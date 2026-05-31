import itertools
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from preframr_tokens import macros
from preframr_tokens.engine_fingerprint import (
    UNKNOWN_CLUSTER,
    compute_fingerprint,
)
from preframr_tokens.macros.ctrl_update_pass import CtrlUpdatePass
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_nudge_pass import FreqNudgePass
from preframr_tokens.macros.freq_onset_pass import FreqOnsetPass
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.skeleton_pass import SkeletonPass
from preframr_tokens.macros.stamp_pass import StampPass
from preframr_tokens.macros.sweep_pass import SweepPass
from preframr_tokens.macros.trajectory_anchor import TrajectoryAnchorPass
from preframr_tokens.macros.wavetable_pass import WavetablePass
from preframr_tokens.macros.lonely_validator import LonelyWriteValidatorPass
from preframr_tokens.macros.patch_pass import PatchPass
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.release_update_pass import ReleaseUpdatePass
from preframr_tokens.macros.voice_track_pass import VoiceTrackPass
from preframr_tokens.reg_mappers import FreqMapper
from preframr_tokens.palette_io import load_palettes_attrs
from preframr_tokens.reg_match import (
    ctrl_match,
    frame_match,
    freq_match,
)
from preframr_tokens.utils import wrapbits
from preframr_tokens.stfconstants import (
    DEFAULT_IRQ_CYCLES,
    DELAY_REG,
    DIFF_PDTYPE,
    IRQ_PDTYPE,
    OP_PDTYPE,
    DUMP_SUFFIX,
    FC_LO_REG,
    FILTER_BITS,
    FILTER_REG,
    FRAME_REG,
    MAX_REG,
    META_FREQ_BITS,
    _MIN_DIFF,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PAD_REG,
    PARSED_SUFFIX,
    PCM_BITS,
    REG_PDTYPE,
    SET_OP,
    SUBREG_PDTYPE,
    TOKEN_KEYS,
    VAL_PDTYPE,
    PWM_SUSTAIN_OP,
    VOICES,
    VOICE_REG,
    VOICE_REG_SIZE,
    WAVETABLE_SUSTAIN_OP,
)

__all__ = [
    "RegLogParser",
    "remove_voice_reg",
    "prepare_df_for_audio",
    "read_initial_irq",
]

_SUSTAIN_MARKER_OPS = {int(PWM_SUSTAIN_OP), int(WAVETABLE_SUSTAIN_OP)}

FRAME_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
    "op": OP_PDTYPE,
}

PY_MTIME = Path(__file__).resolve().stat().st_mtime
pd.set_option("future.no_silent_downcasting", True)


def read_initial_irq(df, default: int = DEFAULT_IRQ_CYCLES) -> int:
    """Read the IRQ cycle interval (frame period) from a parser-output df: the
    first FRAME_REG row with a positive ``diff``. A degenerate leading frame can
    carry ``diff`` 0 (song starts at t=0); using it as the period zeroes every
    DELAY-expanded frame's playback time. Falls back to ``default`` (canonical
    SID ~50.1 Hz raster) when no FRAME row has a positive diff."""
    frame_diffs = df[df["reg"] == FRAME_REG]["diff"]
    positive = frame_diffs[frame_diffs > 0]
    if not positive.empty:
        return int(positive.iloc[0])
    return int(default)


FILTER_SHIFT_DF = pd.DataFrame(
    [{"reg": FILTER_REG, "val": i, "y": wrapbits(i, 3)} for i in range(2**3)],
    dtype=MODEL_PDTYPE,
)


def _build_valid_voiceorders():
    """Compute the set of legal FRAME_REG ``svt`` values: every 2-bit
    packed permutation of voice ids 0..2 (1-based). Length must be 15
    (sum of P(3,1)+P(3,2)+P(3,3)). Pure function of VOICES; computed
    once at module import.
    """
    voiceorders = set()
    for r in range(1, VOICES + 1):
        for perm in itertools.permutations(range(VOICES), r):
            voiceorder = 0
            for i, v in enumerate(perm):
                voiceorder += v + 1 << (2 * i)
            voiceorders.add(voiceorder)
    assert len(voiceorders) == 15, (len(voiceorders), sorted(voiceorders))
    return voiceorders


VALID_VOICEORDERS = _build_valid_voiceorders()


def frame_reg(orig_df):
    """Cumulative frame index (1-based per-FRAME_REG / DELAY_REG marker).
    Returns a Series aligned with ``orig_df.index``.
    """
    df = orig_df[["reg", "val"]].copy()
    df.loc[(df["reg"] == FRAME_REG), "val"] = 1
    m = frame_match(df)
    df.loc[~m, "val"] = 0
    return df["val"].cumsum()


def norm_df(orig_df):
    """Augment ``orig_df`` with the per-frame / per-voice / per-row
    bookkeeping columns (``f``, ``v``, ``n``, ``vd``) every parse-time
    helper that needs cross-frame state walks against. Pure function;
    callers use the result to merge / pivot / groupby on stable keys.
    """
    df = orig_df.copy().reset_index(drop=True)
    df["f"] = frame_reg(df)
    df["v"] = df["reg"].abs().floordiv(VOICE_REG_SIZE).astype(MODEL_PDTYPE)
    df.loc[df["v"] < 0, "v"] = 0
    df["n"] = (df.index + 1) * 10
    df.loc[df["f"].diff() != 0, "v"] = 0
    df["vd"] = df["v"].diff().astype(MODEL_PDTYPE).fillna(0)
    return df


def last_reg_val_frame(orig_df, regs):
    """Yield, per reg in ``regs``, a long-form DataFrame with one row
    per (frame, voice) carrying the latest ``val`` written to that
    reg-family within the frame plus the previous-frame val (``pval``).
    The substrate the per-reg-burst layer uses to detect DIFF / FLIP /
    REPEAT runs across frames.
    """
    assert not len(orig_df[orig_df["reg"] == VOICE_REG])
    pivot_df = norm_df(orig_df.copy())
    pivot_df = (
        pivot_df.pivot(columns="reg", values="val", index=["f", "n", "v"])
        .astype(MODEL_PDTYPE)
        .ffill()
        .fillna(0)
    )
    for reg in regs:
        sub_df = pivot_df.copy()
        if reg < VOICE_REG_SIZE:
            vregs = [v * VOICE_REG_SIZE + reg for v in range(VOICES)]
        else:
            vregs = [reg]
        vregs = [vreg for vreg in vregs if vreg in sub_df.columns]
        sub_df = (
            sub_df[vregs].reset_index()[["f"] + vregs].drop_duplicates("f", keep="last")
        )
        sub_df = (
            pd.melt(sub_df, id_vars=["f"], var_name="v", value_name="val")
            .astype(MODEL_PDTYPE)
            .sort_values("f")
        ).reset_index(drop=True)
        sub_df["v"] = sub_df["v"].floordiv(VOICE_REG_SIZE)
        diff_df = sub_df.copy()
        diff_df["pval"] = diff_df["val"]
        diff_df["f"] += 1
        diff_df = diff_df[["pval", "v", "f"]]
        sub_df = sub_df.merge(diff_df, how="left", on=["v", "f"])
        sub_df = sub_df.fillna(0).astype(MODEL_PDTYPE).sort_values(["f", "v"])
        yield sub_df


def reset_diffs(orig_df, irq, sidq):
    df = orig_df.copy().reset_index(drop=True)
    frame_cond = df["reg"] == FRAME_REG

    if irq is None:
        irq = df[frame_cond]["diff"].iat[0]

    df.loc[df["reg"] == DELAY_REG, "diff"] = df["val"] * irq
    df["delay"] = df["diff"] * sidq

    df["f"] = (frame_cond).cumsum()
    df["fd"] = df["diff"]
    df.loc[df["reg"] < 0, "fd"] = pd.NA

    df["fd"] = df.groupby(["f"])["fd"].transform("sum") * sidq
    df.loc[frame_cond, "delay"] = df[frame_cond]["delay"] - df[frame_cond][
        "fd"
    ].shift().fillna(0)
    return df


def remove_voice_reg(orig_df, reg_widths):
    df = orig_df.copy()
    df["v"] = pd.NA
    df["fn"] = 0
    df["f"] = 0
    df.loc[df["reg"] == FRAME_REG, "f"] = 1
    df["f"] = df["f"].cumsum()
    m = df["reg"].isin({FRAME_REG, VOICE_REG}) | df["op"].isin(_SUSTAIN_MARKER_OPS)
    df.loc[m, "fn"] = 1
    df["fn"] = df.groupby("f")["fn"].cumsum()
    df["fn"] -= 1
    df.loc[df["reg"] == FRAME_REG, "sval"] = (
        df[df["reg"] == FRAME_REG]["val"] & 2**6 - 1
    )
    df["sval"] = df["sval"].ffill().fillna(0)
    sval_at_m = df.loc[m, "sval"].fillna(0).astype(np.int64).to_numpy()
    fn_at_m = df.loc[m, "fn"].fillna(0).astype(np.int64).to_numpy()
    df.loc[m, "v"] = (np.right_shift(sval_at_m, fn_at_m * 2) & 0b11) - 1
    df["v"] = df["v"].ffill().fillna(0)
    df.loc[df["v"] < 0, "v"] = 0
    m = (df["reg"] >= 0) & (df["reg"] < VOICE_REG_SIZE)
    reg_at_m = df.loc[m, "reg"].astype(np.int64).to_numpy()
    v_at_m = df.loc[m, "v"].fillna(0).astype(np.int64).to_numpy()
    df.loc[m, "reg"] = reg_at_m + v_at_m * VOICE_REG_SIZE
    df = df[df["reg"] != VOICE_REG]
    df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
    if orig_df.attrs:
        df.attrs.update(orig_df.attrs)
    for v in range(VOICES):
        v_offset = v * VOICE_REG_SIZE
        for i in range(VOICE_REG_SIZE):
            if i in reg_widths:
                reg_widths[v_offset + i] = reg_widths[i]
    return df, reg_widths


def prepare_df_for_audio(orig_df, reg_widths, irq, sidq, strict=False, prompt_len=None):
    """Audio-render bootstrap. Take a macro-form df (post-tokenisation
    or post-prediction) and run the post-encode chain that yields the
    per-row ``delay`` column the SID renderer consumes:
    ``remove_voice_reg`` → ``expand_ops`` → ``reset_diffs``. ``prompt_len``
    splits ``description`` so prompt rows render with description=0
    """
    if not prompt_len:
        prompt_len = len(orig_df)
    df = orig_df.copy()
    df["description"] = 0
    if prompt_len < len(df):
        df.loc[prompt_len:, "description"] = 1
        assert len(df[df["description"] == 1])
    df, reg_widths = remove_voice_reg(df, reg_widths)
    df = expand_ops(df, strict=strict)
    df = reset_diffs(df, irq, sidq)
    return df, reg_widths


def combine_val(reg_df, reg, reg_range, dtype=MODEL_PDTYPE, lobits=8):
    """Coalesce ``reg_range`` consecutive little-endian byte registers starting at
    ``reg`` into one wide value in ``val`` (each byte forward-filled so a partial
    update reads its last settled byte), re-keyed to ``reg``. The per-byte masked
    assignment leaves NA on non-matching rows, so each byte is coerced to int
    (a plain-int caller upcasts to float there) before the shift.
    """
    origcols = reg_df.columns
    for i in range(reg_range):
        reg_df[str(i)] = reg_df[reg_df["reg"] == (reg + i)]["val"]
        reg_df[str(i)] = reg_df[str(i)].ffill().fillna(0).astype("int64")
        reg_df[str(i)] = np.left_shift(reg_df[str(i)].values, int(lobits * i))
    reg_df["val"] = 0
    reg_df["reg"] = reg
    for i in range(reg_range):
        reg_df["val"] = reg_df["val"].astype(dtype) + reg_df[str(i)]
    return reg_df[origcols]


def combine_reg(orig_df, reg, diffmax=512, bits=0, lobits=8):
    """Settle the 16-bit value spanning ``reg`` (lo) and ``reg+1`` (hi): forward-fill
    both bytes and keep the last settled value per ``clock // diffmax`` bucket, so a
    coordinated lo+hi update is read as one value and a half-updated pair is never
    seen. Non-``reg`` rows pass through. ``bits`` masks off low bits of the result.
    The canonical SID freq/PW/filter combine, shared by the parser and freq audits.
    """
    cond = (orig_df["reg"] == reg) | (orig_df["reg"] == (reg + 1))
    reg_df = orig_df[cond].sort_values("clock", kind="stable").copy()
    non_reg_df = orig_df[~cond]
    reg_df["dclock"] = reg_df["clock"].floordiv(diffmax)
    reg_df = combine_val(reg_df, reg, 2, lobits=lobits)
    reg_df = reg_df.drop_duplicates(["dclock"], keep="last")
    if bits:
        reg_df["val"] = np.left_shift(np.right_shift(reg_df["val"], bits), bits)
    df = pd.concat([non_reg_df, reg_df[orig_df.columns]], ignore_index=True)
    df = df.astype(orig_df.dtypes)
    return df


class RegLogParser:
    def __init__(self, args=None, logger=logging, cluster_table=None):
        self.args = args
        self.logger = logger
        self.freq_mapper = None
        if self.args:
            self.freq_mapper = FreqMapper(cents=self.args.cents)
        self.exclude_set = self._load_exclude_list()
        self.cluster_table = cluster_table

    def _load_exclude_list(self):
        path = getattr(self.args, "exclude_list", None) if self.args else None
        if not path:
            return frozenset()
        try:
            with open(path) as f:
                rows = [ln.strip().split(",", 1)[0] for ln in f if ln.strip()]
        except FileNotFoundError:
            self.logger.warning("exclude-list %s not found", path)
            return frozenset()
        keys = {r for r in rows if r and r != "path"}
        self.logger.info("loaded %d paths from exclude-list %s", len(keys), path)
        return frozenset(keys)

    def _excluded(self, name):
        if not self.exclude_set:
            return False
        if name in self.exclude_set:
            return True
        for tail in ("/" + name.rsplit("/", 2)[-2] + "/" + name.rsplit("/", 1)[-1],):
            if tail.lstrip("/") in self.exclude_set:
                return True
        return False

    def _read_df(self, name):
        try:
            df = pd.read_parquet(name)
        except Exception as e:
            raise ValueError(f"cannot read {name}: {e}") from e

        df = df[df["reg"] <= MAX_REG]
        df["val"] = df["val"].astype(VAL_PDTYPE)
        chips = df["chipno"].nunique()
        df = df[["clock", "irq", "reg", "val"]]
        if chips > 1:
            return df[df["clock"] < 0]
        return df

    def _maskreg(self, df, reg, valmask):
        mask = df["reg"] == reg
        df.loc[mask, ["val"]] = df[mask]["val"] & valmask

    def _highbitmask(self, bits):
        return 255 - (2**bits - 1)

    def _maskregbits(self, df, reg, bits):
        self._maskreg(df, reg, self._highbitmask(bits))

    def _state_df(self, states, dataset, irq):
        tokens = dataset.tokenizer.tokens.copy()
        tokens["diff"] = _MIN_DIFF
        tokens.loc[tokens["reg"] < -MAX_REG, "diff"] = 0
        tokens.loc[tokens["reg"] == PAD_REG, "diff"] = 0
        tokens.loc[tokens["reg"] == FRAME_REG, "diff"] = irq
        df = pd.DataFrame(states, columns=["n"]).merge(tokens, on="n", how="left")
        if "description" not in df.columns:
            df["description"] = 0
        return df

    def _anchor_enabled(self):
        """Whether TrajectoryAnchorPass will run for this parse (mirrors the
        pass's own gate), so ``freq_unq`` is only stashed when it is consumed."""
        return self.args is None or getattr(self.args, "trajectory_anchor_pass", True)

    @staticmethod
    def _stash_freq_unq(df):
        """Copy the full-precision 16-bit freq into a ``freq_unq`` side column
        before lossy cent-quantization, so TrajectoryAnchorPass detects pitch
        origins at semitone resolution; carried through the later row-wise
        transforms and dropped with the other side columns at token-emit time."""
        df = df.copy()
        df["freq_unq"] = df["val"]
        return df

    def _squeeze_changes(self, df):
        prev = df.groupby("reg")["val"].shift()
        mask = prev.isna() | (prev != df["val"])
        cols = ["clock", "irq", "reg", "val"]
        if "freq_unq" in df.columns:
            cols.append("freq_unq")
        return df.loc[mask, cols].reset_index(drop=True)

    def _combine_val(self, reg_df, reg, reg_range, dtype=MODEL_PDTYPE, lobits=8):
        return combine_val(reg_df, reg, reg_range, dtype=dtype, lobits=lobits)

    def _combine_reg(self, orig_df, reg, diffmax=512, bits=0, lobits=8):
        return combine_reg(orig_df, reg, diffmax=diffmax, bits=bits, lobits=lobits)

    def _rotate_filter(self, df, r):
        m = df["reg"] == FILTER_REG
        df.loc[m, "fres"] = df[m]["val"].values & 0xF0
        df.loc[m, "val"] -= df[m]["fres"]
        for _ in range(r):
            df = df.merge(FILTER_SHIFT_DF, how="left", on=["reg", "val"])
            m = df["reg"] == FILTER_REG
            df.loc[m, "val"] = df[m]["y"]
            df = df.drop(["y"], axis=1)
        m = df["reg"] == FILTER_REG
        df.loc[m, "val"] += df[m]["fres"]
        return df

    def _rotate_voice_augment(self, orig_df, max_perm):
        if orig_df.empty:
            return
        if not max_perm:
            yield orig_df
            return
        for r in range(min(VOICES, max_perm)):
            df = orig_df.copy()
            m = df["reg"].abs() < VOICE_REG_SIZE * VOICES
            df["rreg"] = (df[m]["reg"].abs() + (VOICE_REG_SIZE * r)).mod(
                VOICE_REG_SIZE * VOICES
            )
            df.loc[m, "reg"] = df["rreg"]
            df = self._rotate_filter(df, r)
            df = df[orig_df.columns]
            yield df

    def _add_frame_reg(self, orig_df, diffmax, min_irq_prop=0.95):
        df = orig_df.copy()
        df["irqdiff"] = df["irq"].diff().fillna(0).astype(MODEL_PDTYPE)
        df["diff"] = _MIN_DIFF
        df["i"] = df.index * 10
        m = df["irqdiff"] > diffmax
        largest_irqs_sum = 0
        try:
            irq_counts = df["irqdiff"][m].value_counts()
            largest_irqs_sum = sum([k * v for k, v in irq_counts.items()])
            irq = int(irq_counts.nlargest(1).index[0])
        except IndexError:
            irq = 0
        if largest_irqs_sum / df["clock"].max() < min_irq_prop:
            irq = 0
        irq_df = df[m].copy()
        irq_df["i"] -= 1
        irq_df["reg"] = FRAME_REG
        irq_df["diff"] = irq_df["irqdiff"]
        irq_df["val"] = (irq_df["diff"] / irq).round().astype(MODEL_PDTYPE)
        irq_df["diff"] = irq
        irq_df.loc[irq_df["val"] > 1, "reg"] = DELAY_REG
        irq_df.loc[irq_df["reg"] == DELAY_REG, "diff"] = 0
        irq_df.loc[irq_df["reg"] == FRAME_REG, "val"] = 0
        out_cols = ["reg", "val", "diff"]
        if "freq_unq" in df.columns:
            out_cols.append("freq_unq")
        df = (
            pd.concat([df, irq_df], ignore_index=True)
            .sort_values(["i"])[out_cols + ["i"]]
            .astype(MODEL_PDTYPE)
            .reset_index(drop=True)
        )
        return irq, df[out_cols]

    def _cap_delay(self, df):
        """Curve C: exact 1..16, nearest power of 2 17..256; chain via adjacent."""
        m = df["reg"] == DELAY_REG
        df.loc[m & (df["val"] > 255), "val"] = 255

        def _quant(v):
            if v <= 16:
                return v
            for b in (32, 64, 128, 256):
                if v <= b:
                    prev = b // 2
                    return b if (v - prev) >= (b - v) else prev
            return 256

        df.loc[m, "val"] = df.loc[m, "val"].map(_quant)
        return df

    def _split_reg(self, orig_df, reg):
        df = orig_df.copy().reset_index(drop=True)
        df["f"] = df["reg"] == FRAME_REG
        df["f"] = df["f"].cumsum().astype(MODEL_PDTYPE)
        df["fs"] = df.index
        df["prev_f"] = df["f"].shift(1).fillna(-1)
        df.loc[df["f"] == df["prev_f"], "fs"] = pd.NA
        df["fs"] = df["fs"].ffill()
        df["reg_order"] = df.index - df["fs"]
        m = df["reg"] == reg
        reg_df = df[m].copy()
        reg_df["val"] = reg_df["val"].floordiv(256)
        reg_df.loc[:, "reg"] += 1
        df.loc[m, "val"] -= reg_df["val"] * 256
        df = pd.concat([df, reg_df], copy=False, ignore_index=True).sort_values(
            ["f", "reg_order"], ascending=True
        )
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        return df

    def _reduce_val_res(self, df, reg, bits):
        m = df["reg"] == reg
        df.loc[m, "val"] = np.left_shift(np.right_shift(df[m]["val"], bits), bits)
        return df

    def _quantize_freq_to_cents(self, df):
        m = freq_match(df)
        df.loc[m, "val"] = df[m]["val"].map(self.freq_mapper.fi_map)
        return df

    def _norm_pr_order(self, orig_df):
        """Sort rows within each frame by strict numeric voice order."""
        df = norm_df(orig_df.copy())
        df.loc[df["reg"] < 0, "v"] = df["reg"]
        df = df.sort_values(["f", "v", "reg", "op", "n"])
        df = df[orig_df.columns].reset_index(drop=True)
        if orig_df.attrs:
            df.attrs.update(orig_df.attrs)
        return df

    def _add_voice_reg(self, orig_df, zero_voice_reg=True):
        nd = norm_df(orig_df)
        m = (nd["reg"] >= 0) & (nd["v"].isin(set(range(VOICES))))
        first_v = nd[m]
        if first_v.empty:
            return orig_df
        first_v = first_v.iloc[0]
        nd["f"] = (nd["reg"] == FRAME_REG).astype(MODEL_PDTYPE).cumsum()
        df = nd[((nd["vd"] != 0) | (nd["n"] == first_v["n"])) & m].copy()
        nd.loc[m, "reg"] = nd[m]["reg"] % VOICE_REG_SIZE
        df["n"] -= 1
        df["val"] = df["v"]
        df["reg"] = VOICE_REG
        df["op"] = SET_OP
        df = (
            pd.concat([nd, df], ignore_index=True)
            .sort_values(["n"])
            .reset_index(drop=True)
        )
        df["nr"] = df["reg"].shift(-1).astype(REG_PDTYPE)
        df["nval"] = df["val"].shift(-1).astype(VAL_PDTYPE)
        df["pr"] = df["reg"].shift(1).astype(REG_PDTYPE)
        df.loc[((df["reg"] == FRAME_REG) & (df["nr"] == VOICE_REG)), ["v", "val"]] = df[
            "nval"
        ]
        df = df[~((df["reg"] == VOICE_REG) & (df["pr"].fillna(VOICE_REG) == FRAME_REG))]

        df["fn"] = 0
        m = df["reg"].isin({FRAME_REG, VOICE_REG})
        df.loc[m, "fn"] = 1
        df["fn"] = df.groupby("f")["fn"].cumsum()
        df["fn"] -= 1
        df["sv"] = 0
        df.loc[m, "sv"] = np.left_shift(df[m]["v"] + 1, df[m]["fn"] * 2)
        df["svc"] = df.groupby("f")["sv"].cumsum()
        df["svt"] = df.groupby("f")["svc"].transform("max")

        if zero_voice_reg:
            df.loc[df["reg"] == VOICE_REG, "val"] = 0
        else:
            for v in range(VOICES):
                fm = (df["v"] == v) & (df["op"] == SET_OP) & (df["reg"] == 0)
                df.loc[fm, f"{v}freqmeta"] = np.right_shift(
                    df[fm]["val"], self.freq_mapper.bits - META_FREQ_BITS
                )
                freqmeta = df[fm][f"{v}freqmeta"]
                assert (
                    len(freqmeta) == 0 or freqmeta.max() < 2**META_FREQ_BITS
                ), freqmeta.max()
                cm = (df["v"] == v) & (df["op"] == SET_OP) & (df["reg"] == 4)
                df.loc[cm, f"{v}ctrlmeta"] = df[cm]["val"] & 0b11110000
            df = df.ffill().fillna(0)
            m = df["reg"] == VOICE_REG
            df.loc[m, "v"] = df[m]["val"]
            for v in range(VOICES):
                m = (df["reg"] == VOICE_REG) & (df["v"] == v)
                df.loc[m, "val"] = df[m][f"{v}ctrlmeta"] + df[m][f"{v}freqmeta"]

        m = df["reg"] == FRAME_REG
        df.loc[m, "val"] = df[m]["svt"]
        invalid_val = set(df[m]["val"].unique()) - VALID_VOICEORDERS
        assert not invalid_val, invalid_val
        df = df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)
        if orig_df.attrs:
            df.attrs.update(orig_df.attrs)
        return df

    def _simplify_ctrl(self, orig_df):
        df = orig_df.copy()
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            ctrl_reg = v_offset + 4
            df.loc[(df["reg"] == ctrl_reg) & (df["val"] & 0b00010000 == 0), "val"] = (
                df["val"] & 0b11111011
            )
            df.loc[(df["reg"] == ctrl_reg) & (df["val"] & 0b11110000 == 0), "val"] = (
                df["val"] & 0b11111101
            )
        return df

    def _simplify_pcm(self, orig_df):
        """Audible-equivalence rewrite of per-voice PW (pulse-width) writes."""
        df = orig_df.copy()
        df["n"] = df.index.astype(np.int64) * 10
        regs = df["reg"].to_numpy()
        df["v"] = pd.Series(regs // VOICE_REG_SIZE).astype(pd.UInt8Dtype())

        out_dfs = [df[df["v"] >= VOICES]]
        vals = df["val"].to_numpy()
        v_arr = regs // VOICE_REG_SIZE

        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            pcm_reg = v_offset + 2
            ctrl_reg = v_offset + 4
            v_mask = v_arr == v
            if not v_mask.any():
                continue
            v_idx = np.where(v_mask)[0]
            v_regs = regs[v_idx]
            v_vals = vals[v_idx]

            pcm_col = np.where(v_regs == pcm_reg, v_vals.astype(np.float64), np.nan)
            pcm_running = pd.Series(pcm_col).ffill().to_numpy()

            ctrl_mask = v_regs == ctrl_reg
            bit6_set = (v_vals & 0b01000000) == 0b01000000
            p_col = np.full(len(v_idx), np.nan, dtype=np.float64)
            p_col[ctrl_mask & bit6_set] = 1.0
            p_col[ctrl_mask & ~bit6_set] = 0.0
            p_running = pd.Series(p_col).ffill().to_numpy()

            v_df = df.iloc[v_idx].copy()
            override = (v_regs == pcm_reg) & (p_running == 0)
            if override.any():
                new_vals = v_vals.copy()
                new_vals[override] = 0
                v_df["val"] = new_vals

            synth = ctrl_mask & bit6_set
            if synth.any():
                p_df = df.iloc[v_idx[synth]].copy()
                synth_pcm = np.nan_to_num(pcm_running[synth], nan=0.0).astype(np.int64)
                p_df["reg"] = pcm_reg
                p_df["val"] = synth_pcm
                p_df["n"] = p_df["n"] - 1
                v_df = pd.concat([v_df, p_df], ignore_index=True)
            out_dfs.append(v_df)

        df = pd.concat(out_dfs, ignore_index=True).sort_values("n")
        return df[orig_df.columns].reset_index(drop=True)

    def _filter_irq(self, df, name):
        try:
            irq = df["irq"].iloc[0]
        except (IndexError, KeyError):
            self.logger.info(df)
            self.logger.info("skipped %s, no irq", name)
            return False
        if irq < self.args.min_irq or irq > self.args.max_irq:
            self.logger.info("skipped %s, irq %u (outside IRQ range)", name, irq)
            return False
        return True

    def _filter(self, df, name):
        if len(df[df["reg"] == FRAME_REG]) == 0:
            self.logger.info("skipped %s, no frames", name)
            return False
        min_song = getattr(self.args, "min_song_tokens", 256)
        if len(df) < min_song:
            self.logger.info("skipped %s, length %u (< %u)", name, len(df), min_song)
            return False
        is_frame = df["reg"] == FRAME_REG
        frame_idx = is_frame.cumsum()
        vol_mask = df["reg"] == MODE_VOL_REG
        if vol_mask.any():
            vol_per_frame = frame_idx[vol_mask].value_counts()
            max_vpf = int(vol_per_frame.max())
            if max_vpf >= 16:
                self.logger.info(
                    "skipped %s, digi-like vol density (max %u writes per frame)",
                    name,
                    max_vpf,
                )
                return False
        c_df = norm_df(df)
        ctrl_mask = ctrl_match(df)
        if "op" in df.columns:
            ctrl_mask = ctrl_mask & (df["op"] == SET_OP)
        c_df = c_df[ctrl_mask]
        c_df["ccount"] = c_df.groupby(["f", "v"])["reg"].transform("size")
        c_df = c_df[c_df["f"] > 16]
        if len(c_df):
            c_max = c_df["ccount"].max()
            if c_max > 6:
                self.logger.info(
                    "skipped %s, too many (%u) control reg changes per frame",
                    name,
                    c_max,
                )
                return False
        return True

    def _combine_regs(self, df):
        for v in range(VOICES):
            v_offset = v * VOICE_REG_SIZE
            for reg, bits in ((v_offset, 0), ((v_offset + 2), PCM_BITS)):
                df = self._combine_reg(df, reg=reg, bits=bits)
        df = self._combine_reg(df, FC_LO_REG, bits=FILTER_BITS)
        return df.sort_values("clock", kind="stable").reset_index(drop=True)

    def _consolidate_frames(self, orig_df):
        """Collapse each maximal run of marker-only (content-free) frames into one
        DELAY plus a trailing frame, preserving total playback time. A FRAME
        marker is worth ``round(diff / frame_period)`` units, a DELAY its val; the
        run's final marker (the next content frame) is kept verbatim so its
        voice-order survives. Cycle-preserving by construction."""
        rows = norm_df(orig_df.copy()).to_dict("records")
        n = len(rows)
        if not n:
            return orig_df.reset_index(drop=True)
        frame_period = read_initial_irq(orig_df)

        def _units(r):
            if int(r["reg"]) == DELAY_REG:
                return int(r["val"])
            return int(round(int(r["diff"]) / frame_period)) if frame_period else 1

        def _is_marker(r):
            return int(r["reg"]) in (FRAME_REG, DELAY_REG)

        out = []
        i = 0
        while i < n:
            if not _is_marker(rows[i]):
                out.append(rows[i])
                i += 1
                continue
            j = i
            while j < n and _is_marker(rows[j]):
                j += 1
            last = rows[j - 1]
            if int(last["reg"]) == FRAME_REG:
                empty = sum(_units(rows[k]) for k in range(i, j - 1))
                if empty > 0:
                    out.append(
                        {
                            **rows[i],
                            "reg": DELAY_REG,
                            "val": empty,
                            "diff": frame_period,
                        }
                    )
                out.append(last)
            else:
                total = sum(_units(rows[k]) for k in range(i, j))
                if total - 1 > 0:
                    out.append(
                        {
                            **rows[i],
                            "reg": DELAY_REG,
                            "val": total - 1,
                            "diff": frame_period,
                        }
                    )
                out.append(
                    {**rows[i], "reg": FRAME_REG, "val": 0, "diff": frame_period}
                )
            i = j
        df = pd.DataFrame(out)
        return df[orig_df.columns].astype(orig_df.dtypes).reset_index(drop=True)

    def _squeeze_frame_regs(self, orig_df, regs=(0, 2, 21)):
        df = norm_df(orig_df.copy())
        df["dreg"] = pd.NA
        for reg in regs:
            if reg < VOICE_REG_SIZE:
                for v in range(VOICES):
                    dreg = v * VOICE_REG_SIZE + reg
                    df.loc[df["reg"] == dreg, "dreg"] = int(dreg)
            else:
                df.loc[df["reg"] == reg, "dreg"] = df["reg"]
        df = df[~df.duplicated(["f", "dreg"], keep="last") | df["dreg"].isna()]
        df = df[orig_df.columns].reset_index(drop=True)
        return df

    def _add_subreg(self, orig_df):
        df = orig_df.copy()
        sub_dfs = []
        df["subreg"] = int(-1)
        df["n"] = df.index * 10
        for reg in (4, 5, 6, 23, 24):
            m = df["reg"] == reg
            sub_df = df[m].copy()
            sub_df["subreg"] = 1
            sub_df["val"] = np.right_shift(sub_df["val"] & 0b11110000, 4)
            sub_df["n"] += 1
            df.loc[m, "val"] = df[m]["val"] & 0b0001111
            df.loc[m, "subreg"] = 0
            sub_dfs.append(sub_df)
        df = pd.concat([df] + sub_dfs, ignore_index=True).sort_values("n")
        df = df[list(orig_df.columns) + ["subreg"]]
        return df

    def parse(self, name, max_perm=99, require_pq=False, reparse=False):
        if self._excluded(name):
            self.logger.info("skipped %s, exclude-list", name)
            return
        if not reparse:
            parquet_glob = glob.glob(name.replace(DUMP_SUFFIX, PARSED_SUFFIX))
            if parquet_glob:
                for parquet_name in sorted(parquet_glob)[:max_perm]:
                    pf = ParquetFile(parquet_name)
                    try:
                        sample_rows = next(pf.iter_batches(batch_size=1))
                    except StopIteration:
                        continue
                    df = pa.Table.from_batches([sample_rows]).to_pandas()
                    if self._filter_irq(df, parquet_name):
                        df = pd.read_parquet(parquet_name)
                        if self._filter(df, parquet_name):
                            df.attrs.update(load_palettes_attrs(parquet_name))
                            yield df
                return
            if require_pq:
                return
        df = self._read_df(name)
        try:
            from preframr_tokens.dump_meta import meta_path_for, read_meta, write_meta

            existing = read_meta(name)
            if existing is None or existing.stale:
                write_meta(name, df)
        except Exception:  # noqa: BLE001 pylint: disable=broad-except
            pass
        engine_fp_vec = compute_fingerprint(Path(name))
        engine_fp_list = engine_fp_vec.tolist() if engine_fp_vec is not None else None
        engine_fp_cluster = (
            self.cluster_table.cluster_for_path(name)
            if self.cluster_table is not None
            else UNKNOWN_CLUSTER
        )
        df = self._squeeze_changes(df)
        df = self._combine_regs(df)
        if self._anchor_enabled():
            df = self._stash_freq_unq(df)
        if not getattr(self.args, "skeleton_pass", False):
            df = self._quantize_freq_to_cents(df)
        df = self._simplify_ctrl(df)
        df = self._simplify_pcm(df)
        df = self._squeeze_changes(df)
        if df.empty:
            return
        irq, df = self._add_frame_reg(df, diffmax=2048)
        if not self._filter(df, name):
            return
        df = self._squeeze_frame_regs(df)
        df = VoiceTrackPass().apply(df, args=self.args)
        df = TrajectoryAnchorPass().apply(df, args=self.args)
        df = StampPass().apply(df, args=self.args)
        df = SweepPass().apply(df, args=self.args)
        df = SkeletonPass().apply(df, args=self.args)
        df = WavetablePass().apply(df, args=self.args)
        df = FreqTrajectoryPass().apply(df, args=self.args)
        df = FreqOnsetPass().apply(df, args=self.args)
        df = PresetPass().apply(df, args=self.args)
        df = PerRegBurstPass().apply(df, args=self.args)
        df = GateSlopeShiftPass().apply(df, args=self.args)
        df = PatchPass().apply(df, args=self.args)
        df = ReleaseUpdatePass().apply(df, args=self.args)
        df = self._consolidate_frames(df)
        df = self._cap_delay(df)
        delay_val = df[df["reg"] == DELAY_REG]["val"]
        if len(delay_val):
            delay_max = delay_val.max()
            assert delay_max <= 256, delay_max
        irq = min(2 ** (IRQ_PDTYPE.itemsize * 8) - 1, irq)
        df["irq"] = irq
        while not df.empty and frame_match(df.iloc[-1]):
            df = df.head(len(df) - 1)
        while not df.empty and (
            (df.iloc[0]["reg"] == MODE_VOL_REG and df.iloc[0]["val"] == 15)
        ):
            df = df.tail(len(df) - 1)

        if not frame_match(df.iloc[0]):
            first_frame = df[df["reg"] == FRAME_REG].head(1)
            df = pd.concat([first_frame, df], ignore_index=True).reset_index(drop=True)

        for xdf in self._rotate_voice_augment(df, max_perm=max_perm):
            preserved_subreg = None
            if "subreg" in xdf.columns:
                preserved_subreg = xdf["subreg"].copy()
            xdf = xdf[FRAME_DTYPES.keys()].astype(FRAME_DTYPES)
            if preserved_subreg is not None:
                xdf["subreg"] = preserved_subreg.astype(SUBREG_PDTYPE).values
            xdf = self._norm_pr_order(xdf)
            pre_passes_voice_preview = self._add_voice_reg(
                xdf.copy(), zero_voice_reg=True
            )
            if not self._filter(pre_passes_voice_preview, name):
                break
            xdf.attrs["engine_fp_cluster"] = engine_fp_cluster
            xdf = macros.run_passes(xdf, args=self.args)
            xdf = self._norm_pr_order(xdf)
            xdf = macros.run_post_norm_pre_voice_passes(xdf, args=self.args)
            xdf = self._add_voice_reg(xdf, zero_voice_reg=True)
            xdf = FreqNudgePass().apply(xdf, args=self.args)
            xdf = CtrlUpdatePass().apply(xdf, args=self.args)
            xdf = LonelyWriteValidatorPass().apply(xdf, args=self.args)
            xdf = xdf.reset_index(drop=True)
            for k in TOKEN_KEYS:
                if k not in xdf.columns:
                    xdf[k] = int(-1)
            empty_val = xdf[xdf["val"].isna()]
            assert empty_val.empty, (name, empty_val)
            if engine_fp_list is not None:
                xdf.attrs["engine_fingerprint"] = engine_fp_list
            xdf.attrs["engine_fp_cluster"] = engine_fp_cluster
            yield xdf
