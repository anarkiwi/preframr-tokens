"""Gate-/sweep-aware trajectory-anchor detector (annotation-only): recover each
register's true trajectory origins from its value dynamics (pass1 sustained level
changes, pass2 ramp/oscillator collapse) unioned with the voice gate, and mark
them in a boolean ``traj_anchor`` column ``FreqTrajectoryPass`` honors as forced
boundaries. Gated by ``trajectory_anchor_pass``; see the design doc."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from preframr_tokens.macros.passes_base import MacroPass, _ensure_subreg, _frame_index
from preframr_tokens.stfconstants import (
    FC_LO_REG,
    FILTER_BITS,
    SET_OP,
    TRAJ_REGS,
    VOICE_REG_SIZE,
    VOICES,
)

__all__ = [
    "AnchorParams",
    "TrajectoryAnchorPass",
    "detect_anchors",
    "pass1_origins",
    "pass2_collapse",
]

_PW_MASK = 0x0FFF
_CTRL_OFFSET = 4
_GATE_BIT = 0x01


@dataclass(frozen=True)
class AnchorParams:
    """Detector knobs (corpus-tuning dials, not hard constants). Per-register
    median window / level band / minimum hold, plus the shared run-collapse
    parameters. Validated starting values from the upstream prototype."""

    freq_w: int = 5
    freq_band: float = 1.0
    freq_min_hold: int = 3
    pw_w: int = 5
    pw_band: float = 200.0
    pw_min_hold: int = 3
    filter_w: int = 5
    filter_band: float = 64.0
    filter_min_hold: int = 3
    p_max: int = 24
    min_run: int = 4
    ac_thresh: float = 0.6
    match_w: int = 2

    def for_kind(self, kind: str) -> tuple[int, float, int]:
        """``(median window, level band, min_hold)`` for ``kind`` in
        ``{FREQ, PW, FILTER}``."""
        prefix = kind.lower()
        return (
            int(getattr(self, f"{prefix}_w")),
            float(getattr(self, f"{prefix}_band")),
            int(getattr(self, f"{prefix}_min_hold")),
        )


def _reg_kind(reg: int) -> tuple[str, int | None]:
    """Map a TRAJ reg id to ``(kind, voice)``; FILTER (global) -> voice None."""
    reg = int(reg)
    if reg == int(FC_LO_REG):
        return "FILTER", None
    offset = reg % VOICE_REG_SIZE
    voice = reg // VOICE_REG_SIZE
    return ("FREQ" if offset == 0 else "PW"), voice


def _convert(kind: str, raw: int) -> float:
    """Register value -> the natural unit the band is expressed in: FREQ ->
    semitone ``round(12*log2(freq))`` (NaN when silent), PW -> 12-bit pulse
    width, FILTER -> 11-bit cutoff (``combined >> FILTER_BITS``)."""
    if kind == "FREQ":
        if raw <= 0:
            return math.nan
        return float(round(12 * math.log2(raw)))
    if kind == "PW":
        return float(int(raw) & _PW_MASK)
    return float(int(raw) >> int(FILTER_BITS))


def _smooth(val: np.ndarray, w: int) -> np.ndarray:
    """Median-filter (window ``w``) over a NaN-interpolated copy of ``val``;
    suppresses vibrato/PWM jitter so a held-with-modulation level reads flat."""
    valid = ~np.isnan(val)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros(len(val), dtype=float)
    series = np.interp(np.arange(len(val)), idx, val[idx])
    half = w // 2
    return np.array(
        [
            np.median(series[max(0, i - half) : i + half + 1])
            for i in range(len(series))
        ],
        dtype=float,
    )


def pass1_origins(
    val: np.ndarray, band: float, w: int, min_hold: int
) -> tuple[list[int], list[float]]:
    """Sustained level-change origins. Walk the smoothed series tracking a
    reference level; emit a candidate origin where the value leaves the
    reference by more than ``band`` and the new level holds ``>= min_hold``
    frames (a transient returning sooner is modulation). Returns
    ``(origin_frames, level_value_at_each)``; deliberately over-segments."""
    valid = ~np.isnan(val)
    if int(valid.sum()) < 8:
        return [], []
    med = _smooth(val, w)
    frames: list[int] = []
    vals: list[float] = []
    ref = float(med[0])
    cand: float | None = None
    cstart = 0
    crun = 0
    for t in range(1, len(med)):
        if not valid[t]:
            cand = None
            continue
        if abs(med[t] - ref) <= band:
            cand = None
            continue
        if cand is None or abs(med[t] - cand) > band:
            cand, cstart, crun = float(med[t]), t, 1
        else:
            crun += 1
        if crun >= min_hold:
            ref = float(np.median(med[cstart : t + 1]))
            frames.append(cstart)
            vals.append(ref)
            cand = None
    return frames, vals


def _periodic(seg: np.ndarray, ac_thresh: float) -> bool:
    """True when the value waveform ``seg`` is periodic (a strong normalised
    autocorrelation peak at some lag in ``[2, len/2)``) = an oscillator, vs an
    aperiodic melodic line. The principled arp-vs-fast-melody separator."""
    if len(seg) < 8:
        return False
    x = np.asarray(seg, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0):
        return True
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    if ac[0] == 0:
        return True
    ac = ac / ac[0]
    hi = max(3, len(x) // 2)
    return float(ac[2:hi].max()) > ac_thresh


def pass2_collapse(
    frames: list[int],
    vals: list[float],
    med: np.ndarray,
    p_max: int,
    min_run: int,
    ac_thresh: float,
) -> list[int]:
    """Collapse a run of ``>= min_run`` pass-1 origins to its onset for a RAMP (a
    maximal strictly-monotonic value run, gap-independent so an ultra-slow sweep
    is still one ramp; the gate union restores gated per-note onsets) or an
    OSCILLATOR (a dense run, gaps ``<= p_max``, with a periodic value waveform per
    the autocorrelation test); aperiodic non-monotonic runs (a melody) are kept."""
    if len(frames) < min_run:
        return list(frames)
    keep: list[int] = []
    i = 0
    n = len(frames)
    while i < n:
        inc = i
        while inc + 1 < n and vals[inc + 1] > vals[inc]:
            inc += 1
        dec = i
        while dec + 1 < n and vals[dec + 1] < vals[dec]:
            dec += 1
        mono_end = inc if (inc - i) >= (dec - i) else dec
        if mono_end - i + 1 >= min_run:
            keep.append(frames[i])
            i = mono_end + 1
            continue
        j = i
        while j + 1 < n and frames[j + 1] - frames[j] <= p_max:
            j += 1
        run_f = frames[i : j + 1]
        if len(run_f) >= min_run and _periodic(
            med[run_f[0] : run_f[-1] + 1], ac_thresh
        ):
            keep.append(run_f[0])
            i = j + 1
            continue
        keep.extend(run_f)
        i = j + 1
    return keep


def dedup(frames: list[int], match_w: int) -> list[int]:
    """Sorted, de-duplicated frames merging any within ``match_w`` (keep first)."""
    out: list[int] = []
    for f in sorted({int(x) for x in frames}):
        if not out or f - out[-1] > match_w:
            out.append(f)
    return out


def detect_anchors(
    value: np.ndarray,
    gate_on: list[int],
    kind: str,
    *,
    params: AnchorParams,
) -> list[int]:
    """Final per-register anchor frames: ``pass2(pass1(value))`` unioned with the
    gate-on retriggers for voice-gated registers (FREQ/PW); the global FILTER is
    not voice-gated, so it is intrinsic-only."""
    w, band, min_hold = params.for_kind(kind)
    origin_frames, origin_vals = pass1_origins(value, band, w, min_hold)
    med = _smooth(value, w)
    collapsed = pass2_collapse(
        origin_frames, origin_vals, med, params.p_max, params.min_run, params.ac_thresh
    )
    gate = [] if kind == "FILTER" else list(gate_on)
    return dedup(collapsed + gate, params.match_w)


class TrajectoryAnchorPass(MacroPass):
    """Annotate the SET row that begins each trajectory (per TRAJ reg) with a
    boolean ``traj_anchor`` column. Annotation-only: no new atoms, no token-stream
    change by itself -- ``FreqTrajectoryPass`` consumes the column as a forced
    segment boundary."""

    GATE_FLAGS = frozenset({"trajectory_anchor_pass"})

    def __init__(self, params: AnchorParams | None = None):
        self.params = params or AnchorParams()

    def apply(self, df, args=None):
        if args is not None and not getattr(args, "trajectory_anchor_pass", True):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        f_idx = _frame_index(df).to_numpy()
        n_frames = int(f_idx.max()) + 1 if len(f_idx) else 0
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        subregs = df["subreg"].to_numpy()
        ops = df["op"].to_numpy() if "op" in df.columns else None
        freq_unq = df["freq_unq"].to_numpy() if "freq_unq" in df.columns else None
        set_mask = subregs == -1
        if ops is not None:
            set_mask = set_mask & (ops == int(SET_OP))

        anchor = np.zeros(len(df), dtype=bool)
        if n_frames == 0:
            df["traj_anchor"] = anchor
            return df

        gate_on = self._gate_on_frames(regs, vals, f_idx, set_mask)
        for reg in TRAJ_REGS:
            kind, voice = _reg_kind(int(reg))
            reg_mask = set_mask & (regs == int(reg))
            value = self._value_series(kind, reg_mask, vals, freq_unq, f_idx, n_frames)
            gon = [] if kind == "FILTER" else gate_on.get(voice, [])
            frames = detect_anchors(value, gon, kind, params=self.params)
            if not frames:
                continue
            anchor |= reg_mask & np.isin(f_idx, np.asarray(frames))
        df["traj_anchor"] = anchor
        return df

    @staticmethod
    def _gate_on_frames(regs, vals, f_idx, set_mask) -> dict[int, list[int]]:
        """Per-voice frames where the control-register gate bit goes 0->1."""
        ctrl_to_voice = {v * VOICE_REG_SIZE + _CTRL_OFFSET: v for v in range(VOICES)}
        gate_on: dict[int, list[int]] = {v: [] for v in range(VOICES)}
        state = {v: 0 for v in range(VOICES)}
        for i in range(len(regs)):
            if not set_mask[i]:
                continue
            voice = ctrl_to_voice.get(int(regs[i]))
            if voice is None:
                continue
            gate = int(vals[i]) & _GATE_BIT
            if gate and not state[voice]:
                gate_on[voice].append(int(f_idx[i]))
            state[voice] = gate
        return gate_on

    @staticmethod
    def _value_series(kind, reg_mask, vals, freq_unq, f_idx, n_frames) -> np.ndarray:
        """Per-frame carry-forward of one register's value, converted to the
        kind's natural unit (NaN where the register has not yet been written, or
        a freq voice is silent)."""
        src = freq_unq if (kind == "FREQ" and freq_unq is not None) else vals
        per_frame: list[int | None] = [None] * n_frames
        idxs = np.where(reg_mask)[0]
        for i in idxs:
            per_frame[int(f_idx[i])] = int(src[i])
        value = np.full(n_frames, np.nan)
        cur: int | None = None
        for frame in range(n_frames):
            if per_frame[frame] is not None:
                cur = per_frame[frame]
            if cur is not None:
                value[frame] = _convert(kind, cur)
        return value
