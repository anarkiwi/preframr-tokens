"""DecodeState and the per-row state primitives shared by every macro
walker -- decoder dispatch, encoder passes, validators, and the
redundancy reporter all build on what's defined here.
"""

__all__ = ["DecodeState"]

from collections import defaultdict

import numpy as np

from preframr_tokens.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    MAX_REG,
    _MIN_DIFF,
    MODE_VOL_REG,
    SET_OP,
    VOICES,
    VOICE_REG_SIZE,
)


class _FastRow:
    """Lightweight row stand-in for ``decoder.expand``."""

    __slots__ = ("reg", "val", "op", "subreg", "diff", "description", "Index")

    def __init__(self, reg, val, op, subreg, diff, description, Index):
        self.reg = reg
        self.val = val
        self.op = op
        self.subreg = subreg
        self.diff = diff
        self.description = description
        self.Index = Index


def _fastrow_from_arrs(arrs, i):
    """Build a ``_FastRow`` from the per-column arrays produced by
    ``_df_arrays_and_frames`` at row index ``i``. Centralises the
    repeated ``_FastRow(reg=int(regs[i]), val=int(vals[i]), ...)``
    pattern that appeared in 8+ pass walkers; one place to fix if the
    columns ever change.
    """
    return _FastRow(
        reg=int(arrs["reg"][i]),
        val=int(arrs["val"][i]),
        op=int(arrs["op"][i]),
        subreg=int(arrs["subreg"][i]),
        diff=int(arrs["diff"][i]),
        description=int(arrs["description"][i]),
        Index=int(arrs["Index"][i]),
    )


def _frame_arrays(f_df):
    """Extract per-row arrays for a frame group. Used by hot pass loops
    that previously walked via ``itertuples`` (slow: per-row namedtuple +
    per-cell pandas indexing). Caller iterates by integer index.
    """
    cols = {
        "reg": f_df["reg"].to_numpy(),
        "val": f_df["val"].to_numpy(),
        "op": f_df["op"].to_numpy(),
        "diff": f_df["diff"].to_numpy(),
        "Index": f_df.index.to_numpy(),
    }
    if "subreg" in f_df.columns:
        cols["subreg"] = f_df["subreg"].to_numpy()
    else:
        cols["subreg"] = np.full(len(f_df), -1, dtype=np.int64)
    if "description" in f_df.columns:
        cols["description"] = f_df["description"].to_numpy()
    else:
        cols["description"] = np.zeros(len(f_df), dtype=np.int64)
    return cols


def _df_arrays_and_frames(df):
    """Extract whole-df column arrays once, plus frame-start positions."""
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    diffs = df["diff"].to_numpy()
    subregs = (
        df["subreg"].to_numpy()
        if "subreg" in df.columns
        else np.full(len(df), -1, dtype=np.int64)
    )
    descs = (
        df["description"].to_numpy()
        if "description" in df.columns
        else np.zeros(len(df), dtype=np.int64)
    )
    indices = df.index.to_numpy()
    is_marker = (regs == FRAME_REG) | (regs == DELAY_REG)
    frame_starts = np.where(is_marker)[0]
    arrs = {
        "reg": regs,
        "val": vals,
        "op": ops,
        "diff": diffs,
        "subreg": subregs,
        "description": descs,
        "Index": indices,
    }
    return arrs, frame_starts


_FRAME_MARKER_REGS = {FRAME_REG, DELAY_REG}


_PER_VOICE_SUBREG_BASES = (4, 5, 6)
SUBREG_REGS = tuple(
    base + v * VOICE_REG_SIZE for v in range(VOICES) for base in _PER_VOICE_SUBREG_BASES
) + (FILTER_REG, MODE_VOL_REG)

PWM_REGS_BY_VOICE = tuple(2 + v * VOICE_REG_SIZE for v in range(VOICES))
FREQ_REGS_BY_VOICE = tuple(0 + v * VOICE_REG_SIZE for v in range(VOICES))
CTRL_REGS_BY_VOICE = tuple(4 + v * VOICE_REG_SIZE for v in range(VOICES))
AD_REGS_BY_VOICE = tuple(5 + v * VOICE_REG_SIZE for v in range(VOICES))
SR_REGS_BY_VOICE = tuple(6 + v * VOICE_REG_SIZE for v in range(VOICES))
GATE_REGS_BY_VOICE = tuple(
    (CTRL_REGS_BY_VOICE[v], AD_REGS_BY_VOICE[v], SR_REGS_BY_VOICE[v])
    for v in range(VOICES)
)
_GATE_REG_TO_VOICE = {r: v for v in range(VOICES) for r in GATE_REGS_BY_VOICE[v]}
_BUNDLE_REGS_FLAT = frozenset(
    reg for v in range(VOICES) for reg in GATE_REGS_BY_VOICE[v]
)


class DecodeState:
    """Per-stream state shared by all ``MacroDecoder`` invocations."""

    def __init__(
        self,
        frame_diff,
        last_diff=None,
        strict=False,
    ):
        self.frame_diff = frame_diff
        self.last_val = np.zeros(MAX_REG + 1, dtype=np.int64)
        self.last_flip = np.zeros(MAX_REG + 1, dtype=np.int64)
        self.active_flip_regs = set()
        self.last_diff = dict(last_diff) if last_diff else {}
        self.strict = strict
        self.pending_diffs = defaultdict(list)
        self.pending_set_writes = defaultdict(list)
        self.interval_links = []
        self.pending_track_links = []
        self.pending_track_fields = {}
        self.pending_nudge_fields = {}
        self.pending_ft = None
        self.last_freq_v0 = {}
        self.last_skel_note = {}
        self.pending_orn = None
        self.pending_stamp_def = None
        self.stamp_table = {}
        self.pending_stamp_rel = None
        self.pending_patch_def = None
        self.patch_table = {}
        self.pending_ctrl_triple = {}
        self.prev_frame_val = np.zeros(MAX_REG + 1, dtype=np.int64)
        self.pending_subreg_reg = None
        self.pending_subreg_nibbles = set()
        self.last_ctrl = {v: 0 for v in range(VOICES)}
        self.pending_filter_triple_hi = 0
        self.pending_filter_triple_lo = 0
        self.pending_deferred_pre_unroll = []
        self.pending_deferred_post_marker = []

    def diff_for(self, reg):
        return self.last_diff.get(reg, _MIN_DIFF)

    def peek(self, reg):
        """Return the running byte value of ``reg`` without mutating
        ``last_val``. Encoder passes that need to inspect prior register
        state to make encode-time decisions (e.g. SubregPass's nibble
        split, DedupSetPass's redundant-SET drop) should go through this
        accessor instead of indexing ``last_val`` directly so the
        """
        return int(self.last_val[reg])

    def flush_pending_subreg(self):
        """Emit and clear any pending subreg state. Returns at most one
        write tuple ``(reg, val, diff)``; empty list if nothing pending."""
        if self.pending_subreg_reg is None:
            return []
        reg = self.pending_subreg_reg
        write = (reg, int(self.last_val[reg]), self.diff_for(reg))
        self.pending_subreg_reg = None
        self.pending_subreg_nibbles = set()
        return [write]

    def maybe_flush_for(self, incoming_reg, incoming_subreg):
        """Decide whether the incoming row should flush pending subreg
        state. Returns the flush writes (possibly empty)."""
        if self.pending_subreg_reg is None:
            return []
        if self.pending_subreg_reg != incoming_reg:
            return self.flush_pending_subreg()
        if incoming_subreg in (0, 1):
            if incoming_subreg in self.pending_subreg_nibbles:
                return self.flush_pending_subreg()
            return []
        return self.flush_pending_subreg()

    def tick_frame(self):
        """Apply pending REPEAT/FLIP/PWM/INTERVAL/subreg ops at a frame boundary."""
        writes = self.flush_pending_subreg()
        for reg in self.active_flip_regs:
            val = int(self.last_flip[reg])
            self.last_val[reg] += val
            self.last_flip[reg] = -val
            writes.append((reg, int(self.last_val[reg]), self.diff_for(reg)))
        for reg in list(self.pending_diffs.keys()):
            if not self.pending_diffs[reg]:
                del self.pending_diffs[reg]
                continue
            delta = self.pending_diffs[reg].pop(0)
            if not self.pending_diffs[reg]:
                del self.pending_diffs[reg]
            self.last_val[reg] += delta
            writes.append((reg, int(self.last_val[reg]), self.diff_for(reg)))
        for reg in list(self.pending_set_writes.keys()):
            queue = self.pending_set_writes[reg]
            if not queue:
                del self.pending_set_writes[reg]
                continue
            val = queue.pop(0)
            if not queue:
                del self.pending_set_writes[reg]
            self.last_val[reg] = val
            writes.append((reg, val, self.diff_for(reg)))
        for link in list(self.interval_links):
            cur_src = int(self.last_val[link["src"]])
            prev_src = int(self.prev_frame_val[link["src"]])
            delta = cur_src - prev_src
            if delta != 0:
                self.last_val[link["tgt"]] += delta
                writes.append(
                    (
                        link["tgt"],
                        int(self.last_val[link["tgt"]]),
                        self.diff_for(link["tgt"]),
                    )
                )
            link["remaining"] -= 1
            if link["remaining"] <= 0:
                self.interval_links.remove(link)
        for link in list(self.pending_track_links):
            lead = int(self.last_val[link["src"]])
            tgt_val = int(round(lead * link["ratio"])) + link["detune"]
            self.last_val[link["tgt"]] = tgt_val
            writes.append((link["tgt"], tgt_val, self.diff_for(link["tgt"])))
            link["remaining"] -= 1
            if link["remaining"] <= 0:
                self.pending_track_links.remove(link)
        np.copyto(self.prev_frame_val, self.last_val)
        return writes


def _build_last_diff(df):
    """Per-reg first-SET diff lookup. Returns ``{int(reg): int(diff)}``
    with ``_MIN_DIFF`` as the fallback when a reg has no SET row.
    """
    last_diff = {}
    for reg in df["reg"].unique():
        sub = df[(df["reg"] == reg) & (df["op"] == SET_OP)]["diff"]
        last_diff[int(reg)] = int(sub.iloc[0]) if len(sub) else _MIN_DIFF
    return last_diff


def _build_decode_state(df, strict=False):
    """Construct a ``DecodeState`` seeded the same way ``expand_ops`` does:
    ``frame_diff`` from the first FRAME_REG row with a positive ``diff`` (a
    degenerate leading frame can carry diff 0, which would zero every
    DELAY-unrolled frame's playback time), ``last_diff`` per reg from each reg's
    first SET. Returns ``None`` if there is no FRAME_REG.
    """
    fr = df[df["reg"] == FRAME_REG]["diff"]
    if fr.empty:
        return None
    positive = fr[fr > 0]
    frame_diff = int(positive.iloc[0]) if not positive.empty else int(fr.iloc[0])
    last_diff = _build_last_diff(df)
    return DecodeState(
        frame_diff,
        last_diff=last_diff,
        strict=strict,
    )
