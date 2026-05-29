"""SkeletonPass + ornament channel (Stage 1+2 unified pitch): segment each freq reg into
NOTES (semitone-run + ``MIN_HOLD`` UNION gate-on), emit one ``SKEL`` atom per note (LUT
index; abs first per reg, signed interval after) plus one ``ORN`` descriptor collapsing the
note's intra-note arps/vibrato/slide into a classified primitive (PLAIN/OCTAVE/ARP/SLIDE/
VIB/RESID, inline params). Opt-in; ported from audit.unified_pitch (validated)."""

__all__ = [
    "SkeletonPass",
    "LUT",
    "fn_to_note_resid",
    "midi_to_fn",
    "CLOCK_RATE",
    "CENTS_THRESHOLD",
    "MIN_HOLD",
    "fit_descriptor",
]

import math

from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    _splice_rows,
    MacroPass,
)
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_REGS,
    ORN_OP,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_VIB,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
    VOICE_CTRL_REG,
    VOICE_REG_SIZE,
)

CLOCK_RATE = 985248
MIDI_LO, MIDI_HI = 16, 112
CENTS_THRESHOLD = 8.0
MIN_HOLD = 3
VIB_MIN_CENTS = 8.0
ARP_MAX_DISTINCT = 4
_OFFSET_LIMIT = 24
_CTRL_FOR_FREQ = {
    int(reg): int(VOICE_CTRL_REG[reg // VOICE_REG_SIZE]) for reg in FREQ_TRAJ_REGS
}


def midi_to_fn(m):
    """MIDI note -> 16-bit SID freq word, clamped to 0..0xFFFF."""
    return max(
        0,
        min(
            0xFFFF, int(round(440.0 * 2 ** ((m - 69) / 12.0) * 16777216.0 / CLOCK_RATE))
        ),
    )


LUT = [midi_to_fn(m) for m in range(128)]


def fn_to_note_resid(fn):
    """16-bit freq -> (nearest MIDI semitone, residual in cents). None if silent/out of range."""
    if fn < 8:
        return None
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return None
    mf = 69 + 12 * math.log2(hz / 440.0)
    note = int(round(mf))
    if not MIDI_LO <= note <= MIDI_HI:
        return None
    return note, (mf - note) * 100.0


def _vib_depth(resids):
    """Per-note sub-semitone vibrato depth bucket (0 none / 1 light / 2 heavy) from the cents
    amplitude of the note's on-semitone writes (the held-note wobble)."""
    near = [c for c in resids if abs(c) < 50]
    if len(near) < 3:
        return 0
    amp = (max(near) - min(near)) / 2.0
    return 0 if amp < VIB_MIN_CENTS else (1 if amp <= 30 else 2)


def fit_descriptor(base, seg_fns):
    """Classify a note's intra-note freq writes (16-bit, in frame order) into one ornament
    primitive. ``base`` = the note's semitone; ``seg_fns`` = settled freqs AFTER the onset.
    Returns (orn_type, offsets) where offsets is the ordered per-frame note-relative semitone
    cycle (ARP/OCTAVE/SLIDE), () for PLAIN, depth bucket carried in VIB, raw deltas in RESID.
    """
    if not seg_fns:
        return ORN_TYPE_PLAIN, ()
    resolved = [fn_to_note_resid(fn) for fn in seg_fns]
    if any(r is None for r in resolved):
        return ORN_TYPE_RESID, ()
    notes = [r[0] for r in resolved]
    resids = [r[1] for r in resolved]
    offs = [n - base for n in notes]
    nonzero = [o for o in offs if o != 0]
    if not nonzero:
        depth = _vib_depth(resids)
        return (ORN_TYPE_VIB, (depth,)) if depth else (ORN_TYPE_PLAIN, ())
    if any(abs(o) > _OFFSET_LIMIT for o in offs):
        return ORN_TYPE_RESID, ()
    distinct = sorted(set(offs))
    if set(distinct) <= {0, 12} or set(distinct) <= {0, -12}:
        return ORN_TYPE_OCTAVE, tuple(offs)
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    monotone = all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)
    if monotone and abs(offs[-1] - offs[0]) >= 2:
        return ORN_TYPE_SLIDE, tuple(offs)
    if len(distinct) <= ARP_MAX_DISTINCT:
        return ORN_TYPE_ARP, tuple(offs)
    return ORN_TYPE_RESID, ()


def _row(reg, op, subreg, val, diff, irq):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


class SkeletonPass(MacroPass):
    """Dense skeleton + ornament: segment each freq reg into notes (semitone-run + MIN_HOLD
    UNION gate-on), emit one SKEL atom per note and one ORN descriptor collapsing its
    intra-note arps/vibrato/slide. Requires ``freq_trajectory_pass`` / ``freq_onset_pass``
    OFF (skeleton owns the freq channel)."""

    GATE_FLAGS = frozenset({"skeleton_pass"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "skeleton_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        if "traj_anchor" in df.columns:
            df = df.drop(columns=["traj_anchor"])
        ctx = self._context(df)
        irq = _first_irq(df)
        drop_idx = []
        new_rows = []
        for reg in FREQ_TRAJ_REGS:
            self._claim_reg(reg, ctx, irq, drop_idx, new_rows)
        if not new_rows:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    @staticmethod
    def _context(df):
        """Per-row arrays + per-reg ordered (frame, row_index, val, diff) freq SETs and per-voice
        gate-on rising-edge frames (ctrl bit0)."""
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        freq_sets = {int(reg): [] for reg in FREQ_TRAJ_REGS}
        gate_on = {int(reg): set() for reg in FREQ_TRAJ_REGS}
        gate_state = {int(c): 0 for c in _CTRL_FOR_FREQ.values()}
        ctrl_to_freq = {int(c): int(f) for f, c in _CTRL_FOR_FREQ.items()}
        for i in range(len(df)):
            reg = int(regs[i])
            if reg in freq_sets and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                freq_sets[reg].append(
                    (
                        int(f_idx[i]),
                        int(i),
                        int(vals[i]),
                        int(diffs[i]) if diffs is not None else 0,
                    )
                )
            elif reg in gate_state and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                g = int(vals[i]) & 1
                if g and not gate_state[reg]:
                    gate_on[ctrl_to_freq[reg]].add(int(f_idx[i]))
                gate_state[reg] = g
        return freq_sets, gate_on

    def _claim_reg(self, reg, ctx, irq, drop_idx, new_rows):
        freq_sets, gate_on = ctx
        sets = freq_sets[int(reg)]
        if not sets:
            return
        onsets = self._segment_notes(sets, gate_on[int(reg)])
        if len(onsets) < 2:
            return
        by_frame = {s[0]: s for s in sets}
        onset_frames = [fr for fr, _ in onsets]
        prev_note = None
        for k, (onset_fr, note) in enumerate(onsets):
            nxt_fr = (
                onset_frames[k + 1]
                if k + 1 < len(onset_frames)
                else max(s[0] for s in sets) + 1
            )
            anchor = by_frame.get(onset_fr)
            if anchor is None:
                continue
            seg = [s for s in sets if onset_fr < s[0] < nxt_fr]
            per_frame = self._per_frame_fns(note, anchor, seg, onset_fr, nxt_fr)
            orn_type, _ = fit_descriptor(note, [s[2] for s in seg])
            claimed = self._emit_note(
                reg, note, anchor, per_frame, orn_type, prev_note, irq, new_rows
            )
            if not claimed:
                continue
            prev_note = note
            drop_idx.append(anchor[1])
            drop_idx.extend(s[1] for s in seg)

    @staticmethod
    def _per_frame_fns(note, anchor, seg, onset_fr, nxt_fr):
        """Forward-filled per-frame settled freq for frames AFTER the onset up to the next note
        (held frames inherit the last write). Position 0 is the onset frame itself (= anchor
        value, owned by the SKEL atom); the ornament replays positions 1..end byte-exactly.
        """
        span = max(0, nxt_fr - onset_fr - 1)
        if span == 0:
            return []
        seg_at = {s[0] - onset_fr: int(s[2]) for s in seg}
        out = []
        cur = int(anchor[2])
        for k in range(1, span + 1):
            cur = seg_at.get(k, cur)
            out.append(cur)
        return out

    @staticmethod
    def _segment_notes(sets, gate_frames):
        """Note onsets = semitone level-change that HOLDS >= MIN_HOLD frames (so fast arp steps
        are not notes) UNION gate-on frames. Hold duration is measured by frame span (robust to
        per-frame duplicate writes from the block-expand path). Returns sorted [(frame, note)].
        """
        resolved = []
        for s in sets:
            r = fn_to_note_resid(s[2])
            if r is not None:
                resolved.append((s[0], r[0]))
        runs = []
        for fr, note in resolved:
            if runs and runs[-1][1] == note:
                runs[-1] = (runs[-1][0], note, fr)
            else:
                runs.append((fr, note, fr))
        onsets = {}
        for i, (start, note, last) in enumerate(runs):
            end = runs[i + 1][0] if i + 1 < len(runs) else last + MIN_HOLD
            if (end - start) >= MIN_HOLD:
                onsets[start] = note
        fn_at = {fr: note for fr, note in resolved}
        for fr in gate_frames:
            if fr in fn_at:
                onsets.setdefault(fr, fn_at[fr])
        return sorted(onsets.items())

    def _emit_note(
        self, reg, note, anchor, per_frame, orn_type, prev_note, irq, new_rows
    ):
        """Emit the SKEL atom (abs/interval) for one note followed by its ORN descriptor.
        Returns False (claim nothing) if the skeleton interval overflows a signed byte.
        """
        skel = self._skel_row(reg, note, anchor, prev_note, irq)
        if skel is None:
            return False
        orn = self._orn_rows(reg, note, per_frame, orn_type, anchor[3], irq)
        skel["__pos"] = anchor[1]
        new_rows.append(skel)
        for row in orn:
            row["__pos"] = anchor[1]
            new_rows.append(row)
        return True

    @staticmethod
    def _skel_row(reg, note, anchor, prev_note, irq):
        if prev_note is None:
            return _row(reg, SKEL_OP, SKEL_SUBREG_ABS, note, anchor[3], irq)
        interval = note - prev_note
        if not -128 <= interval <= 127:
            return None
        return _row(reg, SKEL_OP, SKEL_SUBREG_INTERVAL, interval & 0xFF, anchor[3], irq)

    @staticmethod
    def _orn_rows(reg, note, per_frame, orn_type, diff, irq):
        """Build the ORN descriptor: a TYPE atom then inline per-frame params. Semitone types
        (OCTAVE/ARP/SLIDE) store one signed P1 offset per frame (decoded LUT[note+off],
        audio-exact at the semitone floor); VIB/RESID store the raw 16-bit freq per frame as a
        P1 hi + P2 lo pair (byte-exact, the escape that keeps the wobble/unstructured tail).
        PLAIN carries no params (the SKEL value holds)."""
        count = 0 if orn_type == ORN_TYPE_PLAIN else len(per_frame)
        count = min(count, 0xFFFF)
        rows = [
            _row(reg, ORN_OP, ORN_SUBREG_TYPE, orn_type, diff, irq),
            _row(reg, ORN_OP, ORN_SUBREG_P2, count, diff, irq),
        ]
        if count == 0:
            return rows
        if orn_type in (ORN_TYPE_VIB, ORN_TYPE_RESID):
            for fn in per_frame[:count]:
                u = int(fn) & 0xFFFF
                rows.append(
                    _row(reg, ORN_OP, ORN_SUBREG_P1, (u >> 8) & 0xFF, diff, irq)
                )
                rows.append(_row(reg, ORN_OP, ORN_SUBREG_P2, u & 0xFF, diff, irq))
            return rows
        for fn in per_frame[:count]:
            res = fn_to_note_resid(int(fn))
            off = (res[0] - note) if res is not None else 0
            rows.append(_row(reg, ORN_OP, ORN_SUBREG_P1, int(off) & 0xFF, diff, irq))
        return rows
