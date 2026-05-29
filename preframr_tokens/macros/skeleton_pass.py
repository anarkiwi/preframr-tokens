"""SkeletonPass: collapse each clean held freq note into ONE ``SKEL`` atom (op54) -- a
note->freq LUT index (absolute for the first claimed note per reg, a small signed semitone
interval after) so a note is one atomic token; intra-note-motion / off-semitone notes stay
raw op0 SET (byte-exact pass-through). Opt-in (``skeleton_pass``); when on,
``freq_trajectory_pass`` and ``freq_onset_pass`` must be OFF (skeleton owns FREQ_TRAJ_REGS).
"""

__all__ = [
    "SkeletonPass",
    "LUT",
    "fn_to_note_resid",
    "midi_to_fn",
    "CLOCK_RATE",
    "CENTS_THRESHOLD",
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
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
)

CLOCK_RATE = 985248
MIDI_LO, MIDI_HI = 16, 112
CENTS_THRESHOLD = 8.0


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


def _row(reg, subreg, val, diff, irq):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(SKEL_OP),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


class SkeletonPass(MacroPass):
    """Collapse clean held freq notes into one SKEL atom (op54) each; requires
    ``freq_trajectory_pass`` and ``freq_onset_pass`` OFF (skeleton owns the freq channel).
    Notes with intra-note motion or an off-semitone held value stay raw op0 SET."""

    GATE_FLAGS = frozenset({"skeleton_pass"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "skeleton_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        anchors = (
            df["traj_anchor"].to_numpy().astype(bool)
            if "traj_anchor" in df.columns
            else None
        )
        if anchors is not None:
            df = df.drop(columns=["traj_anchor"])
        irq = _first_irq(df)
        drop_idx = []
        new_rows = []
        for reg in FREQ_TRAJ_REGS:
            sets = [
                (
                    int(f_idx[i]),
                    int(i),
                    int(vals[i]),
                    int(diffs[i]) if diffs is not None else 0,
                )
                for i in range(len(df))
                if int(regs[i]) == reg
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ]
            if not sets:
                continue
            self._claim_reg(reg, sets, regs, f_idx, anchors, irq, drop_idx, new_rows)
        if not new_rows:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _claim_reg(self, reg, sets, regs, f_idx, anchors, irq, drop_idx, new_rows):
        prev_note = [None]
        for chunk, emit in self._anchor_chunks(reg, sets, regs, f_idx, anchors):
            if not emit or not chunk:
                continue
            self._claim_chunk(reg, chunk, prev_note, irq, drop_idx, new_rows)

    @staticmethod
    def _claim_chunk(reg, chunk, prev_note, irq, drop_idx, new_rows):
        """Claim a note chunk whose every freq value sits on ONE semitone within
        CENTS_THRESHOLD (tolerates held-note jitter; rejects vibrato/slide/multi-
        semitone to RESID) as one SKEL atom; leave any other chunk as raw op0 SET."""
        note = None
        for c in chunk:
            res = fn_to_note_resid(c[2])
            if res is None or abs(res[1]) > CENTS_THRESHOLD:
                return
            if note is None:
                note = res[0]
            elif res[0] != note:
                return
        if note is None:
            return
        first = chunk[0]
        if prev_note[0] is None:
            row = _row(reg, SKEL_SUBREG_ABS, note, first[3], irq)
        else:
            interval = note - prev_note[0]
            if not -128 <= interval <= 127:
                return
            row = _row(reg, SKEL_SUBREG_INTERVAL, interval & 0xFF, first[3], irq)
        prev_note[0] = note
        row["__pos"] = first[1]
        new_rows.append(row)
        drop_idx.extend(c[1] for c in chunk)

    @staticmethod
    def _anchor_chunks(reg, sets, regs, f_idx, anchors):
        """Split a register's SETs into inter-anchor ``(chunk, emit)`` segments cut at
        each ``traj_anchor`` frame, so no note chunk spans an anchor and every emitted one
        begins on one (the leading segment emits only if its first SET is itself an
        anchor). With no column the whole list is one emittable segment."""
        if anchors is None:
            return [(sets, True)]
        anchor_frames = {
            int(f_idx[i])
            for i in range(len(regs))
            if int(regs[i]) == reg and bool(anchors[i])
        }
        if not anchor_frames:
            return [(sets, True)]
        chunks = [[sets[0]]]
        for s in sets[1:]:
            if s[0] in anchor_frames:
                chunks.append([s])
            else:
                chunks[-1].append(s)
        return [
            (chunk, idx > 0 or chunk[0][0] in anchor_frames)
            for idx, chunk in enumerate(chunks)
        ]
