"""InstrumentProgramPass: intern each voice's per-frame timbre PROGRAM -- the forward-filled
``(ctrl, AD, SR)`` walk over a note span -- as a define-on-first INSTR_DEF + exact INSTR_REF, so every
onset-associated ctrl/AD/SR raw SET is consumed and the ctrl/AD/SR residual drains to zero by
construction. A voice-relative per-frame write-series codebook (``instrument_program``).
"""

__all__ = ["InstrumentProgramPass"]

import bisect
import os
from collections import defaultdict

import numpy as np

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.codebook_emit import emit_recurring
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE, VOICES
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    INSTR_DEF_OP,
    INSTR_END_OP,
    INSTR_OFF_AD,
    INSTR_OFF_CTRL,
    INSTR_OFF_SR,
    INSTR_REF_OP,
    INSTR_STEP_OP,
    INSTR_SUBREG_FRAME,
    SET_OP,
)

_INSTR_PRIORITY = -12
_FIELD_OFFSETS = (INSTR_OFF_CTRL, INSTR_OFF_AD, INSTR_OFF_SR)


def _row(reg, op, subreg, val, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=irq, irq=irq)


class InstrumentProgramPass(MacroPass):
    """Mine each voice's per-frame (ctrl, AD, SR) program from a note onset and replace it with an
    inline INSTR_DEF (define-on-first) + INSTR_REF, consuming every ctrl/AD/SR SET in the span. Default
    OFF; byte-exact via a register_state guard that falls back to the unclaimed stream on divergence.
    Set PREFRAMR_INSTR_TRUST=1 to skip the guard (faster) once byte-exactness is corpus-proven.
    """

    GATE_FLAGS = frozenset({"instrument_program"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "instrument_program", False):
            return df
        if df is None or len(df) == 0:
            return df
        if "op" in df.columns and df["op"].isin((INSTR_DEF_OP, INSTR_REF_OP)).any():
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        spans = self._spans(df)
        if not spans:
            return df
        drop_idx, new_rows = self._emit(spans, irq)
        if not new_rows:
            return df
        result = arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_INSTR_PRIORITY,
                    label="instrument",
                )
            ],
        )
        skip_guard = bool(os.environ.get("PREFRAMR_INSTR_TRUST"))
        if not skip_guard and not self._instr_is_lossless(df, result):
            return df
        return result

    @staticmethod
    def _instr_is_lossless(df_in, df_out):
        """Decode both streams and require byte-exact per-frame register_state; on any divergence the
        caller drops the whole claim and keeps the literal stream.
        """
        from preframr_tokens.audit_primitives import register_state

        before = register_state(df_in.copy())
        after = register_state(df_out.copy())
        return before.shape == after.shape and not (before != after).any()

    @staticmethod
    def _real_frames(regs, vals):
        """Decoded real-frame index per row (FRAME=+1, DELAY=+val): a span signature must count the
        frames its per-frame replay actually drains across, since a DELAY collapses many silent frames
        into one marker yet unrolls to that many decode ticks."""
        out = np.empty(len(regs), dtype=np.int64)
        f = -1
        for i in range(len(regs)):
            r = int(regs[i])
            if r == FRAME_REG:
                f += 1
            elif r == DELAY_REG:
                f += max(1, int(vals[i]))
            out[i] = f
        return out

    def _spans(self, df):
        """Partition every voice's timeline into note spans bounded by gate-on retriggers (with a
        leading span from the first ctrl/AD/SR write so pre-onset preamble writes are covered too), so
        each ctrl/AD/SR SET falls in exactly one span -- the basis for draining them all.
        """
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        frames = self._real_frames(regs, vals)
        ctrl_of = {}
        for v in range(VOICES):
            creg = int(CTRL_REGS_BY_VOICE[v])
            for off in range(3):
                ctrl_of[creg + off] = creg
        per_voice = defaultdict(lambda: {0: [], 1: [], 2: []})
        for i in range(len(df)):
            reg = int(regs[i])
            if reg not in ctrl_of or int(ops[i]) != SET_OP or int(subregs[i]) != -1:
                continue
            creg = ctrl_of[reg]
            per_voice[creg][reg - creg].append((int(frames[i]), int(i), int(vals[i])))
        if not per_voice:
            return []
        end_frame = int(frames.max()) + 1
        spans = []
        for v in range(VOICES):
            creg = int(CTRL_REGS_BY_VOICE[v])
            if creg in per_voice:
                spans.extend(self._voice_spans(creg, per_voice[creg], end_frame))
        return spans

    @classmethod
    def _voice_spans(cls, creg, writes, end_frame):
        cw, aw, sw = writes[0], writes[1], writes[2]
        all_w = cw + aw + sw
        if not all_w:
            return []
        start = min(w[0] for w in all_w)
        onsets = cls._onsets(cw)
        bounds = sorted({start} | {o for o in onsets if o > start}) + [end_frame]
        out = []
        for k in range(len(bounds) - 1):
            span = cls._build_span(creg, bounds[k], bounds[k + 1], cw, aw, sw)
            if span is not None:
                out.append(span)
        return out

    @staticmethod
    def _onsets(cw):
        """Gate-on rising-edge real frames on the ctrl reg (bit0 0 -> 1), the note-span boundary."""
        out, prev = [], 0
        for fr, _ri, val in cw:
            g = val & 1
            if g and not prev:
                out.append(fr)
            prev = g
        return out

    @staticmethod
    def _val_at(writes, frame):
        """Forward-filled value in effect at ``frame`` (last write at or before it), 0 if none -- the
        SID power-on / register_state initial state."""
        i = bisect.bisect_right(writes, (frame, float("inf"), float("inf")))
        return writes[i - 1][2] if i > 0 else 0

    @classmethod
    def _build_span(cls, creg, b0, b1, cw, aw, sw):
        """One span [b0, b1): the forward-filled per-frame (ctrl, AD, SR) over [b0, last_write] (held
        tail excluded) and ALL ctrl/AD/SR row indices it consumes. The signature uses each frame's
        SETTLED value (last write wins -- a same-frame hard-restart pair settles at its second byte, the
        only value register_state sees), but every write row is consumed so none survives as residual.
        """
        in_lists = [
            seq[bisect.bisect_left(seq, (b0,)) : bisect.bisect_left(seq, (b1,))]
            for seq in (cw, aw, sw)
        ]
        settled = [{} for _ in range(3)]
        for j in range(3):
            for fr, _ri, v in in_lists[j]:
                settled[j][fr] = v
        write_frames = set().union(*(set(s) for s in settled))
        if not write_frames:
            return None
        cur = [cls._val_at(cw, b0), cls._val_at(aw, b0), cls._val_at(sw, b0)]
        length = max(write_frames) - b0 + 1
        seqs = ([], [], [])
        for i in range(length):
            fr = b0 + i
            for j in range(3):
                if fr in settled[j]:
                    cur[j] = settled[j][fr]
                seqs[j].append(cur[j])
        rows = [ri for lst in in_lists for _fr, ri, _v in lst]
        return {
            "reg": creg,
            "frame": b0,
            "sig": tuple(zip(seqs[0], seqs[1], seqs[2])),
            "ctrls": seqs[0],
            "ads": seqs[1],
            "srs": seqs[2],
            "rows": rows,
        }

    @classmethod
    def _emit(cls, spans, irq):
        """Two phases sharing one id space: cross-voice exact recurrence first (a shared instrument bank
        reused across voices), then per-voice define-on-first over the residue -- so every span emits at
        minimum a DEF and no ctrl/AD/SR SET survives. Both keep each DEF strictly before its REFs in the
        post-norm (frame, voice) order."""
        drop_idx, new_rows = [], []
        consumed = set()
        next_id = cls._cross_voice(spans, irq, drop_idx, new_rows, consumed)
        groups = defaultdict(list)
        for s in spans:
            if id(s) not in consumed:
                groups[(s["reg"], s["sig"])].append(s)

        def emit_first(cb_id, occ):
            return cls._def_rows(cb_id, occ[0], irq) + [
                _row(occ[0]["reg"], INSTR_REF_OP, -1, cb_id, irq)
            ]

        def emit_ref(cb_id, s):
            return [_row(s["reg"], INSTR_REF_OP, -1, cb_id, irq)]

        emit_recurring(
            groups,
            minrep=1,
            group_sort=lambda kv: (min(s["frame"] for s in kv[1]), kv[0][0]),
            occ_sort=lambda s: min(s["rows"]),
            pos_of=lambda s: min(s["rows"]),
            rows_of=lambda s: s["rows"],
            emit_first=emit_first,
            emit_ref=emit_ref,
            start_id=next_id,
            drop_idx=drop_idx,
            new_rows=new_rows,
        )
        return drop_idx, new_rows

    @classmethod
    def _cross_voice(cls, spans, irq, drop_idx, new_rows, consumed):
        """Emit one shared DEF per signature recurring >= 2 times whose earliest occurrence is alone in
        its frame, so the DEF sorts strictly before every REF in both the pre-norm row order and the
        post-norm (frame, voice) order; same-frame groups fall to the per-voice phase. Returns next id.
        """
        groups = defaultdict(list)
        for s in spans:
            groups[s["sig"]].append(s)
        next_id = 0
        for _sig, occ in sorted(
            groups.items(), key=lambda kv: min(min(s["rows"]) for s in kv[1])
        ):
            if len(occ) < 2:
                continue
            occ = sorted(occ, key=lambda s: min(s["rows"]))
            head = occ[0]
            if any(s["frame"] == head["frame"] for s in occ[1:]):
                continue
            cb_id = next_id
            next_id += 1
            head_pos = min(head["rows"])
            head_rows = cls._def_rows(cb_id, head, irq) + [
                _row(head["reg"], INSTR_REF_OP, -1, cb_id, irq)
            ]
            for r in head_rows:
                r["__pos"] = head_pos
                new_rows.append(r)
            drop_idx.extend(head["rows"])
            consumed.add(id(head))
            for s in occ[1:]:
                ref = _row(s["reg"], INSTR_REF_OP, -1, cb_id, irq)
                ref["__pos"] = min(s["rows"])
                new_rows.append(ref)
                drop_idx.extend(s["rows"])
                consumed.add(id(s))
        return next_id

    @staticmethod
    def _def_rows(cb_id, span, irq):
        """Inline DEF: header (val=id), per-frame STEP atoms emitting only changed (ctrl, AD, SR) fields
        with an INSTR_SUBREG_FRAME advance between frames, terminated by INSTR_END (val=id). Voice-
        relative -- reg 0 on the DEF, the voice rides on the REF."""
        rows = [_row(0, INSTR_DEF_OP, -1, cb_id, irq)]
        seqs = (span["ctrls"], span["ads"], span["srs"])
        prev = [None, None, None]
        for i in range(len(span["ctrls"])):
            if i > 0:
                rows.append(_row(0, INSTR_STEP_OP, INSTR_SUBREG_FRAME, 0, irq))
            for j, off in enumerate(_FIELD_OFFSETS):
                val = seqs[j][i]
                if val != prev[j]:
                    rows.append(_row(0, INSTR_STEP_OP, off, val, irq))
                    prev[j] = val
        rows.append(_row(0, INSTR_END_OP, -1, cb_id, irq))
        return rows
