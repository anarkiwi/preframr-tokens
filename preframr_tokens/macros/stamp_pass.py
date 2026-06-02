"""StampPass: percussion/effect stamp codebook (design/percussion_stamp_encoding.md). A drum is the
same exact register write-series stamped repeatedly; this proposer mines recurring exact (freq,ctrl)
gate-on spans from the immutable source and replaces each with an inline-redefinable STAMP_DEF +
per-hit STAMP_REF (a Claim), draining them to byte-exact stamps before the skeleton floor-snaps them
to RESID. Opt-in (``stamp_pass``), default OFF."""

__all__ = ["StampPass", "classify_char"]

import statistics
from collections import defaultdict

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.codebook_emit import emit_recurring
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    MacroPass,
)
from preframr_tokens.macros.skeleton_pass import SkeletonPass, fn_to_note_resid
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_REGS,
    SET_OP,
    STAMP_CHAR_CYMBAL,
    STAMP_CHAR_HAT,
    STAMP_CHAR_KICK,
    STAMP_CHAR_NOISE_FX,
    STAMP_CHAR_OTHER,
    STAMP_CHAR_PITCH_FX,
    STAMP_CHAR_SNARE,
    STAMP_CHAR_TOM,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_MINREP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    STAMP_REL_SUBREG_BASE_HI,
    STAMP_REL_SUBREG_BASE_LO,
    STAMP_REL_SUBREG_ID,
    STAMP_STEP_OP,
    STAMP_SUBREG_FRAME,
    VOICE_REG_SIZE,
)

_STAMP_PRIORITY = -10
_CTRL_OFFSET = 4
_FREQ_OFFSET = 0


def _is_pitched(ctrl):
    """A frame carries melodic pitch iff a pitched waveform (bits 4-6) is set with TEST (bit3) and
    NOISE (bit7) clear -- mirrors SkeletonPass._is_pitched_frame on an explicit ctrl byte.
    """
    return bool(ctrl & 0x70) and not (ctrl & 0x08) and not (ctrl & 0x80)


def classify_char(fns, ctrls):
    """Coarse drum character from the stamp's per-frame (freq, ctrl) writes -> a small global
    transfer vocabulary (KICK/TOM/SNARE/HAT/CYMBAL/NOISE_FX/PITCH_FX/OTHER), deterministic and
    tune-independent. Ported from the resid_drum_codebook prototype's ``character``."""
    n = len(fns)
    noise = sum(1 for c in ctrls if (c & 0x80) and not (c & 0x08))
    pitched = []
    for fn, c in zip(fns, ctrls):
        if not _is_pitched(c):
            continue
        res = fn_to_note_resid(int(fn))
        if res is not None:
            pitched.append(res[0])
    nfrac = noise / max(n, 1)
    pfrac = len(pitched) / max(n, 1)
    base = statistics.median(pitched) if pitched else 0
    sweep = (max(pitched) - min(pitched)) if len(pitched) >= 2 else 0
    down = bool(pitched) and pitched[0] - min(pitched) >= 6
    if nfrac >= 0.6 and pfrac < 0.3:
        return (
            STAMP_CHAR_HAT
            if n <= 5
            else (STAMP_CHAR_CYMBAL if n <= 16 else STAMP_CHAR_NOISE_FX)
        )
    if 0.2 <= nfrac < 0.8 and pfrac >= 0.2:
        return STAMP_CHAR_SNARE
    if pfrac >= 0.5 and down and sweep >= 12:
        return STAMP_CHAR_KICK if base <= 50 else STAMP_CHAR_TOM
    if pfrac >= 0.6 and sweep < 6:
        return STAMP_CHAR_PITCH_FX
    return STAMP_CHAR_OTHER


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


def _rel_sig(span):
    """Transpose-invariant signature: per-frame freq DELTA from the span's onset freq, plus the exact
    ctrl series -- two hits of one gesture at different base freqs share it (byte-exact via raw delta,
    no semitone snap)."""
    base = span["fns"][0]
    return (tuple(fn - base for fn in span["fns"]), tuple(span["ctrls"]))


class StampPass(MacroPass):
    """Mine recurring exact (freq,ctrl) write-series per voice and replace them with inline
    STAMP_DEF + STAMP_REF, consuming the raw writes. Default OFF."""

    GATE_FLAGS = frozenset({"stamp_pass"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "stamp_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        if (
            "op" in df.columns
            and df["op"].isin((STAMP_DEF_OP, STAMP_REF_OP, STAMP_REL_REF_OP)).any()
        ):
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        freq_sets, gate_on, ctrl_writes = SkeletonPass._context(df)
        ctrl_rows = self._ctrl_rows(df)
        end_frame = int(_frame_index(df).max()) + 1
        spans = []
        for reg in FREQ_TRAJ_REGS:
            spans.extend(
                self._reg_spans(
                    int(reg), freq_sets, gate_on, ctrl_writes, ctrl_rows, end_frame
                )
            )
        groups = defaultdict(list)
        for span in spans:
            groups[(span["reg"], span["sig"])].append(span)
        drop_idx, new_rows = self._emit(groups, irq)
        if not new_rows:
            return df
        result = arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_STAMP_PRIORITY,
                    label="stamp",
                )
            ],
        )
        if not self._stamp_is_lossless(df, result):
            return df
        return result

    @staticmethod
    def _stamp_is_lossless(df_in, df_out):
        """Decode both streams and require byte-exact per-frame register_state. A stamp must
        reproduce the writes it consumes exactly; if the codebook replay diverges (the decoder's
        per-frame drain can mis-align freq across hits) fall back to the un-stamped stream so the
        render stays faithful -- the register-level fidelity oracle, not audio."""
        from preframr_tokens.audit_primitives import register_state

        before = register_state(df_in.copy())
        after = register_state(df_out.copy())
        return before.shape == after.shape and not (before != after).any()

    @staticmethod
    def _ctrl_rows(df):
        """Per ctrl-reg ordered (frame, row_index, val) SET writes -- the row indices a stamp claim
        consumes (SkeletonPass._context drops them)."""
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        ctrl_regs = {int(reg) + _CTRL_OFFSET for reg in FREQ_TRAJ_REGS}
        out = defaultdict(list)
        for i in range(len(df)):
            reg = int(regs[i])
            if reg in ctrl_regs and int(ops[i]) == SET_OP and int(subregs[i]) == -1:
                out[reg].append((int(f_idx[i]), int(i), int(vals[i])))
        return out

    @classmethod
    def _reg_spans(cls, reg, freq_sets, gate_on, ctrl_writes, ctrl_rows, end_frame):
        """Note-spans for one voice freq reg, bounded by gate-on retriggers (the true hit boundary
        -- a freq level-change within a gated hit must not split it; fall back to level-change
        onsets only when the voice never gates). Each (onset, next_onset) becomes a span with its
        byte-exact per-frame (freq,ctrl) signature and the freq+ctrl row indices it consumes (freq
        held across a hit is forward-filled from the last write at/before onset)."""
        sets = freq_sets[reg]
        if not sets:
            return []
        onset_set = set(gate_on[reg])
        if not onset_set:
            onset_set = {fr for fr, _ in SkeletonPass._segment_notes(sets, set())}
        if not onset_set:
            return []
        onset_frames = sorted(f for f in onset_set if f < end_frame) + [end_frame]
        cwrites = ctrl_writes[reg]
        crows = ctrl_rows.get(reg + _CTRL_OFFSET, [])
        out = []
        for k in range(len(onset_frames) - 1):
            span = cls._build_span(
                reg, onset_frames[k], onset_frames[k + 1], sets, cwrites, crows
            )
            if span is not None:
                out.append(span)
        return out

    @classmethod
    def _build_span(cls, reg, onset, end, sets, cwrites, crows):
        if end - onset <= 0:
            return None
        ctrl0 = SkeletonPass._ctrl_at(cwrites, onset)
        if ctrl0 is None:
            return None
        fn0 = cls._freq_at(sets, onset)
        if fn0 is None:
            return None
        fwrite = {s[0]: (int(s[1]), int(s[2])) for s in sets if onset <= s[0] < end}
        cwrite = {fr: (ri, v) for fr, ri, v in crows if onset <= fr < end}
        write_frames = set(fwrite) | set(cwrite)
        if not write_frames:
            return None
        length = max(write_frames) - onset + 1
        fns, ctrls = [], []
        cur_fn, cur_ctrl = fn0, ctrl0
        for i in range(length):
            fr = onset + i
            if fr in fwrite:
                cur_fn = fwrite[fr][1]
            if fr in cwrite:
                cur_ctrl = cwrite[fr][1]
            fns.append(cur_fn)
            ctrls.append(cur_ctrl)
        rows = [fwrite[fr][0] for fr in fwrite] + [cwrite[fr][0] for fr in cwrite]
        if not rows:
            return None
        order = (_FREQ_OFFSET, _CTRL_OFFSET)
        for fr in sorted(write_frames):
            if fr in fwrite and fr in cwrite:
                if cwrite[fr][0] < fwrite[fr][0]:
                    order = (_CTRL_OFFSET, _FREQ_OFFSET)
                break
        return {
            "reg": reg,
            "onset": onset,
            "sig": tuple(zip(fns, ctrls)),
            "fns": fns,
            "ctrls": ctrls,
            "rows": rows,
            "order": order,
        }

    @staticmethod
    def _freq_at(sets, frame):
        """Forward-filled freq value in effect at ``frame`` (last freq SET at or before it)."""
        cur = None
        for s in sets:
            if s[0] > frame:
                break
            cur = int(s[2])
        return cur

    def _emit(self, groups, irq):
        """Two passes over the shared recurring-codebook skeleton: exact ABS first (byte-identical to
        before), then transpose-relative REL over the spans ABS did not consume -- a pitched gesture
        played at different base freqs shares one REL def (freq stored as deltas from onset) with
        per-hit base; drains spans no single exact sig reaches MINREP for."""
        consumed = set()

        def abs_first(stamp_id, occ):
            char = classify_char(occ[0]["fns"], occ[0]["ctrls"])
            return self._def_rows(stamp_id, char, occ[0], irq) + [
                _row(occ[0]["reg"], STAMP_REF_OP, -1, stamp_id, irq, irq)
            ]

        def abs_ref(stamp_id, span):
            return [_row(span["reg"], STAMP_REF_OP, -1, stamp_id, irq, irq)]

        drop_idx, new_rows, next_id = emit_recurring(
            groups,
            minrep=STAMP_MINREP,
            group_sort=lambda kv: (min(s["onset"] for s in kv[1]), kv[0][0]),
            occ_sort=lambda s: s["onset"],
            pos_of=lambda s: min(s["rows"]),
            rows_of=lambda s: s["rows"],
            emit_first=abs_first,
            emit_ref=abs_ref,
            consumed=consumed,
        )

        rel_groups = defaultdict(list)
        for occ in groups.values():
            for span in occ:
                if id(span) not in consumed:
                    rel_groups[_rel_sig(span)].append(span)

        def rel_first(stamp_id, occ):
            char = classify_char(occ[0]["fns"], occ[0]["ctrls"])
            return self._rel_def_rows(stamp_id, char, occ[0], irq) + self._rel_ref_rows(
                occ[0], stamp_id, irq
            )

        def rel_ref(stamp_id, span):
            return self._rel_ref_rows(span, stamp_id, irq)

        emit_recurring(
            rel_groups,
            minrep=STAMP_MINREP,
            group_sort=lambda kv: min(s["onset"] for s in kv[1]),
            occ_sort=lambda s: s["onset"],
            pos_of=lambda s: min(s["rows"]),
            rows_of=lambda s: s["rows"],
            emit_first=rel_first,
            emit_ref=rel_ref,
            start_id=next_id,
            drop_idx=drop_idx,
            new_rows=new_rows,
        )
        return drop_idx, new_rows

    @staticmethod
    def _rel_def_rows(stamp_id, char, span, irq):
        """Like _def_rows but freq is stored as a signed 16-bit DELTA from the span's onset freq
        (frame 0 delta = 0), so any transposition of the gesture replays as base + delta.
        """
        rows = [_row(0, STAMP_DEF_OP, char, stamp_id, irq, irq)]
        fns, ctrls = span["fns"], span["ctrls"]
        base = fns[0]
        prev_d = prev_ctrl = None
        for i in range(len(fns)):
            if i > 0:
                rows.append(_row(0, STAMP_STEP_OP, STAMP_SUBREG_FRAME, 0, irq, irq))
            delta = (fns[i] - base) & 0xFFFF
            for off in span["order"]:
                if off == _FREQ_OFFSET and delta != prev_d:
                    rows.append(_row(0, STAMP_STEP_OP, _FREQ_OFFSET, delta, irq, irq))
                    prev_d = delta
                elif off == _CTRL_OFFSET and ctrls[i] != prev_ctrl:
                    rows.append(
                        _row(0, STAMP_STEP_OP, _CTRL_OFFSET, ctrls[i], irq, irq)
                    )
                    prev_ctrl = ctrls[i]
        rows.append(_row(0, STAMP_END_OP, -1, stamp_id, irq, irq))
        return rows

    @staticmethod
    def _rel_ref_rows(span, stamp_id, irq):
        """A REL hit: the stamp id plus this occurrence's onset (base) freq as hi/lo atoms."""
        base = int(span["fns"][0]) & 0xFFFF
        reg = span["reg"]
        return [
            _row(reg, STAMP_REL_REF_OP, STAMP_REL_SUBREG_ID, stamp_id, irq, irq),
            _row(reg, STAMP_REL_REF_OP, STAMP_REL_SUBREG_BASE_HI, base >> 8, irq, irq),
            _row(
                reg, STAMP_REL_REF_OP, STAMP_REL_SUBREG_BASE_LO, base & 0xFF, irq, irq
            ),
        ]

    @staticmethod
    def _def_rows(stamp_id, char, span, irq):
        """Inline def: a header (val=id, subreg=char) then per-internal-frame STEP atoms emitting
        freq (offset 0) and ctrl (offset 4) only when they change, a frame-advance STEP between
        frames, terminated by STAMP_END. Voice-relative (no voice on the def)."""
        rows = [_row(0, STAMP_DEF_OP, char, stamp_id, irq, irq)]
        prev_fn = prev_ctrl = None
        fns, ctrls = span["fns"], span["ctrls"]
        for i in range(len(fns)):
            if i > 0:
                rows.append(_row(0, STAMP_STEP_OP, STAMP_SUBREG_FRAME, 0, irq, irq))
            for off in span["order"]:
                if off == _FREQ_OFFSET and fns[i] != prev_fn:
                    rows.append(_row(0, STAMP_STEP_OP, _FREQ_OFFSET, fns[i], irq, irq))
                    prev_fn = fns[i]
                elif off == _CTRL_OFFSET and ctrls[i] != prev_ctrl:
                    rows.append(
                        _row(0, STAMP_STEP_OP, _CTRL_OFFSET, ctrls[i], irq, irq)
                    )
                    prev_ctrl = ctrls[i]
        rows.append(_row(0, STAMP_END_OP, -1, stamp_id, irq, irq))
        return rows
