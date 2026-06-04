"""CtrlWavetablePass (IMPLEMENT_residual_set_elimination PR2 + PR4(b), length-1 form): per-voice
register-state codebook, the twin of PatchPass. Any voice ctrl byte (``ctrl_wavetable``) or AD/SR
envelope byte (``env_wavetable``) recurring >= CTRL_WT_MINREP times drains to an inline CTRL_WT_DEF +
per-reuse CTRL_WT_SET re-emitting the same write -- a learnable per-tune state alphabet. Default OFF;
decode in ``decoders.py``.
"""

__all__ = ["CtrlWavetablePass"]

from collections import defaultdict

import numpy as np

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.codebook_emit import emit_recurring
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    PWM_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
    VOICES,
)
from preframr_tokens.stfconstants import (
    _MIN_DIFF,
    CTRL_WT_DEF_OP,
    CTRL_WT_MINREP,
    CTRL_WT_SET_OP,
    CTRL_WT_STEP_OP,
    CTRL_WT_SUBREG_ID,
    CTRL_WT_SUBREG_VAL,
    DELAY_REG,
    FC_LO_REG,
    FILTER_REG,
    FRAME_REG,
    MODE_VOL_REG,
    SET_OP,
    VOICE_REG_SIZE,
)

_FILTER_REGS = (int(FC_LO_REG), int(FC_LO_REG) + 1, int(FILTER_REG))

_CTRL_WT_PRIORITY = -3


def _row(reg, op, subreg, val, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=_MIN_DIFF, irq=irq)


def _max_def_id(df):
    """Largest CTRL_WT codebook id already defined in ``df`` (-1 if none), so a re-mine pass
    allocates fresh ids above it instead of colliding with committed DEFs."""
    ops = df["op"].to_numpy()
    subs = df["subreg"].to_numpy()
    vals = df["val"].to_numpy()
    m = -1
    for i in range(len(df)):
        if int(ops[i]) == CTRL_WT_DEF_OP and int(subs[i]) == CTRL_WT_SUBREG_ID:
            m = max(m, int(vals[i]))
    return m


class CtrlWavetablePass(MacroPass):
    """Mine recurring per-reg bytes -- voice ctrl (``ctrl_wavetable``), AD/SR (``env_wavetable``),
    filter cutoff/resonance (``filter_wavetable``), master mode/vol (``modevol_wavetable``) -- and
    replace them with an inline CTRL_WT_DEF + per-reuse CTRL_WT_SET. One codebook, one coordinated id
    space across all register classes. Default OFF."""

    GATE_FLAGS = frozenset(
        {
            "ctrl_wavetable",
            "env_wavetable",
            "filter_wavetable",
            "modevol_wavetable",
            "freq_wavetable",
            "pw_wavetable",
            "onset_instrument",
            "onset_def",
        }
    )

    def apply(self, df, args=None):
        if args is None:
            return df
        target = []
        if getattr(args, "ctrl_wavetable", False):
            target.extend(int(r) for r in CTRL_REGS_BY_VOICE)
        if getattr(args, "env_wavetable", False):
            target.extend(int(r) for r in AD_REGS_BY_VOICE)
            target.extend(int(r) for r in SR_REGS_BY_VOICE)
        if getattr(args, "filter_wavetable", False):
            target.extend(_FILTER_REGS)
        if getattr(args, "modevol_wavetable", False):
            target.append(int(MODE_VOL_REG))
        if getattr(args, "freq_wavetable", False):
            target.extend(int(r) for r in FREQ_REGS_BY_VOICE)
        if getattr(args, "pw_wavetable", False):
            target.extend(int(r) for r in PWM_REGS_BY_VOICE)
        onset = getattr(args, "onset_instrument", False)
        onset_def = getattr(args, "onset_def", False)
        if (not target and not onset and not onset_def) or df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        target_set = set(target)
        base = _max_def_id(df) + 1
        events = self._events(df, target_set)
        claims = self._emit(events, irq, start_id=base) if target_set else []
        if onset:
            onset_start = max(base, self._next_id(claims))
            claims.extend(self._onset_claims(df, irq, onset_start))
        if onset_def:
            def_start = max(base, self._next_id(claims))
            claims.extend(self._onset_def_claims(df, irq, def_start, claims))
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _next_id(claims):
        """The smallest unused codebook id after the write-phase claims, so the onset phase shares the
        coordinated id space (a CTRL_WT_DEF carries its id in val at subreg CTRL_WT_SUBREG_ID).
        """
        nxt = 0
        for c in claims:
            for t in c.tokens:
                if (
                    int(t["op"]) == CTRL_WT_DEF_OP
                    and int(t["subreg"]) == CTRL_WT_SUBREG_ID
                ):
                    nxt = max(nxt, int(t["val"]) + 1)
        return nxt

    @staticmethod
    def _events(df, target_regs):
        """Every plain full-byte SET on a target reg as {reg, val, row, pos, frame}. ``pos`` (row index)
        orders a DEF before its reuses pre-norm; ``frame`` (decoded real-frame) gates the cross-voice
        phase so a shared-instrument DEF stays before its refs under the voice-major _norm_pr_order.
        """
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        events = []
        rf = -1
        for i in range(len(df)):
            r = int(regs[i])
            if r == FRAME_REG:
                rf += 1
            elif r == DELAY_REG:
                rf += max(1, int(vals[i]))
            elif int(ops[i]) == SET_OP and int(subregs[i]) == -1 and r in target_regs:
                events.append(
                    {
                        "reg": r,
                        "val": int(vals[i]),
                        "row": int(i),
                        "pos": int(i),
                        "frame": rf,
                    }
                )
        return events

    @staticmethod
    def _reg_class(reg):
        """A voice reg's class (offset within its voice) so the same waveform/envelope byte on any voice
        keys to one cross-voice id; global regs (filter/mode-vol) key on themselves."""
        return reg % VOICE_REG_SIZE if reg < 21 else reg

    @classmethod
    def _emit(cls, events, irq, start_id=0):
        """Two phases sharing one id space. Phase 1 keys per-voice (reg, val) -- always ordering-safe.
        Phase 2 keys the per-voice SINGLETONS cross-voice (reg_class, val) -- a shared instrument used
        once per voice -- emitting a group only when its DEF's frame is strictly earliest (no same-frame
        occurrence), which keeps the DEF before every ref in both the pre-norm row order and the
        post-norm (frame, voice) order."""
        groups = defaultdict(list)
        for ev in events:
            groups[(ev["reg"], ev["val"])].append(ev)

        def emit_first(cb_id, occ):
            ev = occ[0]
            return [
                _row(ev["reg"], CTRL_WT_DEF_OP, CTRL_WT_SUBREG_ID, cb_id, irq),
                _row(ev["reg"], CTRL_WT_STEP_OP, CTRL_WT_SUBREG_VAL, ev["val"], irq),
            ]

        def emit_ref(cb_id, ev):
            return [_row(ev["reg"], CTRL_WT_SET_OP, -1, cb_id, irq)]

        consumed: set = set()
        grouped, next_id = emit_recurring(
            groups,
            start_id=start_id,
            minrep=CTRL_WT_MINREP,
            group_sort=lambda kv: (min(e["pos"] for e in kv[1]), kv[0]),
            occ_sort=lambda e: e["pos"],
            pos_of=lambda e: e["pos"],
            rows_of=lambda e: (e["row"],),
            emit_first=emit_first,
            emit_ref=emit_ref,
            consumed=consumed,
            per_group=True,
        )
        claims = [
            Claim(tuple(d), r, priority=_CTRL_WT_PRIORITY, label="ctrl_wt")
            for d, r in grouped
        ]
        claims.extend(
            cls._cross_voice_claims(
                [ev for ev in events if id(ev) not in consumed], next_id, irq
            )
        )
        return claims

    @classmethod
    def _cross_voice_claims(cls, leftover, start_id, irq):
        """Cross-voice claims from the per-voice singletons: group by (reg_class, val), and for each
        group of >= MINREP whose earliest occurrence is alone in its frame, emit one shared DEF (on that
        occurrence) + a CTRL_WT_SET on every other occurrence's own reg."""
        groups = defaultdict(list)
        for ev in leftover:
            groups[(cls._reg_class(ev["reg"]), ev["val"])].append(ev)
        claims = []
        next_id = start_id
        for _gkey, occ in sorted(
            groups.items(), key=lambda kv: min(e["pos"] for e in kv[1])
        ):
            if len(occ) < CTRL_WT_MINREP:
                continue
            occ = sorted(occ, key=lambda e: e["pos"])
            head = occ[0]
            if any(o["frame"] == head["frame"] for o in occ[1:]):
                continue
            cb_id = next_id
            next_id += 1
            rows = [
                _row(head["reg"], CTRL_WT_DEF_OP, CTRL_WT_SUBREG_ID, cb_id, irq),
                _row(
                    head["reg"], CTRL_WT_STEP_OP, CTRL_WT_SUBREG_VAL, head["val"], irq
                ),
            ]
            for r in rows:
                r["__pos"] = head["pos"]
            for ev in occ[1:]:
                ref = _row(ev["reg"], CTRL_WT_SET_OP, -1, cb_id, irq)
                ref["__pos"] = ev["pos"]
                rows.append(ref)
            claims.append(
                Claim(
                    tuple(int(o["row"]) for o in occ),
                    rows,
                    priority=_CTRL_WT_PRIORITY,
                    label="ctrl_wt_xv",
                )
            )
        return claims

    @classmethod
    def _onset_claims(cls, df, irq, start_id):
        """Note onsets reusing a HELD instrument: an AD/SR/CTRL value written ONCE but in effect at
        >= CTRL_WT_MINREP gate-rise onsets is the note's instrument, reused by gate retriggers that do
        not rewrite it. Emit a DEF at its setup write + a CTRL_WT_SET at each reusing onset (re-emitting
        the held value -- register-exact, the arbiter validates). Drains the one setup write and links
        the onsets to the instrument."""
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        state = register_state(df)
        ndec = state.shape[0] - 1
        if ndec < 1:
            return []
        frame_idx = cls._row_frames(regs, vals)
        frame_anchor = {}
        once_writes = {}
        seen = defaultdict(int)
        for i in range(len(df)):
            d = int(frame_idx[i])
            if d not in frame_anchor:
                frame_anchor[d] = i
            r = int(regs[i])
            if int(ops[i]) != SET_OP or int(subs[i]) != -1:
                continue
            key = (r, int(vals[i]))
            seen[key] += 1
            once_writes[key] = i
        claims = []
        next_id = start_id
        for v in range(VOICES):
            creg = int(CTRL_REGS_BY_VOICE[v])
            onsets = [
                d
                for d in range(ndec)
                if (int(state[d + 1, creg]) & 1) and not (int(state[d, creg]) & 1)
            ]
            if len(onsets) < CTRL_WT_MINREP:
                continue
            for reg in (int(AD_REGS_BY_VOICE[v]), int(SR_REGS_BY_VOICE[v]), creg):
                for val, n in [
                    (kv[1], kv2) for kv, kv2 in seen.items() if kv[0] == reg
                ]:
                    if n != 1:
                        continue
                    setup_row = once_writes[(reg, val)]
                    ws = int(frame_idx[setup_row])
                    uses = [
                        d
                        for d in onsets
                        if d > ws
                        and int(state[d + 1, reg]) == val
                        and d in frame_anchor
                    ]
                    if len(uses) < CTRL_WT_MINREP:
                        continue
                    rows = [
                        _row(reg, CTRL_WT_DEF_OP, CTRL_WT_SUBREG_ID, next_id, irq),
                        _row(reg, CTRL_WT_STEP_OP, CTRL_WT_SUBREG_VAL, val, irq),
                    ]
                    for r in rows:
                        r["__pos"] = int(setup_row)
                    for d in uses:
                        ref = _row(reg, CTRL_WT_SET_OP, -1, next_id, irq)
                        ref["__pos"] = int(frame_anchor[d])
                        rows.append(ref)
                    claims.append(
                        Claim(
                            (int(setup_row),),
                            rows,
                            priority=_CTRL_WT_PRIORITY,
                            label="onset_inst",
                        )
                    )
                    next_id += 1
        return claims

    @staticmethod
    def _onset_def_targets():
        """The instrument-config register set a note onset writes: per-voice ctrl/AD/SR/freq/PW plus the
        global filter (cutoff/resonance) and master mode-vol -- the regs whose once-written values are
        the per-tune instrument alphabet."""
        regs = set(_FILTER_REGS)
        regs.add(int(MODE_VOL_REG))
        for byvoice in (
            CTRL_REGS_BY_VOICE,
            AD_REGS_BY_VOICE,
            SR_REGS_BY_VOICE,
            FREQ_REGS_BY_VOICE,
            PWM_REGS_BY_VOICE,
        ):
            regs.update(int(r) for r in byvoice)
        return regs

    @classmethod
    def _first_onset_frame(cls, df):
        """Decoded frame of the earliest gate-rise across voices (the first note-on), or 0 if none --
        the floor below which writes are the driver init preamble (InitPass's province, not an onset).
        """
        state = register_state(df)
        first = None
        for v in range(VOICES):
            creg = int(CTRL_REGS_BY_VOICE[v])
            for d in range(state.shape[0] - 1):
                if (int(state[d + 1, creg]) & 1) and not (int(state[d, creg]) & 1):
                    first = d if first is None else min(first, d)
                    break
        return int(first or 0)

    @classmethod
    def _onset_def_claims(cls, df, irq, start_id, prior):
        """Define-on-first: a single-reg instrument SET written ONCE and unclaimed by the recurrence
        phases is still a codebook entry -- emit a lone CTRL_WT_DEF + STEP (the STEP re-emits the write
        byte-exactly, the arbiter validates) so it leaves the raw-SET residual as a named define. Scoped
        to one-write-per-frame writes at/after the first onset (HARD_RESTART owns multiwrites, InitPass
        the pre-onset preamble)."""
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        target = cls._onset_def_targets()
        frame_idx = cls._row_frames(regs, vals)
        floor = cls._first_onset_frame(df)
        claimed = {int(w) for c in prior for w in c.writes}
        fr_reg_count = defaultdict(int)
        for i in range(len(df)):
            if int(ops[i]) == SET_OP and int(subs[i]) == -1 and int(regs[i]) in target:
                fr_reg_count[(int(frame_idx[i]), int(regs[i]))] += 1
        claims = []
        next_id = start_id
        for i in range(len(df)):
            if int(ops[i]) != SET_OP or int(subs[i]) != -1 or int(i) in claimed:
                continue
            r = int(regs[i])
            f = int(frame_idx[i])
            if r not in target or f < floor or fr_reg_count[(f, r)] != 1:
                continue
            rows = [
                _row(r, CTRL_WT_DEF_OP, CTRL_WT_SUBREG_ID, next_id, irq),
                _row(r, CTRL_WT_STEP_OP, CTRL_WT_SUBREG_VAL, int(vals[i]), irq),
            ]
            for row in rows:
                row["__pos"] = int(i)
            claims.append(
                Claim(
                    (int(i),), rows, priority=_CTRL_WT_PRIORITY + 1, label="onset_def"
                )
            )
            next_id += 1
        return claims

    @staticmethod
    def _row_frames(regs, vals):
        """Decoded real-frame index per row (FRAME=+1, DELAY=+val), starting at -1 -- so frame ``d``
        ends at register_state row ``d+1`` (register_state carries a leading initial frame).
        """
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
