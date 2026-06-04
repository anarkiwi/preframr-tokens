"""NoteOffPass (IMPLEMENT_residual_set_elimination PR3): re-label each gate-transition ctrl write as a
named note event -- a gate FALL (bit0 1->0, ``note_off``) ends a note, a gate RISE (0->1, ``note_on``)
starts one. A 1:1 inline re-emission (same reg, value, frame and intra-frame position), byte-exact in
every sense (raw write events included), perturbing nothing downstream; it only re-labels an otherwise
unlearnable literal ctrl poke as the named onset/release event. Default OFF; decode in ``decoders.py``.
"""

__all__ = ["NoteOffPass"]

import numpy as np

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    NOTE_OFF_OP,
    NOTE_ON_OP,
    SET_OP,
)

_NOTE_OFF_PRIORITY = -4


class NoteOffPass(MacroPass):
    """Re-label each gate-transition ctrl write as a NOTE_OFF (fall) or NOTE_ON (rise) atom. A pure 1:1
    tag of a single ctrl SET, byte-exact and pipeline-transparent. Default OFF."""

    GATE_FLAGS = frozenset({"note_off", "note_on"})

    def apply(self, df, args=None):
        """One Claim per gate-transition ctrl write, arbitrated with validate=True. Each Claim drops one
        ctrl SET and emits one NOTE_OFF/NOTE_ON at the same position carrying the same byte, so decode is
        identical."""
        if args is None:
            return df
        want_off = bool(getattr(args, "note_off", False))
        want_on = bool(getattr(args, "note_on", False))
        if not (want_off or want_on) or df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        claims = self._claims(df, irq, want_off, want_on)
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _decoded_frame_index(regs, vals):
        """Cumulative decoded frame per row (FRAME=+1, DELAY=+val), matching register_state's expansion
        so a row's frame indexes into the decoded gate timeline."""
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

    @classmethod
    def _claims(cls, df, irq, want_off, want_on):
        """One Claim per full-byte ctrl SET whose gate bit transitions vs the prior frame: a FALL (gate
        now 0, prior 1) is a NOTE_OFF, a RISE (now 1, prior 0) a NOTE_ON. The prior gate is read from the
        DECODED ctrl timeline (register_state), so a transition produced by an atom (hard restart, ctrl
        oscillation, drum stamp) is seen, not just plain SETs. register_state carries a leading initial
        frame, so ``state[f]`` is the END of the decoded frame BEFORE row-frame ``f``.
        """
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        state = register_state(df)
        n_frames = state.shape[0]
        frame_idx = cls._decoded_frame_index(regs, vals)
        ctrl_regs = {int(r) for r in CTRL_REGS_BY_VOICE}
        claims = []
        for i in range(len(df)):
            reg = int(regs[i])
            if reg not in ctrl_regs or int(ops[i]) != SET_OP or int(subregs[i]) != -1:
                continue
            val = int(vals[i])
            f = int(frame_idx[i])
            if f <= 0 or f >= n_frames:
                continue
            prior = int(state[f, reg]) & 1
            if want_off and not (val & 1) and prior:
                op = NOTE_OFF_OP
            elif want_on and (val & 1) and not prior:
                op = NOTE_ON_OP
            else:
                continue
            row = make_row(reg, val, op=op, diff=irq, irq=irq)
            row["__pos"] = int(i)
            claims.append(
                Claim(
                    writes=(int(i),),
                    tokens=[row],
                    priority=_NOTE_OFF_PRIORITY,
                    label="note_gate",
                )
            )
        return claims
