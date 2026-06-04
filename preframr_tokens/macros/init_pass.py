"""InitPass: relabel the driver's one-time init-routine register writes -- the plain SETs on the
single-byte value regs (ctrl/AD/SR/res-filt/master-vol, ``SUBREG_REGS``) in the frames BEFORE the
first note-on -- as an INIT_OP atom, byte-identical to the literal SET (the decoder re-emits it
inline). A named preamble for the chip-setup writes, bounded by the first note (via register_state, so
it never absorbs playback). Runs last over surviving SETs only. Default OFF (``init_preamble``).
"""

__all__ = ["InitPass"]

import numpy as np

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.passes_base import _ensure_subreg, MacroPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE, SUBREG_REGS
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    INIT_MAX_FRAMES,
    INIT_OP,
    SET_OP,
)

_CTRL_REGS = tuple(int(r) for r in CTRL_REGS_BY_VOICE)
_INIT_REGS = frozenset(int(r) for r in SUBREG_REGS)


class InitPass(MacroPass):
    """Relabel surviving pre-first-note-on SETs on the single-byte value regs as INIT_OP (the driver
    init routine). The note-on boundary comes from register_state so heterogeneous note atoms do not
    hide it; bounded by INIT_MAX_FRAMES. Default OFF (``init_preamble``)."""

    GATE_FLAGS = frozenset({"init_preamble"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "init_preamble", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        boundary = self._first_note_on_frame(df)
        if boundary is None or boundary <= 0:
            return df
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subs = df["subreg"].to_numpy()
        frame = self._decoded_frame(df)
        op_col = ops.copy()
        for i in range(len(df)):
            if (
                int(frame[i]) < boundary
                and int(regs[i]) in _INIT_REGS
                and int(ops[i]) == SET_OP
                and int(subs[i]) == -1
            ):
                op_col[i] = int(INIT_OP)
        df["op"] = op_col
        return df

    @staticmethod
    def _decoded_frame(df):
        """Cumulative decoded frame per row (FRAME=+1, DELAY=+val), matching register_state expansion."""
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        out = np.empty(len(df), dtype=np.int64)
        f = -1
        for i in range(len(df)):
            r = int(regs[i])
            if r == FRAME_REG:
                f += 1
            elif r == DELAY_REG:
                f += max(1, int(vals[i]))
            out[i] = f
        return out

    @classmethod
    def _first_note_on_frame(cls, df):
        """Decoded frame of the first note-on (first frame where any voice ctrl gate bit is set), or
        None if there is none or it is beyond INIT_MAX_FRAMES (a note-less intro that must not be
        swallowed). Read from register_state so it is robust to how ctrl writes are op-labelled.
        """
        try:
            state = register_state(df)
        except Exception:  # noqa: BLE001
            return None
        frames = state.shape[0]
        limit = min(frames, INIT_MAX_FRAMES + 1)
        for f in range(limit):
            for creg in _CTRL_REGS:
                if creg < state.shape[1] and (int(state[f, creg]) & 1) == 1:
                    return f
        return None
