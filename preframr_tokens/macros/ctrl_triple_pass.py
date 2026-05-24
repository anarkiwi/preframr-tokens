"""Collapse three adjacent CTRL writes into one ``CTRL_TRIPLE`` atom (spec #6),
behind the ``ctrl_triple_pass`` arg flag (default OFF): three consecutive
same-voice CTRL SETs each one frame apart (no DELAY between) become a 3-byte
atom, extending CTRL_BIGRAM by one and run before it so triples win."""

__all__ = ["CtrlTriplePass"]

import numpy as np

from preframr_tokens.macros.passes_base import MacroPass, _splice_rows
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_TRIPLE_OP,
    CTRL_TRIPLE_SUBREG_0,
    CTRL_TRIPLE_SUBREG_1,
    CTRL_TRIPLE_SUBREG_2,
    DELAY_REG,
    FRAME_REG,
    SET_OP,
)


def _one_frame_apart(regs, i, j):
    between = regs[i + 1 : j]
    return not (between == DELAY_REG).any() and int((between == FRAME_REG).sum()) == 1


def _triple_rows(reg, bytes3, diff, irq):
    subs = (CTRL_TRIPLE_SUBREG_0, CTRL_TRIPLE_SUBREG_1, CTRL_TRIPLE_SUBREG_2)
    return [
        {
            "reg": int(reg),
            "val": int(b) & 0xFF,
            "diff": int(diff),
            "op": int(CTRL_TRIPLE_OP),
            "subreg": int(sub),
            "irq": int(irq),
            "description": 0,
        }
        for sub, b in zip(subs, bytes3)
    ]


class CtrlTriplePass(MacroPass):
    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        if args is None or not getattr(args, "ctrl_triple_pass", False):
            return df
        if "op" not in df.columns or "reg" not in df.columns:
            return df
        df = df.reset_index(drop=True).copy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        subregs = (
            df["subreg"].to_numpy() if "subreg" in df.columns else np.full(len(df), -1)
        )
        irq_default = (
            int(df["irq"].iloc[0])
            if "irq" in df.columns and len(df) and df["irq"].notna().any()
            else -1
        )
        if FRAME_REG not in regs:
            return df

        drop_idx = []
        new_rows = []
        for ctrl_reg in self.target_regs:
            positions = np.flatnonzero(
                (regs == ctrl_reg) & (ops == SET_OP) & (subregs == -1)
            )
            k = 0
            while k + 2 < len(positions):
                i, j, l = (int(positions[k + o]) for o in range(3))
                if _one_frame_apart(regs, i, j) and _one_frame_apart(regs, j, l):
                    diff = int(diffs[i]) if diffs is not None else 0
                    bytes3 = (int(vals[i]), int(vals[j]), int(vals[l]))
                    atom = _triple_rows(ctrl_reg, bytes3, diff, irq_default)
                    for nr in atom:
                        nr["__pos"] = i
                    new_rows.extend(atom)
                    drop_idx.extend((i, j, l))
                    k += 3
                else:
                    k += 1
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)
