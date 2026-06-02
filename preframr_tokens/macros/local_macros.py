"""Encoder passes for local-context macros (CtrlBigramPass)."""

from __future__ import annotations

__all__ = ["CtrlBigramPass"]

import numpy as np

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_PAIR_TO_IDX,
    DELAY_REG,
    FRAME_REG,
    SET_OP,
)


class CtrlBigramPass(MacroPass):
    GATE_FLAGS = frozenset({"ctrl_bigram_pass"})
    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        if args is None or not getattr(args, "ctrl_bigram_pass", False):
            return df
        if "op" not in df.columns or "reg" not in df.columns:
            return df
        df = df.reset_index(drop=True).copy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irqs = df["irq"].to_numpy() if "irq" in df.columns else None
        subregs = (
            df["subreg"].to_numpy() if "subreg" in df.columns else np.full(len(df), -1)
        )
        if not (FRAME_REG in regs):
            return df

        claims = []
        for ctrl_reg in self.target_regs:
            ctrl_row_mask = (regs == ctrl_reg) & (ops == SET_OP) & (subregs == -1)
            row_positions = np.flatnonzero(ctrl_row_mask)
            if len(row_positions) < 2:
                continue
            k = 0
            while k + 1 < len(row_positions):
                i = int(row_positions[k])
                j = int(row_positions[k + 1])
                between = regs[i + 1 : j]
                if (between == DELAY_REG).any() or int(
                    (between == FRAME_REG).sum()
                ) != 1:
                    k += 1
                    continue
                if k + 2 < len(row_positions):
                    nxt = int(row_positions[k + 2])
                    if int((regs[j + 1 : nxt] == FRAME_REG).sum()) == 0:
                        k += 1
                        continue
                pair = (int(vals[i]) & 0xFF, int(vals[j]) & 0xFF)
                idx = CTRL_BIGRAM_PAIR_TO_IDX.get(pair)
                if idx is None:
                    k += 1
                    continue
                atom = {
                    "reg": int(ctrl_reg),
                    "val": int(idx),
                    "op": int(CTRL_BIGRAM_OP),
                    "subreg": -1,
                    "diff": int(diffs[i]) if diffs is not None else 0,
                    "irq": int(irqs[i]) if irqs is not None else 0,
                    "__pos": i,
                }
                claims.append(Claim(writes=(i, j), tokens=[atom], label="ctrl_bigram"))
                k += 2

        if not claims:
            return df
        return arbitrate(df, claims)
