"""Encoder passes for local-context macros (CtrlBigramPass)."""

from __future__ import annotations

import numpy as np

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
        subregs = (
            df["subreg"].to_numpy() if "subreg" in df.columns else np.full(len(df), -1)
        )
        if not (FRAME_REG in regs):
            return df

        drop_idx = []
        op_overrides = {}
        val_overrides = {}
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
                pair = (int(vals[i]) & 0xFF, int(vals[j]) & 0xFF)
                idx = CTRL_BIGRAM_PAIR_TO_IDX.get(pair)
                if idx is None:
                    k += 1
                    continue
                op_overrides[i] = int(CTRL_BIGRAM_OP)
                val_overrides[i] = int(idx)
                drop_idx.append(j)
                k += 2

        if not drop_idx:
            return df
        new_ops = ops.copy()
        new_vals = vals.copy()
        for pos, op_v in op_overrides.items():
            new_ops[pos] = op_v
            new_vals[pos] = val_overrides[pos]
        df["op"] = new_ops
        df["val"] = new_vals
        return df.drop(index=drop_idx).reset_index(drop=True)
