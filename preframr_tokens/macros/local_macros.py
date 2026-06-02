"""Encoder passes for local-context macros (CtrlBigramPass)."""

from __future__ import annotations

__all__ = ["CtrlBigramPass"]

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.macros.run_collapse import collapse_runs
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import CTRL_BIGRAM_OP, CTRL_BIGRAM_PAIR_TO_IDX


class CtrlBigramPass(MacroPass):
    GATE_FLAGS = frozenset({"ctrl_bigram_pass"})
    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        if args is None or not getattr(args, "ctrl_bigram_pass", False):
            return df
        if "op" not in df.columns or "reg" not in df.columns:
            return df
        df = df.reset_index(drop=True).copy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irqs = df["irq"].to_numpy() if "irq" in df.columns else None

        def build_atom(reg, idxs):
            i, j = idxs
            pair = (int(vals[i]) & 0xFF, int(vals[j]) & 0xFF)
            idx = CTRL_BIGRAM_PAIR_TO_IDX.get(pair)
            if idx is None:
                return None
            return [
                {
                    "reg": int(reg),
                    "val": int(idx),
                    "op": int(CTRL_BIGRAM_OP),
                    "subreg": -1,
                    "diff": int(diffs[i]) if diffs is not None else 0,
                    "irq": int(irqs[i]) if irqs is not None else 0,
                }
            ]

        return collapse_runs(
            df,
            run_len=2,
            target_regs=self.target_regs,
            build_atom=build_atom,
            label="ctrl_bigram",
        )
