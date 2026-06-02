"""Collapse three adjacent CTRL writes into one ``CTRL_TRIPLE`` atom (spec #6),
behind the ``ctrl_triple_pass`` arg flag (default OFF): three consecutive
same-voice CTRL SETs each one frame apart (no DELAY between) become a 3-byte
atom, extending CTRL_BIGRAM by one and run before it so triples win."""

__all__ = ["CtrlTriplePass"]

from preframr_tokens.macros.passes_base import MacroPass, _first_irq
from preframr_tokens.macros.run_collapse import collapse_runs
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_TRIPLE_OP,
    CTRL_TRIPLE_SUBREG_0,
    CTRL_TRIPLE_SUBREG_1,
    CTRL_TRIPLE_SUBREG_2,
)


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
    GATE_FLAGS = frozenset({"ctrl_triple_pass"})
    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        if args is None or not getattr(args, "ctrl_triple_pass", False):
            return df
        if "op" not in df.columns or "reg" not in df.columns:
            return df
        df = df.reset_index(drop=True).copy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq_default = _first_irq(df)

        def build_atom(reg, idxs):
            i, j, l = idxs
            diff = int(diffs[i]) if diffs is not None else 0
            bytes3 = (int(vals[i]), int(vals[j]), int(vals[l]))
            return _triple_rows(reg, bytes3, diff, irq_default)

        return collapse_runs(
            df,
            run_len=3,
            target_regs=self.target_regs,
            build_atom=build_atom,
            label="ctrl_triple",
        )
