"""Residual CTRL catch-all (validation-phase fail-on-lonely), behind the
``lonely_catch_all`` arg flag (default OFF): every full CTRL SET the bigram /
triple passes did not take becomes a ``CTRL_UPDATE`` op (SET-equivalent decode)
so no CTRL write is left lonely for the strict-no-diff validator. Runs after the
control macros, so only their leftovers are tagged."""

__all__ = ["CtrlUpdatePass"]

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _splice_rows,
    _first_irq,
    make_row,
)
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import CTRL_UPDATE_OP, SET_OP

_CTRL_REGS = frozenset(CTRL_REGS_BY_VOICE)


class CtrlUpdatePass(MacroPass):
    GATE_FLAGS = frozenset({"lonely_catch_all"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "lonely_catch_all", False):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq_default = _first_irq(df)
        drop_idx = []
        new_rows = []
        for i in range(len(df)):
            if (
                int(regs[i]) in _CTRL_REGS
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ):
                row = make_row(
                    int(regs[i]),
                    int(vals[i]),
                    op=CTRL_UPDATE_OP,
                    subreg=-1,
                    diff=int(diffs[i]) if diffs is not None else 0,
                    irq=int(irq_default),
                )
                row["__pos"] = i
                new_rows.append(row)
                drop_idx.append(i)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)
