"""Snap plain-SET val on wide-val regs to per-reg preset tables (aggressive)."""

__all__ = ["PresetPass"]

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _splice_rows,
)
from preframr_tokens.stfconstants import (
    FC_PRESET_TABLE,
    PRESET_REG_GRID,
    PRESET_REG_TO_OP,
    PWM_PRESET_TABLE,
    SET_OP,
)


def _snap(val, grid):
    v = int(val)
    if v >= 0:
        return ((v + grid // 2) // grid) * grid
    return -(((-v) + grid // 2) // grid) * grid


def _preset_id_for(reg, snapped_val):
    grid = PRESET_REG_GRID[int(reg)]
    if snapped_val < 0:
        return 0
    pid = snapped_val // grid
    if int(reg) == 21:
        return min(pid, len(FC_PRESET_TABLE) - 1)
    return min(pid, len(PWM_PRESET_TABLE) - 1)


def _preset_row(reg, preset_id, diff_default, irq_default):
    op = int(PRESET_REG_TO_OP[int(reg)])
    return {
        "reg": int(reg),
        "val": int(preset_id),
        "diff": int(diff_default),
        "op": op,
        "subreg": -1,
        "irq": int(irq_default),
        "description": 0,
    }


class PresetPass(MacroPass):
    """Single-row preset snap for wide-val plain SETs on regs 2 + 21."""

    def apply(self, df, args=None):
        if args is not None and not getattr(args, "preset_pass", True):
            return df
        if df is None or len(df) == 0:
            return df
        orig_df = df
        df = df.reset_index(drop=True).copy()
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        df = _ensure_subreg(df)
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        irq_default = (
            int(df["irq"].iloc[0])
            if "irq" in df.columns and len(df) and df["irq"].notna().any()
            else -1
        )
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None

        drop_idx = []
        new_rows = []
        for i in range(len(df)):
            r = int(regs[i])
            if r not in PRESET_REG_TO_OP:
                continue
            if int(ops[i]) != SET_OP:
                continue
            if int(subregs[i]) != -1:
                continue
            grid = PRESET_REG_GRID[r]
            snapped = _snap(vals[i], grid)
            pid = _preset_id_for(r, snapped)
            diff_v = int(diffs[i]) if diffs is not None else 0
            row = _preset_row(r, pid, diff_v, irq_default)
            row["__pos"] = int(i)
            new_rows.append(row)
            drop_idx.append(int(i))
        if not new_rows:
            return orig_df
        return _splice_rows(df, drop_idx, new_rows)
