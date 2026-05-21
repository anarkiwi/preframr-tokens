"""Project eval atoms onto the training alphabet by nearest-val snapping."""

import numpy as np
import pandas as pd


def build_projection_table(train_atoms):
    """Return dict mapping (op, reg, subreg) -> sorted ascending numpy int64 vals."""
    by_key = {}
    for op, reg, subreg, val in train_atoms:
        by_key.setdefault((int(op), int(reg), int(subreg)), set()).add(int(val))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in by_key.items()}


def _nearest(sorted_vals, val):
    """Binary-search nearest neighbour in sorted_vals to val; tie to lower."""
    idx = np.searchsorted(sorted_vals, val)
    if idx == 0:
        return int(sorted_vals[0])
    if idx >= len(sorted_vals):
        return int(sorted_vals[-1])
    lo = int(sorted_vals[idx - 1])
    hi = int(sorted_vals[idx])
    return lo if (val - lo) <= (hi - val) else hi


def _snap_group(sorted_vals, vals):
    """Vectorised nearest-snap of vals onto sorted_vals; ties to lower."""
    sorted_vals = np.asarray(sorted_vals, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.int64)
    if len(sorted_vals) == 0 or len(vals) == 0:
        return vals
    idx = np.searchsorted(sorted_vals, vals)
    idx_lo = np.clip(idx - 1, 0, len(sorted_vals) - 1)
    idx_hi = np.clip(idx, 0, len(sorted_vals) - 1)
    lo = sorted_vals[idx_lo]
    hi = sorted_vals[idx_hi]
    pick_lo = (vals - lo) <= (hi - vals)
    return np.where(pick_lo, lo, hi)


def project_df(df, projection_table):
    """Snap df['val'] onto train alphabet per (op,reg,subreg); unknown keys pass through."""
    if df is None or len(df) == 0:
        return df
    needed = {"op", "reg", "subreg", "val"}
    if not needed.issubset(df.columns):
        return df
    if not projection_table:
        return df
    out = df.copy()
    ops = out["op"].to_numpy(dtype=np.int64)
    regs = out["reg"].to_numpy(dtype=np.int64)
    subs = out["subreg"].to_numpy(dtype=np.int64)
    vals = out["val"].to_numpy(dtype=np.int64).copy()
    keys = pd.MultiIndex.from_arrays([ops, regs, subs])
    key_df = pd.DataFrame({"_row": np.arange(len(out))}, index=keys)
    for key, group in key_df.groupby(level=[0, 1, 2], sort=False):
        tbl = projection_table.get(key)
        if tbl is None or len(tbl) == 0:
            continue
        rows = group["_row"].to_numpy()
        vals[rows] = _snap_group(tbl, vals[rows])
    target_dtype = out["val"].dtype
    if isinstance(target_dtype, np.dtype):
        out["val"] = vals.astype(target_dtype, copy=False)
    else:
        out["val"] = pd.array(vals, dtype=target_dtype)
    return out
