"""``MacroPass`` base class and the helpers (``_frame_index`` /
``_ensure_subreg`` / ``_splice_rows``) every pass uses to manipulate
its row DataFrame.
"""

import functools

import numpy as np
import pandas as pd

from preframr_tokens.macros.state import _build_decode_state
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, SET_OP, _MIN_DIFF

__all__ = ["MacroPass", "make_row", "requires_state"]


class MacroPass:
    """Base class for encode-side passes operating on a token DataFrame."""

    GATE_FLAGS: frozenset = frozenset()

    def apply(self, df, args=None):
        raise NotImplementedError


def make_row(reg, val, *, op=SET_OP, subreg=-1, diff=_MIN_DIFF, irq=0, description=0):
    """The canonical encoder row dict -- the 7-field schema (reg, val, diff, op, subreg, irq,
    description) every pass emits. Per-pass ``_row`` helpers delegate here so the schema lives once.
    """
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": int(description),
    }


def requires_state(method):
    """Decorator for ``MacroPass.apply`` implementations that need a
    built ``DecodeState``.
    """

    @functools.wraps(method)
    def wrapper(self, df, args=None):
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        if "description" not in df.columns:
            df["description"] = 0
        state = _build_decode_state(df)
        if state is None:
            return df
        return method(self, df, state, args)

    return wrapper


def _frame_index(df):
    """Cumulative frame index for each row (boundary at FRAME_REG/DELAY_REG)."""
    return df["reg"].isin({FRAME_REG, DELAY_REG}).astype(int).cumsum()


def _first_irq(df, default=-1):
    """First row's IRQ value, or ``default`` when ``df`` is empty or has no
    usable ``irq`` column; the shared irq-seed every splice / row-emitting pass
    stamps onto the rows it synthesises."""
    if "irq" in df.columns and len(df) and df["irq"].notna().any():
        return int(df["irq"].iloc[0])
    return default


def _ensure_subreg(df):
    if "subreg" not in df.columns:
        df = df.copy()
        df["subreg"] = -1
    return df


def _rows_to_df(rows, columns, defaults=None):
    """Build a DataFrame from dict-rows column-wise over an explicit ``columns`` list,
    skipping pandas' per-row list-of-dicts inference (``_list_of_dict_to_arrays`` +
    per-column ``convert``). Each column goes through a numpy int64 fast path; any
    irregular value (NA/float/non-int) drops that column to object-list inference. A
    missing key uses ``defaults[col]`` (``-1`` if unspecified)."""
    defaults = defaults or {}
    cols = list(columns)
    if not rows:
        return pd.DataFrame(
            {c: np.empty(0, dtype=np.int64) for c in cols}, columns=cols
        )
    data = {}
    for c in cols:
        d = defaults.get(c, -1)
        vals = [r[c] if c in r else d for r in rows]
        try:
            data[c] = np.fromiter(vals, dtype=np.int64, count=len(vals))
        except (TypeError, ValueError, OverflowError):
            data[c] = vals
    return pd.DataFrame(data, columns=cols)


def _splice_rows(df, drop_idx, new_rows):
    """Drop rows by index and splice ``new_rows`` (each carrying ``__pos``)
    into their original positions, preserving the rest of the row order.
    """
    if not new_rows:
        return df
    orig_attrs = dict(df.attrs)
    df = _ensure_subreg(df)
    irq_value = _first_irq(df)
    orig_dtypes = df.dtypes.to_dict()
    df = df.drop(index=drop_idx)
    df["__pos"] = df.index.astype("int64")
    new_df = _rows_to_df(
        new_rows, df.columns, defaults={"description": 0, "irq": irq_value}
    )
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.sort_values("__pos", kind="stable").reset_index(drop=True)
    combined = combined.drop(columns=["__pos"])
    for col, dt in orig_dtypes.items():
        if col == "__pos":
            continue
        try:
            combined[col] = combined[col].astype(dt)
        except (TypeError, ValueError):
            pass
    if orig_attrs:
        combined.attrs.update(orig_attrs)
    return combined
