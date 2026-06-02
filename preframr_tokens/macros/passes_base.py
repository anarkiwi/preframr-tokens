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


def _frame_isolated(frames, pos, gap):
    """True when the same-register SET at sorted ``frames[pos]`` has no same-reg
    neighbour within ``gap`` frames on either side; the shared lonely-SET test
    behind the FREQ_NUDGE and RELEASE_UPDATE catch-all passes."""
    fr = frames[pos]
    prev_ok = pos == 0 or fr - frames[pos - 1] >= gap
    next_ok = pos == len(frames) - 1 or frames[pos + 1] - fr >= gap
    return prev_ok and next_ok


def _ensure_subreg(df):
    if "subreg" not in df.columns:
        df = df.copy()
        df["subreg"] = -1
    return df


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
    new_df = pd.DataFrame(new_rows)
    for col in df.columns:
        if col not in new_df.columns:
            if col == "description":
                new_df[col] = 0
            elif col == "irq":
                new_df[col] = irq_value
            else:
                new_df[col] = -1
    new_df = new_df[df.columns]
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
