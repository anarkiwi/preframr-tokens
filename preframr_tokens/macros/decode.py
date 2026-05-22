"""Canonical decoder walk."""

__all__ = ["expand_ops"]

import pandas as pd

from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.macros.loops import expand_loops
from preframr_tokens.macros.state import _build_decode_state
from preframr_tokens.macros.walker import FrameWalker
from preframr_tokens.stfconstants import MODEL_PDTYPE


def expand_ops(orig_df, strict=False):
    """Walk an encoded token DataFrame back to literal writes."""
    df = expand_loops(orig_df.copy())

    state = _build_decode_state(df, strict=strict)

    class _ExpandOpsWalker(FrameWalker):
        emit_synthetic_frame_marker = True

        def __init__(self, df_, state_):
            super().__init__(df_, state_)
            self.all_rows = []
            self.call_idx = 0

        def before_row(self, i, reg, op):
            assert reg >= 0, (i, reg)
            assert DECODERS.get(op) is not None, f"unknown op {op} reg {reg}"
            return True

        def on_pre_observe(self, writes):
            ci = self.call_idx
            for w in writes:
                desc = w[3] if len(w) > 3 else pd.NA
                self.all_rows.append((w[0], w[1], w[2], desc, ci))
            self.call_idx += 1

    walker = _ExpandOpsWalker(df, state)
    walker.walk()
    all_rows = walker.all_rows

    if not all_rows:
        return pd.DataFrame(
            columns=["reg", "val", "diff", "description"], dtype=MODEL_PDTYPE
        )
    df = pd.DataFrame(
        all_rows,
        columns=["reg", "val", "diff", "description", "__c"],
        dtype=MODEL_PDTYPE,
    )
    df = (
        df.sort_values(["__c", "reg"], kind="stable")
        .drop(columns="__c")
        .reset_index(drop=True)
    )
    df["description"] = df["description"].ffill().fillna(0)
    return df
