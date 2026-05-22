"""SuperFrame Transform scaffold: pack N consecutive frames into one super-frame block."""

from __future__ import annotations

__all__ = ["SuperFrameTransform"]

from typing import Any

import pandas as pd

from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.stfconstants import FRAME_REG, SUPER_FRAME_REG


@register("super_frame")
class SuperFrameTransform(Transform):
    """Pack N consecutive PAL frames into one super-frame block. N=1 is a no-op; N>=2 is intentionally unimplemented (raise on use)."""

    TIER = "bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"super_frame_pass"})
    MUST_FOLLOW = frozenset({"voice_block_order"})
    MUST_PRECEDE = frozenset({"add_voice_reg"})
    IDEMPOTENT = False
    DEFAULT_PARAMS = {"n_frames": 4}
    PARAM_VALIDATORS = {"n_frames": lambda v: isinstance(v, int) and v >= 1}

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        n = int(self.params["n_frames"])
        if n <= 1:
            return df
        if args is not None and not getattr(args, "super_frame_pass", False):
            return df
        return _pack_super_frames(df, n)

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if df.empty:
            return df
        if "reg" in df.columns and not (df["reg"] == SUPER_FRAME_REG).any():
            return df
        return _unpack_super_frames(df)


def _pack_super_frames(df: pd.DataFrame, n: int) -> pd.DataFrame:
    raise NotImplementedError(
        "super_frame N>=2 pack not yet implemented; "
        "design doc never landed; the N>=2 path is intentionally unimplemented"
    )


def _unpack_super_frames(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "super_frame unpack not yet implemented; "
        "design doc never landed; the N>=2 path is intentionally unimplemented"
    )


def _is_no_op_for(df: pd.DataFrame, n: int, args) -> bool:
    if n <= 1:
        return True
    if args is None or not getattr(args, "super_frame_pass", False):
        return True
    return False
