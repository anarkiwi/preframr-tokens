import logging

import numpy as np

__all__ = ["get_logger", "wrapbits", "to_int64_arrays"]


def get_logger(level=None):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


def wrapbits(x: int, reglen: int) -> int:
    """Bit-rotate-left by 1 within a ``reglen``-bit window."""
    base = (x << 1) & (2**reglen - 1)
    lsb = (x >> (reglen - 1)) & 1
    return base ^ lsb


def to_int64_arrays(df, *names, fillna=None):
    """Extract one or more named columns from ``df`` as int64 numpy arrays. ``fillna`` is an optional ``{column_name: value}`` mapping; columns absent from the map are not pre-filled. Centralises the ``df[col].fillna(default).astype(np.int64).to_numpy()`` triple-extraction pattern."""
    fillna = fillna or {}
    return tuple(
        (df[name].fillna(fillna[name]) if name in fillna else df[name])
        .astype(np.int64)
        .to_numpy()
        for name in names
    )
