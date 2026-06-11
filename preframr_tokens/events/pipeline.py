"""Events-native production tokenization (REDESIGN_optionB §7.1): the raw dump is encoded directly by
the canonical stream codec (``events.stream``). A tune is sliced into self-contained frame-window blocks,
each encoded to a flat token-id list (the pre-BPE "n" stream) that decodes to the window's canonical
writes, so the model trains on self-delimiting event tokens (no ids, no per-tune codebook); BPE over
them is the dictionary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stream
from .oracle import MAX_REG, OrderedWrites, ordered_writes

VOCAB_SIZE = stream.VOCAB_SIZE


def window(ow: OrderedWrites, f0: int, f1: int) -> OrderedWrites:
    """The sub-stream of ``ow`` over frames ``[f0, f1)``, re-indexed so frame ``f0`` becomes frame 0."""
    mask = (ow.frame >= f0) & (ow.frame < f1)
    return OrderedWrites(
        frame=ow.frame[mask] - f0,
        reg=ow.reg[mask],
        val=ow.val[mask],
        n_frames=int(f1 - f0),
        irq=ow.irq[f0:f1] if ow.irq.shape[0] >= f1 else ow.irq[f0:],
    )


def iter_windows(ow: OrderedWrites, frames_per_block: int, stride: int | None = None):
    """Yield self-contained :class:`OrderedWrites` frame windows (``frames_per_block`` frames, advancing by
    ``stride``, default = ``frames_per_block``). Each window is independently byte-exact-decodable.
    """
    if stride is None:
        stride = frames_per_block
    n = ow.n_frames
    for f0 in range(0, n, stride):
        f1 = min(f0 + frames_per_block, n)
        if f1 <= f0:
            continue
        yield window(ow, f0, f1)


def block_tokens(ow_window: OrderedWrites) -> list[int]:
    """One frame window -> its flat event token-id list (the pre-BPE block, byte-exact)."""
    return stream.encode(ow_window)


def block_writes(tokens: list[int]) -> list[tuple[int, int, int]]:
    """Inverse of :func:`block_tokens`: a block's tokens -> its ordered ``(frame, reg, val)`` writes."""
    return stream.decode(tokens)


def dump_blocks(
    df: pd.DataFrame, frames_per_block: int, stride: int | None = None
) -> list[list[int]]:
    """Raw dump DataFrame -> list of self-contained event-token blocks (the §7.1 tokenization)."""
    ow = ordered_writes(df)
    return [block_tokens(w) for w in iter_windows(ow, frames_per_block, stride)]


def block_array(
    df: pd.DataFrame, block_size: int, frames_per_block: int, stride: int | None = None
) -> np.ndarray:
    """Materialise a tune into a fixed ``(n_blocks, block_size)`` int32 array of event tokens (zero-padded
    tails, long blocks truncated) -- the events-native replacement for ``blocks.materialize_block_array``.
    NOTE: BPE is applied downstream over the flat token stream, not here (the atoms are self-delimiting).
    """
    ow = ordered_writes(df)
    rows = []
    for w in iter_windows(ow, frames_per_block, stride):
        seq = np.asarray(block_tokens(w), dtype=np.int32)
        row = np.zeros(block_size, dtype=np.int32)
        row[: min(len(seq), block_size)] = seq[:block_size]
        rows.append(row)
    if not rows:
        return np.zeros((0, block_size), dtype=np.int32)
    return np.stack(rows)


def atoms() -> list[int]:
    """The complete pre-BPE alphabet: every atomic event token id ``0 .. VOCAB_SIZE-1``."""
    return list(range(VOCAB_SIZE))


__all__ = [
    "MAX_REG",
    "VOCAB_SIZE",
    "atoms",
    "block_array",
    "block_tokens",
    "block_writes",
    "dump_blocks",
    "iter_windows",
    "window",
]
