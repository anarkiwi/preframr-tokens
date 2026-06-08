"""Events-native generation decode (REDESIGN_optionB §7.1, step 4): turn a model's generated BPE token-id
stream back into the ordered register-write stream and a render-ready raw-dump DataFrame. The inverse of
the tokenization -- ``tokenizer.decode`` (BPE -> n-space atoms) then :func:`dataset.ids_to_writes` (atoms ->
ordered writes). The factored decoder is a strict grammar parser, so an invalid generated stream raises
loudly rather than silently mis-decoding; constrained-sampling masking is a generation-time optimization on
top of this (the decoder is the completeness oracle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import ids_to_writes


def tokens_to_writes(tokenizer, bpe_ids) -> list[tuple[int, int, int]]:
    """Generated BPE ids -> ordered ``(frame, reg, val)`` writes. ``tokenizer.decode`` undoes the BPE to
    the n-space atom stream; trailing PAD (0) is dropped; the factored decoder replays it (raising on a
    grammatically invalid stream)."""
    nspace = list(tokenizer.decode(np.asarray(bpe_ids, dtype=np.uint32)))
    while nspace and int(nspace[-1]) == 0:
        nspace.pop()
    return ids_to_writes(nspace)


def writes_to_dump_df(writes, chipno: int = 0) -> pd.DataFrame:
    """Ordered ``(frame, reg, val)`` writes -> a raw-dump DataFrame (``clock, irq, chipno, reg, val``) the
    reSID renderer consumes. ``irq`` is the dense frame index (the player-call tick); ``clock`` is a
    strictly increasing surrogate (one per write) since intra-frame order is already explicit in the row
    order. A tick/cycle scale is applied downstream by the renderer if it needs absolute cycles.
    """
    if not writes:
        cols = ["clock", "irq", "chipno", "reg", "val"]
        return pd.DataFrame({c: np.empty(0, dtype=np.int64) for c in cols})
    frame = np.array([f for f, _, _ in writes], dtype=np.int64)
    reg = np.array([r for _, r, _ in writes], dtype=np.int64)
    val = np.array([v for _, _, v in writes], dtype=np.int64)
    return pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": frame,
            "chipno": np.full(len(writes), int(chipno), dtype=np.int64),
            "reg": reg,
            "val": val,
        }
    )


def tokens_to_dump_df(tokenizer, bpe_ids, chipno: int = 0) -> pd.DataFrame:
    """Generated BPE ids -> render-ready raw-dump DataFrame (compose of the two steps above)."""
    return writes_to_dump_df(tokens_to_writes(tokenizer, bpe_ids), chipno=chipno)


__all__ = ["tokens_to_dump_df", "tokens_to_writes", "writes_to_dump_df"]
