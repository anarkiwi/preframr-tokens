"""Events-native generation decode: generated BPE ids -> ordered
canonical register writes -> a render-ready raw-dump DataFrame, via ``tokenizer.decode`` (BPE -> n-space
atoms) then :func:`dataset.ids_to_writes`. The stream decoder is a strict grammar parser (an invalid
stream raises loudly); constrained-sampling masking is a generation-time optimization on top (the
decoder is the completeness oracle)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stream
from .dataset import ids_to_writes
from .oracle import ordered_writes


def recanon(n_ids) -> list[int]:
    """Project a grammatically-valid n-space atom stream onto its canonical form: drop PAD, strip
    KEYFRAME conditioning, decode to writes, rebuild the ordered-write oracle, re-encode. Identity on a
    canonical keyframe-free stream, idempotent, write-preserving. The Tier-4 DAgger oracle (rollout ->
    nearest valid SID state); input must be whole-frame and continuous -- a windowed leading KEYFRAME
    carries prior state strip_keyframes cannot restore (see WORK_ORDER_prior_state_recanon.md).
    """
    atoms = [int(t) - 1 for t in n_ids if int(t) > 0]
    writes = stream.decode(stream.strip_keyframes(atoms))
    canon = stream.encode(ordered_writes(writes_to_dump_df(writes)), verify=False)
    return [a + 1 for a in canon]


def tokens_to_writes(tokenizer, bpe_ids, extend=False) -> list[tuple[int, int, int]]:
    """Generated BPE ids -> ordered ``(frame, reg, val)`` writes. ``tokenizer.decode`` undoes the BPE to
    the n-space atom stream; trailing PAD (0) is dropped; the stream decoder replays it (raising on a
    grammatically invalid stream). ``extend=True`` replays continuations past the declared frame count.
    """
    nspace = list(tokenizer.decode(np.asarray(bpe_ids, dtype=np.uint32)))
    while nspace and int(nspace[-1]) == 0:
        nspace.pop()
    return ids_to_writes(nspace, extend=extend)


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


def tokens_to_dump_df(
    tokenizer, bpe_ids, chipno: int = 0, extend=False
) -> pd.DataFrame:
    """Generated BPE ids -> render-ready raw-dump DataFrame (compose of the two steps above)."""
    return writes_to_dump_df(
        tokens_to_writes(tokenizer, bpe_ids, extend=extend), chipno=chipno
    )


__all__ = ["tokens_to_dump_df", "tokens_to_writes", "writes_to_dump_df"]
