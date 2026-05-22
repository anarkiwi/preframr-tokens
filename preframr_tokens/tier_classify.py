"""Per-vocab-id loss-tier classification. Thin wrappers over ``VocabSignature``; consumers that need both tier and frame-weight should build a ``VocabSignature`` directly to avoid two passes over the vocab."""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.transform import LOSS_TIER_NAMES
from preframr_tokens.vocab_signature import (
    CONTENT_TIER,
    VocabSignature,
    _row_tier,
    _op_tier_map,
)

__all__ = ["vocab_id_tier", "build_vocab_tier_ids", "build_vocab_tier_map", "CONTENT_TIER"]


def vocab_id_tier(vid: int, rt, tokens) -> str:
    """Classify one vocab id. Cheap one-shot; for whole-vocab classification use ``build_vocab_tier_ids`` / ``build_vocab_tier_map`` / ``VocabSignature``."""
    if tokens is None or len(tokens) == 0:
        return CONTENT_TIER
    n_base = len(tokens)
    base_ids = rt.decode([vid]) if rt.tkmodel else [vid]
    op_tier = _op_tier_map()
    for bid in base_ids:
        bid = int(bid)
        if bid >= n_base:
            continue
        return _row_tier(tokens.iloc[bid], op_tier)
    return CONTENT_TIER


def build_vocab_tier_ids(
    rt,
    tokens,
    n_vocab: int,
    tier_order: tuple[str, ...] = LOSS_TIER_NAMES,
) -> np.ndarray:
    """Per-vocab-id tier index into ``tier_order`` as int64 numpy array."""
    return VocabSignature(rt, tokens, n_vocab, tier_order=tier_order).tier_ids


def build_vocab_tier_map(rt, tokens, n_vocab: int) -> dict[int, str]:
    """Per-vocab-id ``{id: tier_name}`` dict; consumed by the generalization-gate audit (which keys metrics by tier name)."""
    return VocabSignature(rt, tokens, n_vocab).tier_names
