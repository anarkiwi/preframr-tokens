"""Single-pass per-vocab-id classifier. Folds ``tier_classify`` (loss-tier per id) and ``token_weighting`` (frame-time weight per id) into one walk over the vocab so consumers (loss heads, per-token CE weighting, generalization-gate audit) don't iterate the vocab three times. The free functions in ``tier_classify`` and ``token_weighting`` remain as thin wrappers that materialize a ``VocabSignature`` and slice off the field they want."""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.roles import frame_weight_role
from preframr_tokens.macros.transform import (
    collect_op_loss_tiers,
    ensure_default_transforms_registered,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    LOSS_TIER_NAMES,
    MODE_VOL_REG,
    VOICE_CTRL_REG,
)

__all__ = ["VocabSignature", "CONTENT_TIER"]

CONTENT_TIER = "content"

_VOICE_CTRL_REGS = frozenset(VOICE_CTRL_REG.values())
_OP_TIER_CACHE: dict[int, str] | None = None


def _op_tier_map() -> dict[int, str]:
    global _OP_TIER_CACHE  # pylint: disable=global-statement
    if _OP_TIER_CACHE is None:
        ensure_default_transforms_registered()
        _OP_TIER_CACHE = collect_op_loss_tiers()
    return _OP_TIER_CACHE


def _row_tier(row, op_tier: dict[int, str]) -> str:
    """Tier for a single atomic row. Reg-specific overrides (FRAME, DELAY, FILTER, MODE_VOL, VOICE_CTRL) take precedence over the op-registry mapping; unmapped tokens fall through to ``content``."""
    op = int(row.op)
    reg = int(row.reg)
    if reg == FRAME_REG:
        return "structural"
    if reg == DELAY_REG:
        return "mid"
    if reg in (FILTER_REG, MODE_VOL_REG):
        return "zero"
    if reg in _VOICE_CTRL_REGS:
        return "mid"
    return op_tier.get(op, CONTENT_TIER)


def _row_weight(row) -> float:
    op = int(row.op)
    subreg = int(row.subreg)
    val = int(row.val)
    reg = int(row.reg)
    if frame_weight_role(op, subreg) is not None:
        return float(val)
    if reg == DELAY_REG:
        return float(val)
    if reg == FRAME_REG:
        return 1.0
    return 0.0


class VocabSignature:
    """Per-vocab-id (loss-tier, frame-time-weight) signature, computed in one walk over the vocab. Exposes ``tier_ids`` (int64 indices into ``tier_order``), ``tier_names`` (dict[int, str]), and ``frame_weights`` (float32, default 1.0). Under Unigram sub-tokens the tier is taken from the first atomic id; the frame weight accumulates across all atomic ids."""

    def __init__(
        self,
        rt,
        tokens,
        n_vocab: int,
        tier_order: tuple[str, ...] = LOSS_TIER_NAMES,
    ):
        self.tier_order = tier_order
        name_to_id = {name: i for i, name in enumerate(tier_order)}
        default_id = name_to_id[CONTENT_TIER]
        self.tier_ids = np.full(n_vocab, default_id, dtype=np.int64)
        self.tier_names: dict[int, str] = {}
        self.frame_weights = np.ones(n_vocab, dtype=np.float32)
        if tokens is None or len(tokens) == 0:
            self.tier_names = {vid: CONTENT_TIER for vid in range(n_vocab)}
            return
        op_tier = _op_tier_map()
        n_base = len(tokens)
        for vid in range(n_vocab):
            base_ids = rt.decode([vid]) if rt.tkmodel else [vid]
            tier = CONTENT_TIER
            tier_set = False
            weight = 0.0
            for bid in base_ids:
                bid = int(bid)
                if bid >= n_base:
                    continue
                row = tokens.iloc[bid]
                if not tier_set:
                    tier = _row_tier(row, op_tier)
                    tier_set = True
                weight += _row_weight(row)
            self.tier_names[vid] = tier
            self.tier_ids[vid] = name_to_id.get(tier, default_id)
            if weight > 0.0:
                self.frame_weights[vid] = weight
