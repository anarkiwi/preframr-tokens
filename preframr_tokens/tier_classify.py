"""Per-vocab-id loss-tier classification (structural / mid / content / zero). Single source of truth for the partition the per-tier model heads + the multi-task loss aggregator depend on; consumers should not re-implement the reg/op switch."""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.transform import LOSS_TIER_NAMES, collect_op_loss_tiers
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    MODE_VOL_REG,
    VOICE_CTRL_REG,
)

CONTENT_TIER = "content"

_VOICE_CTRL_REGS = frozenset(VOICE_CTRL_REG.values())
_OP_TIER_CACHE: dict[int, str] | None = None


def _registry_op_tier_map() -> dict[int, str]:
    """Op-code -> loss tier from the Transform registry; imports the macros packages to trigger registration."""
    global _OP_TIER_CACHE  # pylint: disable=global-statement
    if _OP_TIER_CACHE is None:
        # pylint: disable=import-outside-toplevel,unused-import
        from preframr_tokens.macros import (
            transforms_audio_bit_exact,
            transforms_bit_exact,
        )

        _OP_TIER_CACHE = collect_op_loss_tiers()
    return _OP_TIER_CACHE


def vocab_id_tier(vid: int, rt, tokens) -> str:
    """Classify one vocab id into ``CONTENT_TIER`` / one of the other ``LOSS_TIER_NAMES``. Reg-specific overrides (FRAME, DELAY, FILTER, MODE_VOL, VOICE_CTRL) take precedence over the op-registry mapping; unmapped tokens fall through to ``content``."""
    if rt.tkmodel:
        base_ids = rt.decode([vid])
    else:
        base_ids = [vid]
    n_base = len(tokens)
    op_tier = _registry_op_tier_map()
    for bid in base_ids:
        bid = int(bid)
        if bid >= n_base:
            continue
        row = tokens.iloc[bid]
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
        if op in op_tier:
            return op_tier[op]
        return CONTENT_TIER
    return CONTENT_TIER


def build_vocab_tier_ids(
    rt,
    tokens,
    n_vocab: int,
    tier_order: tuple[str, ...] = LOSS_TIER_NAMES,
) -> np.ndarray:
    """Return per-vocab-id tier index into ``tier_order`` as int64 numpy array. Unknown / pad / out-of-range vids default to the index of ``CONTENT_TIER``."""
    name_to_id = {name: i for i, name in enumerate(tier_order)}
    default_id = name_to_id[CONTENT_TIER]
    out = np.full(n_vocab, default_id, dtype=np.int64)
    if tokens is None or len(tokens) == 0:
        return out
    for vid in range(n_vocab):
        tier = vocab_id_tier(vid, rt, tokens)
        out[vid] = name_to_id.get(tier, default_id)
    return out


def build_vocab_tier_map(rt, tokens, n_vocab: int) -> dict[int, str]:
    """Return ``{vocab_id: tier_name}`` for the active pipeline; consumed by the generalization gate (which keys metrics by tier name, not id)."""
    if tokens is None or len(tokens) == 0:
        return {vid: CONTENT_TIER for vid in range(n_vocab)}
    return {vid: vocab_id_tier(vid, rt, tokens) for vid in range(n_vocab)}
