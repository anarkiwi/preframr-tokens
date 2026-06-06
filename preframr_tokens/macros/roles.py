"""Single source of truth for ``(op, subreg)`` → role classification used by the constrained-decode pre-compute, the stream validators, the per-vocab tier classifier, and the frame-time weighting helper. Each role is a tiny tag so callers can dispatch on it with a single dict lookup instead of re-encoding the (op, subreg) constants inline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from preframr_tokens.stfconstants import (
    DO_LOOP_OP,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
)

__all__ = [
    "DistancePairSpec",
    "DISTANCE_PAIR_OPS",
    "distance_pair_role",
    "frame_weight_role",
]


@dataclass(frozen=True)
class DistancePairSpec:
    """Mapping from a distance-pair op (``PATTERN_REPLAY``) to its DIST_HI / DIST_LO / LEN slot ids and any extra trailing slots (``PATTERN_REPLAY`` carries the optional OVERLAY_COUNT)."""

    label: str
    dist_hi: int
    dist_lo: int
    length: int
    extra_subregs: frozenset[int]


DISTANCE_PAIR_OPS: dict[int, DistancePairSpec] = {
    PATTERN_REPLAY_OP: DistancePairSpec(
        label="PR",
        dist_hi=PATTERN_REPLAY_SUBREG_DIST_HI,
        dist_lo=PATTERN_REPLAY_SUBREG_DIST_LO,
        length=PATTERN_REPLAY_SUBREG_LEN,
        extra_subregs=frozenset({PATTERN_REPLAY_SUBREG_OVERLAY_COUNT}),
    ),
}


def distance_pair_role(op: int, subreg: int) -> Optional[str]:
    """``"dist_hi" | "dist_lo" | "len" | "ov_count" | None`` for distance-pair ops; ``None`` if ``op`` is not a distance-pair op or ``subreg`` doesn't match any slot."""
    spec = DISTANCE_PAIR_OPS.get(int(op))
    if spec is None:
        return None
    sr = int(subreg)
    if sr == spec.dist_hi:
        return "dist_hi"
    if sr == spec.dist_lo:
        return "dist_lo"
    if sr == spec.length:
        return "len"
    if sr in spec.extra_subregs:
        return "ov_count"
    return None


def frame_weight_role(op: int, subreg: int) -> Optional[str]:
    """``"pattern_replay_len" | "do_loop_len" | None`` for the op-side weight sources (the reg-side DELAY / FRAME ones live in ``token_weighting``)."""
    op = int(op)
    sr = int(subreg)
    if op == PATTERN_REPLAY_OP and sr == PATTERN_REPLAY_SUBREG_LEN:
        return "pattern_replay_len"
    if op == DO_LOOP_OP and sr == 0:
        return "do_loop_len"
    return None
