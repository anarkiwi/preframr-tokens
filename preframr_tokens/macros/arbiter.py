"""Claim + arbiter: the speculative encoding pipeline core
(``design/speculative_encoding_pipeline.md``). Passes PROPOSE ``Claim``s over the immutable source;
``arbitrate`` accepts a non-overlapping subset (a lossless write-PARTITION; unclaimed writes stay)
maximising the lexicographic score with a deterministic tie-break (priority, then source order).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from preframr_tokens.macros.passes_base import _splice_rows

__all__ = ["Claim", "arbitrate"]


@dataclass
class Claim:
    """One proposed encoding of a span of source writes. ``writes`` are the source row index
    labels this claim consumes (the drop set); ``tokens`` are the replacement rows (each a dict
    carrying ``__pos`` for stable splice positioning); ``score`` is the lexicographic objective
    (higher is better, each element numeric so ``-s`` orders it); ``priority`` is the pass-order
    tie-break (lower = earlier = preferred on score ties)."""

    writes: tuple
    tokens: list = field(default_factory=list)
    score: tuple = (0, 0, 0)
    priority: int = 0
    label: str = ""


def _sort_key(claim):
    """Total deterministic order: best score first (negate each numeric element), then lower
    priority, then earliest source write -- so selection is reproducible (the deterministic suite
    requires it)."""
    first = min(claim.writes) if claim.writes else -1
    return (tuple(-s for s in claim.score), claim.priority, first)


def arbitrate(df, claims):
    """Select non-overlapping claims greedily in ``_sort_key`` order and apply them as one splice.
    Conflicting (write-overlapping) lower-ranked claims are dropped; unclaimed writes remain in
    ``df`` (lossless fallback). A single claim reduces exactly to
    ``_splice_rows(df, drop_idx, new_rows)`` -- byte-identical to the pre-arbiter passes.
    """
    claimed: set = set()
    drop_idx: list = []
    new_rows: list = []
    for claim in sorted(claims, key=_sort_key):
        w = set(claim.writes)
        if w & claimed:
            continue
        claimed |= w
        drop_idx.extend(claim.writes)
        new_rows.extend(claim.tokens)
    return _splice_rows(df, drop_idx, new_rows)
