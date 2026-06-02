"""Claim + arbiter: the speculative encoding pipeline core
(``design/speculative_encoding_pipeline.md``). Passes PROPOSE ``Claim``s over the immutable source;
``arbitrate`` accepts a non-overlapping subset maximising the lexicographic score with a deterministic
tie-break. Under ``PREFRAMR_ARBITER_STRICT`` it raises on any accepted claim that does not decode
byte-identically (register_state), so a buggy register-exact pass surfaces to be root-fixed.
"""

from __future__ import annotations

import os
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


def _decoded_state(df):
    """Per-frame decoded register state (macros expanded), the byte-exactness oracle. Op-less frames
    are treated as plain SETs, matching ``parse_audit``."""
    from preframr_tokens.audit_primitives import register_state
    from preframr_tokens.stfconstants import SET_OP

    return register_state(df if "op" in df.columns else df.assign(op=int(SET_OP)))


def _lossless(src_state, candidate_df):
    """True iff ``candidate_df`` decodes to the same per-frame register state as the source (same shape
    -- frame budget -- and same values). Returns False on any decode error: an undecodable candidate is
    never accepted."""
    try:
        cand = _decoded_state(candidate_df)
    except (
        Exception
    ):  # noqa: BLE001 -- an undecodable claim partition is simply rejected
        return False
    if cand.shape != src_state.shape:
        return False
    return not (cand != src_state).any()


def _strict():
    return os.environ.get("PREFRAMR_ARBITER_STRICT", "") not in ("", "0")


def arbitrate(df, claims):
    """Select non-overlapping claims greedily in ``_sort_key`` order; conflicting (write-overlapping)
    lower-ranked claims are dropped and their writes stay literal. A single claim reduces exactly to
    ``_splice_rows(df, drop_idx, new_rows)`` -- byte-identical to the pre-arbiter passes. Validation is
    opt-in via ``PREFRAMR_ARBITER_STRICT`` (register-exact passes only; content-tier passes change
    decoded state by design); when strict, a non-byte-exact accepted claim RAISES."""
    claimed: set = set()
    selected: list = []
    for claim in sorted(claims, key=_sort_key):
        w = set(claim.writes)
        if w & claimed:
            continue
        claimed |= w
        selected.append(claim)
    if not selected:
        return df

    def _apply(chosen):
        drop_idx: list = []
        new_rows: list = []
        for c in chosen:
            drop_idx.extend(c.writes)
            new_rows.extend(c.tokens)
        return _splice_rows(df, drop_idx, new_rows)

    out = _apply(selected)
    if not _strict():
        return out
    src_state = _decoded_state(df)
    if _lossless(src_state, out):
        return out
    for claim in selected:
        if not _lossless(src_state, _apply([c for c in selected if c is not claim])):
            continue
        raise AssertionError(
            f"ARBITER: claim {claim.label or claim.writes} is not byte-exact "
            f"(decoded register_state diverges); root-fix the proposing pass"
        )
    raise AssertionError("ARBITER: accepted partition is not byte-exact")
