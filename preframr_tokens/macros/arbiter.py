"""Claim + arbiter: the speculative encoding pipeline core
(``design/speculative_encoding_pipeline.md``). Passes PROPOSE ``Claim``s over the immutable source;
``arbitrate`` accepts a non-overlapping subset maximising the lexicographic score with a deterministic
tie-break. Register-exact passes pass ``validate=True`` so any claim that changes the decoded
register_state is dropped (kept literal); ``PREFRAMR_ARBITER_STRICT`` raises on such a drop instead.
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


def arbitrate(df, claims, validate=False):
    """Select non-overlapping claims greedily in ``_sort_key`` order; write-overlapping lower-ranked
    claims drop and stay literal. ``validate`` (set by register-exact passes) decodes the result and
    DROPS any accepted claim that changes the source per-frame register_state -- so a collapse another
    pass's atom would clobber (a stamp's later same-frame control write) stays literal, not wrong
    tokens. ``PREFRAMR_ARBITER_STRICT`` RAISES on such a drop instead, flagging the proposing pass.
    """
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
    strict = _strict()
    if not (validate or strict):
        return out
    src_state = _decoded_state(df)
    if _lossless(src_state, out):
        return out
    accepted: list = []
    dropped: list = []
    for claim in selected:
        if _lossless(src_state, _apply(accepted + [claim])):
            accepted.append(claim)
        else:
            dropped.append(claim)
    if strict and dropped:
        raise AssertionError(
            f"ARBITER: {len(dropped)} claim(s) not byte-exact (e.g. "
            f"{dropped[0].label or dropped[0].writes}); root-fix the proposing pass"
        )
    return _apply(accepted)
