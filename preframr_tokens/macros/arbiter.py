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

__all__ = ["Claim", "arbitrate", "arbitrate_independent_groups"]


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


def _select_nonoverlap(claims):
    """Greedy ``_sort_key``-order selection dropping any claim whose source writes overlap an
    already-selected claim (write-overlapping lower-ranked claims stay literal)."""
    claimed: set = set()
    selected: list = []
    for claim in sorted(claims, key=_sort_key):
        w = set(claim.writes)
        if w & claimed:
            continue
        claimed |= w
        selected.append(claim)
    return selected


def _apply(df, chosen):
    drop_idx: list = []
    new_rows: list = []
    for c in chosen:
        drop_idx.extend(c.writes)
        new_rows.extend(c.tokens)
    return _splice_rows(df, drop_idx, new_rows)


def _greedy_accept(df, selected, src_state):
    """Cumulative byte-exact greedy: keep each claim (in ``selected`` order) only if the result still
    decodes to ``src_state``. Returns the accepted subset."""
    if _lossless(src_state, _apply(df, selected)):
        return list(selected)
    accepted: list = []
    for claim in selected:
        if _lossless(src_state, _apply(df, accepted + [claim])):
            accepted.append(claim)
    return accepted


def arbitrate(df, claims, validate=False):
    """Select non-overlapping claims greedily in ``_sort_key`` order; write-overlapping lower-ranked
    claims drop and stay literal. ``validate`` (register-exact passes) decodes the result and drops any
    claim that changes the source register_state -- CUMULATIVELY, since claims interact (one ctrl
    collapse's frame-tick drain can land in another's frame, so independent checks are unsound) -- a
    clobbered collapse stays literal. ``PREFRAMR_ARBITER_STRICT`` raises on such a drop instead.
    """
    selected = _select_nonoverlap(claims)
    if not selected:
        return df
    strict = _strict()
    if not (validate or strict):
        return _apply(df, selected)
    src_state = _decoded_state(df)
    accepted = _greedy_accept(df, selected, src_state)
    if strict and len(accepted) != len(selected):
        raise AssertionError(
            f"ARBITER: {len(selected) - len(accepted)} claim(s) not byte-exact; "
            f"root-fix the proposing pass"
        )
    return _apply(df, accepted)


def arbitrate_independent_groups(df, groups, validate=False):
    """Like ``arbitrate`` on the flattened claims, but validates each group against ONE shared source
    decode. Sound only when groups are register_state-INDEPENDENT (no group's decode touches another's
    frames -- e.g. ``collapse_runs`` claims separated by more than the tick-drain span): then a claim's
    cumulative-lossless verdict depends only on its own group, so per-group greedy equals global greedy,
    at a fraction of the decodes. ``PREFRAMR_VERIFY_PARTITION`` asserts equality vs the flat path.
    """
    flat = [c for g in groups for c in g]
    selected = _select_nonoverlap(flat)
    if not selected:
        return df
    strict = _strict()
    if not (validate or strict):
        return _apply(df, selected)
    src_state = _decoded_state(df)
    if _lossless(src_state, _apply(df, selected)):
        accepted = selected
    else:
        sel_ids = {id(c) for c in selected}
        accepted = []
        for g in groups:
            g_sel = [c for c in g if id(c) in sel_ids]
            if g_sel:
                accepted.extend(_greedy_accept(df, g_sel, src_state))
    if os.environ.get("PREFRAMR_VERIFY_PARTITION", "") not in ("", "0"):
        ref = _greedy_accept(df, selected, src_state)
        if {id(c) for c in accepted} != {id(c) for c in ref}:
            raise AssertionError(
                f"PARTITION: group result != flat greedy "
                f"({len(accepted)} vs {len(ref)} accepted); groups not independent"
            )
    if strict and len(accepted) != len(selected):
        raise AssertionError(
            f"ARBITER: {len(selected) - len(accepted)} claim(s) not byte-exact; "
            f"root-fix the proposing pass"
        )
    return _apply(df, accepted)
