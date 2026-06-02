"""Shared recurring-codebook emission: the MINE -> GROUP -> DEF-once -> REF-per-occurrence skeleton
common to the inline codebook passes (PatchPass, StampPass). Centralises the DEF-before-REF ``__pos``
invariant -- every emitted row is stamped with its occurrence position so ``_splice_rows`` orders a
def strictly before its refs -- so that byte-exactness rule lives in one place instead of being
re-derived per pass."""

from __future__ import annotations

__all__ = ["emit_recurring"]


def emit_recurring(
    groups,
    *,
    minrep,
    group_sort,
    occ_sort,
    pos_of,
    rows_of,
    emit_first,
    emit_ref,
    start_id=0,
    drop_idx=None,
    new_rows=None,
    skip=None,
    consumed=None,
):
    """Assign one codebook id per group recurring >= ``minrep`` times, emitting a DEF on the first
    occurrence (``emit_first(id, occ)``) and a REF on later ones (``emit_ref(id, ev)``), consuming
    ``rows_of(ev)`` at ``pos_of(ev)``; ``group_sort``/``occ_sort`` make ids deterministic. ``skip`` +
    ``consumed`` + ``start_id`` hand the residue to a later phase. Returns ``(drop, rows, next_id)``.
    """
    drop_idx = [] if drop_idx is None else drop_idx
    new_rows = [] if new_rows is None else new_rows
    next_id = start_id
    for _key, occ in sorted(groups.items(), key=group_sort):
        if skip is not None:
            occ = [ev for ev in occ if not skip(ev)]
        if len(occ) < minrep:
            continue
        cb_id = next_id
        next_id += 1
        occ = sorted(occ, key=occ_sort)
        for j, ev in enumerate(occ):
            rows = emit_first(cb_id, occ) if j == 0 else emit_ref(cb_id, ev)
            pos = pos_of(ev)
            for row in rows:
                row["__pos"] = pos
                new_rows.append(row)
            drop_idx.extend(rows_of(ev))
            if consumed is not None:
                consumed.add(id(ev))
    return drop_idx, new_rows, next_id
