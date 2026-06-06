"""Shared recurring-codebook emission: the MINE -> GROUP -> DEF-once -> REF-per-occurrence skeleton
common to the inline codebook passes (WavetablePass, InstrumentProgramPass). Centralises the DEF-before-REF ``__pos``
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
    per_group=False,
):
    """Assign one codebook id per group recurring >= ``minrep`` times, emitting a DEF on the first
    occurrence (``emit_first(id, occ)``) and a REF on later ones (``emit_ref(id, ev)``), consuming
    ``rows_of(ev)`` at ``pos_of(ev)``; ``group_sort``/``occ_sort`` make ids deterministic, ``skip`` /
    ``consumed`` / ``start_id`` hand a residue to a later phase. ``per_group`` returns ``(groups,
    next_id)`` (one ``(drop, rows)`` per group, for granular per-Claim validate) else flattened.
    """
    drop_idx = [] if drop_idx is None else drop_idx
    new_rows = [] if new_rows is None else new_rows
    grouped = []
    next_id = start_id
    for _key, occ in sorted(groups.items(), key=group_sort):
        if skip is not None:
            occ = [ev for ev in occ if not skip(ev)]
        if len(occ) < minrep:
            continue
        cb_id = next_id
        next_id += 1
        occ = sorted(occ, key=occ_sort)
        g_drop, g_rows = [], []
        for j, ev in enumerate(occ):
            rows = emit_first(cb_id, occ) if j == 0 else emit_ref(cb_id, ev)
            pos = pos_of(ev)
            for row in rows:
                row["__pos"] = pos
                g_rows.append(row)
            g_drop.extend(rows_of(ev))
            if consumed is not None:
                consumed.add(id(ev))
        grouped.append((g_drop, g_rows))
        drop_idx.extend(g_drop)
        new_rows.extend(g_rows)
    if per_group:
        return grouped, next_id
    return drop_idx, new_rows, next_id
