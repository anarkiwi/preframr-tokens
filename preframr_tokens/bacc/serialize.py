"""Shared post-BACC machinery + the GoatTracker/generic id-stream dispatchers.

This module holds the driver-AGNOSTIC token primitives every backend's serializer
reuses: the base-16 LEB128 reader/writers (``_wu``/``_ru``/``_wi``/``_ri``), the
inline backward-LZ with optional TRANSPOSE factoring (``_lz_emit_t``/``_lz_read_t``)
and its REPEAT/TRANSPOSE markers, plus the small length helpers (``_u_len``) and the
vocab constants (``VOCAB``/``PAD_ID``). ``gt_serialize``, ``tracker_serialize`` and
``generic_serialize`` import these names directly.

``program_to_ids``/``ids_to_program``/``measure`` are thin dispatchers that route a
BaccProgram to its driver's serializer -- now only ``goattracker`` and ``generic``
(the hand GoatTracker backend kept for reference, and the generic path under active
development). There is no hand-coded default driver: any other driver routes through
the generic path.
"""

# vocab: base-16 LEB digits 0..31 (0-15 continue, 16-31 terminal) + REPEAT marker
# + TRANSPOSE marker (a backward REPEAT whose copied notes are re-coordinated by a
# constant grid-interval Delta -- factors a phrase repeated at a different pitch,
# exactly what a tracker orderlist's Transpose(semitones) does, while the note
# token stays the ABSOLUTE canonical A440 grid index).
# Legacy inline-LZ markers. These belong to the v1 base-16 LEB alphabet still used
# by the generic/tracker serializer (pending its port to the flat v2 alphabet --
# task #4). The MODEL-FACING alphabet is now the flat, typed, no-LZ scheme defined
# in :mod:`preframr_tokens.bacc.flat_serialize`; ``VOCAB``/``PAD_ID`` below are ITS
# size (must equal ``flat_serialize.VOCAB`` -- asserted in tests).
REPEAT = 32
TRANSPOSE = 33
VOCAB = 544
PAD_ID = VOCAB  # reserved padding id above the codec alphabet

_MIN_COPY = 2


def _wu(out, n):
    n = int(n)
    while True:
        d = n & 0xF
        n >>= 4
        out.append(d if n else 16 + d)
        if not n:
            return


def _wi(out, n):
    n = int(n)
    _wu(out, (n << 1) ^ (n >> 63))


def _ru(ids, i):
    n = shift = 0
    while True:
        d = ids[i]
        i += 1
        n |= (d & 0xF) << shift
        if d >= 16:
            return n, i
        shift += 4


def _ri(ids, i):
    z, i = _ru(ids, i)
    return (z >> 1) ^ -(z & 1), i


# --- shared backward-LZ with TRANSPOSE factoring (post-BACC, driver-agnostic) ---
# The dominant score block of EVERY driver is a list of comparable items (per-voice
# note-on rows, GoatTracker pattern rows, ...). The good compression -- inline
# backward REPEAT for an exact phrase repeat plus TRANSPOSE (a backward REPEAT
# re-coordinated by a constant pitch Delta on the canonical A440 grid, exactly what
# a tracker orderlist's Transpose(semitones) does) -- is the SAME machinery for all
# of them; only the per-item literal and the transpose-delta test are
# driver-specific. Factoring it here (keyed off the common score-item list) is what
# lets every driver's score get REPEAT/TRANSPOSE phrase factoring.
#
# ``delta_of(a, b)`` returns the constant grid-interval that makes ``b`` a
# transposed copy of ``a`` (every non-pitch field identical, both notes
# grid-resolvable), or ``None`` if ``b`` is not a constant-pitch shift of ``a``.
# A driver that has no transposable pitch axis passes ``delta_of=None`` and gets
# plain REPEAT-only LZ.
def _transposed_run(items, i, off, delta, delta_of):
    """Length of the backward run at ``i`` (source ``i-off``) every item of which
    is the prior run's item shifted by exactly ``delta`` (non-pitch fields
    identical). The source items must themselves be grid-resolvable so decode can
    re-add ``delta``."""
    n = 0
    while i + n < len(items):
        if delta_of(items[i - off + n], items[i + n]) != delta:
            break
        n += 1
    return n


def _best_repeat(items, i):
    """Best (length, offset) exact backward run at ``i`` -- the longest run of items
    equal to a prior run, ties broken to the SMALLEST offset (the reference scans
    ``off`` ascending and keeps the first maximum)."""
    best_len, best_off = 0, 0
    for off in range(1, i + 1):
        n = 0
        while i + n < len(items) and items[i - off + n] == items[i + n]:
            n += 1
        if n > best_len:
            best_len, best_off = n, off
    return best_len, best_off


def _best_repeat_indexed(items, i, eqkeys, posns):
    """``_best_repeat`` accelerated by an equality-key index: an exact run can only
    start at a prior position ``p`` with ``eqkeys[p] == eqkeys[i]`` (a necessary
    condition for ``items[p] == items[i]``).  ``posns[key]`` is the ascending list of
    PRIOR positions (``p < i`` -- the caller adds each position only after the cursor
    passes it, so the index never offers a forward source) with that key; scanning it
    DESCENDING visits offsets ascending, so the first strict maximum is the
    smallest-offset longest run -- the SAME ``(best_len, best_off)`` the dense scan
    returns (byte-identical), but probing only matching starts instead of all ``i``."""
    cand = posns.get(eqkeys[i])
    if not cand:
        return 0, 0
    best_len, best_off = 0, 0
    n_items = len(items)
    # A run may overlap its source (length > offset, e.g. a constant stream), so every
    # same-key candidate is probed; the inner loop stops at the first divergence.
    for k in range(len(cand) - 1, -1, -1):
        p = cand[k]
        n = 0
        while i + n < n_items and items[p + n] == items[i + n]:
            n += 1
        if n > best_len:
            best_len, best_off = n, i - p
    return best_len, best_off


def _best_transpose(items, i, delta_of):
    """Best (length, offset, delta) backward run matching a prior run up to a
    single constant non-zero grid-interval (a transposed phrase repeat)."""
    best_len, best_off, best_delta = 0, 0, 0
    for off in range(1, i + 1):
        delta = delta_of(items[i - off], items[i])
        if delta in (None, 0):
            continue  # delta==0 is the exact REPEAT case, handled separately
        n = _transposed_run(items, i, off, delta, delta_of)
        if n > best_len:
            best_len, best_off, best_delta = n, off, delta
    return best_len, best_off, best_delta


def _best_transpose_indexed(items, i, delta_of, xkeys, xposns):
    """``_best_transpose`` accelerated by a transpose-compatibility-key index.

    ``delta_of(items[p], items[i])`` is non-None only when the two items agree on
    every non-pitch field, so a transpose source can only start at a prior ``p`` with
    ``xkeys[p] == xkeys[i]`` (a NECESSARY condition -- ``delta_of`` is still called on
    every probed candidate, so a false-positive key collision is simply rejected).
    ``xposns`` holds only PRIOR positions (``p < i``); scanning them DESCENDING visits
    offsets ascending, so the first strict maximum is the smallest-offset longest
    transposed run -- the SAME ``(best_len, best_off, best_delta)`` the dense scan
    returns."""
    cand = xposns.get(xkeys[i])
    if not cand:
        return 0, 0, 0
    best_len, best_off, best_delta = 0, 0, 0
    # ``cand`` ascending; scanned DESCENDING -> offsets ascending, first strict maximum
    # is the smallest-offset longest transposed run (the dense scan's result).  A
    # transposed run may overlap its source, so every same-key candidate is probed.
    for k in range(len(cand) - 1, -1, -1):
        p = cand[k]
        off = i - p
        delta = delta_of(items[p], items[i])
        if delta in (None, 0):
            continue
        n = _transposed_run(items, i, off, delta, delta_of)
        if n > best_len:
            best_len, best_off, best_delta = n, off, delta
    return best_len, best_off, best_delta


def _factorize(keys):
    """Map a list of hashable keys to dense ``int64`` ids (0..K-1) preserving equality:
    ``ids[a] == ids[b]`` iff ``keys[a] == keys[b]``.  Feeds the numeric njit LZ."""
    import numpy as np

    pool = {}
    ids = np.empty(len(keys), dtype=np.int64)
    for idx, k in enumerate(keys):
        j = pool.get(k)
        if j is None:
            j = len(pool)
            pool[k] = j
        ids[idx] = j
    return ids


def _lz_emit_via_kernel(out, items, lit_cost, lit_emit, delta_of, eqkeys, xkeys, xvecs):
    """Run the backward-LZ on the :mod:`bacc._lz_njit` numeric kernel (byte-identical to
    the Python reference) and replay its plan into ``out``.  Returns False (so the caller
    falls back to the Python path) when the encoding cannot be built numerically -- numba
    absent, or any item carries a non-int positive base the integer matrix can't hold.
    The kernel reproduces the EXACT same per-position copy/transpose/literal decision; we
    only move the O(matches) inner search off Python tuples."""
    from preframr_tokens.bacc import _lz_njit as LK
    from preframr_tokens.bacc.generic._njit import HAVE_NUMBA

    if not HAVE_NUMBA:
        return False
    import numpy as np

    n = len(items)
    use_xpose = 1 if (delta_of is not None and xvecs is not None) else 0
    nlane = len(xvecs[0]) if (use_xpose and n) else 1
    posbase = np.full((n, nlane), LK._NEG, dtype=np.int64)
    xelig = np.zeros(n, dtype=np.int64)
    if use_xpose:
        for r in range(n):
            vec = xvecs[r]
            any_pos = False
            for l, val in enumerate(vec):
                if val is None:
                    continue
                if not isinstance(val, int) or val < 0:
                    return False  # non-int positive base -> use the Python reference
                posbase[r, l] = val
                any_pos = True
            xelig[r] = 1 if any_pos else 0
    eqid = _factorize(eqkeys)
    xid = _factorize(xkeys) if use_xpose else eqid
    litcost = np.fromiter((lit_cost(it) for it in items), dtype=np.int64, count=n)
    prefix = np.zeros(n + 1, dtype=np.int64)
    prefix[1:] = np.cumsum(litcost)
    plan = LK.lz_plan_kernel(
        eqid, xid, posbase, xelig, litcost, prefix, nlane, use_xpose, _MIN_COPY
    )
    for op, a, length, delta in plan:
        if op == LK.OP_TRANSPOSE:
            out.append(TRANSPOSE)
            _wu(out, int(a))
            _wu(out, int(length))
            _wi(out, int(delta))
        elif op == LK.OP_REPEAT:
            out.append(REPEAT)
            _wu(out, int(a))
            _wu(out, int(length))
        else:
            lit_emit(out, items[int(a)])
    return True


def _lz_emit_t(
    out,
    items,
    lit_cost,
    lit_emit,
    delta_of=None,
    eq_key=None,
    xpose_key=None,
    xpose_vec=None,
):
    """Inline backward-LZ over ``items`` with optional TRANSPOSE factoring.

    Literals are emitted via ``lit_emit(out, item)`` and costed via
    ``lit_cost(item)`` (byte length). A copy is REPEAT(offset, length) over prior
    items; when ``delta_of`` is given, a TRANSPOSE(offset, length, Delta) copies a
    prior run re-coordinated by a constant grid-interval. The cheapest of {exact
    REPEAT, transposed REPEAT+Delta, literal} is chosen per position, weighing each
    candidate by tokens-saved so a longer plain REPEAT is not beaten by a shorter
    transposed one (and vice versa) -- so enabling TRANSPOSE never costs more than
    REPEAT-only would have.

    ``eq_key`` / ``xpose_key`` are OPTIONAL hashable-key functions that make the
    backward search sub-quadratic without changing its output: ``eq_key(item)`` must
    be EQUAL for any two ``==`` items (so an exact match never starts at a position
    with a different key), and ``xpose_key(item)`` must be EQUAL for any two items
    ``delta_of`` relates (so a transpose never starts at a position with a different
    key).  When supplied, the match start is restricted to the index of matching
    positions -- byte-identical selection (the dense reference's smallest-offset
    longest run), but O(matches) instead of O(i) probes per position.  Absent (or one
    key None), the dense reference scan is used.

    ``xpose_vec(item)`` (the per-item positive-base vector) additionally enables the
    numeric njit kernel (:mod:`bacc._lz_njit`): with ``eq_key`` it precomputes the
    integer equality / transpose encoding and runs the whole search in compiled code,
    byte-identical to the Python path -- the cover path's dominant cost on a long digi
    row stream.  Any item the matrix can't hold (numba absent / a non-int base) falls
    back to the Python reference below."""
    n = len(items)
    # The numeric kernel needs both the equality key and the positive-base vector (and,
    # for transpose, the xpose key); when present and numba is available it replaces the
    # Python search below, byte-for-byte (the per-position decision is identical).
    if eq_key is not None and xpose_vec is not None and n:
        eqkeys_k = [eq_key(it) for it in items]
        xkeys_k = (
            [xpose_key(it) for it in items]
            if (delta_of is not None and xpose_key is not None)
            else eqkeys_k
        )
        xvecs_k = [xpose_vec(it) for it in items] if delta_of is not None else None
        if _lz_emit_via_kernel(
            out, items, lit_cost, lit_emit, delta_of, eqkeys_k, xkeys_k, xvecs_k
        ):
            return
    # Precompute the per-item keys (computing a key never introduces a match), but build
    # the position indexes INCREMENTALLY: a position ``j`` is added to the index only
    # after the cursor passes it, so a search at ``i`` sees exactly the prior positions
    # ``p < i`` -- the same source window the dense scan's ``off in range(1, i+1)`` walks
    # (a forward source would spuriously match future items).  Every position the cursor
    # advances over (including ones inside an emitted copy run) is indexed, because the
    # dense scan can match a later position against ANY earlier one.
    eqkeys = eqposns = None
    if eq_key is not None:
        eqkeys = [eq_key(it) for it in items]
        eqposns = {}
    xkeys = xposns = None
    if delta_of is not None and xpose_key is not None:
        xkeys = [xpose_key(it) for it in items]
        xposns = {}

    def _index_through(lo, hi):
        # Add positions [lo, hi) to whichever indexes are active.
        for j in range(lo, hi):
            if eqposns is not None:
                eqposns.setdefault(eqkeys[j], []).append(j)
            if xposns is not None:
                xposns.setdefault(xkeys[j], []).append(j)

    indexed_upto = 0
    i = 0
    while i < n:
        if eqposns is not None or xposns is not None:
            _index_through(indexed_upto, i)
            indexed_upto = i
        if eqposns is not None:
            best_len, best_off = _best_repeat_indexed(items, i, eqkeys, eqposns)
        else:
            best_len, best_off = _best_repeat(items, i)
        cost_copy = 1 + _u_len(best_off) + _u_len(best_len)
        lit_copy = sum(lit_cost(items[i + j]) for j in range(best_len))
        use_copy = best_len >= _MIN_COPY and cost_copy < lit_copy
        copy_gain = (lit_copy - cost_copy) if use_copy else 0
        if delta_of is not None:
            if xposns is not None:
                tlen, toff, tdelta = _best_transpose_indexed(
                    items, i, delta_of, xkeys, xposns
                )
            else:
                tlen, toff, tdelta = _best_transpose(items, i, delta_of)
            cost_trans = 1 + _u_len(toff) + _u_len(tlen) + _wi_len(tdelta)
            lit_trans = sum(lit_cost(items[i + j]) for j in range(tlen))
            use_trans = tlen >= _MIN_COPY and cost_trans < lit_trans
            trans_gain = (lit_trans - cost_trans) if use_trans else 0
            if use_trans and trans_gain >= copy_gain:
                out.append(TRANSPOSE)
                _wu(out, toff)
                _wu(out, tlen)
                _wi(out, tdelta)
                i += tlen
                continue
        if use_copy:
            out.append(REPEAT)
            _wu(out, best_off)
            _wu(out, best_len)
            i += best_len
        else:
            lit_emit(out, items[i])
            i += 1


def _lz_read_t(ids, i, count, lit_read, shift=None):
    """Inverse of ``_lz_emit_t``: rebuild ``count`` items, literals via
    ``lit_read(ids, i)``. A TRANSPOSE copies the prior run with each item
    re-coordinated by ``shift(item, delta)`` (a lossless grid re-coordinate; the
    non-pitch fields carry through unchanged). Returns ``(items, new_index)``."""
    items = []
    while len(items) < count:
        if ids[i] == REPEAT:
            i += 1
            off, i = _ru(ids, i)
            length, i = _ru(ids, i)
            base = len(items)
            for j in range(length):
                items.append(items[base - off + j])
        elif ids[i] == TRANSPOSE:
            i += 1
            off, i = _ru(ids, i)
            length, i = _ru(ids, i)
            delta, i = _ri(ids, i)
            base = len(items)
            for j in range(length):
                items.append(shift(items[base - off + j], delta))
        else:
            item, i = lit_read(ids, i)
            items.append(item)
    return items, i


def _u_len(n):
    out = []
    _wu(out, n)
    return len(out)


def _wi_len(n):
    out = []
    _wi(out, n)
    return len(out)


def program_to_ids(program):
    """Serialize a BaccProgram to a flat list of token ids (round-trippable).

    GoatTracker uses the flat v2 alphabet (:mod:`flat_serialize`); the generic
    path still emits the v1 LEB+LZ alphabet pending its port (task #4)."""
    if program.driver == "goattracker":
        from preframr_tokens.bacc.flat_serialize import flat_gt_program_to_ids

        return flat_gt_program_to_ids(program)
    from preframr_tokens.bacc.generic_serialize import generic_program_to_ids

    return generic_program_to_ids(program)


def ids_to_program(ids, driver="generic"):
    """Inverse of program_to_ids -> BaccProgram (instruments tables reconstructed)."""
    if driver == "goattracker":
        from preframr_tokens.bacc.flat_serialize import flat_gt_ids_to_program

        return flat_gt_ids_to_program(ids)
    from preframr_tokens.bacc.generic_serialize import generic_ids_to_program

    return generic_ids_to_program(ids)


def measure(program):
    """Return ({block: tokens}, nframes) for the serialized program."""
    if program.driver == "goattracker":
        from preframr_tokens.bacc.flat_serialize import flat_gt_measure

        return flat_gt_measure(program)
    from preframr_tokens.bacc.generic_serialize import generic_measure

    return generic_measure(program)
