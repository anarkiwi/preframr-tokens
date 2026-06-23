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
REPEAT = 32
TRANSPOSE = 33
VOCAB = 34
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


def _lz_emit_t(out, items, lit_cost, lit_emit, delta_of=None):
    """Inline backward-LZ over ``items`` with optional TRANSPOSE factoring.

    Literals are emitted via ``lit_emit(out, item)`` and costed via
    ``lit_cost(item)`` (byte length). A copy is REPEAT(offset, length) over prior
    items; when ``delta_of`` is given, a TRANSPOSE(offset, length, Delta) copies a
    prior run re-coordinated by a constant grid-interval. The cheapest of {exact
    REPEAT, transposed REPEAT+Delta, literal} is chosen per position, weighing each
    candidate by tokens-saved so a longer plain REPEAT is not beaten by a shorter
    transposed one (and vice versa) -- so enabling TRANSPOSE never costs more than
    REPEAT-only would have."""
    i = 0
    while i < len(items):
        best_len, best_off = 0, 0
        for off in range(1, i + 1):
            n = 0
            while i + n < len(items) and items[i - off + n] == items[i + n]:
                n += 1
            if n > best_len:
                best_len, best_off = n, off
        cost_copy = 1 + _u_len(best_off) + _u_len(best_len)
        lit_copy = sum(lit_cost(items[i + j]) for j in range(best_len))
        use_copy = best_len >= _MIN_COPY and cost_copy < lit_copy
        copy_gain = (lit_copy - cost_copy) if use_copy else 0
        if delta_of is not None:
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
    """Serialize a BaccProgram to a flat list of token ids (round-trippable)."""
    if program.driver == "goattracker":
        from preframr_tokens.bacc.gt_serialize import gt_program_to_ids

        return gt_program_to_ids(program)
    from preframr_tokens.bacc.generic_serialize import generic_program_to_ids

    return generic_program_to_ids(program)


def ids_to_program(ids, driver="generic"):
    """Inverse of program_to_ids -> BaccProgram (instruments tables reconstructed)."""
    if driver == "goattracker":
        from preframr_tokens.bacc.gt_serialize import gt_ids_to_program

        return gt_ids_to_program(ids)
    from preframr_tokens.bacc.generic_serialize import generic_ids_to_program

    return generic_ids_to_program(ids)


def measure(program):
    """Return ({block: tokens}, nframes) for the serialized program."""
    if program.driver == "goattracker":
        from preframr_tokens.bacc.gt_serialize import gt_measure

        return gt_measure(program)
    from preframr_tokens.bacc.generic_serialize import generic_measure

    return generic_measure(program)
