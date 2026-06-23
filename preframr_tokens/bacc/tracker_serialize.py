"""Serialize the canonical Tracker IR to the model-facing token stream.

A ``driver="generic"`` :class:`~preframr_tokens.bacc.primitive.BaccProgram` is
compiled into the :class:`~preframr_tokens.bacc.tracker_ir.TrackerIR` (a shared
instrument/generator pool + per-voice BUNDLED instrument rows + free sibling lanes +
the four global lanes), then serialized through the SHARED post-BACC score machinery
(:func:`serialize._lz_emit_t`):

* the POOL holds each unique seed-and-pitch-invariant instrument struct ONCE,
  token-LZ'd across structs (two instruments sharing a long sub-table collapse), each
  carrying its fixed SEED SCHEMA (the SEED_KEYS that instrument's per-note residue
  uses -- constant per struct, so a row's seed is just bare VALUES, no keys/tags);
* each voice's BUNDLED rows are one LZ stream of ``(dt, dur, refs, bases, seeds)``
  records (a voice's freq + folded pw/ctrl/ad/sr ride ONE row), factored by inline
  backward REPEAT (an exact phrase repeat) and TRANSPOSE (a phrase replayed at a
  constant note-table-index shift -- a tracker orderlist Transpose);
* free sibling lanes and the global lanes are per-lane LZ'd event streams.

A pitched freq onset's pitch rides as a note-table-RELATIVE ``base`` index (the
canonical cross-driver alphabet -- see :mod:`pitch`); an algorithmic/unpitched
segment carries an absolute generator ref instead.  When the tune has no note table
the ``base`` is constant (-1) and is dropped from every row entirely.  No new token
ids -- the base-16 LEB digit alphabet plus the shared REPEAT/TRANSPOSE markers.

The serializer round-trips byte-exact to the IR, which :func:`tracker_ir.unlift`
inverts to the EXACT ``genfits``/``eventfits`` the per-register renderer consumes, so
``render(unlift(deserialize(serialize(ir)))) == state`` byte-for-byte; the
render-equality self-check (HARD RULE #0) RAISES on any mismatch (no silent escape).
"""

from preframr_tokens.bacc.generic.tracker import SEED_KEYS
from preframr_tokens.bacc.primitive import BaccProgram
from preframr_tokens.bacc.serialize import (
    REPEAT,
    _lz_emit_t,
    _lz_read_t,
    _ri,
    _ru,
    _u_len,
    _wi,
    _wu,
)
from preframr_tokens.bacc.tracker_ir import (
    CLASSES,
    GLOBALS,
    TrackerIR,
    VoiceTrack,
)

NREG = 25
_INSTR_REF_BIAS = 1  # ref -1 (un-fit) encodes as 0; a real ref k encodes as k+1
# The pool token stream uses only value tags (0..6) + LEB digits (0..31), all below
# the REPEAT marker (32), so REPEAT is a safe in-stream escape: the pool's flattened
# tokens are backward-LZ'd at TOKEN granularity so two instruments sharing a long
# sub-table (the SAME wavetable read at a different phase) collapse to one copy.
_POOL_MIN_COPY = 3  # a copy costs REPEAT + off + len (>=3 tokens); break even at 3

# --- the seed key alphabet (a compact per-note residue codec) ---------------
# The per-note seed is a small dict over the fixed SEED_KEYS, and the SET of keys an
# instrument uses is CONSTANT per pool struct (the struct IS the seed-invariant
# program).  So the keys are stored ONCE per pool entry as a SCHEMA (a list of
# key-indices), and each row's seed is just the bare VALUES in schema order -- no
# per-row key strings, dict tags, or counts (the dominant per-row saving on a melodic
# hold / per-note-sweep lane).
_SEED_KEY_INDEX = {k: i for i, k in enumerate(SEED_KEYS)}
_SEED_KEY_LIST = list(SEED_KEYS)


# --- generic JSON-value (de)serialization for instrument structs ------------
# The instrument struct is a small JSON-clean dict/list of ints, str keys, and the
# "rel" pitch sentinel; encode it by value-type so it stays correct as the archetype
# set evolves.  Used for the POOL (once per unique struct), never per row.
#
# A homogeneous INT list -- the dominant pool payload, a generator's period-P
# ``table`` (an accum's signed steps, a read's values) -- gets its own tag ``_T_IARR``
# so the per-element ``_T_INT`` type tag is paid ONCE for the whole array, not once per
# element.  A dense-modulation lane's table is dozens-to-hundreds of ints, so dropping
# the per-element tag is the single largest pool saving (the tag was ~a third of every
# table's tokens).  All tags stay < REPEAT (32), so the pool's token-LZ escape is still
# safe.  A mixed/nested list (e.g. a pitch-factored ``["rel", 0, -3]`` -- a str + ints,
# or a piecewise ``["P", ...]`` body) keeps the generic ``_T_LIST`` element-wise form.
_T_NONE, _T_FALSE, _T_TRUE, _T_INT, _T_STR, _T_LIST, _T_DICT, _T_IARR = range(8)


def _is_int_array(value):
    """True for a non-empty list/tuple of plain ints (not bools) -- the homogeneous
    int payload ``_T_IARR`` packs tag-free.  Empty stays generic (no payload to save).
    """
    return bool(value) and all(
        isinstance(x, int) and not isinstance(x, bool) for x in value
    )


def _write_value(out, value):
    if value is None:
        out.append(_T_NONE)
    elif value is True:
        out.append(_T_TRUE)
    elif value is False:
        out.append(_T_FALSE)
    elif isinstance(value, int):
        out.append(_T_INT)
        _wi(out, value)
    elif isinstance(value, str):
        out.append(_T_STR)
        data = value.encode("utf-8")
        _wu(out, len(data))
        for b in data:
            _wu(out, b)
    elif isinstance(value, (list, tuple)):
        if _is_int_array(value):
            out.append(_T_IARR)
            _wu(out, len(value))
            for item in value:
                _wi(out, item)
        else:
            out.append(_T_LIST)
            _wu(out, len(value))
            for item in value:
                _write_value(out, item)
    elif isinstance(value, dict):
        out.append(_T_DICT)
        _wu(out, len(value))
        for key, item in value.items():
            _write_value(out, key)
            _write_value(out, item)
    else:
        raise TypeError(f"tracker_serialize: unsupported value {type(value)!r}")


def _read_value(ids, i):
    tag = ids[i]
    i += 1
    if tag == _T_NONE:
        return None, i
    if tag == _T_TRUE:
        return True, i
    if tag == _T_FALSE:
        return False, i
    if tag == _T_INT:
        return _ri(ids, i)
    if tag == _T_STR:
        n, i = _ru(ids, i)
        data = bytearray()
        for _ in range(n):
            b, i = _ru(ids, i)
            data.append(b)
        return data.decode("utf-8"), i
    if tag == _T_IARR:
        n, i = _ru(ids, i)
        items = []
        for _ in range(n):
            item, i = _ri(ids, i)
            items.append(item)
        return items, i
    if tag == _T_LIST:
        n, i = _ru(ids, i)
        items = []
        for _ in range(n):
            item, i = _read_value(ids, i)
            items.append(item)
        return items, i
    if tag == _T_DICT:
        n, i = _ru(ids, i)
        out = {}
        for _ in range(n):
            key, i = _read_value(ids, i)
            item, i = _read_value(ids, i)
            out[key] = item
        return out, i
    raise ValueError(f"tracker_serialize: unknown value tag {tag}")


# --- the per-ref seed SCHEMA + base shape -----------------------------------
# A pool entry's schema records, for a simple ("S") body, the ordered seed keys; for
# a piecewise ("P") body, the per-piece ordered seed keys.  ``ref == -1`` (un-fit) has
# no body so its events carry no base/seed at all.  The schema is derived from the IR
# events (consistent per ref by construction), emitted once after the pool, and used
# to decode each row's base/seed positionally.
def _schema_of_seed(seed):
    """The ordered seed-key tuple of one (simple) seed dict (``()`` for None/empty)."""
    if not seed:
        return ()
    return tuple(seed.keys())


def _collect_schemas(ir):
    """For each pool index, the seed schema: an ``("S", key_tuple)`` or
    ``("P", [key_tuple, ...])``.  Scans every event/row once (every reference to a ref
    carries the SAME schema, asserted on mismatch -- a structural invariant)."""
    schema = {}

    def note(ref, seed):
        if ref < 0:
            return
        tag = ir.pool[ref][0]
        if tag == "P":
            sc = ("P", tuple(_schema_of_seed(ps) for ps in (seed or [])))
        else:
            sc = ("S", _schema_of_seed(seed))
        prev = schema.get(ref)
        if prev is None:
            schema[ref] = sc
        elif prev != sc:
            raise ValueError(
                f"tracker_serialize: ref {ref} has inconsistent seed schema "
                f"{prev!r} vs {sc!r} (the struct must determine its seed keys)"
            )

    for track in ir.voices:
        for _dur, refs, _bases, seeds in track.rows:
            for ref, seed in zip(refs, seeds):
                note(ref, seed)
        for events in track.free.values():
            for _dur, ref, _base, seed in events:
                note(ref, seed)
    for events in ir.globals.values():
        for _dur, ref, _base, seed in events:
            note(ref, seed)
    # A pool entry no event references (cannot happen via lift) defaults to empty.
    out = []
    for ref in range(len(ir.pool)):
        sc = schema.get(ref)
        if sc is None:
            tag = ir.pool[ref][0]
            sc = ("P", ()) if tag == "P" else ("S", ())
        out.append(sc)
    return out


def _emit_one_schema(out, schema):
    """Emit ONE seed schema: a tag (0=S, 1=P) then the ordered seed-key indices (a
    simple key list, or a per-piece key list)."""
    kind, body = schema
    if kind == "S":
        out.append(0)
        _wu(out, len(body))
        for key in body:
            _wu(out, _SEED_KEY_INDEX[key])
    else:
        out.append(1)
        _wu(out, len(body))
        for piece in body:
            _wu(out, len(piece))
            for key in piece:
                _wu(out, _SEED_KEY_INDEX[key])


def _read_one_schema(ids, i):
    kind = ids[i]
    i += 1
    if kind == 0:
        n, i = _ru(ids, i)
        keys = []
        for _ in range(n):
            k, i = _ru(ids, i)
            keys.append(_SEED_KEY_LIST[k])
        return ("S", tuple(keys)), i
    npiece, i = _ru(ids, i)
    pieces = []
    for _ in range(npiece):
        m, i = _ru(ids, i)
        keys = []
        for _ in range(m):
            k, i = _ru(ids, i)
            keys.append(_SEED_KEY_LIST[k])
        pieces.append(tuple(keys))
    return ("P", tuple(pieces)), i


def _emit_schemas(out, schemas):
    """Emit the per-ref seed schemas as a small DICTIONARY + per-entry index.

    A seed schema is one of only a few distinct shapes (``('seed',)``,
    ``('lead','seed')``, ...) but EVERY pool entry carries one, so emitting the full
    key list per entry repeats the same handful of shapes dozens of times.  Instead the
    DISTINCT schemas (first-seen order) are emitted ONCE as a dictionary, then each pool
    entry is a single index into it -- the per-entry cost drops from a whole key list to
    one LEB index (the dominant schema-block saving on a many-instrument tune)."""
    distinct = []
    index = {}
    for sc in schemas:
        if sc not in index:
            index[sc] = len(distinct)
            distinct.append(sc)
    _wu(out, len(distinct))
    for sc in distinct:
        _emit_one_schema(out, sc)
    for sc in schemas:
        _wu(out, index[sc])


def _read_schemas(ids, i, npool):
    ndistinct, i = _ru(ids, i)
    distinct = []
    for _ in range(ndistinct):
        sc, i = _read_one_schema(ids, i)
        distinct.append(sc)
    schemas = []
    for _ in range(npool):
        idx, i = _ru(ids, i)
        schemas.append(distinct[idx])
    return schemas, i


# --- hashable canonicalisation for the backward-LZ key index ----------------
def _hashable(x):
    """A hashable canonical form of a base/seed value that equals iff the originals are
    ``==``: lists become tuples, dicts become their sorted ``(key, value)`` tuple (so
    insertion order never matters), scalars/None pass through.  Used to derive the
    backward-LZ equality + transpose-compatibility keys (the keys only PRUNE which
    positions the match probes -- ``==`` / ``delta_of`` still decide -- so an exact
    canonical form keeps the LZ byte-identical)."""
    if isinstance(x, list):
        return tuple(_hashable(v) for v in x)
    if isinstance(x, dict):
        return tuple((k, _hashable(v)) for k, v in sorted(x.items()))
    return x


# A sentinel standing in for a free (transposable) positive pitch base in the
# transpose-compatibility key: two rows can only transpose-relate when their negative
# (absolute) bases are EQUAL and every positive base is free, so positive bases all map
# to this one token (``delta_of`` then computes/validates the shift).
_XPOSE_FREE = object()


# --- one (dur, ref, base, seed) free-lane event literal, schema-driven ------
# ``ctx`` carries the decode-shared (schemas, has_note_table).  The segment START is
# NOT stored (the cover tiles a lane contiguously from 0, so start = running sum of
# durs).  base is emitted only when the tune has a note table (else it is the constant
# -1 and is implied); seed is the bare values in the ref's schema order (no per-row
# keys/tags -- the schema is stored once per pool entry).
def _emit_one(out, ev, ctx):
    dur, ref, base, seed = ev
    _wu(out, dur)
    _emit_lane_entry(out, ref, base, seed, ctx)


def _read_one(ids, i, ctx):
    dur, i = _ru(ids, i)
    ref, base, seed, i = _read_lane_entry(ids, i, ctx)
    return (dur, ref, base, seed), i


def _one_cost(ev, ctx):
    tmp = []
    _emit_one(tmp, ev, ctx)
    return len(tmp)


def _make_event_codec(ctx):
    """Bind the decode context into (lit, read, cost, delta, shift) callables for the
    free-lane event LZ (one event = one (dur, ref, base, seed) record)."""

    def lit(out, ev):
        _emit_one(out, ev, ctx)

    def read(ids, i):
        return _read_one(ids, i, ctx)

    def cost(ev):
        return _one_cost(ev, ctx)

    def delta(a, b):
        # dur, ref, seed identical; a single non-zero pitch-base shift (TRANSPOSE).
        if a[0] != b[0] or a[1] != b[1] or a[3] != b[3]:
            return None
        if isinstance(a[2], list) or isinstance(b[2], list):
            return None
        if a[2] < 0 or b[2] < 0:
            return None
        return b[2] - a[2]

    def shift(ev, d):
        dur, ref, base, seed = ev
        return (dur, ref, base + d, seed)

    def eq_key(ev):
        # Equal iff two events are ``==`` (the REPEAT match's necessary condition).
        return (ev[0], ev[1], _hashable(ev[2]), _hashable(ev[3]))

    def xpose_key(ev):
        # Equal for any pair ``delta`` can relate: same dur/ref/seed, and a positive
        # scalar base is FREE (folds to the sentinel); a list / negative base can never
        # transpose, so it carries its own value and never collides with a free one.
        base = ev[2]
        free = (
            _XPOSE_FREE
            if (not isinstance(base, list) and base >= 0)
            else _hashable(base)
        )
        return (ev[0], ev[1], _hashable(ev[3]), free)

    def xpose_vec(ev):
        # The single transposable (positive scalar) base of this event, as a one-lane
        # ``[value-or-None]`` vector (None when the base is a list / negative -- i.e.
        # the lane carries no shift, matching ``delta``'s list/negative handling).  Feeds
        # the njit LZ's numeric transpose encoding (see :mod:`bacc._lz_njit`).
        base = ev[2]
        return [base if (not isinstance(base, list) and base >= 0) else None]

    return lit, read, cost, delta, shift, eq_key, xpose_key, xpose_vec


# --- a bundled voice row (dur, [ref...], [base...], [seed...]) ---------------
def _make_row_codec(ctx, nlane):
    """Bind the decode context + the voice's fixed lane count into the bundled-row LZ
    callables.  A row is one dur + a per-lane (ref, base, seed) at that shared dur
    (each lane via the schema codec).  ``nlane`` (= 1 spine + bundled) is constant per
    voice, so it is emitted ONCE in the voice header, not per row."""

    def lit(out, row):
        dur, refs, bases, seeds = row
        _wu(out, dur)
        for ref, base, seed in zip(refs, bases, seeds):
            _emit_lane_entry(out, ref, base, seed, ctx)

    def read(ids, i):
        dur, i = _ru(ids, i)
        refs, bases, seeds = [], [], []
        for _ in range(nlane):
            ref, base, seed, i = _read_lane_entry(ids, i, ctx)
            refs.append(ref)
            bases.append(base)
            seeds.append(seed)
        return (dur, refs, bases, seeds), i

    def cost(row):
        tmp = []
        lit(tmp, row)
        return len(tmp)

    def delta(a, b):
        # same dur/refs/seeds; the spine (lane 0) shifts by a single interval, every
        # pitch-bearing lane by the same delta, absolute lanes unchanged.
        if a[0] != b[0] or a[1] != b[1] or a[3] != b[3]:
            return None
        d = None
        for ab, bb in zip(a[2], b[2]):
            if isinstance(ab, list) or isinstance(bb, list):
                return None
            if ab < 0 or bb < 0:
                if ab != bb:
                    return None
                continue
            dd = bb - ab
            if d is None:
                d = dd
            elif dd != d:
                return None
        return d

    def shift(row, d):
        dur, refs, bases, seeds = row
        nb = [b + d if (not isinstance(b, list) and b >= 0) else b for b in bases]
        return (dur, refs, nb, seeds)

    def eq_key(row):
        # Equal iff two rows are ``==`` (the REPEAT match's necessary condition).
        dur, refs, bases, seeds = row
        return (
            dur,
            tuple(refs),
            tuple(_hashable(b) for b in bases),
            tuple(_hashable(s) for s in seeds),
        )

    def xpose_key(row):
        # Equal for any pair ``delta`` can relate: same dur/refs/seeds and the same
        # absolute (list / negative) bases at the same lanes; every positive lane base
        # is FREE (folds to the sentinel -- ``delta`` computes the shared shift).
        dur, refs, bases, seeds = row
        bsig = tuple(
            _XPOSE_FREE if (not isinstance(b, list) and b >= 0) else _hashable(b)
            for b in bases
        )
        return (dur, tuple(refs), tuple(_hashable(s) for s in seeds), bsig)

    def xpose_vec(row):
        # The per-lane transposable (positive scalar) bases, as a ``[value-or-None]``
        # vector (None for a list / negative lane -- which carries no shift, matching
        # ``delta``).  Feeds the njit LZ's numeric transpose encoding
        # (:mod:`bacc._lz_njit`); ``nlane`` entries, the row's fixed lane count.
        _dur, _refs, bases, _seeds = row
        return [b if (not isinstance(b, list) and b >= 0) else None for b in bases]

    return lit, read, cost, delta, shift, eq_key, xpose_key, xpose_vec


def _emit_lane_entry(out, ref, base, seed, ctx):
    """One lane's (ref, base, seed) inside a bundled row (schema-driven, no tags)."""
    schemas, has_nt = ctx
    _wu(out, ref + _INSTR_REF_BIAS)
    if ref < 0:
        return
    kind, body = schemas[ref]
    if kind == "P":
        if has_nt:
            for b in base:
                _wu(out, b + 1)
        for keys, ps in zip(body, seed):
            for key in keys:
                _wi(out, ps[key])
    else:
        if has_nt:
            _wu(out, base + 1)
        for key in body:
            _wi(out, seed[key])


def _read_lane_entry(ids, i, ctx):
    schemas, has_nt = ctx
    rref, i = _ru(ids, i)
    ref = rref - _INSTR_REF_BIAS
    if ref < 0:
        return ref, -1, None, i
    kind, body = schemas[ref]
    if kind == "P":
        npiece = len(body)
        bases = [-1] * npiece
        if has_nt:
            bases = []
            for _ in range(npiece):
                b, i = _ru(ids, i)
                bases.append(b - 1)
        seeds = []
        for keys in body:
            ps = {}
            for key in keys:
                v, i = _ri(ids, i)
                ps[key] = v
            seeds.append(ps)
        return ref, bases, seeds, i
    base = -1
    if has_nt:
        b, i = _ru(ids, i)
        base = b - 1
    seed = {}
    for key in body:
        v, i = _ri(ids, i)
        seed[key] = v
    return ref, base, seed, i


# --- instrument pool (defined ONCE up front, then referenced by index) -------
def _flatten_pool(pool):
    """The pool's entries flattened to one value-token stream (no LZ)."""
    flat = []
    for entry in pool:
        _write_value(flat, entry)
    return flat


def _lz_tokens(stream, min_copy=_POOL_MIN_COPY):
    """Backward-LZ a flat token ``stream`` into an output token list, escaping a copy
    with the :data:`REPEAT` marker (safe: the value stream never emits a token >=
    REPEAT).  Hash-of-3-grams accelerated greedy match (O(n)); a match is taken only
    when strictly shorter than the literals it replaces.

    The greedy match runs on the njit :func:`bacc._lz_njit.token_lz_plan_kernel`
    (byte-identical to the former Python loop); the host replays the plan with the LEB
    emitters.  When numba is absent the kernel decorator degrades to plain Python, so the
    behaviour is unchanged either way."""
    import numpy as np

    from preframr_tokens.bacc import _lz_njit as LK

    n = len(stream)
    if n == 0:
        return []
    arr = np.asarray(stream, dtype=np.int64)
    plan = LK.token_lz_plan_kernel(arr, int(min_copy))
    out = []
    for is_copy, off, length in plan:
        if is_copy:
            out.append(REPEAT)
            _wu(out, int(off))
            _wu(out, int(length))
        else:
            out.append(int(stream[int(off)]))
    return out


def _unlz_tokens(ids, i, ntokens):
    """Inverse of :func:`_lz_tokens`: rebuild ``ntokens`` flat value-tokens."""
    flat = []
    while len(flat) < ntokens:
        tok = ids[i]
        if tok == REPEAT:
            i += 1
            off, i = _ru(ids, i)
            length, i = _ru(ids, i)
            base = len(flat)
            for j in range(length):
                flat.append(flat[base - off + j])
        else:
            flat.append(tok)
            i += 1
    return flat, i


def _emit_pool(out, pool):
    """Emit the instrument pool: its count, its flat-token length, then the TOKEN-LZ'd
    flat stream (so shared sub-tables across instruments collapse)."""
    _wu(out, len(pool))
    flat = _flatten_pool(pool)
    _wu(out, len(flat))
    out.extend(_lz_tokens(flat))


def _read_pool(ids, i):
    n, i = _ru(ids, i)
    ntokens, i = _ru(ids, i)
    flat, i = _unlz_tokens(ids, i, ntokens)
    pool, j = [], 0
    for _ in range(n):
        entry, j = _read_value(flat, j)
        pool.append(entry)
    return pool, i


# --- lane / row stream encode (REPEAT + TRANSPOSE, or token-LZ; pick cheaper) ---
# A stream of comparable items (free-lane events / bundled voice rows) is encoded the
# cheaper of two ways, with a 1-token selector:
#   MODE 0  ITEM-LZ: the shared :func:`serialize._lz_emit_t` -- inline backward
#           REPEAT (an exact phrase repeat) + TRANSPOSE (a phrase replayed at a
#           constant note-table-index shift), the cross-driver phrase machinery.
#   MODE 1  TOKEN-LZ: the items flattened to one value-token stream and backward-LZ'd
#           at TOKEN granularity (:func:`_lz_tokens`) -- collapses VALUE-level
#           redundancy a phrase-granularity REPEAT misses (the same few dt / ref /
#           seed values recurring out of phase, the dominant generic-lane redundancy).
# TRANSPOSE wins on a pitch-shifted tracker phrase; TOKEN-LZ wins on a busy lane with
# no clean phrase structure -- so the encoder measures both and keeps the smaller, the
# prompt's "whichever is fewer tokens" applied per stream (decode reads the selector).
_MODE_ITEM_LZ = 0
_MODE_TOKEN_LZ = 1


def _emit_stream(
    out, items, lit, cost, delta, eq_key=None, xpose_key=None, xpose_vec=None
):
    """Emit ``len(items)`` then the cheaper of {item-LZ (REPEAT+TRANSPOSE), token-LZ}
    of ``items`` (a 1-token selector picks the mode; ``lit``/``cost`` render/measure a
    single item, ``delta`` is the TRANSPOSE interval test).  ``eq_key``/``xpose_key``
    are the optional hashable-key accelerators for the backward-LZ (byte-identical);
    ``xpose_vec`` is the optional per-item positive-base vector that lets the search run
    on the numeric njit kernel (also byte-identical)."""
    _wu(out, len(items))
    if not items:
        return
    item_body = []
    _lz_emit_t(
        item_body,
        items,
        cost,
        lit,
        delta,
        eq_key=eq_key,
        xpose_key=xpose_key,
        xpose_vec=xpose_vec,
    )
    flat = []
    for it in items:
        lit(flat, it)
    token_body = _lz_tokens(flat, _POOL_MIN_COPY)
    # +1 for the mode selector either way; token-LZ also stores the flat length.
    if 1 + len(item_body) <= 1 + _u_len(len(flat)) + len(token_body):
        out.append(_MODE_ITEM_LZ)
        out.extend(item_body)
    else:
        out.append(_MODE_TOKEN_LZ)
        _wu(out, len(flat))
        out.extend(token_body)


def _read_stream(ids, i, read, shift):
    """Inverse of :func:`_emit_stream`: rebuild the item list (``read`` parses one item
    from a flat token list; ``shift`` re-coordinates a TRANSPOSE copy)."""
    n, i = _ru(ids, i)
    if n == 0:
        return [], i
    mode = ids[i]
    i += 1
    if mode == _MODE_ITEM_LZ:
        items, i = _lz_read_t(ids, i, n, read, shift)
        return items, i
    ntokens, i = _ru(ids, i)
    flat, i = _unlz_tokens(ids, i, ntokens)
    items, j = [], 0
    for _ in range(n):
        item, j = read(flat, j)
        items.append(item)
    return items, i


def _emit_events(out, events, ctx):
    """Encode a free lane's ``(dt, dur, ref, base, seed)`` event stream."""
    lit, _read, cost, delta, _shift, eq_key, xpose_key, xpose_vec = _make_event_codec(
        ctx
    )
    _emit_stream(
        out,
        events,
        lit,
        cost,
        delta,
        eq_key=eq_key,
        xpose_key=xpose_key,
        xpose_vec=xpose_vec,
    )


def _read_events(ids, i, ctx):
    _lit, read, _cost, _delta, shift, _eqk, _xk, _xv = _make_event_codec(ctx)
    return _read_stream(ids, i, read, shift)


def _emit_rows(out, rows, ctx, nlane):
    """Encode a voice's BUNDLED row stream (TRANSPOSE factors a transposed bundled
    phrase -- a constant spine-pitch shift).  ``nlane`` is fixed for the voice (1 spine
    + bundled siblings) and derived from the voice header, not stored per row."""
    lit, _read, cost, delta, _shift, eq_key, xpose_key, xpose_vec = _make_row_codec(
        ctx, nlane
    )
    _emit_stream(
        out,
        rows,
        lit,
        cost,
        delta,
        eq_key=eq_key,
        xpose_key=xpose_key,
        xpose_vec=xpose_vec,
    )


def _read_rows(ids, i, ctx, nlane):
    _lit, read, _cost, _delta, shift, _eqk, _xk, _xv = _make_row_codec(ctx, nlane)
    return _read_stream(ids, i, read, shift)


# --- top-level header -------------------------------------------------------
def _emit_header(out, ir):
    _wu(out, ir.nframes)
    for b in ir.boot:
        _wu(out, b)
    if ir.note_table is None:
        _wu(out, 0)
    else:
        _wu(out, 1)
        _wu(out, len(ir.note_table))
        for v in ir.note_table:
            _wu(out, v)


def _read_header(ids, i):
    nframes, i = _ru(ids, i)
    boot = []
    for _ in range(NREG):
        b, i = _ru(ids, i)
        boot.append(b)
    has_nt, i = _ru(ids, i)
    if has_nt:
        n, i = _ru(ids, i)
        note_table = []
        for _ in range(n):
            v, i = _ru(ids, i)
            note_table.append(v)
    else:
        note_table = None
    return nframes, boot, note_table, i


# --- a class is one of CLASSES, encoded by its fixed index ------------------
_CLASS_INDEX = {cls: i for i, cls in enumerate(CLASSES)}
_CLASS_LIST = list(CLASSES)


def _emit_voice(out, track, ctx):
    # the bundled classes (a bitmask over CLASSES, decode-reproducible order); the row
    # lane count (1 spine + bundled) is derived from this mask, not stored per row.
    mask = 0
    for cls in track.bundled:
        mask |= 1 << _CLASS_INDEX[cls]
    _wu(out, mask)
    _emit_rows(out, track.rows, ctx, 1 + len(track.bundled))
    # the free sibling lanes (those not bundled), in CLASSES order
    free_classes = [cls for cls in CLASSES if cls in track.free]
    _wu(out, len(free_classes))
    for cls in free_classes:
        _wu(out, _CLASS_INDEX[cls])
        _emit_events(out, track.free[cls], ctx)


def _read_voice(ids, i, ctx):
    mask, i = _ru(ids, i)
    bundled = [cls for cls in CLASSES if mask & (1 << _CLASS_INDEX[cls])]
    rows, i = _read_rows(ids, i, ctx, 1 + len(bundled))
    nfree, i = _ru(ids, i)
    free = {}
    for _ in range(nfree):
        cidx, i = _ru(ids, i)
        events, i = _read_events(ids, i, ctx)
        free[_CLASS_LIST[cidx]] = events
    return VoiceTrack(bundled=bundled, rows=rows, free=free), i


# --- the IR codec -----------------------------------------------------------
def _ir_to_ids(ir):
    schemas = _collect_schemas(ir)
    ctx = (schemas, ir.note_table is not None)
    out = []
    _emit_header(out, ir)
    _emit_pool(out, ir.pool)
    _emit_schemas(out, schemas)
    for track in ir.voices:
        _emit_voice(out, track, ctx)
    for reg in GLOBALS:
        _emit_events(out, ir.globals[reg], ctx)
    return out


def _ids_to_ir(ids):
    i = 0
    nframes, boot, note_table, i = _read_header(ids, i)
    pool, i = _read_pool(ids, i)
    schemas, i = _read_schemas(ids, i, len(pool))
    ctx = (schemas, note_table is not None)
    voices = []
    for _ in range(3):
        track, i = _read_voice(ids, i, ctx)
        voices.append(track)
    globals_ = {}
    for reg in GLOBALS:
        events, i = _read_events(ids, i, ctx)
        globals_[reg] = events
    return TrackerIR(
        note_table=note_table,
        pool=pool,
        voices=voices,
        globals=globals_,
        nframes=nframes,
        boot=boot,
    )


# --- public interface (generic_serialize delegates here) --------------------
def _lift_program(program):
    """Compile a ``driver="generic"`` program into the Tracker IR via its rendered
    byte-exact state (the IR is a lossless re-expression of the cover, and the cover
    is byte-exact against the rendered state).

    The per-lane covers are computed ONCE; the IR is then built two ways -- with and
    without a synthesized pitch table (pitch-factoring trades a note-table header +
    pitch-invariant pool for 1-token note ``base`` indices and TRANSPOSE) -- and the
    SMALLER serialization is kept (the prompt's "whichever is fewer tokens", applied to
    the pitch axis).  When the tune already carries a driver note table the two builds
    coincide, so this is a no-op there."""
    from preframr_tokens.bacc.generic.recover import render_generic
    from preframr_tokens.bacc.tracker_ir import build_ir, cover_all_lanes

    note_table = program.tables.get("note_table")
    state = render_generic(program)
    covers = cover_all_lanes(state, note_table)
    nframes, boot = program.nframes, list(program.boot)
    ir_pf = build_ir(covers, note_table, nframes, boot, synth_pitch=True)
    if note_table is not None:
        return _ir_to_ids(ir_pf), ir_pf
    ir_plain = build_ir(covers, note_table, nframes, boot, synth_pitch=False)
    ids_pf, ids_plain = _ir_to_ids(ir_pf), _ir_to_ids(ir_plain)
    if len(ids_pf) <= len(ids_plain):
        return ids_pf, ir_pf
    return ids_plain, ir_plain


def tracker_program_to_ids(program):
    """Serialize a tracker-lifted generic program to token ids (inverse of
    :func:`tracker_ids_to_program`)."""
    ids, _ir = _lift_program(program)
    return ids


def tracker_ids_to_program(ids):
    """Inverse of :func:`tracker_program_to_ids` -> a ``driver="generic"`` program
    whose ``tables`` carry the reconstructed ``genfits``/``eventfits``."""
    from preframr_tokens.bacc.tracker_ir import unlift

    ir = _ids_to_ir(ids)
    genfits, eventfits = unlift(ir)
    return BaccProgram(
        driver="generic",
        nframes=ir.nframes,
        boot=ir.boot,
        instruments=[],
        score=[],
        seed={},
        tables={
            "note_table": ir.note_table,
            "genfits": _wrap_genfits(genfits),
            "eventfits": _wrap_eventfits(eventfits),
        },
    )


def _wrap_genfits(genfits):
    """Re-wrap unlift's ``key -> segments`` into the ``recover`` genfits shape
    (``"V:cls" -> {segments, carry}``); carry is recomputed at render, so None."""
    return {key: {"segments": segs, "carry": None} for key, segs in genfits.items()}


def _wrap_eventfits(eventfits):
    return {str(reg): segs for reg, segs in eventfits.items()}


def tracker_measure(program):
    """Return ``({block: tokens}, nframes)`` for the tracker token stream: the
    header / pool / per-voice row+free / globals split."""
    _ids, ir = _lift_program(program)
    return _measure_ir(ir, program.nframes)


def _measure_ir(ir, nframes):
    schemas = _collect_schemas(ir)
    ctx = (schemas, ir.note_table is not None)
    out = []
    _emit_header(out, ir)
    header = len(out)
    out = []
    _emit_pool(out, ir.pool)
    _emit_schemas(out, schemas)
    instr_def = len(out)
    out = []
    for track in ir.voices:
        _emit_voice(out, track, ctx)
    voices = len(out)
    out = []
    for reg in GLOBALS:
        _emit_events(out, ir.globals[reg], ctx)
    globals_ = len(out)
    brk = {
        "header": header,
        "instr_def": instr_def,
        "score": voices + globals_,
        "voices": voices,
        "globals": globals_,
        "n_instruments": len(ir.pool),
        "total": header + instr_def + voices + globals_,
    }
    return brk, nframes
