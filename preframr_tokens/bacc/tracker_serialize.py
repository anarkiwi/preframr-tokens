"""Serialize a tracker-LIFTED generic program to the model-facing token stream.

A generic :class:`~preframr_tokens.bacc.primitive.BaccProgram` lifts
(:mod:`preframr_tokens.bacc.generic.tracker`) into a shared INSTRUMENT pool + a
per-lane NOTE-EVENT stream.  This module serializes that tracker form through the
SHARED post-BACC score machinery (:func:`serialize._lz_emit_t`): each lane's event
stream is factored by inline backward REPEAT (an exact phrase repeat) and TRANSPOSE
(a phrase replayed at a constant note-table-index shift -- a tracker orderlist
Transpose), and every instrument is defined inline on first reference and
referenced by index thereafter (the inline-define-on-first-use dedup the Hubbard /
GoatTracker scores use).  No new token ids -- the base-16 LEB digit alphabet plus
the shared REPEAT/TRANSPOSE markers.

The header carries ``nframes`` + the frame-0 ``boot`` + the bus-recovered note
table; the body is the instrument pool (deferred -- emitted inline on first use)
and the per-lane LZ'd event streams.  Round-trips byte-exact to the lifted program,
which renders byte-exact (residual-zero) to the bus-state via
:func:`tracker.render_from_fits`.
"""

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

NREG = 25
_INSTR_REF_BIAS = 1  # ref -1 (un-fit) encodes as 0; a real ref k encodes as k+1
# The pool token stream uses only value tags (0..6) + LEB digits (0..31), all
# below the REPEAT marker (32), so REPEAT is a safe in-stream escape: the pool's
# flattened tokens are backward-LZ'd at TOKEN granularity (not entry granularity)
# so two instruments that share a long sub-table -- e.g. the SAME wavetable read
# at a different phase, the dominant Grid_Runner redundancy -- collapse to one
# copy.  This is a pure lossless re-encode of the already-deduped pool (HARD
# RULE #0): :func:`_read_pool_lz` rebuilds the identical flat stream byte-for-byte.
_POOL_MIN_COPY = 3  # a copy costs REPEAT + off + len (>=3 tokens); break even at 3


# --- generic JSON-value (de)serialization for instrument structs -----------
# The instrument struct is a small JSON-clean dict/list of ints, str keys, and the
# "rel" pitch sentinel; encode it by value-type (same scheme as generic_serialize)
# so it stays correct as the archetype set evolves.
_T_NONE, _T_FALSE, _T_TRUE, _T_INT, _T_STR, _T_LIST, _T_DICT = range(7)


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


# --- per-note seed / base (the per-event residue) --------------------------
def _write_seed(out, seed):
    """A simple event's seed is a dict; a piecewise event's is a list of dicts.
    ``None`` (an un-fit event) writes a single 0-length marker."""
    if seed is None:
        out.append(_T_NONE)
    elif isinstance(seed, list):
        out.append(_T_LIST)
        _wu(out, len(seed))
        for piece in seed:
            _write_value(out, piece)
    else:
        _write_value(out, seed)


def _read_seed(ids, i):
    return _read_value(ids, i)


def _write_base(out, base):
    """The pitch base index: an int (simple body) or a list of ints (piecewise).
    Written as a tagged value so it round-trips through :func:`_read_value`."""
    _write_value(out, base)


def _read_base(ids, i):
    return _read_value(ids, i)


# --- event literal (the LZ item) -------------------------------------------
def _event_lit(out, event):
    """Emit one note-event literal: (dt, dur, ref+bias, base, seed).  The
    instrument body is NOT emitted here (it is defined inline on first use by the
    pool-aware wrapper in :func:`_emit_lane`)."""
    dt, dur, ref, base, seed = event
    _wu(out, dt)
    _wu(out, dur)
    _wu(out, ref + _INSTR_REF_BIAS)
    _write_base(out, base)
    _write_seed(out, seed)


def _event_read(ids, i):
    dt, i = _ru(ids, i)
    dur, i = _ru(ids, i)
    ref, i = _ru(ids, i)
    base, i = _read_base(ids, i)
    seed, i = _read_seed(ids, i)
    return (dt, dur, ref - _INSTR_REF_BIAS, base, seed), i


def _event_delta(a, b):
    """The constant note-table-index shift making event ``b`` a transposed copy of
    ``a`` (same dt / dur / instrument / seed, both simple pitch-factored bodies),
    else ``None``.  A phrase replayed at a different pitch is a constant-base shift.
    """
    if a[0] != b[0] or a[1] != b[1] or a[2] != b[2] or a[4] != b[4]:
        return None
    if isinstance(a[3], list) or isinstance(b[3], list):
        return None  # piecewise body: no single transpose interval
    if a[3] < 0 or b[3] < 0:
        return None  # absolute (off-grid) body: not pitch-transposable
    return b[3] - a[3]


def _event_shift(event, delta):
    dt, dur, ref, base, seed = event
    return (dt, dur, ref, base + delta, seed)


# --- instrument pool (defined ONCE up front, then referenced by index) -------
# The pool is emitted before the lanes (not inline) so the per-lane event LZ is a
# clean stream of fixed-shape literals: a REPEAT/TRANSPOSE copy never has to carry
# an instrument definition, and cost estimation never has to model whether a
# literal also defines an instrument.  Each instrument is a small JSON struct.
def _flatten_pool(pool):
    """The pool's entries flattened to one value-token stream (no LZ)."""
    flat = []
    for entry in pool:
        _write_value(flat, entry)
    return flat


def _lz_tokens(stream, min_copy=_POOL_MIN_COPY):
    """Backward-LZ a flat token ``stream`` into an output token list, escaping a
    copy with the :data:`REPEAT` marker (safe: the value stream never emits a
    token >= REPEAT).  Hash-of-3-grams accelerated greedy match (O(n)); a match is
    taken only when it is strictly shorter than the literals it replaces.  The
    inverse is :func:`_unlz_tokens`."""
    out = []
    n = len(stream)
    table = {}  # 3-gram -> list of start positions (most recent last)
    i = 0
    while i < n:
        best_len, best_off = 0, 0
        if i + 3 <= n:
            key = (stream[i], stream[i + 1], stream[i + 2])
            for pos in reversed(table.get(key, ())):
                off = i - pos
                length = 0
                while i + length < n and stream[pos + length] == stream[i + length]:
                    length += 1
                    if length >= 4095:
                        break
                if length > best_len:
                    best_len, best_off = length, off
                if best_len >= 64:  # good-enough cap keeps the scan bounded
                    break
        cost_copy = 1 + _u_len(best_off) + _u_len(best_len)
        if best_len >= min_copy and cost_copy < best_len:
            out.append(REPEAT)
            _wu(out, best_off)
            _wu(out, best_len)
            step = best_len
        else:
            out.append(stream[i])
            step = 1
        # index every 3-gram start we just passed over
        for j in range(i, min(i + step, n - 2)):
            table.setdefault((stream[j], stream[j + 1], stream[j + 2]), []).append(j)
        i += step
    return out


def _unlz_tokens(ids, i, ntokens):
    """Inverse of :func:`_lz_tokens`: rebuild ``ntokens`` flat value-tokens from
    the LZ'd stream starting at ``ids[i]``.  Returns ``(flat, new_index)``."""
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
    """Emit the instrument pool: its count, its flat-token length, then the
    TOKEN-LZ'd flat stream (so shared sub-tables across instruments collapse)."""
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


# --- lane encode (pool already defined; events are plain LZ literals) --------
def _emit_lane(out, events):
    """Backward-LZ a lane's event stream.  TRANSPOSE factors a transposed phrase
    repeat (a constant note-table-index shift)."""

    def lit_cost(event):
        tmp = []
        _event_lit(tmp, event)
        return len(tmp)

    _wu(out, len(events))
    _lz_emit_t(out, events, lit_cost, _event_lit, _event_delta)


def _read_lane(ids, i):
    n, i = _ru(ids, i)
    events, i = _lz_read_t(ids, i, n, _event_read, _event_shift)
    return events, i


# --- lane ordering: a stable, decode-reproducible key ----------------------
def _lane_order(lanes):
    """Generator lanes (``("g", "V:freq"|"V:pw")``) first in voice order, then
    event lanes (``("e", reg)``) in register order -- a fixed order both sides
    reproduce so the flat stream needs no per-lane id."""

    def key(lane_id):
        kind, val = lane_id
        if kind == "g":
            voice, cls = val.split(":")
            return (0, int(voice), 0 if cls == "freq" else 1)
        return (1, val, 0)

    return sorted(lanes, key=key)


# --- top-level codec -------------------------------------------------------
def _header(out, program):
    _wu(out, program.nframes)
    for b in program.boot:
        _wu(out, b)
    note_table = program.tables.get("note_table")
    if note_table is None:
        _wu(out, 0)
    else:
        _wu(out, 1)
        _wu(out, len(note_table))
        for v in note_table:
            _wu(out, v)


def _read_header(ids):
    i = 0
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


def tracker_program_to_ids(program):
    """Serialize a tracker-lifted generic program to token ids (inverse of
    :func:`tracker_ids_to_program`)."""
    from preframr_tokens.bacc.generic.tracker import lift

    pool, lanes, _ = lift(program)
    out = []
    _header(out, program)
    _emit_pool(out, pool)
    _wu(out, len(_lane_order(lanes)))
    for lane_id in _lane_order(lanes):
        kind, val = lane_id
        out.append(0 if kind == "g" else 1)
        if kind == "g":
            voice, cls = val.split(":")
            _wu(out, int(voice))
            _wu(out, 0 if cls == "freq" else 1)
        else:
            _wu(out, val)
        _emit_lane(out, lanes[lane_id])
    return out


def tracker_ids_to_program(ids):
    """Inverse of :func:`tracker_program_to_ids` -> a tracker-lifted generic
    program whose ``tables`` carry the reconstructed ``genfits``/``eventfits``."""
    from preframr_tokens.bacc.generic.tracker import unlift

    nframes, boot, note_table, i = _read_header(ids)
    pool, i = _read_pool(ids, i)
    nlanes, i = _ru(ids, i)
    lanes = {}
    for _ in range(nlanes):
        kind = ids[i]
        i += 1
        if kind == 0:
            voice, i = _ru(ids, i)
            cls, i = _ru(ids, i)
            lane_id = ("g", f"{voice}:{'freq' if cls == 0 else 'pw'}")
        else:
            reg, i = _ru(ids, i)
            lane_id = ("e", reg)
        events, i = _read_lane(ids, i)
        lanes[lane_id] = events
    genfits, eventfits = unlift(pool, lanes, note_table)
    return BaccProgram(
        driver="generic",
        nframes=nframes,
        boot=boot,
        instruments=[],
        score=[],
        seed={},
        tables={
            "note_table": note_table,
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
    """Return ``({block: tokens}, nframes)`` for the tracker token stream."""
    from preframr_tokens.bacc.generic.tracker import lift

    pool, lanes, _ = lift(program)
    out = []
    _header(out, program)
    header = len(out)
    out = []
    _emit_pool(out, pool)
    instr_def = len(out)
    out = []
    _wu(out, len(_lane_order(lanes)))
    for lane_id in _lane_order(lanes):
        kind, val = lane_id
        out.append(0 if kind == "g" else 1)
        if kind == "g":
            voice, cls = val.split(":")
            _wu(out, int(voice))
            _wu(out, 0 if cls == "freq" else 1)
        else:
            _wu(out, val)
        _emit_lane(out, lanes[lane_id])
    score = len(out)
    brk = {
        "header": header,
        "instr_def": instr_def,
        "score": score,
        "n_instruments": len(pool),
        "total": header + instr_def + score,
    }
    return brk, program.nframes
