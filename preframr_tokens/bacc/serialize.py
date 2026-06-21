"""Serialize a BaccProgram to an inline backward-LZ token-id stream.

The model-facing form: a startup seed (boot frame + per-voice initial state),
then per-voice score blocks. Each note-on is a literal (dt, note-token,
instrument, length, porta); a repeated phrase is an inline backward REPEAT
(offset, length) over prior note-ons of that voice. The note-token is the
ABSOLUTE canonical A440 12-TET grid index (Part B) -- IDENTICAL to the
GoatTracker token for the same concert pitch -- not a driver-table delta.
Instruments are defined inline on first use (no forward table). Round-trips
byte-exact to the program.
"""

from preframr_tokens.bacc.pitch import hubbard_grid_bijection, hubbard_table_fn
from preframr_tokens.bacc.primitive import BaccProgram, NoteOn

# vocab: base-16 LEB digits 0..31 (0-15 continue, 16-31 terminal) + REPEAT marker
# + TRANSPOSE marker (a backward REPEAT whose copied notes are re-coordinated by a
# constant grid-interval Delta -- factors a phrase repeated at a different pitch,
# exactly what a tracker orderlist's Transpose(semitones) does, while the note
# token stays the ABSOLUTE canonical A440 grid index).
REPEAT = 32
TRANSPOSE = 33
VOCAB = 34
PAD_ID = VOCAB  # reserved padding id above the codec alphabet

_SEED_KEYS = (
    "notenum",
    "instrnr",
    "lnthcc",
    "lenleft",
    "sfl",
    "sfh",
    "porta",
    "vctrl",
    "pdly",
    "pdir",
)
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


# --- canonical note token (Part B, absolute A440 12-TET grid) -------------
# The Hubbard note field is the ABSOLUTE canonical grid index -- IDENTICAL to
# the GoatTracker token for the same concert pitch -- not a driver-table delta.
# It packs two flag bits into the LEB low bits: bit0=0 -> a zig-zag canonical
# grid index in the upper bits; bit0=1 -> a literal note-table INDEX (for rare
# aliased tail notes outside the clean ET run, so the re-coordinate stays
# byte-exact and lossless). bit1 is a porta-present flag: porta is the SAME value
# (0) on >98% of note-ons, so rather than always spend a LEB token on it we fold
# "this row carries a non-zero porta" into the note token here and only emit the
# porta field when the flag is set -- a free fold, since the grid index has slack
# in its low LEB digit. Position-independent (a pure function of the row's note
# and porta), so the row LZ/REPEAT/TRANSPOSE matching stays sound.
def _zz(n):
    n = int(n)
    return (n << 1) ^ (n >> 63)


def _unzz(z):
    return (z >> 1) ^ -(z & 1)


def _note_field(out, note, porta, static_img, grid_of):
    """Emit the canonical note token (bit0=escape, bit1=porta-present)."""
    pf = 2 if porta else 0
    g = grid_of.get(note)
    if g is not None and hubbard_table_fn(static_img, note) > 0:
        _wu(out, (_zz(g) << 2) | pf)  # bit0=0: canonical absolute grid index
    else:
        _wu(out, (note << 2) | pf | 1)  # bit0=1: literal index escape (alias)


def _read_note_field(ids, i, index_of):
    """Inverse of _note_field -> (note index, porta-present flag, i)."""
    z, i = _ru(ids, i)
    has_porta = bool(z & 2)
    if z & 1:
        return z >> 2, has_porta, i  # literal index escape
    return index_of[_unzz(z >> 2)], has_porta, i  # grid index -> note index


def _voice_rows(program):
    static_img = program.tables["static_img"]
    _, index_to_grid = hubbard_grid_bijection(static_img)
    rows = {0: [], 1: [], 2: []}
    prevf = [0, 0, 0]
    for ev in program.score:
        v = ev.voice
        rows[v].append((ev.frame - prevf[v], ev.note, ev.instr, ev.lnth, ev.porta))
        prevf[v] = ev.frame
    return rows, static_img, index_to_grid


def _lit_cost(row, static_img, grid_of):
    out = []
    _wu(out, row[0])
    _note_field(out, row[1], row[4], static_img, grid_of)
    _wu(out, row[2])
    _wu(out, row[3])
    if row[4]:  # porta folded into the note token; emit only when non-zero
        _wu(out, row[4])
    return len(out)


def _grid_delta(a, b, grid_of):
    """Grid-interval Delta(b) - Delta(a) if both rows match up to a constant
    pitch shift (every non-note field identical, both notes grid-resolvable),
    else None. The note token is the absolute canonical grid index, so a phrase
    repeated transposed is a constant-Delta shift of the prior run's notes."""
    if a[0] != b[0] or a[2] != b[2] or a[3] != b[3] or a[4] != b[4]:
        return None
    ga, gb = grid_of.get(a[1]), grid_of.get(b[1])
    if ga is None or gb is None:
        return None  # aliased-tail (literal-escape) note: not grid-transposable
    return gb - ga


# --- shared backward-LZ with TRANSPOSE factoring (post-BACC, driver-agnostic) ---
# The dominant score block of EVERY driver is a list of comparable items (Hubbard
# per-voice note-on rows, GoatTracker pattern rows, DMC pattern tokens). The good
# compression -- inline backward REPEAT for an exact phrase repeat plus TRANSPOSE
# (a backward REPEAT re-coordinated by a constant pitch Delta on the canonical
# A440 grid, exactly what a tracker orderlist's Transpose(semitones) does) -- is
# the SAME machinery for all three; only the per-item literal and the
# transpose-delta test are driver-specific. Factoring it here (keyed off the
# common score-item list) is what lets GoatTracker and DMC scores, not just
# Hubbard's, get REPEAT/TRANSPOSE phrase factoring + the canonical note grid.
#
# ``delta_of(a, b)`` returns the constant grid-interval that makes ``b`` a
# transposed copy of ``a`` (every non-pitch field identical, both notes
# grid-resolvable), or ``None`` if ``b`` is not a constant-pitch shift of ``a``.
# A driver that has no transposable pitch axis passes ``delta_of=None`` and gets
# plain REPEAT-only LZ (byte-identical to the old per-driver ``_lz_emit``).
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


def _emit_rows(out, rows, seen, instruments, static_img, grid_of):
    """Hubbard per-voice note-on rows through the shared transposed-LZ. The literal
    emits (dt, canonical note token, instr [+ inline instr-def on first use],
    length [, porta]); the delta test is the row grid-interval."""

    def lit_emit(o, r):
        _wu(o, r[0])
        _note_field(o, r[1], r[4], static_img, grid_of)
        _wu(o, r[2])
        if r[2] not in seen:
            seen.add(r[2])
            for b in instruments[r[2]]:
                _wu(o, b)
        _wu(o, r[3])
        if r[4]:  # porta-present flag lives in the note token (see _note_field)
            _wu(o, r[4])

    _lz_emit_t(
        out,
        rows,
        lambda r: _lit_cost(r, static_img, grid_of),
        lit_emit,
        lambda a, b: _grid_delta(a, b, grid_of),
    )


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
    if program.driver == "lft":
        from preframr_tokens.bacc.lft_serialize import lft_program_to_ids

        return lft_program_to_ids(program)
    if program.driver == "dmc":
        from preframr_tokens.bacc.dmc_serialize import dmc_program_to_ids

        return dmc_program_to_ids(program)
    if program.driver == "generic":
        from preframr_tokens.bacc.generic_serialize import generic_program_to_ids

        return generic_program_to_ids(program)
    out = []
    _wu(out, program.nframes)
    for b in program.boot:
        _wu(out, b)
    for b in program.tables["static_img"]:
        _wu(out, b)
    for k in _SEED_KEYS:
        for x in program.seed[k]:
            _wu(out, x)
    _wu(out, program.seed["init_speed"])
    _wu(out, program.seed["resetspd"])
    rows, static_img, grid_to_index = _voice_rows(program)
    seen = set()
    for v in range(3):
        _wu(out, len(rows[v]))
        _emit_rows(out, rows[v], seen, program.instruments, static_img, grid_to_index)
    return out


def ids_to_program(ids, driver="hubbard_monty"):
    """Inverse of program_to_ids -> BaccProgram (instruments tables reconstructed)."""
    if driver == "goattracker":
        from preframr_tokens.bacc.gt_serialize import gt_ids_to_program

        return gt_ids_to_program(ids)
    if driver == "lft":
        from preframr_tokens.bacc.lft_serialize import lft_ids_to_program

        return lft_ids_to_program(ids)
    if driver == "dmc":
        from preframr_tokens.bacc.dmc_serialize import dmc_ids_to_program

        return dmc_ids_to_program(ids)
    if driver == "generic":
        from preframr_tokens.bacc.generic_serialize import generic_ids_to_program

        return generic_ids_to_program(ids)
    i = 0
    nframes, i = _ru(ids, i)
    boot = []
    for _ in range(25):
        b, i = _ru(ids, i)
        boot.append(b)
    static_img = []
    for _ in range(256):
        b, i = _ru(ids, i)
        static_img.append(b)
    seed = {}
    for k in _SEED_KEYS:
        vals = []
        for _ in range(3):
            x, i = _ru(ids, i)
            vals.append(x)
        seed[k] = vals
    seed["init_speed"], i = _ru(ids, i)
    seed["resetspd"], i = _ru(ids, i)
    index_of, grid_of = hubbard_grid_bijection(static_img)
    instruments = [[0] * 8 for _ in range(64)]
    seen = set()
    score = []

    def lit_read(idl, j):
        dt, j = _ru(idl, j)
        note, has_porta, j = _read_note_field(idl, j, index_of)
        instr, j = _ru(idl, j)
        if instr not in seen:
            seen.add(instr)
            row = []
            for _ in range(8):
                b, j = _ru(idl, j)
                row.append(b)
            instruments[instr] = row
        lnth, j = _ru(idl, j)
        if has_porta:  # porta field present only when its note-token flag set
            porta, j = _ru(idl, j)
        else:
            porta = 0
        return (dt, note, instr, lnth, porta), j

    def shift(r, delta):
        # lossless grid re-coordinate; non-note fields carry through unchanged
        dt, note, instr, lnth, porta = r
        return (dt, index_of[grid_of[note] + delta], instr, lnth, porta)

    for v in range(3):
        nrows, i = _ru(ids, i)
        rows, i = _lz_read_t(ids, i, nrows, lit_read, shift)
        prevf = 0
        for dt, note, instr, lnth, porta in rows:
            prevf += dt
            score.append(NoteOn(prevf, v, note, instr, lnth, porta))
    score.sort(key=lambda e: (e.frame, e.voice))
    return BaccProgram(
        driver, nframes, boot, instruments, score, seed, {"static_img": static_img}
    )


def measure(program):
    """Return ({block: tokens}, nframes) for the serialized program."""
    if program.driver == "goattracker":
        from preframr_tokens.bacc.gt_serialize import gt_measure

        return gt_measure(program)
    if program.driver == "lft":
        from preframr_tokens.bacc.lft_serialize import lft_measure

        return lft_measure(program)
    if program.driver == "dmc":
        from preframr_tokens.bacc.dmc_serialize import dmc_measure

        return dmc_measure(program)
    if program.driver == "generic":
        from preframr_tokens.bacc.generic_serialize import generic_measure

        return generic_measure(program)
    ids = program_to_ids(program)
    used = sorted({ev.instr for ev in program.score})
    instr_def = sum(_u_len(b) for i in used for b in program.instruments[i])
    seed = sum(_u_len(x) for k in _SEED_KEYS for x in program.seed[k])
    seed += _u_len(program.seed["init_speed"]) + _u_len(program.seed["resetspd"])
    boot = sum(_u_len(b) for b in program.boot)
    table = sum(_u_len(b) for b in program.tables["static_img"])
    score = len(ids) - instr_def - seed - boot - table - _u_len(program.nframes)
    brk = {
        "score": score,
        "instr_def": instr_def,
        "seed": seed,
        "boot": boot,
        "table": table,
        "total": len(ids),
    }
    return brk, program.nframes
