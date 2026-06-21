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
# It packs a 1-bit escape into the LEB LSB: bit0=0 -> a zig-zag canonical grid
# index in the upper bits; bit0=1 -> a literal note-table INDEX (for rare
# aliased tail notes outside the clean ET run, so the re-coordinate stays
# byte-exact and lossless). Position-independent, so the row LZ/REPEAT is sound.
def _zz(n):
    n = int(n)
    return (n << 1) ^ (n >> 63)


def _unzz(z):
    return (z >> 1) ^ -(z & 1)


def _note_field(out, note, static_img, grid_of):
    """Emit the canonical note token for a Hubbard note-table index."""
    g = grid_of.get(note)
    if g is not None and hubbard_table_fn(static_img, note) > 0:
        _wu(out, _zz(g) << 1)  # bit0=0: canonical absolute grid index
    else:
        _wu(out, (note << 1) | 1)  # bit0=1: literal index escape (aliased tail)


def _read_note_field(ids, i, index_of):
    """Inverse of _note_field -> the exact Hubbard note-table index."""
    z, i = _ru(ids, i)
    if z & 1:
        return z >> 1, i  # literal index escape
    return index_of[_unzz(z >> 1)], i  # canonical grid index -> note index


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
    _note_field(out, row[1], static_img, grid_of)
    _wu(out, row[2])
    _wu(out, row[3])
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


def _transposed_run(rows, i, off, delta, grid_of):
    """Length of the backward run at ``i`` (source ``i-off``) whose notes are all
    the prior run's notes shifted by exactly ``delta`` (non-note fields identical).
    The source rows must themselves be grid-resolvable so decode can add delta."""
    n = 0
    while i + n < len(rows):
        src, dst = rows[i - off + n], rows[i + n]
        if _grid_delta(src, dst, grid_of) != delta:
            break
        n += 1
    return n


def _best_transpose(rows, i, grid_of):
    """Best (length, offset, delta) backward run that matches a prior run up to a
    single constant non-zero grid-interval (a transposed phrase repeat)."""
    best_len, best_off, best_delta = 0, 0, 0
    for off in range(1, i + 1):
        delta = _grid_delta(rows[i - off], rows[i], grid_of)
        if delta in (None, 0):
            continue  # delta==0 is the exact REPEAT case, handled separately
        n = _transposed_run(rows, i, off, delta, grid_of)
        if n > best_len:
            best_len, best_off, best_delta = n, off, delta
    return best_len, best_off, best_delta


def _emit_rows(out, rows, seen, instruments, static_img, grid_of):
    i = 0
    while i < len(rows):
        best_len, best_off = 0, 0
        for off in range(1, i + 1):
            n = 0
            while i + n < len(rows) and rows[i - off + n] == rows[i + n]:
                n += 1
            if n > best_len:
                best_len, best_off = n, off
        cost_copy = 1 + _u_len(best_off) + _u_len(best_len)
        tlen, toff, tdelta = _best_transpose(rows, i, grid_of)
        cost_trans = 1 + _u_len(toff) + _u_len(tlen) + _wi_len(tdelta)
        lit_copy = sum(
            _lit_cost(rows[i + j], static_img, grid_of) for j in range(best_len)
        )
        lit_trans = sum(
            _lit_cost(rows[i + j], static_img, grid_of) for j in range(tlen)
        )
        use_copy = best_len >= _MIN_COPY and cost_copy < lit_copy
        use_trans = tlen >= _MIN_COPY and cost_trans < lit_trans
        # Cheapest of {exact REPEAT, transposed REPEAT+Delta, literal}, weighing
        # each candidate by tokens-saved-per-row so a longer plain REPEAT is not
        # beaten by a shorter transposed one (and vice versa).
        copy_gain = (lit_copy - cost_copy) if use_copy else 0
        trans_gain = (lit_trans - cost_trans) if use_trans else 0
        if use_trans and trans_gain >= copy_gain:
            out.append(TRANSPOSE)
            _wu(out, toff)
            _wu(out, tlen)
            _wi(out, tdelta)
            i += tlen
        elif use_copy:
            out.append(REPEAT)
            _wu(out, best_off)
            _wu(out, best_len)
            i += best_len
        else:
            r = rows[i]
            _wu(out, r[0])
            _note_field(out, r[1], static_img, grid_of)
            _wu(out, r[2])
            if r[2] not in seen:
                seen.add(r[2])
                for b in instruments[r[2]]:
                    _wu(out, b)
            _wu(out, r[3])
            _wu(out, r[4])
            i += 1


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
    for v in range(3):
        nrows, i = _ru(ids, i)
        rows = []
        while len(rows) < nrows:
            if ids[i] == REPEAT:
                i += 1
                off, i = _ru(ids, i)
                length, i = _ru(ids, i)
                base = len(rows)
                for j in range(length):
                    rows.append(rows[base - off + j])
            elif ids[i] == TRANSPOSE:
                # Backward REPEAT re-coordinated by a constant grid-interval: copy
                # the prior run, then shift each copied note by delta on the grid
                # (lossless re-coordinate; non-note fields carry through unchanged).
                i += 1
                off, i = _ru(ids, i)
                length, i = _ru(ids, i)
                delta, i = _ri(ids, i)
                base = len(rows)
                for j in range(length):
                    dt, note, instr, lnth, porta = rows[base - off + j]
                    note = index_of[grid_of[note] + delta]
                    rows.append((dt, note, instr, lnth, porta))
            else:
                dt, i = _ru(ids, i)
                note, i = _read_note_field(ids, i, index_of)
                instr, i = _ru(ids, i)
                if instr not in seen:
                    seen.add(instr)
                    row = []
                    for _ in range(8):
                        b, i = _ru(ids, i)
                        row.append(b)
                    instruments[instr] = row
                lnth, i = _ru(ids, i)
                porta, i = _ru(ids, i)
                rows.append((dt, note, instr, lnth, porta))
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
