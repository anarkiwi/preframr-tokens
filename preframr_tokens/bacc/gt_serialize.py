"""Serialize a GoatTracker BaccProgram to/from the COMMON learnable abstraction.

GoatTracker recovers into the SAME shape as the Hubbard codec: per-voice tracker
ROWS + pitch-invariant instrument GENERATOR definitions + a backward ORDERLIST --
NOT raw .SNG bytes. The reconstructed pygoattracker ``Song`` is decomposed into
semantic fields (orderlist entries, pattern rows, instrument params, the four
generator-parameter tables) and rebuilt into a render-equivalent ``Song`` at
decode; the residual-zero gate is render-equality vs the dump, never .SNG-byte
equality, so the tables/commands become abstract generators + effects.

Token shape (all through the shared base-16 LEB digit alphabet, no new ids):

  header   : nframes, subtune, adparam, optimize_pulse, optimize_realtime,
             boot[25], boot1[25]
  tables   : the four generator-parameter columns (wave/pulse/filter/speed),
             each: length, left[], right[]  -- the PWM/arp/filter-sweep params
  instrs   : count, then per instrument-generator the 9 abstract fields
  patterns : count, then per-pattern row-count headers, then ONE global
             backward-LZ row stream over ALL patterns concatenated (the window
             spans pattern boundaries so a phrase repeated in a later pattern can
             copy from an earlier one); decode re-slices the flat row stream into
             patterns by the row counts. Each literal row is
             (note_token, instr_ref, effect, data) where note_token is the
             canonical 12-TET interval (Part B) for a pitched note, or a small
             marker for REST/KEYOFF/KEYON, and effect is the shared effect vocab
  orderlist: per subtune x channel a backward-LZ entry stream (PlayPattern /
             Repeat / Transpose ops) + restart -- backward-only, no forward decl
"""

from preframr_tokens.bacc.backends.goattracker import make_program_from_song
from preframr_tokens.bacc.pitch import fn_to_grid
from preframr_tokens.bacc.serialize import (
    _lz_emit_t,
    _lz_read_t,
    _ri,
    _ru,
    _wi,
    _wu,
)

NREG = 25

# --- canonical note token alphabet ---------------------------------------
# Pitched notes carry the 12-TET grid INTERVAL (signed). Non-pitched markers
# (REST/KEYOFF/KEYON) ride a small reserved namespace so the note field stays
# a single token; rows distinguish them by a 1-byte "kind" prefix.
_KIND_PITCH = 0
_KIND_REST = 1
_KIND_KEYOFF = 2
_KIND_KEYON = 3
# A playable note byte whose GoatTracker freq-table entry is non-positive (e.g. a
# note below FIRSTNOTE, which the playroutine still sounds as a wrapped/transposed
# tone) has no clean 12-TET grid identity. It rides a literal raw-note escape so
# the row stays byte-exact and lossless -- mirroring the Hubbard aliased-tail-note
# escape -- instead of feeding log2(<=0) into the pitch grid.
_KIND_RAWNOTE = 4


# --- helpers shared with serialize ----------------------------------------
def _lit_bytes(lit, item):
    tmp = []
    lit(tmp, item)
    return tmp


def _lz_emit(out, items, lit, delta_of=None, shift=None):
    """Inline backward-LZ over a list (the shared post-BACC ``_lz_emit_t``),
    emitting literals via ``lit(out, item)``. A copy is REPEAT(offset, length) over
    prior items; passing ``delta_of`` adds TRANSPOSE(offset, length, Delta) for a
    prior run re-coordinated by a constant grid-interval (a transposed phrase
    repeat). ``shift`` is unused on encode (kept symmetric with ``_lz_read``)."""
    del shift  # encode side does not re-coordinate; only decode (_lz_read) does
    _lz_emit_t(out, items, lambda it: len(_lit_bytes(lit, it)), lit, delta_of)


def _lz_read(ids, i, count, rd, shift=None):
    """Inverse of _lz_emit (the shared ``_lz_read_t``): rebuild ``count`` items,
    literals via ``rd(ids, i)``; a TRANSPOSE copy re-coordinates each item via
    ``shift(item, delta)``. Returns (items, new_index)."""
    return _lz_read_t(ids, i, count, rd, shift)


# --- note tokenization (Part B canonical grid) ----------------------------
def _note_token(note_byte):
    """(kind, grid_interval) for a pattern note byte.

    A pitched note resolves through GoatTracker's freq table to a SID Fn, snaps to
    the canonical A440 12-TET grid (Part B), and carries that grid index as a
    signed interval from the grid origin (n=0=A440) -- driver-invariant, and
    position-independent so the row LZ stays sound. REST/KEYOFF/KEYON are
    non-pitched markers (interval unused)."""
    from pygoattracker import constants as c

    if note_byte == c.REST:
        return _KIND_REST, 0
    if note_byte == c.KEYOFF:
        return _KIND_KEYOFF, 0
    if note_byte == c.KEYON:
        return _KIND_KEYON, 0
    idx = note_byte - c.FIRSTNOTE
    fn = c.FREQ_TABLE[idx] if 0 <= idx < len(c.FREQ_TABLE) else 0
    if fn <= 0:
        # No clean freq-table pitch (note below FIRSTNOTE / past the table); keep
        # the raw note byte so the row round-trips byte-exact.
        return _KIND_RAWNOTE, note_byte
    return _KIND_PITCH, fn_to_grid(fn)


def _grid_to_note_byte(grid):
    """Inverse: canonical grid index -> the GoatTracker note byte whose freq-table
    entry snaps to that grid index. The freq table is clean 12-TET so this is a
    bijection on the playable range; built once and cached."""
    return _GRID_NOTE[grid]


def _build_grid_note():
    from pygoattracker import constants as c

    table = {}
    for note in range(c.FIRSTNOTE, c.LASTNOTE + 1):
        fn = c.FREQ_TABLE[note - c.FIRSTNOTE]
        if fn > 0:
            table[fn_to_grid(fn)] = note
    return table


_GRID_NOTE = _build_grid_note()


# --- pattern rows ----------------------------------------------------------
def _row_lit(out, row):
    """One abstract row literal: (note_token, instr_ref, effect, data).

    Pitched notes carry the canonical A440 12-TET grid INTERVAL (signed, relative
    to the grid origin n=0=A440 -- Part B), so the same concert pitch is the same
    token across drivers and the literal is position-independent (the LZ over rows
    stays sound). REST/KEYOFF/KEYON are non-pitched markers. instr/command/data
    are the instr_ref + the shared effect vocabulary (GoatTracker command/data)."""
    note, instr, command, data = row
    kind, interval = _note_token(note)
    _wu(out, kind)
    if kind == _KIND_PITCH:
        _wi(out, interval)
    elif kind == _KIND_RAWNOTE:
        _wu(out, interval)  # the raw note byte (no clean grid pitch)
    _wu(out, instr)
    _wu(out, command)
    _wu(out, data)


def _row_read(ids, i):
    kind, i = _ru(ids, i)
    if kind == _KIND_PITCH:
        interval, i = _ri(ids, i)
        note = _grid_to_note_byte(interval)
    elif kind == _KIND_RAWNOTE:
        note, i = _ru(ids, i)
    else:
        from pygoattracker import constants as c

        note = {_KIND_REST: c.REST, _KIND_KEYOFF: c.KEYOFF, _KIND_KEYON: c.KEYON}[kind]
    instr, i = _ru(ids, i)
    command, i = _ru(ids, i)
    data, i = _ru(ids, i)
    return (note, instr, command, data), i


def _row_delta(a, b):
    """Grid-interval making row ``b`` a transposed copy of ``a`` (same instr /
    command / data, both pitched on the clean freq-table grid), else None. Only
    _KIND_PITCH notes are grid-transposable; REST/KEYOFF/KEYON and raw-note
    escapes have no clean interval (a phrase touching them is not transposed)."""
    if a[1] != b[1] or a[2] != b[2] or a[3] != b[3]:
        return None
    ka, ia = _note_token(a[0])
    kb, ib = _note_token(b[0])
    if ka != _KIND_PITCH or kb != _KIND_PITCH:
        return None
    return ib - ia


def _row_shift(row, delta):
    """Re-coordinate a pitched row by ``delta`` grid steps (lossless; the note's
    freq-table-grid interval shifts, the other fields carry through unchanged)."""
    note, instr, command, data = row
    _, interval = _note_token(note)
    return (_grid_to_note_byte(interval + delta), instr, command, data)


# --- orderlist (backward) --------------------------------------------------
# Orderlist entries serialize as (op, value): op 0 PlayPattern(num),
# 1 Repeat(count), 2 Transpose(semitones, zig-zag signed).
def _emit_orderlist(out, entries, restart):
    _wu(out, len(entries))
    _wu(out, restart)

    def lit(o, ent):
        op, val = ent
        _wu(o, op)
        if op == 2:
            _wi(o, val)
        else:
            _wu(o, val)

    _lz_emit(out, entries, lit)


def _read_orderlist(ids, i):
    n, i = _ru(ids, i)
    restart, i = _ru(ids, i)

    def rd(idl, j):
        op, j = _ru(idl, j)
        if op == 2:
            val, j = _ri(idl, j)
        else:
            val, j = _ru(idl, j)
        return (op, val), j

    entries, i = _lz_read(ids, i, n, rd)
    return entries, restart, i


def _orderlist_entries(ol):
    from pygoattracker.model import PlayPattern, Repeat, Transpose

    out = []
    for e in ol.entries:
        if isinstance(e, PlayPattern):
            out.append((0, e.num))
        elif isinstance(e, Repeat):
            out.append((1, e.count))
        elif isinstance(e, Transpose):
            out.append((2, e.semitones))
        else:  # pragma: no cover - model is a closed union
            raise ValueError(f"unknown orderlist entry {e!r}")
    return out


# --- tables (generator parameter columns) ----------------------------------
def _emit_table(out, table):
    _wu(out, len(table.left))
    for b in table.left:
        _wu(out, b)
    for b in table.right:
        _wu(out, b)


def _read_table(ids, i):
    from pygoattracker.model import Table

    n, i = _ru(ids, i)
    left = []
    for _ in range(n):
        b, i = _ru(ids, i)
        left.append(b)
    right = []
    for _ in range(n):
        b, i = _ru(ids, i)
        right.append(b)
    return Table(left=left, right=right), i


# --- instruments (generators) ----------------------------------------------
_INSTR_FIELDS = (
    "attack_decay",
    "sustain_release",
    "wave_ptr",
    "pulse_ptr",
    "filter_ptr",
    "vibrato_param",
    "vibrato_delay",
    "gateoff_timer",
    "first_wave",
)


def _emit_instrument(out, instr):
    for f in _INSTR_FIELDS:
        _wu(out, getattr(instr, f))


def _read_instrument(ids, i, num):
    from pygoattracker.model import Instrument

    vals = []
    for _ in _INSTR_FIELDS:
        v, i = _ru(ids, i)
        vals.append(v)
    return Instrument(*vals, name=f"i{num:02d}"), i


# --- top-level codec -------------------------------------------------------
def _seed_header(out, program):
    _wu(out, program.nframes)
    _wu(out, int(program.seed["subtune"]))
    _wu(out, int(program.seed["adparam"]))
    _wu(out, int(bool(program.seed["optimize_pulse"])))
    _wu(out, int(bool(program.seed["optimize_realtime"])))
    boot = list(program.boot) or [0] * NREG
    boot1 = program.tables.get("boot1") or [0] * NREG
    for b in boot:
        _wu(out, b)
    for b in boot1:
        _wu(out, b)


def _song_of(program):
    """The recovered abstract ``Song``. The serializer only ever sees the
    abstraction -- there is no raw-.SNG-bytes path to fall back to."""
    return program.tables["song"]


def gt_program_to_ids(program):
    song = _song_of(program)
    out = []
    _seed_header(out, program)
    # tables (generator parameter columns), in on-disk order
    for table in (
        song.wavetable,
        song.pulsetable,
        song.filtertable,
        song.speedtable,
    ):
        _emit_table(out, table)
    # instrument-generators
    _wu(out, len(song.instruments))
    for instr in song.instruments:
        _emit_instrument(out, instr)
    # patterns (abstract rows): per-pattern row-count headers, then ONE global
    # backward-LZ row stream over all patterns concatenated. The single window
    # spans pattern boundaries (a phrase repeated in a later pattern copies from
    # an earlier one); decode re-slices the flat stream back by the row counts.
    _wu(out, len(song.patterns))
    flat = []
    for pat in song.patterns:
        rows = [(r.note, r.instrument, r.command, r.data) for r in pat.rows]
        _wu(out, len(rows))
        flat.extend(rows)
    # TRANSPOSE-aware row LZ (shared post-BACC path): a phrase repeated at a
    # different pitch (same instr/command/data) factors as one TRANSPOSE+Delta
    # over the canonical grid, not a fresh literal run.
    _lz_emit(out, flat, _row_lit, _row_delta)
    # orderlists (backward), per subtune x channel
    _wu(out, len(song.subtunes))
    for sub in song.subtunes:
        for ol in sub.channels:
            _emit_orderlist(out, _orderlist_entries(ol), ol.restart)
    return out


def _rows_to_pattern(rows):
    from pygoattracker.model import Pattern, Row

    return Pattern(rows=[Row(n, ins, cmd, dat) for (n, ins, cmd, dat) in rows])


def _entries_to_orderlist(entries, restart):
    from pygoattracker.model import Orderlist, PlayPattern, Repeat, Transpose

    out = []
    for op, val in entries:
        if op == 0:
            out.append(PlayPattern(val))
        elif op == 1:
            out.append(Repeat(val))
        else:
            out.append(Transpose(val))
    return Orderlist(entries=out, restart=restart)


def gt_ids_to_program(ids):
    from pygoattracker.model import Song, Subtune

    i = 0
    nframes, i = _ru(ids, i)
    subtune, i = _ru(ids, i)
    adparam, i = _ru(ids, i)
    optimize_pulse, i = _ru(ids, i)
    optimize_realtime, i = _ru(ids, i)
    boot = []
    for _ in range(NREG):
        b, i = _ru(ids, i)
        boot.append(b)
    boot1 = []
    for _ in range(NREG):
        b, i = _ru(ids, i)
        boot1.append(b)
    wavetable, i = _read_table(ids, i)
    pulsetable, i = _read_table(ids, i)
    filtertable, i = _read_table(ids, i)
    speedtable, i = _read_table(ids, i)
    n_instr, i = _ru(ids, i)
    instruments = []
    for k in range(n_instr):
        instr, i = _read_instrument(ids, i, k + 1)
        instruments.append(instr)
    n_pat, i = _ru(ids, i)
    counts = []
    for _ in range(n_pat):
        c, i = _ru(ids, i)
        counts.append(c)
    flat, i = _lz_read(ids, i, sum(counts), _row_read, _row_shift)
    patterns = []
    off = 0
    for c in counts:
        patterns.append(_rows_to_pattern(flat[off : off + c]))
        off += c
    n_sub, i = _ru(ids, i)
    subtunes = []
    for _ in range(n_sub):
        channels = []
        for _ in range(3):
            entries, restart, i = _read_orderlist(ids, i)
            channels.append(_entries_to_orderlist(entries, restart))
        subtunes.append(Subtune(channels=channels))
    song = Song(
        name="",
        subtunes=subtunes,
        instruments=instruments,
        patterns=patterns,
        wavetable=wavetable,
        pulsetable=pulsetable,
        filtertable=filtertable,
        speedtable=speedtable,
    )
    seed = {
        "subtune": subtune,
        "adparam": adparam,
        "optimize_pulse": optimize_pulse,
        "optimize_realtime": optimize_realtime,
    }
    program = make_program_from_song(song, seed, nframes)
    program.boot = boot
    program.tables["boot1"] = boot1
    return program


def gt_measure(program):
    song = _song_of(program)
    out = []
    _seed_header(out, program)
    header = len(out)
    out = []
    for table in (song.wavetable, song.pulsetable, song.filtertable, song.speedtable):
        _emit_table(out, table)
    tables = len(out)
    out = []
    _wu(out, len(song.instruments))
    for instr in song.instruments:
        _emit_instrument(out, instr)
    instr_def = len(out)
    out = []
    _wu(out, len(song.patterns))
    flat = []
    for pat in song.patterns:
        rows = [(r.note, r.instrument, r.command, r.data) for r in pat.rows]
        _wu(out, len(rows))
        flat.extend(rows)
    _lz_emit(out, flat, _row_lit, _row_delta)
    score = len(out)  # the dominant block: TRANSPOSE-aware pattern-row LZ
    out = []
    _wu(out, len(song.subtunes))
    for sub in song.subtunes:
        for ol in sub.channels:
            _emit_orderlist(out, _orderlist_entries(ol), ol.restart)
    orders = len(out)
    total = len(gt_program_to_ids(program))
    brk = {
        "header": header,
        "tables": tables,
        "instr_def": instr_def,
        "score": score,
        "orders": orders,
        "total": total,
    }
    return brk, program.nframes
