"""Flat, learnability-first token alphabet (v2) -- replaces the base-16 LEB +
inline-LZ scheme.

Design goal: MAXIMISE learnability for a decoder-only LM, trading away the
compression of the v1 alphabet (no place-value arithmetic, no numeric back-offset
copies). Three principles:

  1. TYPED ATOMS.  A token id's RANGE encodes its kind, so the model never has to
     decode place value or infer type from grammar position:
       - NOTE_*      one token per canonical A440 grid index (pitch proximity ->
                     embedding proximity); plus REST/KEYOFF/KEYON/RAW markers.
       - INSTR_REF_* one token per instrument slot.
       - CMD_*       one token per effect/command.
       - BYTE_*      one token per 0..255 byte value (data, params, regs); 16-bit
                     values are a fixed-width (lo, hi) BYTE pair -- positional, not
                     arithmetic.
  2. SELF-DELIMITING STRUCTURE.  Every container is bracketed by BEGIN/END / row
     separators instead of a length prefix, so "emit until END" replaces counting
     (the Dyck-language weakness) and a stray token re-anchors at the next
     delimiter instead of desyncing the stream.
  3. NO LZ.  Repetition is left to the model's attention (and, at the musical
     level, to the orderlist's pattern references) rather than emitted as
     REPEAT/TRANSPOSE numeric back-offsets the model cannot compute reliably.

Losslessness is unchanged: this is a pure re-serialization of the same recovered
``BaccProgram``; the gate stays render-equality (``verify_residual``) plus the
token round-trip (ids -> program -> render byte-exact).
"""

from preframr_tokens.bacc.gt_common import (
    _INSTR_FIELDS,
    _KIND_KEYOFF,
    _KIND_KEYON,
    _KIND_PITCH,
    _KIND_RAWNOTE,
    _KIND_REST,
    _grid_to_note_byte,
    _note_token,
    _orderlist_entries,
    make_program_from_song,
)

NREG = 25

# --- token alphabet (typed, contiguous ranges) ----------------------------
# Structural / control block widened to 0..63 (was 0..31) so the GEN_* generator
# enum, ORDER_CALL (one-level pattern call), the inline REF markers and TEMPO fit
# with headroom; the typed value ranges shift up once (one VOCAB-shape bump). See
# FLAT_VOCAB_MIGRATION.md Phase 1 / RETRACKERS_COMPAT_REVIEW.md Axis 2-6.
BOS = 0
EOS = 1
SEC_HEADER = 2
SEC_TABLES = 3
SEC_INSTRUMENTS = 4
SEC_PATTERNS = 5
SEC_ORDERLISTS = 6
TABLE_BEGIN = 7
TABLE_END = 8
INSTR_BEGIN = 9
INSTR_END = 10
PATTERN_BEGIN = 11
PATTERN_END = 12
ROW = 13
SUBTUNE_BEGIN = 14
SUBTUNE_END = 15
CHANNEL = 16
NOTE_REST = 17
NOTE_KEYOFF = 18
NOTE_KEYON = 19
NOTE_RAW = 20  # followed by one BYTE: a note byte with no clean 12-TET pitch
ORDER_PLAY = 21  # followed by one BYTE: pattern number
ORDER_REPEAT = 22  # followed by one BYTE: repeat count
ORDER_TRANSPOSE = 23  # followed by one BYTE: semitones (two's-complement int8)
ORDER_CALL = 24  # followed by one BYTE: one-level pattern call (Axis 3, JCH)
SEC_NOTE_TABLE = 25  # generic value->grid note table section (Phase 2/3a)
SEC_PROGRAMS = 26  # generic shared-program / accumulator section (Phase 3c)
REF = 27  # followed by one BYTE: content-addressed pattern ref (inline, Sec 2e)
# The generator-kind enum (Axis 5 + PW/filter). Each GEN_* is one structural token
# followed by its fixed param slot (BYTE/NOTE-offset atoms), defined once per
# instrument and expanded by the renderer -- never stored per-frame. See the table
# in FLAT_VOCAB_MIGRATION.md Sec 2a.
GEN_HOLD = 28  # constant pitch (zero-accumulator)
GEN_RAMP = 29  # porta/slide, PW-sweep, filter-sweep
GEN_QUAD = 30  # accelerating porta
GEN_VIBRATO = 31  # vibrato triangle
GEN_ARP = 32  # arp (note-index offset walk)
GEN_TABLEWALK = 33  # $FF/$7F-loop wave/arp sub-table
ACC_BEGIN = 34
ACC_END = 35
LANE_BEGIN = 36
LANE_END = 37
SEG = 38  # generic accumulator/lane segment separator
# 39..63 reserved for future structural tokens

GEN_KINDS = (
    GEN_HOLD,
    GEN_RAMP,
    GEN_QUAD,
    GEN_VIBRATO,
    GEN_ARP,
    GEN_TABLEWALK,
)

# typed value ranges (shifted up once for the widened structural block)
NOTE_BASE = 64
NOTE_SPAN = 160
NOTE_ZERO = NOTE_BASE + 80  # grid index g (A440, n=0=A4) -> NOTE_ZERO + g == 144
NOTE_MIN = -80
NOTE_MAX = 79

INSTR_REF_BASE = NOTE_BASE + NOTE_SPAN  # 224
INSTR_REF_SPAN = 64

CMD_BASE = INSTR_REF_BASE + INSTR_REF_SPAN  # 288
CMD_SPAN = 32

BYTE_BASE = CMD_BASE + CMD_SPAN  # 320
BYTE_SPAN = 256

VOCAB = BYTE_BASE + BYTE_SPAN  # 576
PAD_ID = VOCAB  # reserved padding id, above the codec alphabet


# --- low-level emit/read helpers ------------------------------------------
def _byte(out, v):
    v = int(v)
    if not 0 <= v <= 255:
        raise ValueError(f"byte out of range: {v}")
    out.append(BYTE_BASE + v)


def _read_byte(ids, i):
    t = ids[i]
    if not BYTE_BASE <= t < BYTE_BASE + BYTE_SPAN:
        raise ValueError(f"expected BYTE token at {i}, got {t}")
    return t - BYTE_BASE, i + 1


def _u16(out, v):
    v = int(v)
    if not 0 <= v <= 0xFFFF:
        raise ValueError(f"u16 out of range: {v}")
    _byte(out, v & 0xFF)
    _byte(out, (v >> 8) & 0xFF)


def _read_u16(ids, i):
    lo, i = _read_byte(ids, i)
    hi, i = _read_byte(ids, i)
    return lo | (hi << 8), i


def _expect(ids, i, tok, name):
    if ids[i] != tok:
        raise ValueError(f"expected {name} ({tok}) at {i}, got {ids[i]}")
    return i + 1


# --- note field -----------------------------------------------------------
def _emit_note(out, note_byte):
    kind, interval = _note_token(note_byte)
    if kind == _KIND_PITCH:
        if not NOTE_MIN <= interval <= NOTE_MAX:
            raise ValueError(f"grid index {interval} outside NOTE range")
        out.append(NOTE_ZERO + interval)
    elif kind == _KIND_REST:
        out.append(NOTE_REST)
    elif kind == _KIND_KEYOFF:
        out.append(NOTE_KEYOFF)
    elif kind == _KIND_KEYON:
        out.append(NOTE_KEYON)
    elif kind == _KIND_RAWNOTE:
        out.append(NOTE_RAW)
        _byte(out, interval)  # the raw note byte


def _read_note(ids, i):
    from pygoattracker import constants as c

    t = ids[i]
    if NOTE_BASE <= t < NOTE_BASE + NOTE_SPAN:
        return _grid_to_note_byte(t - NOTE_ZERO), i + 1
    if t == NOTE_REST:
        return c.REST, i + 1
    if t == NOTE_KEYOFF:
        return c.KEYOFF, i + 1
    if t == NOTE_KEYON:
        return c.KEYON, i + 1
    if t == NOTE_RAW:
        nb, i = _read_byte(ids, i + 1)
        return nb, i
    raise ValueError(f"expected note token at {i}, got {t}")


# --- GoatTracker program <-> flat ids -------------------------------------
def _song_of(program):
    return program.tables["song"]


def _emit_header(out, program):
    out.append(SEC_HEADER)
    _u16(out, program.nframes)
    _byte(out, int(program.seed["subtune"]))
    _u16(out, int(program.seed["adparam"]))
    _byte(out, int(bool(program.seed["optimize_pulse"])))
    _byte(out, int(bool(program.seed["optimize_realtime"])))
    boot = list(program.boot) or [0] * NREG
    boot1 = program.tables.get("boot1") or [0] * NREG
    for b in boot:
        _byte(out, b)
    for b in boot1:
        _byte(out, b)


def _emit_tables(out, song):
    out.append(SEC_TABLES)
    for table in (song.wavetable, song.pulsetable, song.filtertable, song.speedtable):
        out.append(TABLE_BEGIN)
        for left, right in zip(table.left, table.right):
            _byte(out, left)
            _byte(out, right)
        out.append(TABLE_END)


def _emit_instruments(out, song):
    out.append(SEC_INSTRUMENTS)
    for instr in song.instruments:
        out.append(INSTR_BEGIN)
        for f in _INSTR_FIELDS:
            _byte(out, getattr(instr, f))
        out.append(INSTR_END)


def _emit_patterns(out, song):
    out.append(SEC_PATTERNS)
    for pat in song.patterns:
        out.append(PATTERN_BEGIN)
        for r in pat.rows:
            out.append(ROW)
            _emit_note(out, r.note)
            if not 0 <= r.instrument < INSTR_REF_SPAN:
                raise ValueError(f"instr ref {r.instrument} outside range")
            out.append(INSTR_REF_BASE + r.instrument)
            if not 0 <= r.command < CMD_SPAN:
                raise ValueError(f"command {r.command} outside range")
            out.append(CMD_BASE + r.command)
            _byte(out, r.data)
        out.append(PATTERN_END)


def _emit_orderlists(out, song):
    out.append(SEC_ORDERLISTS)
    for sub in song.subtunes:
        out.append(SUBTUNE_BEGIN)
        for ol in sub.channels:
            out.append(CHANNEL)
            _byte(out, ol.restart)
            for op, val in _orderlist_entries(ol):
                if op == 0:
                    out.append(ORDER_PLAY)
                    _byte(out, val)
                elif op == 1:
                    out.append(ORDER_REPEAT)
                    _byte(out, val)
                else:  # op == 2, transpose semitones (signed, int8)
                    out.append(ORDER_TRANSPOSE)
                    _byte(out, int(val) & 0xFF)
        out.append(SUBTUNE_END)


def flat_gt_program_to_ids(program):
    song = _song_of(program)
    out = [BOS]
    _emit_header(out, program)
    _emit_tables(out, song)
    _emit_instruments(out, song)
    _emit_patterns(out, song)
    _emit_orderlists(out, song)
    out.append(EOS)
    return out


# --- decode ---------------------------------------------------------------
def _read_header(ids, i):
    i = _expect(ids, i, SEC_HEADER, "SEC_HEADER")
    nframes, i = _read_u16(ids, i)
    subtune, i = _read_byte(ids, i)
    adparam, i = _read_u16(ids, i)
    optimize_pulse, i = _read_byte(ids, i)
    optimize_realtime, i = _read_byte(ids, i)
    boot = []
    for _ in range(NREG):
        b, i = _read_byte(ids, i)
        boot.append(b)
    boot1 = []
    for _ in range(NREG):
        b, i = _read_byte(ids, i)
        boot1.append(b)
    seed = {
        "subtune": subtune,
        "adparam": adparam,
        "optimize_pulse": optimize_pulse,
        "optimize_realtime": optimize_realtime,
    }
    return nframes, seed, boot, boot1, i


def _read_tables(ids, i):
    from pygoattracker.model import Table

    i = _expect(ids, i, SEC_TABLES, "SEC_TABLES")
    tables = []
    for _ in range(4):
        i = _expect(ids, i, TABLE_BEGIN, "TABLE_BEGIN")
        left, right = [], []
        while ids[i] != TABLE_END:
            l, i = _read_byte(ids, i)
            r, i = _read_byte(ids, i)
            left.append(l)
            right.append(r)
        i += 1  # TABLE_END
        tables.append(Table(left=left, right=right))
    return tables, i


def _read_instruments(ids, i):
    from pygoattracker.model import Instrument

    i = _expect(ids, i, SEC_INSTRUMENTS, "SEC_INSTRUMENTS")
    instruments = []
    while ids[i] == INSTR_BEGIN:
        i += 1
        vals = []
        for _ in _INSTR_FIELDS:
            v, i = _read_byte(ids, i)
            vals.append(v)
        i = _expect(ids, i, INSTR_END, "INSTR_END")
        instruments.append(Instrument(*vals, name=f"i{len(instruments):02d}"))
    return instruments, i


def _read_patterns(ids, i):
    from pygoattracker.model import Pattern, Row

    i = _expect(ids, i, SEC_PATTERNS, "SEC_PATTERNS")
    patterns = []
    while ids[i] == PATTERN_BEGIN:
        i += 1
        rows = []
        while ids[i] == ROW:
            i += 1
            note, i = _read_note(ids, i)
            tr = ids[i]
            if not INSTR_REF_BASE <= tr < INSTR_REF_BASE + INSTR_REF_SPAN:
                raise ValueError(f"expected INSTR_REF at {i}, got {tr}")
            instr = tr - INSTR_REF_BASE
            i += 1
            tc = ids[i]
            if not CMD_BASE <= tc < CMD_BASE + CMD_SPAN:
                raise ValueError(f"expected CMD at {i}, got {tc}")
            command = tc - CMD_BASE
            i += 1
            data, i = _read_byte(ids, i)
            rows.append(Row(note, instr, command, data))
        i = _expect(ids, i, PATTERN_END, "PATTERN_END")
        patterns.append(Pattern(rows=rows))
    return patterns, i


def _read_orderlists(ids, i):
    from pygoattracker.model import (
        Orderlist,
        PlayPattern,
        Repeat,
        Subtune,
        Transpose,
    )

    i = _expect(ids, i, SEC_ORDERLISTS, "SEC_ORDERLISTS")
    subtunes = []
    while ids[i] == SUBTUNE_BEGIN:
        i += 1
        channels = []
        while ids[i] == CHANNEL:
            i += 1
            restart, i = _read_byte(ids, i)
            entries = []
            while ids[i] in (ORDER_PLAY, ORDER_REPEAT, ORDER_TRANSPOSE):
                op = ids[i]
                i += 1
                val, i = _read_byte(ids, i)
                if op == ORDER_PLAY:
                    entries.append(PlayPattern(val))
                elif op == ORDER_REPEAT:
                    entries.append(Repeat(val))
                else:
                    entries.append(Transpose(val - 256 if val > 127 else val))
            channels.append(Orderlist(entries=entries, restart=restart))
        i = _expect(ids, i, SUBTUNE_END, "SUBTUNE_END")
        subtunes.append(Subtune(channels=channels))
    return subtunes, i


def flat_gt_ids_to_program(ids):
    from pygoattracker.model import Song

    i = _expect(ids, 0, BOS, "BOS")
    nframes, seed, boot, boot1, i = _read_header(ids, i)
    tables, i = _read_tables(ids, i)
    instruments, i = _read_instruments(ids, i)
    patterns, i = _read_patterns(ids, i)
    subtunes, i = _read_orderlists(ids, i)
    i = _expect(ids, i, EOS, "EOS")
    # _read_tables always returns exactly 4 tables (wave/pulse/filter/speed).
    # pylint: disable-next=unbalanced-tuple-unpacking
    wavetable, pulsetable, filtertable, speedtable = tables
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
    program = make_program_from_song(song, seed, nframes)
    program.boot = boot
    program.tables["boot1"] = boot1
    return program


def flat_gt_measure(program):
    song = _song_of(program)
    blocks = {}
    out = [BOS]
    _emit_header(out, program)
    blocks["header"] = len(out)
    base = len(out)
    _emit_tables(out, song)
    blocks["tables"] = len(out) - base
    base = len(out)
    _emit_instruments(out, song)
    blocks["instr_def"] = len(out) - base
    base = len(out)
    _emit_patterns(out, song)
    blocks["score"] = len(out) - base
    base = len(out)
    _emit_orderlists(out, song)
    blocks["orders"] = len(out) - base
    out.append(EOS)
    blocks["total"] = len(out)
    blocks["fmt"] = "flat_v2"
    return blocks, program.nframes
