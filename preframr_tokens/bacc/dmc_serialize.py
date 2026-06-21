"""Serialize a DMC (Demo Music Creator) BaccProgram to/from token ids.

The recovered DMC program is the SONG MODEL (per-voice orderlists + patterns +
instruments + the note table) plus the engine template that is the driver VM
(re-run by ``backends/dmc.render``). The token form carries:

  header  : nframes, load/init/play addrs, subtune index/speed/master-vol, boot[25]
  image   : engine template length + bytes (the VM; render overwrites its
            song-data regions from the model below, so the model drives render)
  freq    : the 96-entry note->Fn table (defines the A440 grid anchor)
  orders  : per voice -- terminator + (transpose, pattern) entries
  patterns: per referenced pattern -- addr + token rows; NOTE tokens are the
            ABSOLUTE canonical A440 grid index (``pitch.fn_to_grid`` of the note's
            table Fn) -- IDENTICAL to the GoatTracker token for the same concert
            pitch, not a raw note byte
  patptr  : the pattern-pointer table (lo/hi)
  instr   : referenced instrument records (11 bytes each)

All values ride the shared base-16 LEB digit alphabet (no new ids). The round-trip
is exact and ``render`` reproduces the dump byte-exact (modulo the +1 boot frame).
"""

from preframr_tokens.bacc.pitch import fn_to_grid
from preframr_tokens.bacc.primitive import BaccProgram
from preframr_tokens.bacc.serialize import _ri, _ru, _wi, _wu

NREG = 25

# Pattern-token kind codes (small ints, ride the LEB digit alphabet).
_K_NOTE, _K_INS, _K_DUR, _K_FX, _K_TIE, _K_REST, _K_END = range(7)


def _grid_anchor(freq):
    """Grid index of note-table entry 0 (the table is a clean 12-TET bijection, so
    grid(note n) = anchor + n). Returns the absolute A440 grid index of note 0."""
    return fn_to_grid(freq[0])


def _emit_note(out, note, freq, anchor):
    """Emit a note as its absolute canonical A440 grid index when the note resolves
    through the clean freq-table bijection; else a literal-index escape (LEB LSB:
    bit0=0 canonical grid index, bit0=1 literal table index). Keeps the token
    GoatTracker-identical for in-range notes, lossless for any tail note."""
    if (
        0 <= note < len(freq)
        and freq[note] > 0
        and fn_to_grid(freq[note]) == anchor + note
    ):
        _wu(out, (note << 1))  # bit0=0: canonical (grid = anchor + note)
    else:
        _wu(out, (note << 1) | 1)  # bit0=1: literal table index escape


def _read_note(ids, i):
    z, i = _ru(ids, i)
    return z >> 1, i  # the (anchor + n) bijection makes index == grid - anchor


def _emit_pattern_toks(out, toks, freq, anchor):
    _wu(out, len(toks))
    for tok in toks:
        kind = tok[0]
        if kind == "note":
            _wu(out, _K_NOTE)
            _emit_note(out, tok[1], freq, anchor)
        elif kind == "ins":
            _wu(out, _K_INS)
            _wu(out, tok[1])
        elif kind == "dur":
            _wu(out, _K_DUR)
            _wu(out, tok[1])
        elif kind == "fx":
            _wu(out, _K_FX)
            _wu(out, tok[1])
            _wu(out, len(tok[2]))
            for b in tok[2]:
                _wu(out, b)
        elif kind == "tie":
            _wu(out, _K_TIE)
        elif kind == "rest":
            _wu(out, _K_REST)
        elif kind == "end":
            _wu(out, _K_END)
        else:
            raise ValueError(f"unknown DMC pattern token {tok!r}")


def _read_pattern_toks(ids, i):
    n, i = _ru(ids, i)
    toks = []
    for _ in range(n):
        kind, i = _ru(ids, i)
        if kind == _K_NOTE:
            note, i = _read_note(ids, i)
            toks.append(("note", note))
        elif kind == _K_INS:
            v, i = _ru(ids, i)
            toks.append(("ins", v))
        elif kind == _K_DUR:
            v, i = _ru(ids, i)
            toks.append(("dur", v))
        elif kind == _K_FX:
            cmd, i = _ru(ids, i)
            nparam, i = _ru(ids, i)
            params = []
            for _ in range(nparam):
                b, i = _ru(ids, i)
                params.append(b)
            toks.append(("fx", cmd, tuple(params)))
        elif kind == _K_TIE:
            toks.append(("tie",))
        elif kind == _K_REST:
            toks.append(("rest",))
        elif kind == _K_END:
            toks.append(("end",))
        else:
            raise ValueError(f"unknown DMC pattern kind {kind}")
    return toks, i


def _term_code(term):
    return 0 if term == "loop" else 1


def dmc_program_to_ids(program):
    """Serialize a DMC BaccProgram to a flat list of token ids."""
    out = []
    seed = program.seed
    song = program.tables["song"]
    freq = song["freq"]
    anchor = _grid_anchor(freq)

    _wu(out, program.nframes)
    _wu(out, seed["load_addr"])
    _wu(out, seed["init_addr"])
    _wu(out, seed["play_addr"])
    _wu(out, song["subtune"])
    _wu(out, song["song_speed"])
    _wu(out, song["master_vol"])
    _wu(out, song["a_sub"])
    _wu(out, song["a_patlo"])
    _wu(out, song["a_pathi"])
    for b in program.boot:
        _wu(out, b)

    image = seed["image"]
    _wu(out, len(image))
    for b in image:
        _wu(out, b)

    # freq table (defines the grid anchor; 96 little-endian entries)
    _wu(out, len(freq))
    for fn in freq:
        _wu(out, fn)

    # orderlists (per voice): pointer, terminator, entries (transpose, pattern)
    for order in song["orders"]:
        _wu(out, order["ptr"])
        _wu(out, _term_code(order["term"]))
        _wu(out, len(order["entries"]))
        for tr, pat, has_prefix in order["entries"]:
            _wi(out, tr)
            _wu(out, pat)
            _wu(out, has_prefix)

    # patterns: count, then (number, addr, token rows) each
    pats = song["patterns"]
    _wu(out, len(pats))
    for num in sorted(pats):
        pat = pats[num]
        _wu(out, num)
        _wu(out, pat["addr"])
        _emit_pattern_toks(out, pat["toks"], freq, anchor)

    # pattern-pointer table (lo/hi)
    _wu(out, len(song["pat_ptr"]))
    for lo, hi in song["pat_ptr"]:
        _wu(out, lo)
        _wu(out, hi)

    # instruments: count, then (index, 11 bytes) each
    instrs = song["instruments"]
    _wu(out, len(instrs))
    for idx in sorted(instrs):
        _wu(out, idx)
        for b in instrs[idx]:
            _wu(out, b)
    return out


def dmc_ids_to_program(ids):
    """Inverse of ``dmc_program_to_ids`` -> BaccProgram (byte-exact round-trip)."""
    i = 0
    nframes, i = _ru(ids, i)
    load_addr, i = _ru(ids, i)
    init_addr, i = _ru(ids, i)
    play_addr, i = _ru(ids, i)
    subtune, i = _ru(ids, i)
    song_speed, i = _ru(ids, i)
    master_vol, i = _ru(ids, i)
    a_sub, i = _ru(ids, i)
    a_patlo, i = _ru(ids, i)
    a_pathi, i = _ru(ids, i)
    boot = []
    for _ in range(NREG):
        b, i = _ru(ids, i)
        boot.append(b)

    nimg, i = _ru(ids, i)
    image = []
    for _ in range(nimg):
        b, i = _ru(ids, i)
        image.append(b)

    nfreq, i = _ru(ids, i)
    freq = []
    for _ in range(nfreq):
        fn, i = _ru(ids, i)
        freq.append(fn)

    orders = []
    for _ in range(3):
        ptr, i = _ru(ids, i)
        term_code, i = _ru(ids, i)
        nent, i = _ru(ids, i)
        entries = []
        for _ in range(nent):
            tr, i = _ri(ids, i)
            pat, i = _ru(ids, i)
            has_prefix, i = _ru(ids, i)
            entries.append((tr, pat, has_prefix))
        orders.append(
            {
                "ptr": ptr,
                "term": "loop" if term_code == 0 else "stop",
                "entries": entries,
            }
        )

    npats, i = _ru(ids, i)
    patterns = {}
    for _ in range(npats):
        num, i = _ru(ids, i)
        addr, i = _ru(ids, i)
        toks, i = _read_pattern_toks(ids, i)
        patterns[num] = {"addr": addr, "toks": toks}

    nptr, i = _ru(ids, i)
    pat_ptr = []
    for _ in range(nptr):
        lo, i = _ru(ids, i)
        hi, i = _ru(ids, i)
        pat_ptr.append((lo, hi))

    ninstr, i = _ru(ids, i)
    instruments = {}
    for _ in range(ninstr):
        idx, i = _ru(ids, i)
        rec = []
        for _ in range(11):
            b, i = _ru(ids, i)
            rec.append(b)
        instruments[idx] = rec

    vptr = [orders[0]["ptr"], orders[1]["ptr"], orders[2]["ptr"]]
    song = {
        "subtune": subtune,
        "song_speed": song_speed,
        "master_vol": master_vol,
        "a_sub": a_sub,
        "a_patlo": a_patlo,
        "a_pathi": a_pathi,
        "vptr": vptr,
        "orders": orders,
        "patterns": patterns,
        "pat_ptr": pat_ptr,
        "instruments": instruments,
        "freq": freq,
    }
    return BaccProgram(
        driver="dmc",
        nframes=nframes,
        boot=boot,
        instruments=[],
        score=[],
        seed={
            "load_addr": load_addr,
            "init_addr": init_addr,
            "play_addr": play_addr,
            "image": image,
        },
        tables={"song": song},
    )


def dmc_measure(program):
    """Return ({block: tokens}, nframes) for the serialized DMC program."""
    ids = dmc_program_to_ids(program)
    seed = program.seed
    song = program.tables["song"]
    header = (
        _w(program.nframes)
        + _w(seed["load_addr"])
        + _w(seed["init_addr"])
        + _w(seed["play_addr"])
        + _w(song["subtune"])
        + _w(song["song_speed"])
        + _w(song["master_vol"])
        + _w(song["a_sub"])
        + _w(song["a_patlo"])
        + _w(song["a_pathi"])
    )
    boot = sum(_w(b) for b in program.boot)
    image = _w(len(seed["image"])) + sum(_w(b) for b in seed["image"])
    freq = _w(len(song["freq"])) + sum(_w(f) for f in song["freq"])
    score = len(ids) - header - boot - image - freq
    brk = {
        "header": header,
        "boot": boot,
        "image": image,
        "freq": freq,
        "score": score,
        "total": len(ids),
    }
    return brk, program.nframes


def _w(n):
    out = []
    _wu(out, n)
    return len(out)
