"""DMC (Demo Music Creator) v7.62 song model -- ported into preframr-tokens.

Decodes the song-data byte regions of a DMC v7.62 ``$1000`` player image into the
common BACC abstraction (per-voice orderlists + patterns + instruments + the
note-frequency table) and re-emits them byte-exact. The parsing logic is ported
from the reverse-engineered ``undmc`` ``targets/dmc/format.py`` model (memory map
in ``undmc/targets/dmc/DMC_FORMAT.md``); nothing is imported at runtime, exactly
like ``gt_unpack`` brought the GoatTracker layout in-tree.

The musically meaningful structure is recovered into the abstraction; the player
engine code + its fixed effect/wave/filter tables are the driver VM (re-run by
``backends/dmc.render`` exactly as lft re-runs its image and GoatTracker re-runs
pygoattracker). The note bytes in patterns map through the 96-entry freq table at
``$1647/$16A7`` -- a clean ascending 12-TET bijection -- onto the canonical A440
grid, so a DMC note serializes to the SAME absolute grid token as a GoatTracker
note at the same concert pitch (``pitch.fn_to_grid``).

Round-trip contract: ``emit(parse(image)) == image`` over every song-data region,
so a render that rebuilds those regions from the recovered model and re-runs the
engine reproduces the dump byte-exact (modulo the +1 boot frame the GoatTracker
``_align`` already absorbs).
"""

# Player constants (absolute C64 addresses, DMC v7.62 / $1000 load). The subtune
# table and the pattern-pointer tables RELOCATE between v7.62 builds (their song
# data sits at the image tail, whose length varies); they are DERIVED from the
# player code (``derive_layout``) rather than hardcoded. The freq table, the
# instrument table, and the orderlist working pointers are fixed in this player
# build, so they ride as constants (their defaults below double as the layout's
# fallback when a code probe somehow misses).
A_SUBTUNE = 0x17F0  # per-subtune table, 8 bytes each (DEFAULT; derived per tune)
A_PATLO, A_PATHI = 0x1829, 0x182D  # pattern ptr lo/hi (DEFAULT; derived per tune)
A_INSTR = 0x17B0  # instrument table, 11 bytes each (fixed)
A_FREQ_LO, A_FREQ_HI = 0x1647, 0x16A7  # note -> SID Fn (96 notes; fixed)
INSTR_LEN = 11
NUM_NOTES = 96
SUBTUNE_LEN = 8


def derive_layout(mem):
    """Derive the relocating table addresses from the player code.

    Returns ``(subtune_addr, patlo_addr, pathi_addr)``. The subtune table is read
    by ``init`` (``LDA subtune,Y / STA ord_ptr_lo,X / LDA subtune+1,Y / STA
    ord_ptr_hi,X`` -- ``B9 lo hi 9D .. .. B9 lo hi 9D``). The pattern-pointer
    tables are read by the orderlist walker (``LDA patlo,Y / STA $F8 / LDA pathi,Y
    / STA $F9`` -- ``B9 lo hi 85 F8 B9 lo hi 85 F9``). Falls back to the build
    defaults if a probe misses.
    """
    init = mem.b(0x1001) | (mem.b(0x1002) << 8)
    subtune = A_SUBTUNE
    for off in range(init, init + 96):
        if (
            mem.b(off) == 0xB9
            and mem.b(off + 3) == 0x9D
            and mem.b(off + 6) == 0xB9
            and mem.b(off + 9) == 0x9D
        ):
            subtune = mem.b(off + 1) | (mem.b(off + 2) << 8)
            break
    patlo, pathi = A_PATLO, A_PATHI
    for off in range(0x1000, 0x1700):
        if (
            mem.b(off) == 0xB9
            and mem.b(off + 3) == 0x85
            and mem.b(off + 4) == 0xF8
            and mem.b(off + 5) == 0xB9
            and mem.b(off + 8) == 0x85
            and mem.b(off + 9) == 0xF9
        ):
            patlo = mem.b(off + 1) | (mem.b(off + 2) << 8)
            pathi = mem.b(off + 6) | (mem.b(off + 7) << 8)
            break
    return subtune, patlo, pathi


# Orderlist terminators.
ORD_LOOP = 0xFF
ORD_STOP = 0xFE
# Pattern stream control bytes.
PAT_END = 0xFF
PAT_REST = 0xFE
PAT_TIE = 0xFD
PAT_FX = 0xC0  # $C0..$FC effect (bit $10 selects 1- vs 2-byte parameter form)
PAT_DUR = 0x80  # $80..$BF set note length
PAT_INS = 0x60  # $60..$7F set instrument
# notes are $00..$5F (0..95)


class Mem:
    """64K image accessor (load a DMC image, read bytes/words)."""

    def __init__(self, base, data):
        self.m = bytearray(0x10000)
        self.m[base : base + len(data)] = data

    def b(self, a):
        return self.m[a]

    def w(self, a):
        return self.m[a] | (self.m[a + 1] << 8)


def freq_table(mem):
    """The 96-entry note->Fn table (little-endian lo/hi)."""
    return [
        mem.b(A_FREQ_LO + n) | (mem.b(A_FREQ_HI + n) << 8) for n in range(NUM_NOTES)
    ]


def parse_orderlist(mem, ptr):
    """Walk one voice's orderlist into ``(entries, term, nbytes)``.

    ``entries`` is a list of ``(transpose, pattern)``; ``transpose`` is the signed
    ``value - $A0`` prefix that applies to the following pattern (0 if none).
    ``term`` is ``"loop"`` ($FF) or ``"stop"`` ($FE). ``nbytes`` is the consumed
    byte length (so the emitter writes back exactly the same span).
    """
    entries, i, transpose = [], 0, 0
    while i < 512:
        v = mem.b(ptr + i)
        i += 1
        if v == ORD_LOOP:
            return entries, "loop", i
        if v == ORD_STOP:
            return entries, "stop", i
        if v >= 0x80:
            transpose = v - 0xA0
            pat = mem.b(ptr + i)
            i += 1
            # has_prefix=1: this entry carried an explicit transpose-prefix byte in
            # the image (DMC re-emits an $A0 even when transpose is unchanged), so
            # the emitter must reproduce it for a byte-exact round-trip.
            entries.append((transpose, pat, 1))
        else:
            entries.append((transpose, v, 0))
    raise ValueError(f"DMC orderlist overrun at ${ptr:04X}")


def emit_orderlist(entries, term):
    """Inverse of ``parse_orderlist``: model -> exact orderlist bytes.

    Each entry's ``has_prefix`` flag records whether the image carried an explicit
    transpose-prefix byte before the pattern (DMC emits a redundant $A0 even when
    the transpose is unchanged); re-emitting exactly that prefix structure makes
    the byte stream round-trip identically.
    """
    out = []
    for tr, pat, has_prefix in entries:
        if has_prefix:
            out.append((tr + 0xA0) & 0xFF)
        out.append(pat)
    out.append(ORD_LOOP if term == "loop" else ORD_STOP)
    return out


def parse_pattern(mem, ptr):
    """Walk one pattern into a token row list + consumed byte length.

    Tokens are tuples whose first element is a kind string:
      ("note", n)        note 0..95 (absolute table index)
      ("ins", i)         set instrument 0..31
      ("dur", d)         set note length 0..63
      ("fx", cmd, params)  effect ($C0..$FC); ``params`` is the raw parameter
                           byte tuple (1 or 2 bytes per the $10 form bit)
      ("tie",)           $FD tie/legato
      ("rest",)          $FE rest
      ("end",)           $FF end of pattern
    """
    toks, i = [], 0
    while i < 1024:
        v = mem.b(ptr + i)
        i += 1
        if v == PAT_END:
            toks.append(("end",))
            return toks, i
        if v == PAT_REST:
            toks.append(("rest",))
            continue
        if v == PAT_TIE:
            toks.append(("tie",))
            continue
        if v >= PAT_FX:
            cmd = v & 0x1F
            # bit $10 set -> 1 parameter byte; clear -> 2 parameter bytes
            nparam = 1 if (v & 0x10) else 2
            params = tuple(mem.b(ptr + i + k) for k in range(nparam))
            i += nparam
            toks.append(("fx", cmd, params))
            continue
        if v >= PAT_DUR:
            toks.append(("dur", v & 0x3F))
            continue
        if v >= PAT_INS:
            toks.append(("ins", v & 0x1F))
            continue
        toks.append(("note", v))
    raise ValueError(f"DMC pattern overrun at ${ptr:04X}")


def emit_pattern(toks):
    """Inverse of ``parse_pattern``: token rows -> exact pattern bytes."""
    out = []
    for tok in toks:
        kind = tok[0]
        if kind == "note":
            out.append(tok[1])
        elif kind == "ins":
            out.append(PAT_INS | (tok[1] & 0x1F))
        elif kind == "dur":
            out.append(PAT_DUR | (tok[1] & 0x3F))
        elif kind == "fx":
            cmd, params = tok[1], tok[2]
            byte = PAT_FX | cmd | (0x10 if len(params) == 1 else 0)
            out.append(byte)
            out.extend(params)
        elif kind == "tie":
            out.append(PAT_TIE)
        elif kind == "rest":
            out.append(PAT_REST)
        elif kind == "end":
            out.append(PAT_END)
        else:
            raise ValueError(f"unknown DMC pattern token {tok!r}")
    return out


def instrument(mem, ins):
    """The 11-byte instrument record for instrument index ``ins``."""
    return [mem.b(A_INSTR + ins * INSTR_LEN + k) for k in range(INSTR_LEN)]


def parse_song(mem, subtune):
    """Parse a DMC image into the structured song model for one subtune.

    Returns a dict with: ``subtune`` index, ``song_speed``, ``master_vol``,
    per-voice ``vptr`` orderlist addresses, ``orders`` (entries/term per voice),
    ``patterns`` {n: (addr, toks)} for every referenced pattern, ``pat_ptr``
    (the lo/hi pattern-pointer table truncated to the max referenced pattern),
    ``instruments`` {i: record} for every referenced instrument, and the 96-entry
    ``freq`` table. This IS the recovered abstraction (no raw engine bytes).
    """
    a_sub, a_patlo, a_pathi = derive_layout(mem)
    t = a_sub + subtune * SUBTUNE_LEN
    vptr = [mem.w(t), mem.w(t + 2), mem.w(t + 4)]
    song_speed = mem.b(t + 6)
    master_vol = mem.b(t + 7)

    orders, used = [], set()
    for v in range(3):
        entries, term, nbytes = parse_orderlist(mem, vptr[v])
        orders.append({"ptr": vptr[v], "entries": entries, "term": term, "n": nbytes})
        for _, pat, _ in entries:
            used.add(pat)

    patterns, instr_used = {}, set()
    for pat in sorted(used):
        paddr = mem.b(a_patlo + pat) | (mem.b(a_pathi + pat) << 8)
        toks, nbytes = parse_pattern(mem, paddr)
        patterns[pat] = {"addr": paddr, "toks": toks, "n": nbytes}
        for tok in toks:
            if tok[0] == "ins":
                instr_used.add(tok[1])

    max_pat = max(used) if used else 0
    pat_ptr = [(mem.b(a_patlo + p), mem.b(a_pathi + p)) for p in range(max_pat + 1)]
    instruments = {i: instrument(mem, i) for i in sorted(instr_used)}
    return {
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
        "freq": freq_table(mem),
    }


def emit_song(image, song):
    """Write the structured model's song-data regions back into ``image`` (a
    mutable bytearray, the engine template) byte-exact, returning it.

    Every region the parser read is re-derived from the model: the subtune record,
    the three orderlists, the pattern-pointer table, every pattern body, and every
    instrument record. The freq table is the model's (it defines the grid). The
    result equals the original image over all song-data regions, so re-running the
    engine reproduces the dump byte-exact.
    """
    sub = song["subtune"]
    a_sub = song.get("a_sub", A_SUBTUNE)
    a_patlo = song.get("a_patlo", A_PATLO)
    a_pathi = song.get("a_pathi", A_PATHI)
    t = a_sub + sub * SUBTUNE_LEN
    vptr = song["vptr"]
    for v in range(3):
        image[t + v * 2] = vptr[v] & 0xFF
        image[t + v * 2 + 1] = (vptr[v] >> 8) & 0xFF
    image[t + 6] = song["song_speed"]
    image[t + 7] = song["master_vol"]

    for v, order in enumerate(song["orders"]):
        bytes_out = emit_orderlist(order["entries"], order["term"])
        for k, b in enumerate(bytes_out):
            image[order["ptr"] + k] = b

    for p, (lo, hi) in enumerate(song["pat_ptr"]):
        image[a_patlo + p] = lo
        image[a_pathi + p] = hi

    for pat in song["patterns"].values():
        bytes_out = emit_pattern(pat["toks"])
        for k, b in enumerate(bytes_out):
            image[pat["addr"] + k] = b

    for ins, rec in song["instruments"].items():
        for k, b in enumerate(rec):
            image[A_INSTR + ins * INSTR_LEN + k] = b

    for n, fn in enumerate(song["freq"]):
        image[A_FREQ_LO + n] = fn & 0xFF
        image[A_FREQ_HI + n] = (fn >> 8) & 0xFF
    return image


def song_regions(song):
    """The (start, length) byte spans the model owns -- the song-data regions the
    parser read and ``emit_song`` rebuilds. ``recover`` blanks these in the stored
    engine template so the model is the SOLE source of the song data (no raw song
    bytes are retained; the template carries only the player engine + fixed tables).
    """
    sub = song["subtune"]
    a_sub = song.get("a_sub", A_SUBTUNE)
    a_patlo = song.get("a_patlo", A_PATLO)
    a_pathi = song.get("a_pathi", A_PATHI)
    spans = [(a_sub + sub * SUBTUNE_LEN, SUBTUNE_LEN)]
    for order in song["orders"]:
        spans.append(
            (order["ptr"], len(emit_orderlist(order["entries"], order["term"])))
        )
    npat = len(song["pat_ptr"])
    spans.append((a_patlo, npat))
    spans.append((a_pathi, npat))
    for pat in song["patterns"].values():
        spans.append((pat["addr"], len(emit_pattern(pat["toks"]))))
    for ins in song["instruments"]:
        spans.append((A_INSTR + ins * INSTR_LEN, INSTR_LEN))
    spans.append((A_FREQ_LO, NUM_NOTES))
    spans.append((A_FREQ_HI, NUM_NOTES))
    return spans
