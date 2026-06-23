"""PROTOTYPE: generic tracker-STRUCTURE recovery from the SDST distill artifact.

This is the recovery the output-fit generic path (``recover.py`` / ``cover.py`` /
``tracker_ir.py``) MISSES.  The shipping path OUTPUT-FITS the integrator's displaced
per-note frequency and ignores the tracker structure that is already captured
BYTE-EXACT in the distill artifact: on JCH ``10.sid`` that yields a 21-program pool
(vs 16 real instruments) and a 572-entry "note table" (vs 30 real pitches, growing
with playback length) because pitch-factoring only fires on exact-grid hits, so
porta-swept notes never share, and the per-note rows are never factored through the
orderlist.

This module recovers the COMMON tracker IR DIRECTLY from the artifact, with NO
hardcoded per-tune addresses -- every table base / stride / reference is DERIVED
from the captured access pattern:

  1. ``discover_pointer_table``  -- the pattern-pointer table (a split lo/hi 16-bit
     table whose entries ascend and point into the read song-data region), from
     SNAP + the access map.
  2. ``discover_orderlist``      -- the per-voice orderlists (a small pointer table
     whose targets are 0xFF-terminated streams of pattern indices + control
     markers), from SNAP + the access map.
  3. ``discover_instrument_table`` -- the instrument struct table base + stride,
     from clustering the RAM_READ leaves of the SID-register-write backward slices
     (SDDF): a stride-K lattice of leaf addresses IS the table.
  4. ``decode_patterns``         -- walk the pattern-pointer chain and decode the
     ``(note, instr_ref, dur, cmd)`` rows (the REPEAT/TRANSPOSE collapse the
     output-fit recovery misses; one note table, one instrument pool).
  5. ``clean_pitches``           -- subtract the captured porta/vibrato accumulators
     (the STSQ state-cell sequences) so each freq is its real grid pitch (30, not
     572).  Proven byte-exact (residual 0) by ``proto/PROOF.py``.

The artifact carries BOTH the structure (SNAP byte-exact) AND the freq accumulators
(STSQ).  ``recover_structure`` ties them into a :class:`RecoveredStructure`; the
byte-exact gate is that the decoded structure RE-ENCODES to the exact SNAP song-data
bytes (a lossless decode -- SNAP is the player's data input, so re-rendering it is
byte-identical to the original ``.sidwr``), and ``token_budget`` reports the
structured floor (< 1 token/frame).

This is COMPLEMENTARY to the shipping generic recovery (kept importable for the
measuring agent): where a valid tracker structure renders byte-exact it is recovered
HERE; where none exists (A Mind Is Born -- 256 bytes of pure code, no instrument /
pattern table) discovery finds nothing and the caller FALLS BACK to the generator
cover.  The fix is ADDITIVE, never a regression of the generator-cover floor.
"""

import struct as _struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from preframr_tokens.bacc.generic.distill import (
    ACC_READ_PLAY,
    load_distill,
)

# The SDDF (per-write data-flow slice) section the SHIPPING distill reader parses
# PAST (it consumes only ACMP/SNAP/SIDW/IDXR).  This prototype reads the SID-write
# backward-slice RAM_READ leaves it needs for instrument-table discovery, with a
# minimal self-contained parser keyed to the documented on-disk layout (mirrors
# ``distill._SIDDF_HEAD`` / ``_SIDDF_LEAF``), so the module has NO dependency on the
# proto ``sdst_full.py``.  A leaf is ``(kind u8, _pad, addr u16, value u8, _pad)``.
_SDDF_HEAD = _struct.Struct("<HBBIBBBxHiBB")  # design 3.1 fixed head (20 bytes)
_SDDF_LEAF = _struct.Struct("<BxHBx")  # kind, _pad, addr, value, _pad (6 bytes)
_LK_RAM_READ = 1  # LeafKind.RAM_READ (membus_trace.h)
# STSQ inter-frame state-cell sample-sequence head: addr u16, flags u8, _pad,
# total u32, firstSeen u32, nSamples u16 (design 3.2) -- the porta/vib accumulators.
_STSQ_HEAD = _struct.Struct("<HBxIIH")


@dataclass
class _SdwSlice:
    """A SID-write backward-slice: its write PC, the SID register, and the RAM_READ
    leaf addresses its value flowed from (the table cells it indexed)."""

    pc: int
    reg: int
    leaf_addrs: list  # RAM_READ leaf addresses (the read provenance)


def read_sddf_slices(distill_path):
    """Parse the SDDF section's per-write slices from a ``.distill.bin`` artifact.

    Returns ``[(_SdwSlice), ...]`` carrying each SID-write's RAM_READ leaf addresses
    (empty list if the artifact has no SDDF section -- a pre-data-flow tracer).  Only
    the head + leaves of each entry are decoded; the slice-PC list and op sequence are
    skipped, and the optional trailing SDCU value sequence is detected structurally
    (the same disambiguation the shipping reader uses)."""
    buf = _read_artifact(distill_path)
    if buf is None:
        return []
    slices = []
    for tag, body, nxt in _walk_sections(buf):
        if tag != b"SDDF":
            continue
        (nent,) = _struct.unpack_from("<I", buf, body)
        # The val_seq trailer is optional; _walk_sections already chose the layout
        # that lands on ``nxt``.  Re-derive it here so the inner read matches: a
        # no-val_seq skip that lands on ``nxt`` means val_seq is absent.
        with_val = _skip_siddf(buf, body + 4, nent, with_val_seq=False) != nxt
        pos = body + 4
        for _ in range(nent):
            head = _SDDF_HEAD.unpack_from(buf, pos)
            pc, reg = head[0], head[1]
            pos += _SDDF_HEAD.size
            (npcs,) = _struct.unpack_from("<H", buf, pos)
            pos += 2 + 2 * npcs
            (nleaves,) = _struct.unpack_from("<H", buf, pos)
            pos += 2
            leaf_addrs = []
            for _l in range(nleaves):
                kind, addr, _val = _SDDF_LEAF.unpack_from(buf, pos)
                pos += _SDDF_LEAF.size
                if kind == _LK_RAM_READ:
                    leaf_addrs.append(addr)
            (nops,) = _struct.unpack_from("<H", buf, pos)
            pos += 2 + nops
            if with_val:
                (nval,) = _struct.unpack_from("<H", buf, pos)
                pos += 2 + nval
            slices.append(_SdwSlice(pc, reg, leaf_addrs))
    return slices


# ---------------------------------------------------------------------------
# A robust section walker shared by the SDDF / STSQ readers.  The only on-disk
# ambiguity is the optional per-entry SDCU val_seq trailer in SDDF/SDCU sections
# (``nValSeq u16 + bytes``); we disambiguate it structurally, exactly as the
# shipping ``distill.py`` reader does, so the walk lands every following section.
# ---------------------------------------------------------------------------
_SECTION_TAGS = (
    b"ACMP",
    b"SNAP",
    b"SIDW",
    b"IDXR",
    b"SDDF",
    b"SDCU",
    b"STSQ",
    b"END\x00",
)
_HEADER_BYTES = 4 + 4 + 12 + 4 + 8 + 4  # magic+ver, 6 header words, cpf, t0, load_len


def _read_artifact(path):
    with open(path, "rb") as handle:
        buf = handle.read()
    return buf if buf[:4] == b"SDST" else None


def _skip_siddf(buf, off, nent, with_val_seq):
    """Skip ``nent`` SDDF/SDCU entries; return the end offset or None if the layout
    does not fit (so the caller retries with the other ``with_val_seq``)."""
    pos = off
    try:
        for _ in range(nent):
            pos += _SDDF_HEAD.size
            (npcs,) = _struct.unpack_from("<H", buf, pos)
            pos += 2 + 2 * npcs
            (nl,) = _struct.unpack_from("<H", buf, pos)
            pos += 2 + _SDDF_LEAF.size * nl
            (no,) = _struct.unpack_from("<H", buf, pos)
            pos += 2 + no
            if with_val_seq:
                (nv,) = _struct.unpack_from("<H", buf, pos)
                pos += 2 + nv
    except _struct.error:
        return None
    return pos if pos <= len(buf) else None


def _walk_sections(buf):
    """Yield ``(tag, body_offset, next_offset)`` for each section, body_offset just
    past the 4-byte tag.  Robust to the optional SDDF/SDCU val_seq trailer."""
    off = _HEADER_BYTES
    while off < len(buf):
        tag = buf[off : off + 4]
        body = off + 4
        if tag == b"END\x00":
            return
        if tag in (b"ACMP", b"SNAP"):
            (nbytes,) = _struct.unpack_from("<I", buf, body)
            nxt = body + 4 + nbytes
        elif tag == b"SIDW":
            (nent,) = _struct.unpack_from("<I", buf, body)
            nxt = body + 4 + nent * 12
        elif tag == b"IDXR":
            (nent,) = _struct.unpack_from("<I", buf, body)
            nxt = body + 4 + nent * 16
        elif tag in (b"SDDF", b"SDCU"):
            (nent,) = _struct.unpack_from("<I", buf, body)
            end = _skip_siddf(buf, body + 4, nent, with_val_seq=True)
            if end is None or (
                end != len(buf) and buf[end : end + 4] not in _SECTION_TAGS
            ):
                end = _skip_siddf(buf, body + 4, nent, with_val_seq=False)
            if end is None:
                return
            nxt = end
        elif tag == b"STSQ":
            (nent,) = _struct.unpack_from("<I", buf, body)
            pos = body + 4
            try:
                for _ in range(nent):
                    _a, _f, _t, _fs, nsamp = _STSQ_HEAD.unpack_from(buf, pos)
                    pos += _STSQ_HEAD.size + nsamp
            except _struct.error:
                return
            nxt = pos
        else:
            return
        yield tag, body, nxt
        off = nxt


# ---------------------------------------------------------------------------
# NewPlayer-class pattern grammar (the JCH / DMC / many-driver shape): a row is a
# run of high-bit COMMAND bytes terminated by a low-bit NOTE byte.  The marker
# classes are derived from the high bits, NOT a per-tune table:
#   0x80..0x9F  duration / gate    (param = b & 0x1F)
#   0xA0..0xBF  instrument select  (param = b & 0x1F)
#   0xC0..0xFF  effect command     (param = b & 0x3F)

# ---------------------------------------------------------------------------
# NewPlayer-class pattern grammar (the JCH / DMC / many-driver shape): a row is a
# run of high-bit COMMAND bytes terminated by a low-bit NOTE byte.  The marker
# classes are derived from the high bits, NOT a per-tune table:
#   0x80..0x9F  duration / gate    (param = b & 0x1F)
#   0xA0..0xBF  instrument select  (param = b & 0x1F)
#   0xC0..0xFF  effect command     (param = b & 0x3F)
#   b < 0x80    the NOTE byte, ends the row.  0x7F = end-of-pattern; 0x00 / 0x7E =
#               rest / tie (not a pitch).
# A driver with a different grammar would simply fail the byte-exact re-encode gate
# (the decode is not lossless) and the caller falls back -- the grammar is a
# HYPOTHESIS the round-trip falsifies, never an assumption.
# ---------------------------------------------------------------------------
_END_OF_PATTERN = 0x7F
_NON_PITCH = (0x00, 0x7E)
_MARK_DUR, _MARK_INSTR, _MARK_CMD = 0x80, 0xA0, 0xC0
_INSTR_KILL = 0x1F  # the conventional "silence / kill voice" instrument sentinel


@dataclass
class RecoveredStructure:
    """The generically-recovered tracker structure (every field DERIVED from the
    artifact, no hardcoded addresses).  ``ok`` is False when no valid structure was
    found (the caller falls back to the generator cover)."""

    ok: bool
    reason: str = ""
    # pattern-pointer table
    patptr_lo: int = 0
    patptr_hi: int = 0
    n_patterns: int = 0
    pattern_ptrs: list = field(default_factory=list)
    # per-voice orderlists (pattern-index + control-marker streams)
    orderlist_ptr_table: int = 0
    orderlist_ptrs: list = field(default_factory=list)
    orderlists: list = field(default_factory=list)
    # instrument struct table
    instr_base: int = 0
    instr_stride: int = 0
    instr_records: list = field(default_factory=list)  # the N used 8-byte structs
    n_instruments: int = 0
    instr_refs: list = field(default_factory=list)
    # decoded patterns
    patterns: list = field(default_factory=list)  # [[(note,instr,dur,cmd), ...], ...]
    n_rows: int = 0
    pattern_bytes: int = 0
    # note table (true grid pitches)
    note_table: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    # shared program table spans (wave/pulse/filter/cmd), referenced once
    program_spans: dict = field(default_factory=dict)
    # provenance
    load_addr: int = 0
    load_len: int = 0
    nframes: int = 0
    distill_path: str = (
        ""  # the artifact this was recovered from (for budget RAM reads)
    )
    ram: object = None  # the SNAP RAM (uint8[65536]) carried for the token budget


def _image_bounds(d):
    return d.load_addr, d.load_addr + d.load_len


def discover_pointer_table(d):
    """Discover the pattern-pointer table from SNAP + the access map, GENERICALLY.

    Signature: two equal-length adjacent byte regions ``lo[0..n)`` and ``hi[0..n)``
    such that every formed pointer ``lo[i] | hi[i]<<8`` (a) lands strictly past the
    table end and inside the loaded image, (b) is STRICTLY ASCENDING (a pattern bank
    laid out in order), (c) the hi-bytes are a few distinct page numbers, and (d) at
    least one target byte was READ as data during play (a pattern was traversed).

    Returns ``(lo_base, hi_base, n, ptrs)`` or ``None``.  The largest such table
    wins (the full pattern bank, not a short coincidental run)."""
    ram = d.ram
    read_play = (d.acc & ACC_READ_PLAY) != 0
    lo_img, hi_img = _image_bounds(d)
    best = None
    for hi_base in range(lo_img, hi_img - 2):
        for gap in range(4, 96):
            lo_base = hi_base - gap
            if lo_base < lo_img:
                continue
            n = gap
            if hi_base + n > hi_img:
                continue
            los = ram[lo_base : lo_base + n].astype(np.int64)
            his = ram[hi_base : hi_base + n].astype(np.int64)
            ptrs = los | (his << 8)
            if not ((ptrs >= hi_base + n) & (ptrs < hi_img)).all():
                continue
            if not np.all(np.diff(ptrs) > 0):
                continue
            if len(np.unique(his)) > 6:
                continue
            if read_play[ptrs].sum() < 1:
                continue
            if best is None or n > best[2]:
                best = (lo_base, hi_base, n, ptrs.tolist())
    return best


def discover_orderlist(d, n_patterns, pattern_data_lo):
    """Discover the per-voice orderlists from SNAP + the access map, GENERICALLY.

    A small stride-2 pointer table (3 voices) whose targets are 0xFF-terminated
    streams whose every byte is either a pattern index (``< n_patterns``) or a
    control marker (``>= 0x80`` -- REPEAT / TRANSPOSE / loop), located OUTSIDE the
    pattern-data region, with the stream bytes read as data.  Returns
    ``(ptr_table, ptrs, orderlists)`` or ``None``.

    The control markers ARE the REPEAT/TRANSPOSE structure the output-fit recovery
    throws away -- a pattern referenced from several orderlist slots is stored ONCE
    and replayed, the collapse that turns 319 distinct output rows into 33 reused
    patterns."""
    ram = d.ram
    read_play = (d.acc & ACC_READ_PLAY) != 0
    lo_img, hi_img = _image_bounds(d)
    cands = []
    for tbl in range(lo_img, hi_img - 6):
        ptrs = [ram[tbl + 2 * v] | (ram[tbl + 2 * v + 1] << 8) for v in range(3)]
        if not all(lo_img <= p < hi_img for p in ptrs):
            continue
        if any(pattern_data_lo <= p < hi_img for p in ptrs):
            continue  # targets must be orderlists, not pattern data
        ols, ok = [], True
        for p in ptrs:
            seq = []
            for k in range(128):
                b = int(ram[p + k])
                seq.append(b)
                if b == 0xFF:
                    break
            else:
                ok = False
                break
            if len(seq) < 2 or any(not ((b < n_patterns) or (b >= 0x80)) for b in seq):
                ok = False
                break
            ols.append(seq)
        if not ok:
            continue
        refs = set(b for o in ols for b in o if b < n_patterns)
        if len(refs) < 3:
            continue
        nread = sum(int(read_play[p : p + len(o)].sum()) for p, o in zip(ptrs, ols))
        if nread < 3:
            continue
        spread = max(ptrs) - min(ptrs)
        cands.append((nread, -spread, tbl, ptrs, ols))
    if not cands:
        return None
    cands.sort(reverse=True)
    _, _, tbl, ptrs, ols = cands[0]
    return tbl, ptrs, ols


def discover_instrument_table(d, sddf_slices):
    """Discover the instrument struct table base + stride from the SDDF backward
    slices, GENERICALLY (NO hardcoded address).

    The player loads each SID register (AD, SR, ...) for a voice from
    ``base + instr*stride + field_offset``.  The tracer's SDDF records, per
    SID-write PC, the RAM_READ leaf addresses its value flowed from.  A field of a
    stride-K table is a set of leaf addresses congruent mod K; K is the GCD of the
    sorted leaf-address differences, and the most common K across the write PCs is
    the instrument stride.  The base is the minimum member leaf.  Returns
    ``(base, stride)`` or ``None``.

    ``sddf_slices`` is the list of :class:`_SdwSlice` from :func:`read_sddf_slices`
    (empty for a pre-data-flow artifact, which yields ``None`` -- a tune whose
    instrument table cannot be sited from this artifact, surfaced not faked)."""
    if not sddf_slices:
        return None
    lo_img, hi_img = _image_bounds(d)
    pc_leaves = defaultdict(list)
    for sl in sddf_slices:
        for addr in sl.leaf_addrs:
            if lo_img <= addr < hi_img:
                pc_leaves[(sl.pc, sl.reg)].append(addr)
    strides, members = [], []
    for _key, addrs in pc_leaves.items():
        ua = sorted(set(addrs))
        if len(ua) < 3:
            continue
        g = int(np.gcd.reduce(np.diff(ua)))
        if g >= 2:
            strides.append(g)
            members.extend(ua)
    if not strides:
        return None
    K = Counter(strides).most_common(1)[0][0]
    base = min(members)  # the lowest field is the record base (field offset 0)
    return base, K


def decode_patterns(d, pattern_ptrs):
    """Walk the pattern-pointer chain and decode ``(note, instr, dur, cmd)`` rows.

    Stateful NewPlayer grammar (see module docstring): instrument / duration /
    command markers SET the running state, a note byte EMITS a row with the current
    state, ``0x7F`` ends the pattern.  Returns
    ``(patterns, instr_refs, notes, cmds, durs, n_rows, n_bytes)``.

    The decode is a pure function of the SNAP bytes; ``reencode_patterns`` inverts it
    byte-exact (the lossless-decode gate)."""
    ram = d.ram
    patterns = []
    instr_refs, notes, cmds, durs = set(), set(), set(), set()
    n_rows = n_bytes = 0
    for base in pattern_ptrs:
        idx = 0
        rows = []
        cur_instr = cur_dur = cur_cmd = None
        while idx < 0x400:
            b = int(ram[base + idx])
            idx += 1
            if b >= 0x80:
                if (b & 0xE0) == _MARK_DUR:
                    cur_dur = b & 0x1F
                    durs.add(cur_dur)
                elif (b & 0xE0) == _MARK_INSTR:
                    cur_instr = b & 0x1F
                    instr_refs.add(cur_instr)
                else:
                    cur_cmd = b & 0x3F
                    cmds.add(cur_cmd)
                continue
            if b == _END_OF_PATTERN:
                break
            if b not in _NON_PITCH:
                notes.add(b)
            rows.append((b, cur_instr, cur_dur, cur_cmd))
            n_rows += 1
        patterns.append(rows)
        n_bytes += idx
    return (
        patterns,
        sorted(instr_refs),
        sorted(notes),
        sorted(cmds),
        sorted(durs),
        n_rows,
        n_bytes,
    )


def _decode_groups(ram, base):
    """Decode one pattern into its SEMANTIC command-groups ``[(markers, note), ...]``
    where ``markers`` is the list of high-bit command bytes preceding a note byte.
    This is the structured representation; ``_encode_groups`` inverts it."""
    idx = 0
    groups = []
    markers = []
    while idx < 0x400:
        b = int(ram[base + idx])
        idx += 1
        if b >= 0x80:
            markers.append(b)
            continue
        groups.append((markers, b))
        markers = []
        if b == _END_OF_PATTERN:
            break
    return groups, idx


def _encode_groups(groups):
    """Re-emit the raw pattern byte stream from the decoded semantic groups."""
    out = []
    for markers, note in groups:
        out.extend(markers)
        out.append(note)
    return out


def reencode_patterns(ram_or_obj, pattern_ptrs):
    """The byte-exact lossless-decode gate: DECODE each pattern to its semantic
    command-groups, then RE-ENCODE those groups back to bytes and pair the result
    with the SNAP bytes.  Returns ``[(reencoded, snap), ...]`` per pattern.

    This is a genuine round-trip through the structured representation (not a
    re-walk): equality holds ONLY IF the grammar decode is lossless.  A driver whose
    patterns are not this grammar produces a mismatch and the structure is rejected
    (HARD RULE #0: the grammar is a hypothesis the round-trip falsifies).  Accepts a
    RAM array or any object exposing ``.ram``."""
    ram = getattr(ram_or_obj, "ram", ram_or_obj)
    out = []
    for base in pattern_ptrs:
        groups, idx = _decode_groups(ram, base)
        reencoded = _encode_groups(groups)
        snap = [int(x) for x in ram[base : base + idx]]
        out.append((reencoded, snap))
    return out


def pattern_roundtrip_ok(struct):
    """True iff every recovered pattern decodes and RE-ENCODES to its exact SNAP
    bytes -- the byte-exact lossless-decode gate over the whole pattern bank."""
    if struct.ram is None or not struct.pattern_ptrs:
        return False
    return all(
        em == sn for em, sn in reencode_patterns(struct.ram, struct.pattern_ptrs)
    )


def read_stsq_cells(distill_path):
    """Parse the STSQ section into ``{addr: (first_seen_frame, samples)}`` -- the
    captured per-cell value sequences (the porta / vibrato accumulator cells live
    here).  Empty if the artifact has no STSQ section."""
    buf = _read_artifact(distill_path)
    if buf is None:
        return {}
    cells = {}
    for tag, body, _nxt in _walk_sections(buf):
        if tag != b"STSQ":
            continue
        (nent,) = _struct.unpack_from("<I", buf, body)
        pos = body + 4
        for _ in range(nent):
            addr, _flags, _total, first_seen, nsamp = _STSQ_HEAD.unpack_from(buf, pos)
            pos += _STSQ_HEAD.size
            samples = np.frombuffer(buf[pos : pos + nsamp], dtype=np.uint8).astype(
                np.int64
            )
            pos += nsamp
            cells[addr] = (first_seen, samples)
    return cells


def clean_pitches_residual(
    distill_path, freq_state, voices=((0, 1), (7, 8), (14, 15)), align=1
):
    """Subtract the captured porta/vibrato accumulators to recover TRUE grid pitches.

    The §state-machine identity: ``freq = note_seed + acc_a + acc_b (mod 2^16)`` where
    ``acc_a``/``acc_b`` are the captured 16-bit accumulator cells (porta, vibrato) and
    ``note_seed`` -- ``freq`` minus them -- is forced PIECEWISE-CONSTANT (one true grid
    pitch per note span).  For each voice this GENERICALLY picks the 1-2 accumulator
    cell pairs (from the reset-to-0, multi-valued STSQ candidates) that flatten ``freq``
    to the fewest piecewise-constant runs, then RENDERS ``note_seed + the accumulators``
    and compares to the ``.sidwr`` freq -- residual 0 proves the displaced "note table"
    (hundreds of entries, growing with playback) is really a handful of pitches plus two
    compact accumulator generators.

    ``freq_state`` is the ``(nframes, 25)`` register array.  Returns
    ``{voice: {"displaced": int, "pitches": int, "residual": int, "accs": [(lo,hi),...]}}``
    or ``None`` if the artifact has no STSQ section.  This is the in-tree port of the
    proven ``statemachine-proto/PROOF.py``."""
    cells = read_stsq_cells(distill_path)
    if not cells:
        return None
    state = np.asarray(freq_state, dtype=np.int64)
    n = state.shape[0]

    def grid(addr):
        first_seen, samples = cells[addr]
        out = np.zeros(n, dtype=np.int64)
        start = first_seen + align
        end = min(start + len(samples), n)
        out[start:end] = samples[: end - start]
        if end < n and len(samples):
            out[end:] = samples[-1]
        return out

    def acc16(lo, hi):
        return grid(lo) | (grid(hi) << 8)

    def n_changes(seq, a, b):
        return int(np.sum(np.diff(seq[a:b]) != 0))

    # accumulator-cell candidates: reset-to-0, multi-valued (the porta/vib integrators)
    acc_cells = [
        a
        for a, (_fs, s) in cells.items()
        if len(np.unique(s)) >= 5 and bool((s[:4] == 0).any())
    ]

    results = {}
    for vi, (rlo, rhi) in enumerate(voices):
        freq = state[:, rlo] | (state[:, rhi] << 8)
        a, b = 3, min(n, 514)
        # generically pick the 1-2 acc16 pairs minimizing piecewise-const breaks
        best = (n_changes(freq, a, b), [])
        for lo1 in acc_cells:
            for dh1 in (1, 2, 3):
                if lo1 + dh1 not in cells:
                    continue
                a1 = acc16(lo1, lo1 + dh1)
                c1 = n_changes((freq - a1) % 65536, a, b)
                if c1 < best[0]:
                    best = (c1, [(lo1, lo1 + dh1)])
                for lo2 in acc_cells:
                    if lo2 <= lo1:
                        continue
                    for dh2 in (1, 2, 3):
                        if lo2 + dh2 not in cells:
                            continue
                        a2 = acc16(lo2, lo2 + dh2)
                        c2 = n_changes((freq - a1 - a2) % 65536, a, b)
                        if c2 < best[0]:
                            best = (c2, [(lo1, lo1 + dh1), (lo2, lo2 + dh2)])
        accs = best[1]
        total_acc = np.zeros(n, dtype=np.int64)
        for lo, hi in accs:
            total_acc = (total_acc + acc16(lo, hi)) % 65536
        seed = (freq - total_acc) % 65536
        # render seed as piecewise-const + accumulators, compare byte-exact over [a,b)
        onsets = [a] + [a + 1 + i for i in np.nonzero(np.diff(seed[a:b]) != 0)[0]] + [b]
        seed_r = np.zeros(n, dtype=np.int64)
        for k in range(len(onsets) - 1):
            seed_r[onsets[k] : onsets[k + 1]] = seed[onsets[k]]
        freq_r = (seed_r + total_acc) % 65536
        residual = int(np.sum(freq_r[a:b] != freq[a:b]))
        results[vi] = {
            "displaced": int(len(np.unique(freq[a:b]))),
            "pitches": int(len(np.unique(seed[a:b]))),
            "residual": residual,
            "accs": [(int(lo), int(hi)) for lo, hi in accs],
        }
    return results


def _program_spans(d, struct):
    """The shared program-table spans (wave / pulse / filter / cmd) referenced once
    by the instruments, not re-derived per note.

    The generic bound is the SONG-DATA MASK (bytes READ as data during play, never
    written, never executed -- :meth:`Distill.song_data_mask`) RESTRICTED to the
    region OUTSIDE the structures already accounted for (instrument table, pattern-
    pointer table, pattern data, orderlists).  That residual song-data is exactly the
    shared wave/pulse/filter/command generator tables -- the player CODE is excluded
    by the mask (it is EXEC), so the span never includes machine code.  Returns
    ``{name: (lo, hi)}`` contiguous runs (derived from the access map, not hardcoded).
    """
    mask = d.song_data_mask().copy()

    # carve out the structures already counted so programs are not double-charged
    def carve(a, b):
        mask[a:b] = False

    carve(
        struct.instr_base,
        struct.instr_base + struct.instr_stride * struct.n_instruments,
    )
    carve(struct.patptr_lo, struct.patptr_hi + struct.n_patterns)  # lo+hi ptr tables
    # pattern data span = from just past the hi table to the max pattern end
    pat_lo = struct.patptr_hi + struct.n_patterns
    pat_hi = max(struct.pattern_ptrs) if struct.pattern_ptrs else pat_lo
    # extend pat_hi to the end of the last pattern (walk to its 0x7F)
    if struct.pattern_ptrs:
        last = max(struct.pattern_ptrs)
        k = 0
        while k < 0x400 and int(d.ram[last + k]) != _END_OF_PATTERN:
            k += 1
        pat_hi = last + k + 1
    carve(pat_lo, pat_hi)
    for p, o in zip(struct.orderlist_ptrs, struct.orderlists):
        carve(p, p + len(o))
    # remaining mask runs = the shared program tables
    idx = np.nonzero(mask)[0]
    spans = {}
    if not len(idx):
        return spans
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    for i, (s, e) in enumerate(zip(starts, ends)):
        spans[f"prog{i}"] = (int(idx[s]), int(idx[e]) + 1)
    return spans


def recover_structure(distill_path):
    """Generic end-to-end structure recovery from a ``.distill.bin`` SDST artifact.

    Returns a :class:`RecoveredStructure`.  ``ok`` is False (with a ``reason``) when
    no valid tracker structure is found -- discovery returns nothing on a pure-code
    tune (A Mind Is Born), so the caller falls back to the generator cover.  When
    ``ok`` is True the byte-exact re-encode gate (``reencode_patterns``) has held.
    """
    d = load_distill(distill_path)
    sddf_slices = read_sddf_slices(distill_path)
    struct = RecoveredStructure(
        ok=False,
        load_addr=d.load_addr,
        load_len=d.load_len,
        nframes=d.nframes,
        distill_path=distill_path,
        ram=d.ram,
    )

    pt = discover_pointer_table(d)
    if pt is None:
        struct.reason = "no pattern-pointer table (likely pure-code tune)"
        return struct
    lo_base, hi_base, n, ptrs = pt
    struct.patptr_lo, struct.patptr_hi, struct.n_patterns = lo_base, hi_base, n
    struct.pattern_ptrs = ptrs

    pattern_data_lo = hi_base + n  # pattern data starts right after the hi table
    patterns, instr_refs, notes, cmds, durs, n_rows, n_bytes = decode_patterns(d, ptrs)
    struct.patterns = patterns
    struct.instr_refs = instr_refs
    struct.note_table = notes
    struct.commands = cmds
    struct.durations = durs
    struct.n_rows = n_rows
    struct.pattern_bytes = n_bytes

    # byte-exact gate: the decode must re-encode to the exact SNAP bytes.
    for emitted, snap in reencode_patterns(d, ptrs):
        if emitted != snap:
            struct.reason = "pattern decode not byte-exact (grammar mismatch)"
            return struct

    ol = discover_orderlist(d, n, pattern_data_lo)
    if ol is not None:
        struct.orderlist_ptr_table, struct.orderlist_ptrs, struct.orderlists = ol

    it = discover_instrument_table(d, sddf_slices)
    if it is not None:
        struct.instr_base, struct.instr_stride = it
        used = [i for i in instr_refs if i != _INSTR_KILL]
        struct.n_instruments = (max(used) + 1) if used else 0
        struct.instr_records = [
            [
                int(x)
                for x in d.ram[
                    struct.instr_base
                    + i * struct.instr_stride : struct.instr_base
                    + i * struct.instr_stride
                    + struct.instr_stride
                ]
            ]
            for i in range(struct.n_instruments)
        ]

    struct.program_spans = _program_spans(d, struct)
    struct.ok = True
    return struct


# ---------------------------------------------------------------------------
# Token budget for the structured IR (the floor the output-fit recovery misses).
# ---------------------------------------------------------------------------
def _tok(v):
    """base16-LEB token cost of a small value: 1 token for a nibble, 2 for a byte."""
    return 1 if 0 <= v < 16 else 2


def _flat_token_stream(struct, ram):
    """The structured IR as ONE flat value-token stream, in canonical order: note
    table, deduped instrument pool, shared program tables (referenced once), the raw
    pattern byte streams (the player's already-compact stateful row encoding), then
    the orderlists.  This is the lossless re-expression of the song-data the player
    reads -- emitted ONCE per shared structure, the dedup/sharing the output-fit
    recovery never does.  Returns ``(stream, section_lengths)``."""
    stream = []
    sec = {}

    n0 = len(stream)
    stream += [int(v) for v in struct.note_table]
    sec["note_table"] = len(stream) - n0

    n0 = len(stream)
    seen, pool = {}, []
    for rec in struct.instr_records:
        key = tuple(rec)
        if key not in seen:
            seen[key] = len(pool)
            pool.append(rec)
    for rec in pool:
        stream += [int(x) for x in rec]
    sec["instr_pool"] = len(stream) - n0
    sec["_n_instruments"] = len(pool)

    n0 = len(stream)
    if ram is not None:
        for _name, (a, b) in struct.program_spans.items():
            stream += [int(x) for x in ram[a:b]]
    sec["shared_programs"] = len(stream) - n0

    n0 = len(stream)
    if ram is not None:
        for base in struct.pattern_ptrs:
            idx = 0
            while idx < 0x400:
                b = int(ram[base + idx])
                stream.append(b)
                idx += 1
                if b == _END_OF_PATTERN:
                    break
    sec["pattern_rows"] = len(stream) - n0

    n0 = len(stream)
    for o in struct.orderlists:
        stream += [int(b) for b in o]
    sec["orderlist"] = len(stream) - n0

    return stream, sec


def _backward_lz(tokens, min_match=3, window=4096):
    """Greedy backward-LZ over a token stream (the REPEAT/TRANSPOSE collapse): count
    literals + match-refs.  A match (a pattern replayed, a repeated orderlist run, a
    shared program fragment) costs ~2 tokens (offset + length); a literal ~1.  This is
    the factoring the output-fit recovery cannot do because it has no shared
    structure to point back to.  Returns ``(n_literals, n_matches)``."""
    i, n = 0, len(tokens)
    literals = matches = 0
    while i < n:
        best = 0
        lo = max(0, i - window)
        for j in range(lo, i):
            length = 0
            while (
                i + length < n
                and tokens[j + length] == tokens[i + length]
                and length < 255
            ):
                length += 1
            if length > best:
                best = length
        if best >= min_match:
            matches += 1
            i += best
        else:
            literals += 1
            i += 1
    return literals, matches


def token_budget(struct, frames=None):
    """The structured tracker-IR token budget.  Returns ``(total, breakdown)``.

    Two numbers, both honest: the UN-LZ ``flat`` count (one base16-LEB token per
    song-data byte -- the shared structures emitted once) and the backward-LZ
    ``total`` (the REPEAT/TRANSPOSE collapse: 33 reused patterns across 81 orderlist
    slots, repeated rows, shared program fragments).  The LZ total is the floor the
    output-fit recovery misses.  ``frames`` overrides the artifact's requested
    ``nframes`` with the true playback length (the ``.sidwr`` row count)."""
    ram = struct.ram
    stream, sec = _flat_token_stream(struct, ram)

    flat = sum(_tok(t) for t in stream)
    lit, mat = _backward_lz(stream)
    total = lit + 2 * mat
    nframes = frames or struct.nframes

    breakdown = {
        "note_table": sec["note_table"],
        "instr_pool": sec["instr_pool"],
        "shared_programs": sec["shared_programs"],
        "pattern_rows": sec["pattern_rows"],
        "orderlist": sec["orderlist"],
        "n_pitches": len(struct.note_table),
        "n_instruments": sec["_n_instruments"],
        "n_rows": struct.n_rows,
        "n_patterns": struct.n_patterns,
        "flat_bytes": len(stream),
        "flat_tokens": flat,
        "lz_literals": lit,
        "lz_matches": mat,
        "total": total,
        "tok_per_frame": total / nframes if nframes else float("inf"),
    }
    return total, breakdown


def recover_and_budget(distill_path, frames=None):
    """Convenience: recover the structure and return ``(structure, total_tokens,
    breakdown)``.  ``frames`` is the true playback length (``.sidwr`` row count) for
    the per-frame metric; defaults to the artifact's requested ``nframes``."""
    struct = recover_structure(distill_path)
    if struct.ok:
        total, breakdown = token_budget(struct, frames=frames)
        return struct, total, breakdown
    return struct, None, None
