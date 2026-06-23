"""Read the compact SDST artifact preframr-sidtrace DISTILLS in the emulator.

The old pipeline streamed the cycle-by-cycle CPU bus (GBs/tune) to ``.bus.bin``
and partitioned it offline -- petabytes of I/O at 60,000 tunes.  preframr-sidtrace
now does the analysis IN C++ during emulation (``membus_trace.h``) and emits ONE
few-KB ``<prefix>.distill.bin`` per tune.  This module parses that artifact into a
small dataclass the SMC-correct recovery (:mod:`identity`) consumes -- there is no
raw trace to read.

SDST format (little-endian; documented in ``preframr-sidtrace/src/sidtrace.cpp``)::

    magic   "SDST"   | version u16 | reserved u16
    init u16 | play u16 | load u16 | subtune u16 | nframes u16 | reserved u16
    cycles_per_frame u32 | t0_cycle i64
    sections, each: tag char[4] then a tag-specific body, terminated by "END\\0":
      ACMP  nbytes u32, then (run u16, bits u8) pairs over 0..65535: the
            per-address ACCESS-TYPE map (OR of EXEC/READ/WRITE x INIT/PLAY).
      SNAP  nbytes u32, then (addr u16, len u16, bytes[len]) runs: the
            post-init RAM snapshot for the SONG-DATA region only (RAM, read as
            data during play, never written during play, never executed).
      SIDW  nentries u32, then (pc u16, reg u8, _pad u8, count u32, lastVal u8,
            _pad u8[3]) -- PC-tagged SID-write summary (voice-lane attribution).
      IDXR  nentries u32, then (pc u16, base u16, stride i32, idxMin u8,
            idxMax u8, _pad u16, count u32) -- indexed-read VSA summary.
      SDDF  nentries u32, then per-write data-flow slice summaries (design 3.1):
            a fixed head + (slice_pcs, leaves, op_seq) variable arrays, optionally
            followed by an SDCU mid-call value sequence (nValSeq u16 + bytes).
      SDCU  same entry layout as SDDF, keyed by state-cell address: the per-cell
            UPDATE DAG, carrying the mid-call value sequence (design 2.5/2.7) the
            host feeds to Berlekamp-Massey for the LFSR-vs-not verdict.
      STSQ  nentries u32, then (addr u16, flags u8, _pad u8, total u32,
            firstSeen u32, nSamples u16, bytes[nSamples]) -- inter-frame
            state-cell sample sequences (design 3.2).

This reader consumes only ACMP/SNAP/SIDW/IDXR (the access map, RAM snapshot, and
write/index summaries the SMC-correct :mod:`identity` recovery needs); the
data-flow sections (SDDF/SDCU/STSQ) added by the current tracer are parsed past
structurally so the reader stays byte-compatible with both pre- and post-data-flow
artifacts -- the authoritative full reader is the decompiler-backend ``sdst.py``.

The access bits are the SMC-correct classifier: ``SMC = EXEC_PLAY & WRITE_PLAY``;
``song-data = READ_PLAY & ~WRITE_PLAY & ~EXEC`` (code modification, not data).
"""

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"SDST"

# Per-address access-type bits (mirror membus_trace.h AccBits).
ACC_EXEC_INIT = 1 << 0
ACC_READ_INIT = 1 << 1
ACC_WRITE_INIT = 1 << 2
ACC_EXEC_PLAY = 1 << 3
ACC_READ_PLAY = 1 << 4
ACC_WRITE_PLAY = 1 << 5

_SIDW_ENTRY = struct.Struct("<HBBIB3x")  # pc, reg, pad, count, lastVal, pad[3]
_IDXR_ENTRY = struct.Struct("<HHiBBHI")  # pc, base, stride, idxMin, idxMax, pad, count
# SDDF/SDCU per-write data-flow slice head (design 3.1) and one slice leaf.
_SIDDF_HEAD = struct.Struct("<HBBIBBBxHiBB")
_SIDDF_LEAF = struct.Struct("<BxHBx")
# STSQ per-cell inter-frame sample-sequence head (design 3.2).
_STSQ_HEAD = struct.Struct("<HBxIIH")

# Section tags a well-formed stream may begin a section with (used to
# disambiguate the optional SDDF/SDCU val_seq field structurally -- see below).
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


@dataclass
class SidWrite:
    """One PC-tagged SID-register-write summary entry (voice-lane attribution)."""

    pc: int
    reg: int
    count: int
    last_val: int


@dataclass
class IdxRead:
    """One indexed-read VSA summary entry: a table the player walks during play."""

    pc: int
    base: int
    stride: int
    idx_min: int
    idx_max: int
    count: int

    @property
    def length(self):
        """Element count the traversal swept (``idx_max - idx_min + 1``)."""
        return self.idx_max - self.idx_min + 1


@dataclass
class Distill:
    """The parsed SDST artifact: the in-emulator distillation of one tune."""

    version: int
    init_addr: int
    play_addr: int
    load_addr: int
    subtune: int
    nframes: int
    cycles_per_frame: int
    t0_cycle: int
    load_len: int  # loaded program-image length (song data lives in this span)
    acc: np.ndarray  # uint8[65536] per-address AccBits
    ram: np.ndarray  # uint8[65536] post-init RAM (song-data region; 0 elsewhere)
    sid_writes: list = field(default_factory=list)  # list[SidWrite]
    idx_reads: list = field(default_factory=list)  # list[IdxRead]

    # --- the SMC-correct access-type classifier ---
    def exec_mask(self):
        """Addresses fetched as code in either phase (instruction-fetch)."""
        return (self.acc & (ACC_EXEC_INIT | ACC_EXEC_PLAY)) != 0

    def smc_mask(self):
        """Self-modifying code: a location BOTH executed AND written during play.

        SMC operand writes are code modification, never song data; this is the
        access-TYPE definition (not the fragile write-set subtraction)."""
        return ((self.acc & ACC_EXEC_PLAY) != 0) & ((self.acc & ACC_WRITE_PLAY) != 0)

    def eligible_mask(self):
        """Bytes ELIGIBLE to be song data: RAM, never written during play, never
        executed -- the SMC-correct "not code, not mutable state" filter.  SMC
        locations (EXEC & WRITE) and plain code (EXEC) are both excluded here by
        access TYPE, not by write-set subtraction."""
        in_ram = np.zeros(65536, dtype=bool)
        in_ram[0x0002:0xD000] = True
        # Bound to the loaded program image -- the song tables live in the
        # player's own image (HVSC contract); this drops stray reads of
        # untouched high RAM. ``load_len == 0`` means "whole RAM" (unknown span).
        if self.load_len:
            img = np.zeros(65536, dtype=bool)
            end = min(self.load_addr + self.load_len, 65536)
            img[self.load_addr : end] = True
            in_ram = in_ram & img
        write_play = (self.acc & ACC_WRITE_PLAY) != 0
        return in_ram & ~write_play & ~self.exec_mask()

    def song_data_mask(self):
        """The song-data region: maximal :meth:`eligible_mask` runs that contain at
        least one byte actually READ as data during play.

        Read-coverage is partial in a finite capture (not every pattern/instrument
        is traversed), so we keep the GAPS inside a read-containing eligible run --
        the bytes between traversed table entries are genuine song data the player
        will read on a later pass, lifted verbatim from RAM (HARD RULE #0).  This
        matches the emulator's SNAP gating, so the snapshot already holds exactly
        these bytes; the mask is here for the recovery + the round-trip."""
        elig = self.eligible_mask()
        read_play = (self.acc & ACC_READ_PLAY) != 0
        out = np.zeros(65536, dtype=bool)
        idx = np.nonzero(elig)[0]
        if not len(idx):
            return out
        # contiguous eligible runs
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(idx) - 1]))
        for s, e in zip(starts, ends):
            lo, hi = int(idx[s]), int(idx[e])
            if read_play[lo : hi + 1].any():
                out[lo : hi + 1] = True
        return out


def _unrle_acc(body):
    """Expand the ACMP (run u16, bits u8) stream into a uint8[65536] array."""
    acc = np.zeros(65536, dtype=np.uint8)
    off = 0
    addr = 0
    n = len(body)
    while off + 3 <= n and addr < 65536:
        run = body[off] | (body[off + 1] << 8)
        bits = body[off + 2]
        off += 3
        acc[addr : addr + run] = bits
        addr += run
    return acc


def _unpack_snap(body):
    """Expand the SNAP (addr u16, len u16, bytes[len]) runs into uint8[65536]."""
    ram = np.zeros(65536, dtype=np.uint8)
    off = 0
    n = len(body)
    while off + 4 <= n:
        addr = body[off] | (body[off + 1] << 8)
        length = body[off + 2] | (body[off + 3] << 8)
        off += 4
        ram[addr : addr + length] = np.frombuffer(
            body[off : off + length], dtype=np.uint8
        )
        off += length
    return ram


def _at_section_boundary(buf, off):
    """True iff ``off`` is EOF or points at a recognized 4-byte section tag."""
    if off == len(buf):
        return True
    return buf[off : off + 4] in _SECTION_TAGS


def _skip_siddf_section(buf, off, nent, with_val_seq):
    """Walk past ``nent`` SDDF/SDCU entries starting at ``off`` without retaining
    them (this reader does not consume the data-flow sections).

    Returns the offset just past the section, or ``None`` if the layout does not
    fit the buffer (so the caller can retry with the other ``with_val_seq``
    choice). The only layout difference is the trailing per-entry val_seq
    (nValSeq u16 + bytes), absent in pre-data-flow artifacts."""
    try:
        for _ in range(nent):
            off += _SIDDF_HEAD.size
            (npcs,) = struct.unpack_from("<H", buf, off)
            off += 2 + 2 * npcs
            (nleaves,) = struct.unpack_from("<H", buf, off)
            off += 2 + _SIDDF_LEAF.size * nleaves
            (nops,) = struct.unpack_from("<H", buf, off)
            off += 2 + nops
            if with_val_seq:
                (nval,) = struct.unpack_from("<H", buf, off)
                off += 2 + nval
    except struct.error:
        return None
    if off > len(buf):
        return None
    return off


def load_distill(path):
    """Parse a ``<prefix>.distill.bin`` SDST artifact into a :class:`Distill`."""
    with open(path, "rb") as handle:
        buf = handle.read()
    return parse_distill(buf)


def parse_distill(buf):
    """Parse SDST bytes into a :class:`Distill` (the in-memory round-trip entry)."""
    if buf[:4] != MAGIC:
        raise ValueError(f"not an SDST artifact (magic {buf[:4]!r})")
    off = 4
    version, _res = struct.unpack_from("<HH", buf, off)
    off += 4
    init, play, load, subtune, nframes, _res2 = struct.unpack_from("<6H", buf, off)
    off += 12
    (cpf,) = struct.unpack_from("<I", buf, off)
    off += 4
    (t0,) = struct.unpack_from("<q", buf, off)
    off += 8
    (load_len,) = struct.unpack_from("<I", buf, off)
    off += 4

    acc = np.zeros(65536, dtype=np.uint8)
    ram = np.zeros(65536, dtype=np.uint8)
    sid_writes = []
    idx_reads = []

    while off < len(buf):
        tag = buf[off : off + 4]
        off += 4
        if tag == b"END\x00":
            break
        if tag in (b"ACMP", b"SNAP"):
            (nbytes,) = struct.unpack_from("<I", buf, off)
            off += 4
            body = buf[off : off + nbytes]
            off += nbytes
            if tag == b"ACMP":
                acc = _unrle_acc(body)
            else:
                ram = _unpack_snap(body)
        elif tag == b"SIDW":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, reg, _pad, count, last_val = _SIDW_ENTRY.unpack_from(buf, off)
                off += _SIDW_ENTRY.size
                sid_writes.append(SidWrite(pc, reg, count, last_val))
        elif tag == b"IDXR":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, base, stride, imin, imax, _pad, count = _IDXR_ENTRY.unpack_from(
                    buf, off
                )
                off += _IDXR_ENTRY.size
                idx_reads.append(IdxRead(pc, base, stride, imin, imax, count))
        elif tag in (b"SDDF", b"SDCU"):
            # Per-write/per-cell data-flow slices (design 3.1 / 2.2). This reader
            # does not consume them, but it must parse PAST them exactly so the
            # following sections still land. Two on-disk layouts coexist: the
            # current tracer appends a per-entry SDCU mid-call value sequence
            # (nValSeq u16 + bytes); older artifacts predate that field. The
            # header is v1 in both, so we DISAMBIGUATE structurally -- skip the
            # section assuming val_seq is present, and only if that fails to land
            # on a valid next section tag (or EOF) do we re-skip without it.
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            end_off = _skip_siddf_section(buf, off, nent, with_val_seq=True)
            if end_off is None or not _at_section_boundary(buf, end_off):
                end_off = _skip_siddf_section(buf, off, nent, with_val_seq=False)
            if end_off is None:
                raise ValueError(f"could not parse {tag!r} section")
            off = end_off
        elif tag == b"STSQ":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                _addr, _flags, _total, _first_seen, nsamp = _STSQ_HEAD.unpack_from(
                    buf, off
                )
                off += _STSQ_HEAD.size + nsamp
        else:
            raise ValueError(f"unknown SDST section {tag!r} at offset {off - 4}")

    return Distill(
        version=version,
        init_addr=init,
        play_addr=play,
        load_addr=load,
        subtune=subtune,
        nframes=nframes,
        cycles_per_frame=cpf,
        t0_cycle=t0,
        load_len=load_len,
        acc=acc,
        ram=ram,
        sid_writes=sid_writes,
        idx_reads=idx_reads,
    )


def build_distill(dist):
    """Serialise a :class:`Distill` back to SDST bytes (round-trip / test aid).

    Re-derives the ACMP/SNAP runs from ``acc``/``ram`` and the song-data mask, so
    ``parse_distill(build_distill(d))`` reproduces ``d`` for the fields the
    artifact carries.
    """
    out = bytearray()
    out += MAGIC
    out += struct.pack("<HH", dist.version, 0)
    out += struct.pack(
        "<6H",
        dist.init_addr,
        dist.play_addr,
        dist.load_addr,
        dist.subtune,
        dist.nframes,
        0,
    )
    out += struct.pack("<I", dist.cycles_per_frame)
    out += struct.pack("<q", dist.t0_cycle)
    out += struct.pack("<I", dist.load_len)

    # ACMP: RLE the access map.
    acmp = bytearray()
    acc = dist.acc
    addr = 0
    while addr < 65536:
        bits = int(acc[addr])
        run = 1
        while addr + run < 65536 and int(acc[addr + run]) == bits and run < 0xFFFF:
            run += 1
        acmp += struct.pack("<HB", run, bits)
        addr += run
    out += b"ACMP" + struct.pack("<I", len(acmp)) + acmp

    # SNAP: sparse runs over the song-data mask.
    snap = bytearray()
    mask = dist.song_data_mask()
    addr = 0
    while addr < 65536:
        if not mask[addr]:
            addr += 1
            continue
        start = addr
        while addr < 65536 and mask[addr] and (addr - start) < 0xFFFF:
            addr += 1
        snap += struct.pack("<HH", start, addr - start)
        snap += dist.ram[start:addr].tobytes()
    out += b"SNAP" + struct.pack("<I", len(snap)) + snap

    # SIDW.
    sidw = bytearray()
    for sw in dist.sid_writes:
        sidw += _SIDW_ENTRY.pack(sw.pc, sw.reg, 0, sw.count, sw.last_val)
    out += b"SIDW" + struct.pack("<I", len(dist.sid_writes)) + sidw

    # IDXR.
    idxr = bytearray()
    for ir in dist.idx_reads:
        idxr += _IDXR_ENTRY.pack(
            ir.pc, ir.base, ir.stride, ir.idx_min, ir.idx_max, 0, ir.count
        )
    out += b"IDXR" + struct.pack("<I", len(dist.idx_reads)) + idxr

    out += b"END\x00"
    return bytes(out)
