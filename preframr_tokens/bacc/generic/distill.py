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

The RECOVERY-OFFLOAD sections move the expensive Python table/orderlist/relocation
searches into the C++ emulator (it observes the facts for free). They are APPENDED
after SDCU, so ACMP/SNAP/SIDW/IDXR/SDDF/STSQ/SDCU stay byte-identical::

      IDXS  nentries u32 (one per IDXR pc), then (pc u16, flags u8, nSamp u8,
            scale i32, baseFit u16, feedsRegMask u32, samp0Idx u8, samp1Idx u8,
            samp0Addr u16, samp1Addr u16) -- the scaled-index affine fit
            (addr = baseFit + scale*idx, from two correct-index samples) + which
            $D4xx register(s) the table feeds + pointer-table signals (item #1/#5).
      PWLK  nentries u32, then (zp u16, flags u8, yMin u8, yMax u8, _pad u8,
            count u32, nAdv u16, nAdv*ptrVal u16, nAdv*advFrame u32) -- the
            (zp),Y pointer-walk value sequence = the resolved orderlist->pattern
            stream + per-advance frame onset (item #2 + the item #6 tempo events).
      RELO  nentries u32, then (storePc u16, srcReadPc u16, srcBase u16,
            dstBase u16, srcStride i32, dstStride i32, idxMin u8, idxMax u8,
            _pad u16, count u32) -- init block-copy: delta = dstBase-srcBase (#3).
      SDAC  nentries u32, then (pc u16, reg u8, _pad u8, nAddends u16,
            nAddends*(op u8, cell u16)) -- the SIDDF sites tagged accumulated
            (>=2 distinct PCs wrote the source shadow per call) + addends (#7).
      DIGI  (writesPerFrameMean*1000 u32, maxSubframeD418 u32,
            noteTableIdxrPresent u8, _pad u8[3], nFrames u32, nSidWrites u32) --
            the PCM-digi vs tracker write-density signature (item #4).
      TMPO  nCand u16, then nCand*(cell u16, reload u8, _pad u8) -- frame-divider
            reload constants (DEC counter; BNE; reload const) (item #6).

This reader parses ALL sections additively: the recovery-offload captures are
exposed on :class:`Distill`; the data-flow slices (SDDF/SDCU) are still parsed
past structurally (the authoritative full slice reader is the backend ``sdst.py``).

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

# ALU op codes (mirror membus_trace.h AluOp) -- the op of an accumulator addend
# (SDAC) or a state-cell update. Exposed so the recovery can read the addend ops.
(
    ALU_NONE,
    ALU_ADC,
    ALU_SBC,
    ALU_AND,
    ALU_ORA,
    ALU_EOR,
    ALU_ASL,
    ALU_LSR,
    ALU_ROL,
    ALU_ROR,
    ALU_INC,
    ALU_DEC,
    ALU_CMP,
) = range(13)
ALU_NAME = {
    ALU_NONE: "NONE",
    ALU_ADC: "ADC",
    ALU_SBC: "SBC",
    ALU_AND: "AND",
    ALU_ORA: "ORA",
    ALU_EOR: "EOR",
    ALU_ASL: "ASL",
    ALU_LSR: "LSR",
    ALU_ROL: "ROL",
    ALU_ROR: "ROR",
    ALU_INC: "INC",
    ALU_DEC: "DEC",
    ALU_CMP: "CMP",
}

# Leaf kinds (mirror membus_trace.h LeafKind) -- used by the SDDF/SDCU slices.
LK_IMMEDIATE, LK_RAM_READ, LK_STATE_CELL, LK_EXOGENOUS, LK_OUT_OF_WINDOW = range(5)

_SIDW_ENTRY = struct.Struct("<HBBIB3x")  # pc, reg, pad, count, lastVal, pad[3]
_IDXR_ENTRY = struct.Struct("<HHiBBHI")  # pc, base, stride, idxMin, idxMax, pad, count
# SDDF/SDCU per-write data-flow slice head (design 3.1) and one slice leaf.
_SIDDF_HEAD = struct.Struct("<HBBIBBBxHiBB")
_SIDDF_LEAF = struct.Struct("<BxHBx")
# STSQ per-cell inter-frame sample-sequence head (design 3.2).
_STSQ_HEAD = struct.Struct("<HBxIIH")

# Recovery-offload sections (the structural-capture additions; sidtrace
# emit_distill appends these after SDCU, leaving the sections above byte-exact).
_IDXS_ENTRY = struct.Struct(
    "<HBBiHIBBHH"
)  # pc,flags,nSamp,scale,baseFit,feedsMask,s0i,s1i,s0a,s1a
_PWLK_HEAD = struct.Struct("<HBBBBIH")  # zp,flags,yMin,yMax,_pad,count,nAdv
_RELO_ENTRY = struct.Struct(
    "<HHHHiiBBHI"
)  # storePc,srcReadPc,srcBase,dstBase,srcStride,dstStride,idxMin,idxMax,_pad,count
_SDAC_HEAD = struct.Struct("<HBBH")  # pc,reg,_pad,nAddends
_SDAC_ADDEND = struct.Struct("<BH")  # op,cell
# IWLK per-(pc,voice) instrument-table walk-index: the per-frame u8 index into the
# RAM instrument freq-table the freq-feeding IDXR pc walked (recovery-offload: the
# per-frame freq-modulation generator -- freq = note_base + instr_freq_table[index]).
_IWLK_HEAD = struct.Struct("<HBxI")  # pc, voice, _pad, nFrames
_DIGI_REC = struct.Struct(
    "<IIBBBBII"
)  # mean*1000,maxD418,noteTbl,_pad[3],nframes,nSidWrites
_TMPO_ENTRY = struct.Struct("<HBB")  # cell,reload,_pad

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
    b"IDXS",
    b"PWLK",
    b"RELO",
    b"SDAC",
    b"DIGI",
    b"TMPO",
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
class IdxSupp:
    """IDXR SUPPLEMENT entry (recovery-offload item #1 + #5): the scaled-index
    affine fit and the SID-register attribution for one indexed-read PC.

    Parallels the :class:`IdxRead` at the same ``pc``. ``scale``/``base_fit`` is
    the affine table fit (``addr = base_fit + scale*idx``) recovered from two
    distinct-index samples USING THE OPCODE'S ACTUAL INDEX REGISTER -- correct for
    an interleaved stride-2 (``note*2``) table the single-sample ``IdxRead.base``
    garbles. ``scale_set`` is False when the two samples did not fit an affine
    relation (an SMC-varying base); then trust ``IdxRead.base``. ``feeds_reg_mask``
    has bit ``r`` set when this table's values reach SID register ``$D4(r)``.
    ``targets_in_image``/``targets_read_as_data`` flag a pointer table (the 16-bit
    values at ``base+idx`` are themselves in-image addresses read as data)."""

    pc: int
    scale_set: bool
    scale: int
    base_fit: int
    feeds_reg_mask: int
    targets_in_image: bool
    targets_read_as_data: bool
    n_samp: int
    samp_idx: tuple  # (idx0, idx1)
    samp_addr: tuple  # (addr0, addr1)

    def feeds_reg(self, reg):
        """True iff this table's values reach SID register ``reg`` (0..24)."""
        return bool(self.feeds_reg_mask & (1 << reg))


@dataclass
class PtrWalk:
    """(zp),Y pointer-walk capture (recovery-offload item #2): the resolved
    orderlist->pattern stream for one zero-page pointer pair.

    ``ptr_vals`` is the SEQUENCE of distinct consecutive pointer values the pair
    took (each = a pattern start address the orderlist advanced to);
    ``adv_frames[i]`` is the frame (play-call) index of advance ``i`` (the row /
    note onset -- the item #6 tempo events); ``y_min``/``y_max`` is the per-call
    index range (the row span). Consecutive-equal pointer values are deduped into
    advance events. ``is_load``/``is_store`` record the access direction (0xB1 /
    0x91).

    ``adv_y[i]`` is the Y index (the AUTHORED orderlist POSITION) and ``adv_pc[i]``
    the PC of the ``(zp),Y`` read (the consuming VOICE proxy -- each voice reads the
    shared orderlist at its own play PC) at advance ``i`` (sidtrace #11). Together
    they let the recovery collapse the over-expanded per-voice/per-row pointer-read
    stream back to the single AUTHORED orderlist (the Y-walk ``0..y_max``) + its loop
    point (where Y wraps). EMPTY when read from a pre-#11 artifact (back-compat); the
    reader disambiguates the two on-disk layouts structurally."""

    zp: int
    is_load: bool
    is_store: bool
    y_min: int
    y_max: int
    count: int
    ptr_vals: list  # list[int]
    adv_frames: list  # list[int]
    adv_y: list = field(default_factory=list)  # list[int] (Y index per advance; #11)
    adv_pc: list = field(default_factory=list)  # list[int] (read PC per advance; #11)


@dataclass
class ReloCopy:
    """Init block-copy capture (recovery-offload item #3): one relocation copy
    loop observed during PHASE_INIT.

    ``delta = (dst_base - src_base) & 0xFFFF`` is the relocation offset the
    recovery needs to read image-relative tables (the table was copied from
    ``src_base`` in the image to ``dst_base`` at runtime). ``src_stride`` /
    ``dst_stride`` are the per-iteration element strides; ``idx_min``/``idx_max``
    bound the copy length."""

    store_pc: int
    src_read_pc: int
    src_base: int
    dst_base: int
    src_stride: int
    dst_stride: int
    idx_min: int
    idx_max: int
    count: int

    @property
    def delta(self):
        """Relocation offset ``dst_base - src_base`` (mod 16-bit)."""
        return (self.dst_base - self.src_base) & 0xFFFF

    @property
    def length(self):
        """Copy length the index span swept (``idx_max - idx_min + 1``)."""
        return self.idx_max - self.idx_min + 1


@dataclass
class SidAccum:
    """SIDDF ACCUMULATED tag (recovery-offload item #7): a ``$D4xx`` write whose
    source shadow is written >=2x per play-call by distinct PCs.

    ``addends`` is the list of contributing ``(op, cell)`` pairs -- the per-frame
    addend updates that compose the written value (``freq = base (+) vib_acc (+)
    porta_acc``). ``op`` is the ALU op of the contributing write (see ALU_*)."""

    pc: int
    reg: int
    addends: list  # list[(op, cell)]


@dataclass
class DigiSig:
    """Header / write-density signature (recovery-offload item #4).

    A PCM digi streamer writes ``$D418`` hundreds of times per frame and has no
    note-table IDXR; a tracker writes ~tens of registers/frame and indexes a
    96-entry note table. ``writes_per_frame_mean`` and ``max_subframe_d418`` are
    the density signals; ``note_table_idxr_present`` is the structural one."""

    writes_per_frame_mean: float
    max_subframe_d418: int
    note_table_idxr_present: bool
    n_frames: int
    n_sid_writes: int

    @property
    def is_digi(self):
        """Heuristic digi classification: PCM sub-frame streaming to $D418 with no
        note-table IDXR. (Advisory; the recovery's authoritative carve also weighs
        the player-name list.)"""
        return self.max_subframe_d418 >= 16 and not self.note_table_idxr_present


@dataclass
class TempoCand:
    """Frame-divider candidate (recovery-offload item #6): a state cell that
    decrements and reloads from an immediate (``DEC counter; BNE; reload const``).

    ``reload`` is the divider reload constant the host uses as the row tick rate;
    per-row advance frames are in the :class:`PtrWalk` ``adv_frames``."""

    cell: int
    reload: int


@dataclass
class StsqCell:
    """One STSQ inter-frame state-cell sample sequence (design 3.2): the per-frame
    value sequence of a RAM state cell (the porta / vibrato / sweep accumulators).

    ``samples`` is the captured value-per-frame sequence starting at ``first_seen``;
    the recovery feeds these to the accumulator-FIT (a stored ramp is unrecovered
    structure -- HARD RULE #0).  Spans the FULL playback (PR0 lifted the tracer's
    legacy ~512-frame window; the accumulator-fit now sees the whole run)."""

    addr: int
    flags: int
    first_seen: int
    samples: np.ndarray  # uint8 per-frame values

    @property
    def n_unique(self):
        """Distinct value count (a multi-valued reset-to-0 cell is an accumulator)."""
        return int(len(np.unique(self.samples)))


@dataclass
class IwlkWalk:
    """One IWLK instrument-table walk-index sequence (recovery-offload: the
    per-frame freq-modulation index).

    For one freq-feeding IDXR ``pc`` driving one ``voice`` (the freq-lo register
    index 0/7/14, i.e. the SID voice the write fed), ``index`` is the per-frame u8
    index the playroutine walked into the RAM instrument freq-table -- retriggered
    (reset) at note onset. The freq-mod fitter (PR-C) recovers
    ``freq_v = note_base_v + instr_freq_table[index_{v,frame}]`` from this, rather
    than leaning on the ``_state`` anchor (a stored per-frame freq sequence is
    unrecovered structure -- HARD RULE #0). Spans the full playback."""

    pc: int
    voice: int  # freq-lo reg / voice the IDXR write feeds (0,7,14)
    index: np.ndarray  # uint8 per-frame walk index


@dataclass
class SddfSlice:
    """One SID-write backward data-flow slice (design 3.1): the RAM_READ leaf
    addresses a ``$D4xx`` write's value flowed from (the table cells it indexed).

    A stride-K lattice of leaf addresses across the AD/SR/PW/ctrl write PCs IS the
    instrument table (cross-checked against the IDXR stride in S5)."""

    pc: int
    reg: int
    leaf_addrs: list  # RAM_READ leaf addresses (the read provenance)


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
    # Recovery-offload captures (the structural-capture additions; additive --
    # absent -> empty, so old artifacts parse unchanged).
    idx_supp: list = field(default_factory=list)  # list[IdxSupp]   (item #1/#5)
    ptr_walks: list = field(default_factory=list)  # list[PtrWalk]   (item #2)
    relo_copies: list = field(default_factory=list)  # list[ReloCopy] (item #3)
    sid_accum: list = field(default_factory=list)  # list[SidAccum]  (item #7)
    digi: object = None  # DigiSig or None                            (item #4)
    tempo_cands: list = field(default_factory=list)  # list[TempoCand] (item #6)
    # Data-flow sections (THIS reader now parses them, not just past them -- the ONE
    # artifact reader the generic recovery consumes; design §3 consolidation).
    stsq_cells: list = field(default_factory=list)  # list[StsqCell] (design 3.2)
    sddf_slices: list = field(default_factory=list)  # list[SddfSlice] (design 3.1)
    # Instrument-table freq-modulation walk-index (the per-frame index into the RAM
    # instrument freq-table). Additive: absent -> empty, so old artifacts parse
    # unchanged. Consumed by the freq-mod fitter (render freq from a recovered
    # generator + schedule, dropping the _state anchor).
    iwlk_walks: list = field(default_factory=list)  # list[IwlkWalk]

    def idx_supp_by_pc(self):
        """``{pc: IdxSupp}`` for cross-referencing IDXR entries with their
        scaled-index fit + register attribution (recovery-offload items #1/#5)."""
        return {s.pc: s for s in self.idx_supp}

    def stsq_by_addr(self):
        """``{addr: StsqCell}`` for cross-referencing accumulator cells by address."""
        return {c.addr: c for c in self.stsq_cells}

    def iwlk_by_pc_voice(self):
        """``{(pc, voice): IwlkWalk}`` for pairing a freq-feeding IDXR with its
        per-frame instrument-table walk-index (the freq-mod generator input)."""
        return {(w.pc, w.voice): w for w in self.iwlk_walks}

    def idxr_by_pc(self):
        """``{pc: IdxRead}`` for pairing an IDXR entry with its :class:`IdxSupp`."""
        return {r.pc: r for r in self.idx_reads}

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
        # Bound to the loaded program image -- the song tables live in the
        # player's own image (HVSC contract); this drops stray reads of
        # untouched high RAM. ``load_len == 0`` means "whole RAM" (unknown span).
        if self.load_len:
            # The image span is the authoritative RAM region. When the player
            # loads INTO the RAM banked under I/O ($d000-$dfff) / KERNAL/BASIC
            # ($e000-$ffff) -- e.g. MoN_Deenen @ $e800, Stephen_Ruddy @ $f000 --
            # the tracer's SNAP captures it from the UNDERLYING 64 KiB RAM (raw
            # ram[], banking-independent), so those bytes ARE in the artifact. We
            # therefore extend the eligible ceiling to $ffff and let the image
            # intersection gate the high region to exactly what was loaded. This is
            # ADDITIVE: an image loaded entirely < $d000 never overlaps $d000-$ffff,
            # so its eligible set is byte-identical to the old $d000-capped mask.
            in_ram[0x0002:0x10000] = True
            end = min(self.load_addr + self.load_len, 65536)
            in_ram[: max(self.load_addr, 0x0002)] = False
            in_ram[end:] = False
        else:
            # Unknown span ("whole RAM"): keep the conservative $d000 ceiling so the
            # I/O/ROM-banked region is NOT admitted without an image to bound it --
            # preserves the prior behavior exactly for the no-load-len case.
            in_ram[0x0002:0xD000] = True
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


def _skip_pwlk_section(buf, off, nent, with_adv_yp):
    """Walk past ``nent`` PWLK entries from ``off`` without retaining them, to choose
    the on-disk layout structurally (same idiom as :func:`_skip_siddf_section`).

    The post-#11 layout appends, per entry, ``nAdv*(advY u8) + nAdv*(advPc u16)``
    after the legacy ``nAdv*(ptrVal u16) + nAdv*(advFrame u32)`` arrays. Returns the
    offset just past the section, or ``None`` if the chosen layout does not fit the
    buffer (so the caller retries with the other ``with_adv_yp`` choice)."""
    try:
        for _ in range(nent):
            _zp, _fl, _ymin, _ymax, _pad, _count, nadv = _PWLK_HEAD.unpack_from(
                buf, off
            )
            off += _PWLK_HEAD.size + 2 * nadv + 4 * nadv  # ptrVals + advFrames
            if with_adv_yp:
                off += nadv + 2 * nadv  # advY (u8) + advPc (u16)
    except struct.error:
        return None
    if off > len(buf):
        return None
    return off


def _decode_siddf_slices(buf, off, nent, with_val_seq, out):
    """Decode ``nent`` SDDF entries' RAM_READ leaves into ``out`` (list[SddfSlice]).

    The layout (``with_val_seq`` chosen by :func:`_skip_siddf_section`) is the design
    3.1 head + (slice_pcs, leaves, op_seq) variable arrays + optional SDCU val_seq;
    only the head ``(pc, reg)`` and the RAM_READ leaf addresses are retained."""
    for _ in range(nent):
        head = _SIDDF_HEAD.unpack_from(buf, off)
        pc, reg = head[0], head[1]
        off += _SIDDF_HEAD.size
        (npcs,) = struct.unpack_from("<H", buf, off)
        off += 2 + 2 * npcs
        (nleaves,) = struct.unpack_from("<H", buf, off)
        off += 2
        leaf_addrs = []
        for _l in range(nleaves):
            kind, addr, _val = _SIDDF_LEAF.unpack_from(buf, off)
            off += _SIDDF_LEAF.size
            if kind == LK_RAM_READ:
                leaf_addrs.append(addr)
        (nops,) = struct.unpack_from("<H", buf, off)
        off += 2 + nops
        if with_val_seq:
            (nval,) = struct.unpack_from("<H", buf, off)
            off += 2 + nval
        out.append(SddfSlice(pc=pc, reg=reg, leaf_addrs=leaf_addrs))


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
    idx_supp = []
    ptr_walks = []
    relo_copies = []
    sid_accum = []
    digi = None
    tempo_cands = []
    stsq_cells = []
    sddf_slices = []
    iwlk_walks = []

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
            # Per-write/per-cell data-flow slices (design 3.1 / 2.2). This reader now
            # PARSES the SDDF RAM_READ leaves (the instrument-table provenance the
            # generic recovery uses) -- the ONE artifact reader (design §3
            # consolidation; the duplicate parser in structure_recover is removed).
            # Two on-disk layouts coexist: the current tracer appends a per-entry SDCU
            # mid-call value sequence (nValSeq u16 + bytes); older artifacts predate
            # that field.  The header is v1 in both, so we DISAMBIGUATE structurally:
            # skip assuming val_seq is present, and only if that fails to land on a
            # valid next section tag (or EOF) do we re-skip without it.
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            with_val = True
            end_off = _skip_siddf_section(buf, off, nent, with_val_seq=True)
            if end_off is None or not _at_section_boundary(buf, end_off):
                with_val = False
                end_off = _skip_siddf_section(buf, off, nent, with_val_seq=False)
            if end_off is None:
                raise ValueError(f"could not parse {tag!r} section")
            if tag == b"SDDF":
                _decode_siddf_slices(buf, off, nent, with_val, sddf_slices)
            off = end_off
        elif tag == b"STSQ":
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                addr, flags, _total, first_seen, nsamp = _STSQ_HEAD.unpack_from(
                    buf, off
                )
                off += _STSQ_HEAD.size
                samples = np.frombuffer(buf[off : off + nsamp], dtype=np.uint8).copy()
                off += nsamp
                stsq_cells.append(
                    StsqCell(
                        addr=addr, flags=flags, first_seen=first_seen, samples=samples
                    )
                )
        elif tag == b"IDXS":
            # IDXR SUPPLEMENT: scaled-index fit + reg attribution (item #1/#5).
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, flags, n_samp, scale, base_fit, mask, s0i, s1i, s0a, s1a = (
                    _IDXS_ENTRY.unpack_from(buf, off)
                )
                off += _IDXS_ENTRY.size
                idx_supp.append(
                    IdxSupp(
                        pc=pc,
                        scale_set=bool(flags & 0x01),
                        scale=scale,
                        base_fit=base_fit,
                        feeds_reg_mask=mask,
                        targets_in_image=bool(flags & 0x02),
                        targets_read_as_data=bool(flags & 0x04),
                        n_samp=n_samp,
                        samp_idx=(s0i, s1i),
                        samp_addr=(s0a, s1a),
                    )
                )
        elif tag == b"PWLK":
            # (zp),Y pointer-walk sequences (item #2 + the item #6 advance frames).
            # Two on-disk layouts coexist: the post-#11 tracer appends, per entry, the
            # per-advance advY (u8) + advPc (u16) arrays (the authored orderlist index +
            # consuming read PC); older artifacts predate them.  DISAMBIGUATE structurally
            # exactly as the SDDF reader does: assume the new arrays are present, and only
            # if that fails to land on a valid next section tag (or EOF) re-skip without
            # them (back-compat tolerance, sidtrace IWLK reader #152 idiom).
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            with_adv_yp = True
            end_off = _skip_pwlk_section(buf, off, nent, with_adv_yp=True)
            if end_off is None or not _at_section_boundary(buf, end_off):
                with_adv_yp = False
                end_off = _skip_pwlk_section(buf, off, nent, with_adv_yp=False)
            if end_off is None:
                raise ValueError("could not parse PWLK section")
            for _ in range(nent):
                zp, flags, ymin, ymax, _pad, count, nadv = _PWLK_HEAD.unpack_from(
                    buf, off
                )
                off += _PWLK_HEAD.size
                ptrs = list(struct.unpack_from(f"<{nadv}H", buf, off))
                off += 2 * nadv
                frames = list(struct.unpack_from(f"<{nadv}I", buf, off))
                off += 4 * nadv
                adv_y, adv_pc = [], []
                if with_adv_yp:
                    adv_y = list(struct.unpack_from(f"<{nadv}B", buf, off))
                    off += nadv
                    adv_pc = list(struct.unpack_from(f"<{nadv}H", buf, off))
                    off += 2 * nadv
                ptr_walks.append(
                    PtrWalk(
                        zp=zp,
                        is_load=bool(flags & 0x01),
                        is_store=bool(flags & 0x02),
                        y_min=ymin,
                        y_max=ymax,
                        count=count,
                        ptr_vals=ptrs,
                        adv_frames=frames,
                        adv_y=adv_y,
                        adv_pc=adv_pc,
                    )
                )
        elif tag == b"RELO":
            # Init block-copy summaries (item #3).
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                spc, srpc, sb, db, ss, ds, imin, imax, _pad, count = (
                    _RELO_ENTRY.unpack_from(buf, off)
                )
                off += _RELO_ENTRY.size
                relo_copies.append(
                    ReloCopy(
                        store_pc=spc,
                        src_read_pc=srpc,
                        src_base=sb,
                        dst_base=db,
                        src_stride=ss,
                        dst_stride=ds,
                        idx_min=imin,
                        idx_max=imax,
                        count=count,
                    )
                )
        elif tag == b"SDAC":
            # SIDDF ACCUMULATED supplement (item #7).
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, reg, _pad, nadd = _SDAC_HEAD.unpack_from(buf, off)
                off += _SDAC_HEAD.size
                addends = []
                for _ in range(nadd):
                    op, cell = _SDAC_ADDEND.unpack_from(buf, off)
                    off += _SDAC_ADDEND.size
                    addends.append((op, cell))
                sid_accum.append(SidAccum(pc=pc, reg=reg, addends=addends))
        elif tag == b"DIGI":
            # Header / write-density signature (item #4). One fixed record.
            mean1k, maxd418, notetbl, _p0, _p1, _p2, fc, nwr = _DIGI_REC.unpack_from(
                buf, off
            )
            off += _DIGI_REC.size
            digi = DigiSig(
                writes_per_frame_mean=mean1k / 1000.0,
                max_subframe_d418=maxd418,
                note_table_idxr_present=bool(notetbl),
                n_frames=fc,
                n_sid_writes=nwr,
            )
        elif tag == b"TMPO":
            # Tempo / frame-divider candidates (item #6).
            (ncand,) = struct.unpack_from("<H", buf, off)
            off += 2
            for _ in range(ncand):
                cell, reload, _pad = _TMPO_ENTRY.unpack_from(buf, off)
                off += _TMPO_ENTRY.size
                tempo_cands.append(TempoCand(cell=cell, reload=reload))
        elif tag == b"IWLK":
            # Instrument-table freq-modulation walk-index: per-(pc,voice) per-frame
            # u8 index into the RAM instrument freq-table (the freq-mod generator
            # input the freq-mod fitter consumes). Additive -- absent on older
            # artifacts, so this branch is a no-op when the section is missing.
            (nent,) = struct.unpack_from("<I", buf, off)
            off += 4
            for _ in range(nent):
                pc, voice, nfr = _IWLK_HEAD.unpack_from(buf, off)
                off += _IWLK_HEAD.size
                index = np.frombuffer(buf[off : off + nfr], dtype=np.uint8).copy()
                off += nfr
                iwlk_walks.append(IwlkWalk(pc=pc, voice=voice, index=index))
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
        idx_supp=idx_supp,
        ptr_walks=ptr_walks,
        relo_copies=relo_copies,
        sid_accum=sid_accum,
        digi=digi,
        tempo_cands=tempo_cands,
        stsq_cells=stsq_cells,
        sddf_slices=sddf_slices,
        iwlk_walks=iwlk_walks,
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

    # SDDF: re-emit the RETAINED leaves (head + RAM_READ leaves only; empty
    # slice_pcs / op_seq / val_seq -- the reader decodes only (pc, reg) + leaves, so
    # this round-trips the data the recovery uses).  Written WITHOUT a val_seq, which
    # the structural disambiguation in the reader resolves.
    if dist.sddf_slices:
        sddf = bytearray()
        for sl in dist.sddf_slices:
            sddf += _SIDDF_HEAD.pack(sl.pc, sl.reg, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            sddf += struct.pack("<H", 0)  # nPcs
            sddf += struct.pack("<H", len(sl.leaf_addrs))  # nLeaves
            for addr in sl.leaf_addrs:
                sddf += _SIDDF_LEAF.pack(LK_RAM_READ, addr, 0)
            sddf += struct.pack("<H", 0)  # nOps
        out += b"SDDF" + struct.pack("<I", len(dist.sddf_slices)) + sddf

    # STSQ: the per-cell inter-frame sample sequences (the accumulator cells).
    if dist.stsq_cells:
        stsq = bytearray()
        for c in dist.stsq_cells:
            samples = np.asarray(c.samples, dtype=np.uint8)
            stsq += _STSQ_HEAD.pack(
                c.addr, c.flags, int(samples.sum()), c.first_seen, len(samples)
            )
            stsq += samples.tobytes()
        out += b"STSQ" + struct.pack("<I", len(dist.stsq_cells)) + stsq

    # Recovery-offload sections (round-trip only the ones present; the C++
    # emitter always writes them, so a parsed-then-rebuilt artifact reproduces
    # the captured structure these encode).
    idxs = bytearray()
    for s in dist.idx_supp:
        flags = (
            (0x01 if s.scale_set else 0)
            | (0x02 if s.targets_in_image else 0)
            | (0x04 if s.targets_read_as_data else 0)
        )
        idxs += _IDXS_ENTRY.pack(
            s.pc,
            flags,
            s.n_samp,
            s.scale,
            s.base_fit,
            s.feeds_reg_mask,
            s.samp_idx[0],
            s.samp_idx[1],
            s.samp_addr[0],
            s.samp_addr[1],
        )
    out += b"IDXS" + struct.pack("<I", len(dist.idx_supp)) + idxs

    pwlk = bytearray()
    for p in dist.ptr_walks:
        flags = (0x01 if p.is_load else 0) | (0x02 if p.is_store else 0)
        nadv = len(p.ptr_vals)
        pwlk += _PWLK_HEAD.pack(p.zp, flags, p.y_min, p.y_max, 0, p.count, nadv)
        pwlk += struct.pack(f"<{nadv}H", *p.ptr_vals)
        pwlk += struct.pack(f"<{nadv}I", *p.adv_frames)
        # Emit the post-#11 advY/advPc arrays only when present and length-consistent;
        # a PtrWalk read from a pre-#11 artifact has them empty and round-trips in the
        # OLD layout (the structural disambiguation in the reader picks the right one).
        if len(p.adv_y) == nadv and len(p.adv_pc) == nadv:
            pwlk += struct.pack(f"<{nadv}B", *p.adv_y)
            pwlk += struct.pack(f"<{nadv}H", *p.adv_pc)
    out += b"PWLK" + struct.pack("<I", len(dist.ptr_walks)) + pwlk

    relo = bytearray()
    for r in dist.relo_copies:
        relo += _RELO_ENTRY.pack(
            r.store_pc,
            r.src_read_pc,
            r.src_base,
            r.dst_base,
            r.src_stride,
            r.dst_stride,
            r.idx_min,
            r.idx_max,
            0,
            r.count,
        )
    out += b"RELO" + struct.pack("<I", len(dist.relo_copies)) + relo

    sdac = bytearray()
    for a in dist.sid_accum:
        sdac += _SDAC_HEAD.pack(a.pc, a.reg, 0, len(a.addends))
        for op, cell in a.addends:
            sdac += _SDAC_ADDEND.pack(op, cell)
    out += b"SDAC" + struct.pack("<I", len(dist.sid_accum)) + sdac

    if dist.digi is not None:
        d = dist.digi
        out += b"DIGI" + _DIGI_REC.pack(
            int(round(d.writes_per_frame_mean * 1000)),
            d.max_subframe_d418,
            1 if d.note_table_idxr_present else 0,
            0,
            0,
            0,
            d.n_frames,
            d.n_sid_writes,
        )

    tmpo = bytearray()
    for t in dist.tempo_cands:
        tmpo += _TMPO_ENTRY.pack(t.cell, t.reload, 0)
    out += b"TMPO" + struct.pack("<H", len(dist.tempo_cands)) + tmpo

    if dist.iwlk_walks:
        iwlk = bytearray()
        for w in dist.iwlk_walks:
            idx = np.asarray(w.index, dtype=np.uint8)
            iwlk += _IWLK_HEAD.pack(w.pc, w.voice, len(idx))
            iwlk += idx.tobytes()
        out += b"IWLK" + struct.pack("<I", len(dist.iwlk_walks)) + iwlk

    out += b"END\x00"
    return bytes(out)
