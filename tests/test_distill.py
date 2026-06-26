"""The compact SDST distill artifact: round-trip, the SMC-correct access-type
classifier, and the tiny-artifact-size gate.

Three layers:

* self-contained unit tests of the SDST parser/serialiser round-trip and the
  SMC-correct classifier on SYNTHETIC :class:`Distill` artifacts (always run; no
  binary, no fixtures) -- they pin that song data is read-as-data / never-written
  / never-executed, that SMC (executed AND written during play) is excluded from
  song data by access TYPE (not by write-set subtraction), and that a
  hand-built artifact survives ``parse(build(d)) == d`` for the carried fields;
* a SYNTHETIC self-modifying-code player proving the classification keeps the
  song data correct even when the player patches its own operands during play
  (the required SMC test); and
* a binary-gated tiny-artifact-size gate on the real ``Grid_Runner.sid``
  (skipped without ``SIDTRACE_BIN``), contrasting the few-KB distill with the
  retired multi-GB raw bus trace.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import distill as D


def _empty_distill(load_addr=0x1000, load_len=0x1000):
    return D.Distill(
        version=1,
        init_addr=load_addr,
        play_addr=load_addr + 3,
        load_addr=load_addr,
        subtune=1,
        nframes=100,
        cycles_per_frame=19705,
        t0_cycle=12345,
        load_len=load_len,
        acc=np.zeros(65536, dtype=np.uint8),
        ram=np.zeros(65536, dtype=np.uint8),
        sid_writes=[],
        idx_reads=[],
    )


def test_song_data_is_read_not_written_not_executed():
    d = _empty_distill()
    # $1100..$1103: a table the player READS during play (song data).
    for a in range(0x1100, 0x1104):
        d.acc[a] = D.ACC_READ_PLAY
        d.ram[a] = a & 0xFF
    # $1200: written during play (a SID shadow / scratch byte) -> NOT song data.
    d.acc[0x1200] = D.ACC_READ_PLAY | D.ACC_WRITE_PLAY
    # $1300: executed (player code) -> NOT song data even though also read.
    d.acc[0x1300] = D.ACC_READ_PLAY | D.ACC_EXEC_PLAY

    sm = d.song_data_mask()
    assert sm[0x1100] and sm[0x1103]
    assert not sm[0x1200]
    assert not sm[0x1300]


def test_smc_excluded_from_song_data_by_access_type():
    """The required SMC discipline: a location BOTH executed AND written during
    play is self-modifying code, classified SMC and excluded from song data --
    NOT because we subtract the write set, but because it is EXEC."""
    d = _empty_distill()
    # An SMC operand: code that is executed every frame AND patched in place.
    smc = 0x1180
    d.acc[smc] = D.ACC_EXEC_PLAY | D.ACC_WRITE_PLAY | D.ACC_READ_PLAY
    # Adjacent real song data, only read.
    d.acc[0x1190] = D.ACC_READ_PLAY
    d.ram[0x1190] = 0x42

    assert d.smc_mask()[smc]
    assert not d.song_data_mask()[smc]  # SMC is code, never song data
    assert d.song_data_mask()[0x1190]  # genuine data survives


def test_song_data_keeps_unreached_gaps_inside_a_read_run():
    """Read-coverage is partial in a finite capture; the gaps between traversed
    table entries inside a never-written / never-executed run are kept (genuine
    data the player reads on a later pass)."""
    d = _empty_distill()
    d.acc[0x1100] = D.ACC_READ_PLAY  # read
    # 0x1101..0x1102 eligible (no write, no exec) but not yet read in window
    d.acc[0x1103] = D.ACC_READ_PLAY  # read
    for a in range(0x1100, 0x1104):
        d.ram[a] = 0xA0 + (a & 0xF)
    sm = d.song_data_mask()
    assert sm[0x1100] and sm[0x1101] and sm[0x1102] and sm[0x1103]


def test_song_data_bounded_to_load_image():
    """A stray read of untouched high RAM outside the loaded image is NOT song
    data -- the region is bounded to the player's own image span."""
    d = _empty_distill(load_addr=0x1000, load_len=0x1000)  # [0x1000, 0x2000)
    d.acc[0x1100] = D.ACC_READ_PLAY  # inside image
    d.acc[0x8000] = D.ACC_READ_PLAY  # outside image -> dropped
    sm = d.song_data_mask()
    assert sm[0x1100]
    assert not sm[0x8000]


def test_build_parse_round_trip():
    d = _empty_distill()
    for a in range(0x1100, 0x1110):
        d.acc[a] = D.ACC_READ_PLAY
        d.ram[a] = (a * 7) & 0xFF
    d.acc[0x1080] = D.ACC_EXEC_PLAY | D.ACC_WRITE_PLAY  # an SMC site
    d.sid_writes = [D.SidWrite(pc=0x1234, reg=0, count=42, last_val=0x55)]
    d.idx_reads = [
        D.IdxRead(pc=0x1240, base=0x1100, stride=2, idx_min=0, idx_max=7, count=99)
    ]

    blob = D.build_distill(d)
    d2 = D.parse_distill(blob)

    assert d2.version == d.version
    assert d2.init_addr == d.init_addr
    assert d2.play_addr == d.play_addr
    assert d2.load_addr == d.load_addr
    assert d2.load_len == d.load_len
    assert d2.t0_cycle == d.t0_cycle
    assert np.array_equal(d2.acc, d.acc)
    # SNAP carries only the song-data bytes; they must survive verbatim.
    sm = d.song_data_mask()
    assert np.array_equal(d2.ram[sm], d.ram[sm])
    assert d2.sid_writes == d.sid_writes
    assert d2.idx_reads == d.idx_reads


def test_recovery_offload_sections_round_trip():
    """The recovery-offload captures (IDXS/PWLK/RELO/SDAC/DIGI/TMPO) survive
    parse(build(d)) == d, and their derived properties are correct. These are the
    structural-capture additions; they parse additively (absent -> empty)."""
    d = _empty_distill()
    d.idx_supp = [
        D.IdxSupp(
            pc=0x1240,
            scale_set=True,
            scale=2,
            base_fit=0x166D,
            feeds_reg_mask=(1 << 0) | (1 << 1),
            targets_in_image=False,
            targets_read_as_data=False,
            n_samp=2,
            samp_idx=(2, 4),
            samp_addr=(0x1671, 0x1675),
        ),
        D.IdxSupp(
            pc=0x1250,
            scale_set=False,
            scale=1,
            base_fit=0x1900,
            feeds_reg_mask=0,
            targets_in_image=True,
            targets_read_as_data=True,
            n_samp=1,
            samp_idx=(3, 0),
            samp_addr=(0x1903, 0),
        ),
    ]
    d.ptr_walks = [
        D.PtrWalk(
            zp=0xFB,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=76,
            count=5240,
            ptr_vals=[0x1A1B, 0x1AFC, 0x19F4],
            adv_frames=[1, 1, 2],
        )
    ]
    d.relo_copies = [
        D.ReloCopy(
            store_pc=0x37AF,
            src_read_pc=0x37AD,
            src_base=0x3FC4,
            dst_base=0xAACE,
            src_stride=1,
            dst_stride=1,
            idx_min=1,
            idx_max=255,
            count=1020,
        )
    ]
    d.sid_accum = [
        D.SidAccum(
            pc=0x160D, reg=0, addends=[(D.ALU_NONE, 0x1735), (D.ALU_NONE, 0x1792)]
        )
    ]
    d.digi = D.DigiSig(
        writes_per_frame_mean=17.765,
        max_subframe_d418=1,
        note_table_idxr_present=True,
        n_frames=2299,
        n_sid_writes=40843,
    )
    d.tempo_cands = [D.TempoCand(cell=0x1087, reload=2)]

    d2 = D.parse_distill(D.build_distill(d))

    assert d2.idx_supp == d.idx_supp
    assert d2.ptr_walks == d.ptr_walks
    assert d2.relo_copies == d.relo_copies
    assert d2.sid_accum == d.sid_accum
    assert d2.digi == d.digi
    assert d2.tempo_cands == d.tempo_cands
    # derived properties
    assert d2.idx_supp[0].feeds_reg(0) and d2.idx_supp[0].feeds_reg(1)
    assert not d2.idx_supp[0].feeds_reg(4)
    # the affine fit reproduces the samples exactly.
    s = d2.idx_supp[0]
    assert (s.base_fit + s.scale * s.samp_idx[0]) & 0xFFFF == s.samp_addr[0]
    assert (s.base_fit + s.scale * s.samp_idx[1]) & 0xFFFF == s.samp_addr[1]
    assert d2.relo_copies[0].delta == 0x6B0A
    assert d2.relo_copies[0].length == 255
    assert d2.idx_supp_by_pc()[0x1240].scale == 2


def test_recovery_offload_absent_parses_empty():
    """An artifact with NO recovery-offload sections (a pre-change tracer) parses
    with the new fields defaulting to empty -- the additions are non-breaking."""
    d = _empty_distill()
    d2 = D.parse_distill(D.build_distill(d))
    assert d2.idx_supp == [] and d2.ptr_walks == [] and d2.relo_copies == []
    assert d2.sid_accum == [] and d2.tempo_cands == []
    assert d2.iwlk_walks == []
    # build_distill emits an (empty-ish) DIGI; a hand-built absent one is None.
    assert d2.digi is None or isinstance(d2.digi, D.DigiSig)


def test_iwlk_walk_round_trip():
    """An IWLK instrument-table walk-index section survives parse(build(d)): the
    per-(pc,voice) per-frame u8 index round-trips, and the (pc,voice) lookup pairs
    a freq-feeding IDXR with its freq-mod generator input."""
    d = _empty_distill()
    d.iwlk_walks = [
        D.IwlkWalk(
            pc=0x1240,
            voice=0,
            index=np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.uint8),
        ),
        D.IwlkWalk(
            pc=0x1250,
            voice=7,
            index=np.array([5, 5, 4, 4], dtype=np.uint8),
        ),
    ]
    d2 = D.parse_distill(D.build_distill(d))
    assert len(d2.iwlk_walks) == 2
    for w_in, w_out in zip(d.iwlk_walks, d2.iwlk_walks):
        assert w_out.pc == w_in.pc
        assert w_out.voice == w_in.voice
        assert np.array_equal(w_out.index, w_in.index)
        assert w_out.index.dtype == np.uint8
    lut = d2.iwlk_by_pc_voice()
    assert np.array_equal(lut[(0x1240, 0)].index, d.iwlk_walks[0].index)
    assert lut[(0x1250, 7)].voice == 7


def test_iwlk_section_absent_is_noop():
    """A stream with no IWLK section parses with iwlk_walks empty -- the new reader
    branch is purely additive and does not break the current reader. This guards the
    invariant that PR-A lands BEFORE any emitter, so main stays compatible with both
    pre- and post-IWLK artifacts."""
    d = _empty_distill()
    blob = D.build_distill(d)
    assert b"IWLK" not in blob  # empty walks -> no section emitted
    d2 = D.parse_distill(blob)
    assert d2.iwlk_walks == []


def test_pwlk_adv_y_pc_round_trip():
    """The post-#11 per-advance advY (authored orderlist position) + advPc (consuming
    read PC) round-trip through build/parse, length-consistent with the advance count.
    """
    d = _empty_distill()
    d.ptr_walks = [
        D.PtrWalk(
            zp=0xFA,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=72,
            count=10759,
            ptr_vals=[0x157E, 0x14EB, 0x183F],
            adv_frames=[2, 2, 3],
            adv_y=[0, 0, 3],
            adv_pc=[0x10AE, 0x13F3, 0x14C0],
        )
    ]
    d2 = D.parse_distill(D.build_distill(d))
    w = d2.ptr_walks[0]
    assert w.adv_y == [0, 0, 3]
    assert w.adv_pc == [0x10AE, 0x13F3, 0x14C0]
    assert len(w.adv_y) == len(w.ptr_vals) == len(w.adv_pc) == len(w.adv_frames)
    assert all(0 <= y <= w.y_max for y in w.adv_y)


def test_pwlk_pre11_artifact_parses_old_layout():
    """A pre-#11 PtrWalk (no advY/advPc) emits + parses in the OLD layout -- the reader
    DISAMBIGUATES the two on-disk layouts structurally (back-compat tolerance, the
    sidtrace IWLK reader #152 idiom): empty adv_y/adv_pc means the appended arrays are
    absent, and the next section tag lands exactly after the legacy arrays."""
    d = _empty_distill()
    d.ptr_walks = [
        D.PtrWalk(
            zp=0xFB,
            is_load=True,
            is_store=False,
            y_min=0,
            y_max=76,
            count=5240,
            ptr_vals=[0x1A1B, 0x1AFC, 0x19F4],
            adv_frames=[1, 1, 2],
        )
    ]
    blob = D.build_distill(d)
    d2 = D.parse_distill(blob)
    w = d2.ptr_walks[0]
    assert w.adv_y == [] and w.adv_pc == []
    assert w.ptr_vals == [0x1A1B, 0x1AFC, 0x19F4]
    assert w.adv_frames == [1, 1, 2]


def test_digi_signature_classifies():
    """The DIGI signature: a PCM streamer (hundreds of $D418 writes/frame, no note
    table) is_digi; a tracker (tens of writes/frame, note table) is not."""
    digi = D.DigiSig(
        writes_per_frame_mean=11311.0,
        max_subframe_d418=23465,
        note_table_idxr_present=False,
        n_frames=40,
        n_sid_writes=452470,
    )
    assert digi.is_digi
    tracker = D.DigiSig(
        writes_per_frame_mean=18.0,
        max_subframe_d418=2,
        note_table_idxr_present=True,
        n_frames=2300,
        n_sid_writes=41000,
    )
    assert not tracker.is_digi


def test_idx_read_length():
    ir = D.IdxRead(pc=0, base=0x2000, stride=4, idx_min=0, idx_max=15, count=64)
    assert ir.length == 16


def test_parse_rejects_bad_magic():
    with pytest.raises(ValueError, match="not an SDST"):
        D.parse_distill(b"XXXX" + b"\x00" * 40)


# --------------------------------------------------------------------------- #
# A SYNTHETIC self-modifying-code player run through the REAL preframr-sidtrace
# binary (binary-gated). It proves the access-type classifier, end to end on a
# real emulation, excludes SMC from the song data and keeps the data correct.
# --------------------------------------------------------------------------- #
def _sidtrace_bin():
    from preframr_tokens.bacc.generic.sidtrace import sidtrace_bin

    return sidtrace_bin()


def _smc_psid():
    """A minimal hand-assembled PSID whose play routine SELF-MODIFIES an operand
    each frame (it patches the low byte of an ``LDA abs`` instruction) and also
    reads a small read-only data table. Returns the .sid bytes.

    Memory map (load $1000):
      init ($1000): RTS
      play ($1003):
        INC $1010        ; self-modify: bump the operand of the LDA below
        LDA $1234        ; operand low byte at $1010 is the SMC target
        STA $D400        ; emit something to anchor the play cadence
        LDA DATA,X-ish   ; read the read-only table
        STA $D401
        RTS
      DATA  ($1040): 4 read-only bytes
    """
    load = 0x1000
    code = {}

    def emit(addr, *bs):
        for i, b in enumerate(bs):
            code[addr + i] = b & 0xFF

    # init: CLD, LDX #0, RTS  (init at $1000)
    emit(0x1000, 0xD8, 0xA2, 0x00, 0x60)
    # play at $1004:
    #   INC $1009         ($1004) -- SMC: patches the operand low of LDA $xx34 below
    #   LDA $1234         ($1007) -- operand low byte lives at $1009 (self-modified)
    #   STA $D400         ($100A)
    #   LDA $1040         ($100D) -- read the read-only data table
    #   STA $D401         ($1010)
    #   RTS               ($1013)
    emit(0x1004, 0xEE, 0x09, 0x10)  # INC $1009
    emit(0x1007, 0xAD, 0x34, 0x12)  # LDA $1234 ; $1009 holds 0x34
    emit(0x100A, 0x8D, 0x00, 0xD4)  # STA $D400
    emit(0x100D, 0xAD, 0x40, 0x10)  # LDA $1040 (read-only data)
    emit(0x1010, 0x8D, 0x01, 0xD4)  # STA $D401
    emit(0x1013, 0x60)  # RTS
    # read-only data table at $1040
    for i, b in enumerate((0x11, 0x22, 0x33, 0x44)):
        emit(0x1040 + i, b)

    lo = min(code)
    hi = max(code)
    image = bytearray(hi - lo + 1)
    for a, b in code.items():
        image[a - lo] = b

    body = bytes([load & 0xFF, load >> 8]) + bytes(image)  # load addr prefix

    # PSID v2 header (0x7C bytes), init=$1000 play=$1004.
    import struct

    header = bytearray(0x7C)
    header[0:4] = b"PSID"
    struct.pack_into(">H", header, 4, 2)  # version
    struct.pack_into(">H", header, 6, 0x7C)  # data offset
    struct.pack_into(">H", header, 8, 0)  # load addr (0 => take from body prefix)
    struct.pack_into(">H", header, 10, 0x1000)  # init
    struct.pack_into(">H", header, 12, 0x1004)  # play
    struct.pack_into(">H", header, 14, 1)  # songs
    struct.pack_into(">H", header, 16, 1)  # start song
    struct.pack_into(">I", header, 18, 0)  # speed
    return bytes(header) + body


@pytest.mark.skipif(_sidtrace_bin() is None, reason="no preframr-sidtrace binary")
def test_smc_player_classified_correctly_end_to_end(tmp_path):
    from preframr_tokens.bacc.generic.distill import load_distill
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace

    sid = tmp_path / "smc.sid"
    sid.write_bytes(_smc_psid())
    prefix = str(tmp_path / "smc")
    _, distill_path = run_sidtrace(str(sid), prefix, subtune=1, nframes=60)
    dist = load_distill(distill_path)

    # The self-modified operand byte ($1009) is EXEC (part of the LDA) AND written
    # during play -> classified SMC, and excluded from the song data.
    smc_addr = 0x1009
    assert dist.smc_mask()[smc_addr], "SMC operand not classified as SMC"
    assert not dist.song_data_mask()[smc_addr], "SMC leaked into song data"

    # The read-only data table ($1040..$1043) is read as data, never written,
    # never executed -> classified song data, lifted byte-exact from the snapshot.
    sm = dist.song_data_mask()
    assert sm[0x1040] and sm[0x1043]
    assert not dist.acc[0x1040] & (D.ACC_WRITE_INIT | D.ACC_WRITE_PLAY)
    # Lifted byte-exact from the snapshot of the player's own RAM (HARD RULE #0).
    assert bytes(dist.ram[0x1040:0x1044]) == bytes([0x11, 0x22, 0x33, 0x44])


@pytest.mark.skipif(_sidtrace_bin() is None, reason="no preframr-sidtrace binary")
def test_grid_runner_artifact_is_tiny(tmp_path):
    """The per-tune artifact is a few KB -- the whole point of distilling in the
    emulator (vs the retired multi-GB raw bus trace). Reported, and gated."""
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace

    fixtures = os.path.join(os.path.dirname(__file__), "test_fixtures")
    grid = os.path.join(fixtures, "Grid_Runner.sid")
    prefix = str(tmp_path / "grid")
    _, distill_path = run_sidtrace(grid, prefix, subtune=1, nframes=400)
    size = os.path.getsize(distill_path)
    assert size < 64 * 1024, f"distill artifact {size} bytes is not tiny"
