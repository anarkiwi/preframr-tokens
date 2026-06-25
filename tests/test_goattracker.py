"""GoatTracker backend: dispatch + serialize round-trip on an in-process song,
and the full recover -> byte-exact -> < 1 token/frame proof on a real packed
GoatTracker SID (Grid_Runner by Jammer).

The in-process tests need no fixtures (a song is built via pygoattracker's model);
the Grid_Runner test acquires the .sid + dump the same way the Monty gate does
(download the .sid, render the dump in the headlessvice container, cache both)."""

import os

import numpy as np
import pytest

from preframr_tokens import (
    CPF,
    VOCAB,
    ids_to_program,
    measure,
    program_to_ids,
    recover_program,
    render_program,
    verify_residual,
)
from preframr_tokens.bacc.backends import select_backend
from preframr_tokens.bacc.backends.goattracker import (
    _align,
    _align_offset,
    _boot1_offset,
    make_program,
    render_song,
)
from tests._dump_fixture import _HVSC_BASE, acquire

pygoattracker = pytest.importorskip("pygoattracker")

_DEMO_SEED = {
    "subtune": 0,
    "adparam": 0x0900,
    "optimize_pulse": 0,
    "optimize_realtime": 0,
}

_GR_REL = "MUSICIANS/J/Jammer/Grid_Runner.sid"
_GR_URL = os.environ.get(
    "GRID_RUNNER_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/J/Jammer/Grid_Runner.sid",
)


def _demo_sng():
    from pygoattracker import Instrument, Pattern, Row, Song, build_sng
    from pygoattracker.constants import note_value

    song = Song(name="DEMO", author="T", copyright="2026")
    wave_ptr = song.wavetable.add(0x41, 0x00)
    song.wavetable.add(0xFF, 0x00)
    song.instruments.append(
        Instrument(
            attack_decay=0x09,
            sustain_release=0x00,
            wave_ptr=wave_ptr,
            gateoff_timer=2,
            first_wave=0x09,
            name="LEAD",
        )
    )
    pattern = Pattern.empty(16)
    pattern.rows[0] = Row(note=note_value("C-4"), instrument=1)
    pattern.rows[8] = Row(note=note_value("G-4"), instrument=1)
    song.patterns = [pattern]
    return build_sng(song)


def test_vocab_matches_flat_alphabet():
    """The exported VOCAB/PAD_ID must equal the flat v2 alphabet's size (the
    model-facing alphabet)."""
    from preframr_tokens.bacc import flat_serialize as F

    assert VOCAB == F.VOCAB
    from preframr_tokens import PAD_ID

    assert PAD_ID == F.PAD_ID == F.VOCAB


def test_flat_tokens_are_typed_and_self_delimiting():
    """A flat-encoded program begins with BOS, ends with EOS, and every token id
    falls inside the declared typed ranges (no v1 LEB digits / LZ markers)."""
    from preframr_tokens.bacc import flat_serialize as F

    program = make_program(_demo_sng(), _DEMO_SEED, 128)
    ids = program_to_ids(program)
    assert ids[0] == F.BOS and ids[-1] == F.EOS
    assert all(0 <= t < F.VOCAB for t in ids)
    # structural section markers appear in order
    for sec in (
        F.SEC_HEADER,
        F.SEC_TABLES,
        F.SEC_INSTRUMENTS,
        F.SEC_PATTERNS,
        F.SEC_ORDERLISTS,
    ):
        assert sec in ids


def test_render_song_shape_and_masking():
    state = render_song(_demo_sng(), _DEMO_SEED, 128)
    assert state.shape == (128, 25)
    assert state[:, [3, 10, 17]].max() <= 0x0F
    assert state[:, 21].max() <= 0x07


def test_token_roundtrip_renders_byte_exact():
    program = make_program(_demo_sng(), _DEMO_SEED, 128)
    ids = program_to_ids(program)
    assert ids and all(0 <= t < VOCAB for t in ids)
    program2 = ids_to_program(ids, driver="goattracker")
    # The program is the abstract Song (rows + instrument-generators + orderlist),
    # NOT raw .SNG bytes; the gate is render-equality, not byte-equality.
    assert "sng" not in program2.tables and "song" in program2.tables
    assert program2.seed["adparam"] == program.seed["adparam"]
    assert np.array_equal(render_program(program2), render_program(program))


def test_global_pattern_lz_reslices_patterns():
    """The global cross-pattern row-LZ runs ONE backward window over all patterns
    concatenated, so a phrase repeated in a later pattern copies from an earlier
    one; decode must re-slice the flat row stream back into the SAME patterns
    (right count, right per-pattern row counts, right rows). Two patterns that
    share an identical phrase exercise the cross-pattern copy + the re-slice."""
    from pygoattracker import Instrument, Pattern, Row, Song, build_sng
    from pygoattracker.constants import note_value

    from preframr_tokens.bacc.gt_serialize import (
        gt_ids_to_program,
        gt_program_to_ids,
    )

    song = Song(name="GLZ", author="T", copyright="2026")
    wave_ptr = song.wavetable.add(0x41, 0x00)
    song.wavetable.add(0xFF, 0x00)
    song.instruments.append(
        Instrument(
            attack_decay=0x09,
            sustain_release=0x00,
            wave_ptr=wave_ptr,
            gateoff_timer=2,
            first_wave=0x09,
            name="LEAD",
        )
    )
    phrase = [
        Row(note=note_value("C-4"), instrument=1),
        Row(note=note_value("E-4"), instrument=1),
        Row(note=note_value("G-4"), instrument=1),
    ]
    pat_a = Pattern.empty(16)
    pat_b = Pattern.empty(8)  # different length -> exercises per-pattern counts
    for k, r in enumerate(phrase):
        pat_a.rows[k] = r
        pat_b.rows[k] = r  # same phrase, later pattern: cross-pattern copy
    song.patterns = [pat_a, pat_b]

    program = make_program(build_sng(song), _DEMO_SEED, 128)
    ids = gt_program_to_ids(program)
    program2 = gt_ids_to_program(ids)
    song2 = program2.tables["song"]
    # re-slice fidelity: same pattern count + same per-pattern row counts + rows
    assert len(song2.patterns) == 2
    assert [len(p.rows) for p in song2.patterns] == [16, 8]
    src = program.tables["song"]
    for p_in, p_out in zip(src.patterns, song2.patterns):
        assert [(r.note, r.instrument, r.command, r.data) for r in p_in.rows] == [
            (r.note, r.instrument, r.command, r.data) for r in p_out.rows
        ]
    assert np.array_equal(render_program(program2), render_program(program))


def test_pattern_rows_transpose_factor_roundtrip():
    """A phrase repeated at a transposed pitch (same instr/command/data) factors as
    one TRANSPOSE+Delta over the shared post-BACC row LZ, not a fresh literal run --
    byte-exact round-trip and fewer tokens than REPEAT-only would emit."""
    from pygoattracker.constants import note_value

    from preframr_tokens.bacc.gt_serialize import (
        _lz_emit,
        _lz_read,
        _row_delta,
        _row_lit,
        _row_read,
        _row_shift,
    )
    from preframr_tokens.bacc.serialize import TRANSPOSE

    phrase = [(note_value("C-4"), 1, 0, 0), (note_value("E-4"), 1, 0, 0)]
    up = [(note_value("G-4"), 1, 0, 0), (note_value("B-4"), 1, 0, 0)]  # +7 semis
    rows = phrase + up  # phrase then the same shape a fifth higher
    out = []
    _lz_emit(out, rows, _row_lit, _row_delta)
    assert TRANSPOSE in out  # the transposed phrase factored
    back, consumed = _lz_read(out, 0, len(rows), _row_read, _row_shift)
    assert consumed == len(out)
    assert back == rows  # byte-exact inverse re-coordinates the pitch


def test_measure_breaks_down_program():
    program = make_program(_demo_sng(), _DEMO_SEED, 128)
    brk, frames = measure(program)
    assert frames == 128
    assert 0 < brk["header"] <= brk["total"]
    assert brk["score"] <= brk["total"]  # the dominant pattern-row LZ block


class _GtPsid:
    load_addr = 0x1000
    init_addr = 0x1000
    play_addr = 0x1003
    data = bytes([0x4C, 0xA3, 0x10, 0x4C, 0xA7, 0x10]) + bytes(16)


def test_select_backend_dispatches_goattracker():
    assert select_backend(_GtPsid()).name == "goattracker"


# --- boot-frame alignment (the dominant corpus byte-exact failure) ----------
def test_align_drops_leading_init_frames():
    """Standard framing: the render holds some leading init/hard-restart frames
    before the dump's boot frame; _align must drop exactly those (Grid_Runner
    class). boot==render[2], boot1==render[3] here, so offset is 2."""
    boot = [1] + [0] * 24
    boot1 = [2] + [0] * 24
    rendered = np.array(
        [[9] * 25, [8] * 25, boot, boot1, [3] * 25, [4] * 25], dtype=np.int64
    )
    assert _align_offset(rendered, boot, boot1) == 2
    out = _align(rendered, boot, boot1, 3)
    assert np.array_equal(out, np.array([boot, boot1, [3] * 25], dtype=np.int64))


def test_align_deep_offset_beyond_old_slack():
    """The dump's first_play_cycle can start well into playback: the render needs
    >32 frames of ADSR/init ramp before it reaches the dump's boot frame. The old
    32-frame window fell back to offset 0 and shifted the whole render; the wider
    window finds it (this is the ~Need_More_NOPs deep-offset class)."""
    boot = [7] + [0] * 24
    boot1 = [6] + [0] * 24
    lead = [[0] * 24 + [15]] * 40  # 40 cold init frames, past the old slack
    rendered = np.array(lead + [boot, boot1, [5] * 25], dtype=np.int64)
    assert _align_offset(rendered, boot, boot1) == 40
    out = _align(rendered, boot, boot1, 2)
    assert np.array_equal(out, np.array([boot, boot1], dtype=np.int64))


def test_align_prepends_dump_leading_silence_frame():
    """Leading-silence dump (the Alienator class): the dump captured an all-zero
    silence frame 0 that the player's render never emits -- the render starts
    straight at boot1. _align must prepend the dump's frame 0 so the aligned
    render reproduces it byte-for-byte, instead of falling back to offset 0 and
    shifting the entire render one frame."""
    boot = [0] * 24 + [15]  # the leading silence frame the render omits
    boot1 = [34] + [0] * 23 + [15]
    rendered = np.array([boot1, [68] + [0] * 23 + [15], boot1], dtype=np.int64)
    # render has no frame equal to boot, so the boot+boot1 search yields None ...
    assert _align_offset(rendered, boot, boot1) is None
    # ... and the boot1 anchor is at render frame 0.
    assert _boot1_offset(rendered, boot1) == 0
    out = _align(rendered, boot, boot1, 3)
    assert np.array_equal(out[0], np.array(boot))  # frame 0 restored exactly
    assert np.array_equal(out[1], np.array(boot1))  # render begins right after


@pytest.fixture(scope="module")
def grid_runner_paths():
    return acquire(_GR_REL, _GR_URL, subtune=1)


def test_grid_runner_byte_exact(grid_runner_paths):
    sid, dump = grid_runner_paths
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "GoatTracker backend is NOT residual-zero on Grid_Runner"


def test_grid_runner_context_budget(grid_runner_paths):
    sid, dump = grid_runner_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    assert program.driver == "goattracker"
    brk, frames = measure(program)
    # FLAT v2 alphabet (learnability-first): typed atoms + self-delimiting
    # structure, NO inline LZ and NO base-16 LEB place-value. Streams are longer
    # than the v1 LEB+LZ scheme (Grid_Runner ~9,480 vs ~2,817) -- the deliberate
    # compression-for-predictability trade -- but still well under 1 token/frame
    # and inside a 16k context window. The hard gate (residual-0) is unchanged
    # and covered by test_grid_runner_byte_exact.
    assert brk["fmt"] == "flat_v2"
    assert brk["total"] / frames < 1.0, (
        f"Grid_Runner: {brk['total']} tokens / {frames} frames = "
        f"{brk['total'] / frames:.3f} tok/frame (must be < 1)"
    )
    assert brk["total"] < 16384, (
        f"Grid_Runner: {brk['total']} flat tokens for the whole song >= 16384 "
        f"(must fit a 16k context window)"
    )


def test_grid_runner_token_roundtrip(grid_runner_paths):
    sid, dump = grid_runner_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    program2 = ids_to_program(program_to_ids(program), driver="goattracker")
    assert np.array_equal(render_program(program2), render_program(program))


# --- real-tune regressions for the boot-align fix + the two render/measure bugs
_NMN_REL = "MUSICIANS/F/Fegolhuzz/Need_More_NOPs.sid"
_NMN_URL = f"{_HVSC_BASE}/{_NMN_REL}"
_NEH_REL = "MUSICIANS/C/Crowley_Owen/Not_Even_Human.sid"
_NEH_URL = f"{_HVSC_BASE}/{_NEH_REL}"
_FAMI_REL = "DEMOS/A-F/FamiCommodore.sid"
_FAMI_URL = f"{_HVSC_BASE}/{_FAMI_REL}"
_TWILIGHT_REL = "MUSICIANS/N/No-XS/Twilight.sid"
_TWILIGHT_URL = f"{_HVSC_BASE}/{_TWILIGHT_REL}"


def test_deep_offset_tune_byte_exact():
    """Need_More_NOPs: the dump starts ~36 frames into playback, past the old
    32-frame alignment window, so the old backend fell back to offset 0 and
    mismatched everywhere (the ~893-tune boot-frame class). With the widened,
    window-based alignment it recovers byte-exact."""
    sid, dump = acquire(_NMN_REL, _NMN_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "deep-offset boot alignment is NOT residual-zero on Need_More_NOPs"


def test_measure_guards_nonpitched_note_byte():
    """Not_Even_Human carries note bytes below FIRSTNOTE (no clean freq-table
    pitch). Previously fn_to_grid(<=0) raised ValueError: math domain error in
    measure; now they ride the raw-note escape and the program measures and
    round-trips structurally.

    This tune reaches out-of-range notes (>= MAX_NOTES) that the PACKED player
    resolves through its UNPADDED freq table (overrunning into adjacent image
    bytes). Those overrun frequencies are a deterministic function of the
    packer's relocation layout (constant freqtblhi bytes + the low bytes of the
    relocated orderlist/pattern addresses) -- not irreducible, but the abstract
    token stream (the recovered Song) does not encode that layout, so the token
    round-trip is not byte-for-byte render-equal here; equality for in-range
    tunes is covered
    by test_grid_runner_token_roundtrip / test_token_roundtrip_renders_byte_exact.
    """
    sid, dump = acquire(_NEH_REL, _NEH_URL, subtune=1)
    program = recover_program(sid, dump, CPF, subtune=0)
    brk, frames = measure(program)  # must not raise
    assert brk["total"] > 0 and frames > 0
    program2 = ids_to_program(program_to_ids(program), driver="goattracker")
    rendered = render_program(program2)  # decoded program renders without error
    assert rendered.shape == render_program(program).shape


def test_table_overrun_fails_cleanly_not_indexerror():
    """FamiCommodore recovers but a table pointer overruns at render. The bare
    pygoattracker IndexError is now surfaced as a clear backend RuntimeError
    naming the table-pointer overrun (no fabricated frames)."""
    sid, dump = acquire(_FAMI_REL, _FAMI_URL, subtune=1)
    program = recover_program(sid, dump, CPF, subtune=0)
    with pytest.raises(RuntimeError, match="table pointer overran"):
        render_program(program)


def test_packed_freqtable_overrun_byte_exact():
    """Twilight: the packed player lays the freq table out UNPADDED
    (freqtbllo[fn..ln] | freqtblhi[fn..ln] | songtbl, L=80, firstnote=16) and a
    wavetable relative-note step indexes ``mt_freqtbllo-FIRSTNOTE,y`` past the
    table's end, reading adjacent relocated image bytes -- an overrun frequency
    determined by the packer's relocation layout (constant freqtblhi +
    relocated address low-bytes), not musical content.
    pygoattracker's editor table is zero-padded to 128 and
    returns freq 0 there, so the render diverged. Recovering the exact 128-entry
    packed-image table (overrun bytes included) and passing it to the player
    reproduces those frequencies, so the whole song recovers byte-exact."""
    sid, dump = acquire(_TWILIGHT_REL, _TWILIGHT_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "packed freq-table overrun is NOT residual-zero on Twilight"


def test_packed_freqtable_recovers_overrun_bytes():
    """The recovered packed-image freq table differs from the editor's
    zero-padded table exactly where the packed player overruns -- it carries
    real image bytes (not zeros) for out-of-range notes."""
    from pygoattracker import constants as C

    from preframr_tokens.bacc.backends import gt_unpack

    sid, _ = acquire(_TWILIGHT_REL, _TWILIGHT_URL, subtune=1)
    img = gt_unpack._Image(sid)
    lay = gt_unpack._derive_layout(img)
    table = gt_unpack._packed_freq_table(img, lay)
    assert len(table) == 128
    # in-range notes match the standard table exactly
    fn, ln = lay["firstnote"], lay["lastnote"]
    for n in range(fn, ln + 1):
        assert table[n] == ((C.FREQ_HI[n] << 8) | C.FREQ_LO[n])
    # at least one out-of-range note reads a nonzero overrun byte the editor
    # table would have returned 0 for (the divergence this fix resolves).
    assert any(
        table[n] != 0 and (n >= C.MAX_NOTES or table[n] != C.FREQ_TABLE[n])
        for n in range(128)
        if not fn <= n <= ln
    )


_VIC64_REL = "DEMOS/UNKNOWN/VIC64Demo_tune_5.sid"
_VIC64_URL = f"{_HVSC_BASE}/{_VIC64_REL}"


def test_adsr_hardrestart_byte_exact():
    """VIC64Demo_tune_5: the ADSR hard-restart (gate-off timer) class. The
    player slams the SID envelope to the hard-restart ADPARAM/SRPARAM
    (ad=$0F, sr=$00) for the 1-2 frames before each note-on (player.s
    mt_normalnote 1303/1313), then writes the note's real ADSR. The old
    ADPARAM recovery (first AD/SR write the emulator emits) captured the first
    NOTE's ADSR instead, so the render held a stale envelope through the
    gate-off frames and diverged on reg 5/6/19/20 in consecutive pairs.
    Recovering ADPARAM straight from the image's hard-restart store -- by store
    TARGET (lower-address store == AD), which is the value the chip executes --
    makes the whole song byte-exact."""
    sid, dump = acquire(_VIC64_REL, _VIC64_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "ADSR hard-restart recovery is NOT residual-zero on VIC64Demo_tune_5"


def test_adparam_recovered_from_image_store():
    """The hard-restart ADPARAM/SRPARAM is read from the player image's HR store
    (the immediates the chip executes), keyed by store TARGET so the AD/SR pair
    is correct regardless of which order greloc emits the two stores."""
    from preframr_tokens.bacc.backends import gt_unpack

    sid, _ = acquire(_VIC64_REL, _VIC64_URL, subtune=1)
    img = gt_unpack._Image(sid)
    # VIC64Demo_tune_5 stores SRPARAM($00) first then ADPARAM($0F): (AD<<8)|SR.
    assert gt_unpack._adparam_from_image(img) == 0x0F00


_JETTA_REL = "DEMOS/G-L/Jetta.sid"
_JETTA_URL = f"{_HVSC_BASE}/{_JETTA_REL}"


def test_simplepulse_byte_exact():
    """Jetta: greloc's SIMPLEPULSE optimization packs a pulse SET step into one
    byte ``(pulsehi & 0x0f) | (pulselo & 0xf0)`` and the packed player stores it
    to BOTH ghost pulse registers (player.s mt_setpulse, .IF SIMPLEPULSE != 0),
    so pulse-hi = the packed byte's low nibble -- not 0 like the editor player.
    gt_unpack now detects the SIMPLEPULSE build flag from the packed
    mt_pulsemod ``adc mt_pulsespdtbl-1,y ; adc #$00`` signature and runs the
    packed pulse path on the (un-inverted) packed pulse table, so the whole song
    recovers byte-exact (was a whole-song pulse-hi mismatch)."""
    sid, dump = acquire(_JETTA_REL, _JETTA_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "SIMPLEPULSE pulse path is NOT residual-zero on Jetta"


def test_simplepulse_flag_detected():
    """The SIMPLEPULSE build flag is recovered from the packed player image
    (present in Jetta, absent in the full-mod Grid_Runner build)."""
    from preframr_tokens.bacc.backends import gt_unpack

    jetta, _ = acquire(_JETTA_REL, _JETTA_URL, subtune=1)
    grid, _ = acquire(_GR_REL, _GR_URL, subtune=1)
    jimg = gt_unpack._Image(jetta)
    gimg = gt_unpack._Image(grid)
    assert gt_unpack._detect_simplepulse(jimg, gt_unpack._derive_layout(jimg)) is True
    assert gt_unpack._detect_simplepulse(gimg, gt_unpack._derive_layout(gimg)) is False
    # the flag rides the recovered program's render params
    gt_unpack.reconstruct_song(jetta)
    assert gt_unpack.render_params()["simplepulse"] is True


_CATALYST_REL = "MUSICIANS/J/Jammer/Catalyst.sid"
_CATALYST_URL = f"{_HVSC_BASE}/{_CATALYST_REL}"


def test_simplepulse_setpulse_only_byte_exact():
    """Catalyst: a SIMPLEPULSE build whose pulse table is all SET steps (no pulse
    MODULATION). With no mt_pulsemod accumulate the old ``adc #$00`` signature
    cannot fire, so SIMPLEPULSE went undetected and the editor pulse path gave
    pulse-hi 0 on every frame. The SET-PULSE step itself is the tell: SIMPLEPULSE
    drops the editor's pulse-hi store (from the mt_pulsetimetbl byte) and feeds
    the SPEED byte to both registers, so the speed-table load immediately follows
    the time-table load with no store between (player.s mt_setpulse,
    .IF SIMPLEPULSE != 0). Detecting that ``lda mt_pulsetimetbl-1,y ; lda
    mt_pulsespdtbl-1,y`` pair recovers the flag and the song renders byte-exact."""
    sid, dump = acquire(_CATALYST_REL, _CATALYST_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "set-pulse-only SIMPLEPULSE is NOT residual-zero on Catalyst"


def test_simplepulse_setpulse_only_flag_detected():
    """The SET-PULSE-only SIMPLEPULSE flag is recovered from Catalyst (which has
    no mt_pulsemod accumulate) while the full-mod Grid_Runner build stays False --
    the new set-pulse signature never false-positives on full-mod set steps."""
    from preframr_tokens.bacc.backends import gt_unpack

    cat, _ = acquire(_CATALYST_REL, _CATALYST_URL, subtune=1)
    grid, _ = acquire(_GR_REL, _GR_URL, subtune=1)
    cimg = gt_unpack._Image(cat)
    gimg = gt_unpack._Image(grid)
    assert gt_unpack._detect_simplepulse(cimg, gt_unpack._derive_layout(cimg)) is True
    assert gt_unpack._detect_simplepulse(gimg, gt_unpack._derive_layout(gimg)) is False


_HAMMURABI_REL = "GAMES/G-L/Hammurabi.sid"
_HAMMURABI_URL = f"{_HVSC_BASE}/{_HAMMURABI_REL}"


def test_freq_latch_timing_byte_exact():
    """Hammurabi: the GoatTracker frequency-latch timing class. The packed
    instrument-vibrato-only build (greloc NOEFFECTS != 0) reads the continuous
    instrument-vibrato param LIVE from the channel's current instrument every
    tick (player.s mt_chninstr -> mt_insvibparam-1). At a gate-off boundary
    mt_getnewnote flips mt_chninstr to the NEW note's instrument 1-2 frames
    before the note inits, so the held (old) frequency is wobbled by the new
    instrument's vibrato (e.g. instr 8 speed-table[1]=5 -> 5299+5 = 5304, freq-lo
    184). The full-FX model used the param LATCHED at note-init, holding the old
    frequency and diverging on freq-lo in consecutive gate-off pairs (34 frames).
    Detecting the live-vibrato build flag and driving the live param read makes
    the whole song byte-exact."""
    sid, dump = acquire(_HAMMURABI_REL, _HAMMURABI_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "freq-latch timing is NOT residual-zero on Hammurabi"


def test_live_vibrato_flag_detected():
    """The live-vibrato build flag is recovered from the packed player image:
    present in the instrument-vibrato-only build (NOEFFECTS != 0, Hammurabi),
    absent in the full-FX build (NOEFFECTS == 0, Grid_Runner, whose pattern FX
    force mt_chnfx/mt_chnparam dispatch with a latched param)."""
    from preframr_tokens.bacc.backends import gt_unpack

    ham, _ = acquire(_HAMMURABI_REL, _HAMMURABI_URL, subtune=1)
    grid, _ = acquire(_GR_REL, _GR_URL, subtune=1)
    himg = gt_unpack._Image(ham)
    gimg = gt_unpack._Image(grid)
    assert gt_unpack._detect_live_vibrato(himg, gt_unpack._derive_layout(himg)) is True
    assert gt_unpack._detect_live_vibrato(gimg, gt_unpack._derive_layout(gimg)) is False
    # the flag rides the recovered program's render params
    gt_unpack.reconstruct_song(ham)
    assert gt_unpack.render_params()["live_vibrato"] is True


_TRUCKON_REL = "MUSICIANS/L/Linus/Truck-On.sid"
_TRUCKON_URL = f"{_HVSC_BASE}/{_TRUCKON_REL}"


def test_toneporta_lastnote_vibrato_byte_exact():
    """Truck-On: the GoatTracker toneporta-lastnote latch class. When a
    toneportamento reaches/crosses its target note the packed player stores the
    target note into mt_chnlastnote (player.s mt_effect_3_found -> mt_wavenoteabs,
    under NOCALCULATEDSPEED == 0). A following calculated-speed (high-bit
    speedtable) vibrato then derives its step from the target-note interval. The
    render kept a STALE lastnote after a toneporta, so a post-porta calculated-
    speed vibrato used the pre-porta note interval and its depth was off by the
    porta's note distance, diverging on freq-lo. pygoattracker>=0.1.6 latches
    lastnote on toneporta completion, making the song byte-exact."""
    sid, dump = acquire(_TRUCKON_REL, _TRUCKON_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "toneporta-lastnote vibrato is NOT residual-zero on Truck-On"


_DUMDEEDUM_REL = "MUSICIANS/J/Jakim/Dum-dee-dum.sid"
_DUMDEEDUM_URL = f"{_HVSC_BASE}/{_DUMDEEDUM_REL}"


def test_fixedparams_gatetimer_counter_byte_exact():
    """Dum-dee-dum: the FIXEDPARAMS=1 GATETIMERPARAM recovery class. The packed
    player (PULSEOPTIMIZATION=0) bakes the gate-off timer as an immediate
    (cmp #GATETIMERPARAM=$02). The old recovery picked the largest-forward-branch
    ``BD..;C9..;F0`` site, but here the real gate-off check sits just before
    mt_getnewnote (branch +3) while the two toneporta-skip checks (lda
    mt_chnnewfx ; cmp #TONEPORTA=$03) branch far (+61/+15) -- so it mis-recovered
    gatetimer 3 instead of the true 2 and fired the ADSR hard-restart one frame
    early. Keying GATETIMERPARAM on the site that loads mt_chncounter (the operand
    of the unique ``dec <abs>,x ; beq`` in mt_execchn) selects the real gate-off
    site (cmp #$02), making the song byte-exact. FIXEDPARAMS=0 builds
    (VIC64Demo_tune_5) read gatetimer from per-instrument columns and are
    unaffected."""
    sid, dump = acquire(_DUMDEEDUM_REL, _DUMDEEDUM_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "FIXEDPARAMS gatetimer recovery is NOT residual-zero on Dum-dee-dum"


def test_fixedparams_gatetimer_recovers_counter_site():
    """The GATETIMERPARAM recovery keys on mt_chncounter: on Dum-dee-dum
    (FIXEDPARAMS=1, PULSEOPTIMIZATION=0) it recovers the true 2 from the
    counter-loading gate-off site, not the far-branch toneporta ``cmp #$03``."""
    from preframr_tokens.bacc.backends import gt_unpack

    sid, _ = acquire(_DUMDEEDUM_REL, _DUMDEEDUM_URL, subtune=1)
    img = gt_unpack._Image(sid)
    lay = gt_unpack._derive_layout(img)
    gatetimer, _firstwave = gt_unpack._recover_fixed_params(img, lay["wave_L"])
    assert gatetimer == 2, f"expected GATETIMERPARAM 2, recovered {gatetimer}"


def test_grid_runner_is_abstract_not_bytes(grid_runner_paths):
    """Part A: the recovered GoatTracker program is the COMMON abstraction --
    per-voice tracker ROWS + instrument-GENERATOR defs + a backward ORDERLIST --
    NOT raw .SNG bytes. There is no byte path: tables['sng'] never exists, the
    decoded program carries the abstract Song, and it renders byte-exact."""
    sid, dump = grid_runner_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    assert "sng" not in program.tables, "raw-.SNG-bytes escape hatch must be gone"
    program2 = ids_to_program(program_to_ids(program), driver="goattracker")
    song = program2.tables["song"]
    # rows + instrument-generators + orderlist all present as structure
    assert song.patterns and song.patterns[0].rows
    assert song.instruments and song.instruments[0].wave_ptr is not None
    assert song.subtunes[0].channels[0].entries
    # the generator tables are abstract parameter columns, not opaque bytes
    assert len(song.wavetable) and len(song.pulsetable)
    assert np.array_equal(render_program(program2), render_program(program))
