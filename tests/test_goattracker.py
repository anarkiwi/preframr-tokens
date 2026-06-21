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


def test_measure_breaks_down_program():
    program = make_program(_demo_sng(), _DEMO_SEED, 128)
    brk, frames = measure(program)
    assert frames == 128
    assert 0 < brk["header"] <= brk["total"]


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
    # The global cross-pattern row-LZ (one backward window over ALL patterns
    # concatenated instead of a fresh window per pattern) brings Grid_Runner to
    # ~2,817 tokens (was 4,132 with per-pattern windows). It still must fit
    # < 1 token/frame AND the 8192-token context window, and now also under 4096.
    assert brk["total"] / frames < 1.0, (
        f"Grid_Runner: {brk['total']} tokens / {frames} frames = "
        f"{brk['total'] / frames:.3f} tok/frame (must be < 1)"
    )
    assert brk["total"] < 8192, (
        f"Grid_Runner: {brk['total']} tokens for the whole song >= 8192 "
        f"(must fit the 8192-token context window)"
    )
    assert brk["total"] < 4096, (
        f"Grid_Runner: {brk['total']} tokens >= 4096 -- the global "
        f"cross-pattern row-LZ should keep it well under 4096 (~2,817)"
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
    round-trips render-equal."""
    sid, dump = acquire(_NEH_REL, _NEH_URL, subtune=1)
    program = recover_program(sid, dump, CPF, subtune=0)
    brk, frames = measure(program)  # must not raise
    assert brk["total"] > 0 and frames > 0
    program2 = ids_to_program(program_to_ids(program), driver="goattracker")
    assert np.array_equal(render_program(program2), render_program(program))


def test_table_overrun_fails_cleanly_not_indexerror():
    """FamiCommodore recovers but a table pointer overruns at render. The bare
    pygoattracker IndexError is now surfaced as a clear backend RuntimeError
    naming the table-pointer overrun (no fabricated frames)."""
    sid, dump = acquire(_FAMI_REL, _FAMI_URL, subtune=1)
    program = recover_program(sid, dump, CPF, subtune=0)
    with pytest.raises(RuntimeError, match="table pointer overran"):
        render_program(program)


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
