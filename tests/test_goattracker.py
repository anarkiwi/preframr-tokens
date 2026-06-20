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
from preframr_tokens.bacc.backends.goattracker import make_program, render_song
from tests._dump_fixture import acquire

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
    assert program2.tables["sng"] == program.tables["sng"]
    assert program2.seed["adparam"] == program.seed["adparam"]
    assert np.array_equal(render_program(program2), render_program(program))


def test_measure_breaks_down_program():
    program = make_program(_demo_sng(), _DEMO_SEED, 128)
    brk, frames = measure(program)
    assert frames == 128
    assert 0 < brk["sng"] <= brk["total"]


class _GtPsid:
    load_addr = 0x1000
    init_addr = 0x1000
    play_addr = 0x1003
    data = bytes([0x4C, 0xA3, 0x10, 0x4C, 0xA7, 0x10]) + bytes(16)


def test_select_backend_dispatches_goattracker():
    assert select_backend(_GtPsid()).name == "goattracker"


@pytest.fixture(scope="module")
def grid_runner_paths():
    return acquire(_GR_REL, _GR_URL, subtune=1)


def test_grid_runner_byte_exact(grid_runner_paths):
    sid, dump = grid_runner_paths
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "GoatTracker backend is NOT residual-zero on Grid_Runner"


def test_grid_runner_under_one_token_per_frame(grid_runner_paths):
    sid, dump = grid_runner_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    assert program.driver == "goattracker"
    brk, frames = measure(program)
    assert brk["total"] / frames < 1.0, (
        f"Grid_Runner: {brk['total']} tokens / {frames} frames = "
        f"{brk['total'] / frames:.3f} tok/frame (must be < 1)"
    )


def test_grid_runner_token_roundtrip(grid_runner_paths):
    sid, dump = grid_runner_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    program2 = ids_to_program(program_to_ids(program), driver="goattracker")
    assert np.array_equal(render_program(program2), render_program(program))
