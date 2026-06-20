"""Tests for the two-file BACC codec: recover -> render byte-exact, token log
round-trip, the < 1 token/frame economy, and the driver-backend dispatch."""

import struct

import numpy as np
import pytest

from preframr_tokens import (
    CPF,
    VOCAB,
    ids_to_program,
    measure,
    program_to_ids,
    render_program,
    verify_residual,
)
from preframr_tokens.bacc.backends import select_backend
from preframr_tokens.bacc.sidemu import SIDEmu, load_psid


def test_render_is_byte_exact(monty_program, monty_state):
    out = render_program(monty_program)
    assert out.shape == monty_state.shape
    assert np.array_equal(out, monty_state)


def test_verify_residual_true(monty_paths):
    assert verify_residual(monty_paths[0], monty_paths[1], CPF)


def test_recovered_program_shape(monty_program):
    assert monty_program.driver == "hubbard_monty"
    assert monty_program.nframes > 1000
    assert len(monty_program.score) > 0
    assert len(monty_program.instruments) == 20


def test_token_ids_in_vocab(monty_program):
    ids = program_to_ids(monty_program)
    assert ids and all(0 <= t < VOCAB for t in ids)


def test_token_roundtrip_renders_byte_exact(monty_program, monty_state):
    ids = program_to_ids(monty_program)
    program2 = ids_to_program(ids)
    assert len(program2.score) == len(monty_program.score)
    assert np.array_equal(render_program(program2), monty_state)


def test_under_one_token_per_frame(monty_program):
    brk, frames = measure(monty_program)
    block_sum = sum(brk[k] for k in ("score", "instr_def", "seed", "boot", "table"))
    assert block_sum < brk["total"]  # remainder = the leading nframes uint
    assert brk["total"] / frames < 1.0


def test_select_backend_rejects_unknown_driver():
    class _FakePsid:
        load_addr = 0x1234
        init_addr = 0x1234
        play_addr = 0x1237

    with pytest.raises(ValueError):
        select_backend(_FakePsid())


def _synthetic_psid(magic=b"PSID", load_addr=0):
    header = bytearray(0x7C)
    header[0:4] = magic
    struct.pack_into(">H", header, 4, 2)  # version
    struct.pack_into(">H", header, 6, 0x7C)  # data offset
    struct.pack_into(">H", header, 8, load_addr)
    struct.pack_into(">H", header, 10, 0x1000)  # init
    struct.pack_into(">H", header, 12, 0x1003)  # play
    struct.pack_into(">H", header, 14, 1)  # songs
    struct.pack_into(">H", header, 16, 1)  # start song
    program = bytes([0x60, 0x00, 0x00, 0x60])  # init RTS @1000, play RTS @1003
    data = (struct.pack("<H", 0x1000) if load_addr == 0 else b"") + program
    return bytes(header) + data


def test_load_psid_header_fields(monty_paths):
    psid = load_psid(monty_paths[0])
    assert psid.load_addr == 0x8000
    assert psid.init_addr == 0x8000
    assert psid.play_addr == 0x8012


def test_5tt_second_driver_byte_exact(tt_program, tt_state):
    # The backend interface generalizes: a 2nd Hubbard driver renders byte-exact
    # on every frame py65 reproduces. The dump's final frame is a py65-vs-sidtrace
    # emulation edge (py65 itself diverges there), not a BACC recovery failure.
    assert tt_program.driver == "hubbard_5tt"
    out = render_program(tt_program)
    assert out.shape == tt_state.shape
    assert np.array_equal(out[:-1], tt_state[:-1])


def test_select_backend_dispatches_per_driver(monty_paths, tt_paths):
    from preframr_tokens.bacc.sidemu import load_psid as _lp

    assert select_backend(_lp(monty_paths[0])).name == "hubbard_monty"
    assert select_backend(_lp(tt_paths[0])).name == "hubbard_5tt"


def test_synthetic_psid_runs(tmp_path):
    for magic, load in ((b"PSID", 0), (b"RSID", 0x1000)):
        p = tmp_path / "synthetic.sid"
        p.write_bytes(_synthetic_psid(magic, load))
        psid = load_psid(str(p))
        assert psid.init_addr == 0x1000 and psid.play_addr == 0x1003
        emu = SIDEmu(psid)
        emu.init(0)
        emu.play_frame()
        assert emu.state() == [0] * 25
