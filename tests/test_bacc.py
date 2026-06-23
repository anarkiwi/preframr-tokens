"""Driver-agnostic BACC plumbing: the PSID loader/emulator and the hand-backend
dispatch rejection. The per-driver recover -> render byte-exact + token round-trip
+ token-economy proofs live in the backend-specific suites (GoatTracker in
tests/test_goattracker.py; the generic path in tests/test_generic_recovery.py)."""

import struct

import pytest

from preframr_tokens.bacc.backends import select_backend
from preframr_tokens.bacc.sidemu import SIDEmu, load_psid


def test_select_backend_rejects_unknown_driver():
    # No hand backend matches a PSID that is not a GoatTracker image -> raise
    # (the generic path handles everything else; there is no silent fallback).
    class _FakePsid:
        load_addr = 0x1234
        init_addr = 0x1234
        play_addr = 0x1237
        data = b""

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
