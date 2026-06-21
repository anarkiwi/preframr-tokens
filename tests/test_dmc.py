"""DMC (Demo Music Creator) v7.62 backend: dispatch + the byte-exact recovery of
Brian/Graffity tunes into the common BACC abstraction, plus the token round-trip.

DMC is the single most-used editor in the HVSC (~10,700 tunes). The recovery
decomposes the tune into the COMMON learnable abstraction -- canonical A440-grid
notes (the DMC note table is a clean 12-TET bijection, so a DMC note serializes
to the SAME grid token as a GoatTracker note at the same concert pitch) +
instrument generators + a backward per-voice orderlist -- exactly like the other
backends. render() rebuilds the player image's song-data regions FROM that
recovered abstraction and re-runs the engine, so the round-trip is residual-0
(byte-exact), not a replay of the untouched image.

Pins:
  * the DMC fingerprint (BRIAN/GRAFFITY v7.62 $1000) matches its own tunes and not
    the GoatTracker / Hubbard / lft drivers;
  * recover->render is byte-exact (verify_residual) on Ode_to_Music;
  * the recovered program is the abstraction (grid notes + orderlist + instruments),
    NOT raw song bytes -- the song-data regions are blanked in the stored template;
  * program_to_ids/ids_to_program round-trips and re-renders identically.

The fixture (.sid + headlessvice-rendered dump) is acquired exactly like the
Monty/GoatTracker/lft gates and is in GATE_FIXTURES so CI pre-renders + mounts it.
"""

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
from preframr_tokens.bacc import dmc_format as df
from preframr_tokens.bacc.backends import select_backend
from preframr_tokens.bacc.backends.dmc import DmcBackend
from preframr_tokens.bacc.dmc_serialize import (
    _emit_pattern_toks,
    _read_pattern_toks,
    _tok_delta,
    _tok_lit,
    _tok_lit_len,
    _tok_read,
    _tok_shift,
)
from preframr_tokens.bacc.serialize import REPEAT, TRANSPOSE, _lz_emit_t, _lz_read_t
from preframr_tokens.bacc.pitch import fn_to_grid
from preframr_tokens.bacc.sidemu import load_psid
from tests._dump_fixture import acquire

_ODE_REL = "MUSICIANS/A/Ass_It/Ode_to_Music.sid"
_ODE_URL = "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/A/Ass_It/Ode_to_Music.sid"


class _GtPsid:
    """gt2reloc single-speed shape (play = init + 3) WITHOUT the DMC signature:
    must NOT match DMC (and falls through to GoatTracker)."""

    load_addr = 0x1000
    init_addr = 0x1000
    play_addr = 0x1003
    data = bytes([0x4C, 0xA3, 0x10, 0x4C, 0xA7, 0x10]) + bytes(16)


class _MontyPsid:
    """Hubbard Monty shape (play != init + 3): must NOT match DMC."""

    load_addr = 0x8000
    init_addr = 0x8000
    play_addr = 0x8012
    data = bytes(64)


class _DmcWrongAddr:
    """Carries the DMC signature but loads outside the $1000 build window."""

    load_addr = 0x8000
    init_addr = 0x8000
    play_addr = 0x8003
    data = bytes(16) + b"-PLAYER (C) BRIAN/GRAFFITY!-" + bytes(16)


def test_dmc_matches_only_its_own_signature():
    backend = DmcBackend()
    assert backend.matches(_GtPsid()) is False  # no signature
    assert backend.matches(_MontyPsid()) is False  # wrong play vector + no signature
    assert backend.matches(_DmcWrongAddr()) is False  # wrong load/init/play addrs


class _FakeMem:
    """Minimal image accessor for the format-module unit tests."""

    def __init__(self, base, data):
        self.m = bytearray(0x10000)
        self.m[base : base + len(data)] = data

    def b(self, a):
        return self.m[a]


def test_pattern_parse_emit_roundtrip_all_token_kinds():
    """Every pattern token kind -- note, set-instrument, set-duration, effect (both
    the 2-byte $C0 and 1-byte $D0 parameter forms), tie, rest, end -- parses and
    re-emits byte-exact (the effect $10 form bit drives 1- vs 2-byte consumption)."""
    base = 0x2000
    seq = [0x05, 0x65, 0x82, 0xC3, 0x11, 0x22, 0xD7, 0x44, 0xFD, 0xFE, 0xFF]
    toks, n = df.parse_pattern(_FakeMem(base, bytes(seq)), base)
    assert n == len(seq)
    assert toks == [
        ("note", 5),
        ("ins", 5),
        ("dur", 2),
        ("fx", 3, (0x11, 0x22)),  # $C0 form: 2 parameter bytes
        ("fx", 23, (0x44,)),  # $D0 form ($10 bit set): 1 parameter byte
        ("tie",),
        ("rest",),
        ("end",),
    ]
    assert df.emit_pattern(toks) == seq


def test_orderlist_parse_emit_roundtrip_prefix_and_stop():
    """Orderlist round-trips byte-exact, preserving redundant transpose prefixes
    (DMC re-emits $A0 even when the transpose is unchanged) and the $FE stop."""
    base = 0x2000
    # $A0 -> transpose 0 + pat2; $82 -> transpose -30 + pat3; bare pat1 latches
    # the running transpose (-30, no prefix byte); $FE stop.
    seq = [0xA0, 0x02, 0x82, 0x03, 0x01, 0xFE]
    entries, term, n = df.parse_orderlist(_FakeMem(base, bytes(seq)), base)
    assert term == "stop"
    assert n == len(seq)
    assert entries == [(0, 2, 1), (-30, 3, 1), (-30, 1, 0)]
    assert df.emit_orderlist(entries, term) == seq


def test_pattern_token_serializer_roundtrip_with_fx_and_escape():
    """The pattern-token serializer round-trips effects and the literal-index note
    escape (a note that does not resolve through the freq-table bijection)."""
    freq = [0] * 96
    freq[0] = 0x10C  # only note 0 is grid-resolvable; others force the escape
    anchor = fn_to_grid(freq[0])
    toks = [
        ("note", 0),  # grid-resolvable -> canonical token
        ("note", 5),  # freq 0 -> literal-index escape
        ("fx", 7, (0x12, 0x34)),
        ("tie",),
        ("rest",),
        ("end",),
    ]
    out = []
    _emit_pattern_toks(out, toks, freq, anchor)
    back, consumed = _read_pattern_toks(out, 0)
    assert consumed == len(out)
    assert back == toks


def test_dmc_shared_score_lz_transpose_roundtrip_and_saves():
    """The DMC pattern-token stream now rides the SHARED post-BACC transposed LZ.
    A phrase repeated exactly factors as REPEAT; the same phrase repeated a fifth
    up factors as TRANSPOSE+Delta -- both round-trip byte-exact and emit fewer
    tokens than the raw literal stream."""
    freq = list(range(0x100, 0x100 + 96))  # clean bijection: every note canonical
    anchor = fn_to_grid(freq[0])
    # an all-note phrase: a transposed repeat factors as one TRANSPOSE+Delta (a
    # mixed note/control phrase breaks the run at the non-note tokens, by design)
    phrase = [("note", 10), ("note", 12), ("note", 14), ("note", 15)]
    transposed = [("note", n + 7) for (_, n) in phrase]  # a fifth up
    toks = phrase + phrase + transposed  # literal run, exact repeat, transposed
    out = []
    _lz_emit_t(
        out,
        toks,
        lambda t: _tok_lit_len(t, freq, anchor),
        lambda o, t: _tok_lit(o, t, freq, anchor),
        _tok_delta,
    )
    assert REPEAT in out and TRANSPOSE in out  # both factorings fired
    raw = sum(_tok_lit_len(t, freq, anchor) for t in toks)
    assert len(out) < raw  # the shared LZ is a real, lossless reduction
    back, consumed = _lz_read_t(out, 0, len(toks), _tok_read, _tok_shift)
    assert consumed == len(out)
    assert back == toks  # byte-exact inverse


@pytest.fixture(scope="module")
def ode_paths():
    return acquire(_ODE_REL, _ODE_URL, subtune=1)


def test_select_backend_dispatches_dmc(ode_paths):
    sid, _ = ode_paths
    psid = load_psid(sid)
    assert psid.load_addr == 0x1000
    assert psid.play_addr == psid.init_addr + 3  # same shape as GoatTracker...
    assert select_backend(psid).name == "dmc"  # ...but the signature disambiguates
    # and the DMC signature does NOT claim a plain GoatTracker tune
    assert select_backend(_GtPsid()).name == "goattracker"


def test_ode_to_music_residual_zero(ode_paths):
    """Byte-exact over the whole tune: render(recover(sid)) == dump.

    The render rebuilds the image's song-data regions from the recovered
    abstraction and re-runs the player, so this proves the abstraction round-trips
    losslessly (HARD RULE #0: no raw-byte escape), not merely that the untouched
    image replays."""
    sid, dump = ode_paths
    assert verify_residual(
        sid, dump, CPF, subtune=0
    ), "dmc backend is NOT residual-zero on Ode_to_Music"


def test_ode_to_music_recovers_the_common_abstraction(ode_paths):
    """The recovered program IS the song model: a 3-voice canon orderlist, patterns
    whose notes resolve onto the canonical A440 grid, and instrument records. The
    stored engine template has its song-data regions BLANKED (no raw song bytes)."""
    sid, dump = ode_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    assert program.driver == "dmc"
    song = program.tables["song"]
    # the classic Ode_to_Music 3-voice round (staggered entries), straight from bytes
    pats = [[pat for _, pat, _ in o["entries"]] for o in song["orders"]]
    assert pats == [
        [1, 2, 3, 1, 2],
        [0, 1, 2, 3, 1],
        [0, 0, 1, 2, 3],
    ]
    # every pattern note resolves through the clean freq-table 12-TET bijection
    anchor = fn_to_grid(song["freq"][0])
    for pat in song["patterns"].values():
        for tok in pat["toks"]:
            if tok[0] == "note":
                n = tok[1]
                assert fn_to_grid(song["freq"][n]) == anchor + n
    # the engine template is the VM, with the song-data regions blanked out: the
    # subtune record the model owns is all zero in the stored image.
    load = program.seed["load_addr"]
    a_sub = song["a_sub"] - load
    assert all(program.seed["image"][a_sub + k] == 0 for k in range(8))


def test_ode_to_music_token_roundtrip(ode_paths):
    sid, dump = ode_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    ids = program_to_ids(program)
    assert ids and all(0 <= t < VOCAB for t in ids)
    program2 = ids_to_program(ids, driver="dmc")
    assert program2.driver == "dmc"
    assert program_to_ids(program2) == ids
    assert np.array_equal(render_program(program2), render_program(program))


def test_ode_to_music_token_budget(ode_paths):
    """Token-budget measurement: the score (the musical abstraction) is a small
    fraction of the program; the bulk is the one-time engine template (the VM)."""
    sid, dump = ode_paths
    program = recover_program(sid, dump, CPF, subtune=0)
    brk, frames = measure(program)
    assert brk["total"] == len(program_to_ids(program))
    assert frames > 1000
    assert brk["score"] < brk["total"]  # score is the abstraction, not the bulk
    assert brk["total"] / frames < 5.0
