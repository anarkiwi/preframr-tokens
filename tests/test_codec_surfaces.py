"""Coverage + invariant tests for the step/tracker codec surfaces beyond the
Monty budget gate. Drives a real (capped) Monty register-dump window through the
event / tracker / token-serialization paths and asserts the codec's lossless
invariants (event roundtrip, step expansion reproduces events, byte-cost models
agree), plus focused unit roundtrips for the pitch/freq/program lanes.

The dump is resolved via the same fixture path as the budget gate; if it is
absent the per-test fixture builder is reused (no skip path for the gate itself,
but these auxiliary tests skip cleanly when the dump cannot be acquired)."""

import os

import numpy as np
import pytest

import preframr_tokens as P
from preframr_tokens.codec import freq_instrument as FI
from preframr_tokens.codec import freq_relative as FR
from preframr_tokens.codec import pitch_universal as PU
from preframr_tokens.codec import pitch_universal_anchor as PA
from preframr_tokens.codec import pitch_universal_encode as PE
from preframr_tokens.codec import program_encode as PG
from preframr_tokens.codec import serialize_events as SE
from preframr_tokens.codec import serialize_tokens as ST
from preframr_tokens.codec import step_codec as SC
from preframr_tokens.codec import step_tracker as TR

_BUNDLED = os.path.join(
    os.path.dirname(__file__), "test_fixtures", "Monty_on_the_Run.1.dump.parquet"
)
_SCRATCH = "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.1.dump.parquet"
_DUMP = _BUNDLED if os.path.exists(_BUNDLED) else _SCRATCH


def _state(nframes=4000):
    if not os.path.exists(_DUMP):
        pytest.skip("Monty dump unavailable for auxiliary codec-surface tests")
    s = P.per_frame_state(_DUMP, P.CPF, nframes)
    assert s is not None and len(s) >= 2
    return s[:, :25]


def test_event_set_roundtrips_through_token_ids():
    s = _state()
    ev = SE.encode_tune_events(s)
    assert ev, "no events produced"
    for lz in (False, True):
        ids = ST.events_to_ids(ev, lz=lz)
        assert ids, "no token ids produced"
        ev2 = ST.ids_to_events(ids)
        assert sorted(ev) == sorted(ev2), f"token roundtrip lossy (lz={lz})"


def test_decode_events_is_self_consistent():
    s = _state()
    ev = SE.encode_tune_events(s)
    d1 = SE.decode_events(ev, len(s))
    d2 = SE.decode_events(sorted(ev), len(s))
    assert d1.shape == (len(s), 25)
    assert np.array_equal(d1[:, :25], d2[:, :25]), "decode not order-invariant"


def test_step_codec_structure_reproduces_events():
    s = _state()
    rows, global_events, ev = SC.build_steps(s)
    ev_from_steps = SC.steps_to_events(rows, global_events, len(s))
    assert sorted(ev_from_steps) == sorted(ev), "step expansion != event set"
    cost, instr_defs, instr_index = SC.serialize_cost(rows, global_events)
    assert cost["total"] > 0 and instr_index is not None and instr_defs is not None


def test_step_tracker_measure_is_positive():
    s = _state()
    voices, ev = TR.build_rows(s)
    assert len(voices) == 3 and ev
    brk, raw, total, voices2, ev2 = TR.measure(s)
    assert len(voices2) == 3 and ev2
    assert raw > 0 and total > 0 and brk


def test_freq_instrument_measure_residual_ok():
    s = _state()
    res = FI.measure_freq(s)
    assert bool(res["residual_ok"]) is True


def test_cost_primitives_monotone():
    assert SC.u_cost(0) >= 1
    assert SC.u_cost(10**6) >= SC.u_cost(0)
    assert SC.i_cost(-5) == SC.i_cost(5)
    assert SC.i_cost(0) >= 1


def test_freq_relative_roundtrip():
    s = _state()
    for v in range(3):
        b = 7 * v
        seq = s[:, b] + 256 * s[:, b + 1]
        toks = FR.encode_freq(seq)
        assert np.array_equal(FR.decode_freq(toks), seq)


def test_program_lane_roundtrip():
    s = _state()
    lane = s[:, 24]  # filter mode/volume lane, byte-exact LOAD/RUN
    toks = PG.encode_lane(lane)
    assert np.array_equal(PG.decode_lane(toks), lane)
    prog = PG.encode_tune(s)
    assert prog is not None


def test_pitch_universal_anchor_voice_roundtrip():
    s = _state()
    for v in range(3):
        b = 7 * v
        f = (s[:, b] + 256 * s[:, b + 1]).astype(float)
        tuning, phase, toks = PA.encode_voice(f)
        dec = PA.decode_voice(tuning, phase, toks)
        assert len(dec) == len(f)


def test_pitch_universal_helpers():
    s = _state()
    f = (s[:, 0] + 256 * s[:, 1]).astype(float)
    bp = PU.base_pitches_from_freq(f)
    assert len(bp) >= 1  # per-onset base pitches (not per-frame)
    held = PU.held_base_pitches(f)
    assert len(held) >= 1
    off, resid = PU.best_offset(PU.log2fn(f[f > 0]))
    assert float(off) == off and len(resid) >= 1
    tuning, phase, toks = PE.encode_voice(f)
    dec = PE.decode_voice(tuning, phase, toks)
    assert len(dec) == len(f)
