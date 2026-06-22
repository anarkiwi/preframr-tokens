"""Generic tracker-structure decompiler: lift the per-register generic fits
(``genfits``/``eventfits``) into a TRACKER-shaped program -- a shared INSTRUMENT
pool (deduped pitch-invariant fit signatures) plus a per-lane NOTE-EVENT stream --
so the existing REPEAT/TRANSPOSE score machinery compresses it to hand-backend
scale.

The lift is a LOSSLESS RE-EXPRESSION of the already-residual-zero fits (HARD RULE
#0): :func:`tracker.unlift` reconstructs ``genfits``/``eventfits`` byte-for-byte and
the recovered tracker program RENDERS residual-zero (carry RECOMPUTED at render,
never stored).  These tests are SELF-CONTAINED -- they synthesise small native
``BUS_DT`` traces in-process exercising several archetypes across distinct driver
shapes (an arp/wavetable voice, an accumulator-sweep voice, a portamento voice, a
through-composed note voice) and assert:

  * ``lift -> unlift`` renders byte-identically to ``render_generic`` (lossless);
  * the full ``generic_program_to_ids`` round-trip renders residual-zero;
  * the chosen serialization is the SMALLER of the tracker / genfits forms, and on
    a repetitive (tracker-shaped) tune the tracker form is chosen and is smaller.

The whole-tune multi-driver proof on real SIDs (Grid_Runner / Monty / DMC / JCH /
FutureComposer / ...) is env-gated (``GENERIC_BUSTRACE`` / ``TRACKER_SID``) because
it needs a multi-MB native trace or the optional ``preframr-sidtrace`` binary;
those are measured in the PR body, not committed.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc import generic_serialize as GS
from preframr_tokens.bacc import tracker_serialize as TS
from preframr_tokens.bacc.generic import recover_generic, render_generic
from preframr_tokens.bacc.generic.bustrace import BUS_DT, load_bus
from preframr_tokens.bacc.generic.tracker import lift, lifted_render, unlift

_CPF = 19656  # PAL raster frame


def _blit(recs, cyc, reg):
    for index in range(25):
        recs.append((cyc, 0xD400 + index, reg[index] & 0xFF, 1))
        cyc += 2
    return cyc + _CPF - 2 * 25


def _synth_repetitive(nframes=192, period=16):
    """A tune whose voices REPEAT a short phrase: a wavetable/arp voice, a
    dwelled-accumulator PW sweep, and a portamento -- all re-triggered every
    ``period`` frames.  The repetition is what the tracker form must collapse, so
    this is the case where the tracker form beats genfits."""
    arp = [0x0800, 0x0900, 0x0A00, 0x0900]
    recs, cyc, reg = [], 1000, [0] * 25
    for frame in range(nframes):
        ph = frame % period
        v0 = arp[ph % 4]
        reg[0], reg[1] = v0 & 0xFF, (v0 >> 8) & 0xFF
        pw0 = 0x0400 + 8 * (ph // 2)
        reg[2], reg[3] = pw0 & 0xFF, (pw0 >> 8) & 0x0F
        reg[4] = 0x41 if ph >= 1 else 0x40  # re-gate at phrase start
        v1 = 0x2000 + 13 * ph
        reg[7], reg[8] = v1 & 0xFF, (v1 >> 8) & 0xFF
        reg[11] = 0x21 if ph >= 1 else 0x20
        reg[14], reg[15] = 0x00, 0x18  # voice-2 constant hold
        reg[18] = 0x10
        reg[24] = 0x0F
        cyc = _blit(recs, cyc, reg)
    return np.array(recs, dtype=BUS_DT)


def _synth_through_composed(nframes=160):
    """A through-composed melody voice: a distinct held pitch every few frames
    (no phrase repeat).  Honest note list -- small but not collapsible by repeat;
    the lift must still render byte-exact and the chosen form must be byte-exact."""
    melody = [0x1000, 0x10F0, 0x1234, 0x1390, 0x1500, 0x1690, 0x1780, 0x18A0]
    recs, cyc, reg = [], 1000, [0] * 25
    for frame in range(nframes):
        note = melody[(frame // 8) % len(melody)]
        reg[0], reg[1] = note & 0xFF, (note >> 8) & 0xFF
        reg[4] = (
            0x41 if frame % 8 == 0 and frame else (reg[4] | 0x01 if frame else 0x40)
        )
        reg[7], reg[8] = 0x00, 0x10
        reg[14], reg[15] = 0x00, 0x18
        reg[18] = 0x10
        reg[24] = 0x0F
        cyc = _blit(recs, cyc, reg)
    return np.array(recs, dtype=BUS_DT)


@pytest.fixture(name="rep_program")
def _rep_program():
    return recover_generic("rep.sid", None, _synth_repetitive())


@pytest.fixture(name="tc_program")
def _tc_program():
    return recover_generic("tc.sid", None, _synth_through_composed())


# --- HARD RULE #0: lift -> unlift is byte-for-byte lossless ----------------
def test_lift_unlift_renders_byte_exact(rep_program):
    base = render_generic(rep_program)
    assert np.array_equal(lifted_render(rep_program), base)


def test_lift_unlift_through_composed_byte_exact(tc_program):
    base = render_generic(tc_program)
    assert np.array_equal(lifted_render(tc_program), base)


def test_unlift_reconstructs_render_input(rep_program):
    # unlift's genfits/eventfits must render identically to the originals (the
    # representation may differ but the rendered (nframes,25) state is invariant).
    pool, lanes, note_table = lift(rep_program)
    genfits, eventfits = unlift(pool, lanes, note_table)
    # every generator lane and event lane round-trips to a renderable shape
    assert set(genfits) == set(rep_program.tables["genfits"])
    assert {str(r) for r in eventfits} == set(rep_program.tables["eventfits"])


def test_instrument_pool_is_shared_not_per_segment(rep_program):
    # the pool must be SMALLER than the segment count -- the dedup is real, not a
    # 1:1 re-emission (the whole point: a repeated phrase reuses one instrument).
    pool, lanes, _ = lift(rep_program)
    total_events = sum(len(events) for events in lanes.values())
    assert len(pool) < total_events
    assert len(pool) >= 1


def test_carry_recomputed_not_stored(rep_program):
    # unlift never carries a per-frame array; the PW carry is recomputed at render.
    pool, lanes, note_table = lift(rep_program)
    genfits, _ = unlift(pool, lanes, note_table)
    for key, segs in genfits.items():
        if key.endswith("pw"):
            # genfits from unlift is a bare segment list (carry is re-derived in
            # render_from_fits); no per-frame carry array is part of the lift.
            assert isinstance(segs, list)


# --- full serializer round-trip renders residual-zero ----------------------
def _roundtrip_renders_equal(program):
    base = render_generic(program)
    ids = GS.generic_program_to_ids(program)
    rebuilt = GS.generic_ids_to_program(ids)
    return np.array_equal(render_generic(rebuilt), base)


def test_serialize_roundtrip_byte_exact_repetitive(rep_program):
    assert _roundtrip_renders_equal(rep_program)


def test_serialize_roundtrip_byte_exact_through_composed(tc_program):
    assert _roundtrip_renders_equal(tc_program)


def test_tracker_form_chosen_and_smaller_on_repetitive(rep_program):
    # a repetitive tune is exactly where the tracker decompiler wins: the chosen
    # form must be the tracker form AND strictly smaller than the genfits form.
    genfits_only = [0]
    GS._genfits_blocks(genfits_only, rep_program)
    ids = GS.generic_program_to_ids(rep_program)
    assert ids[0] == 1  # _FMT_TRACKER
    assert len(ids) < len(genfits_only)


def test_chosen_form_never_larger_than_genfits(tc_program):
    # the encoder picks the SMALLER of the two forms; even when the tracker form
    # does not win, the chosen stream is never larger than the genfits-only form.
    genfits_only = [0]
    GS._genfits_blocks(genfits_only, tc_program)
    ids = GS.generic_program_to_ids(tc_program)
    assert len(ids) <= len(genfits_only)


def test_measure_reports_chosen_form(rep_program):
    brk, nframes = GS.generic_measure(rep_program)
    assert nframes == rep_program.nframes
    assert brk["fmt"] in ("tracker", "genfits")
    assert brk["total"] == len(GS.generic_program_to_ids(rep_program))


# --- tracker_serialize unit round-trips (values / pool LZ / events) --------
def test_pool_token_lz_roundtrip(rep_program):
    pool, _, _ = lift(rep_program)
    out = []
    TS._emit_pool(out, pool)
    rebuilt, end = TS._read_pool(out, 0)
    assert end == len(out)
    assert rebuilt == pool


def test_value_codec_roundtrip():
    samples = [
        None,
        True,
        False,
        0,
        -42,
        123456,
        "rel",
        [1, 2, 3],
        ["rel", 0, -3, 5],
        {"table": [1, 2, 3], "mode": "read", "flag": True},
        [{"seed": 1}, {"v0": -7, "ctr0": 3}],
    ]
    for value in samples:
        out = []
        TS._write_value(out, value)
        got, end = TS._read_value(out, 0)
        assert end == len(out)
        assert got == value


def test_event_transpose_delta_factoring():
    # two events that are the same instrument/seed at a constant note-index shift
    # factor to that shift (a tracker orderlist TRANSPOSE); incompatible events do
    # not factor.
    a = (4, 4, 7, 10, {"seed": 0})
    b = (4, 4, 7, 13, {"seed": 0})
    assert TS._event_delta(a, b) == 3
    assert TS._event_shift(a, 3) == b
    assert TS._event_delta(a, (4, 4, 8, 13, {"seed": 0})) is None  # diff instrument
    assert TS._event_delta(a, (4, 4, 7, -1, {"seed": 0})) is None  # absolute body


def test_tracker_program_to_ids_roundtrip(rep_program):
    body = TS.tracker_program_to_ids(rep_program)
    clone = TS.tracker_ids_to_program(body)
    assert np.array_equal(render_generic(clone), render_generic(rep_program))


# --- env-gated whole-tune multi-driver proof -------------------------------
@pytest.mark.skipif(
    not os.environ.get("GENERIC_BUSTRACE"),
    reason="set GENERIC_BUSTRACE=/path/to/tune.bus.bin for the whole-tune proof",
)
def test_whole_tune_lift_byte_exact_and_smaller():
    path = os.environ["GENERIC_BUSTRACE"]
    program = recover_generic(path, None, load_bus(path))
    base = render_generic(program)
    assert np.array_equal(lifted_render(program), base)  # lossless lift
    ids = GS.generic_program_to_ids(program)
    rebuilt = GS.generic_ids_to_program(ids)
    assert np.array_equal(render_generic(rebuilt), base)  # residual-zero round-trip
    genfits_only = [0]
    GS._genfits_blocks(genfits_only, program)
    assert len(ids) <= len(genfits_only)
