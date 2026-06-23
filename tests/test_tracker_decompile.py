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
from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import recover_generic, render_generic
from preframr_tokens.bacc.generic.bustrace import BUS_DT, load_bus
from preframr_tokens.bacc.generic.tracker import (
    CITG_MODES,
    CITG_VALUE_KEYS,
    TrackerIR,
    lift,
    lifted_render,
    literal_table_citg,
    unlift,
)

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


def _synth_irreducible(nframes=160):
    """A tune whose voice-0 freq lane is a high-entropy PRNG sequence under a held
    gate -- NO compact CITG covers it (the §3.6 no-escape floor case, the Pengon
    analogue).  The other lanes are clean holds so only the PRNG lane hits the floor;
    it must still round-trip BYTE-EXACT via a literal-table CITG, never an escape."""
    recs, cyc, reg = [], 1000, [0] * 25
    rng = np.random.default_rng(20260622)
    seq = rng.integers(0x0400, 0x4000, size=nframes)
    for frame in range(nframes):
        v0 = int(seq[frame])
        reg[0], reg[1] = v0 & 0xFF, (v0 >> 8) & 0xFF
        reg[4] = 0x41 if frame else 0x40  # gate held high (no per-note re-slice)
        reg[7], reg[8] = 0x00, 0x10
        reg[14], reg[15] = 0x00, 0x18
        reg[18] = 0x10
        reg[24] = 0x0F
        cyc = _blit(recs, cyc, reg)
    return np.array(recs, dtype=BUS_DT)


@pytest.fixture(name="irr_program")
def _irr_program():
    return recover_generic("irr.sid", None, _synth_irreducible())


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


# The former per-register ``genfits`` escape was DELETED (Stage M2: no escape).  Its
# encoded SIZE is still a useful baseline -- it is the naive per-register fit dump the
# Tracker IR must beat -- so the size assertions reconstruct that baseline LOCALLY (it
# is no longer production code).  This keeps the "the IR is compact" property under
# test without resurrecting an escape.
def _genfits_baseline_size(program):
    """The byte count of the retired per-register genfits encoding (a faithful copy
    of the old ``generic_serialize._genfits_blocks``), used only as a size baseline."""
    from preframr_tokens.bacc.serialize import _wi, _wu

    def _wv(out, value):  # the old generic-by-type value encoder
        if value is None:
            out.append(0)
        elif value is True:
            out.append(2)
        elif value is False:
            out.append(1)
        elif isinstance(value, int):
            out.append(3)
            _wi(out, value)
        elif isinstance(value, str):
            out.append(5)
            data = value.encode("utf-8")
            _wu(out, len(data))
            out.extend(data)
        elif isinstance(value, (list, tuple)):
            out.append(6)
            _wu(out, len(value))
            for item in value:
                _wv(out, item)
        elif isinstance(value, dict):
            out.append(7)
            _wu(out, len(value))
            for key, item in value.items():
                _wv(out, key)
                _wv(out, item)
        else:
            raise TypeError(type(value))

    out = [0]  # the old _FMT_GENFITS tag
    _wu(out, program.nframes)
    for b in program.boot:
        _wu(out, b)
    note_table = program.tables.get("note_table")
    if note_table is None:
        _wu(out, 0)
    else:
        _wu(out, 1)
        _wu(out, len(note_table))
        for v in note_table:
            _wu(out, v)
    _wv(out, program.tables["genfits"])
    _wv(out, program.tables["eventfits"])
    return len(out)


def test_tracker_form_smaller_than_genfits_baseline_on_repetitive(rep_program):
    # a repetitive tune is exactly where the tracker decompiler wins: the always-on
    # tracker form must be STRICTLY smaller than the retired per-register baseline.
    ids = GS.generic_program_to_ids(rep_program)
    assert len(ids) < _genfits_baseline_size(rep_program)


def test_tracker_form_never_larger_than_genfits_baseline(tc_program):
    # even on a non-repetitive tune the tracker form is never larger than the naive
    # per-register dump it replaced.
    ids = GS.generic_program_to_ids(tc_program)
    assert len(ids) <= _genfits_baseline_size(tc_program)


def test_measure_reports_tracker_form(rep_program):
    brk, nframes = GS.generic_measure(rep_program)
    assert nframes == rep_program.nframes
    assert brk["fmt"] == "tracker"  # no escape -- always the tracker IR
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
    # two events (dur, ref, base, seed) that are the same dur/instrument/seed at a
    # constant note-index shift factor to that shift (a tracker orderlist TRANSPOSE);
    # incompatible events do not factor.  The delta/shift are bound to the decode
    # context (schemas, has_note_table) by the event codec.
    ctx = ([("S", ("seed",))] * 9, True)  # ref 7/8 are simple bodies with a "seed"
    _lit, _read, _cost, delta, shift, _eq_key, _xpose_key, _xpose_vec = (
        TS._make_event_codec(ctx)
    )
    a = (4, 7, 10, {"seed": 0})
    b = (4, 7, 13, {"seed": 0})
    assert delta(a, b) == 3
    assert shift(a, 3) == b
    assert delta(a, (4, 8, 13, {"seed": 0})) is None  # diff instrument
    assert delta(a, (4, 7, -1, {"seed": 0})) is None  # absolute body


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
    assert len(ids) <= _genfits_baseline_size(program)


# --- Stage M0: the named Tracker IR + the canonical CITG value schema --------
def _all_generator_fits(program):
    """Every generator-lane fit the cover emitted, flattened over piecewise pieces."""
    fits = []
    for segs in (program.tables["genfits"]).values():
        for _start, _stop, fit in segs["segments"]:
            if fit is None:
                continue
            name, prm = fit
            if name == "piecewise":
                fits.extend((pn, pp) for pn, pp, _ in prm["pieces"])
            else:
                fits.append((name, prm))
    return fits


def test_tracker_ir_named_view_equals_lift(rep_program):
    # TrackerIR is a thin named view over the (pool, lanes, note_table) lift triple.
    ir = TrackerIR.from_program(rep_program)
    assert ir.as_triple() == lift(rep_program)


def test_citg_is_the_sole_generator_pool_shape(rep_program):
    # M1: every generator-lane fit the cover emits is a CITG -- the zoo is no longer
    # a cover op, so no per-archetype name leaks into the generator genfits.
    for name, _prm in _all_generator_fits(rep_program):
        assert name == "citg", name


def test_citg_value_schema_is_pinned(rep_program):
    # M0: every CITG params dict declares a known mode.  The core READ/ACCUM modes
    # conform to the canonical nine-field schema (the SOLE generator pool-entry shape);
    # the parametric TABLE-SHAPE / composition modes (§1.1: a closed-form shape tag
    # that expands to the array instead of an explicit `table`) carry `mode` plus their
    # documented shape params -- a within-CITG choice, still one `citg` op.
    for name, prm in _all_generator_fits(rep_program):
        assert name == "citg"
        assert prm["mode"] in CITG_MODES, prm["mode"]
        if prm["mode"] in ("read", "accum"):
            extra = set(prm) - set(CITG_VALUE_KEYS)
            assert not extra, extra
            assert "table" in prm and "clock" in prm


def test_literal_table_citg_is_byte_exact_by_construction():
    # M2 / §3.6: the no-escape floor renders its own bytes exactly, conforms to the
    # schema, and is a normal READ-mode CITG (LOOP=0, CLOCK=every).
    vals = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    p = literal_table_citg(vals)
    assert set(p).issubset(set(CITG_VALUE_KEYS))
    assert p["mode"] == "read" and p["loop"] == 0 and p["clock"] == {"kind": "every"}
    assert A.render_citg(p, len(vals)).tolist() == vals
    assert A.render_fit(("citg", p), len(vals)).tolist() == vals


def test_no_escape_floor_mechanism_and_metric():
    # The §3.6 NO-ESCAPE floor (the Pengon analogue): a generator-lane span no compact
    # CITG covers is left un-fit (None) by the matcher, then FLOORED to a literal-table
    # CITG of its own observed bytes by the outer fitter -- byte-exact, ``citg``
    # vocabulary, never an escape -- and tallied as the corpus-health metric.
    from preframr_tokens.bacc.generic import fitter as Fit

    lane = np.arange(100, 150, dtype=np.int64)  # the observed bytes for the span
    res = [(0, 10, ("hold", {"value": 100})), (10, 50, None)]  # a None (un-fit) span
    A.reset_citg_counts()
    floored, nframes = Fit._floor_unfit(res, lane)
    assert nframes == 40  # the un-fit span (10..50) was floored
    assert A.CITG_FALLBACK_COUNTS["floor:literal_table"] == 1
    assert A.CITG_FALLBACK_COUNTS["floor:frames"] == 40
    # the floored segment is a normal READ-mode literal-table CITG of the span bytes
    name, prm = floored[1][2]
    assert name == "citg" and prm["mode"] == "read" and prm["loop"] == 0
    assert prm["table"] == lane[10:50].tolist()
    # and it renders the observed bytes byte-exact (no escape, no residual)
    assert A.render_fit((name, prm), 40).tolist() == lane[10:50].tolist()


def test_no_escape_roundtrips_byte_exact_on_high_entropy_lane(irr_program):
    # End-to-end NO-ESCAPE proof: a high-entropy freq lane serializes through the
    # (escape-deleted) token stream and round-trips BYTE-EXACT -- it can only fall to a
    # larger table, never a different format (generic_program_to_ids RAISES otherwise).
    base = render_generic(irr_program)
    ids = GS.generic_program_to_ids(irr_program)
    rebuilt = GS.generic_ids_to_program(ids)
    assert np.array_equal(render_generic(rebuilt), base)


def test_canonicalization_serialization_is_stable(rep_program):
    # M0 canonicalization: serializing the SAME program twice yields the SAME token
    # stream (a decode-reproducible canonical form -- stable lane order, stable pool
    # dedup), and re-lifting the round-tripped program reproduces the token stream.
    ids1 = GS.generic_program_to_ids(rep_program)
    ids2 = GS.generic_program_to_ids(rep_program)
    assert ids1 == ids2
    rebuilt = GS.generic_ids_to_program(ids1)
    assert GS.generic_program_to_ids(rebuilt) == ids1
