"""The canonical Tracker IR (Stage 2): lift/build/unlift round-trip + serializer.

These tests cover :mod:`preframr_tokens.bacc.tracker_ir` (the per-voice BUNDLED,
pitch-factored, pool-deduped IR built from the byte-exact Stage-1 lane covers) and
its serializer :mod:`preframr_tokens.bacc.tracker_serialize`:

  * ``lift -> unlift -> render`` is byte-for-byte lossless (HARD RULE #0);
  * the IR bundles a voice's aligned lanes and keeps a dense/misaligned lane free;
  * the per-ref seed SCHEMA is consistent and the row codec round-trips;
  * the full ``generic_program_to_ids`` round-trip renders byte-exact and the chosen
    stream encoding (item-LZ REPEAT/TRANSPOSE vs token-LZ) is the smaller.

The whole-tune tok/frame GATE on the real gate tunes (A Mind Is Born / Grid_Runner /
DefMon) needs the optional ``preframr-sidtrace`` binary + a few seconds of cover, so
it is env-gated (``TRACKER_SID_DIR`` pointing at an HVSC ``C64Music`` root); the
default CI exercises only the self-contained synthetic round-trips.  This folds the
former ``generic/validate_cover.py`` script into a proper, opt-in test.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc import generic_serialize as GS
from preframr_tokens.bacc import tracker_ir as IR
from preframr_tokens.bacc import tracker_serialize as TS
from preframr_tokens.bacc.generic import recover_generic, render_generic
from preframr_tokens.bacc.generic.tracker import render_from_fits

# reuse the self-contained synthetic bus traces from the decompile test
from tests.test_tracker_decompile import (
    _synth_irreducible,
    _synth_repetitive,
    _synth_through_composed,
)

_SYNTHS = {
    "rep": _synth_repetitive,
    "tc": _synth_through_composed,
    "irr": _synth_irreducible,
}


def _program(name):
    return recover_generic(f"{name}.sid", None, _SYNTHS[name]())


# --- HARD RULE #0: lift -> unlift -> render is byte-for-byte lossless --------
@pytest.mark.parametrize("name", list(_SYNTHS))
def test_ir_lift_unlift_render_byte_exact(name):
    prog = _program(name)
    base = render_generic(prog)
    ir = IR.lift(base, prog.tables.get("note_table"), prog.nframes, list(prog.boot))
    assert np.array_equal(IR.render(ir), base)


@pytest.mark.parametrize("name", list(_SYNTHS))
def test_ir_unlift_renders_through_render_from_fits(name):
    # unlift's genfits/eventfits render byte-exact through the per-register renderer.
    prog = _program(name)
    base = render_generic(prog)
    ir = IR.lift(base, prog.tables.get("note_table"))
    genfits, eventfits = IR.unlift(ir)
    rendered = render_from_fits(genfits, eventfits, ir.note_table, ir.nframes)
    assert np.array_equal(rendered, base)


# --- the IR structure: pool dedup + per-voice bundling ----------------------
def test_pool_is_shared_not_per_segment():
    ir = IR.lift(render_generic(_program("rep")))
    nseg = sum(len(t.rows) for t in ir.voices)
    nseg += sum(len(ev) for t in ir.voices for ev in t.free.values())
    nseg += sum(len(ev) for ev in ir.globals.values())
    assert 1 <= len(ir.pool) < nseg  # the dedup is real, not a 1:1 re-emission


def test_bundling_folds_aligned_lanes_and_frees_misaligned():
    # the repetitive tune's voices have aligned freq/pw/ctrl/ad/sr -> bundled; a voice
    # whose pw has its own dense structure keeps it free (chosen per voice).
    ir = IR.lift(render_generic(_program("rep")))
    bundled = [set(t.bundled) for t in ir.voices]
    free = [set(t.free) for t in ir.voices]
    for v in range(3):
        # every CLASS is either bundled or free, never both, and the union is total.
        assert bundled[v].isdisjoint(free[v])
        assert bundled[v] | free[v] == set(IR.CLASSES)


def test_build_ir_synth_pitch_toggle_both_byte_exact():
    # both the pitch-factored and the plain build are byte-exact (the serializer picks
    # the smaller; correctness must not depend on the choice).
    base = render_generic(_program("rep"))
    covers = IR.cover_all_lanes(base, None)
    boot = [int(v) for v in base[0]]
    for synth in (False, True):
        ir = IR.build_ir(covers, None, len(base), boot, synth_pitch=synth)
        assert np.array_equal(IR.render(ir), base)


# --- sibling re-slice / spine-aligned bundling (the forced-breakpoint lever) -------
def _state_midnote_sibling(nframes=192, note=16):
    """A (nframes, 25) state whose voice-0 freq is a clean held-note melody (the spine)
    and whose ctrl changes BOTH at note onsets AND mid-note -- so re-slicing ctrl at the
    freq onsets leaves >1 ctrl segment inside some notes (the PIECEWISE fold case)."""
    pitches = [0x1000, 0x1200, 0x1400, 0x1600]
    state = np.zeros((nframes, 25), dtype=np.int64)
    for f in range(nframes):
        n = (f // note) % len(pitches)
        v0 = pitches[n]
        state[f, 0], state[f, 1] = v0 & 0xFF, (v0 >> 8) & 0xFF
        # ctrl: re-gate at the note onset, then flip the test-bit MID-note (a second,
        # off-spine ctrl change inside the note) -- the structure the spine must re-slice.
        gate = 0x41
        if f % note >= note // 2:
            gate = 0x49  # mid-note ctrl change (test bit set)
        state[f, 4] = gate
        state[f, 5], state[f, 6] = 0x00, 0xF0  # ad/sr held
        state[f, 14], state[f, 15] = 0x00, 0x18  # voice-2 hold (keeps render happy)
        state[f, 24] = 0x0F
    return state


def test_forced_breakpoint_cover_is_byte_exact_and_contained():
    from preframr_tokens.bacc.generic import archetypes as A
    from preframr_tokens.bacc.generic import cover as C

    state = _state_midnote_sibling()
    ctrl = state[:, 4]
    fcov = C.cover_lane(
        state[:, 0] + (state[:, 1] << 8), 0xFFFF, None, None, None, True
    )
    onsets = {s for s, _t, _f in fcov}
    forced = C.cover_lane(ctrl, 0xFF, None, None, None, False, force_breakpoints=onsets)
    # byte-exact and no segment straddles a forced onset
    out = np.zeros(len(ctrl), dtype=np.int64)
    for s, t, fit in forced:
        out[s:t] = A.render_fit(fit, t - s, None, None, 0)[: t - s]
    assert np.array_equal(out, ctrl)
    assert all(not (s < o < t) for s, t, _f in forced for o in onsets)


def test_span_fits_groups_multi_segment_notes_into_piecewise():
    state = _state_midnote_sibling()
    covers = IR.cover_all_lanes(state, None)
    fcov = covers["0:freq"]
    fits = IR._span_fits(covers, 0, "ctrl", fcov)
    assert fits is not None and len(fits) == len(fcov)  # exactly one fit per note
    # a note with a mid-note ctrl change folds to a piecewise row entry
    assert any(f[0] == "piecewise" for f in fits)


def test_midnote_sibling_lift_unlift_render_byte_exact():
    # the whole IR (whatever the cost test bundles, including a piecewise-folded ctrl)
    # reconstructs byte-exact through unlift -> render (HARD RULE #0).
    state = _state_midnote_sibling()
    nf = len(state)
    boot = [int(v) for v in state[0]]
    for synth in (False, True):
        ir = IR.lift(state, None, nf, boot, synth_pitch=synth)
        assert np.array_equal(IR.render(ir), state)
        # and through the serializer round-trip
        ir2 = TS._ids_to_ir(TS._ir_to_ids(ir))
        assert np.array_equal(IR.render(ir2), state)


# --- the serializer: schema codec + stream modes + full round-trip ----------
@pytest.mark.parametrize("name", list(_SYNTHS))
def test_ir_codec_roundtrips(name):
    ir = IR.lift(render_generic(_program(name)))
    ids = TS._ir_to_ids(ir)
    ir2 = TS._ids_to_ir(ids)
    # the reconstructed IR renders byte-exact (the canonical equality is render-level).
    assert np.array_equal(IR.render(ir2), IR.render(ir))


def test_seed_schema_is_consistent_per_ref():
    # every event referencing a pool entry carries the SAME seed-key schema (the struct
    # determines its seed keys) -- the invariant the compact row codec relies on.
    ir = IR.lift(render_generic(_program("rep")))
    schemas = TS._collect_schemas(ir)  # raises on inconsistency
    assert len(schemas) == len(ir.pool)


def test_same_struct_different_seed_schema_gets_distinct_refs():
    # REGRESSION: two fits with an IDENTICAL struct but a DIFFERENT seed-key SET (one note's
    # citg carries an optional ``phase`` residue, another's does not -- both strip to the same
    # struct) must NOT collide on one pool ref, or _collect_schemas raises "inconsistent seed
    # schema ('S',('seed','phase')) vs ('S',('seed',))".  The seed-schema discriminator in
    # _fit_to_pool gives them distinct refs (identical struct entries), each with ONE schema.
    ref_of, pool = IR._pool_builder()
    common = {
        "mode": "accum",
        "table": [1, 2, 3],
        "clock": {"kind": "every"},
        "loop": 0,
    }
    fit_phase = ("citg", {**common, "seed": 5, "phase": 3})  # carries phase residue
    fit_plain = ("citg", {**common, "seed": 7})  # no phase residue
    r1, b1, s1 = IR._fit_to_pool(fit_phase, False, None, ref_of)
    r2, b2, s2 = IR._fit_to_pool(fit_plain, False, None, ref_of)
    assert r1 != r2, "same struct + different seed schema must not share a ref"

    ir = IR.TrackerIR(
        note_table=None,
        pool=pool,
        voices=[
            IR.VoiceTrack(bundled=[], rows=[], free={"pw": [(10, r1, b1, s1)]}),
            IR.VoiceTrack(bundled=[], rows=[], free={"pw": [(10, r2, b2, s2)]}),
            IR.VoiceTrack(bundled=[], rows=[], free={}),
        ],
        globals={21: [], 22: [], 23: [], 24: []},
        nframes=20,
        boot=[0] * 25,
    )
    schemas = TS._collect_schemas(ir)  # must NOT raise
    assert len(schemas) == len(ir.pool)
    ids = TS._ir_to_ids(ir)  # the full codec must round-trip the distinct schemas
    back = TS._ids_to_ir(ids)
    assert back.pool == ir.pool
    assert back.voices[0].free["pw"] == [(10, r1, b1, s1)]
    assert back.voices[1].free["pw"] == [(10, r2, b2, s2)]


def test_value_and_pool_codec_roundtrip():
    samples = [
        None,
        True,
        False,
        -42,
        123456,
        "rel",
        [1, 2, 3],
        ["rel", 0, -3, 5],
        {"table": [1, 2, 3], "mode": "read", "flag": True},
        # _T_IARR edge cases: empty list, a bool-bearing list (NOT an int array),
        # a nested list (mixed), and a long signed table (the dense-modulation payload).
        [],
        [True, 1, 0],
        [[1, 2], [3, 4]],
        [-127, 0, 127, -254, 0, 88, -32],
    ]
    for value in samples:
        out = []
        TS._write_value(out, value)
        got, end = TS._read_value(out, 0)
        assert end == len(out) and got == value
    # the homogeneous-int array (_T_IARR) drops the per-element type tag, so it is
    # strictly smaller than the generic element-wise list encoding would be.
    table = list(range(-20, 21))
    iarr = []
    TS._write_value(iarr, table)
    assert iarr[0] == TS._T_IARR
    generic_len = (
        1 + TS._u_len(len(table)) + sum(2 for _ in table)
    )  # _T_LIST + per-int tag
    assert len(iarr) < generic_len
    # a list that is NOT a pure int array keeps the generic _T_LIST tag.
    mixed = []
    TS._write_value(mixed, ["rel", 0, -3])
    assert mixed[0] == TS._T_LIST
    ir = IR.lift(render_generic(_program("rep")))
    out = []
    TS._emit_pool(out, ir.pool)
    rebuilt, end = TS._read_pool(out, 0)
    assert end == len(out) and rebuilt == ir.pool


def test_row_transpose_delta_and_shift():
    # a bundled row codec factors a constant spine-pitch shift (TRANSPOSE) and is a
    # no-op on the absolute (base -1) lanes.
    ctx = ([("S", ("seed",))] * 4, True)
    _lit, _read, _cost, delta, shift = TS._make_row_codec(ctx, 2)
    a = (8, [0, 1], [5, -1], [{"seed": 0}, {"seed": 9}])
    b = (8, [0, 1], [7, -1], [{"seed": 0}, {"seed": 9}])
    assert delta(a, b) == 2  # spine base 5 -> 7; the absolute lane (-1) unchanged
    assert shift(a, 2) == b


@pytest.mark.parametrize("name", list(_SYNTHS))
def test_full_generic_roundtrip_byte_exact(name):
    prog = _program(name)
    base = render_generic(prog)
    ids = GS.generic_program_to_ids(prog)  # includes the render-equality self-check
    rebuilt = GS.generic_ids_to_program(ids)
    assert np.array_equal(render_generic(rebuilt), base)


def test_measure_matches_total():
    prog = _program("rep")
    brk, nframes = GS.generic_measure(prog)
    assert nframes == prog.nframes
    assert brk["fmt"] == "tracker"
    assert brk["total"] == len(GS.generic_program_to_ids(prog))


# --- env-gated whole-tune tok/frame GATE (folds validate_cover.py) ----------
# Each tune has (rel_path, ceiling) where ``ceiling`` is a REGRESSION ceiling on
# tok/frame, not an aspiration.  Byte-exactness (below) is the MANDATORY invariant;
# the ceiling locks in the achieved compactness so a future change cannot silently
# inflate the stream.
#
# The algorithmic / pattern tunes (A Mind Is Born, DefMon 20_Years) are well under
# 1.0 tok/frame.  Grid_Runner is a genuine high-ENTROPY case: ~every frame carries
# fresh dense PWM/vibrato modulation (voice pw lanes change on 90 %+ of frames over
# only a handful of distinct values), so its content sits near the entropy floor a
# general compressor reaches -- a column-major LZMA of the raw register state scores
# ~0.96 tok/frame at 2500 frames (and higher at this 3000-frame density).  No
# structural primitive can beat a pure entropy coder on it, and the model-facing IR
# trades some compactness for a LEARNABLE structure, so its ceiling is set above that
# floor rather than at the unreachable <1.0.  The duplicate-frame redundancy a tracker
# orderlist would target is ALREADY collapsed by the per-lane backward-LZ + shared
# instrument pool (measured: an explicit frame-range orderlist is strictly larger,
# because its copy references cost more than the LZ back-references they replace).
_SID_DIR = os.environ.get("TRACKER_SID_DIR")
_GATE_TUNES = {
    "A_Mind_Is_Born": ("MUSICIANS/L/Lft/A_Mind_Is_Born.sid", 0.5),
    "Grid_Runner": ("MUSICIANS/J/Jammer/Grid_Runner.sid", 1.8),
    "20_Years_Is_Nothing": ("MUSICIANS/G/Goto80/20_Years_Is_Nothing.sid", 1.0),
}


@pytest.mark.skipif(
    not _SID_DIR,
    reason="set TRACKER_SID_DIR to an HVSC C64Music root for the whole-tune tok/frame gate",
)
@pytest.mark.parametrize("tune", list(_GATE_TUNES))
def test_whole_tune_byte_exact_and_tok_per_frame(tune):
    rel, ceiling = _GATE_TUNES[tune]
    os.environ.setdefault(
        "SIDTRACE_BIN", "/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace"
    )
    from preframr_tokens.bacc.generic.sidtrace import (
        sidwr_state,
        sidwr_to_bus,
        run_sidtrace,
    )

    sw, _ = run_sidtrace(os.path.join(_SID_DIR, rel), f"/tmp/{tune}", 1, 3000)
    _state, t0 = sidwr_state(sw)
    prog = recover_generic(f"{tune}.sid", None, sidwr_to_bus(sw), t0=t0)
    base = render_generic(prog)
    ids = GS.generic_program_to_ids(prog)  # byte-exact verify inside
    rebuilt = GS.generic_ids_to_program(ids)
    assert np.array_equal(render_generic(rebuilt), base)  # MANDATORY: byte-exact
    tpf = len(ids) / prog.nframes
    assert tpf < ceiling, f"{tune}: {tpf:.4f} tok/frame >= ceiling {ceiling}"


# --- env-gated SPARSE-WRITER framing regression (locks in the sidwr_state fix) ----
# A SPARSE writer plays its routine every raster frame but WRITES the SID only on a
# fraction of them (Master Composer: ~234 write-bursts across ~2300 play-calls).  The
# former blit-group framing yielded one row per WRITE, dropping the held-value frames
# between writes, so the fixed boot/pool/header cost blew tok/frame to >4.  Framing at
# the detected play period with forward-fill (one row per play-call, values held
# between writes -- the SAME grid the corpus dump uses) recovers the true ~2300 frames
# and the tune drops well under 1 tok/frame, byte-exact.  This pins both invariants:
# the period-framed frame count and the resulting tok/frame, so a regression to the
# write-burst framing (the original bug) is caught.  It uses the prompt's encode path
# (sidwr_state -> tracker IR lift -> tracker serialize), which is the one the framing
# feeds; the render-equality round-trip is the byte-exact gate (HARD RULE #0).
_SPARSE_TUNES = {
    # tune: (rel_path, play_period_cycles, min_frames, tok/frame ceiling)
    "Master_Composer": ("MUSICIANS/0-9/2121/8_Days_a_Week.sid", 19656, 2000, 1.0),
}


@pytest.mark.skipif(
    not _SID_DIR,
    reason="set TRACKER_SID_DIR to an HVSC C64Music root for the sparse-writer gate",
)
@pytest.mark.parametrize("tune", list(_SPARSE_TUNES))
def test_sparse_writer_period_framed_byte_exact_and_under_one(tune):
    rel, _period, min_frames, ceiling = _SPARSE_TUNES[tune]
    os.environ.setdefault(
        "SIDTRACE_BIN", "/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace"
    )
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace, sidwr_state
    from preframr_tokens.bacc.generic.tracker import render_from_fits

    sw, _ = run_sidtrace(os.path.join(_SID_DIR, rel), f"/tmp/{tune}_sparse", 1, 2500)
    state, _t0 = sidwr_state(sw)
    nframes = len(state)
    boot = [int(v) for v in state[0]]
    # The period framing recovers one row per play-call (not one per sparse write).
    assert nframes >= min_frames, f"{tune}: {nframes} frames (expected period-framed)"
    # Byte-exact: lift -> unlift -> render reproduces the (re-framed) state exactly.
    genfits, eventfits = IR.unlift(IR.lift(state, None, nframes, boot))
    rendered = render_from_fits(genfits, eventfits, None, nframes)
    assert np.array_equal(rendered, state), f"{tune}: render != state (not byte-exact)"
    # tok/frame: the production path picks min over synth_pitch True/False.
    best = min(
        len(TS._ir_to_ids(IR.lift(state, None, nframes, boot, synth_pitch=sp)))
        for sp in (True, False)
    )
    tpf = best / nframes
    assert tpf < ceiling, f"{tune}: {tpf:.4f} tok/frame >= ceiling {ceiling}"
