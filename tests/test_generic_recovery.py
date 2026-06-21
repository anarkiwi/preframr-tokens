"""Driver-agnostic generic recovery from the trusted preframr-sidtrace bus trace.

The recovery (``bacc/generic``) reconstructs the per-frame 25-register state from
a native ``.bus.bin`` and fits a GENERIC per-register BACC program with no
per-driver constants; the self-contained ``render_generic`` reproduces the
bus-state byte-exact (residual=0).

These tests are SELF-CONTAINED: the default tests synthesise a small native
``BUS_DT`` trace in-process (no committed blob, no container), exercise the
archetype library + the full recover/render/residual path, and assert
residual-zero.  The whole-tune residual-zero proof on a real GoatTracker / Hubbard
SID is env-gated (``GENERIC_BUSTRACE``) -- it needs a multi-MB native trace that
is never committed; document how to produce it with ``preframr-sidtrace`` below.

To run the gated whole-tune proof on a cached trace::

    # produce a native .bus.bin (Phase 1 tool; one subtune, N frames):
    #   build/sidtrace Grid_Runner.sid 0 1504 /tmp/grid
    GENERIC_BUSTRACE=/tmp/grid.bus.bin pytest tests/test_generic_recovery.py
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import recover_generic, render_generic, residual
from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import fitter as F
from preframr_tokens.bacc.generic.busstate import per_frame_state_from_bus
from preframr_tokens.bacc.generic.bustrace import BUS_DT, load_bus
from preframr_tokens.bacc.primitive import BaccProgram

_CPF = 19656  # PAL raster frame; the synthetic trace uses a steady cadence
_GEN_REGS = {0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17}  # freq + pw lanes


def _synth_bus(nframes=160):
    """Build a small native ``BUS_DT`` bus trace that re-blits all 25 registers
    every play-call at a steady cadence, exercising several BACC archetypes:

      - voice 0 freq: a 4-step octave/multi arp (table-walk)
      - voice 0 pw:   a dwelled accumulator sweep
      - voice 1 freq: a linear portamento accumulator
      - voice 2 freq: a constant hold
      - volume:       hold; gates rise after the boot prolog

    Returns the ``BUS_DT`` record array (absolute cycles)."""
    arp = [0x0800, 0x0900, 0x0A00, 0x0900]
    recs = []
    cyc = 1000
    reg = [0] * 25
    for frame in range(nframes):
        v0f = arp[frame % 4]
        reg[0], reg[1] = v0f & 0xFF, (v0f >> 8) & 0xFF
        pw0 = 0x0400 + 4 * (frame // 2)  # dwelled accumulator (step every 2 frames)
        reg[2], reg[3] = pw0 & 0xFF, (pw0 >> 8) & 0x0F
        reg[4] = 0x41 if frame >= 2 else 0x40  # voice-0 gate rises at frame 2
        v1f = 0x2000 + 7 * frame  # voice-1 linear portamento
        reg[7], reg[8] = v1f & 0xFF, (v1f >> 8) & 0xFF
        reg[11] = 0x21 if frame >= 2 else 0x20
        reg[14], reg[15] = 0x00, 0x18  # voice-2 constant hold
        reg[18] = 0x10
        reg[24] = 0x0F  # master volume
        for index in range(25):
            recs.append((cyc, 0xD400 + index, reg[index], 1))
            cyc += 2
        cyc += _CPF - 2 * 25  # inter-play-call gap (a blit-group boundary)
    return np.array(recs, dtype=BUS_DT)


@pytest.fixture(name="synth_bus")
def _synth_bus_fixture():
    return _synth_bus()


def test_recover_render_residual_zero(synth_bus):
    program = recover_generic("synth.sid", None, synth_bus)
    assert isinstance(program, BaccProgram)
    assert program.driver == "generic"
    assert program.nframes > 100
    resid, rendered, state = residual(program, synth_bus)
    assert sum(resid.values()) == 0  # whole-tune, all 25 registers byte-exact
    assert np.array_equal(rendered[: len(state)], state[: len(rendered)])


def test_generator_lanes_residual_zero(synth_bus):
    program = recover_generic("synth.sid", None, synth_bus)
    resid, _, _ = residual(program, synth_bus)
    assert sum(resid[reg] for reg in _GEN_REGS) == 0


def test_archetypes_recovered_not_raw(synth_bus):
    program = recover_generic("synth.sid", None, synth_bus)
    tally = program.tables["archetypes"]
    # the freq/pw lanes are described by GENERATOR programs, never per-frame data:
    # the synthetic tune must recover the arp + an accumulator (not all holds).
    assert tally.get("arp", 0) >= 1
    assert tally.get("accum", 0) + tally.get("dwellaccum", 0) >= 1


def test_render_is_self_contained(synth_bus):
    # render_generic consumes only the BaccProgram (no .sid, no bus, no backend).
    program = recover_generic("synth.sid", None, synth_bus)
    rendered = render_generic(program)
    assert rendered.shape == (program.nframes, 25)
    # a second render is deterministic / idempotent.
    assert np.array_equal(rendered, render_generic(program))


def test_program_round_trips_through_serialised_tables(synth_bus):
    program = recover_generic("synth.sid", None, synth_bus)
    # the recovered program's tables are plain-Python (JSON-clean); rebuild a
    # fresh BaccProgram from them and confirm it renders identically.
    clone = BaccProgram(
        driver=program.driver,
        nframes=program.nframes,
        boot=list(program.boot),
        instruments=[],
        score=[],
        seed=dict(program.seed),
        tables=dict(program.tables),
    )
    assert np.array_equal(render_generic(program), render_generic(clone))


def test_archetype_renderers_byte_exact():
    # the BACC library renders its own parameterised generators byte-exact.
    arp = A.render_arp(12, [10, 20, 30], 3, 0, 1)
    assert arp.tolist() == [10, 20, 30] * 4
    accum = A.render_accum(5, 100, 7, 0xFFFF)
    assert accum.tolist() == [100, 107, 114, 121, 128]
    walk = A.render_tablewalk(6, [1, 2, 5], 0)
    assert walk.tolist() == [1, 2, 5, 1, 2, 5]


def _fit_round_trips(lane, note_table=None, carry=None):
    """The fitter recovers ``lane`` as a single-segment cover that renders back
    byte-exact (residual-zero)."""
    lane = np.asarray(lane, dtype=np.int64)
    fit = A.fit_segment(lane, 0, note_table, carry)
    assert fit is not None
    rendered = A.render_fit(fit, len(lane), note_table, carry)
    assert np.array_equal(rendered, lane)
    return fit


def test_fit_vibrato_round_trip():
    lane, _ = A.render_vibrato_exact(64, 0x1000, 0x40, 0)
    fit = _fit_round_trips(lane)
    assert fit[0] in ("vibrato", "vibrato_exact")


def test_fit_glide_round_trip():
    note_table = np.array([0x1000 + 0x80 * i for i in range(64)], dtype=np.int64)
    lane = A.render_glide(48, 10, 1, 3, 4, note_table)
    fit = _fit_round_trips(lane, note_table=note_table)
    assert fit[0] in ("glide", "accum", "dwellaccum", "piecewise")


def test_fit_decay_round_trip():
    body = A.render_decay(40, 0x4000, 0x0100, 2, 0)
    fit = _fit_round_trips(body)
    assert fit[0] in ("decay", "accum", "dwellaccum", "piecewise")


def test_fit_pingpong_round_trip():
    lane = A.render_pingpong(40, 0x0A, 1, 0x07, 0x0F, 0, 0, 0)
    _fit_round_trips(lane)


def test_fit_wrapaccum_round_trip():
    lane = A.render_wrapaccum(48, 0x100, 0x40, 0x100, 0x800)
    _fit_round_trips(lane)


def test_fit_maskaccum_round_trip():
    lane = A.render_maskaccum(48, 0x200, 0x20, [1, 0, 1, 0])
    _fit_round_trips(lane)


def test_fit_tablewalk_round_trip():
    # a period-9 modulation table (beyond the arp P<=6 cap) with no constant rate
    # and no clean wrap -- recovered byte-exact as a tablewalk last-resort.
    table = [0x40, 0x55, 0x42, 0x6E, 0x41, 0x57, 0x40, 0x52, 0x49]
    lane = A.render_tablewalk(54, table, 0)
    _fit_round_trips(lane)


def test_prefix_tablewalk_recovers_period():
    table = [0x40, 0x55, 0x42, 0x6E, 0x41, 0x57, 0x40, 0x52, 0x49]
    lane = A.render_tablewalk(54, table, 0)
    match = A._prefix_tablewalk(lane)
    assert match is not None
    assert match[1] == "tablewalk"
    assert match[2]["table"][:9] == table


def test_fit_ratewalk_round_trip():
    # a wavetable-rate accumulator (the FamiCommodore-style sub-resolution sweep):
    # a period-5 signed-rate table drives a single wider-internal-width accumulator
    # whose value climbs without a short value period -- recovered as one ratewalk.
    lane = A.render_ratewalk(60, 0x0200, [16, 16, 32, 0, 16])
    fit = _fit_round_trips(lane)
    assert fit[0] in ("ratewalk", "maskaccum", "piecewise")


def test_prefix_ratewalk_recovers_rate_table():
    table = [16, 16, 32, 0, 16]
    lane = A.render_ratewalk(60, 0x0200, table)
    match = A._prefix_ratewalk(lane)
    assert match is not None
    assert match[1] == "ratewalk"
    assert match[2]["rate_table"][:5] == table


def test_fit_tablewalk_lead_round_trip():
    # a delayed period-6 modulation (the Hammurabi-style long hold then LFO table):
    # 20 constant frames, then a period-6 offset table -- one rule, not two pieces.
    table = [0x40, 0x60, 0x80, 0x60, 0x40, 0x20]
    lane = A.render_tablewalk_lead(60, 20, 0x40, table)
    fit = _fit_round_trips(lane)
    assert fit[0] in ("tablewalk_lead", "tablewalk", "piecewise")


def test_prefix_tablewalk_lead_recovers_lead_and_table():
    table = [0x40, 0x60, 0x80, 0x60, 0x40, 0x20]
    lane = A.render_tablewalk_lead(60, 20, 0x40, table)
    match = A._prefix_tablewalk_lead(lane)
    assert match is not None
    assert match[1] == "tablewalk_lead"
    assert match[2]["lead"] == 20
    assert match[2]["table"][:6] == table


# A reflecting-triangle PW table over 12 levels (the FamiCommodore-style voice-2
# wavetable) and a non-uniform groove advance clock (some frames step, some hold).
_WPTR_TABLE = [
    0x080,
    0x088,
    0x090,
    0x0A0,
    0x0A8,
    0x0B0,
    0x0C0,
    0x0C8,
    0x0D0,
    0x0E0,
    0x0E8,
    0x0F0,
    0x0E8,
    0x0E0,
    0x0D0,
    0x0C8,
    0x0C0,
    0x0B0,
    0x0A8,
    0x0A0,
    0x090,
    0x088,
]
# a drifting (non-periodic) groove: mostly step, with holds inserted aperiodically.
_WPTR_GROOVE = ([1, 1, 0] * 7 + [1, 1, 1, 0, 1, 0]) * 4


def test_wavetable_ptr_holds_when_advance_clear():
    # the pointer steps on a set advance bit and HOLDS the prior entry on a clear
    # bit -- the value content is the table, the pacing is the external clock.
    out = A.render_wavetable_ptr(6, [10, 20, 30, 40], 0, [1, 0, 1, 1, 0])
    assert out.tolist() == [10, 20, 20, 30, 40, 40]


def test_fit_wavetable_ptr_round_trip():
    # an advance-clocked reflecting-triangle wavetable with a drifting dwell --
    # exactly the wavetable-paced PW that defeats periodic tablewalk/ratewalk.
    advance = _WPTR_GROOVE
    lane = A.render_wavetable_ptr(len(advance) + 1, _WPTR_TABLE, 0, advance)
    fit = _fit_round_trips(lane)
    assert fit[0] in ("wavetable_ptr", "piecewise")


def test_prefix_wavetable_ptr_recovers_table_and_clock():
    advance = _WPTR_GROOVE
    lane = A.render_wavetable_ptr(len(advance) + 1, _WPTR_TABLE, 0, advance)
    match = A._prefix_wavetable_ptr(lane)
    assert match is not None
    assert match[1] == "wavetable_ptr"
    assert match[2]["table"] == _WPTR_TABLE  # the period-22 generator, not raw data
    assert match[2]["advance"] == advance  # the recovered shared groove clock


def test_prefix_wavetable_ptr_rejects_step_every_frame():
    # a plain period-P value table that steps every frame is NOT a groove-paced
    # pointer walk; wavetable_ptr declines it so the cheaper tablewalk handles it.
    lane = A.render_tablewalk(48, _WPTR_TABLE, 0)
    assert A._prefix_wavetable_ptr(lane) is None


def test_prefix_wavetable_ptr_rejects_two_value_pingpong():
    # a 2-value dwell is a pingpong/hold, not a genuine reused value table; the
    # >=3-distinct-values guard keeps wavetable_ptr from absorbing it as raw data.
    lane = A.render_wavetable_ptr(40, [0x10, 0x20], 0, [1, 0, 1, 0] * 9 + [1, 1, 1])
    assert A._prefix_wavetable_ptr(lane) is None


def test_fit_additive_pw_round_trip():
    carry = np.array([1, 0, 0, 0] * 16, dtype=np.int64)
    lane = A.render_additive_pw(48, 0x0A00, 3, carry)
    _fit_round_trips(lane, carry=carry)


def test_fit_vibskydive_round_trip():
    lane = A.render_vibskydive(48, 0x1000, 0x40, 0, 0x30, 1)
    fit = _fit_round_trips(lane)
    assert fit[0] in ("vibskydive", "piecewise", "vibrato_exact")


def test_fit_arp_decay_round_trip():
    lane = A.render_arp_decay(48, [0x0800, 0x0900, 0x0A00, 0x0900], 4, 1, 0x30, 1)
    _fit_round_trips(lane)


def test_event_lane_fit_round_trip():
    col = np.array([0x10] * 8 + [0x12] * 8 + list(range(0x20, 0x2C)), dtype=np.int64)
    segs = A.fit_event_lane(col)
    rendered = A.render_event_lane(segs, len(col))
    assert np.array_equal(rendered, col)


def test_note_table_recovered_from_bus_provenance():
    # build a trace where each freq-lo SID write is preceded by a RAM read of the
    # same value from a contiguous stride-2 region -- the note-table provenance.
    base = 0x1200
    note_freqs = [0x0112 + 0x40 * n for n in range(16)]
    recs = []
    cyc = 1000
    for _ in range(12):  # >= min_hits writes per note-table entry
        for n, freq in enumerate(note_freqs):
            lo, hi = freq & 0xFF, (freq >> 8) & 0xFF
            recs.append((cyc, base + 2 * n, lo, 0))  # RAM read of lo
            cyc += 2
            recs.append((cyc, base + 2 * n + 1, hi, 0))  # RAM read of hi
            cyc += 2
            recs.append((cyc, 0xD400, lo, 1))  # SID freq-lo write
            cyc += 2
            recs.append((cyc, 0xD401, hi, 1))  # SID freq-hi write
            cyc += 2
    records = np.array(recs, dtype=BUS_DT)
    table = F.discover_note_table_from_bus(records)
    assert table is not None
    assert table[:16] == note_freqs


def test_note_table_none_without_provenance():
    records = np.array([(1000, 0xD400, 0x10, 1), (1002, 0xD401, 0x08, 1)], dtype=BUS_DT)
    assert F.discover_note_table_from_bus(records) is None


def test_per_frame_state_empty_trace():
    records = np.array([(1000, 0x0001, 0x00, 0)], dtype=BUS_DT)  # no SID writes
    state, t0, cpf = per_frame_state_from_bus(records)
    assert state is None and t0 is None and cpf is None


def test_recover_rejects_too_short_trace():
    records = np.array([(1000, 0xD400, 0x10, 1)], dtype=BUS_DT)
    with pytest.raises(ValueError, match="did not parse to frames"):
        recover_generic("x.sid", None, records)


def test_gate_noteons_detects_rises():
    state = np.zeros((6, 25), dtype=np.int64)
    state[2:, 4] = 1  # voice-0 gate rises at frame 2
    state[4:, 11] = 1  # voice-1 gate rises at frame 4
    noteons = A.gate_noteons(state)
    assert noteons[0] == [2]
    assert noteons[1] == [4]
    assert noteons[2] == []


def test_load_bus_rejects_rbt1(tmp_path):
    path = tmp_path / "wrong.bus.bin"
    path.write_bytes(b"RBT1" + b"\x00" * 32)
    with pytest.raises(ValueError, match="RBT1"):
        load_bus(str(path))


def test_load_bus_native_roundtrip(tmp_path, synth_bus):
    path = tmp_path / "trace.bus.bin"
    synth_bus.tofile(str(path))
    loaded = load_bus(str(path))
    assert np.array_equal(loaded, synth_bus)


_BUSTRACE = os.environ.get("GENERIC_BUSTRACE")


@pytest.mark.skipif(
    not _BUSTRACE or not os.path.exists(_BUSTRACE),
    reason="set GENERIC_BUSTRACE to a native preframr-sidtrace .bus.bin",
)
def test_whole_tune_residual_zero_real_trace():
    """Whole-tune residual-zero on a real native trace (opt-in).

    The bus-state must be reproduced byte-exact across ALL 25 registers by the
    self-contained render -- the proven 8/8-corpus whole-tune result (up from 5/8).
    Hammurabi is residual-zero via the generic tablewalk_lead archetype (a delayed
    long-period modulation); FamiCommodore is residual-zero via the advance-clocked
    wavetable_ptr archetype (a voice-2 PW groove-paced reflecting triangle over a
    period-22 value table); Not_Even_Human renders byte-exact (its only diff is a
    bus-vs-dump song-end tail, not compared here).  Any genuinely irreducible lane
    would surface as residual rather than be faked with a per-step-storage cover --
    so a non-zero residual here is always a hard failure, never xfail'd."""
    records = load_bus(_BUSTRACE)
    program = recover_generic(_BUSTRACE, None, records)
    resid, _, state = residual(program, records)
    total = sum(resid.values())
    assert total == 0, f"residual on {state.shape}: {resid}"
