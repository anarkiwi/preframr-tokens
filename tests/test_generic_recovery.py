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


def test_prefix_wrapaccum_large_rate_is_fast_and_exact():
    # A WIDE free-running sweep (rate in the thousands, like a multispeed freq lane)
    # must recover byte-exact WITHOUT scanning O(|rate|) wrap boundaries -- the
    # boundary is data-determined, so the matcher derives the few candidates from the
    # wrap points and stays fast.  This is the dense-multispeed perf cliff that made
    # whole tunes exceed the harness budget despite recovering residual-zero.
    import time as _time

    lane = A.render_wrapaccum(60, 0x4000, 0x0C00, 0x0400, 0xF000)
    start = _time.perf_counter()
    match = A._prefix_wrapaccum(lane)
    assert _time.perf_counter() - start < 1.0  # not O(|rate|) ~ thousands of renders
    assert match is not None and match[1] == "wrapaccum"
    rend = A.render_wrapaccum(
        60, match[2]["v0"], match[2]["rate"], match[2]["lo"], match[2]["hi"]
    )
    assert np.array_equal(rend, lane)  # byte-exact


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


def test_prefix_maskaccum_stall_recovers_period_and_rate():
    # a single-rate accumulator that HOLDS on a periodic tick-0 stall mask (the
    # GoatTracker tempo-paced continuous-effect skip): step +0x10 on the period-6
    # mask [1,1,1,1,1,0], over many cycles.  The longest-prefix stall matcher must
    # recover the rate and the period-6 advance mask, not a coincidental short fit.
    mask = [1, 1, 1, 1, 1, 0]
    lane = A.render_maskaccum(120, 0x0200, 0x10, mask)
    match = A._prefix_maskaccum_stall(lane)
    assert match is not None
    assert match[1] == "maskaccum"
    assert match[2]["rate"] == 0x10
    assert match[2]["mask"][:6] == mask
    rendered = A.render_fit((match[1], match[2]), len(lane))
    assert np.array_equal(rendered, lane)


def test_maskaccum_stall_not_promoted_over_short_coincidental_arp():
    # a clean period-4 arp must NOT be re-described as a stall accumulator: the
    # stall matcher only wins when it covers SUBSTANTIALLY more than the proven
    # library's run, so a genuine short generator is never shadowed.
    lane = A.render_arp(48, [0x0800, 0x0900, 0x0A00, 0x0900], 4, 0, 1)
    name, _, _ = A._longest_archetype_aug(lane, 0, None, None, 0xFFFF)
    assert name == "arp"


def test_maskaccum_stall_requires_multiple_cycles():
    # a one-shot dwell (a single stall in an otherwise linear ramp) is NOT a
    # periodic tick-0 stall: the stall matcher requires >= mincycles full mask
    # cycles, so a coincidental lone hold is not promoted to a periodic generator.
    lane = np.array(
        [0x200 + 0x10 * i for i in range(5)]
        + [0x200 + 0x10 * 4] * 1
        + [0x200 + 0x10 * (i + 4) for i in range(1, 6)],
        dtype=np.int64,
    )
    assert A._prefix_maskaccum_stall(lane, 0xFFFF, mincycles=3) is None


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


def test_prefix_wavetable_ptr_folds_longest_prefix_of_multi_section_melody():
    # A FutureComposer voice running one looping note/arp wavetable for several
    # pattern rows and then SWITCHING to a different one is a single un-retriggered
    # note-on segment whose walk does NOT fold onto one loop -- but its FIRST
    # section does.  The prefix fold must recover that section as one wavetable_ptr
    # piece (the period-P table + its advance clock), not decline the whole lane.
    table_a = [0x0800, 0x0900, 0x0A00, 0x0B00, 0x0900]  # section-A note wavetable
    table_b = [0x1000, 0x1100, 0x1200, 0x0F00]  # a DIFFERENT section-B wavetable
    groove = [1, 1, 0] * 12  # dwelled advance clock (some frames hold)
    sec_a = A.render_wavetable_ptr(len(groove) + 1, table_a, 0, groove)
    sec_b = A.render_wavetable_ptr(len(groove) + 1, table_b, 0, groove)
    lane = np.concatenate([sec_a, sec_b])
    match = A._prefix_wavetable_ptr(lane)
    assert match is not None
    assert match[1] == "wavetable_ptr"
    assert match[2]["table"] == table_a  # section A's generator, not the whole melody
    assert match[0] == len(sec_a)  # the prefix stops exactly where section B begins
    assert len(match[2]["advance"]) == match[0]  # clock trimmed to the matched run
    # and the greedy whole-segment cover lays the chain down piece by piece, exact.
    fit = A.fit_segment(lane, 0)
    rendered = A.render_fit(fit, len(lane))
    assert np.array_equal(rendered, lane)


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


def test_per_frame_state_forward_fills_sparse_writes():
    """The (vectorised) per-frame reconstruction must hold a register's value
    across play-calls that do not re-write it -- the running-register-file
    semantics -- and take the LAST write within a play-call.  This is the
    invariant a multispeed/coalesced trace (millions of writes, registers
    written sparsely) relies on; assert it on a tiny hand-built trace."""
    cpf = 19656
    recs = []
    cyc = 1000
    # frame 0: write reg0=0x11, reg24=0x0F  (reg1 never written -> stays 0)
    # frame 1: write reg0 twice (0x20 then 0x22 -> last wins); reg24 NOT rewritten
    # frame 2: write nothing to reg0 (holds 0x22); bump reg24=0x07
    plan = [
        [(0, 0x11), (24, 0x0F)],
        [(0, 0x20), (0, 0x22)],
        [(24, 0x07)],
    ]
    for writes in plan:
        for reg, val in writes:
            recs.append((cyc, 0xD400 + reg, val, 1))
            cyc += 2
        cyc += cpf  # play-call (blit-group) boundary
    records = np.array(recs, dtype=BUS_DT)
    state, _, _ = per_frame_state_from_bus(records, t0=1000)
    assert state.shape == (3, 25)
    assert list(state[:, 0]) == [0x11, 0x22, 0x22]  # last-write + forward-fill
    assert list(state[:, 1]) == [0, 0, 0]  # never written -> zero
    assert list(state[:, 24]) == [0x0F, 0x0F, 0x07]  # held then changed


def _continuous_write_bus(nframes=120, cpf=19656):
    """A trace whose play-call writes are spread EVENLY across the whole frame so
    there is no quiet inter-play gap -- the gap-based blit grouping collapses every
    play-call into one group.  Voice-0 freq-hi ramps one step per frame so a
    correctly-framed reconstruction is unambiguous."""
    recs = []
    t0 = 1000
    # 50 writes per frame, spread across the frame's first half -> the largest
    # inter-write gap (~393 cycles) is well under the 2000-cycle blit-gap, so NO
    # group boundary ever forms and the gap framing degenerates to a single group;
    # keeping them in the first half means every write rounds to its own frame.
    per_frame = 50
    step = (cpf // 2) // per_frame
    for frame in range(nframes):
        base = t0 + frame * cpf
        fhi = (0x10 + frame) & 0xFF  # voice-0 freq-hi ramps one step / frame
        for k in range(per_frame):
            reg = 1 if k == 0 else 24  # reg1 = freq-hi (the ramp); reg24 padding
            val = fhi if k == 0 else 0x0F
            recs.append((base + k * step, 0xD400 + reg, val, 1))
    return np.array(recs, dtype=BUS_DT), t0, cpf, nframes


def test_per_frame_state_cadence_fallback_on_continuous_writes():
    """A player that writes the SID continuously (no per-play gap) collapses the
    gap-based grouping to a single frame; the cadence-binning fallback must instead
    recover one frame per play-call with byte-exact running state."""
    records, t0, _cpf, nframes = _continuous_write_bus()
    state, got_t0, _ = per_frame_state_from_bus(records, t0=t0)
    assert state.shape == (nframes, 25)
    assert got_t0 == t0
    # voice-0 freq-hi (reg 1) is the per-frame ramp 0x10, 0x11, ... one per frame.
    assert list(state[:, 1]) == [(0x10 + f) & 0xFF for f in range(nframes)]


def test_continuous_write_tune_recovers_residual_zero():
    """End to end: the continuous-write tune (which the gap framing could not
    parse) recovers a residual-zero generic program via the cadence fallback."""
    records, _t0, _cpf, _ = _continuous_write_bus()
    program = recover_generic("continuous.sid", None, records)
    assert program.nframes > 100
    resid, _, _ = residual(program, records)
    assert sum(resid.values()) == 0


def _static_hold_bus(nframes=200, cpf=19656):
    """A trace that pokes the SID ONCE at boot and then never writes it again --
    the register file rings on, unchanged, for the rest of the run.

    The boot burst sets a static chord (a value per register); after it the chip is
    quiet but the CPU keeps running (modelled as non-SID RAM reads), so the OVERALL
    bus trace spans ``nframes`` cadence frames even though there is exactly one
    SID-write group.  This is the BASIC-stub / init-only / single-chord-intro
    signature the gap AND cadence framings both collapse to a single frame."""
    t0 = 1000
    boot = {0: 0x49, 1: 0x1C, 4: 0x21, 5: 0x09, 24: 0x0F}  # one static chord
    recs = [
        (t0 + i * 2, 0xD400 + reg, val, 1) for i, (reg, val) in enumerate(boot.items())
    ]
    # the CPU runs on (non-SID reads) to the end of the requested duration so the
    # trace extent spans many cadence frames with no further SID write.
    for frame in range(1, nframes):
        recs.append((t0 + frame * cpf, 0x0314, 0x00, 0))  # a quiet RAM read
    return np.array(recs, dtype=BUS_DT), t0, cpf, nframes, boot


def test_per_frame_state_static_hold_on_single_boot_burst():
    """A single boot burst of SID writes followed by a quiet, ringing chip recovers
    a whole-tune CONSTANT-HOLD state spanning the trace's full cadence extent --
    not a rejected single frame."""
    records, t0, _cpf, nframes, boot = _static_hold_bus()
    state, got_t0, _ = per_frame_state_from_bus(records, t0=t0)
    assert state.shape == (nframes, 25)
    assert got_t0 == t0
    for reg, val in boot.items():
        assert list(state[:, reg]) == [val] * nframes  # held for the whole tune


def test_static_hold_tune_recovers_residual_zero_as_holds():
    """End to end: the static-hold tune (which the framing could not parse before)
    recovers a residual-zero generic program whose every lane is a constant ``hold``
    generator -- a closed-form program, never raw per-frame data."""
    records, _t0, _cpf, _nf, _boot = _static_hold_bus()
    program = recover_generic("static.sid", None, records)
    assert program.nframes > 100
    resid, _, _ = residual(program, records)
    assert sum(resid.values()) == 0
    # the cover is constant holds only -- no per-frame structure was invented.
    assert set(program.tables["archetypes"]) <= {"hold"}


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


def test_note_boundaries_detects_retriggers_without_gate_rise():
    # A legato / hard-restart phrase: the gate bit stays HIGH after frame 2, so
    # gate_noteons sees a single note-on, but the player re-triggers each new note
    # by rewriting the control byte (a waveform change / 1-frame hard restart) or
    # the ADSR registers.  note_boundaries must surface every retrigger.
    state = np.zeros((12, 25), dtype=np.int64)
    state[2:, 4] = 0x41  # voice-0 gate on (bit0) from frame 2, stays high
    state[4:, 4] = 0x21  # frame 4: waveform-bit change, gate bit0 STILL set
    state[8:, 5] = 0x0A  # voice-0 AD rewrite at frame 8 (new note attack/decay)
    assert A.gate_noteons(state)[0] == [2]  # gate-only misses the retriggers
    bounds = A.note_boundaries(state)[0]
    assert 2 in bounds and 4 in bounds and 8 in bounds
    # voices 1 and 2 never sound -> no boundaries.
    assert A.note_boundaries(state)[1] == []
    assert A.note_boundaries(state)[2] == []


def test_pw_sweep_resets_detect_note_reseeds_not_smooth_sweep():
    # A smooth ramping PW sweep (no reset) trips nothing; a sweep that re-seeds
    # (drops sharply back to its start each note) surfaces the drop frames.
    smooth = np.zeros((40, 25), dtype=np.int64)
    for i in range(40):
        smooth[i, 2] = (0x40 + 4 * i) & 0xFF
        smooth[i, 3] = ((0x40 + 4 * i) >> 8) & 0x0F
    assert A.pw_sweep_resets(smooth, 0) == []
    reseed = np.zeros((40, 25), dtype=np.int64)
    for i in range(40):
        val = 0x100 + 0x40 * (i % 8)  # ramps then drops back every 8 frames
        reseed[i, 2] = val & 0xFF
        reseed[i, 3] = (val >> 8) & 0x0F
    resets = A.pw_sweep_resets(reseed, 0)
    assert resets == [8, 16, 24, 32]


def test_legato_vibrato_freq_recovered_via_pw_reset_boundary():
    # A pure-legato phrase: gate stays high, control/ADSR never change, so
    # note_boundaries finds nothing -- but each new note re-seeds the PW sweep.
    # The freq lane (a per-note hold that jumps between notes) is recovered only
    # because the freq lane is also sliced at the PW-sweep resets.
    note_bases = [0x0900, 0x0C00, 0x0700, 0x0A00]
    recs = []
    cyc = 1000
    reg = [0] * 25
    nframes = 64
    for frame in range(nframes):
        note = note_bases[(frame // 8) % len(note_bases)] if frame >= 2 else 0
        reg[0], reg[1] = note & 0xFF, (note >> 8) & 0xFF
        reg[4] = 0x41 if frame >= 2 else 0x40  # gate rises once at frame 2, then HELD
        pw = 0x100 + 0x40 * (frame % 8)  # PW sweep re-seeds every 8 frames (note len)
        reg[2], reg[3] = pw & 0xFF, (pw >> 8) & 0x0F
        reg[24] = 0x0F
        for index in range(25):
            recs.append((cyc, 0xD400 + index, reg[index], 1))
            cyc += 2
        cyc += _CPF - 2 * 25
    bus = np.array(recs, dtype=BUS_DT)
    # control byte never changes after frame 2, so the retrigger-only boundary
    # detector sees a single note-on -- the freq lane would otherwise be unfit.
    state, _, _ = per_frame_state_from_bus(bus)
    assert A.note_boundaries(state)[0] == [2]
    program = recover_generic("legato_vib.sid", None, bus)
    resid, _, _ = residual(program, bus)
    assert resid[0] == 0 and resid[1] == 0  # voice-0 freq byte-exact
    assert sum(resid.values()) == 0


def test_note_boundaries_legato_phrase_renders_residual_zero():
    # Build a synthetic tune whose voice-0 freq is a multi-note phrase with the
    # gate held HIGH throughout: each note holds a distinct freq and re-triggers
    # via a control-byte change.  Gate-only slicing collapses the whole phrase into
    # one over-long segment (unfit, all zeros); note_boundaries slices it per note
    # so it renders byte-exact.
    notes = [0x0800, 0x0A00, 0x0C00, 0x0900]
    recs = []
    cyc = 1000
    reg = [0] * 25
    nframes = 80
    for frame in range(nframes):
        note = notes[(frame // 8) % len(notes)] if frame >= 2 else 0
        reg[0], reg[1] = note & 0xFF, (note >> 8) & 0xFF
        # gate stays high from frame 2; a fresh note every 8 frames re-triggers the
        # control byte (waveform bit toggles) without dropping the gate bit.
        if frame >= 2:
            reg[4] = 0x41 if (frame // 8) % 2 == 0 else 0x21  # gate bit0 stays set
        reg[24] = 0x0F
        for index in range(25):
            recs.append((cyc, 0xD400 + index, reg[index], 1))
            cyc += 2
        cyc += _CPF - 2 * 25
    bus = np.array(recs, dtype=BUS_DT)
    program = recover_generic("legato.sid", None, bus)
    resid, _, _ = residual(program, bus)
    assert resid[0] == 0 and resid[1] == 0  # voice-0 freq lane byte-exact
    assert sum(resid.values()) == 0


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
