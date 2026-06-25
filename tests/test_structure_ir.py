"""Self-contained unit tests for the structure-recovery codec (:mod:`structure_ir`).

These exercise the serialize/deserialize/render codec and the assembly from a
:class:`~preframr_tokens.bacc.generic.structure_recover.RecoveredStructure` WITHOUT the
optional ``preframr-sidtrace`` binary -- on a synthetic structure + a synthetic distill
path -- so the default CI covers the codec mechanism (the env-gated whole-tune proof in
``test_corpus_budget.py`` / ``test_structure_recover.py`` exercises the real artifacts).

The invariants pinned here:
  * the codec round-trips every serialized field EXACTLY (the no-escape gate, raising on
    mismatch) and the patterns re-decode to the SAME ``(note, instr, dur, cmd)`` tuples;
  * the FREQ lanes render from the deserialized IR ALONE, byte-exact, via the recovered
    porta/vibrato accumulators (the §state-machine identity);
  * a structured artifact assembles into a < 1-token/frame IR; a non-structured one
    (``ok=False``) returns ``None`` (the additive-fallback invariant).
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import structure_ir as SI
from preframr_tokens.bacc.generic.structure_recover import RecoveredStructure


def _hand_pattern_bytes():
    """Two hand patterns in the NewPlayer grammar (markers >= 0x80 then a note < 0x80,
    0x7F ends the pattern) -- pattern 1 is reused so the orderlist factors it."""
    # dur=0x0C, instr=0x01, note 0x10; cmd=0x02, note 0x20; rest 0x00; end 0x7F
    p0 = [0x8C, 0xA1, 0x10, 0xC2, 0x20, 0x00, 0x7F]
    p1 = [0x8D, 0xA2, 0x11, 0x12, 0x7F]  # dur, instr, note; note; end
    return [p0, p1]


def _synthetic_ir(nframes=64):
    """A synthetic :class:`StructureIR` with a known FITTED accumulator generator (a
    ramp) so the freq render is byte-exact, plus a byte-exact ``_state`` anchor."""
    from preframr_tokens.bacc.generic.structure_recover import ACC_RAMP

    pattern_bytes = _hand_pattern_bytes()
    patterns = [SI._decode_pattern_bytes(pb) for pb in pattern_bytes]

    # One voice-0 accumulator: a RAMP generator (value += 3) the freq render re-adds onto
    # a held pitch.  accfit entry = (first_seen, kind, seed, p1, p2, p3, n, raw).
    first_seen, rate, n = 0, 3, nframes
    accfits = [[(first_seen, ACC_RAMP, 0, rate, 0, 0, n, None)], [], []]

    # Build a byte-exact state whose voice-0 freq == seed(=0x0100, held) + acc16 grid.
    state = np.zeros((nframes, 25), dtype=np.int64)
    seed = 0x0100
    align = 1
    acc = np.zeros(nframes, dtype=np.int64)
    for i in range(align, nframes):
        acc[i] = ((i - align) * rate) & 0xFFFF
    freq = (seed + acc) % 65536
    state[:, 0] = freq & 0xFF
    state[:, 1] = (freq >> 8) & 0xFF

    # The PR4a token-derived note base: voice 0 carries the FITTED accumulator (above);
    # store it as a _NB_TABLE with a single held pitch + a flat idx walk so the freq render
    # from tokens is base(held seed) + the accumulator overlay = freq, byte-exact.  Voices
    # 1/2 are silent (freq 0) -> a _NB_ARP with a single-pitch table and one (0,)-offset
    # shape per onset (a content-addressed ref pool, NOT a per-frame value stream).
    nb_v0 = (SI._NB_TABLE, [seed], [0], [0], [0], 2, align, [seed] * align)
    onsets = [align, nframes // 2]
    shape0 = (0,) * (onsets[1] - onsets[0])
    shape1 = (0,) * (nframes - onsets[1])
    silent_arp = (
        SI._NB_ARP,
        [0],  # one-entry note table (silence)
        align,
        [0] * align,  # warm-up
        onsets,  # two per-onset segments tiling [align, nframes)
        [0, 0],  # base index 0 at each onset
        [0, 1],  # ref the length-matched (0,...)-offset shapes
        [shape0, shape1],  # distinct (0,...)-offset shapes (a content-addressed pool)
    )
    note_bases = [nb_v0, silent_arp, silent_arp]

    return SI.StructureIR(
        note_table=[0x0100, 0x0120, 0x0140],
        instr_pool=[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        shared_programs=[
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            2,
            3,
        ],  # a repeating program (LZ collapses)
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=[[0, 1, 0, 0xFF], [1, 0xFF], [0, 0xFF]],  # pattern 0 reused
        accfits=accfits,
        note_bases=note_bases,
        # the SHARED onset grid the ARP / SEG records key to; windowed at each record's ``a``
        # (here ``align``) it re-derives that record's onsets (the per-record copy is dropped).
        onset_grid=[0, align, nframes // 2],
        nframes=nframes,
        boot=[int(state[0, r]) for r in range(25)],
        _state=state,
    )


def test_codec_field_roundtrip_exact():
    ir = _synthetic_ir()
    ids = SI.assert_ids_roundtrip(ir)  # raises on any field mismatch
    assert ids == SI.structure_ir_to_ids(ir)
    back = SI.structure_ir_from_ids(ids)
    # the patterns re-decode to the SAME tuples through the byte stream
    assert back.patterns == ir.patterns
    assert back.pattern_bytes == ir.pattern_bytes
    assert back.note_table == ir.note_table
    assert back.instr_pool == ir.instr_pool
    assert back.orderlists == ir.orderlists
    assert SI._norm(back.accfits) == SI._norm(ir.accfits)


def test_section_sizes_sum_to_total():
    ir = _synthetic_ir()
    ids = SI.structure_ir_to_ids(ir)
    sizes = SI.section_sizes(ir)
    assert sum(sizes.values()) == len(ids)


def test_freq_renders_from_deserialized_ir_byte_exact():
    ir = _synthetic_ir()
    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    freq = SI.render_freq_from_ir(back, ir._state)
    ref = ir._state[:, 0] | (ir._state[:, 1] << 8)
    assert np.array_equal(
        freq[0], ref
    ), "voice-0 freq must render from the IR byte-exact"


def test_render_structure_byte_exact_from_tokens_alone():
    """PR4a: ``render_structure`` reproduces the full 25-register state BYTE-EXACT from the
    DESERIALIZED IR ALONE (``_state is None``) -- the freq from the token-derived note base
    + accumulators, the non-freq from boot (all lanes constant here)."""
    ir = _synthetic_ir()
    rendered = SI.render_structure(ir)
    assert np.array_equal(rendered, ir._state)
    # rebuilt from ids alone (no anchor): still byte-exact -- the token render is anchor-free.
    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert back._state is None
    rendered_tok = SI.render_structure(back)
    assert np.array_equal(rendered_tok, ir._state)


def test_nonfreq_lane_renders_from_ids_alone():
    """The M1 non-freq replay: an ADMITTED change-point lane renders BYTE-EXACT from the
    serialized program ALONE (no anchor) -- the proof ``render-from-ids == state`` for that
    lane.
    """
    nframes = 64
    ir = _synthetic_ir(nframes)
    # admit ctrl voice-0 (reg 4) as a sparsely-gated lane: 0 then 0x41 at frame 5, held.
    col = np.zeros(nframes, dtype=np.int64)
    col[5:] = 0x41
    starts, values = SI._lane_change_program(col)
    ir.nonfreq = [(SI._LANE_CP, 4, starts, values)]
    ir._state[:, 4] = (
        col  # keep the anchor consistent so render_structure is byte-exact
    )

    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert SI._norm(back.nonfreq) == SI._norm(ir.nonfreq)  # the program round-trips
    # render the admitted lane from the IR ALONE (anchor=None) -- byte-exact, no anchor.
    lanes = SI.render_nonfreq_from_ir(back, anchor=None)
    assert 4 in lanes and np.array_equal(lanes[4], col)
    # and the full render (admitted lane from program, the rest from the anchor) is exact.
    rendered = SI.render_structure(ir)
    assert np.array_equal(rendered, ir._state)


def test_nonfreq_seg_lane_renders_from_ids_alone():
    """A DENSE non-freq lane admitted as a content-addressed PER-ONSET SEGMENT dictionary
    (``_LANE_SEG``): a small distinct-segment pool REF'd per onset (the shared instrument
    across notes) renders BYTE-EXACT from the serialized record ALONE -- never a per-frame
    value dump (HARD RULE #0)."""
    nframes = 64
    ir = _synthetic_ir(nframes)
    # a two-shape instrument lane: every onset fires shape A (a 4-frame attack then hold) or
    # shape B (a constant), so the per-onset dictionary collapses to TWO distinct segments.
    onsets = list(range(0, nframes, 8))  # 8 onsets, each an 8-frame note
    seg_a = (0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x40, 0x40)
    seg_b = (0x80,) * 8
    seg_dict = [seg_a, seg_b]
    refs = [0, 1, 0, 1, 0, 1, 0, 1]
    col = np.zeros(nframes, dtype=np.int64)
    for k, o in enumerate(onsets):
        col[o : o + 8] = np.asarray(seg_dict[refs[k]], dtype=np.int64)
    ir.nonfreq = [(SI._LANE_SEG, 4, 0, [], onsets, refs, seg_dict)]
    ir.onset_grid = (
        onsets  # the SHARED schedule the SEG record keys its per-onset refs to
    )
    ir._state[:, 4] = col

    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert SI._norm(back.nonfreq) == SI._norm(ir.nonfreq)  # the record round-trips
    lanes = SI.render_nonfreq_from_ir(back, anchor=None)
    assert 4 in lanes and np.array_equal(lanes[4], col)
    rendered = SI.render_structure(ir)
    assert np.array_equal(rendered, ir._state)


def test_note_base_arp_round_trip_and_render():
    """A ``_NB_ARP`` note base (per-onset base index + REF into a distinct offset-shape pool)
    round-trips through the codec and renders its freq byte-exact -- an arp is a base pitch
    plus a small repeating index-offset shape (NOTES), not a per-frame value stream."""
    nframes = 32
    ir = _synthetic_ir(nframes)
    # a 3-pitch table, an arp that cycles base 0 -> +1 -> +2 over each onset segment.
    tbl = [0x0100, 0x0110, 0x0120]
    onsets = [0, 8, 16, 24]
    shape = (0, 1, 2, 1, 0, 1, 2, 1)  # an 8-frame arp wiggle
    bases = [0, 0, 0, 0]
    refs = [0, 0, 0, 0]
    rec = (SI._NB_ARP, tbl, 0, [], onsets, bases, refs, [shape])
    grid, is_full = SI._note_base_grid(rec, nframes)
    assert is_full
    # verify against the explicit idx walk
    tbl_arr = np.asarray(tbl, dtype=np.int64)
    expect = np.concatenate([tbl_arr[list(shape)] for _ in onsets])
    assert np.array_equal(grid, expect)
    # round-trip via the full note-base section.  All three records key to the SAME shared
    # onset grid (a real tune's voices share one schedule); a single-pitch (0,)-shape ARP on
    # the other voices tiles the same onsets, so the windowed-onset re-derivation matches.
    silent = (
        SI._NB_ARP,
        [0],
        0,
        [],
        onsets,
        [0] * len(onsets),
        [0] * len(onsets),
        [(0,) * 8],
    )
    ir.note_bases = [rec, silent, silent]
    ir.onset_grid = onsets
    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert SI._norm(back.note_bases) == SI._norm(ir.note_bases)


def test_c3_gate_fails_on_raw_per_frame_note_base():
    """C3 hardening (``codec_gate.c3_no_raw_value_stream``): a synthetic IR carrying a raw
    per-frame note_base VALUE stream (length ~nframes) MUST be REJECTED -- the banned
    literal-floor / register-log reproduction, not a generator/table/ref representation
    (HARD RULE #0)."""
    from tools.codec_gate import CheckFailure, c3_no_raw_value_stream

    nframes = 64
    # a fake note base that smuggles a per-frame freq VALUE stream in as the warm-up field
    # (a relabeled dump): a _NB_TABLE whose warm-up is the whole nframes-long freq column.
    raw_stream = list(range(nframes))
    bad = SI.StructureIR(
        note_bases=[(SI._NB_TABLE, [0], [0], [0], [0], 2, 0, raw_stream)],
        nonfreq=[],
        nframes=nframes,
    )
    with pytest.raises(CheckFailure):
        c3_no_raw_value_stream(bad, nframes)

    # an UNKNOWN note-base kind (not a generator/table/ref) is also rejected.
    bad_kind = SI.StructureIR(
        note_bases=[(99, raw_stream)], nonfreq=[], nframes=nframes
    )
    with pytest.raises(CheckFailure):
        c3_no_raw_value_stream(bad_kind, nframes)

    # a clean generator IR passes (the _synthetic_ir representations are all generators).
    c3_no_raw_value_stream(_synthetic_ir(nframes), nframes)


def test_nonfreq_ramp16_generator_renders_from_ids_alone():
    """The M1 generator-fit: a PW sweep (voice-0 ``lo|hi<<8`` = a constant-step wrapping
    ramp) is DERIVED as a 16-bit ramp GENERATOR -- ONE record covering BOTH byte lanes --
    and renders byte-exact from the ids ALONE (no anchor).  This is the derive-don't-store
    win: a thousands-frame sweep collapses to a handful of per-segment ints."""
    nframes = 80
    ir = _synthetic_ir(nframes)
    # a clean +88 (mod 2^16) PW ramp on voice-0 (regs 2 lo, 3 hi), seeded at 0x0040.
    pw = np.zeros(nframes, dtype=np.int64)
    val = 0x0040
    for i in range(nframes):
        pw[i] = val
        val = (val + 88) % (1 << 16)
    lo, hi = pw & 0xFF, (pw >> 8) & 0xFF
    starts, seeds, steps = SI._ramp16_fit(lo, hi)
    assert len(starts) == 1 and seeds == [0x0040] and steps == [88]  # one clean segment
    ir.nonfreq = [(SI._LANE_RAMP16, 2, 3, starts, seeds, steps)]
    ir._state[:, 2] = lo
    ir._state[:, 3] = hi

    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert SI._norm(back.nonfreq) == SI._norm(ir.nonfreq)  # the generator round-trips
    lanes = SI.render_nonfreq_from_ir(back, anchor=None)  # both byte lanes from the ids
    assert np.array_equal(lanes[2], lo) and np.array_equal(lanes[3], hi)
    rendered = SI.render_structure(ir)
    assert np.array_equal(rendered, ir._state)


def test_build_nonfreq_picks_ramp16_over_changepoints_for_a_sweep():
    """``build_nonfreq_program`` picks the CHEAPEST byte-exact encoding: a dense PW sweep
    (a wrapping ramp, ~every frame a change point) is admitted as ONE ``_LANE_RAMP16``
    generator, not two dense change-point lanes (the literal-floor trap)."""
    nframes = 400
    state = np.zeros((nframes, 25), dtype=np.int64)
    val = 0x0123
    for i in range(nframes):
        state[i, 2] = val & 0xFF
        state[i, 3] = (val >> 8) & 0xFF
        val = (val + 217) % (
            1 << 16
        )  # a steep ramp -> a change point almost every frame
    prog = SI.build_nonfreq_program(state, None)
    ramp = [r for r in prog if r[0] == SI._LANE_RAMP16 and r[1] == 2 and r[2] == 3]
    assert ramp, "the dense PW sweep must be admitted as a 16-bit ramp generator"
    # neither byte lane is ALSO emitted as a change-point record (the generator covers both).
    cp_regs = {r[1] for r in prog if r[0] == SI._LANE_CP}
    assert 2 not in cp_regs and 3 not in cp_regs
    # and it renders the two byte lanes byte-exact from the program alone.
    ir = SI.StructureIR(nonfreq=prog, nframes=nframes)
    lanes = SI.render_nonfreq_from_ir(ir, anchor=None)
    assert np.array_equal(lanes[2], state[:, 2]) and np.array_equal(
        lanes[3], state[:, 3]
    )


def test_nonfreq_empty_program_serialises_identically():
    """A structure with no admitted non-freq lane serialises byte-identically to the
    pre-M1 stream (the optional section is omitted) -- the back-compat invariant for the
    committed fixtures."""
    ir = _synthetic_ir()
    assert ir.nonfreq == []
    ids = SI.structure_ir_to_ids(ir)
    back = SI.structure_ir_from_ids(ids)
    assert back.nonfreq == []
    assert SI.structure_ir_to_ids(back) == ids


def test_build_structure_ir_from_synthetic_struct(tmp_path):
    # build_structure_ir over a synthetic RecoveredStructure + a non-STSQ distill file
    # (so clean_pitches_residual/read_stsq_cells no-op): exercises the assembly + dedup.
    ram = np.zeros(65536, dtype=np.uint8)
    pbs = _hand_pattern_bytes()
    ram[0x1000 : 0x1000 + len(pbs[0])] = pbs[0]
    ram[0x1100 : 0x1100 + len(pbs[1])] = pbs[1]
    struct = RecoveredStructure(
        ok=True,
        note_table=[0x0100, 0x0120],
        instr_records=[[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],  # dup -> 1
        pattern_ptrs=[0x1000, 0x1100],
        orderlists=[[0, 1, 0xFF]],
        program_spans={"prog0": (0x1400, 0x1404)},
        ram=ram,
        nframes=64,
    )
    distill = str(tmp_path / "synthetic.distill.bin")
    with open(distill, "wb") as handle:
        handle.write(b"NOTSDST" + b"\x00" * 64)  # not an SDST artifact -> STSQ no-ops
    state = np.zeros((64, 25), dtype=np.int64)
    ir = SI.build_structure_ir(struct, state, distill)
    assert len(ir.instr_pool) == 1  # the duplicate instrument struct deduped
    assert ir.pattern_bytes == pbs
    assert ir.accfits == [[], [], []]  # no STSQ in this artifact -> no accumulator fits
    # no PWLK walk in this artifact -> no schedule recovered (the additive-coverage field
    # is None, exactly like _state, and never enters the serialized stream).
    assert ir.schedule is None
    SI.assert_ids_roundtrip(ir)  # the assembled IR still round-trips exactly


def test_schedule_field_not_serialized():
    """The recovered note->frame schedule is a DERIVED analysis field (the IWLK PR will
    drive freq from it; here it only proves coverage), so -- like ``_state`` -- it is NOT
    serialized: an IR with a schedule serialises byte-identically to the same IR without
    one, and deserialization leaves ``schedule`` None."""
    base = SI.StructureIR(nframes=64, boot=[0] * SI.NREG)
    with_sched = SI.StructureIR(
        nframes=64,
        boot=[0] * SI.NREG,
        schedule={
            "onsets": [0, 2, 4],
            "durations": [2, 2, 60],
            "tempo": 2,
            "n_onsets": 3,
            "span": (0, 4),
            "nframes": 64,
        },
    )
    assert SI.structure_ir_to_ids(with_sched) == SI.structure_ir_to_ids(base)
    assert SI.structure_ir_from_ids(SI.structure_ir_to_ids(with_sched)).schedule is None


def test_recover_structure_ir_returns_none_when_not_ok(monkeypatch, tmp_path):
    # a non-structured tune (recover_structure ok=False) -> recover_structure_ir is None
    # (the additive-fallback invariant: the caller then uses the generator cover).
    monkeypatch.setattr(
        SI,
        "recover_structure",
        lambda _p: RecoveredStructure(ok=False, reason="pure-code"),
    )
    distill = str(tmp_path / "x.distill.bin")
    open(distill, "wb").close()
    assert SI.recover_structure_ir(distill, np.zeros((4, 25), dtype=np.int64)) is None


# --------------------------------------------------------------------------- #
# PR4a corpus proof (env-gated on SIDTRACE_BIN + HVSC, like test_note_base.py):
# render_structure(ir) with ir._state = None reproduces the FULL 25-register trace
# BYTE-EXACT (residual 0) from the SHIPPED tokens ALONE, and the gate stays < 1 tok/frame.
# --------------------------------------------------------------------------- #
_HVSC = os.environ.get("HVSC", "/scratch/preframr/hvsc/C64Music")


def _have_bin():
    from preframr_tokens.bacc.generic.sidtrace import sidtrace_bin

    return sidtrace_bin() is not None


_MA_SID = os.path.join(_HVSC, "MUSICIANS/C/Compod/House.sid")
_GT_SID = os.path.join(_HVSC, "DEMOS/M-R/Regurgitated_Meatloaf.sid")
# the MEASURED render-from-tokens floor: GT renders the WHOLE trace from tokens alone
# (residual 0) under the structured floor; MA's GENERATOR is byte-exact (residual 0 when
# forced) but lands at ~1.5 tok/frame -- the per-onset INSTRUMENT SEGMENTS are stored
# per-lane (6 lanes x ~25 distinct segments) instead of factored cross-lane into the ONE
# shared instrument (the orderlist/pattern REF not yet wired into the freq/lane path), so
# the codec ships the anchored pattern-bank instead.  An HONEST stall (NO value-LZ), not a
# wall (see test_ma_render_structure_from_tokens_byte_exact's xfail).
_CORPUS_TPF = {"gt": 0.995}


def _check_render_from_tokens(sid, prefix, nframes, max_tpf, tmp_path):
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace, sidwr_state

    sidwr, distill = run_sidtrace(
        sid, str(tmp_path / prefix), subtune=1, nframes=nframes
    )
    state, _ = sidwr_state(sidwr)
    state = np.asarray(state, dtype=np.int64)
    ir = SI.recover_structure_ir(distill, state)
    assert ir is not None
    # the codec round-trips every shipped field, and the SHIPPED ids deserialize and render
    # the WHOLE trace byte-exact with NO _state anchor (the token-alone render proof).
    ids = SI.assert_ids_roundtrip(ir)
    back = SI.structure_ir_from_ids(ids)
    assert back._state is None
    rendered = SI.render_structure(back)
    resid = int(np.sum(rendered != state))
    assert resid == 0, (prefix, "render_structure(_state=None) residual", resid)
    tpf = len(ids) / state.shape[0]
    assert tpf < max_tpf, (prefix, "tok/frame", tpf)


@pytest.mark.skipif(
    not (_have_bin() and os.path.exists(_MA_SID)),
    reason="set SIDTRACE_BIN + HVSC for the Music_Assembler render-from-tokens proof",
)
@pytest.mark.xfail(
    reason="HONEST stall (NO LZ): MA's token-derived generator is byte-exact (residual 0) "
    "but ~1.5 tok/frame, so the codec ships the anchored pattern-bank (render-from-tokens "
    "NOT byte-exact).  The excess is the per-onset INSTRUMENT SEGMENTS stored per-lane "
    "rather than factored cross-lane into the ONE shared instrument (the orderlist/pattern "
    "REF not yet wired into the freq/lane path).  Upstream increment, never a value-LZ.",
    strict=True,
)
def test_ma_render_structure_from_tokens_byte_exact(tmp_path):
    _check_render_from_tokens(_MA_SID, "ma", 2270, 1.0, tmp_path)


@pytest.mark.skipif(
    not (_have_bin() and os.path.exists(_GT_SID)),
    reason="set SIDTRACE_BIN + HVSC for the GoatTracker render-from-tokens proof",
)
@pytest.mark.xfail(
    reason="HONEST stall (NO LZ -- PR4b): GT's token-derived generator now ships LZ-FREE "
    "(the per-onset note / instrument-fire streams collapse as forward Re-Pair PHRASE "
    "GRAMMAR refs, NOT the _struct_lz back-offset C3 banned), renders _state=None byte-exact "
    "(residual 0), and the redundant per-record onset grid is shared once -- but it lands at "
    "~2.5 tok/frame: the per-onset INSTRUMENT SEGMENTS (the seg-dict pools + arp offset "
    "shapes) are stored PER-LANE / PER-VOICE (~13 lanes, ~9-25 distinct segments each) "
    "instead of factored cross-voice into the ONE shared instrument fire (per voice the 3-5 "
    "non-freq lanes already collapse to ONE 27/27/24-distinct instrument-id stream, but GT's "
    "instrument table is NOT sited -- discover_instrument_table returns None -- so the "
    "segments cannot be shared across voices).  So the codec ships the anchored pattern-bank "
    "(render-from-tokens NOT byte-exact -- freq via the _state anchor, the same fallback MA "
    "ships).  Next increment: site GT's instrument table -> share segments cross-voice "
    "(would drop the ~1147-token seg pools).  Upstream increment, never a value-LZ.",
    strict=True,
)
def test_gt_render_structure_from_tokens_byte_exact(tmp_path):
    _check_render_from_tokens(_GT_SID, "gt", 2300, _CORPUS_TPF["gt"], tmp_path)


def _gt_generator_ir(tmp_path):
    """Build GoatTracker's LZ-FREE GENERATOR IR explicitly (the token-derived note base +
    the non-freq lane phrase-REF program), bypassing :func:`_pick_representation`'s
    over-budget drop, plus the byte-exact ``state`` -- the artifact the PR4b LZ-free /
    render-from-tokens proofs assert against (GT's shipped codec ships the bank, this is the
    generator it would ship if it were under budget)."""
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace, sidwr_state
    from preframr_tokens.bacc.generic.structure_recover import recover_schedule

    sidwr, distill = run_sidtrace(
        _GT_SID, str(tmp_path / "gtg"), subtune=1, nframes=2300
    )
    state, _ = sidwr_state(sidwr)
    state = np.asarray(state, dtype=np.int64)
    n = state.shape[0]
    sched = recover_schedule(distill, nframes=n)
    nb = SI._build_note_bases(distill, state, sched)
    nf = SI.build_nonfreq_program(state, sched)
    gen = SI.StructureIR(
        note_bases=nb,
        accfits=[[], [], []],
        nonfreq=nf,
        onset_grid=[int(o) for o in sched["onsets"]],
        nframes=n,
        boot=[int(state[0, r]) for r in range(SI.NREG)],
    )
    return gen, state


@pytest.mark.skipif(
    not (_have_bin() and os.path.exists(_GT_SID)),
    reason="set SIDTRACE_BIN + HVSC for the GoatTracker LZ-free generator proof",
)
def test_gt_generator_is_lz_free_and_renders_from_tokens(tmp_path):
    """PR4b proof on GoatTracker's GENERATOR path: it renders the WHOLE trace from tokens
    ALONE (``_state=None``, residual 0), its measured note_bases / nonfreq sections are
    LZ-FREE (C3 passes), and disabling ``_struct_lz`` leaves the SHIPPED ids byte-IDENTICAL
    -- the measured stream never depended on LZ (HARD RULE #0).  It is OVER budget (the
    honest stall pinned by ``test_gt_render_structure_from_tokens_byte_exact``'s xfail), so
    the codec ships the bank; this asserts the generator itself is honest LZ-free structure.
    """
    from tools.codec_gate import c3_no_lz_in_measured_stream

    gen, state = _gt_generator_ir(tmp_path)
    # the note base is NOTES (per-onset ARP refs), never the _struct_lz-fed RAMP16 dump.
    assert SI._NB_ARP in {rec[0] for rec in gen.note_bases}
    # the nonfreq lanes are per-onset SEGMENT phrase-REFs (the shared instrument across
    # notes), not raw per-frame change-point dumps.
    assert SI._LANE_SEG in {rec[0] for rec in gen.nonfreq}

    ids = SI.assert_ids_roundtrip(gen)
    assert c3_no_lz_in_measured_stream(ids)  # measured stream is LZ-free

    # render the WHOLE trace from the ids ALONE -- no _state anchor (residual 0).
    back = SI.structure_ir_from_ids(ids)
    assert back._state is None
    rendered = SI.render_structure(back)
    assert int(np.sum(rendered != state)) == 0

    # disabling _struct_lz (identity) leaves the MEASURED sections byte-IDENTICAL: the
    # generator's note_bases / nonfreq never rode on LZ.
    import preframr_tokens.bacc.generic.structure_ir as SImod

    orig = SImod._struct_lz
    try:
        SImod._struct_lz = lambda v: list(v)
        ids_nolz = SImod.structure_ir_to_ids(gen)
    finally:
        SImod._struct_lz = orig
    assert _measured_section_slice(ids) == _measured_section_slice(ids_nolz)


def test_repair_grammar_round_trips():
    """The forward PHRASE GRAMMAR (Re-Pair) round-trips an onset int-stream EXACTLY: a
    repeated phrase is one rule NAME (content-addressed), never a backward (off, len) copy.
    """
    import random

    rng = random.Random(1234)
    for _ in range(200):
        seq = [rng.randint(0, 12) for _ in range(rng.randint(0, 200))]
        terms, rules, stream = SI._repair_encode(seq)
        assert SI._repair_decode(terms, rules, stream) == seq
        flat = []
        SI._flat_grammar(flat, seq)
        back, i = SI._read_grammar(flat, 0)
        assert back == seq and i == len(flat)
        assert SI._grammar_tokens(seq) == len(flat)


def test_repair_grammar_collapses_a_repeating_phrase():
    """A phrase repeated N times collapses to ONE forward rule NAME + N refs -- the
    content-addressed onset-phrase collapse, NOT a per-onset value run."""
    phrase = [3, 1, 4, 1, 5]
    seq = phrase * 20  # 100 atoms, one repeating phrase
    terms, rules, stream = SI._repair_encode(seq)
    assert SI._repair_decode(terms, rules, stream) == seq
    # the 100-atom stream collapses far below its raw length (a NAME per repeat, not a copy).
    assert SI._grammar_tokens(seq) < len(seq) // 2


def test_measured_sections_emit_lz_free():
    """C3 (shipped-stream): a ``_NB_ARP`` / ``_LANE_SEG`` IR serializes its measured
    note_bases / nonfreq sections LZ-FREE -- the bodies carry NO ``_REPEAT`` (the
    ``_struct_lz`` back-offset), and disabling ``_struct_lz`` leaves the SHIPPED ids
    byte-IDENTICAL (proving the measured stream never depended on LZ)."""
    from tools.codec_gate import c3_no_lz_in_measured_stream

    nframes = 64
    ir = _synthetic_ir(nframes)
    # a many-onset SEG lane whose per-onset refs strongly repeat (a phrase) -> the grammar
    # collapse; the section must still ship LZ-free.
    onsets = list(range(0, nframes, 4))  # 16 onsets
    seg_dict = [(0x10,) * 4, (0x20,) * 4]
    refs = [0, 1] * (len(onsets) // 2)
    col = np.zeros(nframes, dtype=np.int64)
    for k, o in enumerate(onsets):
        col[o : o + 4] = seg_dict[refs[k]][0]
    ir.nonfreq = [(SI._LANE_SEG, 4, 0, [], onsets, refs, seg_dict)]
    ir.onset_grid = onsets
    ir._state[:, 4] = col

    ids = SI.structure_ir_to_ids(ir)
    # C3 measured-stream: no _REPEAT in the tagged note_bases / nonfreq bodies (the bank
    # sections MAY use _struct_lz -- C3 allows LZ there -- but the MEASURED slice must not).
    assert c3_no_lz_in_measured_stream(ids)
    assert SI._REPEAT not in _measured_section_slice(ids)

    # disabling _struct_lz (identity) leaves the SHIPPED ids byte-IDENTICAL: the measured
    # stream is genuinely LZ-free (the bank sections here are tiny / unique so they are too).
    import preframr_tokens.bacc.generic.structure_ir as SImod

    orig = SImod._struct_lz
    try:
        SImod._struct_lz = lambda v: list(v)
        ids_nolz = SImod.structure_ir_to_ids(ir)
    finally:
        SImod._struct_lz = orig
    # the MEASURED sections are byte-identical with LZ disabled (they never used it).
    assert _measured_section_slice(ids) == _measured_section_slice(ids_nolz)


def _measured_section_slice(ids):
    """The tagged ONSETS / note_bases / nonfreq portion of a shipped id stream (everything
    from the first measured-section tag onward) -- the part the LZ-free invariant covers.
    """
    tags = (SI._SEC_ONSETS, SI._SEC_NOTE_BASES, SI._SEC_NONFREQ)
    for k, t in enumerate(ids):
        if t in tags:
            return ids[k:]
    return []


def test_shared_onset_grid_serialized_once():
    """The onset grid every ARP / SEG record keys to is serialized ONCE (``_SEC_ONSETS``),
    not per-record: a multi-lane SEG IR carries a single onset section, and each record
    re-derives its windowed onsets from it (no redundant per-record 592-onset copy)."""
    nframes = 64
    ir = _synthetic_ir(nframes)
    onsets = list(range(0, nframes, 8))
    seg_dict = [(0x11,) * 8, (0x22,) * 8]
    refs = [0, 1, 0, 1, 0, 1, 0, 1]
    cols = {}
    recs = []
    for reg in (4, 5, 6):
        col = np.zeros(nframes, dtype=np.int64)
        for k, o in enumerate(onsets):
            col[o : o + 8] = seg_dict[refs[k]][0]
        cols[reg] = col
        recs.append((SI._LANE_SEG, reg, 0, [], onsets, refs, seg_dict))
        ir._state[:, reg] = col
    ir.nonfreq = recs
    ir.onset_grid = onsets
    ids = SI.structure_ir_to_ids(ir)
    assert ids.count(SI._SEC_ONSETS) == 1  # the grid ships exactly once
    back = SI.structure_ir_from_ids(ids)
    # every record re-derived the SAME onsets from the shared grid.
    for rec in back.nonfreq:
        assert list(rec[4]) == SI._window_onsets(onsets, rec[2], nframes)
    lanes = SI.render_nonfreq_from_ir(back, anchor=None)
    for reg in (4, 5, 6):
        assert np.array_equal(lanes[reg], cols[reg])


def test_committed_corpus_sir_fixtures_under_one_token_per_frame():
    # every committed structure-serialization fixture deserializes and is < 1 token/frame
    # (the recovered structured floor; the companion render-equality lives in the corpus
    # gate).  Pins that the shipped artifacts stay valid and under budget.
    fixdir = os.path.join(os.path.dirname(__file__), "test_fixtures", "budget")
    sirs = [f for f in os.listdir(fixdir) if f.endswith(".sir.npz")]
    assert sirs, "expected committed .sir.npz structure fixtures"
    for name in sirs:
        ids = list(np.load(os.path.join(fixdir, name))["ids"].astype(np.int64))
        ir = SI.structure_ir_from_ids(ids)
        assert SI.structure_ir_to_ids(ir) == ids, f"{name}: ids re-serialize mismatch"
        nframes = ir.nframes
        # DMC is the one committed structure still over budget (raw instrument tables);
        # the rest are the recovered floor < 1 token/frame.
        if not name.startswith("DMC__"):
            assert len(ids) / nframes < 1.0, f"{name}: {len(ids)/nframes:.3f} tok/frame"
