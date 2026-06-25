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
    # 1/2 are silent (freq 0) -> a _NB_RAWLZ of zeros.
    nb_v0 = (SI._NB_TABLE, [seed], [0], [0], [0], 2, align, [seed] * align)
    zeros = [0] * nframes
    note_bases = [nb_v0, (SI._NB_RAWLZ, zeros), (SI._NB_RAWLZ, zeros)]

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
    prog = SI.build_nonfreq_program(state)
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
# the MEASURED render-from-tokens floor (residual 0 always; tok/frame must stay < 1).
_CORPUS_TPF = {"ma": 0.97, "gt": 0.50}


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
def test_ma_render_structure_from_tokens_byte_exact(tmp_path):
    _check_render_from_tokens(_MA_SID, "ma", 2270, _CORPUS_TPF["ma"], tmp_path)


@pytest.mark.skipif(
    not (_have_bin() and os.path.exists(_GT_SID)),
    reason="set SIDTRACE_BIN + HVSC for the GoatTracker render-from-tokens proof",
)
def test_gt_render_structure_from_tokens_byte_exact(tmp_path):
    _check_render_from_tokens(_GT_SID, "gt", 2300, _CORPUS_TPF["gt"], tmp_path)


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
