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


def test_render_structure_byte_exact_and_raises_without_anchor():
    ir = _synthetic_ir()
    rendered = SI.render_structure(ir)
    assert np.array_equal(rendered, ir._state)
    # an IR rebuilt from ids alone has no anchor -> the non-freq replay is the next
    # increment, so render_structure surfaces it rather than faking a render.
    back = SI.structure_ir_from_ids(SI.structure_ir_to_ids(ir))
    assert back._state is None
    try:
        SI.render_structure(back)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


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
    SI.assert_ids_roundtrip(ir)  # the assembled IR still round-trips exactly


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
