"""PROTOTYPE gate: generic tracker-STRUCTURE recovery from the SDST artifact.

This proves the structure-recovery fix (#12) the OUTPUT-FIT generic path misses: a
structured player (JCH NewPlayer ``10.sid``) is recovered to its real tracker IR --
a shared instrument pool, per-row note references, factored patterns + orderlist,
and pitch-cleaned note table -- DIRECTLY from the distill artifact, byte-exact, at
< 1 token/frame (vs the 4.94 tok/frame the output-fit recovery floors at).

The recovery is GENERIC: every table base / stride / reference is DERIVED from the
captured access pattern (SNAP + the access map + the SDDF backward-slice leaves), NO
hardcoded per-tune addresses.  It is ADDITIVE: on a pure-code tune (A Mind Is Born --
no instrument / pattern table) discovery finds nothing and the caller FALLS BACK to
the generator cover (the AMIB gate stays green, asserted by
``test_amib_generic_budget.py``).

The whole-tune proof needs the ``preframr-sidtrace-statemachine`` tracer (it emits
the SDDF data-flow section the shipping ``preframr-sidtrace`` does not), so it is
env-gated (skip-if-absent), exactly like the other whole-tune recovery proofs.  The
SELF-CONTAINED unit tests below exercise the generic discoverers + the byte-exact
re-encode gate + the token budget on synthetic artifacts (no binary), so the default
CI gate still covers the mechanism.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import structure_recover as SR

# The statemachine tracer (emits SDDF); a SEPARATE checkout from the default
# ``SIDTRACE_BIN`` (which predates the data-flow section).  Override with
# ``SIDTRACE_SM_BIN``.
_SM_BIN = os.environ.get(
    "SIDTRACE_SM_BIN",
    "/scratch/anarkiwi/preframr/preframr-sidtrace-statemachine/build/sidtrace",
)
_JCH_SID = os.environ.get(
    "JCH_SID", "/scratch/preframr/hvsc/C64Music/MUSICIANS/G/Goto80/10.sid"
)
_AMIB_SID = os.environ.get(
    "AMIB_SID", "/scratch/preframr/hvsc/C64Music/MUSICIANS/L/Lft/A_Mind_Is_Born.sid"
)
_HAVE_SM = os.path.exists(_SM_BIN)


def _trace(sid, prefix, nframes, tmp_path):
    """Run the statemachine tracer to produce ``<prefix>.distill.bin`` (+ sidwr)."""
    import subprocess

    out = str(tmp_path / prefix)
    subprocess.run([_SM_BIN, sid, "1", str(nframes), out], check=True)
    return out + ".distill.bin", out + ".sidwr.bin"


# --------------------------------------------------------------------------- #
# Self-contained mechanism unit tests (no binary).
# --------------------------------------------------------------------------- #
def test_token_cost_nibble_vs_byte():
    assert SR._tok(0) == 1 and SR._tok(15) == 1
    assert SR._tok(16) == 2 and SR._tok(255) == 2


def test_pattern_grammar_roundtrip_lossless():
    # the NewPlayer grammar decode -> re-encode is byte-exact for a hand pattern.
    # markers (>=0x80) then a note (<0x80); 0x7F ends the pattern.
    ram = np.zeros(65536, dtype=np.uint8)
    pat = [
        0x8C,
        0xA1,
        0x10,
        0xC2,
        0x20,
        0x00,
        0x7F,
    ]  # dur, instr, note; cmd, note; rest; end
    ram[0x2000 : 0x2000 + len(pat)] = pat
    pairs = SR.reencode_patterns(ram, [0x2000])
    reencoded, snap = pairs[0]
    assert reencoded == snap == pat


def test_backward_lz_collapses_repeats():
    # a repeated block must factor to a tiny encoding: the first copy as literals, the
    # rest as back-references (one long overlapping run is optimal greedy LZ).  The
    # encoded cost (lit + 2*matches) must be far below the flat token count.
    block = [3, 1, 4, 1, 5, 9, 2, 6]
    stream = block * 8  # 64 tokens
    lit, mat = SR._backward_lz(stream, min_match=3)
    assert mat >= 1
    assert lit + 2 * mat < len(stream) // 2  # collapsed well below half the flat size


def test_instrument_stride_from_leaf_lattice():
    # a stride-8 instrument table is the GCD of the SDDF leaf-address differences.
    base = 0x1892
    slices = [
        SR._SdwSlice(pc=0x13E4, reg=5, leaf_addrs=[base + 8 * i for i in (0, 2, 5, 9)]),
        SR._SdwSlice(
            pc=0x13FB, reg=6, leaf_addrs=[base + 1 + 8 * i for i in (0, 2, 5)]
        ),
    ]

    class _D:
        load_addr = 0x1000
        load_len = 0xC92

    got = SR.discover_instrument_table(_D(), slices)
    assert got == (base, 8)


# --------------------------------------------------------------------------- #
# Whole-tune proof (env-gated on the statemachine tracer).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_HAVE_SM and os.path.exists(_JCH_SID)),
    reason="set SIDTRACE_SM_BIN + JCH_SID for the JCH structure-recovery proof",
)
def test_jch_structure_recovers_byte_exact_under_one_token_per_frame(tmp_path):
    from preframr_tokens.bacc.generic.sidtrace import sidwr_state

    distill, sidwr = _trace(_JCH_SID, "jch", 1500, tmp_path)
    state, _ = sidwr_state(sidwr)
    nframes = state.shape[0]

    struct = SR.recover_structure(distill)
    assert struct.ok, struct.reason

    # GENERIC: derived from the artifact, not hardcoded (the JCH 10.sid ground truth).
    assert (struct.instr_base, struct.instr_stride) == (0x1892, 8)
    assert struct.n_instruments == 16  # 16 real instruments (output-fit pool was 21)
    assert len(struct.note_table) == 30  # 30 grid pitches (output-fit table was 572)
    assert struct.n_rows == 319 and struct.n_patterns == 33

    # BYTE-EXACT: every pattern decodes and re-encodes to its exact SNAP bytes.
    assert SR.pattern_roundtrip_ok(struct)

    # PITCH CLEANING: subtracting the captured porta/vibrato accumulators renders the
    # freq output BYTE-EXACT (residual 0) on every voice -- the displaced "note table"
    # (hundreds of entries, growing with playback) is really a handful of grid pitches.
    clean = SR.clean_pitches_residual(distill, state)
    assert clean is not None
    for voice, info in clean.items():
        assert info["residual"] == 0, (voice, info)
        assert (
            info["pitches"] < info["displaced"]
        )  # collapsed below the displaced count

    # TOKEN FLOOR: < 1 token/frame (vs the 4.94 tok/frame output-fit recovery).
    total, brk = SR.token_budget(struct, frames=nframes)
    assert (
        total < nframes
    ), f"{total} tok >= {nframes} frames ({brk['tok_per_frame']:.3f}/fr)"

    # NO ENTROPY WALL (HARD RULE #0): the STSQ accumulator capture caps at 512 frames,
    # but the freq OUTPUT over the WHOLE tune is a compact piecewise-generator program
    # (CONST holds + AFFINE vibrato + QUADRATIC porta).  The shipping byte-exact cover
    # proves this on every lane already -- recover the program, render it, and require
    # the freq render byte-exact full-length at FAR fewer segments than frames (a wall
    # would force ~one segment per frame).  This forbids quietly accepting a window cap.
    from preframr_tokens.bacc import tracker_ir

    covers = tracker_ir.cover_all_lanes(state)
    for vidx in range(3):
        cov = covers[f"{vidx}:freq"]
        # the cover tiles the lane byte-exact (cover_lane is residual-zero by contract);
        # the segment count is the generator program size -- must be << nframes.
        assert len(cov) < nframes // 2, (
            f"voice {vidx} freq needs {len(cov)} segments over {nframes} frames -- "
            f"that is an entropy wall, not a generator program"
        )


@pytest.mark.skipif(
    not (_HAVE_SM and os.path.exists(_AMIB_SID)),
    reason="set SIDTRACE_SM_BIN + AMIB_SID for the A Mind Is Born fallback proof",
)
def test_amib_pure_code_falls_back_to_generator_cover(tmp_path):
    # A Mind Is Born is 256 bytes of pure code: NO pattern/instrument table, so the
    # structure recovery must find NOTHING and report ``ok=False`` (the caller then
    # uses the generator cover -- the additive-fix invariant; the AMIB budget gate
    # in test_amib_generic_budget.py asserts the generator-cover floor stays 0.41).
    distill, _ = _trace(_AMIB_SID, "amib", 500, tmp_path)
    struct = SR.recover_structure(distill)
    assert not struct.ok
    assert "pure-code" in struct.reason or "pointer table" in struct.reason
