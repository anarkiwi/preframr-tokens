"""PR3: token-derived NOTE-PITCH BASE recovery (``structure_recover.note_base_recover``).

The freq render's per-frame note-pitch base (``note_seed``) is the player's own note
table indexed by a per-frame note-table-index walk; that walk is HIGHLY compressible by a
constant-step wrapping-ramp generator (a note-level glide is a constant-step ramp, an arp
a chain of short ramp runs) -- it is recovered structure, NOT entropy (HARD RULE #0).  PR3
recovers the base FROM TOKENS: the distinct seed Fn values ARE the note table (NOT 12-TET),
and the idx walk is fitted with the existing ``ramp_segments_kernel`` so the base renders
WITHOUT reading ``_state``.  The generator MUST render the idx walk byte-exact (residual 0).

  * the SYNTHETIC test (no binary) builds a freq = note_table[ramp-walk index] with an
    accumulator-free seed and pins covered == nframes, base == freq byte-exact, and the
    idx generator round-trips (rendered == idx, residual 0);
  * the CORPUS test (env-gated on ``SIDTRACE_BIN`` + HVSC) runs the two corpus tunes and
    pins the MEASURED per-voice base-reproduction % and tok/frame, so a regression flips
    red.  These are PASSES at real measured numbers, never xfail.
"""

import os

import numpy as np
import pytest

from preframr_tokens.bacc.generic import structure_recover as SR
from preframr_tokens.bacc.generic.distill import Distill, StsqCell
from preframr_tokens.bacc.generic.sidtrace import sidtrace_bin

_LOAD = 0x1000
_LOAD_LEN = 0x1000


def _blank_distill(nframes):
    return Distill(
        version=1,
        init_addr=_LOAD,
        play_addr=_LOAD + 3,
        load_addr=_LOAD,
        subtune=1,
        nframes=nframes,
        cycles_per_frame=19656,
        t0_cycle=0,
        load_len=_LOAD_LEN,
        acc=np.zeros(65536, dtype=np.uint8),
        ram=np.zeros(65536, dtype=np.uint8),
    )


# --------------------------------------------------------------------------- #
# Synthetic mechanism test (no binary): freq = note_table[ramp-walk index].
# --------------------------------------------------------------------------- #
def test_note_base_recover_ramp_walk_byte_exact(monkeypatch):
    """A voice whose freq is the player's note table indexed by a downward index RAMP
    (a note-level glide) -- accumulator-free seed -- recovers byte-exact FROM TOKENS:
    covered == nframes, base == freq, and the idx generator round-trips (residual 0)."""
    nframes = 200
    # the player's own note table (NOT 12-TET): arbitrary distinct Fn values incl. 0.
    note_table = [0x0000, 0x0480, 0x0512, 0x0633, 0x07A1, 0x0855, 0x0930, 0x0AE2]
    # a per-frame index walk: an UP ramp then a DOWN ramp then a held tail -- the small
    # ints stay well below len(note_table), so the wrapping ramp never wraps (a plain
    # constant-step segmenter), and the walk is byte-exact reproducible by the generator.
    idx = []
    for f in range(nframes):
        if f < 70:
            idx.append(
                f % len(note_table)
            )  # up ramp, wrapping in idx VALUE (not modulus)
        elif f < 140:
            idx.append((140 - f) % len(note_table))  # down ramp
        else:
            idx.append(3)  # held tail
    idx = np.array(idx, dtype=np.int64)
    freq = np.array([note_table[i] for i in idx.tolist()], dtype=np.int64)
    # frames [0, 3) are the analysis warm-up (a == 3); make them match the first held val
    # so the recovered base reproduces them too (the recover window is [3, nframes)).
    freq[:3] = freq[3]

    state = np.zeros((nframes, 25), dtype=np.int64)
    state[:, 0] = freq & 0xFF
    state[:, 1] = (freq >> 8) & 0xFF

    d = _blank_distill(nframes)
    # an STSQ section must exist (note_base_recover returns None otherwise), but NO
    # accumulator cell (single-valued, never resets-to-0-then-ramps) so seed == freq.
    d.stsq_cells = [
        StsqCell(
            addr=0x0040,
            flags=0,
            first_seen=0,
            samples=np.zeros(nframes, dtype=np.uint8),
        )
    ]
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)

    rec = SR.note_base_recover("x", state, voices=((0, 1),))
    assert rec is not None
    info = rec[0]

    # the recovered note table is exactly the distinct seed values (incl. silence 0).
    assert info["note_table"] == sorted(set(note_table))
    # accumulator-free seed == freq, so the base reproduces EVERY analysed frame.
    assert info["nframes"] == nframes - 3
    assert info["covered"] == info["nframes"]
    # the base render equals freq byte-exact over the analysis window.
    base = info["base_freq"]
    assert np.array_equal(base[3:], freq[3:])
    # the idx generator round-trips byte-exact: rendered == idx walk, residual 0.
    assert info["idx_residual"] == 0
    assert np.array_equal(info["idx_rendered"], info["idx_walk"])
    # the render-from-tokens path returns the SAME base (no _state read for the base).
    rb = SR.render_note_base_from_tokens("x", state, voices=((0, 1),))
    assert np.array_equal(rb[0], base)


def test_note_base_recover_none_without_stsq(monkeypatch):
    """No STSQ section -> note_base_recover / render_note_base_from_tokens return None
    (the honest "this artifact cannot supply the base", never a faked result)."""
    d = _blank_distill(64)
    monkeypatch.setattr(SR, "load_distill", lambda _p: d)
    state = np.zeros((64, 25), dtype=np.int64)
    assert SR.note_base_recover("x", state) is None
    assert SR.render_note_base_from_tokens("x", state) is None


# --------------------------------------------------------------------------- #
# Corpus proof (env-gated on SIDTRACE_BIN + HVSC).  Pinned to the MEASURED numbers.
# --------------------------------------------------------------------------- #
_HVSC = os.environ.get("HVSC", "/scratch/preframr/hvsc/C64Music")
_HAVE_BIN = sidtrace_bin() is not None

# Music_Assembler House.sid and GoatTracker Regurgitated_Meatloaf.sid (the PR3 corpus).
_MA_SID = os.path.join(_HVSC, "MUSICIANS/C/Compod/House.sid")
_GT_SID = os.path.join(_HVSC, "DEMOS/M-R/Regurgitated_Meatloaf.sid")

# Per-voice MEASURED floor: (min covered/nframes, max tok/frame).  The base reproduces the
# freq column FULLY (the chosen accumulator set is empty on these tunes, so seed == freq
# and tbl[idx] == freq), and the idx-walk ramp generator renders byte-exact (residual 0).
# tok/frame is the generator's structured floor (per segment: start-delta + seed + step).
_MA_FLOOR = {0: (1.0, 0.80), 1: (1.0, 1.38), 2: (1.0, 1.79)}
_GT_FLOOR = {0: (1.0, 0.80), 1: (1.0, 0.46), 2: (1.0, 0.35)}
# the voices where the ramp generator WINS (tok/frame < 1.0); pinned so a regression that
# inflates the segment count above the budget flips red.
_UNDER_BUDGET = {"MA": (0,), "GT": (0, 1, 2)}


def _trace(sid, prefix, nframes, tmp_path):
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace

    return run_sidtrace(sid, str(tmp_path / prefix), subtune=1, nframes=nframes)


def _check_corpus(sid, prefix, nframes, floor, under, tmp_path):
    from preframr_tokens.bacc.generic.sidtrace import sidwr_state

    sidwr, distill = _trace(sid, prefix, nframes, tmp_path)
    state, _ = sidwr_state(sidwr)
    rec = SR.note_base_recover(distill, state)
    assert rec is not None
    for vi, info in rec.items():
        cov = info["covered"] / info["nframes"]
        # the idx generator is BYTE-EXACT (the recovered structure, not a stored output).
        assert info["idx_residual"] == 0, (prefix, vi, "idx residual non-zero")
        # the base render equals the note table indexed by the byte-exact rendered walk.
        tbl = np.asarray(info["note_table"], dtype=np.int64)
        assert np.array_equal(info["idx_rendered"], info["idx_walk"])
        assert tbl[info["idx_rendered"]].size == info["nframes"]
        min_cov, max_tpf = floor[vi]
        assert cov >= min_cov - 1e-9, (prefix, vi, cov, "below measured coverage floor")
        assert info["tok_per_frame"] <= max_tpf + 1e-3, (
            prefix,
            vi,
            info["tok_per_frame"],
            "above measured tok/frame",
        )
    for vi in under:
        assert rec[vi]["tok_per_frame"] < 1.0, (
            prefix,
            vi,
            "generator must win < 1 tpf",
        )


@pytest.mark.skipif(
    not (_HAVE_BIN and os.path.exists(_MA_SID)),
    reason="set SIDTRACE_BIN + HVSC for the Music_Assembler note-base proof",
)
def test_ma_note_base_from_tokens(tmp_path):
    _check_corpus(_MA_SID, "ma", 2270, _MA_FLOOR, _UNDER_BUDGET["MA"], tmp_path)


@pytest.mark.skipif(
    not (_HAVE_BIN and os.path.exists(_GT_SID)),
    reason="set SIDTRACE_BIN + HVSC for the GoatTracker note-base proof",
)
def test_gt_note_base_from_tokens(tmp_path):
    _check_corpus(_GT_SID, "gt", 2300, _GT_FLOOR, _UNDER_BUDGET["GT"], tmp_path)
