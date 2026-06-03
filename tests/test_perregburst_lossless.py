"""Adversarial round-trip property test for the PerRegBurst freq delta encoder. PerRegBurst re-encodes
per-frame freq SETs as DIFF/FLIP runs; decoding must reproduce the input exactly (the RESID=0 goal).
After the lossless run-detection rewrite (FLIP fires only on genuine >=3 alternation), random
small-delta walks round-trip exactly -- this guards that, with DIFF-only as the lossless floor.
"""

import unittest

import pandas as pd
import pytest

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    SET_OP,
    STAMP_REF_OP,
    _MIN_DIFF,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args

_ARGS = default_tokenizer_args(freq_trajectory_pass=False)
_REG = 7


def _build(values):
    rows = []
    for v in values:
        rows.append(
            {"reg": int(FRAME_REG), "val": 0, "op": int(SET_OP), "diff": int(_MIN_DIFF)}
        )
        if v is not None:
            rows.append(
                {"reg": _REG, "val": int(v), "op": int(SET_OP), "diff": int(_MIN_DIFF)}
            )
    return pd.DataFrame(rows)


def _bad_frames(values):
    df = _build(values)
    out = PerRegBurstPass().apply(df.copy(), args=_ARGS)
    gt = register_state(df)[:, _REG]
    dec = register_state(out)[:, _REG]
    n = min(len(gt), len(dec))
    return [(i, int(gt[i]), int(dec[i])) for i in range(n) if gt[i] != dec[i]]


def _lcg(seed, n):
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x)
    return out


def _walk_seq(seed_base, trial):
    """The shared 12-step small-delta random walk used by both lossless property tests."""
    r = _lcg(seed_base + trial * 7, 14)
    cur = 100 + (r[0] % 40)
    seq = [cur]
    for j in range(1, 12):
        cur = max(1, min(250, cur + [0, 1, -1, 2, -2][r[j] % 5]))
        seq.append(cur if (r[j] % 7) else None)
    return seq


_TRIALS = 300
_DIFF_TRIALS = 150
_CHUNK = 25


@pytest.mark.parametrize("lo", range(0, _TRIALS, _CHUNK))
def test_random_small_delta_walks_round_trip(lo):
    """One ``_CHUNK``-trial slice of the small-delta walk sweep; chunked into independent
    parametrized items so pytest-xdist fans the 300 trials across workers instead of
    pinning one worker for the whole serial loop (the suite wall-clock floor)."""
    fails = []
    for trial in range(lo, min(lo + _CHUNK, _TRIALS)):
        seq = _walk_seq(12345, trial)
        if _bad_frames(seq):
            fails.append(seq)
    assert (
        fails == []
    ), f"{len(fails)} freq walks lossy in [{lo},{lo + _CHUNK}); e.g. {fails[:2]}"


@pytest.mark.parametrize("lo", range(0, _DIFF_TRIALS, _CHUNK))
def test_diff_only_round_trips_lossless(lo):
    """The lossless floor: with FLIP disabled (DIFF-only), every walk must round-trip -- the
    fallback a correct encoder uses, and proof the lossiness is the FLIP/run detection, not DIFF.
    """
    from preframr_tokens.stfconstants import DIFF_OP

    bad = []
    for trial in range(lo, min(lo + _CHUNK, _DIFF_TRIALS)):
        r = _lcg(999 + trial * 13, 14)
        cur = 100 + (r[0] % 40)
        seq = [cur]
        for j in range(1, 12):
            cur = max(1, min(250, cur + [0, 1, -1, 2, -2][r[j] % 5]))
            seq.append(cur if (r[j] % 7) else None)
        df = _build(seq)
        out = PerRegBurstPass(opcodes=[int(DIFF_OP)]).apply(df.copy(), args=_ARGS)
        gt = register_state(df)[:, _REG]
        dec = register_state(out)[:, _REG]
        n = min(len(gt), len(dec))
        if any(gt[i] != dec[i] for i in range(n)):
            bad.append(seq)
    assert (
        bad == []
    ), f"DIFF-only lossy on {len(bad)} walks in [{lo},{lo + _CHUNK}): {bad[:2]}"


class TestPerRegBurstGapPval(unittest.TestCase):
    """Regression for the pval mis-base: a DELAY carries a value across elapsed frames (no row at those
    f-numbers), so a DIFF at the next write must base on the CARRIED value, not 0. The bug encoded
    delta=val-0 and decoded carried+delta (1394 PWM 64 -> 96); here reg7 45 after a carried 40 must
    decode to 45, not 85.
    """

    def test_diff_bases_on_carried_value_across_delay_gap(self):
        def row(reg, val, diff=_MIN_DIFF):
            return {
                "reg": int(reg),
                "val": int(val),
                "op": int(SET_OP),
                "diff": int(diff),
            }

        df = pd.DataFrame(
            [
                row(FRAME_REG, 0, 19000),
                row(_REG, 40),
                row(DELAY_REG, 3, 19000),
                row(_REG, 45),
                row(FRAME_REG, 0, 19000),
                row(_REG, 50),
            ]
        )
        out = PerRegBurstPass().apply(df.copy(), args=_ARGS)
        gt = register_state(df)[:, _REG]
        dec = register_state(out)[:, _REG]
        n = min(len(gt), len(dec))
        bad = [(i, int(gt[i]), int(dec[i])) for i in range(n) if gt[i] != dec[i]]
        self.assertEqual(bad, [], f"DIFF mis-based across a DELAY-carried gap: {bad}")


class TestPerRegBurstEmptyCandBarrier(unittest.TestCase):
    """Regression: a non-empty stamp barrier with an empty candidate set must not
    crash ``_encode``. The old ``cand[[<bool list>]]`` filter, on an empty ``cand``,
    selected empty COLUMNS (``cand[[]]``) -> (0, 0) -> ``change_df["val"]`` KeyError.
    Triggered by a STAMP_REF before a freq SET with zero PWM writes (empty pcm
    candidate); aborted the codebook parse on the first such song.
    """

    def test_empty_pcm_candidate_with_barrier_does_not_crash(self):
        def row(reg, val, op=SET_OP):
            return {
                "reg": int(reg),
                "val": int(val),
                "op": int(op),
                "diff": int(_MIN_DIFF),
            }

        df = pd.DataFrame(
            [
                row(FRAME_REG, 0),
                row(0, 100, op=STAMP_REF_OP),
                row(FRAME_REG, 0),
                row(0, 120),
            ]
        )
        out = PerRegBurstPass().apply(df.copy(), args=_ARGS)
        self.assertIn("val", out.columns)
        self.assertFalse(out.empty)


if __name__ == "__main__":
    unittest.main()
