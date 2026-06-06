"""Tests for ``preframr_tokens.audit_primitives``. Pure-Python token-level audit functions (tier_accuracy, detect_tail_cycle, distinct_n) used by the generalization-gate callback in the main preframr repo and by the post-hoc audit scripts in integration_tests/profile/."""

import unittest

import pandas as pd

from preframr_tokens.audit_primitives import (
    detect_tail_cycle,
    distinct_n,
    op_atom_profile,
    register_state,
    tier_accuracy,
)
from preframr_tokens.stfconstants import FRAME_REG, SET_OP


def _stream(reg, values, gap=1):
    rows = []
    for v in values:
        for _ in range(gap):
            rows.append(
                {"reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1, "diff": 32}
            )
        rows.append({"reg": reg, "val": v, "op": SET_OP, "subreg": -1, "diff": 32})
    rows.append({"reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1, "diff": 32})
    return pd.DataFrame(rows)


class TestDistinctN(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(distinct_n([], n=4), 0)

    def test_too_short(self):
        self.assertEqual(distinct_n([1, 2, 3], n=4), 0)

    def test_unique(self):
        self.assertEqual(distinct_n([1, 2, 3, 4, 5], n=4), 2)

    def test_repeats(self):
        self.assertEqual(distinct_n([1, 1, 1, 1, 1, 1], n=4), 1)


class TestDetectTailCycle(unittest.TestCase):
    def test_short_input_returns_none(self):
        self.assertIsNone(detect_tail_cycle([1, 2, 3], tail_window=128))

    def test_constant_token_period_one(self):
        tokens = [42] * 200
        result = detect_tail_cycle(tokens, tail_window=128, max_period=32)
        self.assertEqual(result, {"period": 1, "repeats": 128})

    def test_period_two(self):
        tokens = [1, 2] * 100
        result = detect_tail_cycle(tokens, tail_window=128, max_period=32)
        self.assertEqual(result["period"], 2)

    def test_random_no_cycle(self):
        tokens = list(range(200))
        self.assertIsNone(
            detect_tail_cycle(tokens, tail_window=128, max_period=32, min_repeats=3)
        )


class TestTierAccuracy(unittest.TestCase):
    def test_all_correct(self):
        predicted = [1, 2, 3]
        ground_truth = [1, 2, 3]
        tier_map = {1: "structural", 2: "content", 3: "content"}
        out = tier_accuracy(predicted, ground_truth, tier_map)
        self.assertEqual(out["per_tier"]["structural"]["acc"], 1.0)
        self.assertEqual(out["per_tier"]["content"]["acc"], 1.0)
        self.assertEqual(out["n_positions"], 3)

    def test_content_over_structural_ratio(self):
        predicted = [1, 1, 2, 2]
        ground_truth = [1, 2, 2, 3]
        tier_map = {1: "structural", 2: "content", 3: "content"}
        out = tier_accuracy(predicted, ground_truth, tier_map)
        self.assertEqual(out["per_tier"]["structural"]["acc"], 1.0)
        self.assertEqual(out["per_tier"]["content"]["acc"], 1 / 3)
        self.assertAlmostEqual(out["content_over_structural"], 1 / 3)

    def test_unknown_tier_bucketed(self):
        predicted = [5]
        ground_truth = [5]
        out = tier_accuracy(predicted, ground_truth, {})
        self.assertIn("_unknown", out["per_tier"])

    def test_length_mismatch_truncates(self):
        out = tier_accuracy([1, 2, 3], [1], {1: "structural"})
        self.assertEqual(out["n_positions"], 1)


class TestProfilingReductions(unittest.TestCase):
    def test_register_state_tracks_per_frame_values(self):
        st = register_state(_stream(0, [100, 102, 100]))
        self.assertEqual(st.shape[1], 25)
        col0 = [int(x) for x in st[:, 0]]
        self.assertIn(100, col0)
        self.assertIn(102, col0)

    def test_op_atom_profile_counts(self):
        df = _stream(0, [10, 20, 30])
        prof = op_atom_profile(df)
        self.assertEqual(prof["total_atoms"], len(df))
        self.assertEqual(prof["op_hist"][SET_OP], len(df))
        self.assertGreater(prof["atoms_per_frame"], 0)
        self.assertIn("per_tier", prof)


if __name__ == "__main__":
    unittest.main()
