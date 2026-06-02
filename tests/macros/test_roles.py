"""Tests for ``preframr_tokens.macros.roles``."""

from __future__ import annotations

import unittest

from preframr_tokens.macros.roles import (
    DISTANCE_PAIR_OPS,
    DistancePairSpec,
    distance_pair_role,
    frame_weight_role,
    slope_subreg_role,
)
from preframr_tokens.stfconstants import (
    DO_LOOP_OP,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    SET_OP,
    FREQ_TRAJ_OP,
    FT_SUBREG_RUNTIME,
    FT_SUBREG_TERMINAL_HI,
    FT_SUBREG_TERMINAL_LO,
)


class TestDistancePairRole(unittest.TestCase):
    def test_pattern_replay_slots(self):
        self.assertEqual(
            distance_pair_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI),
            "dist_hi",
        )
        self.assertEqual(
            distance_pair_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_LO),
            "dist_lo",
        )
        self.assertEqual(
            distance_pair_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_LEN), "len"
        )
        self.assertEqual(
            distance_pair_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_OVERLAY_COUNT),
            "ov_count",
        )

    def test_non_distance_op_returns_none(self):
        self.assertIsNone(distance_pair_role(SET_OP, 0))
        self.assertIsNone(distance_pair_role(DO_LOOP_OP, 0))

    def test_unknown_subreg_returns_none(self):
        self.assertIsNone(distance_pair_role(PATTERN_REPLAY_OP, 99))


class TestDistancePairOps(unittest.TestCase):
    def test_table_contents(self):
        self.assertIn(PATTERN_REPLAY_OP, DISTANCE_PAIR_OPS)
        pr = DISTANCE_PAIR_OPS[PATTERN_REPLAY_OP]
        self.assertEqual(pr.label, "PR")
        self.assertEqual(
            pr.extra_subregs, frozenset({PATTERN_REPLAY_SUBREG_OVERLAY_COUNT})
        )

    def test_spec_is_frozen(self):
        spec = DISTANCE_PAIR_OPS[PATTERN_REPLAY_OP]
        with self.assertRaises(Exception):
            spec.label = "mutated"  # type: ignore[misc]
        self.assertIsInstance(spec, DistancePairSpec)


class TestSlopeSubregRole(unittest.TestCase):
    def test_each_role(self):
        op = FREQ_TRAJ_OP
        self.assertEqual(slope_subreg_role(op, FT_SUBREG_TERMINAL_HI), "terminal_hi")
        self.assertEqual(slope_subreg_role(op, FT_SUBREG_TERMINAL_LO), "terminal_lo")
        self.assertEqual(slope_subreg_role(op, FT_SUBREG_RUNTIME), "runtime")

    def test_non_slope_op_returns_none(self):
        self.assertIsNone(slope_subreg_role(SET_OP, FT_SUBREG_TERMINAL_HI))
        self.assertIsNone(slope_subreg_role(DO_LOOP_OP, FT_SUBREG_TERMINAL_HI))

    def test_unknown_subreg_returns_none(self):
        self.assertIsNone(slope_subreg_role(FREQ_TRAJ_OP, 99))


class TestFrameWeightRole(unittest.TestCase):
    def test_pattern_replay_len(self):
        self.assertEqual(
            frame_weight_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_LEN),
            "pattern_replay_len",
        )

    def test_do_loop_len(self):
        self.assertEqual(frame_weight_role(DO_LOOP_OP, 0), "do_loop_len")

    def test_slope_runtime(self):
        self.assertEqual(
            frame_weight_role(FREQ_TRAJ_OP, FT_SUBREG_RUNTIME), "slope_runtime"
        )

    def test_pattern_replay_dist_hi_is_not_a_weight_source(self):
        self.assertIsNone(
            frame_weight_role(PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI)
        )

    def test_do_loop_non_zero_subreg(self):
        self.assertIsNone(frame_weight_role(DO_LOOP_OP, 1))

    def test_unrelated_op(self):
        self.assertIsNone(frame_weight_role(SET_OP, 0))


if __name__ == "__main__":
    unittest.main()
