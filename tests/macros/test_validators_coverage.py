"""Targeted coverage tests for ``preframr.macros.validators``."""

import unittest

import pandas as pd

from preframr_tokens.macros.validators import (
    validate_back_refs,
    validate_pattern_overlays,
)
from preframr_tokens.stfconstants import (
    DO_LOOP_OP,
    FRAME_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    SET_OP,
)


def _row(**kw):
    """Build a single df row with the columns the validators read.
    Defaults match a no-op SET_OP row so callers only set what they
    care about.
    """
    base = {"op": SET_OP, "reg": 0, "subreg": -1, "val": 0}
    base.update(kw)
    return base


def _df(rows):
    return pd.DataFrame(rows)


class TestValidatePatternOverlaysEarlyReturn(unittest.TestCase):
    def test_no_op_column_returns_true(self):
        df = pd.DataFrame({"reg": [0, 1], "val": [10, 20]})
        self.assertTrue(validate_pattern_overlays(df))


class TestValidateBackRefsEarlyReturns(unittest.TestCase):
    def test_no_op_column_returns_true(self):
        df = pd.DataFrame({"reg": [0, 1], "val": [10, 20]})
        self.assertTrue(validate_back_refs(df))


class TestValidateBackRefsPatternReplayBranch(unittest.TestCase):
    def _frame_padding(self, n):
        """Return ``n`` rows that each advance the output frame
        counter (FRAME_REG SET rows). Lets us build a known
        ``output_frame_count`` before the macro under test.
        """
        return [_row(op=SET_OP, subreg=-1, reg=FRAME_REG, val=0) for _ in range(n)]

    def test_pattern_replay_well_formed_distance_length(self):
        rows = self._frame_padding(5) + [
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=3,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_LEN,
                val=2,
                reg=FRAME_REG,
            ),
        ]
        self.assertTrue(validate_back_refs(_df(rows)))

    def test_pattern_replay_distance_before_frame_zero_raises(self):
        rows = [
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=1,
                reg=FRAME_REG,
            ),
        ]
        with self.assertRaises(AssertionError) as ctx:
            validate_back_refs(_df(rows))
        self.assertIn("reaches before frame 0", str(ctx.exception))

    def test_pattern_replay_length_without_preceding_distance_raises(self):
        rows = [
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_LEN,
                val=2,
                reg=FRAME_REG,
            ),
        ]
        with self.assertRaises(AssertionError) as ctx:
            validate_back_refs(_df(rows))
        self.assertIn("without a complete DIST pair", str(ctx.exception))

    def test_pattern_replay_overlay_count_skipped(self):
        rows = self._frame_padding(3) + [
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=2,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_LEN,
                val=1,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
                val=0,
                reg=FRAME_REG,
            ),
        ]
        self.assertTrue(validate_back_refs(_df(rows)))

    def test_pattern_replay_invalid_subreg_raises(self):
        rows = [
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=9,
                val=0,
                reg=FRAME_REG,
            ),
        ]
        with self.assertRaises(AssertionError) as ctx:
            validate_back_refs(_df(rows))
        self.assertIn("PATTERN_REPLAY subreg=9", str(ctx.exception))


class TestValidateBackRefsPatternOverlayAndLoop(unittest.TestCase):
    def test_pattern_overlay_op_does_not_advance_frames(self):
        rows = [
            _row(
                op=PATTERN_OVERLAY_OP,
                subreg=PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_OVERLAY_OP,
                subreg=PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_OVERLAY_OP,
                subreg=PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=1,
                reg=FRAME_REG,
            ),
        ]
        with self.assertRaises(AssertionError) as ctx:
            validate_back_refs(_df(rows))
        self.assertIn(
            "PATTERN_REPLAY distance=1 reaches before frame 0", str(ctx.exception)
        )

    def test_unmatched_do_loop_end_counts_no_frames(self):
        """A DO_LOOP-end with no open loop (empty do_stack) adds no frame, so a following PATTERN_REPLAY
        with distance 1 still reaches before frame 0."""
        rows = [
            _row(op=DO_LOOP_OP, subreg=-1, val=0, reg=FRAME_REG),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=1,
                reg=FRAME_REG,
            ),
        ]
        with self.assertRaises(AssertionError):
            validate_back_refs(_df(rows))

    def test_do_loop_body_repeats_extend_expanded_frame_count(self):
        """A DO_LOOP body re-executes n_iter times; a PATTERN_REPLAY after the loop whose distance reaches
        into those EXPANDED frames is in bounds (matches expand_loops). Body=2 frames x 3 iters = 6
        expanded frames, so distance=4 resolves to target=2. A linear walk counting the body once
        (output=2) would wrongly flag distance=4 as reaching before frame 0 -- the false positive this
        guards against.
        """
        rows = [
            _row(op=DO_LOOP_OP, subreg=0, val=3, reg=FRAME_REG),
            _row(reg=FRAME_REG),
            _row(reg=FRAME_REG),
            _row(op=DO_LOOP_OP, subreg=1, val=0, reg=FRAME_REG),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
                val=0,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
                val=4,
                reg=FRAME_REG,
            ),
            _row(
                op=PATTERN_REPLAY_OP,
                subreg=PATTERN_REPLAY_SUBREG_LEN,
                val=1,
                reg=FRAME_REG,
            ),
        ]
        self.assertTrue(validate_back_refs(_df(rows)))


if __name__ == "__main__":
    unittest.main()
