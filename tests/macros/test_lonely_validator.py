"""Tests for LonelyWriteValidatorPass (item 11)."""

import unittest

import pandas as pd

from preframr_tokens.macros.lonely_validator import (
    LonelyWriteValidatorPass,
    UnmodelledLonelyWriteError,
)
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FILTER_REG,
    FLIP_OP,
    FRAME_REG,
    MODE_VOL_REG,
    OSC_SUBREG_ANCHOR_HI,
    OSCILLATE_ENV_OP,
    SET_OP,
    SLOPE_FREQ_LO_OP,
    SLOPE_SUBREG_TERMINAL_HI,
    TRANSPOSE_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _df(rows):
    return pd.DataFrame(rows)


def _r(reg, val, op=SET_OP, subreg=-1):
    return {"reg": reg, "val": val, "op": op, "subreg": subreg, "diff": 32}


def _frame():
    return _r(FRAME_REG, 0)


def _validate(rows, **flags):
    flags.setdefault("strict_lonely", True)
    return LonelyWriteValidatorPass().apply(_df(rows), args=FakeArgs(**flags))


class TestLonelyValidator(unittest.TestCase):
    def test_disabled_is_noop(self):
        rows = [_frame(), _r(0, 5)]
        out = LonelyWriteValidatorPass().apply(
            _df(rows), args=FakeArgs(strict_lonely=False)
        )
        self.assertEqual(len(out), len(rows))

    def test_first_voice_write_passes(self):
        rows = [_frame(), _r(0, 5)]
        self.assertIsNotNone(_validate(rows))

    def test_second_lonely_set_raises(self):
        rows = [_frame(), _r(0, 5), _frame(), _r(0, 9)]
        with self.assertRaises(UnmodelledLonelyWriteError):
            _validate(rows)

    def test_diff_op_raises(self):
        rows = [_frame(), _r(0, 5), _frame(), _r(0, 1, op=DIFF_OP)]
        with self.assertRaises(UnmodelledLonelyWriteError):
            _validate(rows)

    def test_filter_route_carveout(self):
        rows = [_frame(), _r(FILTER_REG, 1), _frame(), _r(FILTER_REG, 2)]
        self.assertIsNotNone(_validate(rows))

    def test_master_volume_carveout(self):
        rows = [_frame(), _r(MODE_VOL_REG, 15), _frame(), _r(MODE_VOL_REG, 10)]
        self.assertIsNotNone(_validate(rows))

    def test_gate_off_terminal_carveout(self):
        rows = [
            _frame(),
            _r(4, 0x00),
            _frame(),
            _r(4, 0x10),
        ]
        self.assertIsNotNone(_validate(rows))

    def test_trajectory_anchor_before_slope(self):
        rows = [
            _frame(),
            _r(0, 5),
            _frame(),
            _r(0, 7),
            _r(0, 0, op=SLOPE_FREQ_LO_OP, subreg=SLOPE_SUBREG_TERMINAL_HI),
        ]
        self.assertIsNotNone(_validate(rows))

    def test_trajectory_anchor_before_oscillate(self):
        rows = [
            _frame(),
            _r(0, 5),
            _frame(),
            _r(0, 7),
            _r(0, 0, op=OSCILLATE_ENV_OP, subreg=OSC_SUBREG_ANCHOR_HI),
        ]
        self.assertIsNotNone(_validate(rows))

    def test_anchor_trailing_a_flip(self):
        rows = [
            _frame(),
            _r(2, 5),
            _frame(),
            _r(2, 0, op=FLIP_OP),
            _frame(),
            _r(2, 9),
        ]
        self.assertIsNotNone(_validate(rows))

    def test_anchor_leading_a_transpose(self):
        rows = [
            _frame(),
            _r(0, 5),
            _frame(),
            _r(0, 7),
            _r(0, 3, op=TRANSPOSE_OP),
        ]
        self.assertIsNotNone(_validate(rows))


if __name__ == "__main__":
    unittest.main()
