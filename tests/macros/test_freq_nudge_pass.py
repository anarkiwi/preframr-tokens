"""Tests for FreqNudgePass + FreqNudgeDecoder (spec #3)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_nudge_pass import FreqNudgePass
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FRAME_REG,
    FREQ_NUDGE_OP,
    SET_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _r(reg, val, op=SET_OP, subreg=-1):
    return {
        "reg": reg,
        "val": val,
        "op": op,
        "subreg": subreg,
        "diff": 32,
        "description": 0,
    }


def _frame():
    return _r(FRAME_REG, 0)


def _apply(df, **flags):
    flags.setdefault("freq_nudge_pass", True)
    return FreqNudgePass().apply(df.copy(), args=FakeArgs(**flags))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestFreqNudgePass(unittest.TestCase):
    def test_disabled_is_noop(self):
        df = pd.DataFrame([_frame(), _r(0, 5), _frame(), _r(0, 1, op=DIFF_OP)])
        out = FreqNudgePass().apply(df.copy(), args=FakeArgs(freq_nudge_pass=False))
        self.assertTrue(out.equals(df))

    def test_diff_becomes_nudge_and_round_trips(self):
        rows = [_frame(), _r(0, 100)]
        rows += [_frame(), _r(0, 7, op=DIFF_OP)]
        rows += [_frame(), _r(7, 200)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertEqual(len(out[out["op"] == DIFF_OP]), 0)
        self.assertTrue(bool((out["op"] == FREQ_NUDGE_OP).any()))
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_isolated_set_becomes_absolute_nudge(self):
        rows = [_frame(), _r(0, 100)]
        for _ in range(4):
            rows.append(_frame())
        rows += [_r(0, 555)]
        for _ in range(4):
            rows.append(_frame())
        rows += [_r(7, 10)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        nudges = out[out["op"] == FREQ_NUDGE_OP]
        self.assertGreater(len(nudges), 0)
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_adjacent_sets_not_absolute_nudged(self):
        rows = [_frame(), _r(0, 10), _frame(), _r(0, 20), _frame(), _r(0, 30)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == FREQ_NUDGE_OP).any()))
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_catch_all_nudges_every_residual_set(self):
        rows = [_frame(), _r(0, 10), _frame(), _r(0, 20), _frame(), _r(0, 30)]
        df = pd.DataFrame(rows)
        out = _apply(df, lonely_catch_all=True)
        self.assertEqual(len(out[(out["reg"] == 0) & (out["op"] == SET_OP)]), 0)
        self.assertTrue(bool((out["op"] == FREQ_NUDGE_OP).any()))
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_negative_diff_round_trips(self):
        rows = [
            _frame(),
            _r(0, 500),
            _frame(),
            _r(0, -40, op=DIFF_OP),
            _frame(),
            _r(7, 1),
        ]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))


if __name__ == "__main__":
    unittest.main()
