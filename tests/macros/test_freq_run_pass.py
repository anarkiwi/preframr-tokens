"""Tests for FreqRunPass + FreqRunDecoder (spec #4)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_run_pass import FreqRunPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_RUN_OP,
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


def _run_stream(reg, values):
    rows = []
    for v in values:
        rows.append(_frame())
        rows.append(_r(reg, v))
    return pd.DataFrame(rows)


def _apply(df, **flags):
    flags.setdefault("freq_run_pass", True)
    return FreqRunPass().apply(df.copy(), args=FakeArgs(**flags))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestFreqRunPass(unittest.TestCase):
    def test_disabled_is_noop(self):
        df = _run_stream(0, [10, 90, 30])
        out = FreqRunPass().apply(df.copy(), args=FakeArgs(freq_run_pass=False))
        self.assertTrue(out.equals(df))

    def test_pair_collapses_and_round_trips(self):
        df = _run_stream(0, [100, 300])
        out = _apply(df)
        self.assertTrue(bool((out["op"] == FREQ_RUN_OP).any()))
        self.assertEqual(len(out[(out["reg"] == 0) & (out["op"] == SET_OP)]), 0)
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_nonarithmetic_run_round_trips(self):
        df = _run_stream(0, [100, 305, 207, 999, 12, 640])
        out = _apply(df)
        self.assertTrue(bool((out["op"] == FREQ_RUN_OP).any()))
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_long_run_chunks_and_round_trips(self):
        df = _run_stream(7, list(range(1000, 1020)))
        out = _apply(df)
        self.assertTrue(bool((out["op"] == FREQ_RUN_OP).any()))
        self.assertEqual(_decoded_reg(df, 7), _decoded_reg(out, 7))

    def test_16bit_values_round_trip(self):
        df = _run_stream(0, [40000, 12345, 60000])
        out = _apply(df)
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_single_set_not_collapsed(self):
        rows = [_frame(), _r(0, 5), _frame(), _r(7, 9), _frame(), _r(0, 6)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == FREQ_RUN_OP).any()))


if __name__ == "__main__":
    unittest.main()
