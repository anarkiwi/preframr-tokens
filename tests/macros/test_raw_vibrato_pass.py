"""Tests for RawVibratoEnvelopePass + FreqVibratoDecoder (delta-cycle vibrato)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.raw_vibrato_pass import RawVibratoEnvelopePass
from preframr_tokens.macros.slope_pass import SlopePass
from preframr_tokens.stfconstants import FRAME_REG, FREQ_VIBRATO_OP, SET_OP, SLOPE_OPS


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame():
    return {
        "reg": FRAME_REG,
        "val": 0,
        "diff": 19000,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _row(reg, val, diff=32):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _vibrato_rows(reg, values):
    rows = [_frame(), _row(reg, values[0])]
    for v in values[1:]:
        rows.append(_frame())
        rows.append(_row(reg, v))
    return rows


def _encode(rows):
    df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
    return RawVibratoEnvelopePass().apply(df, args=FakeArgs(vibrato_env_pass=True))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestRawVibratoEnvelopePass(unittest.TestCase):
    def test_disabled_returns_unchanged(self):
        rows = _vibrato_rows(0, [120, 100, 120, 100, 120, 100])
        df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
        out = RawVibratoEnvelopePass().apply(
            df.copy(), args=FakeArgs(vibrato_env_pass=False)
        )
        self.assertTrue(out.equals(df))

    def test_period2_collapses_and_round_trips_exactly(self):
        rows = _vibrato_rows(0, [120, 122, 120, 122, 120, 122, 120, 122])
        encoded = _encode(rows)
        self.assertTrue(bool((encoded["op"] == FREQ_VIBRATO_OP).any()))
        self.assertFalse(bool(encoded["op"].isin(SLOPE_OPS).any()))
        self.assertEqual(_decoded_reg(pd.DataFrame(rows), 0), _decoded_reg(encoded, 0))

    def test_period3_round_trips_exactly(self):
        rows = _vibrato_rows(0, [200, 205, 198, 200, 205, 198, 200, 205, 198])
        encoded = _encode(rows)
        self.assertTrue(bool((encoded["op"] == FREQ_VIBRATO_OP).any()))
        self.assertEqual(_decoded_reg(pd.DataFrame(rows), 0), _decoded_reg(encoded, 0))

    def test_uncapped_long_run_round_trips(self):
        rows = _vibrato_rows(0, [120, 122] * 300)
        encoded = _encode(rows)
        osc = encoded[encoded["op"] == FREQ_VIBRATO_OP]
        self.assertTrue(len(osc) > 0)
        self.assertEqual(_decoded_reg(pd.DataFrame(rows), 0), _decoded_reg(encoded, 0))

    def test_non_periodic_left_for_freq_run(self):
        rows = _vibrato_rows(0, [120, 122, 119, 130, 100, 140])
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == FREQ_VIBRATO_OP).any()))

    def test_below_min_not_collapsed(self):
        rows = _vibrato_rows(0, [120, 122])
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == FREQ_VIBRATO_OP).any()))


if __name__ == "__main__":
    unittest.main()
