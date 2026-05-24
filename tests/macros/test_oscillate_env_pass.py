"""Tests for OscillationEnvelopePass + OscillationEnvelopeDecoder (item 0)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.oscillate_env_pass import OscillationEnvelopePass
from preframr_tokens.macros.slope_pass import SlopePass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    OSC_SUBREG_COUNT,
    OSCILLATE_ENV_OP,
    SET_OP,
    SLOPE_OPS,
)


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


def _oscillation_rows(reg, terminals, runtime, start_val):
    """Build a raw per-frame SET stream: linear ramps from ``start_val``
    through each terminal over ``runtime`` frames each."""
    rows = [_frame(), _row(reg, start_val)]
    cur = start_val
    for term in terminals:
        delta = term - cur
        for j in range(1, runtime + 1):
            rows.append(_frame())
            rows.append(_row(reg, cur + (delta * j) // runtime))
        cur = term
    return rows


def _slope_only(rows):
    return SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))


def _encode(rows):
    df = _slope_only(rows)
    return OscillationEnvelopePass().apply(df, args=FakeArgs(oscillate_env_pass=True))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


def _assert_faithful(test, rows, reg):
    """OscillationEnvelopePass must never change the decoded per-frame writes
    relative to the slope-only encoding (it only re-expresses slope chains)."""
    slope_only = _slope_only(rows)
    with_osc = OscillationEnvelopePass().apply(
        slope_only.copy(), args=FakeArgs(oscillate_env_pass=True)
    )
    test.assertEqual(_decoded_reg(slope_only, reg), _decoded_reg(with_osc, reg))


class TestOscillationEnvelopePass(unittest.TestCase):
    def test_disabled_returns_unchanged(self):
        rows = _oscillation_rows(0, [150, 50, 150, 50], 5, 100)
        df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
        result = OscillationEnvelopePass().apply(
            df.copy(), args=FakeArgs(oscillate_env_pass=False)
        )
        self.assertTrue(result.equals(df))

    def test_symmetric_chain_collapses_to_one_atom(self):
        rows = _oscillation_rows(0, [150, 50, 150, 50], 5, 100)
        encoded = _encode(rows)
        osc = encoded[encoded["op"] == OSCILLATE_ENV_OP]
        self.assertEqual(len(osc), OSC_SUBREG_COUNT)
        self.assertFalse(bool(encoded["op"].isin(SLOPE_OPS).any()))
        self.assertEqual(list(osc["subreg"]), list(range(OSC_SUBREG_COUNT)))

    def test_round_trip_symmetric(self):
        rows = _oscillation_rows(0, [150, 50, 150, 50], 5, 100)
        baseline = pd.DataFrame(rows)
        encoded = _encode(rows)
        self.assertEqual(_decoded_reg(baseline, 0), _decoded_reg(encoded, 0))

    def test_osc_faithful_to_slope_only_freq(self):
        rows = _oscillation_rows(0, [150, 50, 150, 50], 5, 100)
        _assert_faithful(self, rows, 0)

    def test_round_trip_pw_reg(self):
        rows = _oscillation_rows(2, [1024, 2048, 1024, 2048], 4, 2048)
        encoded = _encode(rows)
        self.assertTrue(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))
        _assert_faithful(self, rows, 2)

    def test_non_alternating_chain_left_as_slopes(self):
        rows = [_frame(), _row(0, 0)]
        for k in range(1, 30):
            rows.append(_frame())
            rows.append(_row(0, k * 5))
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))

    def test_two_slopes_below_min_not_collapsed(self):
        rows = _oscillation_rows(0, [150, 50], 5, 100)
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))

    def test_uniform_subrun_collapses_when_chain_runtime_changes(self):
        """A chain whose per-slope runtime changes mid-way still collapses its
        leading uniform-runtime sub-run (>=3 slopes) and round-trips exactly."""
        head = _oscillation_rows(0, [150, 50, 150, 50], 5, 100)
        cur = 50
        for term in (150, 50):
            for j in range(1, 4):
                head.append(_frame())
                head.append(_row(0, cur + ((term - cur) * j) // 3))
            cur = term
        encoded = _encode(head)
        self.assertTrue(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))
        self.assertEqual(_decoded_reg(pd.DataFrame(head), 0), _decoded_reg(encoded, 0))


if __name__ == "__main__":
    unittest.main()
