"""Tests for RawVibratoEnvelopePass + step-mode OscillationEnvelopeDecoder."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.raw_vibrato_pass import RawVibratoEnvelopePass
from preframr_tokens.macros.slope_pass import SlopePass
from preframr_tokens.stfconstants import FRAME_REG, OSCILLATE_ENV_OP, SET_OP, SLOPE_OPS


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


def _vibrato_rows(reg, values, gap=1):
    """A raw per-frame FREQ-SET stream: each value on its own frame, with
    ``gap`` frames (the value held) between successive SETs."""
    rows = [_frame(), _row(reg, values[0])]
    for v in values[1:]:
        for _ in range(gap):
            rows.append(_frame())
        rows.append(_row(reg, v))
    return rows


def _encode(rows):
    df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
    return RawVibratoEnvelopePass().apply(df, args=FakeArgs(vibrato_env_pass=True))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


def _squeeze(seq):
    """Collapse consecutive duplicate writes: re-writing a held FREQ value is
    inaudible, so the change-sequence is the audio-equivalence invariant."""
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


class TestRawVibratoEnvelopePass(unittest.TestCase):
    def test_disabled_returns_unchanged(self):
        rows = _vibrato_rows(0, [120, 100, 120, 100, 120, 100])
        df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
        result = RawVibratoEnvelopePass().apply(
            df.copy(), args=FakeArgs(vibrato_env_pass=False)
        )
        self.assertTrue(result.equals(df))

    def test_slope_pass_leaves_short_vibrato_raw(self):
        """The >=5-frame SLOPE gate never sees alternating per-frame vibrato."""
        rows = _vibrato_rows(0, [120, 100, 120, 100, 120, 100])
        df = SlopePass().apply(pd.DataFrame(rows), args=FakeArgs(slope_pass=True))
        self.assertFalse(bool(df["op"].isin(SLOPE_OPS).any()))

    def test_gap1_collapses_and_round_trips_exactly(self):
        rows = _vibrato_rows(0, [120, 100, 120, 100, 120, 100], gap=1)
        encoded = _encode(rows)
        self.assertTrue(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))
        self.assertEqual(_decoded_reg(pd.DataFrame(rows), 0), _decoded_reg(encoded, 0))

    def test_gap2_collapses_and_round_trips_audio_equivalent(self):
        """gap>1 holds each terminal; the forward-filled per-frame value
        sequence (what the SID actually plays) is preserved."""
        rows = _vibrato_rows(0, [120, 100, 120, 100, 120, 100], gap=2)
        encoded = _encode(rows)
        self.assertTrue(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))
        self.assertEqual(
            _squeeze(_decoded_reg(pd.DataFrame(rows), 0)),
            _squeeze(_decoded_reg(encoded, 0)),
        )

    def test_monotonic_not_collapsed(self):
        rows = _vibrato_rows(0, [10, 20, 30, 40, 50], gap=1)
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))

    def test_below_min_not_collapsed(self):
        rows = _vibrato_rows(0, [120, 100], gap=1)
        encoded = _encode(rows)
        self.assertFalse(bool((encoded["op"] == OSCILLATE_ENV_OP).any()))


if __name__ == "__main__":
    unittest.main()
