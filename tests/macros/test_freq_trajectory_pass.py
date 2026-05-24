"""Round-trip + subtype tests for FreqTrajectoryPass + FreqTrajectoryDecoder:
ramp/oscillate/run classification, gap-tolerant holds, the signed-byte delta
escape, periodic collapse, and the isolated-write fall-through."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_TRAJ_OP,
    FT_PERIODIC_BIT,
    FT_SUBREG_FLAGS,
    FT_SUBTYPE_MASK,
    FT_SUBTYPE_MONOTONE_RAMP,
    FT_SUBTYPE_OSCILLATE,
    FT_SUBTYPE_RUN,
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


def _stream(reg, values, gap=1):
    rows = []
    for v in values:
        for _ in range(gap):
            rows.append(_r(FRAME_REG, 0))
        rows.append(_r(reg, v))
    rows.append(_r(FRAME_REG, 0))
    return pd.DataFrame(rows)


def _per_frame(df, reg):
    dec = expand_ops(df, strict=False).reset_index(drop=True)
    regs = dec["reg"].to_numpy()
    vals = dec["val"].to_numpy()
    cur = 0
    out = []
    for i in range(len(dec)):
        r = int(regs[i])
        if r == FRAME_REG:
            out.append(cur)
        elif r == reg:
            cur = int(vals[i])
    return out


def _apply(df):
    return FreqTrajectoryPass().apply(
        df.copy(), args=FakeArgs(freq_trajectory_pass=True)
    )


def _flags(out):
    f = out[(out["op"] == FREQ_TRAJ_OP) & (out["subreg"] == FT_SUBREG_FLAGS)]
    return [int(v) for v in f["val"].tolist()]


class TestFreqTrajectoryPass(unittest.TestCase):
    def _assert_lossless(self, reg, values, gap=1):
        df = _stream(reg, values, gap=gap)
        out = _apply(df)
        self.assertEqual(_per_frame(df, reg), _per_frame(out, reg))
        return out

    def test_disabled_is_noop(self):
        df = _stream(0, [120, 122, 120, 122, 120, 122])
        out = FreqTrajectoryPass().apply(
            df.copy(), args=FakeArgs(freq_trajectory_pass=False)
        )
        self.assertTrue(out.equals(df))

    def test_monotone_ramp(self):
        out = self._assert_lossless(0, list(range(100, 110)))
        subtypes = [v & FT_SUBTYPE_MASK for v in _flags(out)]
        self.assertIn(FT_SUBTYPE_MONOTONE_RAMP, subtypes)

    def test_fast_vibrato_gap1(self):
        out = self._assert_lossless(7, [120, 122, 120, 122, 120, 122, 120, 122])
        self.assertIn(FT_SUBTYPE_OSCILLATE, [v & FT_SUBTYPE_MASK for v in _flags(out)])

    def test_intermittent_vibrato_gap2(self):
        out = self._assert_lossless(14, [120, 122, 120, 122, 120, 122], gap=2)
        self.assertIn(FT_SUBTYPE_OSCILLATE, [v & FT_SUBTYPE_MASK for v in _flags(out)])

    def test_drifting_amplitude_oscillation(self):
        out = self._assert_lossless(0, [100, 110, 98, 112, 96, 114, 94])
        self.assertIn(FT_SUBTYPE_OSCILLATE, [v & FT_SUBTYPE_MASK for v in _flags(out)])

    def test_large_delta_escape(self):
        out = self._assert_lossless(21, [100, 5000, 120, 5200, 90])
        self.assertGreater(len(_flags(out)), 0)

    def test_periodic_collapse(self):
        out = self._assert_lossless(0, [100, 101, 103] * 4)
        self.assertTrue(
            any(v & FT_PERIODIC_BIT for v in _flags(out)), "expected periodic collapse"
        )

    def test_generic_run(self):
        out = self._assert_lossless(7, [100, 150, 200, 260, 330])
        self.assertIn(FT_SUBTYPE_RUN, [v & FT_SUBTYPE_MASK for v in _flags(out)])

    def test_isolated_write_not_collapsed(self):
        rows = [_r(FRAME_REG, 0), _r(0, 5), _r(FRAME_REG, 0), _r(7, 9)]
        rows += [_r(FRAME_REG, 0)]
        out = _apply(pd.DataFrame(rows))
        self.assertEqual(int((out["op"] == FREQ_TRAJ_OP).sum()), 0)


if __name__ == "__main__":
    unittest.main()
