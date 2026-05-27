"""Interval-coded freq V0 (``--freq-v0-interval``): byte-exact round-trip, transposition
invariance of the onset encoding, freq-regs-only scope, and default-off equivalence."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_TRAJ_OP,
    FT_SUBREG_FLAGS,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
    FT_V0_INTERVAL_BIT,
    SET_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _r(reg, val):
    return {
        "reg": reg,
        "val": val,
        "op": SET_OP,
        "subreg": -1,
        "diff": 32,
        "description": 0,
    }


def _groups_stream(reg, groups, gap=20):
    """Each group is a short osc burst; big gaps (> OSC_MAX_GAP) split groups into
    separate trajectories so the inter-onset interval coding has >= 2 onsets."""
    rows = []
    for grp in groups:
        for v in grp:
            rows.append(_r(FRAME_REG, 0))
            rows.append(_r(reg, v))
        for _ in range(gap):
            rows.append(_r(FRAME_REG, 0))
    rows.append(_r(FRAME_REG, 0))
    return pd.DataFrame(rows)


def _per_frame(df, reg):
    dec = expand_ops(df, strict=False).reset_index(drop=True)
    cur = 0
    out = []
    for _, row in dec.iterrows():
        if int(row["reg"]) == FRAME_REG:
            out.append(cur)
        elif int(row["reg"]) == reg:
            cur = int(row["val"])
    return out


def _apply(df, interval):
    return FreqTrajectoryPass().apply(
        df.copy(),
        args=FakeArgs(freq_trajectory_pass=True, freq_v0_interval=interval),
    )


def _v0_rows(out):
    o = out[
        (out["op"] == FREQ_TRAJ_OP)
        & (out["subreg"].isin([FT_SUBREG_FLAGS, FT_SUBREG_V0_HI, FT_SUBREG_V0_LO]))
    ]
    return [(int(s), int(v)) for s, v in zip(o["subreg"], o["val"])]


_MELODY = [[100, 103, 100], [140, 143, 140], [120, 123, 120]]


class TestFreqV0Interval(unittest.TestCase):
    def test_roundtrip_interval_lossless(self):
        df = _groups_stream(0, _MELODY)
        raw = _per_frame(df, 0)
        self.assertEqual(_per_frame(_apply(df, True), 0), raw)
        self.assertEqual(_per_frame(_apply(df, False), 0), raw)

    def test_interval_bit_set_after_first_trajectory(self):
        out = _apply(_groups_stream(0, _MELODY), True)
        flags = [int(v) for s, v in _v0_rows(out) if s == FT_SUBREG_FLAGS]
        self.assertFalse(flags[0] & FT_V0_INTERVAL_BIT)
        self.assertTrue(all(f & FT_V0_INTERVAL_BIT for f in flags[1:]))

    def test_transposition_invariance(self):
        df = _groups_stream(0, _MELODY)
        df_t = _groups_stream(0, [[v + 50 for v in g] for g in _MELODY])
        base = _v0_rows(_apply(df, True))
        trans = _v0_rows(_apply(df_t, True))
        self.assertEqual(base[3:], trans[3:])
        self.assertNotEqual(base[:3], trans[:3])

    def test_freq_regs_only_pw_untouched(self):
        df = _groups_stream(2, _MELODY)
        self.assertEqual(_v0_rows(_apply(df, True)), _v0_rows(_apply(df, False)))

    def test_default_off_unchanged(self):
        df = _groups_stream(0, _MELODY)
        no_flag = FreqTrajectoryPass().apply(
            df.copy(), args=FakeArgs(freq_trajectory_pass=True)
        )
        self.assertEqual(_v0_rows(no_flag), _v0_rows(_apply(df, False)))


if __name__ == "__main__":
    unittest.main()
