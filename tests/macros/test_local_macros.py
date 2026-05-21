"""Roundtrip + decode smoke tests for the local-context macros."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.local_macros import CtrlBigramPass
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    DELAY_REG,
    FRAME_REG,
    SET_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame(diff=19000):
    return {
        "reg": FRAME_REG,
        "val": 1,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _row(reg, val, op=SET_OP, subreg=-1, diff=32):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": op,
        "subreg": subreg,
        "description": 0,
    }


def _sid_writes(df):
    keep = df["reg"] >= 0
    return df.loc[keep, ["reg", "val", "diff"]].reset_index(drop=True)


class TestCtrlBigramPass(unittest.TestCase):
    CTRL_REG_V0 = 4

    def test_collapses_known_bigram_pair(self):
        prev_ctrl, cur_ctrl = CTRL_BIGRAM_TABLE[0]
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_REG_V0, prev_ctrl),
                _frame(),
                _row(self.CTRL_REG_V0, cur_ctrl),
                _frame(),
            ]
        )
        out = CtrlBigramPass().apply(df, args=FakeArgs(ctrl_bigram_pass=True))
        ops = out["op"].tolist()
        self.assertIn(int(CTRL_BIGRAM_OP), ops)
        ctrl_set_rows = out[(out["reg"] == self.CTRL_REG_V0) & (out["op"] == SET_OP)]
        self.assertEqual(len(ctrl_set_rows), 0)
        bigram_rows = out[out["op"] == CTRL_BIGRAM_OP]
        self.assertEqual(len(bigram_rows), 1)
        self.assertEqual(int(bigram_rows.iloc[0]["val"]), 0)

    def test_off_by_default(self):
        prev_ctrl, cur_ctrl = CTRL_BIGRAM_TABLE[0]
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_REG_V0, prev_ctrl),
                _frame(),
                _row(self.CTRL_REG_V0, cur_ctrl),
                _frame(),
            ]
        )
        out = CtrlBigramPass().apply(df, args=FakeArgs())
        self.assertEqual(int((out["op"] == CTRL_BIGRAM_OP).sum()), 0)

    def test_skips_delay_between(self):
        prev_ctrl, cur_ctrl = CTRL_BIGRAM_TABLE[0]
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_REG_V0, prev_ctrl),
                {
                    "reg": DELAY_REG,
                    "val": 3,
                    "diff": 19000,
                    "op": SET_OP,
                    "subreg": -1,
                    "description": 0,
                },
                _frame(),
                _row(self.CTRL_REG_V0, cur_ctrl),
                _frame(),
            ]
        )
        out = CtrlBigramPass().apply(df, args=FakeArgs(ctrl_bigram_pass=True))
        self.assertEqual(int((out["op"] == CTRL_BIGRAM_OP).sum()), 0)

    def test_skips_unknown_pair(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_REG_V0, 0x77),
                _frame(),
                _row(self.CTRL_REG_V0, 0x88),
                _frame(),
            ]
        )
        out = CtrlBigramPass().apply(df, args=FakeArgs(ctrl_bigram_pass=True))
        self.assertEqual(int((out["op"] == CTRL_BIGRAM_OP).sum()), 0)

    def test_roundtrip_matches_baseline_ctrl_state(self):
        prev_ctrl, cur_ctrl = CTRL_BIGRAM_TABLE[0]
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_REG_V0, prev_ctrl),
                _frame(),
                _row(self.CTRL_REG_V0, cur_ctrl),
                _frame(),
                _frame(),
            ]
        )
        baseline_writes = _sid_writes(expand_ops(baseline.copy(), strict=False))
        encoded = CtrlBigramPass().apply(
            baseline.copy(), args=FakeArgs(ctrl_bigram_pass=True)
        )
        encoded_writes = _sid_writes(expand_ops(encoded, strict=False))
        ctrl_baseline = baseline_writes[baseline_writes["reg"] == self.CTRL_REG_V0][
            "val"
        ].tolist()
        ctrl_encoded = encoded_writes[encoded_writes["reg"] == self.CTRL_REG_V0][
            "val"
        ].tolist()
        self.assertEqual(ctrl_baseline, ctrl_encoded)


if __name__ == "__main__":
    unittest.main()
