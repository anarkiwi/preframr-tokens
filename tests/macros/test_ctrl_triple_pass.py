"""Tests for CtrlTriplePass + CtrlTripleDecoder (spec #6)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.ctrl_triple_pass import CtrlTriplePass
from preframr_tokens.stfconstants import (
    CTRL_TRIPLE_OP,
    CTRL_TRIPLE_SUBREG_COUNT,
    FRAME_REG,
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
    flags.setdefault("ctrl_triple_pass", True)
    return CtrlTriplePass().apply(df.copy(), args=FakeArgs(**flags))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestCtrlTriplePass(unittest.TestCase):
    def test_disabled_is_noop(self):
        rows = [_frame(), _r(4, 0x11), _frame(), _r(4, 0x41), _frame(), _r(4, 0x40)]
        df = pd.DataFrame(rows)
        out = CtrlTriplePass().apply(df.copy(), args=FakeArgs(ctrl_triple_pass=False))
        self.assertTrue(out.equals(df))

    def test_triple_collapses_and_round_trips(self):
        rows = [_frame(), _r(4, 0x11), _frame(), _r(4, 0x41), _frame(), _r(4, 0x40)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertEqual(
            len(out[out["op"] == CTRL_TRIPLE_OP]), CTRL_TRIPLE_SUBREG_COUNT
        )
        self.assertEqual(len(out[(out["reg"] == 4) & (out["op"] == SET_OP)]), 0)
        self.assertEqual(_decoded_reg(df, 4), _decoded_reg(out, 4))

    def test_only_pair_not_collapsed(self):
        rows = [_frame(), _r(4, 0x11), _frame(), _r(4, 0x41)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == CTRL_TRIPLE_OP).any()))

    def test_six_writes_make_two_triples(self):
        rows = []
        for v in (0x11, 0x21, 0x41, 0x40, 0x80, 0x81):
            rows.append(_frame())
            rows.append(_r(11, v))
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertEqual(
            len(out[out["op"] == CTRL_TRIPLE_OP]), 2 * CTRL_TRIPLE_SUBREG_COUNT
        )
        self.assertEqual(_decoded_reg(df, 11), _decoded_reg(out, 11))

    def test_gap_breaks_triple(self):
        rows = [
            _frame(),
            _r(4, 0x11),
            _frame(),
            _r(4, 0x41),
            _frame(),
            _frame(),
            _r(4, 0x40),
        ]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == CTRL_TRIPLE_OP).any()))
        self.assertEqual(_decoded_reg(df, 4), _decoded_reg(out, 4))


if __name__ == "__main__":
    unittest.main()
