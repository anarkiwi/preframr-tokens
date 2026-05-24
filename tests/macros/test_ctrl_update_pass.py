"""Tests for CtrlUpdatePass + CtrlUpdateDecoder (residual CTRL catch-all)."""

import unittest

import pandas as pd

from preframr_tokens.macros.ctrl_update_pass import CtrlUpdatePass
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import CTRL_UPDATE_OP, FRAME_REG, SET_OP

CTRL = CTRL_REGS_BY_VOICE[0]


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


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestCtrlUpdatePass(unittest.TestCase):
    def test_disabled_is_noop(self):
        df = pd.DataFrame([_frame(), _r(CTRL, 0x11), _frame(), _r(CTRL, 0x41)])
        out = CtrlUpdatePass().apply(df.copy(), args=FakeArgs(lonely_catch_all=False))
        self.assertTrue(out.equals(df))

    def test_catch_all_tags_every_ctrl_set_and_round_trips(self):
        df = pd.DataFrame(
            [
                _frame(),
                _r(CTRL, 0x11),
                _frame(),
                _r(CTRL, 0x41),
                _frame(),
                _r(CTRL, 0x40),
            ]
        )
        out = CtrlUpdatePass().apply(df.copy(), args=FakeArgs(lonely_catch_all=True))
        self.assertEqual(len(out[(out["reg"] == CTRL) & (out["op"] == SET_OP)]), 0)
        self.assertEqual(int((out["op"] == CTRL_UPDATE_OP).sum()), 3)
        self.assertEqual(_decoded_reg(df, CTRL), _decoded_reg(out, CTRL))

    def test_non_ctrl_untouched(self):
        df = pd.DataFrame([_frame(), _r(0, 100), _frame(), _r(0, 200)])
        out = CtrlUpdatePass().apply(df.copy(), args=FakeArgs(lonely_catch_all=True))
        self.assertFalse(bool((out["op"] == CTRL_UPDATE_OP).any()))


if __name__ == "__main__":
    unittest.main()
