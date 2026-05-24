"""Tests for ReleaseUpdatePass + ReleaseUpdateDecoder (spec #5)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.release_update_pass import ReleaseUpdatePass
from preframr_tokens.stfconstants import FRAME_REG, RELEASE_UPDATE_OP, SET_OP


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
    flags.setdefault("release_update_pass", True)
    return ReleaseUpdatePass().apply(df.copy(), args=FakeArgs(**flags))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestReleaseUpdatePass(unittest.TestCase):
    def test_disabled_is_noop(self):
        rows = [_frame(), _r(6, 0xF0), _frame(), _frame(), _r(6, 0xA0)]
        df = pd.DataFrame(rows)
        out = ReleaseUpdatePass().apply(
            df.copy(), args=FakeArgs(release_update_pass=False)
        )
        self.assertTrue(out.equals(df))

    def test_isolated_sr_becomes_release_update(self):
        rows = [_frame(), _r(6, 0xF0)]
        for _ in range(3):
            rows.append(_frame())
        rows += [_r(6, 0xA0)]
        for _ in range(3):
            rows.append(_frame())
        rows += [_r(0, 1)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertTrue(bool((out["op"] == RELEASE_UPDATE_OP).any()))
        self.assertEqual(len(out[(out["reg"] == 6) & (out["op"] == SET_OP)]), 0)
        self.assertEqual(_decoded_reg(df, 6), _decoded_reg(out, 6))

    def test_isolated_ad_becomes_release_update(self):
        rows = [_frame(), _r(5, 0x11)]
        for _ in range(3):
            rows.append(_frame())
        rows += [_r(0, 9)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertTrue(bool((out["op"] == RELEASE_UPDATE_OP).any()))
        self.assertEqual(_decoded_reg(df, 5), _decoded_reg(out, 5))

    def test_adjacent_sr_writes_not_converted(self):
        rows = [_frame(), _r(6, 0xF0), _frame(), _r(6, 0xA0), _frame(), _r(6, 0x80)]
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == RELEASE_UPDATE_OP).any()))
        self.assertEqual(_decoded_reg(df, 6), _decoded_reg(out, 6))

    def test_catch_all_converts_adjacent_sr_writes(self):
        rows = [_frame(), _r(6, 0xF0), _frame(), _r(6, 0xA0), _frame(), _r(6, 0x80)]
        df = pd.DataFrame(rows)
        out = _apply(df, lonely_catch_all=True)
        self.assertEqual(len(out[(out["reg"] == 6) & (out["op"] == SET_OP)]), 0)
        self.assertTrue(bool((out["op"] == RELEASE_UPDATE_OP).any()))
        self.assertEqual(_decoded_reg(df, 6), _decoded_reg(out, 6))


if __name__ == "__main__":
    unittest.main()
