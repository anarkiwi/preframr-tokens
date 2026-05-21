"""Regression test: PresetPass must run inside PASSES so re-encoding
during iter_self_contained_row_blocks collapses SET reg=21/2 rows back
to PRESET ops. Missing it exploded the alphabet > tkvocab."""

from __future__ import annotations

import unittest

import pandas as pd

from preframr_tokens import macros
from preframr_tokens.macros.blocks import iter_self_contained_row_blocks
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.stfconstants import (
    FC_PRESET_OP,
    FRAME_REG,
    MODEL_PDTYPE,
    PWM_PRESET_OP,
    SET_OP,
)


class FakeArgs:
    seq_len = 64
    cents = 50
    instrument_pass = False
    hard_restart_pass = False
    legato_pass_c2 = False
    legato_pass_c3 = False
    legato_pass_c4 = False
    legato_pass_c7 = False
    fuzzy_loop_pass = False
    preset_pass = True


def _frame_row(diff=19656):
    return {
        "reg": FRAME_REG,
        "val": 0,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _set_row(reg, val, diff=32):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _make_literal_df():
    rows = []
    for fc in (0, 256, 8192, 16384, 32768, 65280):
        rows.extend(
            [
                _frame_row(),
                _set_row(21, fc),
            ]
        )
    for pw in (0, 128, 1024, 2048, 3072, 3968):
        for reg in (2, 9, 16):
            rows.extend(
                [
                    _frame_row(),
                    _set_row(reg, pw),
                ]
            )
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


class TestPresetPassInPasses(unittest.TestCase):
    def test_preset_pass_is_in_passes_list(self):
        self.assertTrue(
            any(isinstance(p, PresetPass) for p in macros.PASSES),
            "PresetPass must be in macros.PASSES",
        )

    def test_literal_set_reg21_collapses_via_iter_blocks(self):
        df = _make_literal_df()
        args = FakeArgs()
        collected_reg21_set = 0
        collected_reg2_set = 0
        collected_preset = 0
        for block in iter_self_contained_row_blocks(df, 8, args=args):
            if block.empty:
                continue
            m21_set = (
                (block["op"] == SET_OP) & (block["reg"] == 21) & (block["subreg"] == -1)
            )
            m2_set = (
                (block["op"] == SET_OP) & (block["reg"] == 2) & (block["subreg"] == -1)
            )
            collected_reg21_set += int(m21_set.sum())
            collected_reg2_set += int(m2_set.sum())
            collected_preset += int(
                block["op"].isin([PWM_PRESET_OP, FC_PRESET_OP]).sum()
            )
        self.assertEqual(
            collected_reg21_set,
            0,
            "literal SET reg=21 rows must be collapsed to FC_PRESET_OP "
            "inside iter_self_contained_row_blocks; got "
            f"{collected_reg21_set} surviving SET reg=21 rows",
        )
        self.assertEqual(
            collected_reg2_set,
            0,
            "literal SET reg=2 rows must be collapsed to PWM_PRESET_OP; got "
            f"{collected_reg2_set} surviving SET reg=2 rows",
        )
        self.assertGreater(
            collected_preset,
            0,
            "expected PRESET_OP rows after PresetPass runs in run_passes",
        )


if __name__ == "__main__":
    unittest.main()
