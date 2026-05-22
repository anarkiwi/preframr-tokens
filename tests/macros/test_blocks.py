"""Direct unit tests for ``preframr_tokens.macros.blocks`` helpers."""

import unittest

import pandas as pd

from preframr_tokens.macros.blocks import (
    expand_to_literal_form,
    iter_self_contained_row_blocks,
    self_contain_slice,
)
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, MODEL_PDTYPE, SET_OP


def _row(reg, val, op=SET_OP, diff=32, subreg=-1):
    return {
        "reg": int(reg),
        "val": int(val),
        "op": int(op),
        "diff": int(diff),
        "subreg": int(subreg),
        "description": 0,
    }


def _df(rows):
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


class TestExpandToLiteralForm(unittest.TestCase):
    def test_set_only_df_passes_through(self):
        df = _df(
            [
                _row(FRAME_REG, 0, diff=19000),
                _row(0, 5),
                _row(FRAME_REG, 0, diff=19000),
                _row(0, 7),
            ]
        )
        literal = expand_to_literal_form(df)
        self.assertIn("op", literal.columns)
        self.assertIn("subreg", literal.columns)
        self.assertEqual(literal["op"].iloc[1], int(SET_OP))

    def test_missing_subreg_column_filled(self):
        df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "op": SET_OP, "diff": 19000},
                {"reg": 0, "val": 3, "op": SET_OP, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        literal = expand_to_literal_form(df)
        self.assertIn("subreg", literal.columns)
        self.assertTrue((literal["subreg"] == -1).all())

    def test_preserves_attrs_on_input(self):
        df = _df([_row(FRAME_REG, 0, diff=19000), _row(0, 5)])
        df.attrs["sentinel"] = "preserved"
        expand_to_literal_form(df)
        self.assertEqual(df.attrs.get("sentinel"), "preserved")


class TestSelfContainSlice(unittest.TestCase):
    def setUp(self):
        self.df = _df(
            [
                _row(FRAME_REG, 0, diff=19000),
                _row(0, 5),
                _row(FRAME_REG, 0, diff=19000),
                _row(0, 7),
                _row(FRAME_REG, 0, diff=19000),
                _row(0, 9),
            ]
        )

    def test_slice_middle_frame(self):
        out = self_contain_slice(self.df, slice_lo_frame=1, slice_hi_frame=2)
        self.assertGreater(len(out), 0)
        self.assertEqual(int(out["reg"].iloc[0]), FRAME_REG)

    def test_slice_lo_past_end_returns_empty(self):
        out = self_contain_slice(self.df, slice_lo_frame=99, slice_hi_frame=100)
        self.assertEqual(len(out), 0)

    def test_slice_to_end_when_hi_past_end(self):
        out = self_contain_slice(self.df, slice_lo_frame=0, slice_hi_frame=99)
        self.assertGreater(len(out), 0)


class TestIterSelfContainedRowBlocksNoOpColumn(unittest.TestCase):
    """The ``if "op" not in df.columns`` early-exit branch — operates on raw frame slices without invoking the macro pipeline."""

    def test_yields_frame_slices(self):
        df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 5, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 7, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 9, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        blocks = list(iter_self_contained_row_blocks(df, frames_per_block=2))
        self.assertGreater(len(blocks), 0)
        for block in blocks:
            self.assertIn("reg", block.columns)
            self.assertNotIn("op", block.columns)

    def test_no_markers_yields_whole_df(self):
        df = pd.DataFrame(
            [{"reg": 0, "val": 5, "diff": 32}],
            dtype=MODEL_PDTYPE,
        )
        blocks = list(iter_self_contained_row_blocks(df, frames_per_block=2))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0]), 1)

    def test_stride_smaller_than_block_overlaps(self):
        df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 5, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 7, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 9, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 11, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        no_stride = list(iter_self_contained_row_blocks(df, frames_per_block=2))
        with_stride = list(
            iter_self_contained_row_blocks(df, frames_per_block=2, stride=1)
        )
        self.assertGreater(len(with_stride), len(no_stride))


if __name__ == "__main__":
    unittest.main()
