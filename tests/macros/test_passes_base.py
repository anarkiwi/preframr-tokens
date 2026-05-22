"""Direct unit tests for ``passes_base`` helpers + ``requires_state`` decorator."""

import unittest

import pandas as pd

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
    requires_state,
)
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, MODEL_PDTYPE, SET_OP


def _df(rows):
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


class TestMacroPassBase(unittest.TestCase):
    def test_apply_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            MacroPass().apply(_df([{"reg": 0, "val": 0, "diff": 0}]))


class TestRequiresState(unittest.TestCase):
    def test_state_none_returns_df_unchanged(self):
        class _Probe:
            @requires_state
            def apply(self, df, state, args):
                self.called = True
                return df

        probe = _Probe()
        df = _df([{"reg": 0, "val": 1, "op": SET_OP, "diff": 32}])
        out = probe.apply(df)
        self.assertFalse(hasattr(probe, "called"))
        self.assertIn("subreg", out.columns)

    def test_state_built_method_called(self):
        class _Probe:
            @requires_state
            def apply(self, df, state, args):
                self.called = True
                return df

        probe = _Probe()
        df = _df(
            [
                {"reg": FRAME_REG, "val": 0, "op": SET_OP, "diff": 19000},
                {"reg": 0, "val": 5, "op": SET_OP, "diff": 32},
            ]
        )
        probe.apply(df)
        self.assertTrue(probe.called)


class TestFrameIndex(unittest.TestCase):
    def test_cumulative_index_increments_at_markers(self):
        df = _df(
            [
                {"reg": FRAME_REG, "val": 0},
                {"reg": 0, "val": 1},
                {"reg": DELAY_REG, "val": 1},
                {"reg": 0, "val": 2},
            ]
        )
        idx = _frame_index(df).tolist()
        self.assertEqual(idx, [1, 1, 2, 2])


class TestEnsureSubreg(unittest.TestCase):
    def test_adds_subreg_when_missing(self):
        df = _df([{"reg": 0, "val": 5}])
        out = _ensure_subreg(df)
        self.assertIn("subreg", out.columns)
        self.assertTrue((out["subreg"] == -1).all())

    def test_passes_through_when_present(self):
        df = _df([{"reg": 0, "val": 5, "subreg": 2}])
        out = _ensure_subreg(df)
        self.assertEqual(int(out["subreg"].iloc[0]), 2)


class TestSpliceRows(unittest.TestCase):
    def _base_df(self):
        return _df(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": SET_OP},
                {"reg": 0, "val": 2, "diff": 32, "op": SET_OP},
                {"reg": 0, "val": 3, "diff": 32, "op": SET_OP},
            ]
        )

    def test_no_new_rows_returns_input(self):
        df = self._base_df()
        out = _splice_rows(df, drop_idx=[], new_rows=[])
        self.assertIs(out, df)

    def test_drops_rows_and_inserts_new(self):
        df = self._base_df()
        out = _splice_rows(
            df,
            drop_idx=[1],
            new_rows=[{"reg": 99, "val": 42, "diff": 32, "op": SET_OP, "__pos": 1}],
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(int(out.iloc[1]["reg"]), 99)

    def test_fills_description_when_absent(self):
        df = _df(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": SET_OP, "description": 5},
                {"reg": 0, "val": 2, "diff": 32, "op": SET_OP, "description": 5},
            ]
        )
        out = _splice_rows(
            df,
            drop_idx=[0],
            new_rows=[{"reg": 99, "val": 42, "diff": 32, "op": SET_OP, "__pos": 0}],
        )
        new_row = out[out["reg"] == 99].iloc[0]
        self.assertEqual(int(new_row["description"]), 0)

    def test_fills_irq_when_absent(self):
        df = _df(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": SET_OP, "irq": 19656},
                {"reg": 0, "val": 2, "diff": 32, "op": SET_OP, "irq": 19656},
            ]
        )
        out = _splice_rows(
            df,
            drop_idx=[0],
            new_rows=[{"reg": 99, "val": 42, "diff": 32, "op": SET_OP, "__pos": 0}],
        )
        new_row = out[out["reg"] == 99].iloc[0]
        self.assertEqual(int(new_row["irq"]), 19656)

    def test_preserves_attrs(self):
        df = self._base_df()
        df.attrs["sentinel"] = "kept"
        out = _splice_rows(
            df,
            drop_idx=[],
            new_rows=[{"reg": 99, "val": 42, "diff": 32, "op": SET_OP, "__pos": 1}],
        )
        self.assertEqual(out.attrs.get("sentinel"), "kept")


if __name__ == "__main__":
    unittest.main()
