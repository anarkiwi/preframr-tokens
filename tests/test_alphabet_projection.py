"""Unit tests for preframr_tokens.alphabet_projection."""

import unittest

import numpy as np
import pandas as pd

from preframr_tokens.alphabet_projection import (
    _nearest,
    build_projection_table,
    project_df,
)


class TestBuildProjectionTable(unittest.TestCase):
    def test_build_projection_table_groups_by_key(self):
        atoms = [
            (0, 21, -1, 64),
            (0, 21, -1, 32),
            (0, 21, -1, 32),
            (0, 22, -1, 100),
            (1, 21, -1, 64),
        ]
        table = build_projection_table(atoms)
        self.assertEqual(set(table.keys()), {(0, 21, -1), (0, 22, -1), (1, 21, -1)})
        np.testing.assert_array_equal(table[(0, 21, -1)], np.array([32, 64]))
        np.testing.assert_array_equal(table[(0, 22, -1)], np.array([100]))
        np.testing.assert_array_equal(table[(1, 21, -1)], np.array([64]))
        self.assertEqual(table[(0, 21, -1)].dtype, np.int64)


class TestNearest(unittest.TestCase):
    def test_nearest_picks_closest(self):
        sv = np.array([0, 32, 64, 128], dtype=np.int64)
        self.assertEqual(_nearest(sv, 20), 32)
        self.assertEqual(_nearest(sv, 48), 32)
        self.assertEqual(_nearest(sv, 200), 128)
        self.assertEqual(_nearest(sv, -5), 0)
        self.assertEqual(_nearest(sv, 64), 64)


class TestProjectDf(unittest.TestCase):
    def test_project_df_snaps_unseen_atoms(self):
        table = {(0, 21, -1): np.array([64], dtype=np.int64)}
        df = pd.DataFrame(
            {
                "op": [0, 0],
                "reg": [21, 99],
                "subreg": [-1, -1],
                "val": [63, 50],
            }
        )
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 64)
        self.assertEqual(int(out["val"].iloc[1]), 50)

    def test_project_df_preserves_non_atomic_rows(self):
        table = {(0, 21, -1): np.array([64], dtype=np.int64)}
        df = pd.DataFrame(
            {
                "op": [0, 0],
                "reg": [21, 254],
                "subreg": [-1, -1],
                "val": [63, 7],
            }
        )
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 64)
        self.assertEqual(int(out["val"].iloc[1]), 7)

    def test_aggressive_snap_no_fidelity_bound(self):
        table = {(0, 21, -1): np.array([100], dtype=np.int64)}
        df = pd.DataFrame({"op": [0], "reg": [21], "subreg": [-1], "val": [10000]})
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 100)

    def test_project_df_empty_inputs(self):
        self.assertIsNone(project_df(None, {}))
        empty = pd.DataFrame({"op": [], "reg": [], "subreg": [], "val": []})
        out = project_df(empty, {(0, 0, 0): np.array([1], dtype=np.int64)})
        self.assertEqual(len(out), 0)

    def test_project_df_missing_columns_passthrough(self):
        df = pd.DataFrame({"reg": [21], "val": [63]})
        out = project_df(df, {(0, 21, -1): np.array([64], dtype=np.int64)})
        pd.testing.assert_frame_equal(out, df)

    def test_project_df_empty_table_passthrough(self):
        df = pd.DataFrame({"op": [0], "reg": [21], "subreg": [-1], "val": [63]})
        out = project_df(df, {})
        pd.testing.assert_frame_equal(out, df)

    def test_tie_breaks_to_lower(self):
        table = {(0, 21, -1): np.array([0, 32, 64, 128], dtype=np.int64)}
        df = pd.DataFrame({"op": [0], "reg": [21], "subreg": [-1], "val": [48]})
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 32)

    def test_vectorised_matches_scalar(self):
        rng = np.random.default_rng(0)
        train_vals = sorted(set(rng.integers(0, 256, size=20).tolist()))
        table = {(0, 21, -1): np.array(train_vals, dtype=np.int64)}
        eval_vals = rng.integers(-10, 300, size=64).tolist()
        df = pd.DataFrame(
            {
                "op": [0] * len(eval_vals),
                "reg": [21] * len(eval_vals),
                "subreg": [-1] * len(eval_vals),
                "val": eval_vals,
            }
        )
        out = project_df(df, table)
        expected = [_nearest(table[(0, 21, -1)], v) for v in eval_vals]
        self.assertEqual(out["val"].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
