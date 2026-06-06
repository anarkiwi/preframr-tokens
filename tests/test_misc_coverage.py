"""Small coverage-tightening tests for branches missed by the topical suites."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preframr_tokens.audit_primitives import detect_tail_cycle
from preframr_tokens.dump_meta import DumpMeta, meta_path_for, read_meta
from preframr_tokens.stfconstants import DUMP_SUFFIX


class TestDetectTailCycleSkipsImpossiblePeriods(unittest.TestCase):
    def test_period_skipped_when_repeats_dont_fit(self):
        """Force the ``continue`` branch where ``tail_window < period * min_repeats``."""
        tokens = [1] * 4 + [1, 2, 1, 2]
        out = detect_tail_cycle(tokens, tail_window=8, max_period=4, min_repeats=3)
        self.assertIsNone(out)


class TestDumpMetaProperties(unittest.TestCase):
    def test_per_frame_max_properties(self):
        meta = DumpMeta(
            dump_path="/tmp/x.dump.parquet",
            fields={
                "vol_changes_per_frame_max": 7,
                "ctrl_changes_per_frame_max": 11,
                "freq_writes_per_frame_max": 13,
            },
        )
        self.assertEqual(meta.vol_changes_per_frame_max, 7)
        self.assertEqual(meta.ctrl_changes_per_frame_max, 11)
        self.assertEqual(meta.freq_writes_per_frame_max, 13)

    def test_per_frame_max_defaults_to_zero(self):
        meta = DumpMeta(dump_path="/tmp/x.dump.parquet", fields={})
        self.assertEqual(meta.vol_changes_per_frame_max, 0)
        self.assertEqual(meta.ctrl_changes_per_frame_max, 0)


class TestReadMetaCorruptParquet(unittest.TestCase):
    def test_corrupt_parquet_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"x{DUMP_SUFFIX}"
            mp = meta_path_for(dump_path)
            mp.write_bytes(b"not a parquet")
            self.assertIsNone(read_meta(dump_path))


class TestSnapGroup(unittest.TestCase):
    def test_empty_sorted_vals_passes_through(self):
        from preframr_tokens.alphabet_projection import _snap_group
        import numpy as np

        vals = np.array([1, 2, 3], dtype=np.int64)
        out = _snap_group(np.array([], dtype=np.int64), vals)
        np.testing.assert_array_equal(out, vals)

    def test_empty_vals_passes_through(self):
        from preframr_tokens.alphabet_projection import _snap_group
        import numpy as np

        vals = np.array([], dtype=np.int64)
        out = _snap_group(np.array([1, 2, 3], dtype=np.int64), vals)
        np.testing.assert_array_equal(out, vals)


if __name__ == "__main__":
    unittest.main()
