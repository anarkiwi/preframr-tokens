"""Smoke tests for ``preframr_tokens.blocks``. Comprehensive coverage of the block-iteration helpers lives in the main `preframr` repo's RegDataset tests (where the full corpus + parser fixtures are available); here we verify the module imports cleanly and the trivial-input contracts."""

import os
import tempfile
import unittest

from preframr_tokens.blocks import glob_dumps, reg_widths_path


class TestRegWidthsPath(unittest.TestCase):
    def test_sidecar_suffix(self):
        self.assertEqual(
            reg_widths_path("/tmp/df-map.csv"), "/tmp/df-map_reg_widths.json"
        )

    def test_no_extension(self):
        self.assertEqual(reg_widths_path("/tmp/df-map"), "/tmp/df-map_reg_widths.json")


class TestGlobDumps(unittest.TestCase):
    def test_empty_glob_returns_empty(self):
        out = glob_dumps(
            "/nonexistent/path/*.dump.parquet", max_files=10, require_pq=False
        )
        self.assertEqual(out, [])

    def test_basic_glob(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.dump.parquet", "b.dump.parquet", "c.txt"):
                open(os.path.join(d, name), "w").close()
            out = glob_dumps(f"{d}/*.dump.parquet", max_files=10, require_pq=False)
            self.assertEqual(len(out), 2)
            for path in out:
                self.assertTrue(path.endswith(".dump.parquet"))


if __name__ == "__main__":
    unittest.main()
