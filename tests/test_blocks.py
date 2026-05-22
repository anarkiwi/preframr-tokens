"""Smoke tests for ``preframr_tokens.blocks``. Comprehensive coverage of the block-iteration helpers lives in the main `preframr` repo's RegDataset tests (where the full corpus + parser fixtures are available); here we verify the module imports cleanly and the trivial-input contracts."""

import inspect
import os
import tempfile
import unittest

from preframr_tokens.blocks import (
    glob_dumps,
    reg_widths_path,
    self_contained_prompt_df,
)


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


class TestSelfContainedPromptDfApi(unittest.TestCase):
    """``self_contained_prompt_df`` has no callers inside
    ``preframr_tokens`` but is imported by ``preframr/train/regdataset.py``.
    Behavioural coverage lives in that repo's RegDataset tests (which
    have the full loader / dataset / tokenizer fixtures). Here we only
    pin the signature so an accidental rename or removal breaks loudly.
    """

    def test_signature(self):
        sig = inspect.signature(self_contained_prompt_df)
        self.assertEqual(
            list(sig.parameters),
            ["loader", "dataset", "seq", "seq_meta", "start", "prompt_seq_len", "irq"],
        )


if __name__ == "__main__":
    unittest.main()
