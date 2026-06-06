"""Melody-skeleton layer 2 learnability (AGENT_TASK_melody_skeleton.md §4): the key-invariant INTERVAL
token must generalize across held-out tunes better than the absolute-pitch baseline. A unit check of the
probe on a hand-built two-tune transposition, plus a corpus-sample cross-tune transfer assertion that
interval beats absolute (skipped cleanly if the corpus is unavailable)."""

import glob
import os
import unittest

from preframr_tokens.melody_audit import (
    ceiling_2gram,
    extract_sequences,
    measure,
    transfer_accuracy,
)

_HVSC = "/scratch/preframr/hvsc"


class TestTransferProbeUnit(unittest.TestCase):
    def test_interval_is_transposition_invariant_absolute_is_not(self):
        up = {"intervals": [2, 3, 1, -4], "notes": [60, 62, 65, 66, 62]}
        down = {"intervals": [2, 3, 1, -4], "notes": [48, 50, 53, 54, 50]}
        self.assertEqual(transfer_accuracy([up], [down], "intervals"), 1.0)
        self.assertEqual(transfer_accuracy([up], [down], "notes"), 0.0)

    def test_ceiling_is_in_pool_upper_bound(self):
        seqs = [{"intervals": [1, 1, 1, 1], "notes": [60, 61, 62, 63, 64]}]
        self.assertEqual(ceiling_2gram(seqs, "intervals"), 1.0)


class TestTransferCorpus(unittest.TestCase):
    def test_interval_beats_absolute_cross_tune(self):
        paths = sorted(
            glob.glob(os.path.join(_HVSC, "**", "*.dump.parquet"), recursive=True)
        )
        if not paths:
            self.skipTest("HVSC corpus unavailable")
        sample = paths[:: max(1, len(paths) // 60)][:60]
        result = measure(sample)
        if result["n_test"] < 4 or result["n_train"] < 4:
            self.skipTest("too few melodic voices in sample")
        self.assertGreater(
            result["interval_transfer"],
            result["absolute_transfer"],
            f"interval must transfer better than absolute: {result}",
        )


if __name__ == "__main__":
    unittest.main()
