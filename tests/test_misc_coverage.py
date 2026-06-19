"""Small coverage-tightening tests for branches missed by the topical suites."""

import tempfile
import unittest
from pathlib import Path

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


class TestFreqLutNoteResid(unittest.TestCase):
    def test_silent_returns_none(self):
        from preframr_tokens.macros.freq_lut import fn_to_note_resid

        self.assertIsNone(fn_to_note_resid(0))
        self.assertIsNone(fn_to_note_resid(10))

    def test_below_midi_lo_returns_none(self):
        from preframr_tokens.macros.freq_lut import fn_to_note_resid, midi_to_fn

        self.assertIsNone(fn_to_note_resid(midi_to_fn(8)))

    def test_in_range_returns_note_and_small_resid(self):
        from preframr_tokens.macros.freq_lut import fn_to_note_resid, LUT

        note, cents = fn_to_note_resid(LUT[60])
        self.assertEqual(note, 60)
        self.assertLess(abs(cents), 1.0)


class TestVocabIdTier(unittest.TestCase):
    def test_no_tokens_is_content_tier(self):
        from preframr_tokens.tier_classify import vocab_id_tier, CONTENT_TIER

        self.assertEqual(vocab_id_tier(0, None, None), CONTENT_TIER)

    def test_id_beyond_base_falls_through_to_content(self):
        from types import SimpleNamespace

        import pandas as pd

        from preframr_tokens.tier_classify import vocab_id_tier, CONTENT_TIER

        tokens = pd.DataFrame([{"op": 0, "reg": 1, "subreg": -1, "val": 5}])
        rt = SimpleNamespace(tkmodel=None)
        self.assertEqual(vocab_id_tier(99, rt, tokens), CONTENT_TIER)


class TestFlagRegistryResolve(unittest.TestCase):
    def test_resolve_expands_requires_and_detects_conflicts(self):
        from preframr_tokens.macros import flag_registry as fr

        orig_req, orig_con = fr.FLAG_REQUIRES, fr.FLAG_CONFLICTS
        fr.FLAG_REQUIRES = {"a": frozenset({"b"})}
        fr.FLAG_CONFLICTS = {"b": frozenset({"c"})}
        try:
            self.assertEqual(fr.resolve_flags({"a"}), {"a", "b"})
            self.assertTrue(fr.valid_combo({"a", "b"}))
            self.assertFalse(fr.valid_combo({"b", "c"}))
            with self.assertRaises(ValueError):
                fr.resolve_flags({"b", "c"})
            with self.assertRaises(ValueError):
                fr.resolve_flags({"c", "b"})
        finally:
            fr.FLAG_REQUIRES, fr.FLAG_CONFLICTS = orig_req, orig_con


if __name__ == "__main__":
    unittest.main()
