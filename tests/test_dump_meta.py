"""DumpMeta sidecar tests: write/read round-trip, stale detection, glob_dumps filter."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preframr_tokens.dump_meta import (
    filter_dump_paths,
    meta_code_hash,
    meta_path_for,
    read_meta,
    write_meta,
)
from preframr_tokens.stfconstants import DUMP_SUFFIX, MODE_VOL_REG


def _tiny_raw_df(irq=19656, vol_changes_per_frame=2, n_frames=10):
    rows = []
    for f in range(n_frames):
        frame_irq = irq * (f + 1)
        for _ in range(vol_changes_per_frame):
            rows.append(
                {
                    "clock": frame_irq,
                    "irq": frame_irq,
                    "chipno": 0,
                    "reg": MODE_VOL_REG,
                    "val": 15,
                }
            )
        rows.append(
            {"clock": frame_irq + 1, "irq": frame_irq, "chipno": 0, "reg": 0, "val": 1}
        )
    return pd.DataFrame(rows)


def _pw_raw_df(irq=19656, pw_changes_per_frame=2, n_frames=10):
    rows = []
    for f in range(n_frames):
        frame_irq = irq * (f + 1)
        for i in range(pw_changes_per_frame):
            rows.append(
                {
                    "clock": frame_irq,
                    "irq": frame_irq,
                    "chipno": 0,
                    "reg": 2,
                    "val": i % 256,
                }
            )
        rows.append(
            {"clock": frame_irq + 1, "irq": frame_irq, "chipno": 0, "reg": 0, "val": 1}
        )
    return pd.DataFrame(rows)


class TestDumpMetaWriteRead(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"test{DUMP_SUFFIX}"
            df = _tiny_raw_df()
            df.to_parquet(dump_path, index=False)
            mp = write_meta(dump_path, df)
            self.assertTrue(mp.exists())
            meta = read_meta(dump_path)
            self.assertIsNotNone(meta)
            self.assertEqual(meta.irq, 19656)
            self.assertEqual(meta.n_frames, 10)
            self.assertEqual(meta.vol_changes_per_frame_max, 2)

    def test_stale_when_hash_differs(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"test{DUMP_SUFFIX}"
            df = _tiny_raw_df()
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            mp = meta_path_for(dump_path)
            tampered = pd.read_parquet(mp)
            tampered["meta_code_hash"] = "bogus_hash_value"
            tampered.to_parquet(mp, index=False)
            meta = read_meta(dump_path)
            self.assertTrue(meta.stale)

    def test_fresh_meta_is_not_stale(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"test{DUMP_SUFFIX}"
            df = _tiny_raw_df()
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            meta = read_meta(dump_path)
            self.assertFalse(meta.stale)

    def test_missing_meta_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"absent{DUMP_SUFFIX}"
            self.assertIsNone(read_meta(dump_path))


class TestDigiDetection(unittest.TestCase):
    def test_high_vol_density_flagged_as_digi(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"digi{DUMP_SUFFIX}"
            df = _tiny_raw_df(vol_changes_per_frame=50)
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            meta = read_meta(dump_path)
            self.assertTrue(meta.is_digi)

    def test_low_vol_density_not_digi(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"clean{DUMP_SUFFIX}"
            df = _tiny_raw_df(vol_changes_per_frame=2)
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            meta = read_meta(dump_path)
            self.assertFalse(meta.is_digi)

    def test_high_pw_density_flagged_as_digi(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"pwdigi{DUMP_SUFFIX}"
            df = _pw_raw_df(pw_changes_per_frame=50)
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            meta = read_meta(dump_path)
            self.assertTrue(meta.is_digi)
            self.assertGreaterEqual(meta.pw_changes_per_frame_max, 50)

    def test_low_pw_density_not_digi(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"pwclean{DUMP_SUFFIX}"
            df = _pw_raw_df(pw_changes_per_frame=6)
            df.to_parquet(dump_path, index=False)
            write_meta(dump_path, df)
            meta = read_meta(dump_path)
            self.assertFalse(meta.is_digi)
            self.assertLessEqual(meta.reg_write_density_max, 12.0)


class TestFilterDumpPaths(unittest.TestCase):
    def _build_corpus(self, td):
        paths = {}
        for name, vol_dens, irq in (
            ("clean_a", 2, 19656),
            ("clean_b", 3, 19656),
            ("digi", 60, 19656),
            ("ntsc_clean", 2, 16640),
        ):
            p = Path(td) / f"{name}{DUMP_SUFFIX}"
            df = _tiny_raw_df(irq=irq, vol_changes_per_frame=vol_dens)
            df.to_parquet(p, index=False)
            write_meta(p, df)
            paths[name] = str(p)
        return paths

    def test_exclude_digi(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._build_corpus(td)
            admitted, dropped = filter_dump_paths(paths.values(), exclude_digi=True)
            self.assertIn(paths["clean_a"], admitted)
            self.assertIn(paths["clean_b"], admitted)
            self.assertIn(paths["ntsc_clean"], admitted)
            self.assertNotIn(paths["digi"], admitted)
            self.assertEqual(dropped[paths["digi"]], "digi")

    def test_irq_range(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._build_corpus(td)
            admitted, dropped = filter_dump_paths(
                paths.values(), irq_range=(19000, 20000)
            )
            self.assertIn(paths["clean_a"], admitted)
            self.assertIn(paths["clean_b"], admitted)
            self.assertIn(paths["digi"], admitted)
            self.assertNotIn(paths["ntsc_clean"], admitted)
            self.assertIn("irq_out_of_range", dropped[paths["ntsc_clean"]])

    def test_missing_meta_kept_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"orphan{DUMP_SUFFIX}"
            df = _tiny_raw_df()
            df.to_parquet(p, index=False)
            admitted, dropped = filter_dump_paths([str(p)], exclude_digi=True)
            self.assertIn(str(p), admitted)
            self.assertNotIn(str(p), dropped)

    def test_missing_meta_dropped_when_require_meta(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"orphan{DUMP_SUFFIX}"
            df = _tiny_raw_df()
            df.to_parquet(p, index=False)
            admitted, dropped = filter_dump_paths([str(p)], require_meta=True)
            self.assertNotIn(str(p), admitted)
            self.assertEqual(dropped[str(p)], "no_meta")

    def test_stale_meta_kept_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._build_corpus(td)
            mp = meta_path_for(paths["clean_a"])
            tampered = pd.read_parquet(mp)
            tampered["meta_code_hash"] = "stale"
            tampered.to_parquet(mp, index=False)
            admitted, _ = filter_dump_paths([paths["clean_a"]], exclude_digi=True)
            self.assertIn(paths["clean_a"], admitted)

    def test_stale_meta_dropped_when_require_meta(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self._build_corpus(td)
            mp = meta_path_for(paths["clean_a"])
            tampered = pd.read_parquet(mp)
            tampered["meta_code_hash"] = "stale"
            tampered.to_parquet(mp, index=False)
            admitted, dropped = filter_dump_paths([paths["clean_a"]], require_meta=True)
            self.assertNotIn(paths["clean_a"], admitted)
            self.assertEqual(dropped[paths["clean_a"]], "stale_meta")


class TestMetaCodeHashStability(unittest.TestCase):
    def test_hash_is_deterministic_within_session(self):
        self.assertEqual(meta_code_hash(), meta_code_hash())

    def test_hash_length_is_16_hex_chars(self):
        h = meta_code_hash()
        self.assertEqual(len(h), 16)
        int(h, 16)


if __name__ == "__main__":
    unittest.main()
