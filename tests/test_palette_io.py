"""Tests for ``preframr_tokens.palette_io``."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from preframr_tokens.palette_io import (
    _palettes_sidecar_path,
    dump_palettes_attrs,
    load_palettes_attrs,
)


class TestPaletteIo(unittest.TestCase):
    def _parquet_path(self, tmpdir: str) -> str:
        return os.path.join(tmpdir, "song.0.parquet")

    def test_sidecar_path_suffix(self):
        self.assertEqual(
            _palettes_sidecar_path("/x/y.parquet"), "/x/y.parquet.palettes.json"
        )

    def test_round_trip_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            attrs = {
                "engine_fingerprint": [0.1, 0.2, 0.3, 0.4],
                "engine_fp_cluster": 7,
            }
            dump_palettes_attrs(attrs, pq)
            loaded = load_palettes_attrs(pq)
            self.assertEqual(loaded["engine_fingerprint"], [0.1, 0.2, 0.3, 0.4])
            self.assertEqual(loaded["engine_fp_cluster"], 7)

    def test_round_trip_fingerprint_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            attrs = {"engine_fingerprint": [1.0, 2.0]}
            dump_palettes_attrs(attrs, pq)
            loaded = load_palettes_attrs(pq)
            self.assertEqual(loaded, {"engine_fingerprint": [1.0, 2.0]})
            self.assertNotIn("engine_fp_cluster", loaded)

    def test_round_trip_cluster_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            attrs = {"engine_fp_cluster": 3}
            dump_palettes_attrs(attrs, pq)
            loaded = load_palettes_attrs(pq)
            self.assertEqual(loaded, {"engine_fp_cluster": 3})

    def test_missing_sidecar_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            self.assertEqual(load_palettes_attrs(pq), {})

    def test_empty_attrs_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            dump_palettes_attrs({}, pq)
            self.assertFalse(os.path.exists(_palettes_sidecar_path(pq)))

    def test_none_attrs_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            dump_palettes_attrs(None, pq)
            self.assertFalse(os.path.exists(_palettes_sidecar_path(pq)))

    def test_attrs_without_known_keys_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            dump_palettes_attrs({"something_else": 42}, pq)
            self.assertFalse(os.path.exists(_palettes_sidecar_path(pq)))

    def test_fingerprint_serialised_as_floats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = self._parquet_path(tmpdir)
            dump_palettes_attrs({"engine_fingerprint": [1, 2, 3]}, pq)
            with open(_palettes_sidecar_path(pq)) as f:
                raw = json.load(f)
            self.assertEqual(raw["engine_fingerprint"], [1, 2, 3])
            loaded = load_palettes_attrs(pq)
            self.assertEqual(
                [type(x) for x in loaded["engine_fingerprint"]],
                [float, float, float],
            )


if __name__ == "__main__":
    unittest.main()
