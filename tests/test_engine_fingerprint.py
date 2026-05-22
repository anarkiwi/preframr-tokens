"""Engine fingerprint: computation determinism + ClusterTable shape (empty-table, unknown-composer, tiny-fixture-JSON paths). Corpus-specific cluster assignments live in the consumer repo."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from preframr_tokens.engine_fingerprint import (
    CTRL_2GRAM_DIM,
    CTRL_3GRAM_DIM,
    DEFAULT_FINGERPRINT_WRITES,
    DELTA_BUCKETS,
    ENGINE_FP_K,
    FEATURE_DIM,
    REG_DENSITY_DIM,
    SLICE_REG_DENSITY,
    UNKNOWN_CLUSTER,
    ClusterTable,
    _ctrl_ngrams,
    _ctrl_state,
    _delta_histogram,
    _filter_touch_ratio,
    _read_writes,
    _reg_density,
    composer_from_dump_path,
    compute_fingerprint,
)
from preframr_tokens.stfconstants import FC_LO_REG, FILTER_REG, MAX_REG

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "integration_tests"
    / "fixtures"
    / "hvsc_authored"
    / "Commando.1.1500.dump.parquet"
)


@unittest.skipUnless(_FIXTURE.exists(), f"fixture missing: {_FIXTURE}")
class TestEngineFingerprint(unittest.TestCase):
    def test_shape_and_finite(self):
        vec = compute_fingerprint(_FIXTURE)
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertTrue((vec == vec).all())

    def test_deterministic(self):
        v1 = compute_fingerprint(_FIXTURE)
        v2 = compute_fingerprint(_FIXTURE)
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)
        self.assertEqual(v1.tobytes(), v2.tobytes())

    def test_truncation_changes_vector(self):
        v_full = compute_fingerprint(_FIXTURE, n_writes=DEFAULT_FINGERPRINT_WRITES)
        v_small = compute_fingerprint(_FIXTURE, n_writes=200)
        self.assertIsNotNone(v_full)
        self.assertIsNotNone(v_small)
        self.assertNotEqual(v_full.tobytes(), v_small.tobytes())


class TestComposerExtraction(unittest.TestCase):
    def test_composer_from_dump_path(self):
        p = "/scratch/preframr/training-dumps/Hubbard_Rob/Commando.1.dump.parquet"
        self.assertEqual(composer_from_dump_path(p), "Hubbard_Rob")

    def test_composer_from_dump_path_no_parent(self):
        self.assertIsNone(composer_from_dump_path("/foo"))


class TestClusterTableEmpty(unittest.TestCase):
    def test_no_path_means_empty(self):
        table = ClusterTable()
        self.assertEqual(len(table), 0)
        self.assertFalse(bool(table))

    def test_empty_returns_unknown(self):
        table = ClusterTable()
        self.assertEqual(table.cluster_for_composer("Hubbard_Rob"), UNKNOWN_CLUSTER)
        self.assertEqual(table.cluster_for_composer(""), UNKNOWN_CLUSTER)
        self.assertEqual(table.cluster_for_composer(None), UNKNOWN_CLUSTER)

    def test_empty_path_lookup_returns_unknown(self):
        table = ClusterTable()
        p = "/scratch/preframr/training-dumps/Hubbard_Rob/Commando.1.dump.parquet"
        self.assertEqual(table.cluster_for_path(p), UNKNOWN_CLUSTER)


class TestClusterTableFromFixture(unittest.TestCase):
    def _write_fixture(self, data):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_loads_from_fixture(self):
        path = self._write_fixture(
            {
                "composer_stats": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ],
                "cluster_assignments": {
                    str(ENGINE_FP_K): [3, 5],
                },
            }
        )
        table = ClusterTable(path)
        self.assertEqual(len(table), 2)
        self.assertTrue(bool(table))
        self.assertEqual(table.cluster_for_composer("Alice"), 3)
        self.assertEqual(table.cluster_for_composer("Bob"), 5)
        self.assertEqual(table.cluster_for_composer("Carol"), UNKNOWN_CLUSTER)

    def test_path_lookup_via_fixture(self):
        path = self._write_fixture(
            {
                "composer_stats": [{"name": "Hubbard_Rob"}],
                "cluster_assignments": {str(ENGINE_FP_K): [7]},
            }
        )
        table = ClusterTable(path)
        dump = "/scratch/preframr/training-dumps/Hubbard_Rob/Commando.1.dump.parquet"
        self.assertEqual(table.cluster_for_path(dump), 7)

    def test_missing_key_returns_empty(self):
        path = self._write_fixture({"composer_stats": []})
        table = ClusterTable(path)
        self.assertEqual(len(table), 0)

    def test_size_mismatch_returns_empty(self):
        path = self._write_fixture(
            {
                "composer_stats": [{"name": "Alice"}, {"name": "Bob"}],
                "cluster_assignments": {str(ENGINE_FP_K): [1]},
            }
        )
        table = ClusterTable(path)
        self.assertEqual(len(table), 0)

    def test_corrupt_json_returns_empty(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write("{not json")
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        table = ClusterTable(Path(tmp.name))
        self.assertEqual(len(table), 0)

    def test_missing_file_returns_empty(self):
        table = ClusterTable("/nonexistent/path.json")
        self.assertEqual(len(table), 0)


class TestCtrlState(unittest.TestCase):
    def test_zero_waveform_nibble_yields_zero_wave_idx(self):
        self.assertEqual(_ctrl_state(0x00), 0)
        self.assertEqual(_ctrl_state(0x01), 1)

    def test_waveform_bit_index_uses_lowest_set_bit(self):
        self.assertEqual(_ctrl_state(0x10) >> 1, 0)
        self.assertEqual(_ctrl_state(0x20) >> 1, 1)
        self.assertEqual(_ctrl_state(0x40) >> 1, 2)
        self.assertEqual(_ctrl_state(0x80) >> 1, 3)
        self.assertEqual(_ctrl_state(0x30) >> 1, 0)
        self.assertEqual(_ctrl_state(0xF0) >> 1, 0)

    def test_gate_bit_in_low_position(self):
        self.assertEqual(_ctrl_state(0x11) & 1, 1)
        self.assertEqual(_ctrl_state(0x10) & 1, 0)


class TestRegDensity(unittest.TestCase):
    def test_uniform_distribution_normalised(self):
        regs = np.array([0, 1, 2, 3], dtype=np.int64)
        dens = _reg_density(regs)
        self.assertEqual(dens.shape, (REG_DENSITY_DIM,))
        self.assertAlmostEqual(dens.sum(), 1.0)

    def test_invalid_regs_dropped(self):
        regs = np.array([-1, 999, 5], dtype=np.int64)
        dens = _reg_density(regs)
        self.assertAlmostEqual(dens.sum(), 1.0)
        self.assertAlmostEqual(float(dens[5]), 1.0)

    def test_empty_input_zero_vector(self):
        dens = _reg_density(np.array([], dtype=np.int64))
        self.assertEqual(dens.shape, (REG_DENSITY_DIM,))
        self.assertEqual(float(dens.sum()), 0.0)


class TestDeltaHistogram(unittest.TestCase):
    def test_fewer_than_two_clocks_returns_zero(self):
        h = _delta_histogram(np.array([100], dtype=np.int64))
        self.assertEqual(h.shape, (DELTA_BUCKETS,))
        self.assertEqual(float(h.sum()), 0.0)

    def test_negative_deltas_clamped_to_zero(self):
        clocks = np.array([100, 50], dtype=np.int64)
        h = _delta_histogram(clocks)
        self.assertAlmostEqual(h.sum(), 1.0)

    def test_normal_deltas_normalised(self):
        clocks = np.array([0, 10, 100, 1000], dtype=np.int64)
        h = _delta_histogram(clocks)
        self.assertAlmostEqual(h.sum(), 1.0)


class TestFilterTouchRatio(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(_filter_touch_ratio(np.array([], dtype=np.int64)), 0.0)

    def test_no_valid_regs_returns_zero(self):
        self.assertEqual(_filter_touch_ratio(np.array([-1, 999], dtype=np.int64)), 0.0)

    def test_all_filter_writes_returns_one(self):
        regs = np.array([FC_LO_REG, FC_LO_REG + 1, FILTER_REG], dtype=np.int64)
        self.assertAlmostEqual(_filter_touch_ratio(regs), 1.0)

    def test_partial_filter_writes_returns_fraction(self):
        regs = np.array([FC_LO_REG, 0, 0, 0], dtype=np.int64)
        self.assertAlmostEqual(_filter_touch_ratio(regs), 0.25)


class TestCtrlNgrams(unittest.TestCase):
    def test_bigram_and_trigram_dims(self):
        regs = np.array([4, 4, 4, 4], dtype=np.int64)
        vals = np.array([0x10, 0x11, 0x10, 0x11], dtype=np.int64)
        bg, tg = _ctrl_ngrams(regs, vals)
        self.assertEqual(bg.shape, (CTRL_2GRAM_DIM,))
        self.assertEqual(tg.shape, (CTRL_3GRAM_DIM,))
        self.assertAlmostEqual(bg.sum(), 1.0)
        self.assertAlmostEqual(tg.sum(), 1.0)

    def test_no_ctrl_writes_returns_zero_vectors(self):
        regs = np.array([0, 1, 2], dtype=np.int64)
        vals = np.array([0, 0, 0], dtype=np.int64)
        bg, tg = _ctrl_ngrams(regs, vals)
        self.assertEqual(float(bg.sum()), 0.0)
        self.assertEqual(float(tg.sum()), 0.0)

    def test_single_ctrl_write_no_ngrams(self):
        regs = np.array([4], dtype=np.int64)
        vals = np.array([0x11], dtype=np.int64)
        bg, tg = _ctrl_ngrams(regs, vals)
        self.assertEqual(float(bg.sum()), 0.0)
        self.assertEqual(float(tg.sum()), 0.0)


def _make_synthetic_parquet(path, n_rows=10):
    """Build a tiny parquet matching the schema _read_writes expects."""
    clock = np.arange(n_rows, dtype=np.int64) * 100
    irq = np.full(n_rows, 19656, dtype=np.int64)
    reg = np.arange(n_rows, dtype=np.int64) % MAX_REG
    val = np.full(n_rows, 0x11, dtype=np.int64)
    table = pa.table({"clock": clock, "irq": irq, "reg": reg, "val": val})
    pq.write_table(table, path)


class TestReadWritesAndComputeFingerprintSynthetic(unittest.TestCase):
    def _tmpfile(self, suffix=".parquet"):
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_read_writes_missing_file(self):
        self.assertIsNone(_read_writes(Path("/nonexistent.parquet"), 100))

    def test_read_writes_returns_shape(self):
        p = self._tmpfile()
        _make_synthetic_parquet(p, n_rows=20)
        arr = _read_writes(p, 10)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (10, 4))

    def test_read_writes_zero_request_returns_none(self):
        p = self._tmpfile()
        _make_synthetic_parquet(p, n_rows=5)
        self.assertIsNone(_read_writes(p, 0))

    def test_compute_fingerprint_full_pipeline(self):
        p = self._tmpfile()
        _make_synthetic_parquet(p, n_rows=200)
        vec = compute_fingerprint(p, n_writes=DEFAULT_FINGERPRINT_WRITES)
        self.assertIsNotNone(vec)
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(vec).all())
        self.assertAlmostEqual(float(vec[SLICE_REG_DENSITY].sum()), 1.0)

    def test_compute_fingerprint_too_few_writes_returns_none(self):
        p = self._tmpfile()
        _make_synthetic_parquet(p, n_rows=1)
        self.assertIsNone(compute_fingerprint(p, n_writes=DEFAULT_FINGERPRINT_WRITES))

    def test_compute_fingerprint_missing_file_returns_none(self):
        self.assertIsNone(compute_fingerprint(Path("/nonexistent.parquet")))


if __name__ == "__main__":
    unittest.main()
