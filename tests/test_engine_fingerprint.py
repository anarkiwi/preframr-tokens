"""Engine fingerprint: computation determinism + ClusterTable contract.

ClusterTable is library glue around a caller-provided JSON; the
specific corpus-snapshot assignments (Hubbard_Rob -> 7, etc.) are
test-data concerns that live in the consumer repo. Here we only
test the table's shape + the empty-table / unknown-composer paths
+ the table built from a tiny fixture JSON.
"""

import json
import tempfile
import unittest
from pathlib import Path

from preframr_tokens.engine_fingerprint import (
    DEFAULT_FINGERPRINT_WRITES,
    ENGINE_FP_K,
    FEATURE_DIM,
    UNKNOWN_CLUSTER,
    ClusterTable,
    composer_from_dump_path,
    compute_fingerprint,
)

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


if __name__ == "__main__":
    unittest.main()
