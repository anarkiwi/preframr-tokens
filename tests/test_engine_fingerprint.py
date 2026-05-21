"""Engine-fingerprint plumbing: probe determinism + parser-side
attachment. The probe's structural correctness is exercised by the
audit scripts in ``integration_tests/profile/``; this module guards
the pieces that wire the fingerprint into the parse pipeline so
downstream macro passes can key palettes on it.
"""

import unittest
from pathlib import Path

from preframr_tokens.engine_fingerprint import (
    DEFAULT_FINGERPRINT_WRITES,
    FEATURE_DIM,
    UNKNOWN_CLUSTER,
    cluster_for_composer,
    cluster_for_path,
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


class TestClusterLookup(unittest.TestCase):
    """Composer -> cluster lookup against the committed
    ``engine_families.json`` (k=7). The audit fixed these assignments;
    if they shift, this test surfaces the change rather than letting
    the parse pipeline silently re-bucket dumps.
    """

    def test_known_composers(self):
        self.assertEqual(cluster_for_composer("Hubbard_Rob"), 7)
        self.assertEqual(cluster_for_composer("Galway_Martin"), 6)
        self.assertEqual(cluster_for_composer("Daglish_Ben"), 4)

    def test_unknown_composer(self):
        self.assertEqual(cluster_for_composer("not_a_real_composer"), UNKNOWN_CLUSTER)
        self.assertEqual(cluster_for_composer(""), UNKNOWN_CLUSTER)
        self.assertEqual(cluster_for_composer(None), UNKNOWN_CLUSTER)

    def test_path_extraction(self):
        p = "/scratch/preframr/training-dumps/Hubbard_Rob/Commando.1.dump.parquet"
        self.assertEqual(composer_from_dump_path(p), "Hubbard_Rob")
        self.assertEqual(cluster_for_path(p), 7)

    def test_fixture_path_falls_to_unknown(self):
        self.assertEqual(cluster_for_path(_FIXTURE), UNKNOWN_CLUSTER)


if __name__ == "__main__":
    unittest.main()
