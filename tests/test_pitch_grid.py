"""Recovered-table pitch model: lossless round-trip, on-grid purity, per-voice chorus (shared note
index, distinct per-voice tables), and vibrato-as-modulation (design/universal_multiresolution_pitch.md).
"""

import unittest

import numpy as np

from preframr_tokens.macros import pitch_grid as pg


def _melody_freqs(notes):
    return np.round(pg._ANCHOR * 2.0 ** (np.asarray(notes) / 12.0)).astype(np.int64)


class TestPitchGrid(unittest.TestCase):
    def _assert_lossless(self, freqs):
        dec = pg.decompose_voice(freqs)
        rec = pg.reconstruct(dec)
        self.assertTrue(
            np.array_equal(rec.astype(np.int64), np.asarray(freqs, dtype=np.int64))
        )
        return dec

    def test_random_full_range_is_lossless(self):
        """Arbitrary 16-bit freqs round-trip exactly under the recovered table (lossless for any table)."""
        rng = np.random.RandomState(0)
        self._assert_lossless(rng.randint(1, 65536, 3000))

    def test_on_grid_melody_is_pure(self):
        """An equal-tempered melody is 100% pure notes: every frame is an exact table entry."""
        rng = np.random.RandomState(1)
        dec = self._assert_lossless(_melody_freqs(rng.randint(20, 90, 256)))
        self.assertEqual(pg.pure_fraction(dec), 1.0)

    def test_chorus_shared_index_distinct_tables(self):
        """Two voices on the same notes, one detuned +12c: identical NOTE index stream, DIFFERENT
        per-voice tables, both 100% pure (the chorus is in the table values, not a residual).
        """
        rng = np.random.RandomState(2)
        base = _melody_freqs(rng.randint(30, 80, 200))
        det = np.round(base * 2.0 ** (12.0 / 1200.0)).astype(np.int64)
        da = self._assert_lossless(base)
        db = self._assert_lossless(det)
        self.assertTrue(np.array_equal(da["note"], db["note"]))
        self.assertNotEqual(da["table"], db["table"])
        self.assertEqual(pg.pure_fraction(da), 1.0)
        self.assertEqual(pg.pure_fraction(db), 1.0)

    def test_vibrato_is_one_note_plus_modulation(self):
        """A vibrato on a held note is one NOTE index plus a nonzero CENTS trajectory, lossless."""
        t = np.arange(400)
        vib = np.round(
            pg._ANCHOR * 2.0 ** (60.0 / 12.0) * 2.0 ** (20.0 * np.sin(t / 6.0) / 1200.0)
        ).astype(np.int64)
        dec = self._assert_lossless(vib)
        self.assertEqual(len({int(n) for n in dec["note"][dec["voiced"]]}), 1)
        self.assertGreater(int((dec["mod"] != 0).sum()), 0)
        self.assertLess(pg.pure_fraction(dec), 1.0)

    def test_modulation_is_tuning_invariant(self):
        """The SAME +/-Xc vibrato gesture at two DIFFERENT tunings yields the SAME cents trajectory
        (model learns "X cents", not absolute freq) -- the transferability requirement.
        """
        t = np.arange(200)
        gesture = np.concatenate(
            [np.ones(120), 2.0 ** (20.0 * np.sin(t / 5.0) / 1200.0)]
        )
        low = np.round(pg._ANCHOR * 2.0 ** (36.0 / 12.0) * gesture).astype(np.int64)
        high = np.round(pg._ANCHOR * 2.0 ** (72.0 / 12.0) * gesture).astype(np.int64)
        dl = pg.decompose_voice(low)
        dh = pg.decompose_voice(high)
        self.assertLessEqual(int(np.abs(dl["mod"] - dh["mod"]).max()), 2)
        raw_l = low - pg._table_vec(dl["table"], dl["note"])
        raw_h = high - pg._table_vec(dh["table"], dh["note"])
        self.assertGreater(int(np.abs(raw_l - raw_h).max()), 20)

    def test_small_table_and_silence(self):
        """The recovered table is the distinct notes used; unvoiced frames round-trip as zero."""
        dec = self._assert_lossless(
            np.array([0, 0, 4455, 4455, 0, 8910, 8910, 0], dtype=np.int64)
        )
        self.assertLessEqual(len(dec["table"]), 2)


if __name__ == "__main__":
    unittest.main()
