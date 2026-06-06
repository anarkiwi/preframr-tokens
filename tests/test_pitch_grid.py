"""Universal multi-resolution pitch grid: lossless round-trip, on-grid purity, per-voice chorus,
and vibrato-as-modulation (design/universal_multiresolution_pitch.md)."""

import unittest

import numpy as np

from preframr_tokens.macros import pitch_grid as pg

_SUB = 32


def _melody_freqs(notes):
    return np.round(pg._ANCHOR * 2.0 ** (np.asarray(notes) / 12.0)).astype(np.int64)


class TestPitchGrid(unittest.TestCase):
    def _assert_lossless(self, freqs, sub=_SUB):
        dec = pg.decompose_voice(freqs, sub)
        rec = pg.reconstruct(dec)
        self.assertTrue(
            np.array_equal(rec.astype(np.int64), np.asarray(freqs, dtype=np.int64))
        )
        return dec

    def test_random_full_range_is_lossless(self):
        """Worst case: arbitrary 16-bit freqs round-trip exactly, LSB tail stays bounded."""
        rng = np.random.RandomState(0)
        dec = self._assert_lossless(rng.randint(1, 65536, 3000))
        self.assertLess(int(np.abs(dec["lsb"]).max()), 128)

    def test_on_grid_melody_is_pure_note(self):
        """An equal-tempered melody decomposes to a pure NOTE stream: zero tuning deviation, zero LSB."""
        rng = np.random.RandomState(1)
        dec = self._assert_lossless(_melody_freqs(rng.randint(20, 90, 256)))
        self.assertEqual(int(np.abs(dec["sub_dev"]).max()), 0)
        self.assertEqual(int(np.abs(dec["lsb"]).max()), 0)

    def test_chorus_is_per_voice_tuning_same_note(self):
        """Two voices on the same notes, one detuned +12 cents: identical NOTE stream, different
        per-voice tuning (the chorus is content, not flattened). Both lossless."""
        rng = np.random.RandomState(2)
        base = _melody_freqs(rng.randint(30, 80, 200))
        det = np.round(base * 2.0 ** (12.0 / 1200.0)).astype(np.int64)
        da = self._assert_lossless(base)
        db = self._assert_lossless(det)
        self.assertTrue(np.array_equal(da["note"], db["note"]))
        self.assertNotEqual(da["tuning"], db["tuning"])

    def test_vibrato_is_one_note_plus_modulation(self):
        """A vibrato on a held note is one NOTE plus a sub-grid deviation trajectory, lossless."""
        t = np.arange(400)
        vib = np.round(
            pg._ANCHOR * 2.0 ** (60.0 / 12.0) * 2.0 ** (20.0 * np.sin(t / 6.0) / 1200.0)
        ).astype(np.int64)
        dec = self._assert_lossless(vib)
        self.assertEqual(len({int(n) for n in dec["note"]}), 1)
        self.assertGreater(int((dec["sub_dev"] != 0).sum()), 0)

    def test_silence_round_trips(self):
        """Zero/unvoiced frames stay zero through decompose/reconstruct."""
        self._assert_lossless(np.array([0, 0, 4455, 0, 8910, 0], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
