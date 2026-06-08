"""Recovered-table pitch model (raw-Hz, post-cents-retirement): note_index<->note_freq round-trip, the
exact per-voice note table (recover_table), the raw-Hz two-layer split freq = note_freq(note) + delta the
MDL gesture parse rides, shared note index under per-voice detune, and held-note-plus-modulation.
"""

import unittest

import numpy as np

from preframr_tokens.macros import pitch_grid as pg


def _melody_freqs(notes, tuning=0.0):
    return pg.note_freq(np.asarray(notes), tuning)


class TestPitchGrid(unittest.TestCase):
    def test_note_index_inverts_note_freq(self):
        """note_index(note_freq(n)) == n across the playable range (before note_freq saturates the
        16-bit word at the extreme low/high ends) -- the grid the note layer indexes."""
        for n in range(-48, 95):
            self.assertEqual(int(pg.note_index(pg.note_freq_at(n, 0.0), 0.0)), n)

    def test_two_layer_split_is_byte_exact(self):
        """Any 16-bit freq series splits into note_index + raw-Hz delta and reconstructs exactly:
        freq == (note_freq(note_index(freq)) + (freq - note_freq(note_index(freq)))), the MDL freq cover.
        """
        rng = np.random.RandomState(0)
        f = rng.randint(1, 65536, 3000).astype(np.int64)
        note = pg.note_index(f, 0.0)
        base = pg.note_freq(note, 0.0)
        delta = f - base
        self.assertTrue(np.array_equal((base + delta).astype(np.int64), f))
        self.assertLess(int(np.abs(delta).max()), 32768)

    def test_on_grid_melody_has_zero_delta(self):
        """An equal-tempered melody sits exactly on the recovered table: every voiced frame has delta 0."""
        rng = np.random.RandomState(1)
        f = _melody_freqs(rng.randint(20, 90, 256))
        note = pg.note_index(f, 0.0)
        self.assertTrue(np.array_equal(pg.note_freq(note, 0.0), f))

    def test_recover_table_is_exact_modal_entry(self):
        """recover_table maps each played note to the exact 16-bit freq word it was played at."""
        notes = [49, 53, 56, 60]
        f = _melody_freqs(np.repeat(notes, 8))
        table = pg.recover_table(f, 0.0)
        for n in notes:
            self.assertEqual(
                int(table[int(pg.note_index(pg.note_freq_at(n, 0.0), 0.0))]),
                int(pg.note_freq_at(n, 0.0)),
            )

    def test_chorus_shares_note_index(self):
        """Two voices on the same notes, one detuned +12 cents, recover the SAME note-index stream
        (the detune lives in the per-voice tuning + table, not the index)."""
        rng = np.random.RandomState(2)
        base = _melody_freqs(rng.randint(34, 78, 200))
        det = np.round(base * 2.0 ** (12.0 / 1200.0)).astype(np.int64)
        ta = pg.voice_tuning(base)
        tb = pg.voice_tuning(det)
        self.assertTrue(np.array_equal(pg.note_index(base, ta), pg.note_index(det, tb)))

    def test_vibrato_is_one_held_note_plus_delta(self):
        """A vibrato around a held note recovers ONE note index (the wobble is a nonzero freq-delta),
        and the raw delta stays well within the signed-16-bit field the gesture anchor carries.
        """
        t = np.arange(400)
        vib = np.round(
            pg.note_freq_at(60, 0.0) * 2.0 ** (20.0 * np.sin(t / 6.0) / 1200.0)
        ).astype(np.int64)
        note = pg.note_index(vib, 0.0)
        self.assertEqual(len(set(int(n) for n in note)), 1)
        delta = vib - pg.note_freq(note, 0.0)
        self.assertGreater(int((delta != 0).sum()), 0)
        self.assertLess(int(np.abs(delta).max()), 4096)

    def test_detuned_scale_recovers_consecutive_indices(self):
        """A Galway-style +44c-detuned ET scale recovers consecutive note indices (the per-voice tuning
        fit handles the near-half-semitone detune naive rounding mis-assigns)."""
        notes = np.arange(40, 76)
        freqs = np.round(
            pg.note_freq_at(0, 0.0) * 2.0 ** ((notes + 44.0 / 100.0) / 12.0)
        ).astype(np.int64)
        f = np.repeat(freqs, 6)
        tuning = pg.voice_tuning(f)
        idx = sorted(set(int(n) for n in pg.note_index(f, tuning)))
        self.assertEqual(idx, list(range(idx[0], idx[0] + len(notes))))


if __name__ == "__main__":
    unittest.main()
