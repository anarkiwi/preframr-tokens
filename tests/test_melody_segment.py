"""Melody-skeleton layer 2 note segmentation: the pass-1 sustained-pitch-change detector unioned with
gate-on retriggers. A held-gate legato line that steps pitch under one sustained gate must segment by
its intrinsic level changes (one onset per sustained note, not over-segmented by vibrato jitter), and a
re-struck same-pitch note must onset on its gate edge."""

import unittest

from preframr_tokens.macros.melody_segment import note_onsets, pass1_origins
from preframr_tokens.macros.generator_fit import recon


def _freqs(notes, ref=0.0):
    """Per-frame 16-bit freqs for a per-frame note list (None -> silent)."""
    return [None if n is None else recon(int(n), ref) for n in notes]


class TestPass1Origins(unittest.TestCase):
    def test_flat_line_has_no_origin(self):
        semis = [60.0] * 40
        self.assertEqual(pass1_origins(_np(semis), 1.0, 5, 3), [])

    def test_vibrato_jitter_is_not_an_origin(self):
        semis = [60.0 + (0.4 if i % 2 else -0.4) for i in range(40)]
        self.assertEqual(pass1_origins(_np(semis), 1.0, 5, 3), [])

    def test_sustained_step_is_one_origin(self):
        semis = [60.0] * 20 + [64.0] * 20
        got = pass1_origins(_np(semis), 1.0, 5, 3)
        self.assertEqual(len(got), 1)
        self.assertTrue(18 <= got[0] <= 22)


class TestNoteOnsets(unittest.TestCase):
    def test_held_gate_legato_segments_by_level_not_overseg(self):
        notes = [60] * 12 + [62] * 12 + [64] * 12 + [66] * 12
        onsets = note_onsets(_freqs(notes), gate_on=[0])
        self.assertEqual(len(onsets), 4)
        self.assertEqual(onsets[0], 0)

    def test_restruck_same_pitch_onsets_on_gate_edge(self):
        notes = [60] * 36
        onsets = note_onsets(_freqs(notes), gate_on=[0, 12, 24])
        self.assertEqual(onsets, [0, 12, 24])

    def test_silent_voice_has_no_onsets(self):
        onsets = note_onsets([None] * 40, gate_on=[])
        self.assertEqual(onsets, [])


def _np(semis):
    import numpy as np

    return np.asarray(semis, dtype=float)


if __name__ == "__main__":
    unittest.main()
