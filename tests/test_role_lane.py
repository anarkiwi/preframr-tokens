"""Layer-3 coarse role tracker: rank voices bass/mid/lead by sustained pitch per block, control-aware and
waveform-agnostic. Pins the causal ordering the byte-exact reorder uses (accompaniment before melody).
"""

import unittest

import numpy as np

from preframr_tokens.role_lane import block_roles, lead_changes, voice_note_series
from preframr_tokens.macros.generator_fit import recon
from preframr_tokens.stfconstants import GEN_FREQ_REGS, INSTR_OFF_CTRL


def _state(per_voice_notes, gated=(True, True, True), frames=None):
    """Build a (frames, 25) register_state with each voice held at a constant note (or a per-frame list),
    gate bit set per ``gated``. Waveform nibble left 0 -- the tracker must not read it.
    """
    cols = 25
    lengths = [
        len(n) if isinstance(n, list) else (frames or 8) for n in per_voice_notes
    ]
    n = max(lengths)
    st = np.zeros((n, cols), dtype=np.int64)
    for v, b in enumerate(GEN_FREQ_REGS):
        notes = per_voice_notes[v] if v < len(per_voice_notes) else None
        is_gated = gated[v] if v < len(gated) else False
        for i in range(n):
            note = notes[i] if isinstance(notes, list) else notes
            if note is not None:
                f = recon(int(note), 0.0)
                st[i, b] = f & 0xFF
                st[i, b + 1] = (f >> 8) & 0xFF
            if is_gated:
                st[i, b + INSTR_OFF_CTRL] = 0x41
    return st


class TestRoleLane(unittest.TestCase):
    def test_pitch_rank_assigns_bass_mid_lead(self):
        st = _state([72, 48, 60])
        roles = block_roles(st, block_frames=256)[0]
        self.assertEqual(roles[1], "bass")
        self.assertEqual(roles[2], "mid")
        self.assertEqual(roles[0], "lead")

    def test_silent_voice_excluded(self):
        st = _state([67, None, 55], gated=(True, False, True))
        roles = block_roles(st, block_frames=256)[0]
        self.assertNotIn(1, roles)
        self.assertEqual(roles[2], "bass")
        self.assertEqual(roles[0], "lead")

    def test_waveform_nibble_ignored(self):
        st = _state([72, 48, 60])
        base = block_roles(st, block_frames=256)[0]
        for b in GEN_FREQ_REGS:
            st[:, b + INSTR_OFF_CTRL] |= 0x80
        self.assertEqual(block_roles(st, block_frames=256)[0], base)

    def test_lead_hops_counted_per_block(self):
        v0 = [72] * 8 + [40] * 8
        v1 = [48] * 8 + [70] * 8
        st = _state([v0, v1, [60] * 16])
        roles = block_roles(st, block_frames=8)
        self.assertEqual(len(roles), 2)
        self.assertEqual(roles[0][0], "lead")
        self.assertEqual(roles[1][1], "lead")
        self.assertEqual(lead_changes(roles), 1)

    def test_voice_note_series_silent_when_ungated(self):
        st = _state([60], gated=(False,), frames=8)
        series = voice_note_series(st)
        self.assertTrue(all(x is None for x in series[0]))


if __name__ == "__main__":
    unittest.main()
