"""Melody-skeleton layer 2: MELODY_INTERVAL atom round-trip + the zig-zag interval codec. A note-onset
re-keyed as (FIRST|interval, residual, delta, len) decodes to the exact same per-frame freqs as the raw
absolute-pitch HOLD/ACCUM atom -- the running interval sum reproducing the absolute note, the residual
carried exact (the generator's losslessness preserved under re-keying)."""

import types
import unittest

from preframr_tokens.macros.generator_fit import note_of, recon, unzig, zig
from preframr_tokens.macros.decoders import MelodyIntervalDecoder
from preframr_tokens.macros.state import DecodeState, FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    MELODY_INTERVAL_SUBREG_DELTA_HI,
    MELODY_INTERVAL_SUBREG_DELTA_LO,
    MELODY_INTERVAL_SUBREG_FIRST,
    MELODY_INTERVAL_SUBREG_INTERVAL_HI,
    MELODY_INTERVAL_SUBREG_INTERVAL_LO,
    MELODY_INTERVAL_SUBREG_LEN,
    MELODY_INTERVAL_SUBREG_RESID_HI,
    MELODY_INTERVAL_SUBREG_RESID_LO,
    MELODY_INTERVAL_SUBREG_VOICE,
)


def _atom_rows(voice, first, token, resid, delta, length):
    """The 9 subreg rows of one MELODY_INTERVAL atom (encoder side), VOICE..LEN in dispatch order."""
    reg = FREQ_REGS_BY_VOICE[voice]
    resid &= 0xFFFF
    delta &= 0xFFFF
    token &= 0xFFFF
    fields = [
        (MELODY_INTERVAL_SUBREG_VOICE, voice),
        (MELODY_INTERVAL_SUBREG_FIRST, 1 if first else 0),
        (MELODY_INTERVAL_SUBREG_INTERVAL_HI, (token >> 8) & 0xFF),
        (MELODY_INTERVAL_SUBREG_INTERVAL_LO, token & 0xFF),
        (MELODY_INTERVAL_SUBREG_RESID_HI, (resid >> 8) & 0xFF),
        (MELODY_INTERVAL_SUBREG_RESID_LO, resid & 0xFF),
        (MELODY_INTERVAL_SUBREG_DELTA_HI, (delta >> 8) & 0xFF),
        (MELODY_INTERVAL_SUBREG_DELTA_LO, delta & 0xFF),
        (MELODY_INTERVAL_SUBREG_LEN, length),
    ]
    return [
        types.SimpleNamespace(reg=reg, val=v, subreg=s, diff=1, description=0, Index=0)
        for s, v in fields
    ]


def _decode(onsets, ref=0.0):
    """Feed a sequence of melody onsets through MelodyIntervalDecoder, return per-voice queued freqs."""
    dec = MelodyIntervalDecoder()
    state = DecodeState(frame_diff=1)
    state.gen_ref = ref
    for rows in onsets:
        for row in rows:
            dec.expand(row, state)
    return {v: list(state.pending_set_writes[FREQ_REGS_BY_VOICE[v]]) for v in range(3)}


class TestZigZag(unittest.TestCase):
    def test_inverse_over_range(self):
        for n in range(-200, 201):
            self.assertEqual(unzig(zig(n)), n)

    def test_low_cardinality_near_zero(self):
        self.assertEqual([zig(n) for n in (0, -1, 1, -2, 2)], [0, 1, 2, 3, 4])


class TestMelodyIntervalRoundTrip(unittest.TestCase):
    def test_first_then_intervals_reconstructs_absolute(self):
        ref = 0.0
        notes = [60, 62, 60, 67]
        deltas = [0, 1, 0, -2]
        resids = [5, -3, 0, 7]
        lengths = [3, 2, 1, 4]
        onsets = []
        prev = None
        for note, delta, resid, length in zip(notes, deltas, resids, lengths):
            if prev is None:
                onsets.append(_atom_rows(0, True, note, resid, delta, length))
            else:
                onsets.append(
                    _atom_rows(0, False, zig(note - prev), resid, delta, length)
                )
            prev = note
        got = _decode(onsets, ref)[0]
        expected = []
        for note, delta, resid, length in zip(notes, deltas, resids, lengths):
            start = (recon(note, ref) + resid) & 0xFFFF
            expected.extend((start + k * delta) & 0xFFFF for k in range(length))
        self.assertEqual(got, expected)

    def test_residual_makes_onset_bit_exact(self):
        ref = 0.0
        freqs = [4321, 5678, 4444]
        onsets = []
        prev_note = None
        for f in freqs:
            note = note_of(f, ref)
            resid = f - recon(note, ref)
            if prev_note is None:
                onsets.append(_atom_rows(1, True, note, resid, 0, 1))
            else:
                onsets.append(_atom_rows(1, False, zig(note - prev_note), resid, 0, 1))
            prev_note = note
        got = _decode(onsets, ref)[1]
        self.assertEqual(got, freqs)

    def test_per_voice_state_is_independent(self):
        ref = 0.0
        onsets = [
            _atom_rows(0, True, 60, 0, 0, 1),
            _atom_rows(2, True, 40, 0, 0, 1),
            _atom_rows(0, False, zig(4), 0, 0, 1),
            _atom_rows(2, False, zig(-3), 0, 0, 1),
        ]
        got = _decode(onsets, ref)
        self.assertEqual(got[0], [recon(60, ref), recon(64, ref)])
        self.assertEqual(got[2], [recon(40, ref), recon(37, ref)])


if __name__ == "__main__":
    unittest.main()
