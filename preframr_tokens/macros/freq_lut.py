"""Leaf module: the fixed MIDI-note -> 16-bit SID freq word table (``LUT``) plus the two
pure note<->freq helpers and the arp-period bound that survive the skeleton/codebook stack.
No imports from the macro stack, so any leaf may depend on it without an import cycle.
"""

__all__ = [
    "LUT",
    "midi_to_fn",
    "fn_to_note_resid",
    "ARP_MAX_PERIOD",
    "CLOCK_RATE",
]

import math

CLOCK_RATE = 985248
MIDI_LO, MIDI_HI = 16, 112
ARP_MAX_PERIOD = 16


def midi_to_fn(m):
    """MIDI note -> 16-bit SID freq word, clamped to 0..0xFFFF."""
    return max(
        0,
        min(
            0xFFFF, int(round(440.0 * 2 ** ((m - 69) / 12.0) * 16777216.0 / CLOCK_RATE))
        ),
    )


LUT = [midi_to_fn(m) for m in range(128)]


def fn_to_note_resid(fn):
    """16-bit freq -> (nearest MIDI semitone, residual in cents). None if silent/out of range."""
    if fn < 8:
        return None
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return None
    mf = 69 + 12 * math.log2(hz / 440.0)
    note = int(round(mf))
    if not MIDI_LO <= note <= MIDI_HI:
        return None
    return note, (mf - note) * 100.0
