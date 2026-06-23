"""Canonical, driver-invariant 12-TET pitch axis (Part B).

The note's IDENTITY is its 12-TET semitone index on a fixed A440 reference,
computed from the onset frequency the driver actually renders -- NOT a note-table
offset, NOT a frequency. Every driver maps onto the same grid:

    onset SID Fn  --snap-->  n = round(12 * log2(Fn / FN_A440))

so concert C-4 is one token whether it came from a generic onset register write
or a GoatTracker note-byte resolved through GoatTracker's freq table.

The SID phase-accumulator frequency register Fn relates to the audible tone by
``f_Hz = Fn * PHI / 2**24`` (PHI = the chip clock, PAL here). Inverting at A440
gives FN_A440 = 440 * 2**24 / PHI, the reference grid anchor. The grid index is
clock-relative only through that single anchor, which cancels in the round() --
the result is the pure musical semitone offset from A440, identical for any
driver that targets the same concert pitch.
"""

import math

# PAL C64 clock (phi2). The SID frequency register Fn maps to Hz as
# f = Fn * PAL_PHI / 2**24; documented here so the A440 anchor is reproducible.
PAL_PHI = 985248
SID_ACC_BITS = 24

# Fn register value that renders exactly concert A440 on the PAL clock. This is
# the canonical grid anchor: n = 0 at A440. (~7492.5 for PAL.)
FN_A440 = 440.0 * (1 << SID_ACC_BITS) / PAL_PHI


def fn_to_hz(fn, phi=PAL_PHI):
    """SID frequency register value -> audible Hz on the given clock."""
    return fn * phi / (1 << SID_ACC_BITS)


def fn_to_grid(fn):
    """Snap a SID onset Fn to its canonical 12-TET semitone index (A440, integer).

    This is the NOTE identity: driver-invariant by construction (only Fn enters).
    Fn must be > 0; a zero/silent onset has no pitch and is rejected by callers.
    """
    return round(12.0 * math.log2(fn / FN_A440))


def grid_to_fn(n):
    """Canonical grid frequency for semitone index n (the A440 12-TET Fn)."""
    return FN_A440 * (2.0 ** (n / 12.0))
