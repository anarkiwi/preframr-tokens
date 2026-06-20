"""Canonical, driver-invariant 12-TET pitch axis (Part B).

The note's IDENTITY is its 12-TET semitone index on a fixed A440 reference,
computed from the onset frequency the driver actually renders -- NOT a note-table
offset, NOT a frequency. Every driver maps onto the same grid:

    onset SID Fn  --snap-->  n = round(12 * log2(Fn / FN_A440))

so concert C-4 is one token whether it came from a Hubbard register write at
onset or a GoatTracker note-byte resolved through GoatTracker's freq table.

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


def hubbard_table_fn(static_img, note):
    """Hubbard onset Fn for a note-table INDEX: ``lo | hi<<8`` little-endian.

    ``static_img`` IS the driver's note-freq table (256 bytes = 128 entries);
    this is the exact onset frequency the Monty-class driver writes for ``note``.
    """
    a = (note & 0xFF) * 2
    return static_img[a] | (static_img[a + 1] << 8)


def hubbard_grid_bijection(static_img):
    """Build the canonical-grid <-> note-index bijection for a Hubbard table.

    Walks the longest ascending clean-12-TET run from index 0 (the real note
    table; the bytes past it are unrelated driver RAM that alias pitch classes
    at slightly different Fn). Over that run grid index <-> note index is an
    exact bijection, so a note resolvable through it serializes as ONE canonical
    grid token identical to GoatTracker's for the same concert pitch. Returns
    ``(grid_to_index, index_to_grid)`` dicts; aliased tail notes (rare, e.g.
    Monty's note 104) are absent and ride a literal-index escape so the
    re-coordinate stays byte-exact and lossless.
    """
    grid_to_index = {}
    index_to_grid = {}
    prev_grid = None
    for note in range(128):
        fn = hubbard_table_fn(static_img, note)
        if fn <= 0:
            break
        g = fn_to_grid(fn)
        if prev_grid is not None and g - prev_grid != 1:
            break  # end of the clean ascending ET run; rest is aliasing RAM
        grid_to_index[g] = note
        index_to_grid[note] = g
        prev_grid = g
    return grid_to_index, index_to_grid
