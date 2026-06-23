"""Recover the song-data tables by IDENTITY -- read off the player's own RAM.

This is Step 0 of the white-box "recover the program, not its output" pipeline
(``design/encoding/sidtrace_program_recovery.md``).  By the player contract (HVSC
SID format: init populates the data once; play only reads it), the RAM region read
as data during play, never written during play, and never executed **is** the song
data, recoverable byte-for-byte from RAM (HARD RULE #0: genuine program data, never
fabricated, never re-fit from output).

The classification is by ACCESS TYPE off the compact SDST distill artifact (the
per-address access map the emulator accumulates), which is SMC-correct: code written
during play is excluded as SMC, not misread as data.  (The legacy raw-bus-trace
write-set/read-set partition is removed -- it misclassified under self-modifying code
and consumed a multi-GB trace; the distill ``song_data_mask`` is the production path.)
:func:`song_regions` is the SMC-correct region partition; :func:`lift_song_region_distill`
lifts the widest run verbatim from the distill RAM snapshot.
"""

import numpy as np


def regions(mask):
    """Contiguous ``(lo, hi)`` byte ranges (inclusive) of a boolean address mask."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    out = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


# --------------------------------------------------------------------------- #
# SMC-correct identity recovery from the compact SDST distill artifact.
#
# The data region is classified by access TYPE off the distill artifact's per-address
# access map: instruction-fetch (code) / data-read / data-write / SMC (= executed AND
# written during play).  Song data = read-as-data during play, never written during
# play, NEVER executed -- SMC code excluded correctly.  This consumes a few-KB
# artifact, never a raw trace.
# --------------------------------------------------------------------------- #
def song_regions(dist):
    """Contiguous ``(lo, hi)`` song-data byte ranges from a :class:`Distill`,
    classified SMC-correctly by access type (see :meth:`Distill.song_data_mask`)."""
    return regions(dist.song_data_mask())


def lift_song_data_distill(dist, lo, hi):
    """Lift the song-data bytes for the inclusive range ``[lo, hi]`` VERBATIM from
    the distill's post-init RAM SNAPSHOT -- the player's own RAM, captured once at
    the init->play boundary, never fabricated (HARD RULE #0).  Returns ``bytes``."""
    return bytes(dist.ram[lo : hi + 1])


def lift_song_region_distill(dist):
    """The widest contiguous SMC-correct song-data region from a distill artifact.

    Returns ``(region_bytes, (lo, hi))`` for the largest run of read-as-data /
    never-written / never-executed RAM the emulator snapshotted, or ``(b"", None)``
    when the artifact isolated no song data (honest fallback, not a fabricated
    region -- HARD RULE #0)."""
    runs = song_regions(dist)
    if not runs:
        return b"", None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return lift_song_data_distill(dist, lo, hi), (lo, hi)


def smc_regions(dist):
    """Contiguous ``(lo, hi)`` ranges classified as SELF-MODIFYING CODE (executed
    AND written during play).  Surfaced so the recovery can report SMC honestly
    and confirm none of it leaked into the song-data region."""
    return regions(dist.smc_mask())
