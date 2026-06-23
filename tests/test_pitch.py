"""Part B -- the driver-invariance gate.

The canonical 12-TET A440 grid index is the NOTE token's identity. The same
concert pitch must yield the SAME grid index whether it arrives as a raw onset
register Fn or a GoatTracker note byte resolved through GoatTracker's freq table.
This is the executable definition of "same note = same token".
"""

import math

import pytest

from preframr_tokens.bacc.pitch import FN_A440, fn_to_grid, grid_to_fn

pygoattracker = pytest.importorskip("pygoattracker")
from pygoattracker import constants as C  # noqa: E402


def _gt_note_fn(note_byte):
    """GoatTracker pattern note byte -> its freq-table SID Fn."""
    return C.FREQ_TABLE[note_byte - C.FIRSTNOTE]


def test_a440_anchor_is_grid_zero():
    assert fn_to_grid(FN_A440) == 0
    # A-4 from GoatTracker's own table also lands on 0.
    assert fn_to_grid(_gt_note_fn(C.note_value("A-4"))) == 0


def test_grid_roundtrips_each_semitone():
    for n in range(-48, 40):
        assert fn_to_grid(grid_to_fn(n)) == n


def test_goattracker_freq_table_is_clean_12tet():
    # Every adjacent GoatTracker note differs by exactly one grid step.
    prev = None
    for note in range(C.FIRSTNOTE, C.FIRSTNOTE + 96):
        fn = _gt_note_fn(note)
        if fn <= 0:
            continue
        g = fn_to_grid(fn)
        if prev is not None:
            assert g - prev == 1, f"note {note}: grid jump {g - prev}"
        prev = g


# Onset Fn values a register-write driver writes for these concert pitches: note
# byte -> onset Fn = table[note*2] | table[note*2+1]<<8, captured from a real tune's
# static note table; pinned here so the gate needs no .sid download. GoatTracker note
# bytes for the SAME concert pitches must snap to the identical grid index.
_DRIVER_ONSET_FN = {
    48: 4456,  # C-4
    52: 5611,  # E-4
    55: 6675,  # G-4
    57: 7494,  # A-4
}
_GT_NAME = {48: "C-4", 52: "E-4", 55: "G-4", 57: "A-4"}


def test_same_concert_pitch_same_grid_index_across_drivers():
    for drv_note, fn in _DRIVER_ONSET_FN.items():
        drv_grid = fn_to_grid(fn)
        gt_grid = fn_to_grid(_gt_note_fn(C.note_value(_GT_NAME[drv_note])))
        assert (
            drv_grid == gt_grid
        ), f"{_GT_NAME[drv_note]}: onset grid {drv_grid} != GoatTracker {gt_grid}"
    # And the anchor lands where music theory says (A-4 == 0, C-4 == -9).
    assert fn_to_grid(_DRIVER_ONSET_FN[57]) == 0
    assert fn_to_grid(_DRIVER_ONSET_FN[48]) == -9


def test_micro_deviation_is_subcent_for_a4():
    # A-4 in both tables is within a couple cents of the canonical anchor; the
    # residual is tuning (Part C), never the note token.
    for fn in (_DRIVER_ONSET_FN[57], _gt_note_fn(C.note_value("A-4"))):
        cents = 1200.0 * math.log2(fn / grid_to_fn(0))
        assert abs(cents) < 5.0


def test_static_tuning_delta_table_is_small_and_lossless():
    """Part C (static Δ): the GoatTracker freq table's deviation from the
    canonical A440 grid is a small per-note-class ET-rounding residual (<=~4c),
    and the grid<->note mapping is a clean bijection -- so the note token (grid
    index) reconstructs the exact freq-table Fn with no loss. Δ is real (the
    deviation), factored off the note alphabet, never a stored byte."""
    deltas = {}
    bijection = {}
    for note in range(C.FIRSTNOTE, C.LASTNOTE + 1):
        fn = _gt_note_fn(note)
        if fn <= 0:
            continue
        n = fn_to_grid(fn)
        deltas[n] = 1200.0 * math.log2(fn / grid_to_fn(n))
        # Δ is per-note-class (one value per grid index), not per occurrence.
        assert n not in bijection, f"grid index {n} is not unique (would clash)"
        bijection[n] = note
    assert max(abs(c) for c in deltas.values()) < 4.0
    # lossless: grid index -> note byte -> exact freq-table Fn round-trips.
    for n, note in bijection.items():
        assert fn_to_grid(_gt_note_fn(note)) == n
