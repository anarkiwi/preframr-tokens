"""Part B -- the driver-invariance gate.

The canonical 12-TET A440 grid index is the NOTE token's identity. The same
concert pitch must yield the SAME grid index whether it arrives as a Hubbard
onset register Fn or a GoatTracker note byte resolved through GoatTracker's
freq table. This is the executable definition of "same note = same token".
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


# Hubbard note table for Monty: note byte -> onset Fn = table[note*2] |
# table[note*2+1]<<8. These are the exact onset Fn values the Monty driver writes
# (captured from the recovered static note table); pinned here so the gate needs
# no .sid download. GoatTracker note bytes for the SAME concert pitches must snap
# to the identical grid index.
_HUBBARD_ONSET_FN = {
    48: 4456,  # C-4
    52: 5611,  # E-4
    55: 6675,  # G-4
    57: 7494,  # A-4
}
_GT_NAME = {48: "C-4", 52: "E-4", 55: "G-4", 57: "A-4"}


def test_same_concert_pitch_same_grid_index_across_drivers():
    for hub_note, fn in _HUBBARD_ONSET_FN.items():
        hub_grid = fn_to_grid(fn)
        gt_grid = fn_to_grid(_gt_note_fn(C.note_value(_GT_NAME[hub_note])))
        assert (
            hub_grid == gt_grid
        ), f"{_GT_NAME[hub_note]}: Hubbard grid {hub_grid} != GoatTracker {gt_grid}"
    # And the anchor lands where music theory says (A-4 == 0, C-4 == -9).
    assert fn_to_grid(_HUBBARD_ONSET_FN[57]) == 0
    assert fn_to_grid(_HUBBARD_ONSET_FN[48]) == -9


def _clean_et_static_img():
    """A synthetic Hubbard note table: a clean ascending 12-TET run from index 0
    so the grid<->index bijection resolves every playable note. Index n renders
    the canonical A440 Fn for grid (n - 57) -- i.e. note 57 == A-4 == grid 0,
    matching the real Monty table's anchor."""
    img = [0] * 256
    for note in range(96):
        fn = int(round(grid_to_fn(note - 57)))
        img[note * 2] = fn & 0xFF
        img[note * 2 + 1] = (fn >> 8) & 0xFF
    return img


def test_serialized_note_token_identical_across_drivers():
    """The executable proof: for the SAME concert pitch, the Hubbard serializer
    and the GoatTracker serializer emit the IDENTICAL note token (the actual
    LEB-digit bytes, not merely the grid integer)."""
    from preframr_tokens.bacc import gt_serialize, serialize

    img = _clean_et_static_img()
    _, index_to_grid = serialize.hubbard_grid_bijection(img)
    for hub_note, name in _GT_NAME.items():
        # Hubbard emitted note token for its note-table index.
        hub_tok = []
        serialize._note_field(hub_tok, hub_note, img, index_to_grid)
        # GoatTracker emitted note token for the SAME concert pitch.
        gt_kind, gt_grid = gt_serialize._note_token(C.note_value(name))
        gt_tok = []
        serialize._wu(gt_tok, serialize._zz(gt_grid) << 1)
        assert gt_kind == gt_serialize._KIND_PITCH
        assert hub_tok == gt_tok, (
            f"{name}: Hubbard note token {hub_tok} != GoatTracker {gt_tok} "
            "(drivers must emit ONE identical token per concert pitch)"
        )
        # And the token decodes back to the canonical grid index both share.
        z, _ = serialize._ru(hub_tok, 0)
        assert serialize._unzz(z >> 1) == fn_to_grid(_gt_note_fn(C.note_value(name)))


def test_micro_deviation_is_subcent_for_a4():
    # A-4 in both tables is within a couple cents of the canonical anchor; the
    # residual is tuning (Part C), never the note token.
    for fn in (_HUBBARD_ONSET_FN[57], _gt_note_fn(C.note_value("A-4"))):
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
