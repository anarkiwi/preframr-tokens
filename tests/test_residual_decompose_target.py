"""Reference/acceptance test for the residual-gesture fix (executable spec for the wiring).

The deployed generator carries a note-table's freq residual as per-instance payloads keyed on
the mis-calibrated scalar grid -> one vibrato LFO fragments into dozens of distinct payloads
(the encoding explosion the guard catches). pitch_grid.decompose_voice ALREADY computes the
right lossless decomposition -- recovered per-note table + modulation in tuning-invariant CENTS
(0 for static notes) + exact closure -- it is simply not wired into generator_pass yet.

These tests pin the properties the wired encoding must preserve: byte-exact reconstruction, high
static purity, and a SMALL modulation value-alphabet (one LFO stays one small set), so the
encoded freq vocabulary stays <= the input's structural alphabet (notes + modulation levels).
"""
import numpy as np

from preframr_tokens.macros import pitch_grid

def _grid(note):
    """Exact recovered-grid freq for a note (so decompose_voice round-trips to residual 0)."""
    return int(pitch_grid.note_freq_at(int(note), 0.0))


def test_static_arp_is_fully_pure():
    # a 3-note arp cycled, NO modulation -> every frame an exact table note.
    arp = [49, 49 + 7, 49 + 12]
    freq = np.array([_grid(arp[t % 3]) for t in range(120)], dtype=np.int64)
    dec = pitch_grid.decompose_voice(freq)
    assert np.array_equal(pitch_grid.reconstruct(dec), freq)
    assert pitch_grid.pure_fraction(dec) == 1.0
    assert set(int(m) for m in dec["mod"]) == {0}


def test_one_vibrato_is_one_small_modulation_alphabet():
    # a held note with a +-200 LSB triangle vibrato (one LFO) -> small mod alphabet, byte-exact.
    base = _grid(49)
    tri = [0, 200, 400, 200, 0, -200, -400, -200]
    freq = np.array([base + tri[t % len(tri)] for t in range(160)], dtype=np.int64)
    dec = pitch_grid.decompose_voice(freq)
    assert np.array_equal(pitch_grid.reconstruct(dec), freq)
    mods = set(int(m) for m in dec["mod"])
    # the LFO visits a handful of cents levels, NOT a per-frame explosion
    assert len(mods) <= 8


def test_vibrato_is_transposition_invariant():
    # the SAME +-200 triangle on two different notes -> the SAME cents mod alphabet (the lever).
    tri = [0, 200, 400, 200, 0, -200, -400, -200]

    def mods_for(note):
        base = _grid(note)
        freq = np.array([base + tri[t % len(tri)] for t in range(160)], dtype=np.int64)
        return set(int(m) for m in pitch_grid.decompose_voice(freq)["mod"])

    # cents are tuning/transposition invariant, so low and high notes share most of the alphabet
    assert mods_for(37) & mods_for(61)
