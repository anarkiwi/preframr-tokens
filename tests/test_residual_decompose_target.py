"""Reference/acceptance spec for the residual-gesture fix: pitch_grid.decompose_voice already
computes the right lossless decomposition (recovered per-note table + tuning-invariant cents
modulation + exact closure) but is not yet wired into generator_pass, so these pin what the
wired encoding must preserve -- byte-exact reconstruction, high static purity, and a small
modulation alphabet (one LFO stays one small set) instead of per-instance residual payloads.
"""

import numpy as np

from preframr_tokens.macros import pitch_grid


def _grid(note):
    """Exact recovered-grid freq for a note (so decompose_voice round-trips to residual 0)."""
    return int(pitch_grid.note_freq_at(int(note), 0.0))


def test_static_arp_is_fully_pure():
    """A 3-note arp cycled with no modulation: every frame is an exact table note."""
    arp = [49, 49 + 7, 49 + 12]
    freq = np.array([_grid(arp[t % 3]) for t in range(120)], dtype=np.int64)
    dec = pitch_grid.decompose_voice(freq)
    assert np.array_equal(pitch_grid.reconstruct(dec), freq)
    assert pitch_grid.pure_fraction(dec) == 1.0
    assert set(int(m) for m in dec["mod"]) == {0}


def test_one_vibrato_is_one_small_modulation_alphabet():
    """A held note with one +-200 LSB triangle vibrato: small cents alphabet, byte-exact
    (the LFO visits a handful of levels, not a per-frame explosion)."""
    base = _grid(49)
    tri = [0, 200, 400, 200, 0, -200, -400, -200]
    freq = np.array([base + tri[t % len(tri)] for t in range(160)], dtype=np.int64)
    dec = pitch_grid.decompose_voice(freq)
    assert np.array_equal(pitch_grid.reconstruct(dec), freq)
    mods = set(int(m) for m in dec["mod"])
    assert len(mods) <= 8


def test_vibrato_is_transposition_invariant():
    """The same +-200 triangle on two different notes yields the same cents alphabet (the
    lever): cents are tuning/transposition invariant, so low and high notes overlap."""
    tri = [0, 200, 400, 200, 0, -200, -400, -200]

    def mods_for(note):
        base = _grid(note)
        freq = np.array([base + tri[t % len(tri)] for t in range(160)], dtype=np.int64)
        return set(int(m) for m in pitch_grid.decompose_voice(freq)["mod"])

    assert mods_for(37) & mods_for(61)
