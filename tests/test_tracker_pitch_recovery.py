"""Pitch-recovery acceptance: render real SWM + defMON tunes through their players, recover each
voice's note->freq table with pitch_grid, and assert the recovered pitches ARE the tracker's own
table values (the abstraction recovers tracker pitches as pure notes). Skips without the optional
tracker deps. Hubbard/Galway (no Python player) are covered by the equal-tempered table test below
plus the corpus measurement in /scratch/tmp/hubbard_galway_pitch.py."""

import glob
import os
from pathlib import Path

import numpy as np
import pytest

from preframr_tokens.macros import pitch_grid as pg

D400 = 0xD400


def _regstate(play_frame, nframes):
    cur = np.zeros(25, dtype=np.int64)
    rows = []
    for _ in range(nframes):
        for reg, val in play_frame():
            off = reg - D400 if reg >= D400 else reg
            if 0 <= off < 25:
                cur[off] = val & 0xFF
        rows.append(cur.copy())
    return np.array(rows, dtype=np.int64)


def _cents_to_nearest(entries, table_vals):
    tv = np.array(sorted(v for v in table_vals if v > 0), dtype=np.float64)
    return [
        float(np.min(np.abs(1200.0 * np.log2(float(f) / tv)))) for f in entries if f > 8
    ]


def _recovered_entries(S):
    out = []
    for b in (0, 7, 14):
        out.extend(
            int(v) for v in pg.recover_table(S[:, b] + 256 * S[:, b + 1]).values()
        )
    return out


def _assert_tracker_pitches(entries, table_vals):
    cents = _cents_to_nearest(entries, table_vals)
    assert cents, "no recovered pitches"
    median = float(np.median(cents))
    within25 = sum(c <= 25 for c in cents) / len(cents)
    assert (
        median <= 5.0
    ), f"recovered pitches are not tracker pitches (median {median:.1f}c off)"
    assert (
        within25 >= 0.7
    ), f"only {within25*100:.0f}% of recovered pitches within 25c of a tracker pitch"


def _swm_paths():
    env = os.environ.get("PREFRAMR_SWM_DIR")
    return (
        sorted(glob.glob(os.path.join(env, "*.swm")))[:4]
        if env and os.path.isdir(env)
        else []
    )


def _defmon_paths():
    env = os.environ.get("PREFRAMR_DEFMON_DIR")
    if env and os.path.isdir(env):
        return sorted(glob.glob(os.path.join(env, "*.prg")))[:4]
    try:
        import pydefmon

        d = Path(pydefmon.__file__).resolve().parent.parent / "build" / "fixtures"
        return sorted(glob.glob(str(d / "*.prg")))[:4]
    except Exception:
        return []


@pytest.mark.parametrize(
    "swm",
    _swm_paths()
    or [pytest.param(None, marks=pytest.mark.skip(reason="set PREFRAMR_SWM_DIR"))],
)
def test_swm_pitches_are_tracker_table(swm):
    pytest.importorskip("pysidwizard")
    from pysidwizard import read_swm
    from pysidwizard.player import SWMPlayer, NOTE_FREQ_LO, NOTE_FREQ_HI

    tv = [(h << 8) | l for l, h in zip(NOTE_FREQ_LO, NOTE_FREQ_HI)]
    S = _regstate(SWMPlayer(read_swm(swm)).play_frame, 1500)
    _assert_tracker_pitches(_recovered_entries(S), tv)


@pytest.mark.parametrize(
    "prg",
    _defmon_paths()
    or [pytest.param(None, marks=pytest.mark.skip(reason="set PREFRAMR_DEFMON_DIR"))],
)
def test_defmon_pitches_are_tracker_table(prg):
    pytest.importorskip("pydefmon")
    from pydefmon import DefmonSong, DefmonPlayer
    from pydefmon.defmon import NOTE_PITCH_LO, NOTE_PITCH_HI

    try:
        song = DefmonSong.from_file(prg)
    except Exception as exc:
        if type(exc).__name__ == "DefmonError":
            pytest.skip(f"non-loadable fixture: {exc}")
        raise
    tv = [(h << 8) | l for l, h in zip(NOTE_PITCH_LO, NOTE_PITCH_HI)]
    S = _regstate(DefmonPlayer(song).play_frame, 1500)
    entries = _recovered_entries(S)
    cents = _cents_to_nearest(entries, tv)
    assert cents
    assert sum(c <= 25 for c in cents) / len(cents) >= 0.6


def test_equal_tempered_table_recovers_pure_at_any_tuning():
    """Hubbard/Galway use the universal note->freq table at the tune's tuning (driver reference). The
    abstraction must recover an ET table (any detune) as pure notes -- proven here for a +33c-detuned
    ET scale built from the canonical curve, standing in for a no-Python-player driver.
    """
    notes = np.arange(28, 88)
    freqs = np.round(pg._ANCHOR * 2.0 ** ((notes + 33.0 / 100.0) / 12.0)).astype(
        np.int64
    )
    seq = np.repeat(freqs, 8)
    dec = pg.decompose_voice(seq)
    assert np.array_equal(pg.reconstruct(dec), seq)
    assert pg.pure_fraction(dec) == 1.0
    assert len(dec["table"]) == len(notes)
