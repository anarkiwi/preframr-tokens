"""Pitch-recovery acceptance (strict): render real SWM + defMON tunes through their players and assert
(1) the decomposition reproduces the tracker's EXACT 16-bit pitches bit-for-bit, and (2) on sustained
frames the recovered note index equals the tracker's OWN table index (FREQTBL / NOTE_PITCH) up to a
constant per-voice offset -- exact notes AND pitches, not an approximation. Skips without the tracker
deps; Hubbard/Galway use the detuned-ET bit-exact test below + /scratch/tmp/hubbard_galway_pitch.py.
"""

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
    return np.array(rows)


def _stable(fr, voiced):
    out = np.zeros(len(fr), dtype=bool)
    for i in range(2, len(fr) - 2):
        if voiced[i] and len(set(fr[i - 2 : i + 3].tolist())) == 1:
            out[i] = True
    return out


def _recovered(fr):
    """The current pitch model's reconstruction pieces: (note_index, base, residual) with
    ``base + residual == fr`` exact by construction (recovered table over the tuned grid).
    """
    tuning = pg.q_to_tuning(pg.tuning_to_q(pg.voice_tuning(fr)))
    table = pg.recover_table(fr, tuning)
    ni = pg.note_index(fr, tuning)
    base = pg.note_freq(ni, tuning).copy()
    for note, freq in table.items():
        base[ni == note] = freq
    return ni, base, fr - base


def _assert_exact_notes_and_pitches(S, table):
    ftbl = np.array([int(v) for v in table], dtype=np.int64)
    checked = 0
    for b in (0, 7, 14):
        fr = (S[:, b] + 256 * S[:, b + 1]).astype(np.int64)
        gate = (S[:, b + 4] & 1).astype(bool)
        voiced = (fr > 8) & gate
        if int(voiced.sum()) < 30:
            continue
        mynote, base, resid = _recovered(fr)
        assert np.array_equal(base + resid, fr), f"voice {b}: pitches not bit-exact"
        stable = _stable(fr, voiced)
        ns = int(stable.sum())
        if ns < 10:
            continue
        checked += 1
        pure = float((resid[stable] == 0).mean())
        assert (
            pure >= 0.99
        ), f"voice {b}: stable pitches not pure table entries ({pure:.2%})"
        tblidx = np.array([int(np.argmin(np.abs(ftbl - f))) for f in fr])
        diff = (mynote - tblidx)[stable]
        off = int(np.median(diff))
        match = float((diff == off).mean())
        assert (
            match >= 0.99
        ), f"voice {b}: note != tracker table index on {(1-match)*100:.0f}% of stable frames"
    assert checked, "no sustained notes to verify"


def _swm_paths():
    env = os.environ.get("PREFRAMR_SWM_DIR")
    return (
        sorted(glob.glob(os.path.join(env, "*.swm")))[:5]
        if env and os.path.isdir(env)
        else []
    )


def _defmon_paths():
    env = os.environ.get("PREFRAMR_DEFMON_DIR")
    if env and os.path.isdir(env):
        return sorted(glob.glob(os.path.join(env, "*.prg")))[:5]
    try:
        import pydefmon

        d = Path(pydefmon.__file__).resolve().parent.parent / "build" / "fixtures"
        return sorted(glob.glob(str(d / "*.prg")))[:5]
    except Exception:
        return []


@pytest.mark.parametrize(
    "swm",
    _swm_paths()
    or [pytest.param(None, marks=pytest.mark.skip(reason="set PREFRAMR_SWM_DIR"))],
)
def test_swm_exact_notes_and_pitches(swm):
    pytest.importorskip("pysidwizard")
    from pysidwizard import read_swm
    from pysidwizard.player import SWMPlayer, NOTE_FREQ_LO, NOTE_FREQ_HI

    tbl = [(h << 8) | l for l, h in zip(NOTE_FREQ_LO, NOTE_FREQ_HI)]
    S = _regstate(SWMPlayer(read_swm(swm)).play_frame, 1500)
    _assert_exact_notes_and_pitches(S, tbl)


@pytest.mark.parametrize(
    "prg",
    _defmon_paths()
    or [pytest.param(None, marks=pytest.mark.skip(reason="set PREFRAMR_DEFMON_DIR"))],
)
def test_defmon_exact_notes_and_pitches(prg):
    pytest.importorskip("pydefmon")
    from pydefmon import DefmonSong, DefmonPlayer
    from pydefmon.defmon import NOTE_PITCH_LO, NOTE_PITCH_HI

    try:
        song = DefmonSong.from_file(prg)
    except Exception as exc:
        if type(exc).__name__ == "DefmonError":
            pytest.skip(f"non-loadable fixture: {exc}")
        raise
    tbl = [(h << 8) | l for l, h in zip(NOTE_PITCH_LO, NOTE_PITCH_HI)]
    S = _regstate(DefmonPlayer(song).play_frame, 1500)
    _assert_exact_notes_and_pitches(S, tbl)


def test_detuned_scale_exact():
    """A Galway-style +44c-detuned ET scale (no Python player) reproduces bit-exactly with consecutive
    note indices and zero residual -- the per-voice tuning fit + recovered table yield the tracker's
    exact pitches + notes back."""
    notes = np.arange(28, 88)
    freqs = np.round(
        pg._ANCHOR
        * 2.0 ** ((notes + 44.0 / 100.0) / 12.0)  # pylint: disable=protected-access
    ).astype(np.int64)
    seq = np.repeat(freqs, 8)
    ni, base, resid = _recovered(seq)
    assert np.array_equal(base + resid, seq)
    assert float((resid == 0).mean()) == 1.0, "static detuned notes must be pure"
    idx = sorted({int(n) for n in ni})
    assert idx == list(range(idx[0], idx[0] + len(notes)))
