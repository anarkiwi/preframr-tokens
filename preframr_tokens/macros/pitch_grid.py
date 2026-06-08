"""Recovered-table pitch model (design/universal_multiresolution_pitch.md). A tracker plays note N as
an EXACT entry from its note->freq table, so the universal abstraction is the NOTE INDEX (shared
semitone grid; note 49 ~ C5 in every tune) and the per-voice TABLE recovers the exact 16-bit freq for
each note (the tracker's table + tuning + per-voice detune). Static notes are PURE (residual 0);
residual is nonzero only for genuine modulation (vibrato/slide). Lossless: reconstruct == freq.
"""

import math

import numpy as np

_FBASE = 16777216.0 / 985248.0
_ANCHOR = _FBASE * 16.0
_LOG_ANCHOR = math.log2(_ANCHOR)


def voice_tuning(freqs):
    """The voice's continuous tuning offset (semitone fraction, -0.5..0.5) -- the tune's tuning + the
    voice's detune (chorus). Circular mean of the fractional semitone so a tune tuned ~half a semitone
    (e.g. Galway +0.44) still assigns notes to the right semitone, and chorus voices keep distinct
    tunings. Notes stay UNIVERSAL (the offset is emitted, not baked into the index)."""
    f = np.asarray(freqs, dtype=np.int64)
    f = f[f > 8]
    if f.size < 8:
        return 0.0
    vals, cnt = np.unique(f, return_counts=True)
    held = vals[cnt >= max(2, int(0.01 * f.size))]
    vals = held if held.size >= 4 else vals
    frac = (12.0 * (np.log2(vals.astype(np.float64)) - _LOG_ANCHOR)) % 1.0
    ang = 2.0 * math.pi * frac
    off = math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2.0 * math.pi)
    return ((off + 0.5) % 1.0) - 0.5


def note_index(freq, tuning=0.0):
    """Nearest-semitone index on the shared grid offset by the per-voice ``tuning`` (note 49 ~ C5). The
    transferable abstraction; intervals are delta-index (transposition + tuning invariant).
    """
    f = np.asarray(freq, dtype=np.float64)
    out = np.zeros(f.shape, dtype=np.int64)
    m = f > 0
    out[m] = np.round(12.0 * (np.log2(f[m]) - _LOG_ANCHOR) - tuning).astype(np.int64)
    return out


def note_freq(note, tuning=0.0):
    """The grid 16-bit freq for a note index at ``tuning`` -- the shared recon the encoder and decoder
    both use (``freq = note_freq(note, tuning) + residual`` is byte-exact). Inverse-consistent with
    note_index: ``note_index(note_freq(n, t), t) == n``."""
    n = np.clip(np.asarray(note, dtype=np.float64), -150.0, 150.0)
    return np.clip(np.round(_ANCHOR * 2.0 ** ((n + tuning) / 12.0)), 0, 65535).astype(
        np.int64
    )


_TUNING_Q = 256.0


def tuning_to_q(tuning):
    """Quantize a per-voice tuning (-0.5..0.5 semitone) to a byte the atom carries; encode + decode
    share this + ``q_to_tuning`` so the grid-recon (``note_freq``) matches and the residual stays exact.
    """
    return int(max(0, min(255, round((float(tuning) + 0.5) * _TUNING_Q))))


def q_to_tuning(q):
    """Inverse of ``tuning_to_q``: byte -> tuning fraction."""
    return (int(q) & 0xFF) / _TUNING_Q - 0.5


def note_freq_at(note, tuning):
    """Scalar ``note_freq`` for one note (decoder convenience)."""
    return int(note_freq(np.asarray([int(note)]), tuning)[0])


def recover_table(freqs, tuning=None):
    """Per-voice note->freq table: the EXACT 16-bit freq each note maps to (modal over the voice's
    frames = the tracker's table entry), with the per-voice ``tuning`` applied so detuned tunes
    (Galway) and chorus voices index correctly. ~20 entries/voice; makes static notes pure.
    """
    f = np.asarray(freqs, dtype=np.int64)
    if tuning is None:
        tuning = voice_tuning(f)
    voiced = f > 8
    note = note_index(f, tuning)
    table = {}
    for n in sorted({int(x) for x in note[voiced]}):
        vals = f[voiced & (note == n)]
        med = float(np.median(vals))
        table[n] = int(vals[int(np.argmin(np.abs(vals.astype(np.float64) - med)))])
    return table
