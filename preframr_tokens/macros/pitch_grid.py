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


def note_index(freq):
    """Nearest-semitone index on the shared universal grid (note 49 ~ C5). The transferable abstraction;
    intervals are delta-index (transposition-invariant)."""
    f = np.asarray(freq, dtype=np.float64)
    out = np.zeros(f.shape, dtype=np.int64)
    m = f > 0
    out[m] = np.round(12.0 * (np.log2(f[m]) - _LOG_ANCHOR)).astype(np.int64)
    return out


def recover_table(freqs):
    """Per-voice note->freq table: the EXACT 16-bit freq each note maps to (modal over the voice's
    frames = the tracker's table entry). ~20 entries/voice; makes static notes pure."""
    f = np.asarray(freqs, dtype=np.int64)
    voiced = f > 8
    note = note_index(f)
    table = {}
    for n in sorted({int(x) for x in note[voiced]}):
        vals = f[voiced & (note == n)]
        u, c = np.unique(vals, return_counts=True)
        table[n] = int(u[int(np.argmax(c))])
    return table


def _table_vec(table, note):
    return np.array([table.get(int(n), 0) for n in note], dtype=np.int64)


def decompose_voice(freqs, table=None):
    """Lossless per-frame decomposition: note[] (shared-grid index), resid[] (freq - table[note]; 0 for
    static notes, the modulation otherwise), voiced[], table. Defaults to ``recover_table(freqs)``.
    """
    f = np.asarray(freqs, dtype=np.int64)
    if table is None:
        table = recover_table(f)
    voiced = f > 8
    note = note_index(f)
    resid = np.where(voiced, f - _table_vec(table, note), 0)
    note = np.where(voiced, note, 0)
    return {"note": note, "resid": resid, "voiced": voiced, "table": table}


def reconstruct(dec):
    """Bit-exact inverse of ``decompose_voice``."""
    f = _table_vec(dec["table"], dec["note"]) + dec["resid"]
    return np.where(dec["voiced"], np.clip(f, 0, 65535), 0).astype(np.int64)


def pure_fraction(dec):
    """Voiced frames that are an exact table note (resid==0) -- the structural-only, pure-note share."""
    v = dec["voiced"]
    n = int(v.sum())
    return float(((dec["resid"] == 0) & v).sum()) / n if n else 0.0
