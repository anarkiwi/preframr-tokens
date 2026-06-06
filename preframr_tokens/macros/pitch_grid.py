"""Universal multi-resolution pitch grid (design/universal_multiresolution_pitch.md). ONE shared
canonical equal-tempered grid (fixed anchor, NOT a per-tune ref); pitch decomposes into NOTE
(semitone, the structural prediction target) + per-voice TUNING (sub-semitone detune, where the
chorus lives) + LSB (exact residue to the 16-bit freq, content tier). Lossless by construction:
``reconstruct(decompose(f)) == f`` exactly. Modulation rides as per-frame sub-grid deviation.
"""

import math

import numpy as np

_FBASE = 16777216.0 / 985248.0
_ANCHOR = _FBASE * 16.0
_LOG_ANCHOR = math.log2(_ANCHOR)


def fine_idx(freq, sub):
    """Universal fine-grid index of a 16-bit freq word: ``round(sub*12*log2(freq/anchor))``, where
    ``sub`` is the per-tune sub-steps per semitone (resolution). Anchor matches the legacy LUT note 0.
    """
    f = np.asarray(freq, dtype=np.float64)
    out = np.zeros(f.shape, dtype=np.int64)
    m = f > 0
    out[m] = np.round(sub * 12.0 * (np.log2(f[m]) - _LOG_ANCHOR)).astype(np.int64)
    return out


def recon_fine(idx, sub):
    """Inverse of ``fine_idx`` up to its rounding: the 16-bit freq nearest the fine-grid index."""
    idx = np.asarray(idx, dtype=np.float64)
    f = _ANCHOR * np.exp2(idx / (sub * 12.0))
    return np.clip(np.round(f), 0, 65535).astype(np.int64)


def split(idx, sub):
    """Fine index -> (note semitone, sub-semitone offset in [-sub/2, sub/2))."""
    note = np.round(idx / float(sub)).astype(np.int64)
    return note, (idx - note * sub).astype(np.int64)


def voice_tuning(freqs, sub):
    """The voice's typical sub-semitone offset (its detune bucket = the chorus signal). Median over the
    voiced frames -- near-constant per voice, differs BETWEEN voices for a chorus."""
    f = np.asarray(freqs, dtype=np.float64)
    f = f[f > 8]
    if f.size == 0:
        return 0
    _, off = split(fine_idx(f, sub), sub)
    return int(np.median(off))


def decompose_voice(freqs, sub):
    """Per-frame lossless decomposition for one voice: returns dict with note[], sub_dev[] (per-frame
    deviation from the voice tuning = modulation, mostly 0), lsb[] (exact residue), and the scalar
    voice tuning. ``reconstruct`` inverts it bit-exactly."""
    f = np.asarray(freqs, dtype=np.int64)
    vt = voice_tuning(f, sub)
    idx = fine_idx(f, sub)
    note, off = split(idx, sub)
    lsb = (f - recon_fine(idx, sub)).astype(np.int64)
    voiced = f > 0
    note = np.where(voiced, note, 0)
    sub_dev = np.where(voiced, off - vt, 0)
    lsb = np.where(voiced, lsb, 0)
    return {
        "sub": sub,
        "tuning": vt,
        "note": note,
        "sub_dev": sub_dev,
        "lsb": lsb,
        "voiced": voiced,
    }


def reconstruct(dec):
    """Bit-exact inverse of ``decompose_voice``."""
    sub, vt = dec["sub"], dec["tuning"]
    idx = dec["note"] * sub + (vt + dec["sub_dev"])
    f = recon_fine(idx, sub) + dec["lsb"]
    return np.where(dec["voiced"], np.clip(f, 0, 65535), 0).astype(np.int64)
