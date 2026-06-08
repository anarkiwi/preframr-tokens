"""Per-tune semitone-LUT pitch helpers shared by the gesture decoder + note-index layer: ``recon`` maps
a note index to its exact 16-bit freq word and ``note_of`` is its nearest-semitone inverse (bit-exact
round-trip), with the zig-zag interval map (``zig``/``unzig``) and the triangle replay (``_tri_seq``)
the surviving generator/melody decoders still read."""

import numpy as np

_FBASE = 16777216.0 / 985248.0

__all__ = [
    "recon",
    "note_of",
    "zig",
    "unzig",
]


_NOTES = 128
_lut_cache = {}


def _lut(ref):
    """Per-tune equal-tempered note->16-bit-freq LUT (the driver substrate): ``LUT[n] = round(2**((n+ref)
    /12) * _FBASE * 16)`` clamped to 65535, covering the full freq range over 128 semitones. ``ref`` is
    the per-tune tuning offset. Cached by ref so the per-frame decoder is not rebuilding it each call.
    """
    key = round(float(ref), 6)
    lut = _lut_cache.get(key)
    if lut is None:
        lut = np.array(
            [
                min(65535, int(round(2.0 ** ((n + ref) / 12.0) * _FBASE * 16.0)))
                for n in range(_NOTES)
            ],
            dtype=np.int64,
        )
        _lut_cache[key] = lut
    return lut


def recon(note, ref):
    """The exact 16-bit freq word for ``note`` on the per-tune LUT (the inverse ``note_of`` keys off, so
    ``recon(note_of(f)) + (f - recon(note_of(f))) == f`` is bit-exact). Clamped to the 128-note range.
    """
    lut = _lut(ref)
    return int(lut[max(0, min(_NOTES - 1, int(note)))])


def note_of(f, ref):
    """Nearest-semitone index for a 16-bit freq word on the per-tune LUT (searchsorted, so ``|f -
    recon(note_of(f))|`` is bounded by half a semitone -- the residual always fits 16 bits, unlike the
    open-form ``round(12*log2 f)`` which overflows when the note saturates the LUT)."""
    if f <= 0:
        return 0
    lut = _lut(ref)
    idx = int(min(max(int(np.searchsorted(lut, f)), 1), _NOTES - 1))
    return idx - 1 if (f - int(lut[idx - 1])) <= (int(lut[idx]) - f) else idx


def zig(n):
    """Zig-zag bias map: signed semitone interval -> small non-negative token (0,-1,1,-2 -> 0,1,2,3),
    so a melody's clustered-near-zero intervals stay low-cardinality. Inverse of ``unzig``.
    """
    n = int(n)
    return (n << 1) ^ (n >> 63)


def unzig(z):
    """Inverse of ``zig``: non-negative token -> signed semitone interval."""
    z = int(z)
    return (z >> 1) ^ -(z & 1)


def _tri_seq(start, step, lo, hi, dir0, ln):
    out = [start]
    cur = start
    d = dir0
    for _ in range(ln - 1):
        nxt = cur + step * d
        if nxt > hi or nxt < lo:
            d = -d
            nxt = cur + step * d
        cur = nxt
        out.append(cur)
    return out
