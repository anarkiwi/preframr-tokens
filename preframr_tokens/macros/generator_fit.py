"""The canonical generator-MDL fitter (design/generator_mdl_representation.md), embedded verbatim from
the validated decompose5.py prototype. Decomposes a per-frame value series into the longest-wins
generator set {HOLD, ACCUM, TABLE, TRIANGLE}; each candidate is accepted only for the longest prefix
its OWN decoder reproduces, so reconstruction equals source by construction. recon/note_of are the
SAME functions the encoder and the decoder use so freq residuals are bit-exact.
"""

import math

import numpy as np

from preframr_tokens.stfconstants import GEN_FREQ_REGS, GEN_SCALAR_REGS

_MAXP = 24
_MINTRI = 6
_FBASE = 16777216.0 / 985248.0

__all__ = [
    "tune_ref",
    "recon",
    "note_of",
    "decompose",
    "channels",
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


def tune_ref(freqs):
    f = np.asarray([x for x in freqs if x > 8], dtype=np.float64)
    if len(f) < 16:
        return 0.0
    frac = (12.0 * np.log2(f)) % 1.0
    ang = 2.0 * math.pi * frac
    return (
        float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2.0 * math.pi))
        % 1.0
    )


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


def gen_hold(s, i, n):
    v = s[i]
    j = i
    while j < n and s[j] == v:
        j += 1
    return j - i


def gen_accum(s, i, n):
    if i + 1 >= n:
        return 1, 0
    d = s[i + 1] - s[i]
    j = i + 1
    while j + 1 < n and s[j + 1] - s[j] == d:
        j += 1
    return j + 1 - i, d


def gen_table(s, i, n, p):
    if i + p >= n:
        return 0
    j = i + p
    while j < n and s[j] == s[j - p]:
        j += 1
    ln = j - i
    if ln < max(3, p + 1) or len(set(s[i : i + p])) < 2:
        return 0
    return ln


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


def gen_tri(s, i, n):
    if i + 1 >= n or s[i + 1] == s[i]:
        return None
    step = abs(s[i + 1] - s[i])
    dir0 = 1 if s[i + 1] > s[i] else -1
    cur = s[i]
    d = dir0
    j = i
    hit, lot = [], []
    while j + 1 < n:
        if s[j + 1] == cur + step * d:
            cur += step * d
        elif s[j + 1] == cur - step * d:
            (hit if d > 0 else lot).append(cur)
            d = -d
            cur += step * d
        else:
            break
        j += 1
    if not (hit and lot) or len(set(hit)) != 1 or len(set(lot)) != 1:
        return None
    lo, hi = min(lot), max(hit)
    seq = _tri_seq(s[i], step, lo, hi, dir0, n - i)
    k = 0
    while k < len(seq) and s[i + k] == seq[k]:
        k += 1
    if k < _MINTRI:
        return None
    return k, (step, lo, hi, dir0)


def fit_run(s, i):
    n = len(s)
    best = ("HOLD", gen_hold(s, i, n), None)
    al, d = gen_accum(s, i, n)
    if d != 0 and al > best[1]:
        best = ("ACCUM", al, d)
    tr = gen_tri(s, i, n)
    if tr and tr[0] > best[1]:
        best = ("TRI", tr[0], tr[1])
    for p in range(2, _MAXP + 1):
        tl = gen_table(s, i, n, p)
        if tl > best[1]:
            best = ("TABLE", tl, p)
    return best


def decompose(s):
    n = len(s)
    out = []
    i = 0
    while i < n:
        kind, ln, params = fit_run(s, i)
        out.append((kind, i, max(1, ln), params))
        i += max(1, ln)
    return out


def channels(state):
    """Yield ``(reg, is_freq, series)`` for the 13 generator-owned value channels of one
    ``register_state`` array. Freq is the combined 16-bit word per voice (``S[:,b]+256*S[:,b+1]``);
    the per-byte scalar channels (pw/cut/res/modevol) read the settled register directly. ctrl/AD/SR
    are excluded -- ``InstrumentProgramPass`` owns them."""
    for b in GEN_FREQ_REGS:
        series = (
            state[:, b].astype(np.int64) + 256 * state[:, b + 1].astype(np.int64)
        ).tolist()
        yield int(b), True, series
    for reg in GEN_SCALAR_REGS:
        yield int(reg), False, state[:, reg].astype(np.int64).tolist()


def all_freqs(state):
    """Every voice's combined 16-bit freq across all frames -- the population ``tune_ref`` calibrates
    the per-tune semitone LUT over."""
    out = []
    for b in GEN_FREQ_REGS:
        out.extend(
            (
                state[:, b].astype(np.int64) + 256 * state[:, b + 1].astype(np.int64)
            ).tolist()
        )
    return out
