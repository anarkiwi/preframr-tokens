"""Numba-JIT kernels for the generic recovery's hot per-frame loops.

The driver-agnostic generic recovery (:mod:`archetypes`) covers each note-on
segment by re-rendering many parameterised archetype candidates and taking the
longest byte-exact prefix.  On a large multispeed / 2SID trace the cover search
re-renders the per-frame accumulator / table-walk / vibrato kernels millions of
times, and the longest-prefix match (:func:`archetypes._match_prefix`) runs once
per candidate -- the measured hot loops (cProfile on Grid_Runner: ``render_arp``,
``_match_prefix``, the vibrato phase sequence, ``render_pingpong`` ...).

These are PURE speedups: every kernel here is byte-for-byte identical to the
pure-Python reference it replaces (validated by a parametrized equality test,
:mod:`tests.test_njit_kernels`).  The chip math masks to fixed widths
(``& 0xFFFF``, ``& 0xFFF``, ``& 0xFF``); Numba uses fixed-width machine ``int64``
with wraparound where pure Python uses arbitrary precision, so every accumulator
here is ``int64`` with the SAME explicit masks the pure-Python renderers apply and
every value stays well within ``int64`` range, so the two agree exactly.

Numba is a declared dependency (``pyproject``), so the docker CI build installs it
and exercises the fast path.  As a belt-and-suspenders fallback, if numba is not
importable this module degrades to an identity ``njit`` decorator and the kernels
run as ordinary (slower) Python -- still byte-exact -- so the package never
hard-fails on a numba-less interpreter.
"""

import numpy as np

try:  # pragma: no cover - exercised by whichever path the environment provides
    from numba import njit as _numba_njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a declared dependency
    _numba_njit = None
    HAVE_NUMBA = False


def _identity_decorator(func):  # pragma: no cover - numba-absent fallback only
    return func


def njit(*args, **kwargs):
    """``numba.njit`` (``cache=True`` defaulted on so the per-tune / per-worker
    compile cost is amortised across the campaign) when numba is importable, else
    an identity decorator so the kernel runs as plain Python -- byte-exact either
    way, just slower without numba.  Supports both ``@njit`` and ``@njit(...)``."""
    if _numba_njit is None:  # pragma: no cover - numba is a declared dependency
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _identity_decorator
    kwargs.setdefault("cache", True)
    if args and callable(args[0]) and not kwargs:
        return _numba_njit(cache=True)(args[0])
    return _numba_njit(*args, **kwargs)


# ---------------------------------------------------------------------------
# Longest-prefix byte-exact match.
# ---------------------------------------------------------------------------
@njit
def match_prefix(rend, seg):
    """First index where ``rend`` and ``seg`` differ over their common length, or
    that common length when they agree on it -- the kernel form of
    :func:`archetypes._match_prefix` (``argmin`` of the equality array)."""
    n = len(rend)
    m = len(seg)
    length = n if n < m else m
    for i in range(length):
        if rend[i] != seg[i]:
            return i
    return length


# ---------------------------------------------------------------------------
# Triangle vibrato phase + value renderers.
# ---------------------------------------------------------------------------
@njit
def tri_phase_seq(ctr0, seg_len):
    """The triangle phase 0..3 for ``seg_len`` consecutive counters from ``ctr0``:
    ``osc = ctr & 7; if osc >= 4: osc ^= 7`` -- the vectorised ``tri_phase``."""
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        osc = (ctr0 + i) & 7
        if osc >= 4:
            osc ^= 7
        out[i] = osc
    return out


@njit
def render_arp(seg_len, notes_freqs, period, ctr0, dwell):
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        step = (ctr0 + i) // dwell
        out[i] = notes_freqs[step % period]
    return out


@njit
def render_accum(seg_len, v0, rate, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val += rate
    return out


@njit
def render_wrapaccum(seg_len, v0, rate, lo_b, hi_b):
    out = np.empty(seg_len, dtype=np.int64)
    span = hi_b - lo_b
    val = v0
    for i in range(seg_len):
        out[i] = val
        val += rate
        if rate > 0 and val >= hi_b:
            val -= span
        elif rate < 0 and val < lo_b:
            val += span
    return out


@njit
def render_glide(seg_len, n0, step, dwell, lead, note_table):
    out = np.empty(seg_len, dtype=np.int64)
    nt = len(note_table)
    for i in range(seg_len):
        k = 0 if i < lead else (i - lead) // dwell
        idx = (n0 + step * k) & 0xFF
        out[i] = note_table[idx] if 0 <= idx < nt else 0
    return out


@njit
def hi_overlay(base_lane, sfh0, par, ctr0):
    length = len(base_lane)
    out = np.empty(length, dtype=np.int64)
    sfh = sfh0 & 0xFF
    for i in range(length):
        lo = base_lane[i] & 0xFF
        hi = (base_lane[i] >> 8) & 0xFF
        if ((ctr0 + i) & 1) == par and sfh != 0:
            old = sfh
            sfh = (sfh - 1) & 0xFF
            hi = old
        out[i] = lo | (hi << 8)
    return out


@njit
def render_additive_pw(seg_len, p0, pulsevalue, carry_seq, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    clen = len(carry_seq)
    hi = p0 & ~0xFF
    lo = p0 & 0xFF
    for i in range(seg_len):
        out[i] = (hi | lo) & width_mask
        carry = carry_seq[i] if i < clen else 0
        lo = (lo + pulsevalue + carry) & 0xFF
    return out


@njit
def render_pingpong(seg_len, v0, rate, lo_b, hi_b, dwell, d0, dir0):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    dwell_left = d0
    direction = dir0
    for i in range(seg_len):
        out[i] = val
        dwell_left -= 1
        if dwell_left < 0:
            dwell_left = dwell
            if direction:
                nxt = val - rate
                if nxt < lo_b:
                    direction = 0
                    nxt = val + rate
                val = nxt
            else:
                nxt = val + rate
                if nxt > hi_b:
                    direction = 1
                    nxt = val - rate
                val = nxt
    return out


@njit
def _pingpong_match(seg, length, v0, rate, lo_b, hi_b, dwell, d0, dir0):
    """Longest byte-exact prefix of ``render_pingpong(...)`` against ``seg`` WITHOUT
    materialising the render: step the reflect accumulator and compare each emitted
    value to ``seg`` in place, stopping at the first divergence -- the exact value
    :func:`match_prefix` would compute on the full render, but allocation-free so the
    fused search avoids one numpy array per candidate."""
    val = v0
    dwell_left = d0
    direction = dir0
    for i in range(length):
        if val != seg[i]:
            return i
        dwell_left -= 1
        if dwell_left < 0:
            dwell_left = dwell
            if direction:
                nxt = val - rate
                if nxt < lo_b:
                    direction = 0
                    nxt = val + rate
                val = nxt
            else:
                nxt = val + rate
                if nxt > hi_b:
                    direction = 1
                    nxt = val - rate
                val = nxt
    return length


@njit
def pingpong_search(seg, base, rates, bound_los, bound_his, max_dwell, minrun):
    """Fused :func:`archetypes._prefix_pingpong` inner search.  Enumerates the SAME
    nested candidate grid -- ``rate`` (the few dominant deltas), the two reflection
    ``(lo, hi)`` bound conventions, ``dwell`` in ``[0, max_dwell)``, ``d0`` in
    ``[0, dwell]``, ``dir0`` in ``(0, 1)`` -- in the SAME order, keeping the first
    candidate with a strictly longer byte-exact prefix (``match > best`` with
    first-wins ties) and returning at once on a full-length match.  Returns
    ``(best_match, rate, lo, hi, dwell, d0, dir0)`` with ``best_match == -1`` when no
    candidate reaches ``minrun`` -- byte-identical to the Python loop, but with no
    per-candidate render allocation or dispatch."""
    length = len(seg)
    nrates = len(rates)
    nbounds = len(bound_los)
    best_match = -1
    best_rate = 0
    best_lo = 0
    best_hi = 0
    best_dwell = 0
    best_d0 = 0
    best_dir0 = 0
    for ri in range(nrates):
        rate = rates[ri]
        for bi in range(nbounds):
            refl_lo = bound_los[bi]
            refl_hi = bound_his[bi]
            for dwell in range(0, max_dwell):
                for d0 in range(dwell + 1):
                    for dir0 in range(2):
                        match = _pingpong_match(
                            seg, length, base, rate, refl_lo, refl_hi, dwell, d0, dir0
                        )
                        if match >= minrun and match > best_match:
                            best_match = match
                            best_rate = rate
                            best_lo = refl_lo
                            best_hi = refl_hi
                            best_dwell = dwell
                            best_d0 = d0
                            best_dir0 = dir0
                            if match == length:
                                return (
                                    best_match,
                                    best_rate,
                                    best_lo,
                                    best_hi,
                                    best_dwell,
                                    best_d0,
                                    best_dir0,
                                )
    return (
        best_match,
        best_rate,
        best_lo,
        best_hi,
        best_dwell,
        best_d0,
        best_dir0,
    )


@njit
def render_pingfold(seg_len, step, frac, lo, hi, acc0, dir0):
    out = np.empty(seg_len, dtype=np.int64)
    acc = acc0
    direction = dir0
    for i in range(seg_len):
        out[i] = acc >> frac
        acc += step * direction
        if acc > hi:
            acc = 2 * hi - acc
            direction = -direction
        elif acc < lo:
            acc = 2 * lo - acc
            direction = -direction
    return out


@njit
def render_vibreflect(seg_len, center, speed, cmpvalue, delay, vibtime0):
    out = np.empty(seg_len, dtype=np.int64)
    freq = center & 0xFFFF
    vibtime = vibtime0 & 0xFF
    vibdelay = delay
    for i in range(seg_len):
        if vibdelay > 1:
            vibdelay -= 1
            out[i] = freq
            continue
        if vibtime < 0x80 and vibtime > cmpvalue:
            vibtime ^= 0xFF
        vibtime = (vibtime + 2) & 0xFF
        if vibtime & 1:
            freq = (freq - speed) & 0xFFFF
        else:
            freq = (freq + speed) & 0xFFFF
        out[i] = freq
    return out


@njit
def render_decay(seg_len, v0, rate, every, ctr0):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val
        if (ctr0 + i + 1) % every == 0:
            val = (val - rate) & 0xFFFF
    return out


@njit
def render_dwell_accum(seg_len, v0, rate, dwell, lead):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    counter = 0
    for i in range(seg_len):
        out[i] = val & 0xFFFF
        if i >= lead:
            counter += 1
            if counter % dwell == 0:
                val = (val + rate) & 0xFFFF
    return out


@njit
def render_maskaccum(seg_len, v0, rate, mask, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    period = len(mask)
    for i in range(seg_len):
        out[i] = val & width_mask
        if mask[i % period]:
            val = (val + rate) & width_mask
    return out


@njit
def render_tablewalk(seg_len, table, ctr0):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(table)
    for i in range(seg_len):
        out[i] = table[(ctr0 + i) % period]
    return out


@njit
def render_ratewalk(seg_len, v0, rate_table, ctr0, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(rate_table)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[(ctr0 + i) % period]) & width_mask
    return out


@njit
def render_dwellratewalk(seg_len, v0, rate_table, dwell, ctr0, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(rate_table)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[((ctr0 + i) // dwell) % period]) & width_mask
    return out


@njit
def render_tablewalk_lead(seg_len, lead, value0, table, ctr0):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(table)
    for i in range(seg_len):
        out[i] = value0 if i < lead else table[(ctr0 + i - lead) % period]
    return out


@njit
def render_wavetable_ptr(seg_len, table, phase, advance):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(table)
    alen = len(advance)
    ptr = phase % period
    for i in range(seg_len):
        if i > 0 and advance[(i - 1) % alen]:
            ptr = (ptr + 1) % period
        out[i] = table[ptr]
    return out
