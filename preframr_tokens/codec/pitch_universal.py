"""Universal pitch basis prototype.

Investigates whether all SID tunes can be normalized onto ONE canonical 12-TET
note->freq table (LOSSY on absolute base pitch, faithful on relative modulation).

Pitch physics (PAL):
  f_Hz = Fn * f_clk / 2^24 ;  Fn = f_Hz * 2^24 / f_clk
Work in log2(Fn). A semitone = 1/12 in log2(freq) (tuning-independent because
f_clk and 2^24 are constant multipliers that vanish in log differences). Cents =
1200 * dlog2. Tuning offset = constant log shift; the SID-vs-Hz scale constant is
also just a constant log offset, so we can grid directly on log2(Fn) without ever
converting to Hz.

BASE pitch extraction: reuse freq_relative.encode_freq. A NOTE token marks a note
onset; the absolute freq AFTER applying that note's interval is the base-pitch
TARGET. MOD spans (vibrato/glide/arp) are excluded from the base-pitch set so
modulation spread does not inflate it. We collect the post-NOTE absolute freq
values per voice = that voice's base-pitch set.
"""

import numpy as np

from preframr_tokens.codec import freq_relative as FR

LOG2 = np.log2(np.e)
SEMI = 1.0 / 12.0  # one semitone in log2(freq)


def base_pitches_from_freq(f):
    """Given a per-frame freq series for one voice, return the list of distinct
    absolute base-pitch freq values (Fn) at NOTE onsets (MOD spans excluded).

    We re-walk the encode_freq token stream and emit the running absolute freq
    at each NOTE token. MOD spans are skipped for base-pitch collection but we
    still advance the running value through them so subsequent NOTE targets stay
    correct."""
    toks = FR.encode_freq([int(x) for x in f])
    cur = 0
    bases = []
    for t in toks:
        if t[0] == "NOTE":
            cur += t[1]
            if cur > 0:
                bases.append(cur)
        else:
            _, e, n = t
            p = len(e)
            for k in range(n):
                cur += e[k % p]
            # the freq at the end of a MOD is a continuation, not a fresh onset
    return bases


def held_base_pitches(f, min_hold=3):
    """Cleaner base-pitch extractor: the freq VALUES that the voice holds steady
    for >= min_hold consecutive frames (sustained note targets). Returns a list of
    (fn, weight) where weight = total frames held at that value. This excludes
    glide steps / transients (which never hold) and is unaffected by vibrato
    spread (vibrato never holds one value). It can miss notes that are ALWAYS
    vibrato'd, but those are a minority and symmetric around the grid anyway."""
    f = np.asarray(f, dtype=np.int64)
    out = {}
    if len(f) == 0:
        return out
    run_v, run = int(f[0]), 1
    for x in f[1:]:
        x = int(x)
        if x == run_v:
            run += 1
        else:
            if run >= min_hold and run_v > 0:
                out[run_v] = out.get(run_v, 0) + run
            run_v, run = x, 1
    if run >= min_hold and run_v > 0:
        out[run_v] = out.get(run_v, 0) + run
    return out


def log2fn(fn):
    return np.log2(np.asarray(fn, dtype=np.float64))


def snap_error_cents(logf, offset):
    """For an array of log2(Fn) base pitches and a grid phase `offset` (in log2),
    snap each to nearest 12-TET semitone and return signed error in cents.

    Grid: semitone lines at offset + k*SEMI for integer k. snap residual r =
    (logf - offset) mod SEMI, mapped to [-SEMI/2, SEMI/2). cents = 1200*r."""
    r = (logf - offset) / SEMI
    err_semi = r - np.round(r)
    return err_semi * 100.0  # 100 cents per semitone


def _wmedian(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    return x[np.searchsorted(cw, cw[-1] / 2.0)]


def best_offset(logf, robust=True, w=None):
    """Data-driven grid phase: minimise snap error over offset in [0,SEMI).
    Coarse-to-fine. robust=True minimises (weighted) MEDIAN |err| (insensitive to
    a minority of off-grid glide/detune notes); False minimises MEAN. w = optional
    per-pitch weights (e.g. hold-frame counts)."""
    if len(logf) == 0:
        return 0.0, np.array([])
    if w is None:
        w = np.ones(len(logf))
    w = np.asarray(w, dtype=np.float64)

    def cost(o):
        a = np.abs(snap_error_cents(logf, o))
        return _wmedian(a, w) if robust else np.average(a, weights=w)

    grid = np.linspace(0, SEMI, 200, endpoint=False)
    best_o = min(grid, key=cost)
    grid2 = np.linspace(best_o - SEMI / 200, best_o + SEMI / 200, 100)
    best_o = min(grid2, key=cost)
    return best_o, snap_error_cents(logf, best_o)
