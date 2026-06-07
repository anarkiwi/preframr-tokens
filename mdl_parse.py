"""MDL optimal-parse over the driver's primitive basis (UNTRACKED, throwaway).

Recovering the gestures in a per-frame register series is the OPTIMAL-PARSING problem (LZ-optimal /
Knuth-Plass / Viterbi DP): shortest path on the description-length cost DAG. The generator basis is the
driver's own simple code -- NO residuals, everything is exactly one of:

  HOLD            constant value (degree-0)
  POLY(N)         forward-differenced polynomial, degree N=1..MAXDEG -- constant N-th difference, i.e. N
                  nested accumulators (RAMP=deg1 slide, deg2=parabolic/smooth vibrato `f+=v; v+=a`, ...).
                  The driver computes smooth curves with additions only via a difference table.
  PERIOD(cell)    looped table / wavetable (arp, looped LFO, PWM/WGl table).

The cost makes LOW degree + LONG run + REUSED shape win, so a constant-2nd-diff vibrato over 30 frames
beats cubic fragments. Byte-exact by construction: a POLY replays its initial difference table by
forward differencing; a PERIOD replays its delta cell; nothing is approximated.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

_HDR = 6  # per-token header bits
_MAXDEG = 3  # max forward-difference degree
_PMAX = 32  # max period probed


def nbits(x) -> int:
    x = int(x)
    return 1 if x == 0 else 2 + abs(x).bit_length()


def _wd(d, wrap):
    d = int(d)
    return ((d + 32768) % 65536) - 32768 if wrap else d


def _diffs(s, wrap):
    """Forward-difference arrays d[k] (k=0..MAXDEG); d[k][i] = k-th difference of s at i (wrap-aware)."""
    out = [s.astype(np.int64)]
    cur = s.astype(np.int64)
    for _ in range(_MAXDEG):
        dd = cur[1:] - cur[:-1]
        if wrap:
            dd = ((dd + 32768) % 65536) - 32768
        out.append(dd)
        cur = dd
    return out


def _poly_runs(diffs, N, n):
    """For each i, the last frame of the maximal degree-N run from i (constant N-th difference), and
    that N-th difference value. A run of constant d[N] over [a,b) covers frames [a, b-1+N]."""
    end = np.arange(n)
    aval = np.zeros(n, dtype=np.int64)
    dN = diffs[N]
    m = len(dN)
    k = 0
    while k < m:
        j = k
        while j < m and dN[j] == dN[k]:
            j += 1
        last = (j - 1) + N
        end[k:j] = last
        aval[k:j] = dN[k]
        k = j
    return end, aval


def _period_edges(s, n):
    edges = defaultdict(list)
    for p in range(2, min(_PMAX, n // 2) + 1):
        rep = (s[p:] == s[:-p]).astype(np.int8)
        d = np.diff(np.concatenate([[0], rep, [0]]))
        for a, e in zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()):
            if e - a < p:
                continue
            cell = tuple(int(x) for x in (s[a + 1 : a + p + 1] - s[a : a + p]))
            cost = _HDR + nbits(int(s[a])) + nbits(p) + sum(nbits(c) for c in cell)
            edges[a].append((e - 1 + p + 1, cost, ("P", cell)))
    return edges


def mdl_parse(series, wrap=False):
    """Globally optimal min-description-length parse into HOLD / POLY(N) / PERIOD tokens via shortest
    path on the position DAG. Returns [(kind, i, j, param)]: kind "H" (param=value), "D" (param=(N, a)
    = degree and constant N-th diff), or "P" (param=delta cell)."""
    s = np.asarray(series, dtype=np.int64)
    n = len(s)
    if n == 0:
        return []
    diffs = _diffs(s, wrap)
    poly = [None] + [_poly_runs(diffs, N, n) for N in range(1, _MAXDEG + 1)]
    per_edges = _period_edges(s, n)

    INF = float("inf")
    cost = [INF] * (n + 1)
    cost[0] = 0.0
    back = [None] * (n + 1)

    def relax(a, b, c, tok):
        if c < cost[b]:
            cost[b] = c
            back[b] = (a, tok)

    for i in range(n):
        if cost[i] == INF:
            continue
        base = cost[i]
        hv = int(s[i])
        # HOLD: maximal constant run, and the length-1 literal fallback
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        relax(i, j, base + _HDR + nbits(hv), ("H", hv))
        relax(i, i + 1, base + _HDR + nbits(hv), ("H", hv))
        # POLY(N): cost = header + anchor value + initial diffs 1..N (the N-th is the reusable shape)
        for N in range(1, _MAXDEG + 1):
            if i + N >= n:
                break
            end, aval = poly[N]
            e = int(end[i])
            if e - i + 1 < N + 2:  # a degree-N poly is trivial on <= N+1 points
                continue
            pcost = _HDR + nbits(hv) + sum(nbits(int(diffs[k][i])) for k in range(1, N + 1))
            relax(i, e + 1, base + pcost, ("D", (N, int(aval[i]))))
        for b, c, tok in per_edges.get(i, ()):
            relax(i, b, base + c, tok)

    toks = []
    j = n
    while j > 0:
        a, tok = back[j]
        toks.append((tok[0], a, j, tok[1]))
        j = a
    toks.reverse()
    return toks


_MINHOLD = 4


def channel_report(series, wrap=False, palette_map=None):
    """Optimal-parse one channel; report modulation-value diversity vs gesture-shape alphabet, plus
    description-length collapse. A POLY shape = (degree, N-th diff); a PERIOD shape = the cell. With the
    complete basis there is no literal floor -- a one-off frame is a length-1 HOLD and is rare."""
    s = np.asarray(series, dtype=np.int64)
    toks = mdl_parse(s, wrap)
    shapes: set = set()
    mod_vals: set = set()
    hold_count: Counter = Counter()
    hold_maxlen: Counter = Counter()
    inst_bits = 0
    for kind, i, j, p in toks:
        if kind == "D":
            shapes.add(("D", p[0], p[1]))
            mod_vals.update(int(v) for v in s[i:j] if v)
            inst_bits += _HDR + nbits(int(s[i])) + (p[0] - 1) * 8  # anchor + lower diffs
        elif kind == "P":
            shapes.add(("P", p))
            mod_vals.update(int(v) for v in s[i:j] if v)
            inst_bits += _HDR + nbits(int(s[i]))
        else:
            if p != 0:
                hold_count[p] += 1
                hold_maxlen[p] = max(hold_maxlen[p], j - i)
                inst_bits += _HDR + nbits(int(p)) if hold_count[p] == 1 else _HDR
    setpoints = {v for v in hold_count if hold_count[v] >= 2 or hold_maxlen[v] >= _MINHOLD}
    literal = sum(
        (j - i) for kind, i, j, p in toks if kind == "H" and p != 0 and p not in setpoints
    )
    palette = {palette_map(v) for v in setpoints} if palette_map else set(setpoints)
    active = s[s != 0]
    naive_bits = int(sum(_HDR + nbits(int(v)) for v in active))
    def_bits = sum(
        (nbits(sh[2]) if sh[0] == "D" else _HDR + sum(nbits(c) for c in sh[1])) for sh in shapes
    ) + sum(nbits(v) for v in setpoints)
    return {
        "naive_bits": naive_bits,
        "struct_bits": int(inst_bits + def_bits),
        "modDV": len(mod_vals),
        "modGEN": len(shapes),
        "palette": len(palette),
        "literal": int(literal),
    }
