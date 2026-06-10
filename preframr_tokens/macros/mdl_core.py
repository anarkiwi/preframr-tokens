"""MDL optimal-parse over the driver's primitive basis (ported from the validated mdl_parse/mdl_codec
prototypes). Recovering the gestures in a per-frame value series is the optimal-parsing problem (an
LZ-optimal / Knuth-Plass / Viterbi shortest-path DP on the description-length cost DAG); every edge is
exactly one of HOLD (constant), POLY(N) (forward-differenced polynomial, constant N-th difference), or
PERIOD (looped delta cell), so the cover is lossless by construction with a length-1 HOLD fallback.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

_HDR = 6
_MAXDEG = 3
_PMAX = 32


def nbits(x) -> int:
    """Elias-gamma-ish bit cost of a signed integer magnitude (``nbits(0) == 1``); the per-token cost
    model's value term, so low-degree / long-run / reused shapes win the shortest path.
    """
    x = int(x)
    return 1 if x == 0 else 2 + abs(x).bit_length()


def _wd(d, wrap):
    """One difference value reduced to the signed 16-bit range when ``wrap`` (freq is mod-65536), else
    passed through unchanged."""
    d = int(d)
    return ((d + 32768) % 65536) - 32768 if wrap else d


def _diffs(s, wrap):
    """Forward-difference arrays ``d[k]`` (k=0..MAXDEG); ``d[k][i]`` is the k-th difference of ``s`` at
    ``i``, reduced to signed 16-bit per level when ``wrap``."""
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
    """For each ``i`` the last frame of the maximal degree-``N`` run from ``i`` (constant N-th
    difference) and that difference value; a constant ``d[N]`` over ``[a, b)`` covers ``[a, b-1+N]``.
    """
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


def _hold_runs(s, n):
    """Exclusive end of the maximal constant run starting at each frame, in one O(n) backward pass, so
    the HOLD edge is O(1) per position instead of an O(run) rescan (the O(n^2) blowup on long silent
    channels) -- the rest of the DP already precomputes its run tables vectorised."""
    end = np.empty(n, dtype=np.int64)
    if n:
        end[n - 1] = n
    for i in range(n - 2, -1, -1):
        end[i] = end[i + 1] if s[i] == s[i + 1] else i + 1
    return end


def _period_edges(s, n, wrap=False):
    """Candidate PERIOD edges keyed by start frame: for each probed period ``p`` the maximal exact
    looped-cell spans, each as ``(end, cost, ("P", cell))`` with ``cell`` the ``p`` looped deltas
    (reduced to signed 16-bit when ``wrap`` so the cell fits the codec's 2-byte field; decode wraps the
    value level, so the looped sum is unchanged mod-65536).
    """
    edges = defaultdict(list)
    for p in range(2, min(_PMAX, n // 2) + 1):
        rep = (s[p:] == s[:-p]).astype(np.int8)
        d = np.diff(np.concatenate([[0], rep, [0]]))
        for a, e in zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()):
            if e - a < p:
                continue
            cell = tuple(
                _wd(int(x), wrap) for x in (s[a + 1 : a + p + 1] - s[a : a + p])
            )
            cost = _HDR + nbits(int(s[a])) + nbits(p) + sum(nbits(c) for c in cell)
            edges[a].append((e - 1 + p + 1, cost, ("P", cell)))
    return edges


def mdl_parse(series, wrap=False):
    """Globally optimal min-description-length parse of ``series`` into HOLD / POLY(N) / PERIOD tokens
    via shortest path on the position DAG. Returns ``[(kind, i, j, param)]`` with kind ``"H"``
    (param=value), ``"D"`` (param=(N, constant N-th diff)), or ``"P"`` (param=delta cell).
    """
    s = np.asarray(series, dtype=np.int64)
    n = len(s)
    if n == 0:
        return []
    diffs = _diffs(s, wrap)
    poly = [None] + [_poly_runs(diffs, N, n) for N in range(1, _MAXDEG + 1)]
    per_edges = _period_edges(s, n, wrap)
    hold_end = _hold_runs(s, n)
    inf = float("inf")
    cost = [inf] * (n + 1)
    cost[0] = 0.0
    back = [None] * (n + 1)

    def relax(a, b, c, tok):
        if c < cost[b]:
            cost[b] = c
            back[b] = (a, tok)

    for i in range(n):
        if cost[i] == inf:
            continue
        base = cost[i]
        hv = int(s[i])
        relax(i, int(hold_end[i]), base + _HDR + nbits(hv), ("H", hv))
        relax(i, i + 1, base + _HDR + nbits(hv), ("H", hv))
        for N in range(1, _MAXDEG + 1):
            if i + N >= n:
                break
            end, aval = poly[N]
            e = int(end[i])
            if e - i + 1 < N + 2:
                continue
            pcost = (
                _HDR + nbits(hv) + sum(nbits(int(diffs[k][i])) for k in range(1, N + 1))
            )
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


def difftable(s, i, N, wrap):
    """Initial forward-difference table ``[v0, d1, .., dN]`` of ``s`` at frame ``i`` (signed 16-bit per
    level when ``wrap``); ``v0`` and the lower diffs are per-instance, ``dN`` is the reusable shape.
    """
    cur = [int(s[i + m]) for m in range(N + 1)]
    dt = [cur[0]]
    for _ in range(N):
        cur = [_wd(cur[m + 1] - cur[m], wrap) for m in range(len(cur) - 1)]
        dt.append(cur[0])
    return dt
