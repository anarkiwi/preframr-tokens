"""Numba-JIT kernels for the shared backward-LZ with TRANSPOSE factoring.

The cover path's dominant cost (cProfile on a digi-dense tune: ~250s of a 300s
recovery) is :func:`serialize._lz_emit_t` over a long bundled-row stream -- the
backward search ``_best_transpose_indexed`` -> ``_transposed_run`` -> ``delta``
(``delta`` called 100M+ times), each call re-walking Python tuples-with-lists.

These kernels move that O(matches) inner search onto PRECOMPUTED integer arrays so
the per-step compare is a machine-int op, not a tuple ``__eq__`` / a per-lane
``isinstance`` loop.  They are PURE speedups: the plan a kernel emits is replayed by
the SAME ``_wu``/``_wi`` emitters, so the id stream is byte-for-byte identical to the
pure-Python reference (pinned by :mod:`tests.test_lz_njit` + the corpus-budget gate's
committed token counts).  When numba is absent the wrapper degrades to identity and
the Python reference path is used instead (``serialize._lz_emit_t`` keeps its
non-kernel branch), so the package never hard-fails on a numba-less interpreter.

The integer encoding (built ONCE per ``_lz_emit_t`` call, in
:func:`serialize._lz_emit_t`) reproduces the codec's equality + transpose contract:

  * ``eqid[i]``   -- ``eqid[a] == eqid[b]``  iff  ``items[a] == items[b]``
    (factorised ``eq_key``; the REPEAT match's necessary-and-sufficient condition).
  * ``xid[i]``    -- ``xid[a] == xid[b]``     iff  ``xpose_key(a) == xpose_key(b)``
    (factorised ``xpose_key``; a NECESSARY condition for ``delta`` to relate a, b).
  * ``posbase[i, l]`` -- the positive (transposable) base at lane ``l`` (or ``_NEG``
    when that lane is non-positive / a list); ``xelig[i]`` is 1 iff the row can take
    part in a transpose (no list base AND at least one positive lane -- exactly when
    ``delta`` can return a non-None shift).
  * ``litcost[i]`` -- ``lit_cost(items[i])`` (the per-item literal byte length), so the
    copy-vs-literal gain is a prefix-sum, never a re-serialisation.

Given ``xid[s] == xid[t]`` and both eligible, ``delta(items[s], items[t])`` equals the
single shift ``d`` such that every positive lane satisfies
``posbase[t, l] - posbase[s, l] == d`` (the negative / list lanes are already equal by
``xid``); the kernel computes ``d`` from the first positive lane and verifies the rest,
exactly mirroring ``serialize.py``'s ``delta``.
"""

import numpy as np

from preframr_tokens.bacc.generic._njit import njit

# Plan op codes (one row per cursor advance, replayed by the Python emitter).
OP_LIT = 0
OP_REPEAT = 1
OP_TRANSPOSE = 2

# Sentinel for a non-positive / list lane in ``posbase`` (a real base is >= 0).
_NEG = -1


@njit(cache=True)
def _ulen(n):
    """``len`` of the base-16 LEB encoding of a non-negative ``n`` (matches
    :func:`serialize._u_len`)."""
    c = 1
    n >>= 4
    while n:
        c += 1
        n >>= 4
    return c


@njit(cache=True)
def _ilen(n):
    """``len`` of the zig-zag base-16 LEB encoding of a signed ``n`` (matches
    :func:`serialize._wi_len`)."""
    z = (n << 1) ^ (n >> 63)
    return _ulen(z)


@njit(cache=True)
def _repeat_run(eqid, p, i, n_items):
    """Length of the exact backward run: longest ``n`` with ``eqid[p+n]==eqid[i+n]``
    (``eqid`` equality is item ``==`` by the ``eq_key`` contract)."""
    n = 0
    while i + n < n_items and eqid[p + n] == eqid[i + n]:
        n += 1
    return n


@njit(cache=True)
def _trans_run(xid, posbase, xelig, s0, t0, delta, n_items, nlane):
    """Length of the transposed backward run from source ``s0`` against ``t0`` with the
    fixed shift ``delta`` (mirrors :func:`serialize._transposed_run` composed with
    ``delta``): each step's rows must be transpose-eligible, share ``xid`` (so dur /
    refs / seeds / absolute bases match), and every positive lane must differ by exactly
    ``delta``."""
    n = 0
    while t0 + n < n_items:
        s = s0 + n
        t = t0 + n
        if xelig[s] == 0 or xelig[t] == 0 or xid[s] != xid[t]:
            break
        ok = True
        for l in range(nlane):
            bs = posbase[s, l]
            bt = posbase[t, l]
            if bs == _NEG or bt == _NEG:
                # a non-positive lane: equal by xid, contributes no shift constraint.
                continue
            if bt - bs != delta:
                ok = False
                break
        if not ok:
            break
        n += 1
    return n


@njit(cache=True)
def _pair_delta(xid, posbase, xelig, s, t, nlane):
    """The transpose shift relating rows ``s`` (source) and ``t`` (mirrors
    :func:`serialize`'s ``delta``): ``(found, d)`` where ``found`` is 0 when the rows do
    not transpose-relate (ineligible, mismatched ``xid``, or no positive lane / an
    inconsistent shift)."""
    if xelig[s] == 0 or xelig[t] == 0 or xid[s] != xid[t]:
        return 0, 0
    have = False
    d = 0
    for l in range(nlane):
        bs = posbase[s, l]
        bt = posbase[t, l]
        if bs == _NEG or bt == _NEG:
            continue
        dd = bt - bs
        if not have:
            d = dd
            have = True
        elif dd != d:
            return 0, 0
    if not have:
        return 0, 0
    return 1, d


@njit(cache=True)
def lz_plan_kernel(
    eqid, xid, posbase, xelig, litcost, prefix, nlane, use_xpose, min_copy
):
    """Reproduce :func:`serialize._lz_emit_t`'s per-position decision over the integer
    encoding and return a plan ``int64[m, 4]`` of ``(op, off, length, delta)`` rows --
    one per cursor advance.  ``prefix[k] = sum(litcost[:k])`` is the literal-cost prefix
    sum (so a copied range's literal cost is ``prefix[i+len]-prefix[i]``).

    Byte-identical to the dense reference: the equality / transpose indexes are walked
    DESCENDING (offsets ascending), keeping the first strict-maximum run -- the
    smallest-offset longest run the Python scan returns -- and the same copy-vs-transpose
    -vs-literal gain tiebreak (``use_trans and trans_gain >= copy_gain``)."""
    n = len(eqid)
    plan = np.empty((n, 4), dtype=np.int64)
    m = 0

    # Incremental equality / transpose position indexes, bucketed by a dense rank of the
    # id values (eqid / xid are pre-factorised to 0..K-1, so a bucket head/next linked
    # list over ranks is O(1) per probe -- no Python dict).  A position is appended only
    # after the cursor passes it (the prior-source window the dense scan walks).
    keqmax = 0
    for k in range(n):
        if eqid[k] + 1 > keqmax:
            keqmax = eqid[k] + 1
    eq_head = np.full(keqmax, -1, dtype=np.int64)  # most-recent pos with this eqid
    eq_prev = np.full(n, -1, dtype=np.int64)  # previous pos with the same eqid

    if use_xpose:
        kxmax = 0
        for k in range(n):
            if xid[k] + 1 > kxmax:
                kxmax = xid[k] + 1
        x_head = np.full(kxmax, -1, dtype=np.int64)
        x_prev = np.full(n, -1, dtype=np.int64)
    else:
        x_head = np.full(1, -1, dtype=np.int64)
        x_prev = np.full(1, -1, dtype=np.int64)

    indexed = 0
    i = 0
    while i < n:
        # index positions [indexed, i): each becomes the new head of its id bucket, so
        # walking head->prev visits prior positions MOST-RECENT-first == offsets
        # ASCENDING (the dense scan's ``off in range(1, i+1)`` order, first max kept).
        for j in range(indexed, i):
            e = eqid[j]
            eq_prev[j] = eq_head[e]
            eq_head[e] = j
            if use_xpose:
                xq = xid[j]
                x_prev[j] = x_head[xq]
                x_head[xq] = j
        indexed = i

        # best exact REPEAT (smallest-offset longest run over same-eqid prior positions)
        best_len = 0
        best_off = 0
        p = eq_head[eqid[i]]
        while p != -1:
            rl = _repeat_run(eqid, p, i, n)
            if rl > best_len:
                best_len = rl
                best_off = i - p
            p = eq_prev[p]

        cost_copy = 1 + _ulen(best_off) + _ulen(best_len)
        lit_copy = prefix[i + best_len] - prefix[i]
        use_copy = best_len >= min_copy and cost_copy < lit_copy
        copy_gain = lit_copy - cost_copy if use_copy else 0

        if use_xpose and xelig[i] == 1:
            tlen = 0
            toff = 0
            tdelta = 0
            q = x_head[xid[i]]
            while q != -1:
                found, d = _pair_delta(xid, posbase, xelig, q, i, nlane)
                if found and d != 0:
                    rl = _trans_run(xid, posbase, xelig, q, i, d, n, nlane)
                    if rl > tlen:
                        tlen = rl
                        toff = i - q
                        tdelta = d
                q = x_prev[q]
            cost_trans = 1 + _ulen(toff) + _ulen(tlen) + _ilen(tdelta)
            lit_trans = prefix[i + tlen] - prefix[i]
            use_trans = tlen >= min_copy and cost_trans < lit_trans
            trans_gain = lit_trans - cost_trans if use_trans else 0
            if use_trans and trans_gain >= copy_gain:
                plan[m, 0] = OP_TRANSPOSE
                plan[m, 1] = toff
                plan[m, 2] = tlen
                plan[m, 3] = tdelta
                m += 1
                i += tlen
                continue

        if use_copy:
            plan[m, 0] = OP_REPEAT
            plan[m, 1] = best_off
            plan[m, 2] = best_len
            plan[m, 3] = 0
            m += 1
            i += best_len
        else:
            plan[m, 0] = OP_LIT
            plan[m, 1] = i
            plan[m, 2] = 1
            plan[m, 3] = 0
            m += 1
            i += 1

    return plan[:m]


# Bucket count for the 3-gram hash linked list (a power of two; a token alphabet is
# tiny, so collisions are rare and a probe falls back to the literal value compare).
_TOK_BUCKETS = 1 << 16


@njit(cache=True)
def token_lz_plan_kernel(stream, min_copy):
    """Reproduce :func:`tracker_serialize._lz_tokens`'s hash-of-3-grams greedy
    backward-LZ over a flat integer token ``stream`` and return a plan ``int64[m, 3]``
    of ``(is_copy, off, length)`` rows -- one per cursor advance (a literal carries
    ``length == 1`` and the host re-emits ``stream[i]``).  Byte-identical to the Python
    reference: same 3-gram bucket walked MOST-RECENT-first, same ``length >= 512`` and
    ``length >= 4095`` caps, same ``cost_copy < best_len`` accept test, and the SAME
    incremental indexing of ``[i, i+step)`` capped at ``n - 2`` (so the last two
    positions seed no 3-gram, exactly as the reference's ``min(i+step, n-2)``)."""
    n = len(stream)
    plan = np.empty((n, 3), dtype=np.int64)
    m = 0
    head = np.full(_TOK_BUCKETS, -1, dtype=np.int64)  # bucket -> most recent start pos
    nxt = np.full(n if n > 0 else 1, -1, dtype=np.int64)  # pos -> previous same-bucket
    mask = _TOK_BUCKETS - 1
    i = 0
    while i < n:
        best_len = 0
        best_off = 0
        if i + 3 <= n:
            h = ((stream[i] * 1000003 + stream[i + 1]) * 1000003 + stream[i + 2]) & mask
            pos = head[h]
            while pos != -1:
                # confirm the 3-gram (the hash can collide); only then extend the run.
                if (
                    stream[pos] == stream[i]
                    and stream[pos + 1] == stream[i + 1]
                    and stream[pos + 2] == stream[i + 2]
                ):
                    length = 0
                    while i + length < n and stream[pos + length] == stream[i + length]:
                        length += 1
                        if length >= 4095:
                            break
                    if length > best_len:
                        best_len = length
                        best_off = i - pos
                    if best_len >= 512:
                        break
                pos = nxt[pos]
        cost_copy = 1 + _ulen(best_off) + _ulen(best_len)
        if best_len >= min_copy and cost_copy < best_len:
            plan[m, 0] = 1
            plan[m, 1] = best_off
            plan[m, 2] = best_len
            step = best_len
        else:
            plan[m, 0] = 0
            plan[m, 1] = i
            plan[m, 2] = 1
            step = 1
        m += 1
        hi = i + step
        if hi > n - 2:
            hi = n - 2
        for j in range(i, hi):
            h2 = (
                (stream[j] * 1000003 + stream[j + 1]) * 1000003 + stream[j + 2]
            ) & mask
            nxt[j] = head[h2]
            head[h2] = j
        i += step
    return plan[:m]


@njit(cache=True)
def backward_lz_counts_kernel(tokens, min_match, window):
    """Reproduce :func:`structure_recover._backward_lz`: the greedy O(n . window)
    backward-LZ that COUNTS ``(literals, matches)``.  At each cursor it finds the
    longest backward run in ``[i-window, i)`` (capped at 255), takes it as a match when
    ``>= min_match`` else emits a literal -- byte-identical counts to the Python loop.
    """
    n = len(tokens)
    literals = 0
    matches = 0
    i = 0
    while i < n:
        best = 0
        lo = i - window
        if lo < 0:
            lo = 0
        for j in range(lo, i):
            length = 0
            while (
                i + length < n
                and tokens[j + length] == tokens[i + length]
                and length < 255
            ):
                length += 1
            if length > best:
                best = length
        if best >= min_match:
            matches += 1
            i += best
        else:
            literals += 1
            i += 1
    return literals, matches
