"""Token-minimizing, byte-exact cover of a recovered SID register lane.

The per-position matcher in :mod:`archetypes` (:func:`_longest_archetype_aug` ->
:func:`_prefix_citg`) returns the LONGEST byte-exact generator run at a position.
A cover built by chaining "longest run wins" is byte-exact but NOT token-minimal:
where a compact ``accum`` (rate=-256, a 3-int struct) would reproduce a span, the
longest-run rule can instead return a 32-value ``tablewalk`` (a ~64-token table),
and a perfect ``+1 every 128 frames`` accumulator is fragmented into ~51 holds.

This module covers a lane to minimize the SERIALIZED cost
``stream + pool`` (the metric :mod:`tracker_serialize` actually emits):

  * the POOL holds each UNIQUE generator struct ONCE (after stripping the per-note
    SEED_KEYS so phase/base/seed ride the row), token-LZ'd across structs;
  * the STREAM is the per-segment rows ``(dt, dur, pool_ref, seed)``, backward-LZ'd.

So a generator's true marginal cost is ``struct_tokens / run_length`` (its pool
share amortized over the frames it covers) plus the per-row stream overhead -- a
compact generator that extends FAR, and especially one whose seed-stripped struct
RECURS (paid once in the pool, near-free per reuse), is strongly preferred.  A lane
that no compact recurrence covers is FLOORED to a §3.6 literal-table CITG of its own
bytes (:func:`archetypes.literal_table_citg`) -- byte-exact by construction and the
same ``citg`` vocabulary, the no-escape last resort.

:func:`cover_lane` covers one lane; :func:`cover_tokens` covers all 16 register
lanes of a ``(nframes, 25)`` state, dedups the seed-stripped structs into a pool,
LZ-measures each lane's row stream, and reports the token split.
"""

import json

import numpy as np

from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import _render_njit as _rnj
from preframr_tokens.bacc.generic.tracker import SEED_KEYS, _PITCH_LIST, _REL
from preframr_tokens.bacc.serialize import _wu
from preframr_tokens.bacc.tracker_serialize import (
    _flatten_pool,
    _lz_tokens,
    _write_value,
)

# A structured generator must beat a same-length chain of holds to be worth a pool
# entry; below this it is just stored data wearing a generator's clothes.
_MINRUN = 3
# The pool-amortization weight: a candidate's marginal pool cost is
# ``struct_tokens / run_length`` only for a NOVEL struct.  A struct already chosen
# elsewhere in the cover (the dominant case for an algorithmic tune -- the SAME
# accum/table recurs every phrase) is near-free, so it is costed at the per-reuse
# stream overhead alone.  This is what makes a compact generator that recurs win
# decisively over a long-but-unique table.
_ROW_OVERHEAD = 4  # ~dt+dur+ref+seed value-tokens a segment row costs in the stream

# A unique sentinel for "absent from cache" so a legitimately cached ``None`` result
# (the matcher declined this window) is distinguished from a miss without a re-run.
_MISS = object()


# ---------------------------------------------------------------------------
# Struct token cost + seed stripping (mirrors tracker_serialize / tracker.lift).
# ---------------------------------------------------------------------------
def _struct_tokens(struct):
    """The value-token length of a pool struct (``["S", [name, struct]]`` entry),
    exactly as :func:`tracker_serialize._write_value` would emit it."""
    out = []
    _write_value(out, struct)
    return len(out)


def _strip_seed(_name, params, is_freq, idx_of):
    """Split a fit ``(name, params)`` into ``(pool_struct, seed, base)`` the SAME
    way :func:`tracker._split_fit` does: the seed-and-pitch-invariant struct is the
    dedup key, the SEED_KEYS ride the row, and a freq table that lies entirely on
    the note-table grid is recoded to relative indices (pitch-invariant).  ``_name``
    is unused (the split is name-agnostic) but kept for call symmetry with the fit."""
    struct = {k: v for k, v in params.items() if k not in SEED_KEYS}
    seed = {k: params[k] for k in params if k in SEED_KEYS}
    base = -1
    if is_freq and idx_of is not None:
        for key in _PITCH_LIST:
            seq = struct.get(key)
            if isinstance(seq, list) and seq and not (seq and seq[0] == _REL):
                idxs = [idx_of.get(int(v)) for v in seq]
                if all(j is not None for j in idxs):
                    base = idxs[0]
                    struct[key] = [_REL] + [j - base for j in idxs]
    return struct, seed, base


# The struct-token cost of a pool entry depends only on its JSON key (a pure function
# of the seed-stripped struct), and the DP re-asks it for the SAME few structs at every
# position; memoizing by key turns the repeated _write_value recursion into a dict hit.
_STRUCT_TOK_MEMO = {}


def _struct_tokens_cached(key, entry):
    tok = _STRUCT_TOK_MEMO.get(key)
    if tok is None:
        tok = _struct_tokens(entry)
        if len(_STRUCT_TOK_MEMO) < 100000:
            _STRUCT_TOK_MEMO[key] = tok
    return tok


def _pool_key(name, params, is_freq, idx_of):
    """The dedup key (a JSON string of the ``["S", [name, struct]]`` pool entry)
    and its token cost, for a single fit."""
    struct, _seed, _base = _strip_seed(name, params, is_freq, idx_of)
    entry = ["S", [name, struct]]
    key = json.dumps(entry, sort_keys=True)
    return key, _struct_tokens_cached(key, entry)


def _rotatable_periodic(name, params):
    """The (table, period) of a periodic-walk CITG whose rotation is byte-exactly
    foldable into ``phase`` -- a full-period (``loop==0``) every-frame ``accum``/``read``
    with a >=2-element table -- else ``None``.  A partial loop (``loop != 0``) wraps to a
    fixed mid-table index, so a rotation would change which entries repeat; it is not
    rotatable."""
    if name != "citg" or not isinstance(params, dict):
        return None
    if params.get("mode") not in ("accum", "read"):
        return None
    if (params.get("clock") or {}).get("kind") != "every":
        return None
    if int(params.get("loop", 0)) != 0:
        return None
    table = params.get("table")
    if not isinstance(table, list) or len(table) < 2:
        return None
    return table, len(table)


def _canon_rotation(table):
    """The lexicographically-minimal rotation of ``table`` and the rotation index ``r``
    that produces it (``min_rot[j] = table[(j+r)%P]``)."""
    period = len(table)
    best_r, best_rot = 0, None
    for r in range(period):
        rot = table[r:] + table[:r]
        if best_rot is None or rot < best_rot:
            best_rot, best_r = rot, r
    return best_rot, best_r


def _merge_rotations(cover):
    """Collapse periodic-walk segments that are ROTATIONS of the same cycle to ONE
    canonical struct (the rotation rides the per-note ``phase`` seed) -- the dedup +
    REPEAT lever -- but ONLY for a cycle that actually occurs at >=2 distinct rotations
    in this lane.

    Where the cover slices a recurring vibrato / arp / ratewalk cell, each occurrence
    starts at a different point in its cycle, so the matcher returns the table ROTATED
    to that start (one note's ``[-10,-10,10,10,10,-10]`` is another's
    ``[10,10,10,-10,-10,-10]`` -- the SAME triangle).  Stored as distinct structs they
    defeat the pool dedup AND the segment-level REPEAT.  Rotating every occurrence to the
    canonical table and carrying the rotation as ``phase`` makes them share ONE struct.
    Restricting this to cycles seen at >=2 rotations keeps a singleton periodic generator
    UNTOUCHED -- so a tune whose periodic structs already dedup (no phase needed) is not
    charged a per-row ``phase`` token it cannot amortize.  Byte-EXACT: with the rotated
    table and ``phase' = (phase - r) % P`` the render is identical at every frame
    (verified by the serializer render-equality self-check; HARD RULE #0)."""
    # First pass: for each rotatable cycle (its canonical key), the set of rotations seen.
    canon_rots = {}
    for _s, _t, fit in cover:
        rp = _rotatable_periodic(*fit)
        if rp is None:
            continue
        table, _period = rp
        canon, r = _canon_rotation(table)
        canon_rots.setdefault(tuple(canon), set()).add(r)
    multi = {k for k, rs in canon_rots.items() if len(rs) >= 2}
    if not multi:
        return cover
    out = []
    for s, t, fit in cover:
        rp = _rotatable_periodic(*fit)
        if rp is not None:
            table, period = rp
            canon, r = _canon_rotation(table)
            if tuple(canon) in multi:
                name, params = fit
                new = dict(params)
                new["table"] = canon
                new["phase"] = (int(params.get("phase", 0)) - r) % period
                fit = (name, new)
        out.append((s, t, fit))
    return out


# ---------------------------------------------------------------------------
# Candidate generators at a position (reuse the proven matchers, EXTEND each).
# ---------------------------------------------------------------------------
def _extend(name, params, lane, start, note_table, carry, _width, cap=None):
    """The maximal byte-exact run length of generator ``(name, params)`` from
    ``start`` over the remaining lane, so a compact generator is credited for every
    frame it truly covers -- the per-frame accumulator / periodic table that runs for
    hundreds of frames is one piece, not a window-capped stub.

    Rendered in a doubling window (``cap`` start, grown only while the render still
    matches to the window's end) so the common SHORT run costs O(run) rather than
    O(remaining): a 4-frame accum at frame 10 of a 3000-frame lane renders ~8 frames,
    not 2990 -- the difference between an O(n) and an O(n^2) per-lane cover."""
    remain = len(lane) - start
    cseq = carry[start:] if carry is not None else None
    seg = lane[start:]
    win = cap if cap is not None else 64
    while True:
        win = min(win, remain)
        rend = A.render_fit((name, params), win, note_table, cseq, 0)
        m = A._match_prefix(rend, seg[:win])
        if m < win or win >= remain:
            return m
        win *= 2


# A candidate whose seed-stripped struct embeds a per-frame array (the ``advance``
# clock of a ``wavetable_ptr``, or a literal ``table`` as long as the run) is stored
# data wearing a generator's clothes -- it never dedups and bloats the pool.  A
# genuine reusable generator's struct is SMALL relative to the frames it covers; we
# admit a novel structured candidate only when its embedded payload is at most this
# fraction of its run, and otherwise let the lane fall to hold-runs + the §3.6
# literal-table floor (which is itself the canonical, measurable stored-data form).
_MAX_PAYLOAD_FRAC = 0.34


def _payload_len(params):
    """The number of stored array elements inside a CITG struct -- the table plus
    any embedded clock array (advance / mask).  The reusable closed-form generators
    (accum/arp/short tablewalk) have a tiny payload; the per-frame-data forms
    (a ``wavetable_ptr`` advance clock, a literal-table floor) have a payload as
    long as the run."""
    n = 0
    table = params.get("table")
    if isinstance(table, list):
        n += len(table)
    clock = params.get("clock")
    if isinstance(clock, dict):
        for k in ("advance", "mask"):
            arr = clock.get(k)
            if isinstance(arr, list):
                n += len(arr)
    for k in ("freqs", "rate_table", "carry_table"):
        arr = params.get(k)
        if isinstance(arr, list):
            n += len(arr)
    return n


def _longdwell_accum(seg, width):
    """A constant-rate accumulator stepped on a UNIFORM long dwell -- the compact
    ``+r every D frames`` generator (a ~6-int CITG ``accum`` with a ``dwell`` clock)
    the existing matchers miss because their dwell / mask period is capped small
    (``_longest_dwell_accum`` dwell<=8, ``_prefix_maskaccum`` period<=12).  This is
    the prompt's ``+1 every 128 frames`` filter-cutoff lane: recovered here as ONE
    period-spanning piece instead of dozens of distinct holds.

    Returns ``(name, params)`` or None.  The dwell clock fires on frame ``D-1`` (the
    first step), so a lane whose first change is at index ``D-1`` and whose nonzero
    deltas are a single ``rate`` on a uniform gap ``D`` is exactly this generator;
    the caller re-renders and verifies byte-exact, so a near-miss is rejected."""
    seg = np.asarray(seg, dtype=np.int64)
    n = len(seg)
    if n < 4:
        return None
    diff = np.diff(seg)
    change = np.nonzero(diff)[0]  # diff[k]!=0 -> change between k and k+1
    if len(change) < 2:
        return None
    rates = diff[change]
    if len(np.unique(rates)) != 1:
        return None
    rate = int(rates[0])
    if rate == 0:
        return None
    firsts = (change + 1).tolist()  # frame indices of each new value
    gaps = np.diff(firsts)
    if len(np.unique(gaps)) != 1:
        return None
    dwell = int(gaps[0])
    if dwell < 1:
        return None
    w = int(width) if width is not None else 0xFFFF
    # Verify over a bounded window (a few dwell periods prove the clock); the candidate
    # is re-extended to its true length by :func:`_extend` afterward, so a short verify
    # window keeps the pure-Python CITG dwell render cheap.
    vn = min(n, max(4 * dwell + firsts[0] + 4, 64))
    # The dwell clock fires at frame ``dwell-1`` (emitting the first new value at
    # ``dwell``); a lead-stall shifts that first fire.  Solve the lead so the first
    # new value lands at ``firsts[0]``, then re-render and accept only on a byte-exact
    # cover (no phase guesswork survives the check).
    lead = firsts[0] - dwell
    params = {
        "mode": "accum",
        "table": [rate],
        "clock": {"kind": "dwell", "dwell": dwell},
        "seed": int(seg[0]),
        "width": w,
    }
    if lead > 0:
        params["lead"] = lead
    if A._match_prefix(A.render_citg(params, vn), seg[:vn]) >= vn:
        return ("citg", params)
    # Fallback: scan a few small leads in case the first-fire convention differs.
    for lead in range(0, min(dwell, 8)):
        p = dict(params)
        if lead:
            p["lead"] = lead
        else:
            p.pop("lead", None)
        if A._match_prefix(A.render_citg(p, vn), seg[:vn]) >= vn:
            return ("citg", p)
    return None


def _hold_len_array(lane):
    """``hold_len[i]`` = the length of the constant run at ``i`` for the whole lane, in
    ONE vectorised O(n) pass -- so the DP's per-position ``hold`` length is an O(1)
    lookup instead of an O(run) rescan at every frame (the per-frame rescan was the
    cover's single largest cost: a long held lane is O(n^2))."""
    lane = np.asarray(lane, dtype=np.int64)
    n = len(lane)
    if n == 0:
        return np.ones(0, dtype=np.int64)
    # Each maximal constant run [s, e) has hold_len[s+k] = e - (s+k).  Find run ends
    # (a change or the lane end), broadcast each position's run-end, subtract index.
    idx = np.arange(n, dtype=np.int64)
    change = np.empty(n, dtype=bool)
    change[-1] = True
    np.not_equal(lane[1:], lane[:-1], out=change[:-1])
    # next_change[i] = the nearest run-end index >= i (a change frame or the lane end);
    # the constant run at i then has length next_change[i] - i + 1.
    cand = np.where(change, idx, n)
    next_change = np.minimum.accumulate(cand[::-1])[::-1]
    return next_change - idx + 1


def _hold_run(lane, start, cache=None):
    """The length of the constant run at ``start`` (the natural ``hold`` length).  Uses
    the lane's precomputed ``hold_len`` array (cached under ``"_hold"`` in ``cache``) so
    the lookup is O(1); falls back to a direct scan when no cache is threaded."""
    if cache is not None:
        arr = cache.get("_hold")
        if arr is None:
            arr = _hold_len_array(lane)
            cache["_hold"] = arr
        return int(arr[start])
    n = len(lane)
    j = start + 1
    v = lane[start]
    while j < n and lane[j] == v:
        j += 1
    return j - start


def _periodic_candidates(seg, width):
    """Candidate long-period periodic generators at the start of ``seg``: the value
    ``tablewalk`` ``out[i]=table[i%P]`` and the signed-rate ``ratewalk``
    ``val+=rate[i%P]``, for the smallest period ``P`` whose loop covers a substantial
    run.  Only periods up to a cap are tried (a longer period than this is just stored
    data); the payload-vs-run filter in :func:`_candidates` then keeps a period only
    when it is small relative to the frames the loop actually covers, so a genuine
    macro-loop is admitted and a one-shot table is rejected."""
    seg = np.asarray(seg, dtype=np.int64)
    n = len(seg)
    if n < 16:
        return []
    w = int(width) if width is not None else 0xFFFF
    maxp = min(256, n // 2)
    # Detect the period over a bounded window (a genuine loop is visible within a few
    # cycles), then EXTEND the winning loop over the full remaining segment -- so the
    # O(maxp * window) period scan is bounded regardless of lane length.
    # A periodic loop is admitted only when its body repeats for at least this many
    # cycles within the window -- a 2-cycle "loop" is usually a coincidence whose
    # unique table bloats the pool, where a genuine macro-loop (the prompt's tune is a
    # period-128 ratewalk for 1024 frames = 8 cycles) recurs many times.
    mincyc = 4
    win = min(n, mincyc * maxp + 16)
    sw = seg[:win]
    out = []
    # The signed-rate ratewalk (ACCUM) is tried FIRST and preferred: its table holds
    # signed STEPS (small ints), far cheaper than a value tablewalk's absolute values
    # for the same period.  The period of the ratewalk IS the period of the per-frame
    # DIFF, so detect it by a direct array comparison (``diff[P:] == diff[:-P]`` over the
    # window) -- O(maxp) cheap array ops -- instead of RENDERING a ratewalk per candidate
    # period (the former O(maxp) renders per breakpoint, the cover's hot loop).  The full
    # matcher (:func:`archetypes._prefix_ratewalk`) already covers small periods, so only
    # the LONG macro-loop periods this finds add anything; one render verifies the winner.
    diffw = (np.diff(sw).astype(np.int64)) % (w + 1)
    m = len(diffw)
    pmax = min(maxp, m // mincyc)
    # The smallest period whose diff body repeats for ``mincyc`` cycles (``diffw[i] ==
    # diffw[i-P]`` over the first ``mincyc*P`` diffs).  The ``for P: np.array_equal(...)``
    # scan -- up to ``pmax`` whole-array compares per breakpoint -- was the cover's
    # dominant remaining ``array_equal`` cost; the fused njit detector
    # :func:`_render_njit.periodic_diff_period` does the SAME ascending-``P`` prefix
    # comparison element-wise with a first-mismatch short-circuit, returning the SAME
    # first qualifying ``P`` (``-1`` when none), and one render below still verifies it.
    diffw_c = diffw if diffw.flags["C_CONTIGUOUS"] else np.ascontiguousarray(diffw)
    best_P = _rnj.periodic_diff_period(diffw_c, int(pmax), int(mincyc))
    if best_P < 0:
        best_P = None
    if best_P is not None:
        P = best_P
        rt0 = diffw[:P]
        run = A._match_prefix(
            A.render_ratewalk(win, int(sw[0]), rt0.tolist(), 0, w), sw
        )
        if run >= mincyc * P:
            diff = (np.diff(seg).astype(np.int64)) % (w + 1)
            # Re-center the rates into signed form so the table is small ints.
            rt = diff[:P].copy()
            rt[rt > w // 2] -= w + 1
            out.append(
                (
                    "citg",
                    {
                        "mode": "accum",
                        "table": rt.tolist(),
                        "clock": {"kind": "every"},
                        "seed": int(seg[0]),
                        "width": w,
                    },
                )
            )
    return out


# The matcher inspects at most this many frames at a breakpoint: its longest single
# byte-exact run never needs more (a longer recurrence is re-extended by _extend), and
# capping the input bounds the cost of its length-proportional sub-searches (the
# vibrato render sweep) so a trivial 3000-frame lane is not charged a 3000-frame
# vibrato scan at every breakpoint.
_MATCH_WINDOW = 512


def _citg_cached(lane, start, note_table, carry, width, cache):
    """:func:`archetypes._prefix_citg` at ``start`` (over a capped window), memoized by
    window CONTENT (the expensive matcher -- its vibrato / glide / composite sub-searches
    dominate the cover runtime -- is a pure function of its inputs, so it is computed
    once per distinct window).  The window cap bounds the matcher's length-proportional
    cost; a recurrence longer than the window is recovered by re-extending the candidate
    (:func:`_extend`).

    The matcher is invoked with a POSITION-INDEPENDENT phase (``ctr0=0``), not
    ``start & 0xFF``: the canonical cover must choose the SAME segment for the SAME
    value-subsequence wherever it occurs (so a repeated phrase tiles identically and
    the serializer's REPEAT/TRANSPOSE fires), which it cannot do if the matcher's
    assumed phase depends on the absolute frame.  A generator whose true phase is
    non-zero still recovers byte-exact -- its phase is carried in the per-note seed,
    and ``_consider`` re-renders + ``_extend`` re-verifies, so a wrong assumed phase
    only ever shortens a match, never corrupts one.

    Because that phase is position-independent and ``note_table`` / ``width`` are
    constant for a whole ``cover_lane`` call, the matcher's result depends ONLY on the
    window bytes and the (additive_pw) carry-slice bytes.  An algorithmic / periodic
    tune visits the SAME 512-frame window at many distinct positions (its lanes loop),
    so keying the cache by the window+carry CONTENT -- not by ``start`` -- collapses
    those duplicate positions to one matcher call (on A_Mind ~70% of the breakpoint
    matcher calls are exact-duplicate windows).  Byte-exact and token-identical: the
    matcher is a pure function, and every consumer (:func:`_strip_seed` /
    :func:`_merge_rotations` / :func:`_pool_key`) copies the ``params`` dict before
    mutating it, so a window's cached result is never aliased into a later edit.  A
    fast ``start``-keyed alias is layered on top so the DP's refinement passes (which
    revisit the SAME positions) skip even the content-hash."""
    if cache is not None:
        pos = cache.get(start, _MISS)
        if pos is not _MISS:
            return pos
    end = min(len(lane), start + _MATCH_WINDOW)
    cseg = carry[start:end] if carry is not None else None
    if cache is None:
        return A._prefix_citg(lane[start:end], note_table, width, 0, cseg)
    ckey = (
        "_citg",
        lane[start:end].tobytes(),
        cseg.tobytes() if cseg is not None else None,
    )
    res = cache.get(ckey, _MISS)
    if res is _MISS:
        res = A._prefix_citg(lane[start:end], note_table, width, 0, cseg)
        cache[ckey] = res
    cache[start] = res
    return res


def _candidates(lane, start, note_table, carry, width, full=True, cache=None):
    """The extended candidate list at ``start``, MEMOIZED across the DP passes.

    The candidate list ``[(name, params, run), ...]`` at a position is PASS-INVARIANT:
    it depends only on ``(lane, start, note_table, carry, width, full)`` -- NOT on the
    DP's ``amort`` cost relaxation (which only re-weights the *cost* of an edge in
    :func:`_cover_dp`, never which candidates exist or how far they extend).  The
    :func:`cover_lane` DP runs up to three times (the initial cover + two refinement
    passes); recomputing the expensive ``_extend`` / ``_prefix_citg`` /
    ``_longdwell_accum`` / ``_periodic_candidates`` work at every position on every pass
    is the cover's dominant redundant cost.  Keying the finished list by ``start`` in the
    shared ``cache`` (``full`` is itself a deterministic function of ``start`` within one
    ``cover_lane`` -- ``start in full_at`` -- so the position alone identifies the list)
    collapses that to ONCE per position for the whole ``cover_lane`` call.  Byte-exact
    and token-identical: the SAME candidate tuples, merely not rebuilt (the returned
    ``params`` dicts are never mutated by the DP -- :func:`_pool_key` /
    :func:`_merge_rotations` copy before reading)."""
    if cache is not None:
        ck = ("_cands", start)
        cached = cache.get(ck)
        if cached is not None:
            return cached
    out = _candidates_uncached(lane, start, note_table, carry, width, full, cache)
    if cache is not None:
        cache[("_cands", start)] = out
    return out


def _candidates_uncached(lane, start, note_table, carry, width, full=True, cache=None):
    """Yield ``(name, params, run)`` candidate generators that byte-exactly cover at
    least ``_MINRUN`` frames from ``start`` (the ``hold`` excepted), each already
    EXTENDED to its maximal byte-exact run length.

    The CHEAP candidates -- the constant ``hold`` (extended to its full constant
    run), the every-frame ``accum`` and the long-dwell ``accum`` -- are always
    produced (O(1) synthesis).  The EXPENSIVE unified matcher
    (:func:`archetypes._prefix_citg`, which enumerates every CITG clock/table-shape
    and returns the longest byte-exact prefix) plus the ``wrapaccum`` synthesizer run
    only when ``full`` is set (at the DP's sparse breakpoints), so the per-frame DP
    stays affordable while the structured forms are still offered where a segment can
    begin.  A stored-data candidate (a ``wavetable_ptr`` advance clock, a literal
    table) is DROPPED when its payload is large relative to its run
    (:data:`_MAX_PAYLOAD_FRAC`); such a span is covered by the compact pieces plus the
    literal-table floor so the pool never silently absorbs per-frame data."""
    seg = lane[start:]
    length = len(seg)
    raw = []

    # The compact every-frame constant-rate accumulator (the 3-int ``accum``): the
    # single cheapest structured generator and the one the longest-run rule most
    # often shadows with a bigger table.
    if length >= _MINRUN:
        delta = int(seg[1]) - int(seg[0])
        if delta != 0:
            raw.append(
                (
                    "citg",
                    {
                        "mode": "accum",
                        "table": [int(delta)],
                        "clock": {"kind": "every"},
                        "seed": int(seg[0]),
                        "width": int(width) if width is not None else 0xFFFF,
                    },
                )
            )

    # The constant baseline (``hold``): a degenerate period-1 read whose held value
    # is a SEED (the ``hold`` archetype, ``{"value": v}``), so EVERY hold -- at every
    # pitch -- strips to the SAME empty struct and the pool carries ONE hold entry,
    # the value riding the row.  Offered so a long flat run (or a held melody note)
    # is one shared-struct hold, never a per-value literal table.  This is the cover's
    # main lever on a melodic generator lane: a phrase of distinct held notes becomes
    # many rows of ONE hold instrument, not many distinct tables.
    raw.append(("hold", {"value": int(seg[0])}))

    if full:
        # The compact constant-rate accumulator stepped on a UNIFORM long dwell (the
        # ``+r every D frames`` generator the small-period matchers miss -- the
        # prompt's ``+1 every 128 frames`` cutoff lane).  Its dwell-clock render is the
        # pure-Python CITG loop, so it is restricted to the sparse breakpoints.
        ld = _longdwell_accum(seg, width)
        if ld is not None:
            raw.append(ld)

        # A free-running modulo accumulator (``wrapaccum``): the sawtooth PWM that
        # wraps by a span, a 4-int struct covering an unbounded sweep in one piece.
        wrap = A._prefix_wrapaccum(seg)
        if wrap is not None:
            p = A.citg_preset(wrap[1], wrap[2])
            if p is not None:
                raw.append(("citg", p))

        # A LONG-PERIOD periodic generator -- a value ``tablewalk`` or a signed-rate
        # ``ratewalk`` whose period P is far smaller than the frames it covers (so it
        # is a genuine compact LOOP, not stored data).  The unified matcher caps its
        # table period small (arp<=6, tablewalk<=48, ratewalk<=48); an algorithmic
        # tune's macro-loop (the prompt's tune is period-256 in places) exceeds that,
        # so the period is detected directly and the maximal-extent loop offered.  One
        # such piece covers a periodic block in ONE row instead of dozens of accum
        # rows -- the dominant stream lever once the pool is cheap.
        for name_p, params_p in _periodic_candidates(seg, width):
            raw.append((name_p, params_p))

        # The unified matcher's single best (longest) byte-exact prefix -- the
        # richest candidate, covering every clock/table-shape the CITG admits (arp /
        # tablewalk / ratewalk / dwell / vibrato / glide / ...).  Dropped below if it
        # is a stored per-frame form (advance clock) whose payload is large vs run.
        citg = _citg_cached(lane, start, note_table, carry, width, cache)
        if citg is not None:
            raw.append(("citg", citg[2]))

    out = []
    for name, params in raw:
        if name == "hold":
            run = _hold_run(lane, start, cache)
        else:
            run = _extend(name, params, lane, start, note_table, carry, width)
        if run <= 0:
            continue
        is_hold = name == "hold"
        if not is_hold and run < _MINRUN:
            continue
        # Reject a novel stored-data candidate (payload large vs run); the hold and
        # the genuine compact generators always pass.  A LOOP whose table fits at
        # least twice within its run (period <= run/2) is a genuine periodic generator
        # -- the table IS reused every cycle -- so it is exempt from the payload cap;
        # only a one-shot table (covers <2 cycles) is the stored-data form the cap
        # rejects.
        payload = _payload_len(params)
        is_loop = payload > 0 and run >= 2 * payload
        if not is_hold and not is_loop and payload > _MAX_PAYLOAD_FRAC * run:
            continue
        out.append((name, params, run))
    return out


# ---------------------------------------------------------------------------
# Token-minimizing cover of one lane: a recurrence-aware shortest-path DP, with a
# CANONICAL per-position segmentation tie-break so a repeated value-subsequence tiles
# identically (the segment-level REPEAT/TRANSPOSE lever) and a sparse breakpoint set +
# memoized matcher (the perf lever).
# ---------------------------------------------------------------------------
def _force_array(force_breakpoints, n):
    """Per-position ``next_force[i]`` = the smallest forced breakpoint STRICTLY greater
    than ``i`` (or ``n`` when none) -- the cap a segment starting at ``i`` may not
    cross.  ``None``/empty disables forcing (``next_force[i] == n`` everywhere, so no
    edge is clamped)."""
    nxt = np.full(n + 1, n, dtype=np.int64)
    if not force_breakpoints:
        return nxt
    for p in sorted(force_breakpoints, reverse=True):
        if 0 < p < n:  # a boundary at 0 or n is implicit (the lane edges)
            nxt[:p] = p
    return nxt


def _candidate_keys(start, cands, is_freq, idx_of, cache):
    """The per-candidate ``(pool_key, key_tok)`` list for position ``start``, aligned
    1:1 with ``cands`` and MEMOIZED across the DP passes (keyed ``("_keys", start)`` in
    the shared ``cache``).

    The pool key + token cost is a pure function of ``(name, params, is_freq, idx_of)``,
    and ``is_freq`` / ``idx_of`` are constant for a whole ``cover_lane`` call, so this is
    pass-invariant -- exactly like the candidate list itself.  Computing it once (the
    ``json.dumps`` keying in :func:`_pool_key`, the dominant per-edge cost after the
    renders are njit'd) and reusing it on the refinement passes removes ~2/3 of the
    ``_pool_key`` calls.  Token-identical: the SAME ``(key, key_tok)`` the DP would
    recompute, merely cached."""
    if cache is not None:
        kk = ("_keys", start)
        cached = cache.get(kk)
        if cached is not None:
            return cached
    out = [_pool_key(name, params, is_freq, idx_of) for name, params, _run in cands]
    if cache is not None:
        cache[("_keys", start)] = out
    return out


def _cover_dp(
    lane,
    width,
    note_table,
    carry,
    idx_of,
    is_freq,
    amort,
    full_at,
    cache,
    force_at=None,
):
    """Minimum-cost byte-exact cover of ``lane`` by a left-to-right shortest-path DP.

    ``cost[i]`` is the least serialized cost to cover ``lane[i:]``.  At each position
    the extended candidates (:func:`_candidates`) plus the always-available ``hold``
    each define an edge of length ``run`` and marginal cost ``row_overhead +
    pool_share`` -- the per-row stream cost plus the struct's amortized pool share
    (``amort[key]``, or its full token cost when not yet amortized).  The DP relaxes
    ``cost[i] = min over edges (edge_cost + cost[i+run])`` from the back, so a short
    cheap edge that ENABLES a long cheap edge later (a 1-frame hold that lets a
    3000-frame hold follow) is chosen over a locally-longer but expensive structured
    edge -- the look-ahead a greedy per-position rule cannot do.

    A TIE in total cost is broken by a POSITION-INDEPENDENT key (longer run, then the
    compact closed-form family ``_family_of`` / ``_FAMILY_RANK``, then the struct JSON),
    so two identical value-subsequences pick the SAME edge wherever the future cost ties,
    which is what lets a repeated phrase tile consistently and the serializer's
    segment-level REPEAT/TRANSPOSE collapse it.

    ``amort`` (built from pass 1's use counts) maps a struct key to the per-USE pool
    share it should be charged -- ``key_tok / uses`` -- so a struct's one-time pool
    cost is spread across every segment that references it: a compact generator reused
    20 times is nearly free per use, while a big table used twice still costs half its
    tokens each time (so it only wins if it genuinely saves more stream than that).
    A key absent from ``amort`` (pass 1, or a struct pass 1 never chose) is charged its
    FULL pool cost -- conservative, never under-charged, so the cover never inflates the
    pool on a mis-estimate (HARD RULE #0: byte-exact regardless of the cost estimate).
    """
    lane = np.asarray(lane, dtype=np.int64)
    n = len(lane)
    if force_at is None:
        force_at = _force_array(None, n)
    INF = float("inf")
    cost = [INF] * (n + 1)
    choice = [None] * n  # (run, name, params)
    cost[n] = 0.0
    # The EXPENSIVE unified matcher is invoked only at the sparse breakpoints
    # ``full_at`` (precomputed once and shared across both DP passes), and its result
    # at each breakpoint is memoized in ``cache`` so the two passes never recompute it.
    for i in range(n - 1, -1, -1):
        cands = _candidates(
            lane, i, note_table, carry, width, full=(i in full_at), cache=cache
        )
        # The pool KEY + token cost of each candidate at ``i`` is pass-invariant
        # (``is_freq`` / ``idx_of`` are constant for a ``cover_lane`` call), so it is
        # computed ONCE per position and cached alongside the candidate list -- the DP's
        # second/third passes reuse it instead of re-running the ``json.dumps`` keying
        # (the dominant per-edge cost once the renders are njit'd).
        keys = _candidate_keys(i, cands, is_freq, idx_of, cache)
        # A forced breakpoint caps how far an edge from ``i`` may reach: clamp every
        # candidate run to the next forced boundary so no segment straddles it.  A
        # clamped run is still byte-exact (a prefix of a byte-exact render), so this only
        # constrains the segmentation, never the bytes.
        cap = int(force_at[i])
        best = INF
        best_tb = None
        best_choice = None
        for (name, params, run), (key, key_tok) in zip(cands, keys):
            run = min(run, cap - i)
            if run <= 0:
                continue
            pool_share = amort.get(key, key_tok)  # full cost if not yet amortized
            edge = _ROW_OVERHEAD + pool_share
            tot = edge + cost[i + run]
            # A position-independent tie-break makes the SAME value-subsequence segment
            # identically wherever its future cost ties (the REPEAT/TRANSPOSE lever).
            tb = (-run, _FAMILY_RANK.get(_family_of(name, params), 1), key)
            if tot < best - 1e-9 or (abs(tot - best) <= 1e-9 and tb < best_tb):
                best = tot
                best_tb = tb
                best_choice = (run, name, params)
        if best_choice is None:  # always feasible: a unit hold
            best_choice = (1, "hold", {"value": int(lane[i])})
            best = _ROW_OVERHEAD + cost[i + 1]
        cost[i] = best
        choice[i] = best_choice
    # Walk the chosen edges into a segment list.
    segs = []
    i = 0
    while i < n:
        run, name, params = choice[i]
        segs.append((i, i + run, (name, params)))
        i += run
    return _floor_runs(lane, segs, width, is_freq, idx_of, force_at)


# A fixed family priority breaking a cost TIE -- it never overrides cost, only orders
# equal-cost edges so the choice is deterministic AND prefers the truer closed-form
# generator (a compact recurring accum/arp over a coincidental table).  Lower wins.
_FAMILY_RANK = {"accum": 0, "read": 1, "hold": 2}


def _family_of(name, params):
    """A coarse family tag for the deterministic tie-break: the CITG ``mode`` (so an
    ``accum`` ramp out-ranks a ``read`` table on a tie), else the fit name."""
    if name == "citg" and isinstance(params, dict):
        return params.get("mode", "citg")
    return name


def _breakpoints(lane, width, note_table, carry, _idx_of, _is_freq, cache):
    """The sparse set of frame indices at which the DP runs the EXPENSIVE unified
    matcher: the value-change frames (a structured generator can only usefully start
    where the lane's value or slope shifts) UNION the segment starts a fast greedy
    pre-pass (the original longest-run :func:`archetypes.fit_segment` style) placed.
    Everywhere else the cheap hold/accum candidates suffice.  Bounded by the number
    of changes/segments, not the frame count, so a long sweep lane stays affordable.
    The matcher result at each greedy start is memoized into ``cache`` so the DP
    passes reuse it.
    """
    lane = np.asarray(lane, dtype=np.int64)
    n = len(lane)
    pts = {0}
    # The value-change frames are full-matcher breakpoints ONLY when they are sparse
    # (a held/melodic lane): on a dense sweep lane (a change almost every frame) the
    # cheap accum/long-dwell candidates already cover the structure, and running the
    # full matcher at every frame would be quadratic for no gain.
    chg = np.nonzero(np.diff(lane))[0] + 1
    if len(chg) <= n // 2:
        pts.update(int(c) for c in chg.tolist())
    # The fast greedy pre-pass starts (the original longest-run cover): the matcher
    # advances by its longest run, so these are sparse even on a dense lane and pin
    # the few places a long structured generator genuinely begins.
    i = 0
    guard = 0
    while i < n and guard < 4 * n:
        guard += 1
        pts.add(i)
        citg = _citg_cached(lane, i, note_table, carry, width, cache)
        run = None
        if citg is not None:
            run = _extend("citg", citg[2], lane, i, note_table, carry, width)
        if not run or run < 1:
            run = _hold_run(lane, i, cache)
        i += max(1, run)
    return pts


def cover_lane(
    lane,
    width=0xFFFF,
    note_table=None,
    carry=None,
    idx_of=None,
    is_freq=False,
    force_breakpoints=None,
):
    """A byte-exact, token-MINIMIZING cover of one register lane.

    Returns ``[(start, stop, (name, params)), ...]`` whose rendered concatenation
    equals ``lane`` exactly.  A recurrence-aware shortest-path DP (:func:`_cover_dp`)
    minimizes the serialized ``stream + pool`` cost with look-ahead, run TWICE: pass 1
    discovers which seed-stripped structs RECUR (the compact generators a phrase
    re-uses); pass 2 re-runs the DP treating every recurring struct as already paid for
    in the pool, so a compact generator that appears many times wins decisively over a
    long-but-unique table even where the table covers more frames per piece.  A
    position-independent tie-break (the ``(-run, family, key)`` key in :func:`_cover_dp`)
    makes the SAME value-subsequence segment identically wherever it recurs, so a
    repeated phrase collapses under the serializer's segment-level REPEAT/TRANSPOSE.  An
    irreducible span falls to the §3.6 literal-table floor.

    ``force_breakpoints`` is an optional set of frame indices at which a segment
    boundary is MANDATORY: no chosen segment straddles one, so every segment is
    contained inside one inter-breakpoint span.  The Tracker IR uses this to RE-SLICE a
    sibling lane (pw/ctrl/ad/sr) at the freq-note onsets (the spine), so the sibling's
    per-note segments align to the spine and fold into the voice's bundled rows.
    Forcing a boundary only ever SHORTENS an edge -- every prefix of a byte-exact
    generator render is itself byte-exact -- so the cover stays byte-exact."""
    lane = np.asarray(lane, dtype=np.int64)
    if len(lane) == 0:
        return []
    force_at = _force_array(force_breakpoints, len(lane))
    # One matcher cache + breakpoint set, shared across all DP passes (the expensive
    # unified matcher then runs at most once per breakpoint for the whole cover).
    cache = {}
    full_at = _breakpoints(lane, width, note_table, carry, idx_of, is_freq, cache)
    if force_breakpoints:
        # A forced boundary where the lane VALUE also changes is where a structured
        # sibling segment may BEGIN, so the expensive matcher is offered there too.  A
        # forced boundary at a frame whose value is UNCHANGED (the common case -- a long
        # held sibling split at an interior note onset) only ever starts a hold/accum,
        # which the cheap always-on candidates already cover; adding it to the matcher
        # set would re-run the unified matcher at every note onset (quadratic on a
        # fine-spined dense tune) for no gain, so it is excluded.
        n = len(lane)
        chg = set((np.nonzero(np.diff(lane))[0] + 1).tolist())
        full_at = full_at | {p for p in force_breakpoints if 0 <= p < n and p in chg}
    cover = _cover_dp(
        lane, width, note_table, carry, idx_of, is_freq, {}, full_at, cache, force_at
    )
    prev_sig = tuple((s, t) for s, t, _ in cover)
    for _ in range(2):
        amort = _amort_of(cover, is_freq, idx_of)
        nxt = _cover_dp(
            lane,
            width,
            note_table,
            carry,
            idx_of,
            is_freq,
            amort,
            full_at,
            cache,
            force_at,
        )
        sig = tuple((s, t) for s, t, _ in nxt)
        cover = nxt
        if sig == prev_sig:
            break
        prev_sig = sig
    # Collapse periodic-walk segments that are rotations of one cycle (phase rides the
    # row) when the cycle recurs at >=2 rotations -- the dedup + REPEAT lever, applied
    # only where it pays (a singleton periodic struct keeps its bare seed).
    return _merge_rotations(cover)


def _amort_of(cover, is_freq, idx_of):
    """Per-use pool share ``key_tok / uses`` for each struct chosen in ``cover`` -- the
    cost a struct's single deduped pool entry contributes, spread over its uses.  Keyed
    purely by the seed-stripped struct, so it is position-independent."""
    toks, uses = {}, {}
    for _start, _stop, fit in cover:
        key, key_tok = _pool_key(fit[0], fit[1], is_freq, idx_of)
        toks[key] = key_tok
        uses[key] = uses.get(key, 0) + 1
    return {k: toks[k] / uses[k] for k in toks}


def _floor_runs(lane, segs, _width, _is_freq, _idx_of, force_at=None):
    """Collapse a maximal run of consecutive 1-frame holds into ONE literal-table
    CITG (the §3.6 no-escape floor) WHEN that run is genuine per-frame data -- a long
    span of mostly-DISTINCT values, where a length-N read of the lane's own bytes
    beats N hold rows that the stream LZ cannot collapse.  A run whose values REPEAT
    (a held melody) is left as per-frame holds: every hold shares ONE pool struct and
    the repeating rows LZ-compress, which beats a unique literal table.  An isolated
    1-frame hold is likewise left as-is.

    A floored literal-table run must not straddle a forced breakpoint (``force_at``),
    so the 1-hold run is broken there -- keeping every produced segment inside one
    inter-breakpoint span (the spine-alignment invariant the IR bundling relies on)."""
    out = []
    i = 0
    m = len(segs)

    def is_hold1(seg):
        return seg[1] - seg[0] == 1 and seg[2][0] == "hold"

    while i < m:
        if not is_hold1(segs[i]):
            out.append(segs[i])
            i += 1
            continue
        j = i
        while j < m and is_hold1(segs[j]):
            # Stop the floored run at a forced boundary: a segment at force_at[start]-1
            # is the last before the boundary; segs[j] would cross it.
            if (
                force_at is not None
                and j > i
                and int(force_at[segs[i][0]]) <= segs[j][0]
            ):
                break
            j += 1
        rstart, rstop = segs[i][0], segs[j - 1][1]
        run_len = rstop - rstart
        vals = [int(v) for v in lane[rstart:rstop]]
        ndistinct = len(set(vals))
        # Floor only a long, value-DISTINCT run (per-frame data); keep a repeating /
        # short run as shared-struct hold rows.
        if run_len >= 8 and ndistinct >= 0.6 * run_len:
            params = A.literal_table_citg(vals)
            out.append((rstart, rstop, ("citg", params)))
        else:
            out.extend(segs[i:j])
        i = j
    return out


# ---------------------------------------------------------------------------
# Lane extraction (SID chip semantics) for the whole-state cover.
# ---------------------------------------------------------------------------
_FREQ_LO = [0, 7, 14]
_FREQ_HI = [1, 8, 15]
_PW_LO = [2, 9, 16]
_PW_HI = [3, 10, 17]
_CTRL = [4, 11, 18]
_AD = [5, 12, 19]
_SR = [6, 13, 20]
_GLOBALS = [21, 22, 23, 24]


def _lane_specs(state):
    """The 16 register lanes (3 voices x {freq, pw, ctrl, ad, sr} + 4 globals) as
    ``(lane_id, values, width, is_freq, is_pw)`` tuples, each value sequence and its
    width derived purely from SID chip semantics."""
    state = np.asarray(state, dtype=np.int64)
    specs = []
    for v in range(3):
        freq = state[:, _FREQ_LO[v]] + (state[:, _FREQ_HI[v]] << 8)
        pw = state[:, _PW_LO[v]] + ((state[:, _PW_HI[v]] & 0xF) << 8)
        specs.append((f"{v}:freq", freq, 0xFFFF, True, False))
        specs.append((f"{v}:pw", pw, 0xFFF, False, True))
        specs.append((f"{v}:ctrl", state[:, _CTRL[v]], 0xFF, False, False))
        specs.append((f"{v}:ad", state[:, _AD[v]], 0xFF, False, False))
        specs.append((f"{v}:sr", state[:, _SR[v]], 0xFF, False, False))
    for reg in _GLOBALS:
        specs.append((f"g{reg}", state[:, reg], 0xFF, False, False))
    return specs


# ---------------------------------------------------------------------------
# Stream cost: backward-LZ a lane's per-segment rows (delta_of=None this stage).
# ---------------------------------------------------------------------------
def _row_tokens(start, stop, pool_ref, seed, base):
    """One segment row's value-tokens: (dt placeholder filled by caller) dur, ref,
    base, seed.  Used both as the LZ literal and to cost it."""
    out = []
    _wu(out, stop - start)  # dur
    _wu(out, pool_ref + 1)  # ref (biased so -1 is 0; here always >=0)
    _write_value(out, base)
    _write_value(out, seed if seed else {})
    return out


def _lane_stream_tokens(rows):
    """The backward-LZ'd token length of a lane's segment rows.  Each row is a flat
    value-token list ``(dt, dur, ref, base, seed)``; the rows are LZ'd at row
    granularity exactly as :func:`serialize._lz_emit_t` would (delta_of=None this
    stage, no TRANSPOSE), so a recurring row pattern collapses."""
    if not rows:
        return 0
    # Flatten each row to a token tuple so identical rows compare equal, then run
    # the same greedy backward-LZ the serializer uses over the flat token stream.
    flat = []
    for r in rows:
        flat.extend(r)
    lz = _lz_tokens(flat, min_copy=3)
    return len(lz)


def cover_tokens(state, note_table=None, lane_filter=None):
    """Cover ALL 16 register lanes of a ``(nframes, 25)`` state, dedup the
    seed-stripped generator structs into a shared pool, LZ-measure each lane's
    per-segment row stream, and report the token split.

    Returns a dict::

        {
          "total": stream + pool,                 # the model-facing token total
          "stream": <LZ'd row-stream tokens>,     # the per-segment rows, all lanes
          "pool": <token-LZ'd dedup'd struct pool>,
          "npool": <distinct structs>,
          "ntok_per_frame": total / nframes,
          "nframes": nframes,
          "segments": {lane_id: nsegments},
          "covers": {lane_id: [(start, stop, fit), ...]},  # byte-exact covers
          "pool_entries": [["S", [name, struct]], ...],
        }

    The pool entries are token-LZ'd together (the same :func:`_emit_pool` encoding,
    so two structs sharing a long sub-table collapse), and the row streams are
    costed per lane and summed -- exactly the ``stream + pool`` the serializer
    emits.  Pitch factoring uses the bus-recovered ``note_table`` when given."""
    state = np.asarray(state, dtype=np.int64)
    nframes = state.shape[0]
    idx_of = None
    if note_table is not None:
        arr = np.asarray(note_table, dtype=np.int64)
        idx_of = {}
        for j, freq in enumerate(arr):
            idx_of.setdefault(int(freq), j)

    # Voice carries: the PW lane's additive-pw coupling needs the sibling freq
    # lane's carry-out.  Cover freq first, recompute its carry, pass to the pw cover
    # (mirrors render_from_fits).  Here we cover freq with no carry (freq never needs
    # one) and derive each voice's carry from the freq cover for its pw lane.
    specs = _lane_specs(state)
    covers = {}
    seg_counts = {}
    pool_index = {}  # struct key -> pool ref
    pool_entries = []
    lane_rows = {}

    # Pre-compute per-voice freq covers so the pw cover can use the carry.
    freq_cover = {}
    for lane_id, vals, width, is_freq, is_pw in specs:
        if is_freq:
            freq_cover[lane_id] = cover_lane(
                vals, width, note_table, None, idx_of, True
            )

    def carry_for_voice(v):
        cov = freq_cover.get(f"{v}:freq")
        if cov is None:
            return None
        res = [(s, t, f) for (s, t, f) in cov]
        return A.freq_carry_sequence(res, nframes)

    for lane_id, vals, width, is_freq, is_pw in specs:
        if lane_filter is not None and lane_id not in lane_filter:
            continue
        if is_freq:
            cover = freq_cover[lane_id]
        elif is_pw:
            v = int(lane_id.split(":")[0])
            carry = carry_for_voice(v)
            cover = cover_lane(vals, width, note_table, carry, idx_of, False)
        else:
            cover = cover_lane(vals, width, note_table, None, idx_of, False)
        covers[lane_id] = cover
        seg_counts[lane_id] = len(cover)

        # Build the lane's row stream, deduping structs into the shared pool.
        rows = []
        prev = 0
        for start, stop, fit in cover:
            name, params = fit
            struct, seed, base = _strip_seed(name, params, is_freq, idx_of)
            entry = ["S", [name, struct]]
            key = json.dumps(entry, sort_keys=True)
            ref = pool_index.get(key)
            if ref is None:
                ref = len(pool_entries)
                pool_index[key] = ref
                pool_entries.append(entry)
            row = []
            _wu(row, start - prev)  # dt
            row.extend(_row_tokens(start, stop, ref, seed, base))
            rows.append(row)
            prev = start
        lane_rows[lane_id] = rows

    stream = sum(_lane_stream_tokens(rows) for rows in lane_rows.values())
    # Pool cost: the dedup'd structs flattened + token-LZ'd, exactly as _emit_pool.
    flat = _flatten_pool(pool_entries)
    pool = len(_lz_tokens(flat))

    total = stream + pool
    return {
        "total": total,
        "stream": stream,
        "pool": pool,
        "npool": len(pool_entries),
        "ntok_per_frame": total / nframes if nframes else 0.0,
        "nframes": nframes,
        "segments": seg_counts,
        "covers": covers,
        "pool_entries": pool_entries,
    }
