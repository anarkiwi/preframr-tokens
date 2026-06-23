"""Numba-JIT kernel for the unified CITG render loop (the cover's #1 hot path).

:func:`archetypes.render_citg` is the single most-called renderer in the cover
search: the per-position matcher (:func:`archetypes._prefix_citg`) re-renders ~20
CITG candidates at every breakpoint and :func:`cover._extend` re-renders each
chosen candidate in a doubling window, so on a long algorithmic lane (A Mind Is
Born) the pure-Python ``read`` / ``accum`` / ``wrapaccum`` tail loop and its
``_citg_gates`` clock loop together dominate the encode (cProfile: ``render_citg``
11s + ``_citg_gates`` 5s of a ~32s encode).

This module fuses the clock-gate computation and the pointer/accumulator walk into
ONE compiled loop, :func:`citg_walk`, for the COMMON CITG shapes (the ``read`` and
``accum`` table-walk modes under the ``every`` / ``dwell`` / ``dwell_ptr`` / ``mask``
/ ``advance`` clocks, with the optional ``wrap=[lo,hi)`` sawtooth bound).  The
cross-cutting modes (``vibrato`` / ``glide`` / ``additive_pw`` / the composites)
keep their own (already-njit'd) renderers and never reach this kernel.

PURE speedup -- byte-for-byte identical to the pure-Python reference loop it
replaces (the chip math is fixed-width ``& width``; Numba's machine ``int64`` with
the SAME explicit masks agrees exactly, and every value stays well within
``int64`` range).  Validated by the existing render-parity test and the codec gate
(residual 0, token-identical).  As a belt-and-suspenders fallback the caller in
:mod:`archetypes` keeps the pure-Python loop for any shape outside this kernel's
domain (and the kernel itself degrades to plain Python when numba is absent).
"""

import numpy as np

from preframr_tokens.bacc.generic._njit import njit

# Clock-kind integer codes (the kernel takes a scalar, not the {"kind": ...} dict).
CLK_EVERY = 0
CLK_DWELL = 1
CLK_DWELL_PTR = 2
CLK_MASK = 3
CLK_ADVANCE = 4

# Render-mode integer codes.
MODE_READ = 0
MODE_ACCUM = 1
MODE_WRAP = 2


# ---------------------------------------------------------------------------
# Fused triangle-vibrato candidate search (the matcher's #2 hot loop post-CITG).
# ---------------------------------------------------------------------------
@njit
def _tri_phase4(ph0):
    """The triangle phase 0..3 for the four consecutive counters ph0..ph0+3, plus a
    full period -- the kernel only ever needs ``tri_phase(ph0 + i)`` per frame, which
    it recomputes inline; this helper documents the ``osc = ctr & 7; if osc>=4 osc^=7``
    shape :func:`archetypes.tri_phase` defines."""
    out = np.empty(4, dtype=np.int64)
    for k in range(4):
        osc = (ph0 + k) & 7
        if osc >= 4:
            osc ^= 7
        out[k] = osc
    return out


@njit
def vibrato_search(seg, bases, amps, minrun):
    """Fused :func:`archetypes._prefix_vibrato` inner search -- enumerate the SAME
    ``(base, amp, ph0)`` grid (``bases`` x ``amps`` x ``ph0`` in 0..7) for BOTH the
    plain ``vibrato`` (``value = (base + phase*amp) & 0xFFFF``) and the byte-wise
    ``vibrato_exact`` (repeated 16-bit add of ``amp`` over the triangle phase) modes,
    computing each candidate's longest byte-exact prefix WITHOUT materialising a render
    (step the phase, compare to ``seg`` in place, stop at the first divergence -- the
    exact value :func:`match_prefix` would compute), and keep the longest.

    Returns ``(best_match, mode_flag, base, amp, ph0)`` with ``mode_flag`` 0 for
    ``vibrato`` and 1 for ``vibrato_exact`` (``best_match == -1`` when none reaches
    ``minrun``).  Byte-identical to the Python double loop's first-strictly-longer-wins
    selection (it scans bases-outer, amps, ph0, vibrato-before-exact, exactly as the
    reference does) but with no per-candidate numpy render or dispatch."""
    length = len(seg)
    nb = len(bases)
    na = len(amps)
    best_match = -1
    best_mode = 0
    best_base = 0
    best_amp = 0
    best_ph0 = 0
    for bi in range(nb):
        base = bases[bi]
        b_lo = base & 0xFF
        b_hi = (base >> 8) & 0xFF
        for ai in range(na):
            amp = amps[ai]
            d_lo = amp & 0xFF
            d_hi = (amp >> 8) & 0xFF
            # The four phase values for each mode (phase in 0..3).
            # vibrato: (base + phase*amp) & 0xFFFF.
            vib0 = base & 0xFFFF
            vib1 = (base + amp) & 0xFFFF
            vib2 = (base + 2 * amp) & 0xFFFF
            vib3 = (base + 3 * amp) & 0xFFFF
            # vibrato_exact: repeated byte-wise 16-bit add of amp from base.
            lo = b_lo
            hi = b_hi
            ex0 = lo | (hi << 8)
            t = lo + d_lo
            lo = t & 0xFF
            t = hi + d_hi + (t >> 8)
            hi = t & 0xFF
            ex1 = lo | (hi << 8)
            t = lo + d_lo
            lo = t & 0xFF
            t = hi + d_hi + (t >> 8)
            hi = t & 0xFF
            ex2 = lo | (hi << 8)
            t = lo + d_lo
            lo = t & 0xFF
            t = hi + d_hi + (t >> 8)
            hi = t & 0xFF
            ex3 = lo | (hi << 8)
            for ph0 in range(8):
                # vibrato mode (flag 0) -- tried first so a tie keeps it.
                m = 0
                for i in range(length):
                    osc = (ph0 + i) & 7
                    if osc >= 4:
                        osc ^= 7
                    if osc == 0:
                        v = vib0
                    elif osc == 1:
                        v = vib1
                    elif osc == 2:
                        v = vib2
                    else:
                        v = vib3
                    if v != seg[i]:
                        break
                    m += 1
                if m >= minrun and m > best_match:
                    best_match = m
                    best_mode = 0
                    best_base = base
                    best_amp = amp
                    best_ph0 = ph0
                # vibrato_exact mode (flag 1).
                m = 0
                for i in range(length):
                    osc = (ph0 + i) & 7
                    if osc >= 4:
                        osc ^= 7
                    if osc == 0:
                        v = ex0
                    elif osc == 1:
                        v = ex1
                    elif osc == 2:
                        v = ex2
                    else:
                        v = ex3
                    if v != seg[i]:
                        break
                    m += 1
                if m >= minrun and m > best_match:
                    best_match = m
                    best_mode = 1
                    best_base = base
                    best_amp = amp
                    best_ph0 = ph0
    return (best_match, best_mode, best_base, best_amp, best_ph0)


# ---------------------------------------------------------------------------
# Fused vibrato + hi-byte skydive composite candidate search.
# ---------------------------------------------------------------------------
@njit
def vibskydive_search(seg, lo, hi, base, cand_lo, minrun):
    """Fused :func:`archetypes._prefix_vibskydive` inner search -- enumerate the SAME
    ``(amp_lo, amp_hi, ph0, par)`` grid (``amp_lo`` in the precomputed ``cand_lo``
    ascending-set order, ``amp_hi`` in ``0..0x3F``, ``ph0`` in ``0..7``, ``par`` in
    ``0,1``) for the vibrato_exact + descending-hi-byte-overlay composite, matching each
    candidate byte-exact in place WITHOUT a materialised :func:`render_vibrato_exact` /
    :func:`render_vibskydive` per candidate (the pure-Python form rendered ~``cand_lo x
    64 x 8`` full-length lanes per breakpoint -- the cover's #2 archetype hot loop after
    CITG).

    For each ``(amp_lo, amp_hi, ph0)`` the four vibrato_exact phase values are derived by
    the repeated byte-wise 16-bit add of ``amp`` from ``base`` (the
    ``_vibrato_exact_phase_tables`` recurrence); the candidate is GATED on the first 8
    LO-bytes matching ``lo[:8]`` (the reference's ``vib[:8] & 0xFF == lo[:8]`` skip) before
    the per-``par`` skydive overlay (``sfh`` starts at the hi-byte of the first frame of
    that parity and counts down on each parity frame).  The winner is the first
    strictly-longer match (``match >= 8`` floor) in that exact nested order, returning at
    once on a full-length match -- byte-identical selection.  Returns ``(best_match,
    amp, ph0, sfh0, par)`` with ``best_match == -1`` when none reaches the floor."""
    length = len(seg)
    b_lo = base & 0xFF
    b_hi = (base >> 8) & 0xFF
    best_match = -1
    best_amp = 0
    best_ph0 = 0
    best_sfh0 = 0
    best_par = 0
    for ci in range(len(cand_lo)):
        amp_lo = cand_lo[ci]
        for amp_hi in range(0, 0x40):
            amp = amp_lo | (amp_hi << 8)
            if amp == 0:
                continue
            d_lo = amp & 0xFF
            d_hi = (amp >> 8) & 0xFF
            # The four vibrato_exact (lo, hi) phase outcomes (repeated byte-wise add).
            clo = b_lo
            chi = b_hi
            elo0 = clo
            ehi0 = chi
            t = clo + d_lo
            clo = t & 0xFF
            t = chi + d_hi + (t >> 8)
            chi = t & 0xFF
            elo1 = clo
            ehi1 = chi
            t = clo + d_lo
            clo = t & 0xFF
            t = chi + d_hi + (t >> 8)
            chi = t & 0xFF
            elo2 = clo
            ehi2 = chi
            t = clo + d_lo
            clo = t & 0xFF
            t = chi + d_hi + (t >> 8)
            chi = t & 0xFF
            elo3 = clo
            ehi3 = chi
            for ph0 in range(8):
                # Gate: the first 8 vibrato_exact LO bytes must equal lo[:8].
                gate_ok = True
                ng = 8 if length > 8 else length
                for i in range(ng):
                    osc = (ph0 + i) & 7
                    if osc >= 4:
                        osc ^= 7
                    if osc == 0:
                        vlo = elo0
                    elif osc == 1:
                        vlo = elo1
                    elif osc == 2:
                        vlo = elo2
                    else:
                        vlo = elo3
                    if vlo != lo[i]:
                        gate_ok = False
                        break
                if not gate_ok:
                    continue
                for par in range(2):
                    # first frame index of this parity (relative to ph0 counter).
                    first = -1
                    for i in range(length):
                        if ((ph0 + i) & 1) == par:
                            first = i
                            break
                    if first < 0:
                        continue
                    sfh0 = hi[first]
                    # Match the vibskydive render in place: vibrato_exact value with a
                    # descending hi-byte counter overlaid on the parity frames.
                    sfh = sfh0 & 0xFF
                    match = 0
                    for i in range(length):
                        osc = (ph0 + i) & 7
                        if osc >= 4:
                            osc ^= 7
                        if osc == 0:
                            vlo = elo0
                            vhi = ehi0
                        elif osc == 1:
                            vlo = elo1
                            vhi = ehi1
                        elif osc == 2:
                            vlo = elo2
                            vhi = ehi2
                        else:
                            vlo = elo3
                            vhi = ehi3
                        if ((ph0 + i) & 1) == par and sfh != 0:
                            old = sfh
                            sfh = (sfh - 1) & 0xFF
                            vhi = old
                        if (vlo | (vhi << 8)) != seg[i]:
                            break
                        match += 1
                    if match >= minrun and match > best_match:
                        best_match = match
                        best_amp = amp
                        best_ph0 = ph0
                        best_sfh0 = sfh0
                        best_par = par
                        if match == length:
                            return (best_match, best_amp, best_ph0, best_sfh0, best_par)
    return (best_match, best_amp, best_ph0, best_sfh0, best_par)


# ---------------------------------------------------------------------------
# Fused mirror-fold triangle (pingfold) candidate search.
# ---------------------------------------------------------------------------
@njit
def pingfold_search(seg, steps, vis_lo, vis_hi, base, minrun):
    """Fused :func:`archetypes._prefix_pingfold` inner search -- enumerate the SAME
    ``(frac, hi, lo, sub, dir0)`` grid and return the longest byte-exact mirror-fold
    fixed-point triangle prefix WITHOUT a materialised :func:`render_pingfold` per
    candidate (the search renders up to ``4*2*2*scale*2`` candidates per breakpoint).

    ``steps[frac]`` is the precomputed internal increment ``round(mean_step * 2**frac)``
    (computed in Python so the half-to-even rounding is byte-identical; ``-1`` marks a
    non-positive step to skip).  For each ``frac`` it tries the two fold-bound
    conventions (``vis_hi<<frac`` and ``(vis_hi+1)<<frac`` for ``hi``; ``vis_lo<<frac``
    and ``0`` for ``lo``), each sub-resolution seed ``sub`` in ``0..scale-1`` and each
    ``dir0`` in ``(1, -1)`` -- in the SAME nested order as the reference -- matching the
    ``acc >> frac`` reflecting accumulator in place and returning at once on a full match.
    Returns ``(best_match, frac, step, lo, hi, acc0, dir0)`` (``best_match == -1`` when
    none reaches ``minrun``); first-strictly-longer-wins, identical to the Python loop.
    """
    length = len(seg)
    best_match = -1
    best_frac = 0
    best_step = 0
    best_lo = 0
    best_hi = 0
    best_acc0 = 0
    best_dir0 = 0
    for frac in range(4):
        step = steps[frac]
        if step <= 0:
            continue
        scale = 1 << frac
        for hsel in range(2):
            hi = (vis_hi << frac) if hsel == 0 else ((vis_hi + 1) << frac)
            for lsel in range(2):
                lo = (vis_lo << frac) if lsel == 0 else 0
                if hi <= lo:
                    continue
                for sub in range(scale):
                    acc0 = (base << frac) + sub
                    if acc0 > hi or acc0 < lo:
                        continue
                    for ds in range(2):
                        dir0 = 1 if ds == 0 else -1
                        # render_pingfold recurrence, matched in place.
                        acc = acc0
                        direction = dir0
                        match = 0
                        for i in range(length):
                            if (acc >> frac) != seg[i]:
                                break
                            match += 1
                            acc += step * direction
                            if acc > hi:
                                acc = 2 * hi - acc
                                direction = -direction
                            elif acc < lo:
                                acc = 2 * lo - acc
                                direction = -direction
                        if match >= minrun and match > best_match:
                            best_match = match
                            best_frac = frac
                            best_step = step
                            best_lo = lo
                            best_hi = hi
                            best_acc0 = acc0
                            best_dir0 = dir0
                            if match == length:
                                return (
                                    best_match,
                                    best_frac,
                                    best_step,
                                    best_lo,
                                    best_hi,
                                    best_acc0,
                                    best_dir0,
                                )
    return (best_match, best_frac, best_step, best_lo, best_hi, best_acc0, best_dir0)


# ---------------------------------------------------------------------------
# Fused dwelled wavetable-rate accumulator (dwellratewalk) candidate search.
# ---------------------------------------------------------------------------
@njit
def dwellratewalk_search(seg, dsign, cand_dwells, maxp, width_mask, minrun):
    """Fused :func:`archetypes._prefix_dwellratewalk` inner search -- for each candidate
    ``dwell`` (passed pre-sorted as the reference computes them) and each ``period`` in
    ``2..nsteps`` it reads the step table off ``dsign`` at the dwell stride and matches
    the dwelled-rate accumulator render in place (no :func:`render_dwellratewalk`
    allocation per candidate), keeping the first strictly-longer byte-exact prefix with
    the ``match >= max(minrun, 2*dwell*period)`` floor and the ``>=2 distinct steps``
    requirement -- byte-identical to the Python double loop.

    Returns ``(best_match, best_dwell, best_period)`` (``best_match == -1`` when none
    qualifies); the caller rebuilds ``rate_table = [dsign[k*dwell] for k in range(P)]``.
    """
    length = len(seg)
    dlen = len(dsign)
    ndw = len(cand_dwells)
    v0 = seg[0]
    best_match = -1
    best_dwell = 0
    best_period = 0
    for di in range(ndw):
        dwell = cand_dwells[di]
        nsteps = dlen // dwell
        if nsteps > maxp:
            nsteps = maxp
        if nsteps < 2:
            continue
        for period in range(2, nsteps + 1):
            # >=2 distinct steps in table[:period] (table[k] = dsign[k*dwell]).
            distinct = False
            first = dsign[0]
            for k in range(1, period):
                if dsign[k * dwell] != first:
                    distinct = True
                    break
            if not distinct:
                continue
            floor = 2 * dwell * period
            if floor < minrun:
                floor = minrun
            # render_dwellratewalk recurrence, matched in place:
            #   out[i] = val & wm; val = (val + rate_table[((0+i)//dwell) % period]) & wm
            val = v0 & width_mask
            match = 0
            for i in range(length):
                if (val & width_mask) != seg[i]:
                    break
                match += 1
                idx = (i // dwell) % period
                val = (val + dsign[idx * dwell]) & width_mask
            if match >= floor and match > best_match:
                best_match = match
                best_dwell = dwell
                best_period = period
    return (best_match, best_dwell, best_period)


# ---------------------------------------------------------------------------
# Fused signed-rate wavetable-accumulator candidate search.
# ---------------------------------------------------------------------------
@njit
def ratewalk_search(seg, dsign, maxp, width_mask, minrun):
    """Fused :func:`archetypes._prefix_ratewalk` inner search -- enumerate the SAME
    ``period`` grid (``1..min(maxp, len(dsign))``) and return the longest byte-exact
    period-P signed-rate accumulator prefix, in the SAME order and with the SAME
    acceptance the Python loop used, but WITHOUT a materialised
    :func:`render_ratewalk` + numpy ``_match_prefix`` per period (the remaining
    pure-Python render-per-period sweep -- ``render_ratewalk`` was ~44 full-window
    renders per ``_prefix_ratewalk`` call, the matcher's last un-fused hot loop).

    ``dsign`` is the precomputed signed per-frame delta the reference builds in numpy
    (``diff = np.diff(seg) % (wm+1)``; ``dsign = where(diff > wm//2, diff-(wm+1),
    diff)``), so the kernel reads the SAME period-P rate table off ``dsign[:period]``.
    A candidate period is admitted only when its table has a nonzero entry (the
    reference's ``if not any(table): continue`` -- a constant hold is left to the
    cheaper zoo ``hold``); its render is the ``render_ratewalk`` recurrence with
    ``ctr0=0`` matched byte-exact in place: ``out[i] = val & wm`` then
    ``val = (val + dsign[i % period]) & wm`` starting ``val = seg[0]``.  The winner is
    the first STRICTLY-longer match (``match >= max(minrun, 2*period)`` floor),
    scanning ``period`` ascending, stopping early on a full-length match (the
    reference's ``best[0] == length`` break).  Returns ``(best_match, best_period)``
    (``best_match == -1`` when none qualifies); the caller rebuilds
    ``rate_table = dsign[:period]``."""
    length = len(seg)
    dlen = len(dsign)
    v0 = seg[0]
    best_match = -1
    best_period = 0
    pmax = maxp if maxp < dlen else dlen
    if pmax < 1:
        pmax = 1
    for period in range(1, pmax + 1):
        if dlen < period:
            continue
        # The rate table must have a nonzero entry (``if not any(table): continue``).
        nonzero = False
        for k in range(period):
            if dsign[k] != 0:
                nonzero = True
                break
        if not nonzero:
            continue
        floor = 2 * period
        if floor < minrun:
            floor = minrun
        # render_ratewalk(ctr0=0) recurrence, matched in place.
        val = v0
        match = 0
        for i in range(length):
            if (val & width_mask) != seg[i]:
                break
            match += 1
            val = (val + dsign[i % period]) & width_mask
        if match >= floor and match > best_match:
            best_match = match
            best_period = period
        if best_match == length:
            break
    return (best_match, best_period)


# ---------------------------------------------------------------------------
# Fused single-rate periodic-stall accumulator candidate search.
# ---------------------------------------------------------------------------
@njit
def maskaccum_stall_search(seg, advance, rate, maxp, width_mask, minrun, mincycles):
    """Fused :func:`archetypes._prefix_maskaccum_stall` inner search -- enumerate the
    SAME ``period`` grid (``1..min(maxp, len(advance))``) for the dominant-rate periodic
    advance mask and return the longest byte-exact mask-accumulator prefix, in the SAME
    order and with the SAME acceptance the Python loop used, but WITHOUT a materialised
    :func:`render_maskaccum` + numpy ``_match_prefix`` per period (the last per-period
    render loop in the maskaccum family, ~``maxp`` renders per breakpoint).

    ``advance`` is the precomputed 0/1 frame mask the reference builds (``advance =
    (dsign == rate)``, ``dsign`` the signed per-frame delta), ``rate`` the dominant
    nonzero rate.  A candidate ``period`` reads ``mask = advance[:period]`` and is gated
    exactly as the reference (``steps = sum(mask) >= 1`` AND not ``(period > 1 and steps
    == period)`` -- an all-advance mask is a plain accum left to the cheaper rule); its
    render is the ``render_maskaccum`` recurrence matched byte-exact in place: ``out[i] =
    val & wm`` then ``if mask[i % period]: val = (val + rate) & wm`` starting ``val =
    seg[0]``.  The winner is the first STRICTLY-longer match (``match >= max(minrun,
    mincycles*period)`` floor), scanning ``period`` ascending, stopping early on a
    full-length match.  Returns ``(best_match, best_period)`` (``best_match == -1`` when
    none qualifies); the caller rebuilds ``mask = advance[:period]``."""
    length = len(seg)
    alen = len(advance)
    v0 = seg[0]
    best_match = -1
    best_period = 0
    pmax = maxp if maxp < alen else alen
    for period in range(1, pmax + 1):
        if alen < mincycles * period:
            continue
        # steps = sum(mask); gate out an empty or all-advance (period>1) mask.
        steps = 0
        for k in range(period):
            steps += advance[k]
        if steps < 1 or (period > 1 and steps == period):
            continue
        floor = mincycles * period
        if floor < minrun:
            floor = minrun
        # render_maskaccum recurrence, matched in place.
        val = v0
        match = 0
        for i in range(length):
            if (val & width_mask) != seg[i]:
                break
            match += 1
            if advance[i % period]:
                val = (val + rate) & width_mask
        if match >= floor and match > best_match:
            best_match = match
            best_period = period
        if best_match == length:
            break
    return (best_match, best_period)


# ---------------------------------------------------------------------------
# Fused lead-hold-then-table-walk candidate search (the matcher's #1 hot loop).
# ---------------------------------------------------------------------------
@njit
def tablewalk_lead_search(seg, lead0, maxp, minrun):
    """Fused :func:`archetypes._prefix_tablewalk_lead` inner search -- enumerate the
    SAME ``(lead, period)`` grid (``lead`` in ``0..min(lead0, length-1)``, ``period`` in
    ``2..min(maxp, len(body)//2)``) and return the longest byte-exact lead-hold-then-
    period-P-table-walk prefix, in the SAME order and with the SAME acceptance the Python
    loop used, but WITHOUT a numpy ``array_equal`` per period or a materialised render
    per candidate (the matcher's #1 hot loop on a lead-heavy lane: ``1478 x lead0 x
    maxp`` iterations).

    A candidate ``(lead, period)`` is admitted when ``body[:period] ==
    body[period:2*period]`` element-wise (``body = seg[lead:]``) and ``body[:period]``
    has >=2 distinct values; its render is ``out[i] = seg[0] (i<lead) else
    seg[lead + ((i-lead) % period)]`` (``ctr0=0``), matched byte-exact in place.  The
    winner is the first STRICTLY-longer match (``match >= lead + 2*period`` floor),
    scanning ``lead`` ascending then ``period`` ascending; the search stops early on a
    full-length match at any ``lead`` (the ``best[0] == length`` break).  Returns
    ``(best_match, best_lead, best_period)`` (``best_match == -1`` when none qualifies);
    the caller rebuilds ``table = seg[lead:lead+period]``."""
    length = len(seg)
    value0 = seg[0]
    best_match = -1
    best_lead = 0
    best_period = 0
    lead_hi = lead0 if lead0 < (length - 1) else (length - 1)
    for lead in range(0, lead_hi + 1):
        body_len = length - lead
        if body_len < minrun:
            continue
        pmax = maxp if maxp < (body_len // 2) else (body_len // 2)
        for period in range(2, pmax + 1):
            # body[:period] == body[period:2*period] element-wise.
            ok = True
            for k in range(period):
                if seg[lead + k] != seg[lead + period + k]:
                    ok = False
                    break
            if not ok:
                continue
            # >=2 distinct values in the period table.
            distinct = False
            base_v = seg[lead]
            for k in range(1, period):
                if seg[lead + k] != base_v:
                    distinct = True
                    break
            if not distinct:
                continue
            # Longest byte-exact prefix of the lead-hold-then-table-walk render.
            match = 0
            for i in range(length):
                if i < lead:
                    pred = value0
                else:
                    pred = seg[lead + ((i - lead) % period)]
                if pred != seg[i]:
                    break
                match += 1
            if match >= lead + 2 * period and match > best_match:
                best_match = match
                best_lead = lead
                best_period = period
        if best_match == length:
            break
    return (best_match, best_lead, best_period)


@njit
def citg_walk(
    seg_len,
    mode,
    table,
    seed,
    width,
    lead,
    phase,
    loop,
    clock_kind,
    dwell,
    fired0,
    mask,
    advance,
    lo_b,
    span,
):
    """Fused ``_citg_gates`` + pointer/accumulator walk for the table-walk CITG
    modes -- byte-identical to the pure-Python tail of :func:`archetypes.render_citg`.

    ``mode`` is :data:`MODE_READ` / :data:`MODE_ACCUM` / :data:`MODE_WRAP`.  ``table``
    is the int64 value/step table (period ``len(table)``); the pointer starts at
    ``phase % period``, advances on the clock step and wraps to ``loop``.  ``seed`` is
    the read value0 / accum acc0; ``width`` masks the accumulator (ACCUM, unused for
    WRAP whose ``[lo_b, lo_b+span)`` span IS the wrap).  ``lead`` frames hold the seed
    with the clock disarmed; ``clock_kind`` + ``dwell`` + ``fired0`` (the armed-frame
    phase seed) + ``mask`` + ``advance`` reproduce :func:`archetypes._citg_gates`
    exactly (``mask`` / ``advance`` are passed as length>=1 int64 arrays; an unused one
    is a 1-element dummy).

    The gate per armed frame ``i`` (``i >= lead``), with ``fired`` counting armed
    frames from ``fired0``:
      * ``every``     -> add = step = True;
      * ``dwell``     -> add = step = (fired+1) % dwell == 0;
      * ``dwell_ptr`` -> add = True, step = (fired+1) % dwell == 0;
      * ``mask``      -> add = step = mask[fired % len(mask)] != 0;
      * ``advance``   -> add = step = advance[fired % len(advance)] != 0.
    """
    out = np.empty(seg_len, dtype=np.int64)
    period = len(table)
    mlen = len(mask)
    alen = len(advance)
    ptr = (phase % period) if period > 0 else 0
    fired = fired0
    val = seed & width  # accum acc0 (read mode emits seed/table, not val)
    for i in range(seg_len):
        # --- clock gates (inert during the lead stall) -------------------
        add = False
        step = False
        if i >= lead:
            if clock_kind == CLK_EVERY:
                add = True
                step = True
            elif clock_kind == CLK_DWELL:
                hit = (fired + 1) % dwell == 0
                add = hit
                step = hit
            elif clock_kind == CLK_DWELL_PTR:
                add = True
                step = (fired + 1) % dwell == 0
            elif clock_kind == CLK_MASK:
                hit = mask[fired % mlen] != 0
                add = hit
                step = hit
            else:  # CLK_ADVANCE
                hit = advance[fired % alen] != 0
                add = hit
                step = hit
            fired += 1
        # --- emit + step the pointer/accumulator -------------------------
        if mode == MODE_READ:
            out[i] = seed if i < lead else (table[ptr] if period > 0 else 0)
        elif mode == MODE_WRAP:
            out[i] = val
            if period > 0:
                val += table[ptr]
                if val >= lo_b + span:
                    val -= span
                elif val < lo_b:
                    val += span
        else:  # MODE_ACCUM
            out[i] = val
            if add and period > 0:
                val = (val + table[ptr]) & width
        if step and period > 0:
            ptr += 1
            if ptr >= period:
                ptr = loop
    return out


# ---------------------------------------------------------------------------
# Fused long-period detector for the periodic-loop candidate (cover._periodic_candidates).
# ---------------------------------------------------------------------------
@njit
def periodic_diff_period(diffw, pmax, mincyc):
    """The smallest period ``P`` in ``2..pmax`` whose per-frame DIFF body repeats for
    ``mincyc`` cycles -- ``diffw[i] == diffw[i - P]`` over the first ``mincyc*P`` diffs
    -- or ``-1`` when none qualifies.  Fuses the ``for P: np.array_equal(diffw[P:need],
    diffw[:need-P])`` scan in :func:`cover._periodic_candidates` (the cover's dominant
    remaining ``array_equal`` storm: up to ``pmax`` whole-array compares per breakpoint)
    into ONE compiled loop that compares element-wise and SHORT-CIRCUITS on the first
    mismatch -- byte-identical selection (same ascending-``P`` order, same prefix
    ``[P:need)`` vs ``[0:need-P)`` comparison, first qualifying ``P`` wins).

    ``diffw`` is the precomputed windowed signed-diff array; the caller verifies the
    winner with one render exactly as before, so this only locates the SAME candidate
    period the Python scan did."""
    m = len(diffw)
    for P in range(2, pmax + 1):
        need = mincyc * P
        if need > m:
            # A larger P only makes ``need`` larger, so once ``need > m`` no further P
            # can satisfy the Python loop's ``need <= m`` guard -- stop scanning.
            break
        ok = True
        for k in range(need - P):
            if diffw[P + k] != diffw[k]:
                ok = False
                break
        if ok:
            return P
    return -1
