"""Numba-JIT kernels for the generic tracker-STRUCTURE recovery (discovery / decode / fit).

These are the discovery-side counterparts to the render kernels in :mod:`_render_njit`
(the generator-cover) and :mod:`_njit` (the BACC primitives).  Following the SAME
contract: ``from preframr_tokens.bacc.generic._njit import njit``; ``@njit`` (defaults
``cache=True``); TYPED numpy arrays + scalar params (never a dict-of-objects per
element); every kernel byte-identical to a pure-Python reference (pinned by
:mod:`tests.test_discover_njit`).

Canonical array layouts (declared once in the design, used by all kernels):

  * ``ram : uint8[65536]`` -- the SNAP RAM (``Distill.ram``).
  * ``read_play : uint8[65536]`` -- the READ_PLAY access mask (0/1).
  * ``idxr : int64[n, 6]`` -- IDXR flattened ``[pc, base, stride, idx_min, idx_max, count]``.
  * ``state : int64[nframes, 25]`` -- the ``.sidwr`` byte-exact target.
  * ``boundaries : int64[256]`` (field-kind per byte value) + ``op_width : int64[16]``
    (operand byte count per kind) + ``packing_mode : int64`` -- the row-grammar dialect.

The row-grammar field kinds (``boundaries[b]`` -> kind) are a SMALL fixed set shared by
all four dialects (NewPlayer / TFX-prefix / FC / nibble); the dialects differ only in
which byte ranges map to which kind and the per-kind operand widths::

    K_NOTE   = 0   a pitch byte: EMIT a row with the running (instr, dur, cmd)
    K_INSTR  = 1   set running instrument := (b & param_mask) [+ op_width operands]
    K_DUR    = 2   set running duration   := (b & param_mask)
    K_CMD    = 3   set running command    := (b & param_mask) [+ op_width operands]
    K_EOP    = 4   end of pattern
    K_REST   = 5   a rest / tie: EMIT a row with note = REST_NOTE, no pitch
    K_IGN    = 6   a marker consumed with no state change (operands still skipped)

The chip math is integer-only with explicit masks, so the numba ``int64`` kernels and
the pure-Python references agree byte-for-byte (the same reasoning as :mod:`_njit`).
"""

import numpy as np

from preframr_tokens.bacc.generic._njit import njit

# Field-kind codes (boundaries[b] -> one of these).
K_NOTE = 0
K_INSTR = 1
K_DUR = 2
K_CMD = 3
K_EOP = 4
K_REST = 5
K_IGN = 6

# Sentinel rows.  A decoded row is (note, instr, dur, cmd); a field never yet set is
# UNSET (the player's running state before its first marker); a REST is REST_NOTE.
UNSET = -1
REST_NOTE = -2

# Max rows / bytes a single pattern decode emits (a guard, never reached in practice).
_MAX_ROWS = 4096


@njit
def decode_pattern_kernel(
    ram, ptr, boundaries, op_width, param_mask, packing_mode, max_bytes
):
    """Decode ONE pattern's bytes starting at ``ptr`` into row arrays.

    Walks bytes, classifies each via ``boundaries[b]``, consumes ``op_width[kind]``
    operand bytes after a marker, and EMITS a row (the running instr/dur/cmd) on a
    note/rest.  ``param_mask[kind]`` masks the field value out of the marker byte.
    ``packing_mode`` is reserved for a bit-packed dialect (nibble): mode 1 splits a
    note byte into instr=high-nibble, note=low-nibble.

    Returns ``(notes, instr, dur, cmd, n_rows, n_bytes)`` -- preallocated ``int64``
    row arrays (``UNSET`` / ``REST_NOTE`` sentinels) sliced to ``n_rows``, and the
    number of bytes consumed (through the EOP byte inclusive).  The single decode
    skeleton for all four dialects (parameterized only by the small arrays)."""
    notes = np.empty(_MAX_ROWS, dtype=np.int64)
    instr = np.empty(_MAX_ROWS, dtype=np.int64)
    dur = np.empty(_MAX_ROWS, dtype=np.int64)
    cmd = np.empty(_MAX_ROWS, dtype=np.int64)
    cur_instr = UNSET
    cur_dur = UNSET
    cur_cmd = UNSET
    nrows = 0
    idx = 0
    while idx < max_bytes and nrows < _MAX_ROWS:
        b = int(ram[(ptr + idx) & 0xFFFF])
        idx += 1
        kind = boundaries[b]
        if kind == K_EOP:
            break
        if kind == K_INSTR:
            cur_instr = b & param_mask[K_INSTR]
            idx += op_width[K_INSTR]
            continue
        if kind == K_DUR:
            cur_dur = b & param_mask[K_DUR]
            idx += op_width[K_DUR]
            continue
        if kind == K_CMD:
            cur_cmd = b & param_mask[K_CMD]
            idx += op_width[K_CMD]
            continue
        if kind == K_IGN:
            idx += op_width[K_IGN]
            continue
        if kind == K_REST:
            notes[nrows] = REST_NOTE
            instr[nrows] = cur_instr
            dur[nrows] = cur_dur
            cmd[nrows] = cur_cmd
            nrows += 1
            continue
        # K_NOTE
        if packing_mode == 1:
            cur_instr = (b >> 4) & 0x0F
            note = b & 0x0F
        else:
            note = b
        notes[nrows] = note
        instr[nrows] = cur_instr
        dur[nrows] = cur_dur
        cmd[nrows] = cur_cmd
        nrows += 1
    return notes[:nrows], instr[:nrows], dur[:nrows], cmd[:nrows], nrows, idx


@njit
def reencode_kernel(snap, ptr, n_bytes, boundaries):
    """Re-emit a pattern by COPYING its ``n_bytes`` SNAP bytes (the player's own
    compact stateful byte stream), masked to ``[0,256)``.

    The decode skeleton is stateful and lossless: the SNAP bytes ARE the canonical
    compact encoding, so the byte-exact gate is "do the bytes we sliced decode to a
    well-formed pattern (terminated by an EOP within ``n_bytes``)".  This kernel
    returns the sliced bytes for the host to compare to the original SNAP; a decode
    that ran off the end (no EOP) yields ``ok == 0``.  Reusing the bytes verbatim is
    the byte-exact re-encode (the round-trip gate's invariant -- mirrors
    :func:`structure_recover.reencode_patterns`)."""
    out = np.empty(n_bytes, dtype=np.int64)
    ok = 0
    for i in range(n_bytes):
        b = int(snap[(ptr + i) & 0xFFFF])
        out[i] = b
        if boundaries[b] == K_EOP:
            ok = 1
    return out, ok


@njit
def bank_eop_lengths_kernel(ram, ptrs, boundaries, max_bytes):
    """The per-pointer EOP-walk length for a whole pointer array, in ONE typed pass
    (byte-identical to :func:`structure_recover._walk_pattern_bytes` applied to each
    pointer): for ``ptrs[k]`` return ``out[k]`` = the byte count through the first
    ``boundaries[b] == K_EOP`` byte inclusive (``ptr`` arithmetic 16-bit-wrapped), or
    ``0`` if no EOP is hit within ``max_bytes``.

    This is the hot per-candidate scan in :func:`structure_recover._bank_candidate_eop`
    (3 dialects x up to ~256 pointers x up to ``max_bytes`` bytes), which on a behavior-
    keyed digi tune the IDXR enumeration calls thousands of times -- moving it onto a
    machine-int kernel is a PURE speedup (the lengths, hence the sliced bank and the
    selected candidate, are identical)."""
    n = ptrs.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for k in range(n):
        ptr = ptrs[k]
        for j in range(max_bytes):
            b = ram[(ptr + j) & 0xFFFF]
            if boundaries[b] == K_EOP:
                out[k] = j + 1
                break
    return out


@njit
def bank_read_extents_kernel(ram_read, ptrs, lo_img, hi_img, max_bytes):
    """The per-pointer READ-coverage extent for a whole pointer array, in ONE typed pass
    (byte-identical to :func:`structure_recover._read_extent` applied to each pointer):
    for ``ptrs[k]`` return ``out[k]`` = the length of the contiguous run from ``ptr`` of
    bytes that were READ AS DATA (``ram_read[addr] != 0``) and lie in ``[lo_img, hi_img)``,
    capped at ``max_bytes``.  ``0`` when ``ptr`` itself was not read (a phantom pointer).

    The read-coverage counterpart of :func:`bank_eop_lengths_kernel`; the grammar-agnostic
    leg of :func:`structure_recover._bank_candidate_readcov` calls this per pointer."""
    n = ptrs.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for k in range(n):
        ptr = ptrs[k]
        j = 0
        while j < max_bytes:
            addr = ptr + j
            if addr < lo_img or addr >= hi_img:
                break
            if ram_read[addr & 0xFFFF] == 0:
                break
            j += 1
        out[k] = j
    return out


@njit
def split_pointer_table_kernel(ram, read_play, lo_img, hi_img):
    """The legacy split lo/hi pointer-table scan (``structure_recover.discover_pointer_table``)
    on typed arrays -- byte-identical to the Python double loop, just compiled.

    For every ``hi_base`` in ``[lo_img, hi_img)`` and ``gap`` (table length ``n``) in
    ``[4, 96)`` with ``lo_base = hi_base - gap`` in image and ``hi_base + n <= hi_img``,
    the formed pointers ``ram[lo_base+i] | ram[hi_base+i]<<8`` must (a) all land in
    ``[hi_base+n, hi_img)``, (b) strictly ascend, (c) use ``<= 6`` distinct hi-byte
    pages, and (d) cover ``>= 1`` READ_PLAY target.  Returns ``[lo_base, hi_base, n]`` of
    the LARGEST such table (first found at that size -- the host's ``n > best`` tiebreak),
    or ``[-1, -1, 0]`` when none exists."""
    best_lo = -1
    best_hi = -1
    best_n = 0
    seen_pages = np.zeros(256, dtype=np.int64)
    page_stamp = 0
    for hi_base in range(lo_img, hi_img - 2):
        for gap in range(4, 96):
            n = gap
            lo_base = hi_base - gap
            if lo_base < lo_img:
                continue
            if hi_base + n > hi_img:
                continue
            if n <= best_n:
                continue  # only a strictly-larger table can replace the incumbent
            ok = True
            ascending = True
            read_hit = False
            distinct_pages = 0
            page_stamp += 1
            prev = -1
            floor = hi_base + n
            for i in range(n):
                ptr = int(ram[lo_base + i]) | (int(ram[hi_base + i]) << 8)
                if ptr < floor or ptr >= hi_img:
                    ok = False
                    break
                if prev >= 0 and ptr <= prev:
                    ascending = False
                    break
                prev = ptr
                page = int(ram[hi_base + i])
                if seen_pages[page] != page_stamp:
                    seen_pages[page] = page_stamp
                    distinct_pages += 1
                    if distinct_pages > 6:
                        ok = False
                        break
                if read_play[ptr] != 0:
                    read_hit = True
            if not ok or not ascending or not read_hit:
                continue
            best_lo = lo_base
            best_hi = hi_base
            best_n = n
    out = np.empty(3, dtype=np.int64)
    out[0] = best_lo
    out[1] = best_hi
    out[2] = best_n
    return out


@njit
def idxr_score_kernel(idxr, ram, read_play, lo_img, hi_img, reloc_delta):
    """Score every IDXR entry as a POINTER-table candidate, on typed arrays.

    For each entry forms its 16-bit target set ``ram[t] | ram[t+1]<<8`` over the
    swept index span (interpreting ``base`` as a lo-table, ``base+1`` as the adjacent
    hi for the stride-2 interleaving), maps each target through ``reloc_delta``, and
    counts: ``n_valid`` (targets resolving into ``[lo_img,hi_img)``), ``read_cov``
    (valid targets a READ_PLAY byte sits in), ``n`` (span), and ``ascending`` (1 if
    the targets strictly ascend).  Returns ``scores : int64[n_idxr, 4]`` columns
    ``[n_valid, read_cov, n, ascending]`` -- the ranking signal the host sorts on.

    Replaces the O(image^2 . gap) Python scan in the old ``discover_pointer_table``:
    one pass over the <=tens of IDXR entries x their <=256 targets."""
    n_idxr = idxr.shape[0]
    scores = np.zeros((n_idxr, 4), dtype=np.int64)
    for e in range(n_idxr):
        base = idxr[e, 1]
        stride = idxr[e, 2]
        idx_min = idxr[e, 3]
        idx_max = idxr[e, 4]
        if stride <= 0:
            continue
        n = idx_max - idx_min + 1
        if n < 2:
            continue
        n_valid = 0
        read_cov = 0
        ascending = 1
        prev = -1
        for i in range(idx_min, idx_max + 1):
            lo_addr = (base + stride * i) & 0xFFFF
            tlo = int(ram[lo_addr])
            thi = int(ram[(lo_addr + 1) & 0xFFFF])
            target = ((tlo | (thi << 8)) + reloc_delta) & 0xFFFF
            if lo_img <= target < hi_img:
                n_valid += 1
                if read_play[target] != 0:
                    read_cov += 1
            if prev >= 0 and target <= prev:
                ascending = 0
            prev = target
        scores[e, 0] = n_valid
        scores[e, 1] = read_cov
        scores[e, 2] = n
        scores[e, 3] = ascending
    return scores


@njit
def accfit_kernel(samples, n, width_mask, kind_code):
    """Fit a generator to a STSQ accumulator's value sequence; return params + the
    longest byte-exact prefix (the matcher form -- step the recurrence, compare in
    place, no per-candidate allocation).

      kind 0 = RAMP      ``value += rate`` (rate = samples[1]-samples[0])
      kind 1 = QUADRATIC ``rate += accel; value += rate`` (2nd difference constant)
      kind 2 = TRIANGLE  reflecting ``value += step`` within ``[lo, hi]`` (a vibrato)

    Returns ``(p0, p1, p2, p3, match_len)``: for RAMP ``(seed, rate, 0, 0)``; for
    QUADRATIC ``(seed, rate0, accel, 0)``; for TRIANGLE ``(seed, step, lo, hi)``.
    The fit reproduces ``samples[:match_len]`` byte-exact under ``width_mask`` (so a
    fit with ``match_len == n`` replaces the stored sequence; a short prefix means the
    cell is not this generator and the host keeps the next candidate / the raw cell --
    HARD RULE #0: the fit GATE is byte-exact, never approximate)."""
    if n < 2:
        return 0, 0, 0, 0, n
    seed = samples[0]
    if kind_code == 0:
        rate = (samples[1] - samples[0]) & width_mask
        val = seed
        m = 0
        for i in range(n):
            if (val & width_mask) != (samples[i] & width_mask):
                break
            val = (val + rate) & width_mask
            m += 1
        return seed, rate, 0, 0, m
    if kind_code == 1:
        if n < 3:
            return seed, (samples[1] - samples[0]) & width_mask, 0, 0, n
        rate0 = (samples[1] - samples[0]) & width_mask
        accel = (samples[2] - 2 * samples[1] + samples[0]) & width_mask
        val = seed
        rate = rate0
        m = 0
        for i in range(n):
            if (val & width_mask) != (samples[i] & width_mask):
                break
            val = (val + rate) & width_mask
            rate = (rate + accel) & width_mask
            m += 1
        return seed, rate0, accel, 0, m
    # kind 2 = reflecting triangle: derive step from the first move, bounds from the
    # observed extrema.
    lo = samples[0]
    hi = samples[0]
    for i in range(n):
        if samples[i] < lo:
            lo = samples[i]
        if samples[i] > hi:
            hi = samples[i]
    step = (samples[1] - samples[0]) & width_mask
    if step > (width_mask >> 1):
        step = step - (width_mask + 1)  # signed
    val = seed
    direction = 1 if step >= 0 else -1
    astep = step if step >= 0 else -step
    m = 0
    for i in range(n):
        if (val & width_mask) != (samples[i] & width_mask):
            break
        nxt = val + astep * direction
        if nxt > hi:
            nxt = 2 * hi - nxt
            direction = -direction
        elif nxt < lo:
            nxt = 2 * lo - nxt
            direction = -direction
        val = nxt
        m += 1
    return seed, astep, lo, hi, m


@njit
def accumulator_grid_kernel(seed, p1, p2, p3, kind_code, nframes, start, width_mask):
    """Render a fitted accumulator generator to its per-frame 16-bit grid.

    The inverse of :func:`accfit_kernel`: ``kind 0`` ramp, ``kind 1`` quadratic,
    ``kind 2`` reflecting triangle.  The grid is 0 before ``start`` (the cell's
    ``first_seen`` frame) and then steps the recurrence; held at the last value is
    NOT applied here (the caller holds), so a value before ``start`` reads 0 exactly
    as :func:`structure_recover.clean_pitches_residual` builds it."""
    out = np.zeros(nframes, dtype=np.int64)
    if start >= nframes:
        return out
    val = seed
    rate = p1
    direction = 1 if p1 >= 0 else -1
    astep = p1 if p1 >= 0 else -p1
    for i in range(start, nframes):
        out[i] = val & width_mask
        if kind_code == 0:
            val = (val + p1) & width_mask
        elif kind_code == 1:
            val = (val + rate) & width_mask
            rate = (rate + p2) & width_mask
        else:
            nxt = val + astep * direction
            if nxt > p3:
                nxt = 2 * p3 - nxt
                direction = -direction
            elif nxt < p2:
                nxt = 2 * p2 - nxt
                direction = -direction
            val = nxt
    return out


@njit
def freq_integrator_kernel(seed_grid, acc):
    """``freq[i] = (seed_grid[i] + acc[i]) & 0xFFFF`` -- the note-table integrator
    render from a piecewise-constant per-note seed grid + the summed (fitted)
    accumulators.  Generalises ``render_freq_from_ir``'s per-voice combine to a
    single typed-array kernel.  Both inputs are ``int64[nframes]``."""
    n = seed_grid.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = (seed_grid[i] + acc[i]) & 0xFFFF
    return out


@njit
def step_lane_kernel(starts, values, nframes):
    """Render a piecewise-constant non-freq register lane (the M1 replay): hold
    ``values[k]`` over ``[starts[k], starts[k+1])`` (the last value to the end).

    The serialized non-freq program is the lane's CHANGE-POINT stream (``starts`` is
    strictly ascending with ``starts[0] == 0``, ``values`` the held byte at each step);
    this kernel re-renders it byte-exact -- the inverse of the host's change-point
    encode.  ``starts``/``values`` are ``int64[nseg]``; the output is ``int64[nframes]``.
    Empty (``nseg == 0``) yields all-zero (an unused lane, never admitted)."""
    out = np.zeros(nframes, dtype=np.int64)
    nseg = starts.shape[0]
    if nseg == 0:
        return out
    for k in range(nseg):
        s = starts[k]
        e = starts[k + 1] if k + 1 < nseg else nframes
        if s < 0:
            s = 0
        if e > nframes:
            e = nframes
        for i in range(s, e):
            out[i] = values[k]
    return out


@njit
def ramp_segments_kernel(col, modulus):
    """Segment a register column into maximal CONSTANT-STEP WRAPPING-RAMP runs (the
    M1 generator-fit for a sweep lane): the lane's per-frame value is a free-running
    accumulator ``value += step (mod modulus)``, reset/reparametrised only at note-gated
    boundaries.  A segment is the longest run from ``i`` reproducible by stepping the
    recurrence from ``col[i]`` with the fixed step ``col[i+1]-col[i] (mod modulus)``.

    Returns ``(starts, seeds, steps, nseg)`` -- per segment the start frame, the seed
    value ``col[start]``, and the per-frame step; the host serialises the (small) segment
    boundaries + per-segment ``(seed, step)`` (value-LZ'd) instead of the dense change-
    point dump (storing the OUTPUT is the HARD RULE #0 literal-floor trap -- a ramp's
    GENERATOR is a handful of ints).  :func:`ramp_render_kernel` is the exact inverse.
    ``col`` is ``int64[nframes]`` (the 16-bit ``lo | hi<<8`` PW value, or an 8-bit lane);
    ``modulus`` is the wrap (``65536`` for a 16-bit PW combine, ``256`` for a byte lane).
    """
    n = col.shape[0]
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), 0
    # first pass: count segments (so the output arrays are exactly sized, njit-friendly).
    nseg = 0
    i = 0
    while i < n:
        if i + 1 >= n:
            nseg += 1
            break
        step = (col[i + 1] - col[i]) % modulus
        val = col[i]
        j = i
        while j < n and (val % modulus) == (col[j] % modulus):
            val = (val + step) % modulus
            j += 1
        nseg += 1
        i = j
    starts = np.empty(nseg, dtype=np.int64)
    seeds = np.empty(nseg, dtype=np.int64)
    steps = np.empty(nseg, dtype=np.int64)
    s = 0
    i = 0
    while i < n:
        starts[s] = i
        seeds[s] = col[i] % modulus
        if i + 1 >= n:
            steps[s] = 0
            s += 1
            break
        step = (col[i + 1] - col[i]) % modulus
        steps[s] = step
        val = col[i]
        j = i
        while j < n and (val % modulus) == (col[j] % modulus):
            val = (val + step) % modulus
            j += 1
        s += 1
        i = j
    return starts, seeds, steps, nseg


@njit
def ramp_render_kernel(starts, seeds, steps, nframes, modulus):
    """Render a ramp-segment generator to its per-frame value column (the inverse of
    :func:`ramp_segments_kernel`): over ``[starts[k], starts[k+1])`` step the recurrence
    ``value = (seeds[k] + (i - starts[k]) * steps[k]) mod modulus`` (the last segment
    runs to ``nframes``).  Byte-exact: stepping the same recurrence the fit verified
    reproduces ``col`` exactly.  ``starts``/``seeds``/``steps`` are ``int64[nseg]``; the
    output is ``int64[nframes]`` (the 16-bit PW value the host splits back to lo/hi)."""
    out = np.zeros(nframes, dtype=np.int64)
    nseg = starts.shape[0]
    if nseg == 0:
        return out
    for k in range(nseg):
        a = starts[k]
        b = starts[k + 1] if k + 1 < nseg else nframes
        if a < 0:
            a = 0
        if b > nframes:
            b = nframes
        val = seeds[k] % modulus
        step = steps[k] % modulus
        for i in range(a, b):
            out[i] = val
            val = (val + step) % modulus
    return out


@njit
def change_points_kernel(col):
    """The change-point stream of a register column ``col`` (``int64[nframes]``): the
    frame indices where the value changes (always including frame 0).  Returns
    ``starts : int64[nseg]`` (strictly ascending, ``starts[0] == 0``); the host pairs
    it with ``col[starts]`` to serialize the lane as a piecewise-constant program.  The
    inverse of :func:`step_lane_kernel` (``step_lane_kernel(starts, col[starts], n)``
    reproduces ``col`` byte-exact)."""
    n = col.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    nseg = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            nseg += 1
    starts = np.empty(nseg, dtype=np.int64)
    starts[0] = 0
    j = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            starts[j] = i
            j += 1
    return starts


@njit
def piecewise_seed_kernel(freq, acc, start, stop):
    """Recover the piecewise-constant note seed from ``(freq - acc)`` over ``[start,
    stop)`` and re-render it held per note span; return ``(seed_render, n_changes)``.

    ``seed = (freq - acc) mod 2^16`` is forced piecewise-constant (one true grid
    pitch per note span); the render holds each segment's first value across the
    segment.  ``n_changes`` over ``[start, stop)`` is the segment count minus one --
    the host's flatness ranking.  Mirrors ``clean_pitches_residual``'s seed
    derivation, on typed arrays."""
    n = freq.shape[0]
    seed = np.empty(n, dtype=np.int64)
    for i in range(n):
        seed[i] = (freq[i] - acc[i]) & 0xFFFF
    out = np.zeros(n, dtype=np.int64)
    changes = 0
    a = start if start > 0 else 0
    b = stop if stop < n else n
    cur = seed[a] if a < n else 0
    for i in range(n):
        if i == a:
            cur = seed[i]
        elif a < i and i < b:
            if seed[i] != seed[i - 1]:
                cur = seed[i]
                changes += 1
        out[i] = cur
    for i in range(a):
        out[i] = seed[a] if a < n else 0
    for i in range(b, n):
        out[i] = out[b - 1] if b > 0 else 0
    return out, changes
