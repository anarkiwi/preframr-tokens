"""Byte-exact parity: each discovery-side numba-JIT kernel == its pure-Python reference.

The generic tracker-STRUCTURE recovery's hot loops (pattern decode / re-encode, the
IDXR pointer scorer, the accumulator-fit matcher + grid render, the freq integrator,
the piecewise-seed extraction) are njit'd in
:mod:`preframr_tokens.bacc.generic._discover_njit`.  This is a PURE speedup: every
kernel must reproduce an independent pure-Python reference byte-for-byte across a
swept parameter grid (the same contract :mod:`tests.test_njit_kernels` pins for the
render kernels).  The explicit ``& 0xFFFF`` chip masks keep the ``int64`` kernels in
lock-step with arbitrary-precision Python.
"""

import numpy as np
import pytest

from preframr_tokens.bacc.generic import _discover_njit as DJ


# --------------------------------------------------------------------------- #
# Pure-Python references.
# --------------------------------------------------------------------------- #
def ref_decode_pattern(ram, ptr, boundaries, op_width, param_mask, packing_mode, mx):
    notes, instr, dur, cmd = [], [], [], []
    ci = cdr = cc = DJ.UNSET
    idx = 0
    while idx < mx and len(notes) < DJ._MAX_ROWS:
        b = int(ram[(ptr + idx) & 0xFFFF])
        idx += 1
        kind = int(boundaries[b])
        if kind == DJ.K_EOP:
            break
        if kind == DJ.K_INSTR:
            ci = b & int(param_mask[DJ.K_INSTR])
            idx += int(op_width[DJ.K_INSTR])
            continue
        if kind == DJ.K_DUR:
            cdr = b & int(param_mask[DJ.K_DUR])
            idx += int(op_width[DJ.K_DUR])
            continue
        if kind == DJ.K_CMD:
            cc = b & int(param_mask[DJ.K_CMD])
            idx += int(op_width[DJ.K_CMD])
            continue
        if kind == DJ.K_IGN:
            idx += int(op_width[DJ.K_IGN])
            continue
        if kind == DJ.K_REST:
            notes.append(DJ.REST_NOTE)
            instr.append(ci)
            dur.append(cdr)
            cmd.append(cc)
            continue
        if packing_mode == 1:
            ci = (b >> 4) & 0x0F
            note = b & 0x0F
        else:
            note = b
        notes.append(note)
        instr.append(ci)
        dur.append(cdr)
        cmd.append(cc)
    return notes, instr, dur, cmd, len(notes), idx


def ref_accfit(samples, width_mask, kind):
    s = [int(x) for x in samples]
    n = len(s)
    if n < 2:
        return 0, 0, 0, 0, n
    seed = s[0]
    if kind == 0:
        rate = (s[1] - s[0]) & width_mask
        val, m = seed, 0
        for i in range(n):
            if (val & width_mask) != (s[i] & width_mask):
                break
            val = (val + rate) & width_mask
            m += 1
        return seed, rate, 0, 0, m
    if kind == 1:
        if n < 3:
            return seed, (s[1] - s[0]) & width_mask, 0, 0, n
        rate0 = (s[1] - s[0]) & width_mask
        accel = (s[2] - 2 * s[1] + s[0]) & width_mask
        val, rate, m = seed, rate0, 0
        for i in range(n):
            if (val & width_mask) != (s[i] & width_mask):
                break
            val = (val + rate) & width_mask
            rate = (rate + accel) & width_mask
            m += 1
        return seed, rate0, accel, 0, m
    lo, hi = min(s), max(s)
    step = (s[1] - s[0]) & width_mask
    if step > (width_mask >> 1):
        step -= width_mask + 1
    val = seed
    direction = 1 if step >= 0 else -1
    astep = step if step >= 0 else -step
    m = 0
    for i in range(n):
        if (val & width_mask) != (s[i] & width_mask):
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


def ref_accumulator_grid(seed, p1, p2, p3, kind, nframes, start, width_mask):
    out = [0] * nframes
    if start >= nframes:
        return out
    val, rate = seed, p1
    direction = 1 if p1 >= 0 else -1
    astep = p1 if p1 >= 0 else -p1
    for i in range(start, nframes):
        out[i] = val & width_mask
        if kind == 0:
            val = (val + p1) & width_mask
        elif kind == 1:
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


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def _newplayer_grammar():
    bnd = np.zeros(256, dtype=np.int64)
    for b in range(0x80, 0xA0):
        bnd[b] = DJ.K_DUR
    for b in range(0xA0, 0xC0):
        bnd[b] = DJ.K_INSTR
    for b in range(0xC0, 0x100):
        bnd[b] = DJ.K_CMD
    bnd[0x7F] = DJ.K_EOP
    bnd[0x00] = DJ.K_REST
    ow = np.zeros(16, dtype=np.int64)
    pm = np.zeros(16, dtype=np.int64)
    pm[DJ.K_DUR] = 0x1F
    pm[DJ.K_INSTR] = 0x1F
    pm[DJ.K_CMD] = 0x3F
    return bnd, ow, pm


@pytest.mark.parametrize("packing_mode", [0, 1])
def test_decode_pattern_kernel_matches_reference(packing_mode):
    rng = np.random.default_rng(7)
    bnd, ow, pm = _newplayer_grammar()
    for _ in range(40):
        ram = rng.integers(0, 256, size=65536, dtype=np.uint8)
        ptr = int(rng.integers(0x1000, 0x8000))
        # ensure an EOP lands within range so both terminate identically
        ram[(ptr + int(rng.integers(4, 60))) & 0xFFFF] = 0x7F
        got = DJ.decode_pattern_kernel(ram, ptr, bnd, ow, pm, packing_mode, 256)
        ref = ref_decode_pattern(ram, ptr, bnd, ow, pm, packing_mode, 256)
        assert list(got[0]) == ref[0]
        assert list(got[1]) == ref[1]
        assert list(got[2]) == ref[2]
        assert list(got[3]) == ref[3]
        assert got[4] == ref[4] and got[5] == ref[5]


def test_reencode_kernel_is_byte_copy():
    bnd, _, _ = _newplayer_grammar()
    snap = np.zeros(65536, dtype=np.uint8)
    prog = [0xA3, 0x82, 0x10, 0x12, 0x7F]
    snap[0x2000 : 0x2000 + len(prog)] = prog
    out, ok = DJ.reencode_kernel(snap, 0x2000, len(prog), bnd)
    assert list(out) == prog and ok == 1
    out, ok = DJ.reencode_kernel(snap, 0x2000, 3, bnd)  # no EOP in 3 bytes
    assert ok == 0


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_accfit_kernel_matches_reference(kind):
    rng = np.random.default_rng(11)
    for _ in range(60):
        n = int(rng.integers(2, 40))
        samples = rng.integers(0, 65536, size=n).astype(np.int64)
        got = DJ.accfit_kernel(samples, n, 0xFFFF, kind)
        ref = ref_accfit(samples, 0xFFFF, kind)
        assert tuple(int(x) for x in got) == tuple(int(x) for x in ref)


def test_accfit_recovers_exact_ramp_and_quadratic():
    ramp = np.array([100 + 7 * i for i in range(20)], dtype=np.int64) & 0xFFFF
    seed, p1, _p2, _p3, m = DJ.accfit_kernel(ramp, len(ramp), 0xFFFF, 0)
    assert (seed, p1, m) == (100, 7, 20)
    quad = np.array([(i * i) & 0xFFFF for i in range(20)], dtype=np.int64)
    _seed, _p1, p2, _p3, m = DJ.accfit_kernel(quad, len(quad), 0xFFFF, 1)
    assert m == 20 and p2 == 2  # constant 2nd difference


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_accumulator_grid_kernel_matches_reference(kind):
    rng = np.random.default_rng(13)
    for _ in range(40):
        nframes = int(rng.integers(4, 80))
        start = int(rng.integers(0, nframes + 2))
        seed = int(rng.integers(0, 65536))
        p1 = int(rng.integers(-50, 200))
        p2 = int(rng.integers(0, 30))
        lo = int(rng.integers(0, 200))
        hi = lo + int(rng.integers(1, 400))
        if kind == 2:
            got = DJ.accumulator_grid_kernel(
                seed, p1, lo, hi, kind, nframes, start, 0xFFFF
            )
            ref = ref_accumulator_grid(seed, p1, lo, hi, kind, nframes, start, 0xFFFF)
        else:
            got = DJ.accumulator_grid_kernel(
                seed, p1, p2, 0, kind, nframes, start, 0xFFFF
            )
            ref = ref_accumulator_grid(seed, p1, p2, 0, kind, nframes, start, 0xFFFF)
        assert list(got) == ref


def test_accfit_grid_roundtrip_ramp_quadratic():
    # a fitted ramp/quadratic renders back to the exact sequence it was fit from.
    for kind, seq in (
        (0, [50 + 9 * i for i in range(30)]),
        (1, [i * i + 3 * i + 5 for i in range(30)]),
    ):
        s = np.array([v & 0xFFFF for v in seq], dtype=np.int64)
        seed, p1, p2, p3, m = DJ.accfit_kernel(s, len(s), 0xFFFF, kind)
        assert m == len(s)
        grid = DJ.accumulator_grid_kernel(seed, p1, p2, p3, kind, len(s), 0, 0xFFFF)
        assert list(grid) == list(s)


def test_freq_integrator_kernel():
    rng = np.random.default_rng(3)
    seed = rng.integers(0, 65536, size=100).astype(np.int64)
    acc = rng.integers(0, 65536, size=100).astype(np.int64)
    got = DJ.freq_integrator_kernel(seed, acc)
    assert np.array_equal(got, (seed + acc) & 0xFFFF)


def _ref_change_points(col):
    """Pure-Python reference for ``change_points_kernel``: frame 0 plus every index
    where the value differs from the previous frame."""
    starts = [0]
    for i in range(1, len(col)):
        if col[i] != col[i - 1]:
            starts.append(i)
    return np.asarray(starts, dtype=np.int64)


def _ref_step_lane(starts, values, nframes):
    """Pure-Python reference for ``step_lane_kernel``: hold ``values[k]`` over
    ``[starts[k], starts[k+1])`` (last value to the end)."""
    out = np.zeros(nframes, dtype=np.int64)
    for k in range(len(starts)):
        s = max(0, int(starts[k]))
        e = int(starts[k + 1]) if k + 1 < len(starts) else nframes
        e = min(e, nframes)
        out[s:e] = values[k]
    return out


def test_change_points_and_step_lane_roundtrip():
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(1, 300))
        # a piecewise-constant column: a few random runs (the non-freq lane shape).
        col = np.zeros(n, dtype=np.int64)
        i = 0
        while i < n:
            run = int(rng.integers(1, 40))
            col[i : i + run] = int(rng.integers(0, 256))
            i += run
        starts = DJ.change_points_kernel(col)
        assert np.array_equal(starts, _ref_change_points(col))
        values = col[starts]
        # step_lane(change_points(col), col[change_points], n) reproduces col byte-exact.
        got = DJ.step_lane_kernel(starts, values, n)
        assert np.array_equal(got, _ref_step_lane(starts, values, n))
        assert np.array_equal(got, col)


def test_step_lane_kernel_holds_tail_and_empty():
    # the held tail past the last change point fills to nframes.
    starts = np.array([0, 3], dtype=np.int64)
    values = np.array([7, 9], dtype=np.int64)
    got = DJ.step_lane_kernel(starts, values, 6)
    assert np.array_equal(got, np.array([7, 7, 7, 9, 9, 9], dtype=np.int64))
    # an empty program renders all-zero (an unused lane, never admitted).
    empty = DJ.step_lane_kernel(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 4
    )
    assert np.array_equal(empty, np.zeros(4, dtype=np.int64))


def _ref_ramp_segments(col, modulus):
    """Pure-Python reference for ``ramp_segments_kernel``: greedy maximal constant-step
    wrapping-ramp runs.  A segment from ``i`` steps ``col[i]`` by the fixed step
    ``col[i+1]-col[i] (mod modulus)`` as long as it reproduces ``col``."""
    n = len(col)
    starts, seeds, steps = [], [], []
    i = 0
    while i < n:
        starts.append(i)
        seeds.append(int(col[i]) % modulus)
        if i + 1 >= n:
            steps.append(0)
            break
        step = (int(col[i + 1]) - int(col[i])) % modulus
        steps.append(step)
        val = int(col[i])
        j = i
        while j < n and (val % modulus) == (int(col[j]) % modulus):
            val = (val + step) % modulus
            j += 1
        i = j
    return (
        np.asarray(starts, dtype=np.int64),
        np.asarray(seeds, dtype=np.int64),
        np.asarray(steps, dtype=np.int64),
    )


def _ref_ramp_render(starts, seeds, steps, nframes, modulus):
    """Pure-Python reference for ``ramp_render_kernel``: step the recurrence per segment."""
    out = np.zeros(nframes, dtype=np.int64)
    for k in range(len(starts)):
        a = max(0, int(starts[k]))
        b = int(starts[k + 1]) if k + 1 < len(starts) else nframes
        b = min(b, nframes)
        val = int(seeds[k]) % modulus
        step = int(steps[k]) % modulus
        for i in range(a, b):
            out[i] = val
            val = (val + step) % modulus
    return out


def test_ramp_segments_and_render_roundtrip_byte_exact():
    rng = np.random.default_rng(23)
    for modulus in (256, 1 << 16):
        for _ in range(20):
            n = int(rng.integers(1, 400))
            # a sweep column: a few constant-step wrapping ramps with note-gated resets
            # (the PW-sweep shape -- ramps that wrap at the modulus and restart).
            col = np.zeros(n, dtype=np.int64)
            i = 0
            val = int(rng.integers(0, modulus))
            while i < n:
                run = int(rng.integers(1, 60))
                step = int(rng.integers(0, modulus))
                for _k in range(run):
                    if i >= n:
                        break
                    col[i] = val
                    val = (val + step) % modulus
                    i += 1
                val = int(rng.integers(0, modulus))  # a reset at the segment boundary
            starts, seeds, steps, nseg = DJ.ramp_segments_kernel(col, modulus)
            r_starts, r_seeds, r_steps = _ref_ramp_segments(col, modulus)
            assert nseg == len(r_starts)
            assert np.array_equal(starts, r_starts)
            assert np.array_equal(seeds, r_seeds)
            assert np.array_equal(steps, r_steps)
            # the generator renders the column byte-exact (the fit/render are inverses).
            got = DJ.ramp_render_kernel(starts, seeds, steps, n, modulus)
            assert np.array_equal(
                got, _ref_ramp_render(starts, seeds, steps, n, modulus)
            )
            assert np.array_equal(got, col)


def test_ramp_segments_single_clean_ramp_is_one_segment():
    # a clean +88 (mod 2^16) ramp with no reset is ONE segment (seed, step) -- the
    # derive-don't-store win: a long sweep collapses to two ints.
    n = 100
    col = np.zeros(n, dtype=np.int64)
    val = 0x0040
    for i in range(n):
        col[i] = val
        val = (val + 88) % (1 << 16)
    starts, seeds, steps, nseg = DJ.ramp_segments_kernel(col, 1 << 16)
    assert nseg == 1 and int(seeds[0]) == 0x0040 and int(steps[0]) == 88
    got = DJ.ramp_render_kernel(starts, seeds, steps, n, 1 << 16)
    assert np.array_equal(got, col)


def test_idxr_score_kernel_finds_in_image_targets():
    ram = np.zeros(65536, dtype=np.uint8)
    read_play = np.zeros(65536, dtype=np.uint8)
    # a 4-entry ascending pointer table at 0x2000 (lo) / 0x2001 (hi), stride 2,
    # targets 0x3000, 0x3100, 0x3200, 0x3300 -- all in [0x1000, 0x4000).
    base = 0x2000
    targets = [0x3000, 0x3100, 0x3200, 0x3300]
    for i, t in enumerate(targets):
        ram[base + 2 * i] = t & 0xFF
        ram[base + 2 * i + 1] = (t >> 8) & 0xFF
        read_play[t] = 1
    idxr = np.array([[0x1234, base, 2, 0, 3, 999]], dtype=np.int64)
    scores = DJ.idxr_score_kernel(idxr, ram, read_play, 0x1000, 0x4000, 0)
    assert scores[0, 0] == 4  # n_valid
    assert scores[0, 1] == 4  # read_cov
    assert scores[0, 2] == 4  # n
    assert scores[0, 3] == 1  # ascending


# --------------------------------------------------------------------------- #
# Bank-candidate scan kernels (the IDXR perf-gate hot loops) vs reference.
# --------------------------------------------------------------------------- #
def _ref_walk_pattern_bytes(ram, ptr, boundaries, max_bytes=0x400):
    # mirrors structure_recover._walk_pattern_bytes
    for k in range(max_bytes):
        b = int(ram[(ptr + k) & 0xFFFF])
        if boundaries[b] == DJ.K_EOP:
            return k + 1
    return 0


def _ref_read_extent(read_play, base, lo_img, hi_img, max_bytes=0x400):
    # mirrors structure_recover._read_extent
    k = 0
    while (
        k < max_bytes
        and lo_img <= (base + k) < hi_img
        and read_play[(base + k) & 0xFFFF]
    ):
        k += 1
    return k


def test_bank_eop_lengths_kernel_matches_reference():
    rng = np.random.default_rng(23)
    bnd, _, _ = _newplayer_grammar()
    for _ in range(40):
        ram = rng.integers(0, 256, size=65536, dtype=np.uint8)
        ptrs = rng.integers(0x0800, 0xF000, size=int(rng.integers(2, 30))).astype(
            np.int64
        )
        # seed an EOP within range for some pointers, leave others unterminated
        for j, p in enumerate(ptrs):
            if j % 3 == 0:
                ram[(int(p) + int(rng.integers(1, 60))) & 0xFFFF] = 0x7F
        got = DJ.bank_eop_lengths_kernel(ram.astype(np.int64), ptrs, bnd, 0x400)
        ref = [_ref_walk_pattern_bytes(ram, int(p), bnd) for p in ptrs]
        assert list(int(x) for x in got) == ref


def test_bank_read_extents_kernel_matches_reference():
    rng = np.random.default_rng(29)
    lo_img, hi_img = 0x1000, 0xC000
    for _ in range(40):
        read_play = rng.integers(0, 2, size=65536, dtype=np.uint8)
        ptrs = rng.integers(0x0800, 0xF000, size=int(rng.integers(2, 30))).astype(
            np.int64
        )
        got = DJ.bank_read_extents_kernel(read_play, ptrs, lo_img, hi_img, 0x400)
        ref = [_ref_read_extent(read_play, int(p), lo_img, hi_img) for p in ptrs]
        assert list(int(x) for x in got) == ref
