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
