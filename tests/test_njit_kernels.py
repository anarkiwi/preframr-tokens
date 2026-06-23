"""Byte-exact parity: each numba-JIT kernel == its pure-Python reference.

The generic recovery's hot per-frame loops are njit'd for speed
(:mod:`preframr_tokens.bacc.generic._njit`).  This is a PURE speedup: every kernel
must reproduce the pre-njit pure-Python renderer byte-for-byte.  These tests pin
that contract by re-implementing each renderer as an independent pure-Python
reference and asserting equality across a randomized / swept parameter grid --
the explicit ``& 0xFFFF`` / ``& 0xFFF`` / ``& 0xFF`` chip masks keep the
``int64`` kernels in lock-step with arbitrary-precision Python (no wraparound
divergence), and these tests verify that, they do not assume it.
"""

import numpy as np
import pytest

from preframr_tokens.bacc.generic import _njit
from preframr_tokens.bacc.generic import _render_njit as _rnj
from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import cover as C


def _seg_lens():
    return (1, 2, 3, 5, 13, 31, 64, 128, 384)


# ---------------------------------------------------------------------------
# Pure-Python references (the pre-njit renderers, byte-for-byte).
# ---------------------------------------------------------------------------
def ref_match_prefix(rend, seg):
    length = min(len(rend), len(seg))
    eq = rend[:length] == seg[:length]
    if eq.all():
        return length
    return int(np.argmin(eq))


def ref_tri_phase_seq(ctr0, seg_len):
    osc = (ctr0 + np.arange(seg_len, dtype=np.int64)) & 7
    return np.where(osc >= 4, osc ^ 7, osc)


def ref_render_arp(seg_len, freqs, ctr0, dwell):
    out = np.empty(seg_len, dtype=np.int64)
    period = len(freqs)
    for i in range(seg_len):
        step = (ctr0 + i) // dwell
        out[i] = freqs[step % period]
    return out


def ref_render_accum(seg_len, v0, rate, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val += rate
    return out


def ref_render_wrapaccum(seg_len, v0, rate, lo_b, hi_b):
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


def ref_render_glide(seg_len, n0, step, dwell, lead, note_table):
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        k = 0 if i < lead else (i - lead) // dwell
        idx = (n0 + step * k) & 0xFF
        out[i] = note_table[idx] if 0 <= idx < len(note_table) else 0
    return out


def ref_hi_overlay(base_lane, sfh0, par, ctr0):
    length = len(base_lane)
    out = np.empty(length, dtype=np.int64)
    sfh = sfh0 & 0xFF
    for i in range(length):
        lo = int(base_lane[i]) & 0xFF
        hi = (int(base_lane[i]) >> 8) & 0xFF
        if ((ctr0 + i) & 1) == par and sfh != 0:
            old = sfh
            sfh = (sfh - 1) & 0xFF
            hi = old
        out[i] = lo | (hi << 8)
    return out


def ref_render_additive_pw(seg_len, p0, pulsevalue, carry_seq, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    hi = p0 & ~0xFF
    lo = p0 & 0xFF
    for i in range(seg_len):
        out[i] = (hi | lo) & width_mask
        carry = int(carry_seq[i]) if i < len(carry_seq) else 0
        lo = (lo + pulsevalue + carry) & 0xFF
    return out


def ref_render_pingpong(seg_len, v0, rate, lo_b, hi_b, dwell, d0, dir0):
    out = np.empty(seg_len, dtype=np.int64)
    val, dwell_left, direction = v0, d0, dir0
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


def ref_render_pingfold(seg_len, step, frac, lo, hi, acc0, dir0):
    out = np.empty(seg_len, dtype=np.int64)
    acc, direction = acc0, dir0
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


def ref_render_vibreflect(seg_len, center, speed, cmpvalue, delay, vibtime0):
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


def ref_render_decay(seg_len, v0, rate, every, ctr0):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val
        if (ctr0 + i + 1) % every == 0:
            val = (val - rate) & 0xFFFF
    return out


def ref_render_dwell_accum(seg_len, v0, rate, dwell, lead):
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


def ref_render_maskaccum(seg_len, v0, rate, mask, width_mask):
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    period = len(mask)
    for i in range(seg_len):
        out[i] = val & width_mask
        if mask[i % period]:
            val = (val + rate) & width_mask
    return out


def ref_render_tablewalk(seg_len, table, ctr0):
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        out[i] = table[(ctr0 + i) % period]
    return out


def ref_render_ratewalk(seg_len, v0, rate_table, ctr0, width_mask):
    period = len(rate_table)
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[(ctr0 + i) % period]) & width_mask
    return out


def ref_render_dwellratewalk(seg_len, v0, rate_table, dwell, ctr0, width_mask):
    period = len(rate_table)
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[((ctr0 + i) // dwell) % period]) & width_mask
    return out


def ref_render_tablewalk_lead(seg_len, lead, value0, table, ctr0):
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        out[i] = value0 if i < lead else table[(ctr0 + i - lead) % period]
    return out


def ref_render_wavetable_ptr(seg_len, table, phase, advance):
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    ptr = phase % period
    for i in range(seg_len):
        if i > 0 and advance[(i - 1) % len(advance)]:
            ptr = (ptr + 1) % period
        out[i] = table[ptr]
    return out


# ---------------------------------------------------------------------------
# Parity tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_match_prefix_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(20):
        n = rng.randint(1, 200)
        m = rng.randint(1, 200)
        # small alphabet so collisions / matches are common
        a = rng.randint(0, 4, size=n).astype(np.int64)
        b = rng.randint(0, 4, size=m).astype(np.int64)
        assert int(_njit.match_prefix(a, b)) == ref_match_prefix(a, b)
    # identical arrays => full common length
    a = rng.randint(0, 0x10000, size=50).astype(np.int64)
    assert int(_njit.match_prefix(a, a.copy())) == len(a)


@pytest.mark.parametrize("ctr0", range(8))
@pytest.mark.parametrize("seg_len", _seg_lens())
def test_tri_phase_seq_parity(ctr0, seg_len):
    assert np.array_equal(
        _njit.tri_phase_seq(np.int64(ctr0), seg_len),
        ref_tri_phase_seq(ctr0, seg_len),
    )


@pytest.mark.parametrize("seed", range(8))
def test_render_arp_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(40):
        period = rng.randint(2, 8)
        freqs = rng.randint(0, 0x10000, size=period).astype(np.int64)
        ctr0 = rng.randint(0, 256)
        dwell = rng.randint(1, 6)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_arp(
            seg_len, freqs, len(freqs), np.int64(ctr0), np.int64(dwell)
        )
        ref = ref_render_arp(seg_len, freqs, ctr0, dwell)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_accum_parity(seed):
    rng = np.random.RandomState(seed)
    for width in (0xFFFF, 0xFFF):
        for _ in range(40):
            v0 = rng.randint(0, width + 1)
            rate = rng.randint(-4000, 4000)
            seg_len = rng.choice(_seg_lens())
            got = _njit.render_accum(
                seg_len, np.int64(v0), np.int64(rate), np.int64(width)
            )
            ref = ref_render_accum(seg_len, v0, rate, width)
            assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_wrapaccum_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(60):
        lo = rng.randint(0, 0x4000)
        hi = lo + rng.randint(0x40, 0xC000)
        rate = rng.choice([-1, 1]) * rng.randint(1, 0x1000)
        v0 = rng.randint(lo, hi)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_wrapaccum(
            seg_len, np.int64(v0), np.int64(rate), np.int64(lo), np.int64(hi)
        )
        ref = ref_render_wrapaccum(seg_len, v0, rate, lo, hi)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_glide_parity(seed):
    rng = np.random.RandomState(seed)
    nt = rng.randint(0, 0x10000, size=128).astype(np.int64)
    for _ in range(50):
        n0 = rng.randint(0, 128)
        step = rng.choice([-2, -1, 1, 2])
        dwell = rng.randint(1, 8)
        lead = rng.randint(0, 10)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_glide(
            seg_len, np.int64(n0), np.int64(step), np.int64(dwell), np.int64(lead), nt
        )
        ref = ref_render_glide(seg_len, n0, step, dwell, lead, nt)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_hi_overlay_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(40):
        n = rng.choice(_seg_lens())
        base = rng.randint(0, 0x10000, size=n).astype(np.int64)
        sfh0 = rng.randint(0, 256)
        par = rng.randint(0, 2)
        ctr0 = rng.randint(0, 256)
        got = _njit.hi_overlay(base, np.int64(sfh0), np.int64(par), np.int64(ctr0))
        ref = ref_hi_overlay(base, sfh0, par, ctr0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_additive_pw_parity(seed):
    rng = np.random.RandomState(seed)
    for width in (0xFFF, 0xFFFF):
        for _ in range(40):
            p0 = rng.randint(0, width + 1)
            pulsevalue = rng.randint(0, 256)
            seg_len = int(rng.choice(_seg_lens()))
            clen = rng.randint(1, seg_len + 1)
            carry = rng.randint(0, 2, size=clen).astype(np.int64)
            got = _njit.render_additive_pw(
                seg_len, np.int64(p0), np.int64(pulsevalue), carry, np.int64(width)
            )
            ref = ref_render_additive_pw(seg_len, p0, pulsevalue, carry, width)
            assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_pingpong_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(80):
        lo = rng.randint(0, 0x2000)
        hi = lo + rng.randint(2, 0x800)
        rate = rng.randint(1, 0x80)
        v0 = rng.randint(lo, hi + 1)
        dwell = rng.randint(0, 12)
        d0 = rng.randint(0, dwell + 1)
        dir0 = rng.randint(0, 2)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_pingpong(
            seg_len,
            np.int64(v0),
            np.int64(rate),
            np.int64(lo - 1),
            np.int64(hi + 1),
            np.int64(dwell),
            np.int64(d0),
            np.int64(dir0),
        )
        ref = ref_render_pingpong(seg_len, v0, rate, lo - 1, hi + 1, dwell, d0, dir0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_pingfold_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(80):
        frac = rng.randint(0, 4)
        lo = 0
        hi = rng.randint(8, 256) << frac
        step = rng.randint(1, 32)
        acc0 = rng.randint(lo, hi + 1)
        dir0 = rng.choice([-1, 1])
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_pingfold(
            seg_len,
            np.int64(step),
            np.int64(frac),
            np.int64(lo),
            np.int64(hi),
            np.int64(acc0),
            np.int64(dir0),
        )
        ref = ref_render_pingfold(seg_len, step, frac, lo, hi, acc0, dir0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_vibreflect_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(60):
        center = rng.randint(0, 0x10000)
        speed = rng.randint(1, 0x100)
        cmpvalue = rng.randint(1, 0x40)
        delay = rng.randint(1, 10)
        vibtime0 = rng.randint(0, 0x40) * 2
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_vibreflect(
            seg_len,
            np.int64(center),
            np.int64(speed),
            np.int64(cmpvalue),
            np.int64(delay),
            np.int64(vibtime0),
        )
        ref = ref_render_vibreflect(seg_len, center, speed, cmpvalue, delay, vibtime0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_decay_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(60):
        v0 = rng.randint(0, 0x10000)
        rate = rng.randint(1, 0x1000)
        every = rng.randint(1, 5)
        ctr0 = rng.randint(0, every)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_decay(
            seg_len, np.int64(v0), np.int64(rate), np.int64(every), np.int64(ctr0)
        )
        ref = ref_render_decay(seg_len, v0, rate, every, ctr0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_dwell_accum_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(60):
        v0 = rng.randint(0, 0x10000)
        rate = rng.randint(-0x1000, 0x1000)
        dwell = rng.randint(1, 8)
        lead = rng.randint(0, 12)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_dwell_accum(
            seg_len, np.int64(v0), np.int64(rate), np.int64(dwell), np.int64(lead)
        )
        ref = ref_render_dwell_accum(seg_len, v0, rate, dwell, lead)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_maskaccum_parity(seed):
    rng = np.random.RandomState(seed)
    for width in (0xFFF, 0xFFFF):
        for _ in range(40):
            v0 = rng.randint(0, width + 1)
            rate = rng.randint(-0x800, 0x800)
            plen = rng.randint(1, 12)
            mask = rng.randint(0, 2, size=plen).astype(np.int64)
            if not mask.any():
                mask[0] = 1
            seg_len = rng.choice(_seg_lens())
            got = _njit.render_maskaccum(
                seg_len, np.int64(v0), np.int64(rate), mask, np.int64(width)
            )
            ref = ref_render_maskaccum(seg_len, v0, rate, mask, width)
            assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_tablewalk_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(50):
        plen = rng.randint(1, 48)
        table = rng.randint(0, 0x10000, size=plen).astype(np.int64)
        ctr0 = rng.randint(0, 256)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_tablewalk(seg_len, table, np.int64(ctr0))
        ref = ref_render_tablewalk(seg_len, table, ctr0)
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_ratewalk_parity(seed):
    rng = np.random.RandomState(seed)
    for width in (0xFFF, 0xFFFF):
        for _ in range(40):
            plen = rng.randint(1, 48)
            rate_table = rng.randint(-128, 128, size=plen).astype(np.int64)
            v0 = rng.randint(0, width + 1)
            ctr0 = rng.randint(0, 256)
            seg_len = rng.choice(_seg_lens())
            got = _njit.render_ratewalk(
                seg_len, np.int64(v0), rate_table, np.int64(ctr0), np.int64(width)
            )
            ref = ref_render_ratewalk(seg_len, v0, rate_table, ctr0, width)
            assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_dwellratewalk_parity(seed):
    rng = np.random.RandomState(seed)
    for width in (0xFFF, 0xFFFF):
        for _ in range(40):
            plen = rng.randint(2, 24)
            rate_table = rng.randint(-128, 128, size=plen).astype(np.int64)
            v0 = rng.randint(0, width + 1)
            dwell = rng.randint(1, 16)
            ctr0 = rng.randint(0, 256)
            seg_len = rng.choice(_seg_lens())
            got = _njit.render_dwellratewalk(
                seg_len,
                np.int64(v0),
                rate_table,
                np.int64(dwell),
                np.int64(ctr0),
                np.int64(width),
            )
            ref = ref_render_dwellratewalk(seg_len, v0, rate_table, dwell, ctr0, width)
            assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", range(8))
def test_render_tablewalk_lead_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(50):
        plen = rng.randint(1, 24)
        table = rng.randint(0, 0x10000, size=plen).astype(np.int64)
        lead = rng.randint(0, 12)
        value0 = rng.randint(0, 0x10000)
        ctr0 = rng.randint(0, 256)
        seg_len = rng.choice(_seg_lens())
        got = _njit.render_tablewalk_lead(
            seg_len, np.int64(lead), np.int64(value0), table, np.int64(ctr0)
        )
        ref = ref_render_tablewalk_lead(seg_len, lead, value0, table, ctr0)
        assert np.array_equal(got, ref)


def _ref_prefix_pingpong(seg):
    """Pre-fusion pure-Python ``_prefix_pingpong`` (the nested render+match loop)."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < A._MINRUN:
        return None
    base = int(seg[0])
    diffs = np.diff(seg)
    nonzero = np.abs(diffs[diffs != 0])
    if len(nonzero) == 0:
        return None
    lo_b, hi_b = int(seg.min()), int(seg.max())
    bound_pairs = ((lo_b - 1, hi_b + 1), (lo_b, hi_b))
    best = None
    for rate in sorted(set(nonzero.tolist()))[:5]:
        for refl_lo, refl_hi in bound_pairs:
            for dwell in range(0, 12):
                for d0 in range(dwell + 1):
                    for dir0 in (0, 1):
                        rend = ref_render_pingpong(
                            length, base, int(rate), refl_lo, refl_hi, dwell, d0, dir0
                        )
                        match = ref_match_prefix(rend, seg)
                        if match >= A._MINRUN and (best is None or match > best[0]):
                            best = (
                                match,
                                "pingpong",
                                {
                                    "v0": base,
                                    "rate": int(rate),
                                    "lo": refl_lo + 1,
                                    "hi": refl_hi - 1,
                                    "dwell": dwell,
                                    "d0": d0,
                                    "dir0": dir0,
                                },
                            )
                            if match == length:
                                return best
    return best


@pytest.mark.parametrize("seed", range(12))
def test_fused_prefix_pingpong_matches_reference(seed):
    # the fused njit pingpong search is byte-identical to the pre-fusion nested
    # render+match loop -- same winner, same params, same order / tie-breaking.
    rng = np.random.RandomState(seed)
    for _ in range(120):
        length = rng.randint(3, 90)
        kind = rng.randint(0, 3)
        if kind == 0:  # a genuine reflect triangle
            lo = rng.randint(0, 0x800)
            seg = A.render_pingpong(
                length,
                rng.randint(lo, lo + 0x400),
                rng.randint(1, 40),
                lo,
                lo + rng.randint(0x40, 0x800),
                rng.randint(0, 8),
                0,
                rng.randint(0, 2),
            )
        elif kind == 1:  # structureless
            seg = rng.randint(0, 30, size=length).astype(np.int64)
        else:  # a wrapping ramp
            seg = (rng.randint(0, 5, size=length).cumsum() % 50).astype(np.int64)
        assert A._prefix_pingpong(seg) == _ref_prefix_pingpong(seg)


@pytest.mark.parametrize("seed", range(8))
def test_render_wavetable_ptr_parity(seed):
    rng = np.random.RandomState(seed)
    for _ in range(50):
        plen = rng.randint(3, 32)
        table = rng.randint(0, 0x10000, size=plen).astype(np.int64)
        alen = rng.randint(1, 16)
        advance = rng.randint(0, 2, size=alen).astype(np.int64)
        phase = rng.randint(0, plen)
        seg_len = int(rng.choice(_seg_lens()))
        got = _njit.render_wavetable_ptr(seg_len, table, np.int64(phase), advance)
        ref = ref_render_wavetable_ptr(seg_len, table, phase, advance)
        assert np.array_equal(got, ref)


def _ref_prefix_ratewalk(seg, width_mask=0xFFFF, maxp=48, minrun=8):
    """Pre-fusion pure-Python ``_prefix_ratewalk`` (the render-per-period loop)."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    best = None
    for period in range(1, min(maxp, max(1, len(dsign))) + 1):
        if len(dsign) < period:
            continue
        table = dsign[:period].tolist()
        if not any(table):
            continue
        rend = ref_render_ratewalk(length, int(seg[0]), table, 0, width_mask)
        match = ref_match_prefix(rend, seg)
        if match >= max(minrun, 2 * period) and (best is None or match > best[0]):
            best = (
                match,
                "ratewalk",
                {"v0": int(seg[0]), "rate_table": table, "width": width_mask},
            )
        if best and best[0] == length:
            break
    return best


@pytest.mark.parametrize("seed", range(12))
def test_fused_prefix_ratewalk_matches_reference(seed):
    # The fused njit ratewalk search is byte-identical to the pre-fusion nested
    # render+match loop -- same winner, same params, same ascending-period order.
    rng = np.random.RandomState(seed)
    for _ in range(120):
        length = rng.randint(1, 400)
        wm = int(rng.choice([0xFF, 0xFFF, 0xFFFF]))
        kind = rng.randint(0, 5)
        if kind == 0:  # structureless noise
            seg = rng.randint(0, wm + 1, size=length).astype(np.int64)
        elif kind == 1:  # a genuine period-P signed-rate loop
            period = rng.randint(1, 50)
            tbl = rng.randint(-8, 9, size=period).astype(np.int64)
            v = int(rng.randint(0, wm + 1))
            out = []
            for i in range(length):
                out.append(v & wm)
                v = (v + int(tbl[i % period])) & wm
            seg = np.asarray(out, dtype=np.int64)
        elif kind == 2:  # a loop that breaks to noise partway
            period = rng.randint(1, 12)
            tbl = rng.randint(-5, 6, size=period).astype(np.int64)
            v = int(rng.randint(0, wm + 1))
            out = []
            for i in range(length):
                out.append(v & wm)
                v = (v + int(tbl[i % period])) & wm
            seg = np.asarray(out, dtype=np.int64)
            if length:
                br = rng.randint(0, length)
                seg[br:] = rng.randint(0, wm + 1, size=length - br)
        elif kind == 3:  # a single-rate wrapping ramp
            r = int(rng.randint(1, 40))
            v = int(rng.randint(0, wm + 1))
            seg = np.asarray([(v + r * i) & wm for i in range(length)], dtype=np.int64)
        else:  # near-flat small amplitude
            seg = (
                (int(rng.randint(0, wm + 1)) + rng.randint(-2, 3, size=length)) & wm
            ).astype(np.int64)
        for maxp in (48, 24, 12):
            for minrun in (8, 24):
                assert A._prefix_ratewalk(
                    seg, wm, maxp, minrun
                ) == _ref_prefix_ratewalk(seg, wm, maxp, minrun)


def _ref_prefix_maskaccum_stall(
    seg, width_mask=0xFFFF, maxp=24, minrun=12, mincycles=3
):
    """Pre-fusion pure-Python ``_prefix_maskaccum_stall`` (render-per-period loop)."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    nonzero = dsign[dsign != 0]
    if len(nonzero) < 3:
        return None
    vals, cnts = np.unique(nonzero, return_counts=True)
    rate = int(vals[np.argmax(cnts)])
    advance = (dsign == rate).astype(int)
    best = None
    for period in range(1, min(maxp, len(advance)) + 1):
        if len(advance) < mincycles * period:
            continue
        mask = advance[:period].tolist()
        steps = sum(mask)
        if steps < 1 or (period > 1 and steps == period):
            continue
        rend = ref_render_maskaccum(length, int(seg[0]), rate, mask, width_mask)
        match = ref_match_prefix(rend, seg)
        if match >= max(minrun, mincycles * period) and (
            best is None or match > best[0]
        ):
            best = (
                match,
                "maskaccum",
                {"v0": int(seg[0]), "rate": rate, "mask": mask, "width": width_mask},
            )
        if best and best[0] == length:
            break
    return best


@pytest.mark.parametrize("seed", range(12))
def test_fused_prefix_maskaccum_stall_matches_reference(seed):
    # The fused njit periodic-stall search is byte-identical to the pre-fusion
    # render+match loop -- same winner, same params, same mask gate / order.
    rng = np.random.RandomState(seed)
    for _ in range(120):
        length = rng.randint(1, 400)
        wm = int(rng.choice([0xFF, 0xFFF, 0xFFFF]))
        kind = rng.randint(0, 4)
        if kind == 0:  # structureless noise
            seg = rng.randint(0, wm + 1, size=length).astype(np.int64)
        elif kind == 1:  # a genuine rate stepped on a periodic advance mask
            period = rng.randint(2, 26)
            mask = rng.randint(0, 2, size=period)
            if mask.sum() == 0:
                mask[0] = 1
            rate = int(rng.randint(-20, 21)) or 3
            v = int(rng.randint(0, wm + 1))
            out = []
            for i in range(length):
                out.append(v & wm)
                if mask[i % period]:
                    v = (v + rate) & wm
            seg = np.asarray(out, dtype=np.int64)
        elif kind == 2:  # a plain accum (all-advance, must be rejected)
            r = int(rng.randint(1, 25))
            v = int(rng.randint(0, wm + 1))
            seg = np.asarray([(v + r * i) & wm for i in range(length)], dtype=np.int64)
        else:  # near-constant
            seg = np.full(length, int(rng.randint(0, wm + 1)), dtype=np.int64)
        for maxp in (24, 12):
            for minrun in (12, 6):
                for mincycles in (3, 2):
                    assert A._prefix_maskaccum_stall(
                        seg, wm, maxp, minrun, mincycles
                    ) == _ref_prefix_maskaccum_stall(seg, wm, maxp, minrun, mincycles)


def _ref_periodic_diff_period(diffw, pmax, mincyc):
    """Pre-fusion pure-Python period scan (the ``np.array_equal`` loop)."""
    m = len(diffw)
    for period in range(2, pmax + 1):
        need = mincyc * period
        if need <= m and np.array_equal(diffw[period:need], diffw[: need - period]):
            return period
    return None


@pytest.mark.parametrize("seed", range(12))
def test_periodic_diff_period_matches_reference(seed):
    # The fused njit long-period detector returns the SAME first qualifying period
    # the per-P np.array_equal scan in cover._periodic_candidates did (-1 / None).
    rng = np.random.RandomState(seed)
    for _ in range(200):
        m = rng.randint(1, 700)
        kind = rng.randint(0, 4)
        wm = int(rng.choice([0xFF, 0xFFF, 0xFFFF]))
        if kind == 0:
            diffw = rng.randint(0, wm + 1, size=m)
        elif kind == 1:  # periodic body
            period = rng.randint(2, 60)
            body = rng.randint(0, wm + 1, size=period)
            diffw = np.array([body[i % period] for i in range(m)])
        elif kind == 2:  # periodic then breaks
            period = rng.randint(2, 40)
            body = rng.randint(0, wm + 1, size=period)
            diffw = np.array([body[i % period] for i in range(m)])
            if m:
                br = rng.randint(0, m)
                diffw[br:] = rng.randint(0, wm + 1, size=m - br)
        else:
            diffw = np.full(m, int(rng.randint(0, wm + 1)))
        diffw = np.ascontiguousarray(diffw.astype(np.int64))
        for mincyc in (4, 2, 8):
            for maxp in (256, 64, 16):
                pmax = min(maxp, m // mincyc)
                got = _rnj.periodic_diff_period(diffw, int(pmax), int(mincyc))
                got = None if got < 0 else got
                assert got == _ref_periodic_diff_period(diffw, pmax, mincyc)


def test_periodic_candidates_unchanged_by_fused_detector():
    # cover._periodic_candidates (which now calls the fused detector) still yields the
    # SAME candidate list a genuine macro-loop produces (one render verifies the winner).
    rng = np.random.RandomState(99)
    for _ in range(60):
        wm = int(rng.choice([0xFFF, 0xFFFF]))
        period = rng.randint(8, 64)
        tbl = rng.randint(-6, 7, size=period).astype(np.int64)
        if not tbl.any():
            tbl[0] = 1
        length = period * rng.randint(5, 12) + rng.randint(0, period)
        v = int(rng.randint(0, wm + 1))
        out = []
        for i in range(length):
            out.append(v & wm)
            v = (v + int(tbl[i % period])) & wm
        seg = np.asarray(out, dtype=np.int64)
        got = C._periodic_candidates(seg, wm)
        # Re-render any returned candidate and confirm it covers a substantial prefix.
        for name, params in got:
            assert name == "citg"
            rend = A.render_citg(params, length)
            assert A._match_prefix(rend, seg) >= 4 * len(params["table"])
