"""Byte-exact parity: the backward-LZ + split-pointer-table njit kernels == their
pure-Python references.

The cover path's dominant cost (cProfile on a digi-dense tune: ~250s of a 300s
recovery) is :func:`serialize._lz_emit_t` over a long bundled-row stream, and the
legacy structure path's :func:`structure_recover.discover_pointer_table` is an
O(image . gap) double scan.  Both now run on numba kernels (:mod:`bacc._lz_njit`,
:func:`bacc.generic._discover_njit.split_pointer_table_kernel`).  These are PURE
speedups: the id stream / the chosen table must be byte-for-byte identical to the
pre-kernel pure-Python reference.  These tests pin that contract over a randomized
grid (the corpus-budget gate's committed token streams pin it on real tunes too)."""

import numpy as np
import pytest

import preframr_tokens.bacc.serialize as S
from preframr_tokens.bacc import _lz_njit as LK
from preframr_tokens.bacc.generic import _discover_njit as DJ


# ---------------------------------------------------------------------------
# An event-shaped codec (dur, ref, base, seed): a single transposable positive
# ``base`` lane, ``base < 0`` an absolute lane (no shift) -- the same contract the
# real event/row codecs expose to the LZ.
# ---------------------------------------------------------------------------
def _event_codec():
    def lit(out, ev):
        S._wu(out, ev[0])
        S._wu(out, ev[1])
        S._wi(out, ev[2])
        S._wu(out, ev[3])

    def cost(ev):
        tmp = []
        lit(tmp, ev)
        return len(tmp)

    def delta(a, b):
        if a[0] != b[0] or a[1] != b[1] or a[3] != b[3]:
            return None
        if a[2] < 0 or b[2] < 0:
            return None
        return b[2] - a[2]

    def eq_key(ev):
        return (ev[0], ev[1], ev[2], ev[3])

    def xpose_key(ev):
        free = "_FREE" if ev[2] >= 0 else ev[2]
        return (ev[0], ev[1], ev[3], free)

    def xpose_vec(ev):
        return [ev[2] if ev[2] >= 0 else None]

    return lit, cost, delta, eq_key, xpose_key, xpose_vec


def _random_events(rng, n):
    items = []
    for _ in range(n):
        items.append(
            (
                int(rng.integers(0, 3)),
                int(rng.integers(0, 2)),
                int(rng.integers(-1, 5)),  # -1 = absolute lane
                int(rng.integers(0, 2)),
            )
        )
    # inject a transposed phrase repeat so the TRANSPOSE branch is exercised
    if n > 10 and rng.random() < 0.6:
        k = int(rng.integers(3, min(8, n // 2)))
        shift = int(rng.integers(1, 4))
        src = items[2 : 2 + k]
        ins = [(d, r, (b + shift) if b >= 0 else b, s) for (d, r, b, s) in src]
        pos = int(rng.integers(2 + k, n))
        items[pos:pos] = ins
    return items


def _emit(items, *, use_kernel, with_xpose):
    lit, cost, delta, eq_key, xpose_key, xpose_vec = _event_codec()
    out = []
    S._lz_emit_t(
        out,
        items,
        cost,
        lit,
        delta_of=delta if with_xpose else None,
        eq_key=eq_key,
        xpose_key=xpose_key if with_xpose else None,
        xpose_vec=xpose_vec if use_kernel else None,
    )
    return out


@pytest.mark.parametrize("with_xpose", [True, False])
def test_lz_kernel_matches_python_reference(with_xpose):
    """The njit LZ plan (``xpose_vec`` supplied) emits the SAME id stream as the
    indexed Python search (``xpose_vec=None``) -- over REPEAT-only and REPEAT+TRANSPOSE.
    """
    rng = np.random.default_rng(20240613)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        items = _random_events(rng, n)
        ref = _emit(items, use_kernel=False, with_xpose=with_xpose)
        ker = _emit(items, use_kernel=True, with_xpose=with_xpose)
        assert ref == ker, (items, ref, ker)


def test_lz_kernel_handles_constant_and_overlapping_runs():
    """A constant stream (a run that overlaps its source, length > offset) and an
    all-transposable ramp both encode identically on the kernel and the reference."""
    const = [(1, 0, 3, 0)] * 40
    ramp = [(1, 0, i % 7, 0) for i in range(40)]
    for items in (const, ramp):
        for with_xpose in (True, False):
            ref = _emit(items, use_kernel=False, with_xpose=with_xpose)
            ker = _emit(items, use_kernel=True, with_xpose=with_xpose)
            assert ref == ker


# ---------------------------------------------------------------------------
# split_pointer_table_kernel == the original Python double-scan.
# ---------------------------------------------------------------------------
def _ref_pointer_table(ram, read_play, lo_img, hi_img):
    best = None
    for hi_base in range(lo_img, hi_img - 2):
        for gap in range(4, 96):
            lo_base = hi_base - gap
            if lo_base < lo_img:
                continue
            n = gap
            if hi_base + n > hi_img:
                continue
            los = ram[lo_base : lo_base + n].astype(np.int64)
            his = ram[hi_base : hi_base + n].astype(np.int64)
            ptrs = los | (his << 8)
            if not ((ptrs >= hi_base + n) & (ptrs < hi_img)).all():
                continue
            if not np.all(np.diff(ptrs) > 0):
                continue
            if len(np.unique(his)) > 6:
                continue
            if read_play[ptrs].sum() < 1:
                continue
            if best is None or n > best[2]:
                best = (lo_base, hi_base, n)
    return best


def test_split_pointer_table_kernel_matches_reference():
    """The njit split-pointer-table scan picks the SAME (lo, hi, n) as the Python
    double loop over randomized images (with and without a planted ascending table)."""
    rng = np.random.default_rng(11)
    for _ in range(60):
        lo_img = int(rng.integers(0x1000, 0x2000))
        span = int(rng.integers(200, 600))
        hi_img = lo_img + span
        ram = np.zeros(65536, dtype=np.uint8)
        ram[lo_img:hi_img] = rng.integers(0, 256, size=span, dtype=np.uint8)
        read_play = np.zeros(65536, dtype=np.uint8)
        if rng.random() < 0.8:
            nn = int(rng.integers(5, 40))
            hb = lo_img + int(rng.integers(60, max(61, span - nn - 10)))
            lb = hb - nn
            if lb >= lo_img and hb + nn <= hi_img:
                base_t = hb + nn + int(rng.integers(0, 5))
                targets = sorted(
                    set(int(rng.integers(base_t, hi_img)) for _ in range(nn))
                )
                if len(targets) == nn:
                    for i, t in enumerate(targets):
                        ram[lb + i] = t & 0xFF
                        ram[hb + i] = (t >> 8) & 0xFF
                        read_play[t] = 1
        ref = _ref_pointer_table(ram, read_play, lo_img, hi_img)
        out = DJ.split_pointer_table_kernel(ram, read_play, lo_img, hi_img)
        ker = (int(out[0]), int(out[1]), int(out[2])) if out[2] != 0 else None
        assert ref == ker


# ---------------------------------------------------------------------------
# token_lz_plan_kernel == tracker_serialize._lz_tokens (the flat-token TOKEN-LZ mode).
# ---------------------------------------------------------------------------
def _ref_lz_tokens(stream, min_copy=3):
    out = []
    n = len(stream)
    table = {}
    i = 0
    while i < n:
        best_len, best_off = 0, 0
        if i + 3 <= n:
            key = (stream[i], stream[i + 1], stream[i + 2])
            for pos in reversed(table.get(key, ())):
                off = i - pos
                length = 0
                while i + length < n and stream[pos + length] == stream[i + length]:
                    length += 1
                    if length >= 4095:
                        break
                if length > best_len:
                    best_len, best_off = length, off
                if best_len >= 512:
                    break
        cost_copy = 1 + S._u_len(best_off) + S._u_len(best_len)
        if best_len >= min_copy and cost_copy < best_len:
            out.append(S.REPEAT)
            S._wu(out, best_off)
            S._wu(out, best_len)
            step = best_len
        else:
            out.append(stream[i])
            step = 1
        for j in range(i, min(i + step, n - 2)):
            table.setdefault((stream[j], stream[j + 1], stream[j + 2]), []).append(j)
        i += step
    return out


def _ker_lz_tokens(stream, min_copy=3):
    if not stream:
        return []
    plan = LK.token_lz_plan_kernel(np.asarray(stream, dtype=np.int64), min_copy)
    out = []
    for is_copy, off, length in plan:
        if is_copy:
            out.append(S.REPEAT)
            S._wu(out, int(off))
            S._wu(out, int(length))
        else:
            out.append(int(stream[int(off)]))
    return out


def _random_token_stream(rng):
    length = int(rng.integers(0, 200))
    alpha = int(rng.integers(2, 18))  # small alphabet so 3-grams recur
    s = [int(rng.integers(0, alpha)) for _ in range(length)]
    if length > 20 and rng.random() < 0.7:
        k = int(rng.integers(4, min(20, length // 2)))
        src = s[1 : 1 + k]
        pos = int(rng.integers(1 + k, length))
        s[pos:pos] = src
    return s


def test_token_lz_kernel_matches_reference():
    """The njit TOKEN-LZ plan emits the SAME id stream as the Python hash-of-3-grams
    greedy backward-LZ over a flat value-token stream (the pool / token-LZ-mode path).
    """
    rng = np.random.default_rng(424242)
    for _ in range(300):
        s = _random_token_stream(rng)
        assert _ref_lz_tokens(s) == _ker_lz_tokens(s)


def _ref_backward_lz(tokens, min_match=3, window=4096):
    i, n = 0, len(tokens)
    literals = matches = 0
    while i < n:
        best = 0
        lo = max(0, i - window)
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


def test_backward_lz_counts_kernel_matches_reference():
    """The njit backward-LZ COUNTS ``(literals, matches)`` identically to the Python
    O(n . window) double scan (the structure token-budget scorer)."""
    rng = np.random.default_rng(515151)
    for _ in range(300):
        s = _random_token_stream(rng)
        ref = _ref_backward_lz(s)
        if not s:
            ker = (0, 0)
        else:
            lit, mat = LK.backward_lz_counts_kernel(
                np.asarray(s, dtype=np.int64), 3, 4096
            )
            ker = (int(lit), int(mat))
        assert ref == ker
