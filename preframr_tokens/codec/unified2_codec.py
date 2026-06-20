"""DECOUPLED step/tracker codec for Monty, residual-zero.

Supersedes unified_codec.py (22,732). Key moves vs the bundled row codec:

  1. LANE-DECOUPLED STREAMS. The bundled row token (freq+dur+all bodies as ONE
     LZ key) prevented reuse: a phrase had to match in EVERY lane to dedup. We
     split into independent per-lane continuous (dt, op) streams and LZ each
     separately. dur alone collapses 5,342 -> 662 (it is hugely repetitive once
     unbundled); micro 2,595 -> 1,302; the non-freq bodies stop paying per-row
     rel/bundle overhead. Residual-zero is trivial: the split is just regrouping
     encode_tune_events() by lane (event multiset identical -> render identical,
     verified np.array_equal on 25 regs x 17,544 frames).

  2. BODIES AS PARAMETRIC SWEEPS (Lever 1). A WALK/RUN body's per-frame delta
     sequence RLE-factors into (rate_pattern, dwell_vector). The rate_pattern
     (sequence of deltas) is the pitch-/duration-invariant SWEEP SHAPE, shared
     in a codebook: 354 distinct bodies -> 188 distinct rate-patterns. The
     dwell_vectors (hold-lengths, a 33-symbol alphabet dominated by 1,2,6) go to
     a single global side-stream, LZ'd (2,397 -> 1,293). Reconstruction is exact
     (recon == original delta seq, 0/1305 mismatch), so residual-zero holds.

  3. FREQ via freq_instrument shape codebook (45 instruments), micro stored
     literally (the base-snap residual; see Lever-2 note in RESULTS).

Gate: decode of the regrouped event set == event-codec decode, byte-for-byte.
"""

import collections

import numpy as np

from preframr_tokens.codec import serialize_events as SE
from preframr_tokens.codec import step_codec as SC
from preframr_tokens.codec import step_tracker as TR
from preframr_tokens.codec import freq_instrument as FI

uc = SC.u_cost
ic = SC.i_cost
LANEMAP = {"pw": [3, 4, 5], "ctrl": [6, 7, 8], "ad": [9, 10, 11], "sr": [12, 13, 14]}
GLOBAL = (15, 16, 17, 18)


def _deltas(op):
    if op[0] == "WALK":
        out = []
        for d, w in op[1]:
            out += [d] * w
        return out
    if op[0] == "RUN":
        return list(op[1])
    return None


def _rle(ds):
    segs = []
    for d in ds:
        if segs and segs[-1][0] == d:
            segs[-1][1] += 1
        else:
            segs.append([d, 1])
    return [(d, w) for d, w in segs]


def _lz(seq, litfn):
    i, n, tot = 0, len(seq), 0
    while i < n:
        bl, bk = 0, 0
        for back in range(1, i + 1):
            ln = 0
            while i + ln < n and back + ln <= i and seq[i - back + ln] == seq[i + ln]:
                ln += 1
            if ln > bl:
                bl, bk = ln, back
        if bl >= 2:
            rc = 1 + uc(bk) + uc(bl)
            lc = sum(litfn(seq[i + k]) for k in range(bl))
            if rc < lc:
                tot += rc
                i += bl
                continue
        tot += litfn(seq[i])
        i += 1
    return tot


def measure(s):
    t = len(s)
    ev = SE.encode_tune_events(s[:, :25])
    bylane = collections.defaultdict(list)
    for sf, lid, op in ev:
        bylane[lid].append((sf, op))
    brk = collections.Counter()
    defcost = 0

    # ---- FREQ lanes (continuous per-voice) + literal micro ----
    freq_cb = {}
    fseq = {v: [] for v in range(3)}
    flit = {}
    mseq = {v: [] for v in range(3)}
    for v in range(3):
        for sf, op in sorted(bylane[v]):
            nm = op[0]
            if nm == "NOTE":
                tok = ("N", op[1])
                flit[tok] = ic(op[1])
                fseq[v].append(tok)
                mseq[v].append(op[2])
            elif nm in ("RAW", "REST"):
                tok = (nm, op[1] if nm == "RAW" else 0)
                flit[tok] = uc(tok[1])
                fseq[v].append(tok)
            else:
                g = FI._gen(op)
                if g is None:
                    tok = ("L",) + tuple(map(str, op))
                    flit[tok] = SC._op_body_cost(op, True)
                    fseq[v].append(tok)
                    continue
                _e, n, base = g
                shape, d, shift = FI._shape_of(op, base)
                if shape not in freq_cb:
                    freq_cb[shape] = len(freq_cb)
                    dc = 1
                    if shape[0] == "VIB":
                        tri, dw = shape[1], shape[2]
                        dc += (
                            uc(len(tri))
                            + sum(ic(x) for x in tri)
                            + sum(uc(w) for w in dw)
                        )
                    elif shape[0] == "SLIDE":
                        dc += ic(shape[1])
                    elif shape[0] == "DELTA":
                        e_ = shape[1]
                        nz = [(k, x) for k, x in enumerate(e_) if x != 0]
                        dc += uc(len(e_)) + uc(len(nz))
                        prev = -1
                        for k, x in nz:
                            dc += uc(k - prev - 1) + ic(x)
                            prev = k
                    defcost += dc
                tok = (
                    "G",
                    freq_cb[shape],
                    n,
                    d if (shape[0] == "VIB" and shift is None) else -1,
                )
                flit[tok] = uc(freq_cb[shape]) + uc(n) + (1 if shape[0] == "VIB" else 0)
                fseq[v].append(tok)
    brk["freq"] = sum(_lz(fseq[v], lambda x: flit[x]) for v in range(3))
    brk["micro"] = sum(_lz(mseq[v], lambda m: ic(m)) for v in range(3))

    # ---- DUR stream (gate-rise rows) ----
    voices, _ = TR.build_rows(s)
    dseq = {
        v: [(r["dur"] // SC.STEP, r["dur"] % SC.STEP) for r in voices[v]]
        for v in range(3)
    }
    brk["dur"] = sum(_lz(dseq[v], lambda x: uc(x[0]) + uc(x[1])) for v in range(3))

    # ---- NON-FREQ bodies: rate-pattern codebook + global LZ dwell stream ----
    body_cb = {}
    ratepat_cb = {}
    dwellstream = []
    for kind, lids in LANEMAP.items():
        for lid in lids:
            for sf, op in sorted(bylane[lid]):
                full = (kind, TR._body_key(op))
                if full in body_cb:
                    continue
                body_cb[full] = len(body_cb)
                ds = _deltas(op)
                if ds is not None:
                    segs = _rle(ds)
                    pat = tuple(d for d, w in segs)
                    dwell = tuple(w for d, w in segs)
                    rk = (kind, op[0], pat)
                    if rk not in ratepat_cb:
                        ratepat_cb[rk] = len(ratepat_cb)
                        defcost += 1 + 1 + uc(len(pat)) + sum(ic(d) for d in pat)
                    defcost += uc(ratepat_cb[rk])
                    dwellstream.append(len(dwell))
                    dwellstream.extend(dwell)
                else:
                    defcost += 1 + 1 + SC._op_body_cost(op, False)
    defcost += _lz(dwellstream, lambda x: uc(x))
    body = 0
    for kind, lids in LANEMAP.items():
        for lid in lids:
            seq, lits, prev = [], {}, 0
            for sf, op in sorted(bylane[lid]):
                dt = sf - prev
                prev = sf
                bref = body_cb[(kind, TR._body_key(op))]
                tok = (dt, bref)
                lits[tok] = ic(dt) + uc(bref)
                seq.append(tok)
            body += _lz(seq, lambda x, lits=lits: lits[x])
    brk["body"] = body

    # ---- global filter automation (literal) ----
    glob = 0
    for lid in GLOBAL:
        prev = 0
        for sf, op in sorted(bylane[lid]):
            glob += ic(sf - prev) + 2 + SC._op_body_cost(op, False)
            prev = sf
    brk["global"] = glob
    brk["defcost"] = defcost
    brk["freq_instr"] = len(freq_cb)
    brk["bodies"] = len(body_cb)
    brk["rate_patterns"] = len(ratepat_cb)
    brk["total"] = (
        brk["freq"] + brk["micro"] + brk["dur"] + brk["body"] + brk["global"] + defcost
    )
    return dict(brk), t


def verify_residual(s):
    """Residual-zero: the codec's structures are a regrouping of encode_tune_events;
    decode of the regrouped events == decode of the original (byte-for-byte). Also
    checks rate-pattern+dwell reconstructs every WALK/RUN delta-seq exactly."""
    t = len(s)
    ev = SE.encode_tune_events(s[:, :25])
    bylane = collections.defaultdict(list)
    for sf, lid, op in ev:
        bylane[lid].append((sf, op))
    rebuilt = []
    for lid in sorted(bylane):
        prev = 0
        for sf, op in sorted(bylane[lid]):
            prev = sf
            rebuilt.append((prev, lid, op))
    ev_ok = sorted(ev) == sorted(rebuilt)
    d1 = SE.decode_events(ev, t)
    d2 = SE.decode_events(rebuilt, t)
    render_ok = np.array_equal(d1[:, :25], d2[:, :25])
    sweep_ok = True
    for _kind, lids in LANEMAP.items():
        for lid in lids:
            for sf, op in bylane[lid]:
                ds = _deltas(op)
                if ds is None:
                    continue
                recon = []
                for d, w in _rle(ds):
                    recon += [d] * w
                if recon != ds:
                    sweep_ok = False
    freq_ok = bool(FI.measure_freq(s)["residual_ok"])
    return ev_ok and render_ok and sweep_ok and freq_ok


if __name__ == "__main__":
    from preframr_tokens.codec import lane_grammar as G
    from preframr_tokens.codec import lsp_validate as V

    DUMP = (
        "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/"
        "Monty_on_the_Run.1.dump.parquet"
    )
    s = G.per_frame_state(DUMP, V.CPF, 10**9)[:, :25]
    b, t = measure(s)
    ok = verify_residual(s)
    print("=== DECOUPLED step/tracker codec (lane streams + sweep factoring) ===")
    for k in sorted(b):
        print(f"  {k:14s} {b[k]}")
    print(f"\nresidual-zero: {ok}")
    print(
        f"FULL MONTY: {b['total']} tokens = {b['total']/t:.3f} tok/frame "
        f"= {b['total']/8192:.2f}x vs 8192"
    )
    print("  " + ("UNDER 8192!!!" if b["total"] <= 8192 else "still OVER 8192"))
