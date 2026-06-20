"""Integrate the per-lane encoders into ONE time-ordered inline event stream.

Lanes: freq16 x3 (absolute-anchored NOTE/RAW/REST/MOD pitch ops, relative
semitone intervals) + pw12 x3 + ctrl/ad/sr x3 + fc-lo/fc-hi/res/vol (LOAD value /
RUN delta-run). Each lane's ops tile its frames; we flatten to events
(start_frame, lane, op), DROP holds (a lane implicitly holds its value between
events), and sort by start_frame. The stream is inline + continuable: any prefix
is the song up to that frame; new events can appear anytime; no preamble.

Freq lanes use the absolute-anchored pitch encoder (pitch_universal_anchor): every
NOTE onset re-anchors to an EXACT integer 12-TET grid index (relative semitone
interval, the small model-facing alphabet) and every MOD carries its own exact
integer base Fn. NOTE is INTENTIONALLY lossy on the base pitch (bounded grid-snap
<=50c, interior >=2000Fn); MOD/RAW/REST are RAW-Fn byte-exact via carried integers.
The per-voice tuning PHASE is a decode-side reconstruction field carried inline on
each NOTE token (quantized to PHASE_Q cents) -- not a frozen preamble table: any
prefix containing a NOTE carries its own phase, so the stream stays continuable.

Non-freq lanes stay byte-exact (LOAD value / RUN delta-run). Decode of freq lanes
is grid-index based (anchored); decode of non-freq lanes is the running-value
accumulator.
"""

import numpy as np

from preframr_tokens.codec import pitch_universal_anchor as A
from preframr_tokens.codec import program_encode as PE

SEMI = A.SEMI
CENTS_PER_SEMI = A.CENTS_PER_SEMI
PHASE_Q = 1.0  # cents quantum for the carried per-voice phase
_ARP_MAX_LEVELS = 4  # max distinct Fn levels for a genuine arp table

# lane id -> (registers it drives, kind). kind: "freq" / "pw" / "byte"
LANES = []
for v in range(3):
    LANES.append(("freq", (7 * v, 7 * v + 1)))
for v in range(3):
    LANES.append(("pw", (7 * v + 2, 7 * v + 3)))
for v in range(3):
    LANES.append(("ctrl", (7 * v + 4,)))
for v in range(3):
    LANES.append(("ad", (7 * v + 5,)))
for v in range(3):
    LANES.append(("sr", (7 * v + 6,)))
for reg in (21, 22, 23, 24):
    LANES.append(("byte", (reg,)))


def _lane_seq(s, kind, regs):
    if kind == "freq":
        return s[:, regs[0]] + 256 * s[:, regs[1]]
    if kind == "pw":
        return s[:, regs[0]] + 256 * (s[:, regs[1]] & 0xF)
    return s[:, regs[0]]


def _phase_q(phase_log2):
    """Quantize the per-voice phase to PHASE_Q cents (serializable signed int of
    cents). Round-trips to a phase_log2 close enough that the NOTE grid Fn keeps
    the by-design <=50c snap bound (the +-0.5c quantization is sub-snap)."""
    return int(np.round(phase_log2 / SEMI * CENTS_PER_SEMI / PHASE_Q))


def _phase_unq(pq):
    return float(pq) * PHASE_Q / CENTS_PER_SEMI * SEMI


def _freq_ops(seq):
    """Absolute-anchored ops for a freq voice, with the per-voice phase folded
    onto each NOTE as a decode-side field: ("NOTE", interval, micro_cents,
    phase_q). RAW/REST/MOD pass through (MOD already carries its base_fn)."""
    f = [int(x) for x in seq]
    # encode under the SAME quantized phase the decoder reconstructs from pq, so the
    # grid idx / glide endpoints never disagree across the quantization boundary.
    r = A.encode_voice(f, phase_quant=lambda p: _phase_unq(_phase_q(p)))
    if r is None:
        # voice never establishes a grid (e.g. < 2 held base pitches): every
        # value is an exact RAW escape -> byte-exact, no anchoring needed.
        return [("RAW", int(x)) for x in f]
    _, phase, toks = r
    pq = _phase_q(phase)
    out = []
    for t in toks:
        if t[0] == "NOTE":
            out.append(("NOTE", int(t[1]), int(round(t[2])), pq))
        else:
            out.append(t)
    return _recover_generators(out, phase)


def _as_triangle(e):
    """If period pattern e is a triangle LFO half-cycle in ANY phase rotation,
    return (s_up, u, s_dn, d, rot); else None. This is the SHARED-FAMILY triangle
    vibrato generator (Hubbard `vibrato`: counter and #7 / cmp #4 / eor #7 builds
    the symmetric oscilatval ramp 0,1,2,3,3,2,1,0, scaled by diff -- a constant
    +slope up-run of u frames then a constant -slope down-run of d frames on the
    freq accumulator).

    The CANONICAL half-cycle is (s_up,)*u + (s_dn,)*d (up-run then down-run). A real
    Monty run can start at ANY phase of that cycle (vibrato that begins mid-phase),
    so the period delta pattern is a CYCLIC ROTATION of the canonical: the up/down
    runs may be split across the period boundary. `rot` is the index into the
    canonical half-cycle at which e[0] sits, i.e. e[k] == canon[(rot+k) % p]. With
    rot carried, decode rebuilds canon and reads it starting at phase rot, so the
    tiling to n is byte-identical. Length is preserved (u+d == len(e))."""
    a = [int(x) for x in e]
    p = len(a)
    if p < 2 or any(x == 0 for x in a):
        return None  # pure triangle: every frame ramps
    signs = [1 if x > 0 else -1 for x in a]
    # find a cyclic sign-change boundary; all-same-sign is a slide, not a triangle.
    start = next((i for i in range(p) if signs[i] != signs[i - 1]), None)
    if start is None:
        return None
    rot_a = [a[(start + k) % p] for k in range(p)]
    rot_s = [signs[(start + k) % p] for k in range(p)]
    # collapse into sign-runs; a triangle has exactly two (one +run, one -run).
    runs = []
    for sg in rot_s:
        if runs and runs[-1][0] == sg:
            runs[-1][1] += 1
        else:
            runs.append([sg, 1])
    if len(runs) != 2:
        return None
    if runs[0][0] > 0:  # rotated form starts with the up-run
        u, d = runs[0][1], runs[1][1]
        s_up, s_dn = rot_a[0], rot_a[u]
    else:  # starts with down-run: shift to up
        d, u = runs[0][1], runs[1][1]
        s_dn, s_up = rot_a[0], rot_a[d]
        start = (start + d) % p
        rot_a = [a[(start + k) % p] for k in range(p)]
    canon = [s_up] * u + [s_dn] * d
    if rot_a != canon:
        return None  # slopes not constant within a run
    rot = (p - start) % p  # e[k] == canon[(rot + k) % p]
    return s_up, u, s_dn, d, rot


def _as_arp(e, base):
    """If period pattern e is a fixed-interval ARPEGGIO -- the freq accumulator
    DWELLS on each of a small set of discrete Fn levels for some frames, then jumps
    to the next (Hubbard `octarp`: "Simple octave arpeggio", counter and #1 walks
    note vs note+12; chord arps walk a few notefreqs table entries) -- return its
    step program [(level_delta, dwell), ...] else None.

    level_delta is the OUTPUT level (absolute Fn the lane sits at that frame) MINUS
    base: a TRANSPOSITION-RELATIVE offset (the arp interval), grounded in the
    octarp's note offset, and EXACT (the notefreqs table values are not a clean
    ratio of base, so we carry the exact Fn offset, not an approximate 2^(semi/12)).
    Levels are the POST-delta cumulative trajectory (out[k]=base+cumsum(e[:k+1])),
    so _arp_period_deltas inverts it byte-for-byte. The run must close (return to
    base) so tiling to n is exact, and the level set must be small (<=
    _ARP_MAX_LEVELS) -- a genuine arp table, not an opaque RLE of a complex run
    (those stay MOD, byte-exact)."""
    a = [int(x) for x in e]
    acc = base
    levels = []  # (level_delta, dwell) runs of OUTPUT lvl
    for d in a:
        acc += d  # output level for this frame (post-add)
        ld = acc - base
        if levels and levels[-1][0] == ld:
            levels[-1][1] += 1
        else:
            levels.append([ld, 1])
    if acc != base:
        return None  # does not close -> not a clean cycle
    if len(levels) < 2:
        return None  # single level is a hold, not an arp
    if len(set(ld for ld, _ in levels)) > _ARP_MAX_LEVELS:
        return None  # too many distinct levels -> not an arp
    return tuple((int(ld), int(w)) for ld, w in levels)


def _arp_period_deltas(steps):
    """Exact inverse of _as_arp: rebuild the period delta pattern e from the
    (level_delta, dwell) step table so out[k] = base + cumsum(e[:k+1]) walks the
    OUTPUT-level trajectory steps encode. e[k] = level[k] - level[k-1] cyclically
    (level[-1] := level[last] == 0 since the cycle closes to base)."""
    levels = []
    for ld, w in steps:
        levels.extend([ld] * w)
    e = []
    prev = 0  # cycle closes to base (level 0)
    for lv in levels:
        e.append(lv - prev)
        prev = lv
    return e


def _recover_generators(toks, phase):
    """Replace literal freq MOD delta-runs with recovered PARAMETRIC GENERATORS,
    grounded in the driver modulation algorithm (sid_opset_inventory.md s6h):

      HOLD  : an all-zero delta run is NOT modulation -- the lane simply holds.
              It is dropped to nothing (an implicit hold); decode holds the value.
      SLIDE : a period-1 constant-delta run is the linear pitch-slide / portamento
              ACCUM generator (P4): freq += rate each frame. Emitted as
              ("SLIDE", rate, n, base) -- the generator's rate parameter, not an
              RLE of arbitrary deltas. Decode regenerates the run bit-for-bit.

      VIB   : a period pattern that is a single up-ramp then down-ramp (constant
              slope each) is the SHARED-FAMILY triangle vibrato generator. Emitted
              as ("VIB", s_up, u, s_dn, d, n, base) -- the two ramp slopes + lengths,
              not the tiled deltas. Decode rebuilds the half-cycle and regenerates.

      Other periodic runs (arp offset tables / multi-slope PWM) are left as
      MOD(period_pattern, n, base): the recovered period pattern IS the generator's
      per-cycle program -- the per-tune data (the music), correctly NOT compressed.

    The transform is residual-zero by construction: every emitted op regenerates
    the identical Fn trajectory the original MOD produced (verified in roundtrip).
    """
    out = []
    cur_idx = None
    cur_fn = 0
    for t in toks:
        if t[0] == "NOTE":
            interval, micro = t[1], t[2]
            cur_idx = interval if cur_idx is None else cur_idx + interval
            cur_fn = min(
                65535, max(0, int(round(A._grid_fn(_phase_unq(t[3]), cur_idx, micro))))
            )
            out.append(t)
        elif t[0] in ("RAW", "REST"):
            if t[0] == "RAW":
                cur_idx = None
            cur_fn = int(t[1])
            out.append(t)
        else:  # MOD
            e, n, base = t[1], t[2], int(t[3])
            if all(d == 0 for d in e):
                # HOLD: not a generator -- the lane holds. Tag droppable iff the
                # running register Fn already equals base (the emission loop drops
                # it but still advances n frames, so decode holds it exactly). When
                # base != cur_fn the hold must re-latch -> keep it as MOD.
                if base == cur_fn:
                    out.append(("FHOLD", n, base))
                else:
                    out.append(t)
            elif len(e) == 1:
                out.append(("SLIDE", int(e[0]), n, base))
            else:
                vib = _as_triangle(e)
                if vib:
                    out.append(("VIB", *vib, n, base))  # (s_up,u,s_dn,d,rot,n,base)
                else:
                    arp = _as_arp(e, base)
                    if arp:
                        out.append(("ARP", arp, n, base))  # (steps, n, base)
                    else:
                        out.append(t)
            p = len(e)
            acc = float(base)
            for k in range(n):
                acc += e[k % p]
            cur_fn = int(round(acc))
            cur_idx = A._glide_end_idx(phase, e, n, base, cur_idx)
    return out


def _run_costs():
    """Lazy import of the token-cost primitives (serialize_tokens) -- imported here
    to keep the module import order (serialize_tokens imports serialize_events)."""
    from preframr_tokens.codec import serialize_tokens as ST

    return ST


def _sparse_run_cost(e, n):
    ST = _run_costs()
    nz = [(k, d) for k, d in enumerate(e) if d != 0]
    o = []
    ST._uint(o, len(e))
    ST._uint(o, n)
    ST._uint(o, len(nz))
    prev = -1
    for k, d in nz:
        ST._uint(o, k - prev - 1)
        ST._int(o, d)
        prev = k
    return len(o)


def _walk_steps(e):
    """RLE of the per-cycle CUMULATIVE OUTPUT offsets (relative to the lane's value
    entering the run). steps = [(offset, dwell), ...]; the period delta pattern is
    recovered exactly as e[k] = offset[k] - offset[k-1] (offset[-1] := 0). This is
    the non-freq sibling of the freq ARP level-walk, WITHOUT the close-to-base /
    level-count constraints: the walk is the run's exact per-cycle program (the
    table data), recovered, not an opaque RLE -- it is emitted only when its token
    body is strictly cheaper than the sparse-delta form."""
    steps = []
    acc = 0
    for d in e:
        acc += d
        if steps and steps[-1][0] == acc:
            steps[-1][1] += 1
        else:
            steps.append([acc, 1])
    return [(int(o), int(w)) for o, w in steps]


def _walk_period(steps):
    """Inverse of _walk_steps: rebuild the period delta pattern e from the cumulative
    (offset, dwell) steps. e[k] = level[k] - level[k-1]; the run starts from the
    lane's running value (level 0 reference)."""
    levels = []
    for o, w in steps:
        levels.extend([o] * w)
    e = []
    prev = 0
    for lv in levels:
        e.append(lv - prev)
        prev = lv
    return tuple(e)


def _walk_cost(steps, n):
    ST = _run_costs()
    o = []
    ST._uint(o, len(steps))
    ST._uint(o, n)
    for off, w in steps:
        ST._int(o, off)
        ST._uint(o, w)
    return len(o)


def _nslide_cost(d, n):
    ST = _run_costs()
    o = []
    ST._int(o, d)
    ST._uint(o, n)
    return len(o)


def _nvib_cost(tri, n):
    ST = _run_costs()
    s_up, u, s_dn, d, rot = tri
    o = []
    ST._int(o, s_up)
    ST._uint(o, u)
    ST._int(o, s_dn)
    ST._uint(o, d)
    ST._uint(o, rot)
    ST._uint(o, n)
    return len(o)


def _recover_run(op):
    """Replace a non-freq RUN(period, n) delta-run with the cheapest equivalent
    GENERATOR op, residual-zero (each regenerates the identical period):
      NSLIDE(rate, n)             period length 1: value += rate each frame
                                  (PW power-of-two sweeps, filter sweeps -- the SLIDE
                                  family on the RUN-byte lanes).
      NVIB(s_up,u,s_dn,d,rot, n)  triangle (up-ramp/down-ramp) on a RUN-byte lane.
      WALK(steps, n)              general cumulative level-walk (square waves, PWM
                                  duty toggles, multi-step hard-restart tables, the
                                  arp-shaped relatch-holds) -- the run's per-cycle
                                  output program RLE'd.
      RUN(period, n)              kept (sparse-delta) when no generator is cheaper.
    Chosen by strict token-body minimum so the transform never inflates the stream."""
    e, n = op[1], op[2]
    best_cost, best_op = _sparse_run_cost(e, n), op
    if len(e) == 1:
        c = _nslide_cost(int(e[0]), n)
        if c < best_cost:
            best_cost, best_op = c, ("NSLIDE", int(e[0]), n)
    tri = _as_triangle(e)
    if tri is not None:
        c = _nvib_cost(tri, n)
        if c < best_cost:
            best_cost, best_op = c, ("NVIB", *tri, n)
    steps = _walk_steps(e)
    c = _walk_cost(steps, n)
    if c < best_cost:
        best_op = ("WALK", tuple(steps), n)
    return best_op


def _run_period(op):
    """Lower a non-freq run op back to (period_deltas, n) for the replay decoder.
    RUN passes through; NSLIDE/NVIB/WALK regenerate the identical period."""
    if op[0] == "RUN":
        return op[1], op[2]
    if op[0] == "NSLIDE":
        return (int(op[1]),), op[2]
    if op[0] == "NVIB":
        s_up, u, s_dn, d, rot, n = op[1:7]
        canon = (s_up,) * u + (s_dn,) * d
        p = len(canon)
        e = tuple(canon[(rot + k) % p] for k in range(p))
        return e, n
    # WALK
    return _walk_period(op[1]), op[2]


def _freq_hold(op):
    # a same-grid re-onset with no micro change reproduces the held Fn -> drop.
    return op[0] == "NOTE" and op[1] == 0 and op[2] == 0


def _is_hold(op):
    if op[0] == "LOAD":
        return False  # caller checks value==cur
    if op[0] in ("NOTE", "RAW", "REST"):
        return False
    return all(d == 0 for d in op[1])  # MOD/RUN all-zero deltas


def _op_len(op):
    if op[0] in ("NOTE", "LOAD", "RAW", "REST"):
        return 1
    if op[0] == "FHOLD":  # ("FHOLD", n, base): n held frames, dropped
        return op[1]
    if op[0] == "VIB":  # ("VIB", s_up,u,s_dn,d,rot, n, base)
        return op[6]
    if op[0] == "ARP":  # ("ARP", steps, n, base)
        return op[2]
    if op[0] == "NVIB":  # ("NVIB", s_up,u,s_dn,d,rot, n)
        return op[6]
    if op[0] in ("NSLIDE", "WALK"):  # ("NSLIDE", rate, n) / ("WALK", steps, n)
        return op[2]
    return op[2]


def encode_tune_events(s):
    """Return time-sorted [(start_frame, lane_id, op)] with holds dropped."""
    events = []
    for lid, (kind, regs) in enumerate(LANES):
        seq = _lane_seq(s, kind, regs).astype(np.int64)
        if kind == "freq":
            toks = _freq_ops(seq)
            f = 0
            seeded = False  # the decode chain has a cur_idx (not None)
            for op in toks:
                # a NOTE that re-seeds the chain (after RAW/REST reset) carries the
                # absolute seed even if int==0,micro==0 -> never drop it; otherwise
                # decode keeps a stale cur_idx (the Swimming desync). FHOLD is a
                # recovered hold: dropped from the stream but its n frames still
                # advance f (the lane implicitly holds across the gap).
                drop = (_freq_hold(op) and seeded) or op[0] == "FHOLD"
                if not drop:
                    events.append((f, lid, op))
                if op[0] in ("RAW", "REST"):
                    seeded = False
                elif op[0] == "NOTE":
                    seeded = True
                f += _op_len(op)
        else:
            toks = PE.encode_lane([int(x) for x in seq])
            f = 0
            cur = 0
            for op in toks:
                length = 1 if op[0] == "LOAD" else op[2]
                hold = (op[0] == "LOAD" and op[1] == cur) or _is_hold(op)
                if not hold:
                    # RUN delta-runs are replaced by recovered generators
                    # (NSLIDE/NVIB/WALK) when strictly cheaper; LOAD passes through.
                    # Accumulation below stays on the ORIGINAL RUN period (the
                    # generator regenerates the identical period, residual=0).
                    emit = _recover_run(op) if op[0] == "RUN" else op
                    events.append((f, lid, emit))
                if op[0] == "LOAD":
                    cur = op[1]
                else:
                    p = len(op[1])
                    for k in range(length):
                        cur += op[1][k % p]
                f += length
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def _decode_freq_lane(ops, T):
    """Anchored freq decode -> per-frame integer Fn (16-bit). NOTE jumps to the
    snapped grid Fn (lossy <=50c); MOD/RAW/REST reproduce the exact integer Fn."""
    out = np.zeros(T, dtype=np.int64)
    cur = 0  # last emitted integer Fn (for holds)
    cur_idx = None
    phase = 0.0
    fr = 0
    for sf, op in ops:
        if sf > fr:
            out[fr:sf] = cur
        if op[0] == "NOTE":
            interval, micro, pq = op[1], op[2], op[3]
            phase = _phase_unq(pq)
            cur_idx = interval if cur_idx is None else cur_idx + interval
            val = A._grid_fn(phase, cur_idx, micro)
            cur = min(65535, max(0, int(round(val))))  # top-edge 16-bit clamp
            out[sf] = cur
            fr = sf + 1
        elif op[0] == "RAW":
            cur_idx = None
            cur = int(op[1])
            out[sf] = cur
            fr = sf + 1
        elif op[0] == "REST":
            cur = int(op[1])
            out[sf] = cur
            fr = sf + 1
        elif op[0] == "SLIDE":  # linear pitch-slide generator: +rate/frame
            rate, n, base_fn = op[1], op[2], op[3]
            acc = base_fn
            for k in range(n):
                acc += rate
                out[sf + k] = acc
            cur = acc
            cur_idx = A._glide_end_idx(phase, (rate,), n, base_fn, cur_idx)
            fr = sf + n
        elif op[0] == "VIB":  # triangle vibrato generator (any phase rot)
            s_up, u, s_dn, d, rot, n, base_fn = op[1:8]
            canon = (s_up,) * u + (s_dn,) * d
            p = len(canon)
            e = tuple(canon[(rot + k) % p] for k in range(p))  # rotate to run phase
            acc = base_fn
            for k in range(n):
                acc += e[k % p]
                out[sf + k] = acc
            cur = acc
            cur_idx = A._glide_end_idx(phase, e, n, base_fn, cur_idx)
            fr = sf + n
        elif op[0] == "ARP":  # fixed-interval arpeggio (level-table walk)
            steps, n, base_fn = op[1], op[2], op[3]
            e = _arp_period_deltas(steps)
            p = len(e)
            acc = base_fn
            for k in range(n):
                acc += e[k % p]
                out[sf + k] = acc
            cur = acc
            cur_idx = A._glide_end_idx(phase, tuple(e), n, base_fn, cur_idx)
            fr = sf + n
        else:  # MOD: ride raw deltas from carried base
            e, n, base_fn = op[1], op[2], op[3]
            p = len(e)
            acc = float(base_fn)
            for k in range(n):
                acc += e[k % p]
                v = int(round(acc))
                out[sf + k] = v
            cur = v
            cur_idx = A._glide_end_idx(phase, e, n, base_fn, cur_idx)
            fr = sf + n
    if fr < T:
        out[fr:T] = cur
    return out


def decode_events(events, T):
    """Replay events -> 25-register per-frame state."""
    s = np.zeros((T, 25), dtype=np.int64)
    lane_vals = [[] for _ in LANES]
    for sf, lid, op in events:
        lane_vals[lid].append((sf, op))
    for lid, (kind, regs) in enumerate(LANES):
        if kind == "freq":
            out = _decode_freq_lane(lane_vals[lid], T)
            s[:, regs[0]] = out & 0xFF
            s[:, regs[1]] = out >> 8
            continue
        out = np.zeros(T, dtype=np.int64)
        cur = 0
        fr = 0
        for sf, op in lane_vals[lid]:
            if sf > fr:
                out[fr:sf] = cur  # implicit hold
            if op[0] == "LOAD":
                cur = op[1]
                out[sf] = cur
                fr = sf + 1
            else:
                deltas, n = _run_period(op)
                p = len(deltas)
                for k in range(n):
                    cur += deltas[k % p]
                    out[sf + k] = cur
                fr = sf + n
        if fr < T:
            out[fr:T] = cur
        if kind == "pw":
            s[:, regs[0]] = out & 0xFF
            s[:, regs[1]] = out >> 8
        else:
            s[:, regs[0]] = out
    return s


def roundtrip(s):
    """Round-trip. Returns (rec, n_events): rec is the decoded 25-reg state.
    NOTE: freq lanes are INTENTIONALLY lossy on the base pitch (grid-snap <=50c);
    use cents_check for the freq gate and array_equal on the non-freq registers
    for the byte-exact gate. See serialize_corpus for the split gate."""
    ev = encode_tune_events(s)
    rec = decode_events(ev, len(s))
    return rec, len(ev)
