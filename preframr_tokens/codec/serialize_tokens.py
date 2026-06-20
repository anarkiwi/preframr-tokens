"""Flat token serialization of the inline event stream. Small fixed vocabulary
(atoms-only); BPE builds the dictionary on top later. Each event ->
  [DT uint] [LANE atom] [OP atom] [op params]
Numbers are self-delimiting base-16 LEB (digit 0-15 = continue, 16-31 = terminal);
signed values use zigzag. Round-trips: ids -> events -> ids exact (the token tuples
are reproduced exactly; the freq-lane DECODE to registers is intentionally lossy on
the NOTE base pitch, bounded grid-snap <=50c -- see serialize_events).

Freq lanes (0..2) use the absolute-anchored ops NOTE/RAW/REST/MOD:
  NOTE -> (interval_semitones, micro_cents, phase_q)   model-facing interval +
          decode-side micro-detune + per-voice phase (carried inline, not frozen)
  RAW  -> (fn)                exact out-of-grid / extreme-transient Fn
  REST -> (fn)                exact note-off / power-zero Fn
  MOD  -> (deltas, n, base_fn) raw deltas ridden from the carried exact base Fn
A freq MOD's base_fn is serialized as a ZIGZAG DELTA off the per-lane running
register Fn (the 16-bit value the lane holds entering the MOD, reconstructed
identically on both sides by replaying the freq decode). The base equals that
running value ~51% of the time and is within one LEB digit ~80% (median 0) -- so
the base collapses from a ~3-4 digit absolute Fn to ~1 digit. The EVENT tuple
keeps base_fn ABSOLUTE (ids->events->ids round-trips exactly); only the token
encoding is delta'd.
Freq MOD runs are further factored into RECOVERED GENERATORS (serialize_events):
  SLIDE -> (rate, n, base_fn)             linear pitch-slide / portamento accum
  VIB   -> (s_up, u, s_dn, d, n, base_fn) triangle-vibrato half-cycle (two ramps)
an all-zero HOLD run is dropped (implicit hold). These regenerate the run
bit-for-bit (residual = 0); they are NOT an RLE of arbitrary deltas.

Non-freq lanes use LOAD value / RUN delta-run, byte-exact. RUN delta-runs are now
factored into RECOVERED GENERATORS too (serialize_events._recover_run), residual=0:
  NSLIDE -> (rate, n)                  period-1 linear sweep (PW power-of-two rates,
                                       filter sweeps) -- the SLIDE family on RUN-byte
                                       lanes, no base (run starts from the running val)
  NVIB   -> (s_up,u,s_dn,d,rot, n)     triangle on a RUN-byte lane
  WALK   -> (steps, n)                 general cumulative level-walk: steps is a tuple
                                       of (offset, dwell) -- the run's per-cycle OUTPUT
                                       program (square waves, PWM duty toggles, multi-
                                       step hard-restart tables, arp-shaped relatches),
                                       RLE'd. Emitted only when strictly cheaper than
                                       the sparse RUN; hard-restart blips stay RUN.
OP atoms NOTE/MOD are shared between freq and non-freq lanes; the serializer
branches on the lane id (freq = id < 3) to read the extra freq fields.

Vocab layout (62 atoms):
  0..16   LANE (17)
  17..28  OP: NOTE, LOAD, MOD, RUN, RAW, REST, SLIDE, VIB, ARP, NSLIDE, NVIB, WALK
  29      REPEAT_MARKER
  30..61  DIGIT (32: low16 continue, high16 terminal)

Freq MOD runs are factored into RECOVERED GENERATORS (serialize_events):
  SLIDE -> (rate, n, base_fn)                  linear pitch-slide / portamento accum
  VIB   -> (s_up, u, s_dn, d, rot, n, base_fn) triangle vibrato, ANY phase rotation
           (rot = phase into the canonical up-then-down half-cycle the run starts at)
  ARP   -> (steps, n, base_fn)                 fixed-interval arpeggio: steps is a
           tuple of (level_delta, dwell) -- the arp's transposition-relative Fn level
           table walked at a dwell rate (Hubbard octarp). Exact, not an opaque RLE.
"""

from preframr_tokens.codec import pitch_universal_anchor as A
from preframr_tokens.codec import serialize_events as SE

LANE_BASE = 0
OP_BASE = 17
DIGIT_BASE = 26
# REPEAT op atom: an inline backward-LZ copy of N already-emitted EVENTS. It is the
# only op that does NOT carry a (dt, lane) header -- it is read directly after the
# previous event's tokens as REPEAT_MARKER + LEB(offset_events) + LEB(n_events).
# Backward-only, within a sliding event window: at decode it copies the n events
# that occurred `offset` events ago (their dt-relative timing reproduces exactly,
# so absolute frames shift to the current position). No hoisted dictionary, no
# preamble; any prefix stays a valid continuable song. Residual = 0 because the
# copied literal events are byte-identical to the originals.
OPS = {
    "NOTE": 0,
    "LOAD": 1,
    "MOD": 2,
    "RUN": 3,
    "RAW": 4,
    "REST": 5,
    "SLIDE": 6,
    "VIB": 7,
    "ARP": 8,
    "NSLIDE": 9,
    "NVIB": 10,
    "WALK": 11,
}
OPS_INV = {v: k for k, v in OPS.items()}
# REPEAT_MARKER + LREPLAY_MARKER + digit block sit ABOVE the op block.
# LREPLAY: a per-lane backward "replay" op. Where merged REPEAT copies a contiguous
# globally-frame-sorted span (so it cannot capture a single voice's pattern that is
# INTERLEAVED with other voices), LREPLAY copies the next `n_events` events of ONE
# lane from THAT lane's own already-emitted history, `events_back` lane-events ago.
# Tokens: LREPLAY_MARKER + LANE atom + LEB(events_back) + LEB(n_events). Emitted in
# TIME ORDER at the frame where the lane starts repeating; the copied events take
# their absolute frames from the lane's own running history (last lane frame + the
# replayed intra-lane dt deltas), so they do NOT touch the global dt base `prev` --
# splicing a lane's interleaved pattern back in never makes another lane's dt go
# negative. Backward-only, established-on-first-play (the first play of a pattern is
# literal events; later plays are LREPLAY references = the orderlist, inline). No
# frozen table, no preamble; any prefix stays a valid continuable song. Residual=0:
# the copied lane events are byte-identical and fc/micro-table state is advanced over
# them in lane order exactly as the literal stream would.
REPEAT_MARKER = OP_BASE + len(OPS)  # sits just below the digit block
LREPLAY_MARKER = REPEAT_MARKER + 1
DIGIT_BASE = LREPLAY_MARKER + 1  # digits start after the markers
VOCAB = DIGIT_BASE + 32
N_FREQ_LANES = 3

# LZ-over-events parameters (the REPEAT lever).
_LZ_MIN_LEN = 2  # min events to bother emitting a REPEAT
_LZ_WINDOW = 1_000_000  # sliding look-back in events (effectively whole tune)
_LZ_MAX_LEN = 4096


class _FreqCur:
    """Per-freq-lane running register Fn (the 16-bit value the lane holds entering
    the next op), reconstructed identically on encode and decode by replaying the
    anchored freq decode. Used as the delta reference for a freq MOD's base_fn."""

    def __init__(self):
        self.cur = [0, 0, 0]
        self.idx = [None, None, None]
        self.phase_q = [0, 0, 0]  # per-voice phase quantum (one constant/voice)
        self.phase_seeded = [False, False, False]  # phase sent once, on first NOTE
        # Backward-looking per-(lane, absolute-index) micro TABLE. The tune's own
        # recovered note table: the FIRST onset at an absolute index establishes that
        # index's micro residual (its fixed +-4c ET-table deviation under the voice
        # phase); every later onset at that index references the table (delta off it,
        # ==0 for a clean repeat -> byte-identical NOTE tokens). No frozen preamble:
        # the entry is established inline on first occurrence, referenced after, so any
        # prefix stays a valid continuable song. A modulation-at-onset offset that the
        # table cannot absorb rides as a SMALL signed delta off the table residual
        # (the shared, backward-referenced micro alphabet), never a raw cents value.
        self.micro_tbl = {}  # (lane, abs_idx) -> established micro

    def ref(self, lane):
        return self.cur[lane] & 0xFFFF

    def next_idx(self, lane, interval):
        """The absolute grid index this NOTE will sit at (pre-step), matching the
        accumulation in step()/the freq decoder."""
        return interval if self.idx[lane] is None else self.idx[lane] + interval

    def micro_ref(self, lane, idx):
        """Established table micro for (lane, idx), or None if this is first sight."""
        return self.micro_tbl.get((lane, idx))

    def set_micro(self, lane, idx, micro):
        if (lane, idx) not in self.micro_tbl:
            self.micro_tbl[(lane, idx)] = micro

    def phase_ref(self, lane):
        return self.phase_q[lane]

    def step(self, lane, op):
        if op[0] == "NOTE":
            interval, micro, pq = op[1], op[2], op[3]
            self.phase_q[lane] = pq
            self.phase_seeded[lane] = True
            phase = SE._phase_unq(pq)
            ci = interval if self.idx[lane] is None else self.idx[lane] + interval
            self.idx[lane] = ci
            self.set_micro(lane, ci, micro)
            self.cur[lane] = min(
                65535, max(0, int(round(A._grid_fn(phase, ci, micro))))
            )
        elif op[0] == "RAW":
            self.idx[lane] = None
            self.cur[lane] = int(op[1])
        elif op[0] == "REST":
            self.cur[lane] = int(op[1])
        elif op[0] == "SLIDE":
            rate, n, base = op[1], op[2], op[3]
            self.cur[lane] = int(base) + int(rate) * int(n)
        elif op[0] == "VIB":
            s_up, u, s_dn, d, rot, n, base = op[1:8]
            canon = (s_up,) * u + (s_dn,) * d
            p = len(canon)
            e = tuple(canon[(rot + k) % p] for k in range(p))
            acc = int(base)
            for k in range(int(n)):
                acc += e[k % p]
            self.cur[lane] = acc
        elif op[0] == "ARP":
            steps, n, base = op[1], op[2], op[3]
            e = SE._arp_period_deltas(steps)
            p = len(e)
            acc = int(base)
            for k in range(int(n)):
                acc += e[k % p]
            self.cur[lane] = acc
        else:  # MOD
            e, n, base = op[1], op[2], op[3]
            p = len(e)
            acc = float(base)
            for k in range(n):
                acc += e[k % p]
            self.cur[lane] = int(round(acc))


def _uint(out, n):
    n = int(n)
    assert n >= 0, f"_uint requires a non-negative value, got {n}"
    while True:
        d = n & 0xF
        n >>= 4
        out.append(DIGIT_BASE + (d if n else 16 + d))
        if not n:
            return


def _int(out, n):
    n = int(n)
    _uint(out, (n << 1) ^ (n >> 63))  # zigzag


def _ruint(ids, i):
    n = shift = 0
    while True:
        a = ids[i] - DIGIT_BASE
        i += 1
        if a < 16:
            n |= a << shift
            shift += 4
        else:
            n |= (a - 16) << shift
            return n, i


def _rint(ids, i):
    z, i = _ruint(ids, i)
    return (z >> 1) ^ -(z & 1), i


def _emit_event(out, fc, prev, sf, lane, op):
    """Append one literal event's tokens; advance fc; return the new prev frame. The dt
    header is SIGNED (zigzag): the stream is emitted in approximate time order, but an
    LREPLAY may splice one lane's repeating pattern in at the frame it starts repeating,
    leaving a following other-lane literal at a slightly earlier frame -> a small
    negative dt. decode_events groups by lane, so global order is not load-bearing; the
    signed dt only has to reproduce each event's absolute frame (prev + dt)."""
    _int(out, sf - prev)
    prev = sf
    out.append(LANE_BASE + lane)
    out.append(OP_BASE + OPS[op[0]])
    _emit_event_body(out, fc, lane, op)
    return prev


def _emit_event_body(out, fc, lane, op):
    is_freq = lane < N_FREQ_LANES
    if op[0] == "NOTE":
        _int(out, op[1])
        if is_freq:
            # phase: a single per-voice tuning constant, emitted ONCE on the
            # first NOTE of the lane (inline seed), then never again -> repeated
            # notes carry no phase field.
            if not fc.phase_seeded[lane]:
                _uint(out, op[3])  # absolute phase quantum (one/voice)
            # micro: factor against the tune's own recovered note table. The
            # FIRST onset at an absolute grid index establishes that index's
            # micro residual (its fixed ET-table deviation under the voice phase)
            # and stores it ABSOLUTELY; every later onset at the same index stores
            # only the SIGNED DELTA off that established residual (==0 for a clean
            # repeat -> byte-identical NOTE token, which unblocks the LZ-copy op).
            # A modulation-at-onset offset the table cannot absorb rides as that
            # small signed delta (the shared, backward-referenced micro alphabet),
            # never a raw per-note cents value. The table is established inline on
            # first occurrence and only referenced after -> backward-only, no
            # frozen preamble; any prefix stays a valid continuable song. Lossless:
            # decode reconstructs micro = established_residual + stored_delta.
            idx = fc.next_idx(lane, op[1])
            ref = fc.micro_ref(lane, idx)
            if ref is None:
                _int(out, op[2])  # first sight: absolute residual (seeds table)
            else:
                _int(out, op[2] - ref)  # later: signed delta off the table residual
    elif op[0] == "LOAD":
        _uint(out, op[1])
    elif op[0] in ("RAW", "REST"):
        _uint(out, op[1])
    elif op[0] == "SLIDE":  # freq linear-slide generator
        _int(out, op[1])  # rate (signed per-frame delta)
        _uint(out, op[2])  # n frames
        _int(out, op[3] - fc.ref(lane))  # base as zigzag delta off running Fn
    elif op[0] == "VIB":  # freq triangle-vibrato generator
        _int(out, op[1])  # s_up  (positive ramp slope)
        _uint(out, op[2])  # u     (up-run length)
        _int(out, op[3])  # s_dn  (negative ramp slope)
        _uint(out, op[4])  # d     (down-run length)
        _uint(out, op[5])  # rot   (start phase into half-cycle)
        _uint(out, op[6])  # n     (total frames)
        _int(out, op[7] - fc.ref(lane))  # base as zigzag delta off running Fn
    elif op[0] == "ARP":  # freq fixed-interval arpeggio
        steps = op[1]
        _uint(out, len(steps))  # number of (level_delta, dwell) steps
        for ld, w in steps:
            _int(out, ld)  # level delta off base (arp interval Fn)
            _uint(out, w)  # dwell frames at this level
        _uint(out, op[2])  # n     (total frames)
        _int(out, op[3] - fc.ref(lane))  # base as zigzag delta off running Fn
    elif op[0] == "NSLIDE":  # non-freq linear sweep generator
        _int(out, op[1])  # rate (signed per-frame delta)
        _uint(out, op[2])  # n frames
    elif op[0] == "NVIB":  # non-freq triangle generator
        _int(out, op[1])  # s_up
        _uint(out, op[2])  # u
        _int(out, op[3])  # s_dn
        _uint(out, op[4])  # d
        _uint(out, op[5])  # rot
        _uint(out, op[6])  # n
    elif op[0] == "WALK":  # non-freq cumulative level-walk
        steps = op[1]
        _uint(out, len(steps))  # number of (offset, dwell) steps
        for off, w in steps:
            _int(out, off)  # cumulative output offset off entry val
            _uint(out, w)  # dwell frames at this level
        _uint(out, op[2])  # n (total frames)
    elif op[0] == "RUN":  # non-freq RUN: SPARSE period encoding
        # The period delta pattern of a held register is dominated by zeros
        # (the playroutine holds and changes a register only a few frames per
        # cycle: hard-restart -X,0,+X toggles, a PWM step, etc.). Encoding every
        # zero costs ~p tokens for a p-frame period with 2-4 changes. Instead
        # emit only the NONZERO frames: their count, then for each a (gap, delta)
        # where gap = frames since the previous nonzero (so contiguous changes
        # cost 1 gap token each). Pure-zero periods cannot occur here (holds are
        # dropped upstream by _is_hold), so nnz >= 1. Exact: decode rebuilds the
        # length-p zero pattern and writes each delta at its frame, then tiles n.
        e = op[1]
        _uint(out, len(e))  # p (period length)
        _uint(out, op[2])  # n (total frames, n = k*p)
        nz = [(k, d) for k, d in enumerate(e) if d != 0]
        _uint(out, len(nz))  # nnz (nonzero frames in the period)
        prev_k = -1  # NOT `prev`: that is the DT tracker
        for k, d in nz:
            _uint(out, k - prev_k - 1)  # gap to previous nonzero frame
            _int(out, d)  # signed delta at this frame
            prev_k = k
    else:  # MOD (freq lane)
        _uint(out, len(op[1]))
        _uint(out, op[2])
        for d in op[1]:
            _int(out, d)
        if is_freq:  # freq MOD: base as zigzag delta off
            _int(out, op[3] - fc.ref(lane))  # the running register Fn
    if is_freq:
        fc.step(lane, op)


def _opkey(op):
    """Hashable identity of an event op (already hashable tuples, but RUN/MOD carry
    a tuple already). Used for the LZ match key together with dt+lane."""
    return op


def _leb16(x):
    x = int(x)
    n = 1
    while x >= 16:
        x >>= 4
        n += 1
    return n


def _event_spans(events):
    """Exact literal token span of each event under _emit_event (state-faithful)."""
    fc = _FreqCur()
    prev = 0
    spans = []
    for sf, lane, op in events:
        o = []
        prev = _emit_event(o, fc, prev, sf, lane, op)
        spans.append(len(o))
    return spans


def _merged_repeat_parse(events):
    """PASS 1 -- the existing merged REPEAT: greedy longest backward-match LZ over
    EVENTS with key (dt, lane, op), over globally-contiguous spans only. A REPEAT
    consumes [i, i+n) and i jumps past it, so copied frames never overshoot the next
    literal (frame-monotonic by construction). Returns LIT / REPEAT items.
    Identical in behaviour to the pre-LREPLAY baseline."""
    from collections import defaultdict
    import bisect

    n = len(events)
    prev = 0
    keys = []
    for sf, lane, op in events:
        keys.append((sf - prev, lane, _opkey(op)))
        prev = sf
    spans = _event_spans(events)
    pref = [0]
    for sp in spans:
        pref.append(pref[-1] + sp)
    out = []
    table = defaultdict(list)
    K = _LZ_MIN_LEN
    i = 0
    while i < n:
        best_len = 0
        best_pos = -1
        if i + K <= n:
            cands = table.get(tuple(keys[i : i + K]))
            if cands:
                lo = bisect.bisect_left(cands, i - _LZ_WINDOW) if _LZ_WINDOW < i else 0
                for pos in reversed(cands[lo:]):
                    cap = min(_LZ_MAX_LEN, n - i)
                    L = K
                    while L < cap and keys[pos + L] == keys[i + L]:
                        L += 1
                    if L > best_len:
                        best_len, best_pos = L, pos
                        if L >= cap:
                            break
        emit_repeat = False
        if best_len >= K:
            lit_span = pref[i + best_len] - pref[i]
            rep_cost = 1 + _leb16(i - best_pos) + _leb16(best_len)
            emit_repeat = rep_cost < lit_span
        if emit_repeat:
            out.append(("REPEAT", i - best_pos, best_len))
            j = i
            while j + K <= i + best_len:
                table[tuple(keys[j : j + K])].append(j)
                j += 1
            i += best_len
        else:
            out.append(("LIT", events[i]))
            if i + K <= n:
                table[tuple(keys[i : i + K])].append(i)
            i += 1
    return out


def _lreplay_pass(parsed):
    """PASS 2 -- per-lane backward REPLAY. Walks the pass-1 stream in order, maintaining
    each lane's FULL emitted-event history exactly as the decoder reconstructs it
    (literal lane events + REPEAT-copied lane events + earlier LREPLAY-copied lane
    events). For each LITERAL event it tries to extend a backward per-lane match against
    that lane's history (key = (lane_dt, op)); a matched run of the lane's own next
    literal events collapses to one LREPLAY(lane, events_back, n). events_back / n are
    counted over the full lane history, matching decode exactly.

    A match runs only over CONSECUTIVE literal events of the lane that are also
    consecutive in the GLOBAL stream order (no REPEAT span and no other lane event in
    between) -- so an LREPLAY replaces a clean contiguous chunk of the global stream with
    one op, never reorders across a REPEAT, and (because it does not advance the global
    dt base) leaves the surviving literals globally sorted -> dt headers stay
    non-negative. Backward-only, established on first play; cost-aware."""
    from collections import defaultdict

    lane_hist = defaultdict(list)  # lane -> [(lane_dt, op)] full history
    lane_prevf = defaultdict(int)
    lane_tab = defaultdict(lambda: defaultdict(list))  # lane -> Kgram -> hist positions
    K = _LZ_MIN_LEN
    out = []

    def _push(lane, sf, op):
        h = lane_hist[lane]
        pos = len(h)
        h.append((sf - lane_prevf[lane], op))
        lane_prevf[lane] = sf
        if pos - K + 1 >= 0:
            kk = tuple(h[pos - K + 1 : pos + 1])
            lane_tab[lane][kk].append(pos - K + 1)

    # First, expand pass-1 into a flat tagged list so we can look ahead within a
    # same-lane global run for the match length.
    flat = []  # (tag, sf, lane, op) ; tag 'L' lit / 'R' repeat-copied
    rep_events = []  # rebuilt frames for REPEAT expansion
    prevf = 0
    for item in parsed:
        if item[0] == "LIT":
            sf, lane, op = item[1]
            flat.append(("L", sf, lane, op))
            rep_events.append((sf, lane, op))
            prevf = sf
        else:
            _, off, ln = item
            start = len(rep_events) - off
            src_prev = rep_events[start - 1][0] if start > 0 else 0
            for k in range(ln):
                ssf, lane, op = rep_events[start + k]
                prevf = prevf + (ssf - src_prev)
                src_prev = ssf
                flat.append(
                    (
                        "R",
                        prevf,
                        lane,
                        op,
                        item if k == 0 else None,
                        ln if k == 0 else None,
                    )
                )
                rep_events.append((prevf, lane, op))

    n = len(flat)
    i = 0
    while i < n:
        rec = flat[i]
        if rec[0] == "R":
            # emit the REPEAT op once (at its first copied event) and push all its
            # copied events into lane history; they are NOT LREPLAY-foldable.
            if rec[4] is not None:
                out.append(rec[4])
            _push(rec[2], rec[1], rec[3])
            i += 1
            continue
        # rec is a literal. Determine the maximal GLOBAL run of consecutive literals of
        # the same lane starting here (broken by any 'R' or a different lane).
        lane = rec[2]
        j = i
        run = []
        while j < n and flat[j][0] == "L" and flat[j][2] == lane:
            run.append((flat[j][1], flat[j][3]))  # (sf, op)
            j += 1
        m = len(run)
        # local keys (lane_dt, op) for the run, continuing the lane frame timeline.
        prevf2 = lane_prevf[lane]
        rkeys = []
        for sf, op in run:
            rkeys.append((sf - prevf2, op))
            prevf2 = sf
        p = 0
        while p < m:
            best_len = 0
            best_pos = -1
            hist = lane_hist[lane]
            hpos = len(hist)  # backward boundary (source must end here)
            if p + K <= m:
                cands = lane_tab[lane].get(tuple(rkeys[p : p + K]))
                if cands:
                    for pos in reversed(cands[-64:]):
                        # NO self-overlap: the source span [pos, pos+L) stays strictly
                        # within already-emitted history (pos + L <= hpos). This makes
                        # the copied lane_dt sequence a verbatim slice of history, so
                        # reconstructed frames equal the originals exactly.
                        cap = min(m - p, hpos - pos)
                        L = 0
                        while (
                            L < cap
                            and hist[pos + L][0] == rkeys[p + L][0]
                            and hist[pos + L][1] == rkeys[p + L][1]
                        ):
                            L += 1
                        if L > best_len:
                            best_len, best_pos = L, pos
            emit = False
            if best_len >= K:
                span = _lane_run_span(lane, lane_hist, run[p : p + best_len])
                cost = 1 + 1 + _leb16(hpos - best_pos) + _leb16(best_len)
                emit = cost < span
            if emit:
                eback = hpos - best_pos
                # verify reconstructed frames equal the originals (residual guard): the
                # decoder lands the run at lane_prevf + cumulative source lane_dt.
                lf = lane_prevf[lane]
                for k in range(best_len):
                    lf += hist[best_pos + k][0]
                    assert (
                        lf == run[p + k][0]
                    ), f"LREPLAY frame mismatch lane={lane} got {lf} want {run[p + k][0]}"
                out.append(("LREPLAY", lane, eback, best_len))
                for k in range(best_len):
                    sf, op = run[p + k]
                    _push(lane, sf, op)
                p += best_len
            else:
                sf, op = run[p]
                out.append(("LIT", (sf, lane, op)))
                _push(lane, sf, op)
                p += 1
        i = j
    return out


def _lane_run_span(lane, lane_hist, run):
    """Exact token span of `run` events ((sf,op) list) of `lane` under _emit_event,
    with the per-lane fc state replayed over the lane's prior history so freq stateful
    fields (phase seed, micro table) cost exactly as in the real stream."""
    fc = _FreqCur()
    # replay prior history ops into fc (frames irrelevant to fc except via op fields,
    # which the ops already carry); lane_hist stores (lane_dt, op).
    prevf = 0
    for dt, op in lane_hist[lane]:
        prevf += dt
        if lane < N_FREQ_LANES:
            fc.step(lane, op)
    tot = 0
    prev = prevf
    for sf, op in run:
        o = []
        prev = _emit_event(o, fc, prev, sf, lane, op)
        tot += len(o)
    return tot


def _lz_parse(events):
    """Two-pass time-ordered backward-reference parse. Pass 1 = merged REPEAT
    (contiguous multi-lane blocks). Pass 2 = per-lane LREPLAY over the leftover
    literals (one voice's interleaved repeating pattern). Complementary by
    construction: pass 2 only touches LIT items, never events inside a REPEAT span,
    so the two ops never double-count and the stream stays frame-monotonic.
    Returns items: ('LIT', (sf,lane,op)) | ('REPEAT', off_events, n_events)
                 | ('LREPLAY', lane, lane_events_back, n_lane_events)."""
    return _lreplay_pass(_merged_repeat_parse(events))


def _lz_expand(parsed):
    """Inverse of _lz_parse: rebuild the exact literal event list (residual=0). A
    merged REPEAT shifts the copied span to the current global frame by replaying the
    source span's dt deltas. An LREPLAY copies the next n events of one lane from that
    lane's own history, shifting each to the lane's running frame (last lane frame +
    replayed intra-lane dt). Because pass 2 only collapses CONSECUTIVE global same-lane
    literals, an LREPLAY's last copied frame is the run's last global frame, so it
    advances the global cursor too -> the stream stays frame-monotonic for a following
    REPEAT (whose dt replay is anchored at the current global cursor)."""
    events = []
    prevf = 0
    lane_hist = {}  # lane -> list of (sf, op) in lane order
    lane_last = {}  # lane -> last frame emitted for that lane
    for item in parsed:
        if item[0] == "LIT":
            sf, lane, op = item[1]
            events.append((sf, lane, op))
            prevf = sf
            lane_hist.setdefault(lane, []).append((sf, op))
            lane_last[lane] = sf
        elif item[0] == "REPEAT":
            _, off, ln = item
            start = len(events) - off
            src_prev = events[start - 1][0] if start > 0 else 0
            for k in range(ln):
                src_sf, lane, op = events[start + k]
                prevf = prevf + (src_sf - src_prev)
                src_prev = src_sf
                events.append((prevf, lane, op))
                lane_hist.setdefault(lane, []).append((prevf, op))
                lane_last[lane] = prevf
        else:  # LREPLAY
            _, lane, eback, ln = item
            hist = lane_hist[lane]
            start = len(hist) - eback
            src_prev = hist[start - 1][0] if start > 0 else 0
            lf = lane_last[lane]
            for k in range(ln):
                src_sf, op = hist[start + k]
                lf = lf + (src_sf - src_prev)
                src_prev = src_sf
                events.append((lf, lane, op))
                hist.append((lf, op))
            lane_last[lane] = lf
            prevf = lf  # advance the global cursor to the run end
    return events


def events_to_ids(events, lz=True):
    """Serialize events to the flat token-id stream. With lz=True (default) an inline
    backward-LZ pass collapses repeated event spans into REPEAT(offset, n) ops -- the
    repetition lever -- emitted ONLY when cheaper than the literal span. The fc state
    is advanced over the COPIED events too, so stateful fields (phase seed, micro
    table) stay identical to the literal stream: decode of a REPEAT reproduces the
    copied events byte-for-byte. Set lz=False for the pre-REPEAT literal baseline."""
    parsed = _lz_parse(events) if lz else [("LIT", e) for e in events]
    out = []
    prev = 0
    fc = _FreqCur()
    emitted = []  # the running expanded literal-event list
    lane_hist = {}  # lane -> list of (sf, op); LREPLAY source
    lane_last = {}  # lane -> last frame emitted for that lane
    for item in parsed:
        if item[0] == "REPEAT":
            _, off, ln = item
            out.append(REPEAT_MARKER)
            _uint(out, off)
            _uint(out, ln)
            # Advance prev + fc over the COPIED events by replaying the source span's
            # dt deltas (its first dt is part of the matched key, so prev+dt lands on
            # the exact target frame). Subsequent literals then see identical state
            # (dt base, phase seed, micro table). The copied events ARE the song here.
            start = len(emitted) - off
            src_prev = emitted[start - 1][0] if start > 0 else 0
            for k in range(ln):
                src_sf, lane, op = emitted[start + k]
                prev = prev + (src_sf - src_prev)
                src_prev = src_sf
                if lane < N_FREQ_LANES:
                    fc.step(lane, op)
                emitted.append((prev, lane, op))
                lane_hist.setdefault(lane, []).append((prev, op))
                lane_last[lane] = prev
        elif item[0] == "LREPLAY":
            _, lane, eback, ln = item
            out.append(LREPLAY_MARKER)
            out.append(LANE_BASE + lane)
            _uint(out, eback)
            _uint(out, ln)
            # Replay this lane's history; fc is stepped in lane order so the micro table
            # / phase seed stay identical to the literal stream. The run is a block of
            # consecutive global same-lane literals, so its last frame is the current
            # global frame -> advance prev to it for the following REPEAT/literal dt.
            hist = lane_hist[lane]
            start = len(hist) - eback
            src_prev = hist[start - 1][0] if start > 0 else 0
            lf = lane_last[lane]
            for k in range(ln):
                src_sf, op = hist[start + k]
                lf = lf + (src_sf - src_prev)
                src_prev = src_sf
                if lane < N_FREQ_LANES:
                    fc.step(lane, op)
                emitted.append((lf, lane, op))
                hist.append((lf, op))
            lane_last[lane] = lf
            prev = lf
        else:
            sf, lane, op = item[1]
            prev = _emit_event(out, fc, prev, sf, lane, op)
            emitted.append((sf, lane, op))
            lane_hist.setdefault(lane, []).append((sf, op))
            lane_last[lane] = sf
    return out


def ids_to_events(ids):
    events = []
    i = 0
    prev = 0
    N = len(ids)
    fc = _FreqCur()
    lane_hist = {}  # lane -> list of (sf, op); LREPLAY source
    lane_last = {}  # lane -> last frame emitted for that lane
    while i < N:
        if ids[i] == REPEAT_MARKER:
            i += 1
            off, i = _ruint(ids, i)
            ln, i = _ruint(ids, i)
            start = len(events) - off
            src_prev = events[start - 1][0] if start > 0 else 0
            for k in range(ln):
                src_sf, lane, op = events[start + k]
                prev = prev + (src_sf - src_prev)
                src_prev = src_sf
                if lane < N_FREQ_LANES:
                    fc.step(lane, op)
                events.append((prev, lane, op))
                lane_hist.setdefault(lane, []).append((prev, op))
                lane_last[lane] = prev
            continue
        if ids[i] == LREPLAY_MARKER:
            i += 1
            lane = ids[i] - LANE_BASE
            i += 1
            eback, i = _ruint(ids, i)
            ln, i = _ruint(ids, i)
            hist = lane_hist[lane]
            start = len(hist) - eback
            src_prev = hist[start - 1][0] if start > 0 else 0
            lf = lane_last[lane]
            for k in range(ln):
                src_sf, op = hist[start + k]
                lf = lf + (src_sf - src_prev)
                src_prev = src_sf
                if lane < N_FREQ_LANES:
                    fc.step(lane, op)
                events.append((lf, lane, op))
                hist.append((lf, op))
            lane_last[lane] = lf
            prev = lf
            continue
        dt, i = _rint(ids, i)
        prev += dt
        lane = ids[i] - LANE_BASE
        i += 1
        kind = OPS_INV[ids[i] - OP_BASE]
        i += 1
        is_freq = lane < N_FREQ_LANES
        if kind == "NOTE":
            v, i = _rint(ids, i)
            if is_freq:
                if not fc.phase_seeded[lane]:
                    pq, i = _ruint(ids, i)  # one-time per-voice phase seed
                else:
                    pq = fc.phase_ref(lane)
                # mirror the encoder's table factoring: first sight at this absolute
                # index reads the absolute residual (which fc.step will seed into the
                # table); a later onset reads a signed delta off the established
                # residual. idx is computed identically to the encoder/decoder freq
                # accumulator (fc.step has not yet run for this op).
                idx = fc.next_idx(lane, v)
                ref = fc.micro_ref(lane, idx)
                stored, i = _rint(ids, i)
                micro = stored if ref is None else stored + ref
                op = ("NOTE", v, micro, pq)
            else:
                op = ("NOTE", v)
        elif kind == "LOAD":
            v, i = _ruint(ids, i)
            op = ("LOAD", v)
        elif kind in ("RAW", "REST"):
            v, i = _ruint(ids, i)
            op = (kind, v)
        elif kind == "SLIDE":
            rate, i = _rint(ids, i)
            n, i = _ruint(ids, i)
            delta, i = _rint(ids, i)
            op = ("SLIDE", rate, n, delta + fc.ref(lane))
        elif kind == "VIB":
            s_up, i = _rint(ids, i)
            u, i = _ruint(ids, i)
            s_dn, i = _rint(ids, i)
            d, i = _ruint(ids, i)
            rot, i = _ruint(ids, i)
            n, i = _ruint(ids, i)
            delta, i = _rint(ids, i)
            op = ("VIB", s_up, u, s_dn, d, rot, n, delta + fc.ref(lane))
        elif kind == "ARP":
            ns, i = _ruint(ids, i)
            steps = []
            for _ in range(ns):
                ld, i = _rint(ids, i)
                w, i = _ruint(ids, i)
                steps.append((ld, w))
            n, i = _ruint(ids, i)
            delta, i = _rint(ids, i)
            op = ("ARP", tuple(steps), n, delta + fc.ref(lane))
        elif kind == "NSLIDE":
            rate, i = _rint(ids, i)
            n, i = _ruint(ids, i)
            op = ("NSLIDE", rate, n)
        elif kind == "NVIB":
            s_up, i = _rint(ids, i)
            u, i = _ruint(ids, i)
            s_dn, i = _rint(ids, i)
            d, i = _ruint(ids, i)
            rot, i = _ruint(ids, i)
            n, i = _ruint(ids, i)
            op = ("NVIB", s_up, u, s_dn, d, rot, n)
        elif kind == "WALK":
            ns, i = _ruint(ids, i)
            steps = []
            for _ in range(ns):
                off, i = _rint(ids, i)
                w, i = _ruint(ids, i)
                steps.append((off, w))
            n, i = _ruint(ids, i)
            op = ("WALK", tuple(steps), n)
        elif kind == "RUN":  # non-freq RUN: SPARSE period decode
            p, i = _ruint(ids, i)
            n, i = _ruint(ids, i)
            nnz, i = _ruint(ids, i)
            ds = [0] * p
            k = -1
            for _ in range(nnz):
                gap, i = _ruint(ids, i)
                d, i = _rint(ids, i)
                k += gap + 1
                ds[k] = d
            op = ("RUN", tuple(ds), n)
        else:  # MOD (freq lane)
            p, i = _ruint(ids, i)
            n, i = _ruint(ids, i)
            ds = []
            for _ in range(p):
                d, i = _rint(ids, i)
                ds.append(d)
            if is_freq:
                delta, i = _rint(ids, i)  # base delta off running register Fn
                op = (kind, tuple(ds), n, delta + fc.ref(lane))
            else:
                op = (kind, tuple(ds), n)
        if is_freq:
            fc.step(lane, op)
        events.append((prev, lane, op))
        lane_hist.setdefault(lane, []).append((prev, op))
        lane_last[lane] = prev
    return events
