"""STEP-LEVEL (tracker) representation of the SID decompiler codec, scoped to ONE
tune (Monty). The thesis: frames are a playback artifact; the composer wrote
tracker STEPS (pattern rows) that the playroutine RENDERS to frames.

This is a LAYER ON TOP of the existing per-frame event codec (serialize_events):
its NOTE/VIB/SLIDE/ARP/MOD/RUN ops ARE the render, and decode_events() IS the
renderer. We do not touch them. We re-organize the same EVENT SET into a tracker
structure and re-serialize it, then prove decode(step) -> identical events ->
identical 25-register per-frame state (residual = 0 by construction, since the
events fed to decode_events are byte-identical).

Structure built here:
  1. Quantize onsets to the 4-frame STEP grid (validated: gate triggers + freq
     onsets cluster on multiples of 4; a handful of voice-0 off-grid onsets are
     carried as exact frame offsets, not snapped -> still residual-zero).
  2. Per VOICE, group events into ROWS: a row begins at a freq onset (NOTE/RAW/
     REST) and bundles every non-freq voice-lane op (ctrl/ad/sr/pw) that fires
     within the row, expressed RELATIVE to the row's onset step.
  3. INSTRUMENT = the bundle of (ad, sr, pw, ctrl) generators over a row, with
     frame offsets relative to onset. Dedup identical bundles -> a small
     instrument set; a row references an instrument by index, established INLINE
     on first use (backward-only).
  4. NOTE-ROW = (pitch_op, duration_steps, instrument_ref). Dedup identical rows.
  5. ORDERLIST = per-voice backward references to repeated row-blocks (the
     pattern dedup), established inline on first play.

The global filter lanes (fc/res/vol, regs 21..24) are voiceless; they are carried
as a 4th "voice" row stream keyed only on their own op onsets (no instrument
factoring -- they are the tune's filter automation, kept literal/LZ'd).

Serialization mirrors serialize_tokens' base-16 LEB (digit 0-15 continue, 16-31
terminal; zigzag for signed) so the token counts are directly comparable.
"""

import collections
import math

from preframr_tokens.codec import serialize_events as SE

STEP = 4  # DEFAULT frames per tracker step (Monty); per-tune grid detected below


def detect_step_grid(s, default=STEP, lo=1, hi=32):
    """Detect the tracker STEP grid (frames per row) for ONE tune from s.

    The grid is the tune's tempo (frames-per-row) and varies per tune (Monty=4,
    5_Title_Tunes=8, Action_Biker=12). It is recovered from the GATE-RISE
    (note-on) frames across the 3 voices: take the inter-trigger GAPS, keep only
    the FREQUENT gaps (count >= max(3, 5% of gaps)), and take their GCD.

    Keeping only frequent gaps is essential: a naive GCD over ALL gaps collapses
    to 1 on a single off-grid outlier. The GCD of the frequent gaps recovers the
    true grid and handles funktempo/swing (e.g. {4,8} -> 4). Falls back to
    `default` when there are too few triggers or the result is out of [lo, hi].
    """
    gaps = []
    for v in range(3):
        b = 7 * v
        gate = s[:, b + 4] & 1
        if len(gate) == 0:
            continue
        rises = [i for i in range(1, len(gate)) if gate[i] == 1 and gate[i - 1] == 0]
        if gate[0] == 1:
            rises = [0] + rises
        gaps.extend(rises[k + 1] - rises[k] for k in range(len(rises) - 1))
    gaps = [g for g in gaps if g > 0]
    if len(gaps) < 4:
        return default
    cnt = collections.Counter(gaps)
    thr = max(3, int(0.05 * len(gaps)))
    frequent = [g for g, c in cnt.items() if c >= thr]
    if not frequent:
        return default
    g = 0
    for x in frequent:
        g = math.gcd(g, x)
    if g < lo or g > hi:
        return default
    return g


# voice lanes: freq lane id == v, plus the non-freq lanes that share the voice.
# lane ids: 0..2 freq, 3..5 pw, 6..8 ctrl, 9..11 ad, 12..14 sr, 15..18 global.
FREQ_LANE = {0: 0, 1: 1, 2: 2}
VOICE_LANES = {
    0: {"freq": 0, "pw": 3, "ctrl": 6, "ad": 9, "sr": 12},
    1: {"freq": 1, "pw": 4, "ctrl": 7, "ad": 10, "sr": 13},
    2: {"freq": 2, "pw": 5, "ctrl": 8, "ad": 14, "sr": 14},  # fixed below
}
# rebuild correctly: sr lanes are 12,13,14 for v0,1,2
VOICE_LANES = {
    v: {"freq": v, "pw": 3 + v, "ctrl": 6 + v, "ad": 9 + v, "sr": 12 + v}
    for v in range(3)
}
GLOBAL_LANES = [15, 16, 17, 18]  # fc-lo, fc-hi, res, vol


# ---------- base-16 LEB sizing (mirror serialize_tokens, token-count only) ----------
def u_cost(n):
    n = int(n)
    c = 1
    while n > 15:
        n >>= 4
        c += 1
    return c


def i_cost(n):
    n = int(n)
    return u_cost((n << 1) ^ (n >> 63))


def _op_body_cost(op, is_freq):
    """Token cost of an op's PARAMETER body (no dt/lane/op header). Mirrors
    serialize_tokens._emit_event_body field-for-field, but WITHOUT the freq
    base/micro/phase delta context (the step layer re-derives those; we measure
    the absolute-form body so the comparison is conservative / upper-bound)."""
    nm = op[0]
    if nm == "NOTE":
        c = i_cost(op[1])
        if is_freq:
            c += i_cost(op[2])  # micro (phase folded once per voice, see header)
        return c
    if nm in ("LOAD", "RAW", "REST"):
        return u_cost(op[1])
    if nm == "SLIDE":
        return i_cost(op[1]) + u_cost(op[2]) + i_cost(op[3])
    if nm == "VIB":
        return (
            i_cost(op[1])
            + u_cost(op[2])
            + i_cost(op[3])
            + u_cost(op[4])
            + u_cost(op[5])
            + u_cost(op[6])
            + i_cost(op[7])
        )
    if nm == "ARP":
        c = u_cost(len(op[1]))
        for ld, w in op[1]:
            c += i_cost(ld) + u_cost(w)
        return c + u_cost(op[2]) + i_cost(op[3])
    if nm == "NSLIDE":
        return i_cost(op[1]) + u_cost(op[2])
    if nm == "NVIB":
        return (
            i_cost(op[1])
            + u_cost(op[2])
            + i_cost(op[3])
            + u_cost(op[4])
            + u_cost(op[5])
            + u_cost(op[6])
        )
    if nm == "WALK":
        c = u_cost(len(op[1]))
        for off, w in op[1]:
            c += i_cost(off) + u_cost(w)
        return c + u_cost(op[2])
    if nm == "RUN":
        e = op[1]
        c = u_cost(len(e)) + u_cost(op[2])
        nz = [(k, d) for k, d in enumerate(e) if d != 0]
        c += u_cost(len(nz))
        prev_k = -1
        for k, d in nz:
            c += u_cost(k - prev_k - 1) + i_cost(d)
            prev_k = k
        return c
    if nm == "MOD":
        c = u_cost(len(op[1])) + u_cost(op[2])
        for d in op[1]:
            c += i_cost(d)
        if is_freq:
            c += i_cost(op[3])
        return c
    raise ValueError(nm)


# ---------- build the step structure ----------
def build_steps(s, step=None):
    """Return the tracker structure for the 25-register state s.

    rows[v]  = list of row dicts (per voice 0..2): {
        'step'     : onset frame // STEP (quantized),
        'frame_off': onset frame % STEP (0 except a few off-grid v0 onsets),
        'pitch'    : the freq op tuple (NOTE/RAW/REST/SLIDE/VIB/ARP/MOD),
        'dur_steps': duration in steps until the next row onset,
        'instr'    : instrument key (tuple of (rel_frame, lane_kind, op)),
    }
    global_events = list of (frame, lane_id, op) for the filter lanes (kept as-is).
    """
    if step is None:
        step = detect_step_grid(s)
    ev = SE.encode_tune_events(s[:, :25])
    bylane = collections.defaultdict(list)
    for sf, lid, op in ev:
        bylane[lid].append((sf, op))

    rows = {}
    for v in range(3):
        L = VOICE_LANES[v]
        fl = L["freq"]
        freq_ops = sorted(bylane[fl], key=lambda x: x[0])
        # row onsets = freq op start frames (each NOTE/RAW/REST/generator that the
        # freq lane emits begins a row). Generators (SLIDE/VIB/ARP/MOD) that ride a
        # held pitch also start a freq event -> they begin their own row; that is
        # fine, a "row" is just a freq-op-delimited segment.
        onsets = [sf for sf, _ in freq_ops]
        non_freq = []
        for kind in ("pw", "ctrl", "ad", "sr"):
            for sf, op in bylane[L[kind]]:
                non_freq.append((sf, kind, op))
        non_freq.sort()
        vrows = []
        nf_i = 0
        n = len(freq_ops)
        for ri, (sf, fop) in enumerate(freq_ops):
            nxt = onsets[ri + 1] if ri + 1 < n else None
            seg_end = nxt if nxt is not None else len(s)
            # bundle non-freq ops whose start frame is within [sf, seg_end)
            bundle = []
            while nf_i < len(non_freq) and non_freq[nf_i][0] < seg_end:
                nsf, kind, op = non_freq[nf_i]
                if nsf >= sf:
                    bundle.append((nsf - sf, kind, op))  # offset relative to onset
                nf_i += 1
            instr_key = tuple(bundle)
            vrows.append(
                {
                    "step": sf // step,
                    "frame_off": sf % step,
                    "pitch": fop,
                    "dur_frames": (seg_end - sf),
                    "instr": instr_key,
                }
            )
        rows[v] = vrows

    global_events = []
    for lid in GLOBAL_LANES:
        for sf, op in bylane[lid]:
            global_events.append((sf, lid, op))
    global_events.sort()
    return rows, global_events, ev


# ---------- residual-zero decode: reconstruct the identical event set ----------
def steps_to_events(rows, global_events, T, step=STEP):
    """Expand the tracker structure back to the flat (frame, lane, op) event set.
    Must reproduce encode_tune_events(s) exactly so decode_events renders identically.
    """
    ev = []
    for v in range(3):
        L = VOICE_LANES[v]
        for row in rows[v]:
            sf = row["step"] * step + row["frame_off"]
            ev.append((sf, L["freq"], row["pitch"]))
            for rel, kind, op in row["instr"]:
                ev.append((sf + rel, L[kind], op))
    ev.extend(global_events)
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


# ---------- instrument / row / pattern dedup + token accounting ----------
def serialize_cost(rows, global_events, step=STEP):
    """Token-count the tracker serialization. Components:
      INSTRUMENTS: distinct non-freq bundles, each emitted ONCE (inline on first
                   use) as [n_ops] then per op [rel_frame][lane_kind][op-body].
                   Later rows reference by [instr_idx] (a uint).
      NOTE-ROWS:   per voice, per row: [pitch op-body] [dur_steps] [instr_ref].
                   Dedup'd row-blocks (orderlist) replaced by a backward ref
                   [ORDER_MARK][block_back][block_len].
      GLOBAL:      filter automation events, literal [dt][lane][op-body].
    Returns a dict breakdown + total."""
    cost = collections.Counter()

    # ---- instrument table (established inline, referenced by index) ----
    instr_index = {}
    instr_defs = []  # list of bundles in first-use order
    # assign indices in first-use order across all voices in step/time order
    all_rows = []
    for v in range(3):
        for row in rows[v]:
            all_rows.append((row["step"], v, row))
    all_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, row in all_rows:
        key = row["instr"]
        if key not in instr_index:
            instr_index[key] = len(instr_defs)
            instr_defs.append(key)

    # cost of each instrument DEFINITION (emitted once)
    for bundle in instr_defs:
        cost["instr_def"] += u_cost(len(bundle))  # n ops in bundle
        for rel, _kind, op in bundle:
            cost["instr_def"] += u_cost(rel)  # rel frame offset
            cost["instr_def"] += 1  # lane-kind atom (pw/ctrl/ad/sr)
            cost["instr_def"] += 1  # op atom
            cost["instr_def"] += _op_body_cost(op, is_freq=False)

    # ---- per-voice note-row streams with pattern (orderlist) dedup ----
    for v in range(3):
        vrows = rows[v]
        # canonical row token: (pitch_op, dur_steps, instr_idx). Build the literal
        # row token list, then LZ over rows (backward block refs = the orderlist).
        row_tokens = []
        for row in vrows:
            dur_steps = row["dur_frames"] // step
            frac = row["dur_frames"] % step
            row_tokens.append(
                (
                    row["pitch"],
                    dur_steps,
                    frac,
                    row["frame_off"],
                    instr_index[row["instr"]],
                )
            )

        # cost a row literally
        def row_cost(rt):
            pitch, dur_steps, frac, foff, instr = rt
            c = 1  # op atom for the pitch
            c += _op_body_cost(pitch, is_freq=True)
            c += u_cost(dur_steps) + u_cost(frac) + u_cost(foff)
            c += u_cost(instr)  # instrument reference
            return c

        # greedy backward LZ over the row stream (the orderlist)
        i = 0
        N = len(row_tokens)
        # precompute literal cost prefix
        while i < N:
            best_len, best_back = 0, 0
            # find the longest backward match starting at i
            for back in range(1, i + 1):
                ln = 0
                while (
                    i + ln < N
                    and back + ln <= i
                    and row_tokens[i - back + ln] == row_tokens[i + ln]
                ):
                    ln += 1
                    if i - back + ln > i:  # don't overlap past current
                        break
                if ln > best_len:
                    best_len, best_back = ln, back
            ref_cost = 1 + u_cost(best_back) + u_cost(best_len)  # ORDER_MARK+back+len
            lit_cost = row_cost(row_tokens[i])
            if best_len >= 2 and ref_cost < sum(
                row_cost(row_tokens[i + k]) for k in range(best_len)
            ):
                cost["orderlist"] += ref_cost
                i += best_len
            else:
                cost["note_rows"] += lit_cost
                i += 1

    # ---- global filter automation (literal, LZ'd over events) ----
    prev = 0
    for sf, _lid, op in global_events:
        cost["global"] += i_cost(sf - prev)  # dt
        prev = sf
        cost["global"] += 1  # lane atom
        cost["global"] += 1  # op atom
        cost["global"] += _op_body_cost(op, is_freq=False)

    cost["total"] = sum(v for k, v in cost.items() if k != "total")
    return dict(cost), instr_defs, instr_index
