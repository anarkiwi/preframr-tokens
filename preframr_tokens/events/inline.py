"""Inline decompiler codec: a SID per-frame settled register grid becomes ONE
time-ordered event stream of relative pitch and delta-run gestures, then a flat
list of fixed atom ids. Lossless against the settled grid by construction (no SET
op, no escape lane, no frozen table); any prefix is a valid, continuable song
(no preamble, no forward declaration). BPE over the atoms is the dictionary."""

from __future__ import annotations

import numpy as np

PMAX = 96
NUM_REGS = 25

LANES: list[tuple[str, tuple[int, ...]]] = []
for _v in range(3):
    LANES.append(("freq", (7 * _v, 7 * _v + 1)))
for _v in range(3):
    LANES.append(("pw", (7 * _v + 2, 7 * _v + 3)))
for _v in range(3):
    LANES.append(("ctrl", (7 * _v + 4,)))
for _v in range(3):
    LANES.append(("ad", (7 * _v + 5,)))
for _v in range(3):
    LANES.append(("sr", (7 * _v + 6,)))
for _reg in (21, 22, 23, 24):
    LANES.append(("byte", (_reg,)))

NUM_LANES = len(LANES)

LANE_BASE = 0
OP_BASE = NUM_LANES
DIGIT_BASE = OP_BASE + 4
VOCAB_SIZE = DIGIT_BASE + 32

OPS = {"NOTE": 0, "LOAD": 1, "MOD": 2, "RUN": 3}
OPS_INV = {v: k for k, v in OPS.items()}

NOTE_OP = OP_BASE + OPS["NOTE"]
LOAD_OP = OP_BASE + OPS["LOAD"]
MOD_OP = OP_BASE + OPS["MOD"]
RUN_OP = OP_BASE + OPS["RUN"]


def lane_seq(grid: np.ndarray, kind: str, regs: tuple[int, ...]) -> np.ndarray:
    """Recover a lane's per-frame value series from the settled 25-register grid:
    freq combines lo/hi to 16 bits, pw combines lo/hi (masked) to 12 bits, every
    other lane is a single byte."""
    if kind == "freq":
        return grid[:, regs[0]] + 256 * grid[:, regs[1]]
    if kind == "pw":
        return grid[:, regs[0]] + 256 * (grid[:, regs[1]] & 0xF)
    return grid[:, regs[0]]


def _delta_run(df, i, avail):
    best_n, best_p = 0, 1
    for p in range(1, min(PMAX, avail) + 1):
        n = 0
        while n < avail and df[i + n] == df[i + (n % p)]:
            n += 1
        if n >= 2 * p and n > best_n:
            best_n, best_p = n, p
    return best_n, best_p


def encode_freq(v) -> list[tuple]:
    """Relative pitch lane: NOTE(interval) is an inline pitch step, MOD(deltas, n)
    is an n-frame periodic delta run (vibrato/glide). Everything is relative to the
    running freq, so the lane is transposition-invariant and continuable."""
    v = [int(x) for x in v]
    t = len(v)
    if t == 0:
        return []
    df = [v[0]] + [v[k] - v[k - 1] for k in range(1, t)]
    toks: list[tuple] = []
    i = 0
    while i < t:
        best_n, best_p = _delta_run(df, i, t - i)
        if best_n >= 2:
            toks.append(("MOD", tuple(df[i : i + best_p]), best_n))
            i += best_n
        else:
            toks.append(("NOTE", df[i]))
            i += 1
    return toks


def encode_lane(v) -> list[tuple]:
    """Generic lane: LOAD(value) jumps to a value (note/wavetable/table entry),
    RUN(deltas, n) is an n-frame periodic delta run (sweep/sustain/vibrato/PWM)."""
    v = [int(x) for x in v]
    t = len(v)
    if t == 0:
        return []
    df = [v[0]] + [v[k] - v[k - 1] for k in range(1, t)]
    toks: list[tuple] = []
    i = 0
    while i < t:
        best_n, best_p = _delta_run(df, i, t - i)
        if best_n >= 2:
            toks.append(("RUN", tuple(df[i : i + best_p]), best_n))
            i += best_n
        else:
            toks.append(("LOAD", v[i]))
            i += 1
    return toks


def _is_hold(op) -> bool:
    if op[0] == "NOTE":
        return op[1] == 0
    if op[0] == "LOAD":
        return False
    return all(d == 0 for d in op[1])


def encode_events(grid: np.ndarray) -> list[tuple[int, int, tuple]]:
    """Merge all 17 lanes into one time-ordered ``[(start_frame, lane, op)]`` event
    stream with implicit holds dropped, sorted by ``(start_frame, lane)``."""
    events: list[tuple[int, int, tuple]] = []
    for lid, (kind, regs) in enumerate(LANES):
        seq = lane_seq(grid, kind, regs).astype(np.int64)
        toks = encode_freq(seq) if kind == "freq" else encode_lane(seq)
        f = 0
        cur = 0
        for op in toks:
            length = 1 if op[0] in ("NOTE", "LOAD") else op[2]
            hold = (op[0] == "LOAD" and op[1] == cur) or _is_hold(op)
            if not hold:
                events.append((f, lid, op))
            if op[0] == "NOTE":
                cur += op[1]
            elif op[0] == "LOAD":
                cur = op[1]
            else:
                p = len(op[1])
                for k in range(length):
                    cur += op[1][k % p]
            f += length
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def decode_events(events, n_frames: int) -> np.ndarray:
    """Replay an event stream back into the settled ``(n_frames, 25)`` register grid."""
    grid = np.zeros((n_frames, NUM_REGS), dtype=np.int64)
    lane_vals: list[list[tuple[int, tuple]]] = [[] for _ in LANES]
    for sf, lid, op in events:
        lane_vals[lid].append((sf, op))
    for lid, (kind, regs) in enumerate(LANES):
        out = np.zeros(n_frames, dtype=np.int64)
        cur = 0
        fr = 0
        for sf, op in lane_vals[lid]:
            if sf > fr:
                out[fr:sf] = cur
            if op[0] == "NOTE":
                cur += op[1]
                out[sf] = cur
                fr = sf + 1
            elif op[0] == "LOAD":
                cur = op[1]
                out[sf] = cur
                fr = sf + 1
            else:
                deltas, n, p = op[1], op[2], len(op[1])
                for k in range(n):
                    cur += deltas[k % p]
                    out[sf + k] = cur
                fr = sf + n
        if fr < n_frames:
            out[fr:n_frames] = cur
        if kind in ("freq", "pw"):
            grid[:, regs[0]] = out & 0xFF
            grid[:, regs[1]] = out >> 8
        else:
            grid[:, regs[0]] = out
    return grid


def _emit_u(out: list[int], n: int) -> None:
    n = int(n)
    while True:
        d = n & 0xF
        n >>= 4
        out.append(DIGIT_BASE + (d if n else 16 + d))
        if not n:
            return


def _emit_s(out: list[int], n: int) -> None:
    n = int(n)
    _emit_u(out, (n << 1) ^ (n >> 63))


def _read_u(ids, i: int) -> tuple[int, int]:
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


def _read_s(ids, i: int) -> tuple[int, int]:
    z, i = _read_u(ids, i)
    return (z >> 1) ^ -(z & 1), i


def events_to_ids(events) -> list[int]:
    """Flatten the event stream to atom ids: ``[DT][LANE][OP][params]`` per event,
    DT as an unsigned inter-event frame delta."""
    out: list[int] = []
    prev = 0
    for sf, lane, op in events:
        _emit_u(out, sf - prev)
        prev = sf
        out.append(LANE_BASE + lane)
        out.append(OP_BASE + OPS[op[0]])
        if op[0] == "NOTE":
            _emit_s(out, op[1])
        elif op[0] == "LOAD":
            _emit_u(out, op[1])
        else:
            _emit_u(out, len(op[1]))
            _emit_u(out, op[2])
            for d in op[1]:
                _emit_s(out, d)
    return out


def ids_to_events(ids) -> list[tuple[int, int, tuple]]:
    """Inverse of :func:`events_to_ids`."""
    events: list[tuple[int, int, tuple]] = []
    i = 0
    prev = 0
    n = len(ids)
    while i < n:
        dt, i = _read_u(ids, i)
        prev += dt
        lane = ids[i] - LANE_BASE
        i += 1
        kind = OPS_INV[ids[i] - OP_BASE]
        i += 1
        if kind == "NOTE":
            val, i = _read_s(ids, i)
            op: tuple = ("NOTE", val)
        elif kind == "LOAD":
            val, i = _read_u(ids, i)
            op = ("LOAD", val)
        else:
            p, i = _read_u(ids, i)
            cnt, i = _read_u(ids, i)
            deltas = []
            for _ in range(p):
                d, i = _read_s(ids, i)
                deltas.append(d)
            op = (kind, tuple(deltas), cnt)
        events.append((prev, lane, op))
    return events


def is_digit_atom(tok: int) -> bool:
    return DIGIT_BASE <= tok < VOCAB_SIZE


def encode_grid(grid: np.ndarray) -> list[int]:
    """Settled ``(n_frames, 25)`` grid -> flat atom ids."""
    return events_to_ids(encode_events(grid))


def decode_grid(ids, n_frames: int) -> np.ndarray:
    """Flat atom ids + frame count -> settled ``(n_frames, 25)`` grid."""
    return decode_events(ids_to_events(ids), n_frames)
