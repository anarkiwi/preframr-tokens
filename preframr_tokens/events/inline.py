"""Inline decompiler codec over a fixed small atom alphabet (no ids/escape/frozen
table). SETTLED non-env lanes (freq/pw/fc/res/vol) become NOTE/MOD pitch + LOAD/RUN
delta-run gestures (settling is audio-safe); ctrl/AD/SR (env regs) become the ORDERED
WRITE stream (de-duped same-reg-same-val no-ops) preserving envelope/hard-restart
order. Any prefix is a continuable song; BPE over the atoms is the dictionary."""

from __future__ import annotations

import numpy as np

PMAX = 96
NUM_REGS = 25

ENV_REGS: tuple[int, ...] = (4, 5, 6, 11, 12, 13, 18, 19, 20)
ENV_IDX = {r: i for i, r in enumerate(ENV_REGS)}

NONENV_LANES: list[tuple[str, tuple[int, ...]]] = []
for _v in range(3):
    NONENV_LANES.append(("freq", (7 * _v, 7 * _v + 1)))
for _v in range(3):
    NONENV_LANES.append(("pw", (7 * _v + 2, 7 * _v + 3)))
for _reg in (21, 22, 23, 24):
    NONENV_LANES.append(("byte", (_reg,)))

NUM_NONENV = len(NONENV_LANES)
NUM_ENV = len(ENV_REGS)
NUM_LANES = NUM_NONENV + NUM_ENV

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

ENV_SUB = 1 << 20


def lane_seq(grid: np.ndarray, kind: str, regs: tuple[int, ...]) -> np.ndarray:
    """Recover a non-env lane's per-frame value series from the settled grid: freq
    combines lo/hi to 16 bits, pw combines lo/hi (masked) to 12 bits, every other
    lane is a single byte."""
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


def encode_events(grid: np.ndarray, env_writes) -> list[tuple[int, int, tuple]]:
    """Merge the settled non-env lanes (NOTE/MOD/LOAD/RUN gestures, implicit holds
    dropped) with the ORDERED ``(frame, reg, val)`` env writes into one ``[(frame, sub,
    payload)]`` stream sorted by ``(frame, sub)``; a lane payload is ``("L", lane, op)``,
    an env payload is ``("W", reg, val)``, and env events sort after lane events in a
    frame (``ENV_SUB`` offset) keeping their write order."""
    events: list[tuple[int, int, tuple]] = []
    for lid, (kind, regs) in enumerate(NONENV_LANES):
        seq = lane_seq(grid, kind, regs).astype(np.int64)
        toks = encode_freq(seq) if kind == "freq" else encode_lane(seq)
        f = 0
        cur = 0
        for op in toks:
            length = 1 if op[0] in ("NOTE", "LOAD") else op[2]
            hold = (op[0] == "LOAD" and op[1] == cur) or _is_hold(op)
            if not hold:
                events.append((f, lid, ("L", lid, op)))
            if op[0] == "NOTE":
                cur += op[1]
            elif op[0] == "LOAD":
                cur = op[1]
            else:
                p = len(op[1])
                for k in range(length):
                    cur += op[1][k % p]
            f += length
    frame_n: dict[int, int] = {}
    for fr, reg, val in env_writes:
        i = frame_n.get(fr, 0)
        frame_n[fr] = i + 1
        events.append((fr, ENV_SUB + i, ("W", reg, val)))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def decode_events(events, n_frames: int):
    """Replay an event stream into ``(settled (n_frames, 25) grid, ordered env
    writes)``. Env regs in the grid are forward-filled from the replayed writes so
    the grid is a faithful settled view; the ordered env writes are the fidelity
    target."""
    lane_ops: list[list[tuple[int, tuple]]] = [[] for _ in NONENV_LANES]
    env_writes: list[tuple[int, int, int]] = []
    for fr, _sub, payload in events:
        if payload[0] == "L":
            lane_ops[payload[1]].append((fr, payload[2]))
        else:
            env_writes.append((fr, payload[1], payload[2]))
    grid = np.zeros((n_frames, NUM_REGS), dtype=np.int64)
    for lid, (kind, regs) in enumerate(NONENV_LANES):
        out = np.zeros(n_frames, dtype=np.int64)
        cur = 0
        fr = 0
        for sf, op in lane_ops[lid]:
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
    envcols = {r: 0 for r in ENV_REGS}
    by_frame: dict[int, list[tuple[int, int]]] = {}
    for fr, reg, val in env_writes:
        by_frame.setdefault(fr, []).append((reg, val))
    for f in range(n_frames):
        for reg, val in by_frame.get(f, []):
            envcols[reg] = val
        for reg in ENV_REGS:
            grid[f, reg] = envcols[reg]
    return grid, env_writes


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
    """Flatten the event stream to atom ids: ``[DT][SELECTOR][...]`` per event, DT as
    an unsigned inter-event frame delta. A lane selector (``<NUM_NONENV``) is followed
    by ``[OP][params]``; an env selector (``>=NUM_NONENV``) is followed by the written
    value."""
    out: list[int] = []
    prev = 0
    for sf, _sub, payload in events:
        _emit_u(out, sf - prev)
        prev = sf
        if payload[0] == "L":
            out.append(LANE_BASE + payload[1])
            op = payload[2]
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
        else:
            out.append(LANE_BASE + NUM_NONENV + ENV_IDX[payload[1]])
            _emit_u(out, payload[2])
    return out


def ids_to_events(ids) -> list[tuple[int, int, tuple]]:
    """Inverse of :func:`events_to_ids`. Env events get an ascending intra-stream
    ``ENV_SUB`` sub-order so they keep their write order on re-sort."""
    events: list[tuple[int, int, tuple]] = []
    i = 0
    prev = 0
    n = len(ids)
    frame_n: dict[int, int] = {}
    while i < n:
        dt, i = _read_u(ids, i)
        prev += dt
        sel = ids[i] - LANE_BASE
        i += 1
        if sel < NUM_NONENV:
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
            events.append((prev, sel, ("L", sel, op)))
        else:
            val, i = _read_u(ids, i)
            reg = ENV_REGS[sel - NUM_NONENV]
            sub = frame_n.get(prev, 0)
            frame_n[prev] = sub + 1
            events.append((prev, ENV_SUB + sub, ("W", reg, val)))
    return events


def is_digit_atom(tok: int) -> bool:
    return DIGIT_BASE <= tok < VOCAB_SIZE


def encode_target(grid: np.ndarray, env_writes) -> list[int]:
    """Settled non-env grid + ordered env writes -> flat atom ids."""
    return events_to_ids(encode_events(grid, env_writes))


def decode_target(ids, n_frames: int):
    """Flat atom ids + frame count -> ``(settled (n_frames, 25) grid, ordered env
    writes)``."""
    return decode_events(ids_to_events(ids), n_frames)
