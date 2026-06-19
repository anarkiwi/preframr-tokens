"""The shared atom alphabet and the settled non-env lane primitives. SETTLED lanes
(freq/pw/fc/res/vol) become NOTE/MOD pitch + LOAD/RUN delta-run gestures; the alphabet
also carries the env-half instrument markers (VOICE/DEF/REF/RAW/LEAD, see
:mod:`preframr_tokens.events.instrument`) and the backward span-copy SEQREF marker (see
:mod:`preframr_tokens.events.seqref`). Backward-only, no preamble/codebook/escape."""

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
NUM_VOICES = 3
NUM_ITEMS = 4

LANE_BASE = 0
VOICE_BASE = LANE_BASE + NUM_NONENV
ITEM_BASE = VOICE_BASE + NUM_VOICES
OP_BASE = ITEM_BASE + NUM_ITEMS
SEQREF_OP = OP_BASE + 4
DIGIT_BASE = SEQREF_OP + 1
VOCAB_SIZE = DIGIT_BASE + 32

NUM_LANES = NUM_NONENV + NUM_VOICES + NUM_ITEMS

ITEMS = {"DEF": 0, "REF": 1, "RAW": 2, "LEAD": 3}
ITEMS_INV = {v: k for k, v in ITEMS.items()}
DEF_ITEM = ITEM_BASE + ITEMS["DEF"]
REF_ITEM = ITEM_BASE + ITEMS["REF"]
RAW_ITEM = ITEM_BASE + ITEMS["RAW"]
LEAD_ITEM = ITEM_BASE + ITEMS["LEAD"]

OPS = {"NOTE": 0, "LOAD": 1, "MOD": 2, "RUN": 3}
OPS_INV = {v: k for k, v in OPS.items()}

NOTE_OP = OP_BASE + OPS["NOTE"]
LOAD_OP = OP_BASE + OPS["LOAD"]
MOD_OP = OP_BASE + OPS["MOD"]
RUN_OP = OP_BASE + OPS["RUN"]


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


def lane_op_length(op) -> int:
    """The number of frames a lane op advances: 1 for NOTE/LOAD, the run count for
    MOD/RUN."""
    return 1 if op[0] in ("NOTE", "LOAD") else op[2]


def nonenv_lane_events(grid: np.ndarray, skip_lanes=()) -> list[tuple[int, int, tuple]]:
    """The settled non-env lanes as ``(frame, lane, ("L", lane, op))`` gesture events
    (NOTE/MOD pitch, LOAD/RUN delta-runs; implicit holds dropped), sorted by
    ``(frame, lane)``. ``skip_lanes`` omits lanes whose values are carried elsewhere
    (e.g. a synced-PW voice's pw lane folded into its instrument)."""
    events: list[tuple[int, int, tuple]] = []
    skip = frozenset(skip_lanes)
    for lid, (kind, regs) in enumerate(NONENV_LANES):
        if lid in skip:
            continue
        seq = lane_seq(grid, kind, regs).astype(np.int64)
        toks = encode_freq(seq) if kind == "freq" else encode_lane(seq)
        f = 0
        cur = 0
        for op in toks:
            length = lane_op_length(op)
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
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def lane_events_to_grid(
    events, n_frames: int, env_writes=(), skip_lanes=()
) -> np.ndarray:
    """Replay non-env lane gesture events back into a settled ``(n_frames, 25)`` grid;
    ``skip_lanes`` are left at 0 (their values are carried in the env writes). Env regs
    are forward-filled from ``env_writes`` so the grid is a faithful settled view (the
    ordered env writes are the actual fidelity target)."""
    skip = frozenset(skip_lanes)
    lane_ops: list[list[tuple[int, tuple]]] = [[] for _ in NONENV_LANES]
    for fr, _sub, payload in events:
        if payload[0] == "L":
            lane_ops[payload[1]].append((fr, payload[2]))
    grid = np.zeros((n_frames, NUM_REGS), dtype=np.int64)
    for lid, (kind, regs) in enumerate(NONENV_LANES):
        if lid in skip:
            continue
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


def emit_u(out: list[int], n: int) -> None:
    """Append an unsigned base-16 LEB varint over the DIGIT range."""
    _emit_u(out, n)


def emit_s(out: list[int], n: int) -> None:
    """Append a zigzag-signed base-16 LEB varint over the DIGIT range."""
    _emit_s(out, n)


def read_u(ids, i: int) -> tuple[int, int]:
    """Read an unsigned varint at ``i``; returns ``(value, next_index)``."""
    return _read_u(ids, i)


def read_s(ids, i: int) -> tuple[int, int]:
    """Read a zigzag-signed varint at ``i``; returns ``(value, next_index)``."""
    return _read_s(ids, i)


def emit_lane_op(out: list[int], lane: int, op) -> None:
    """Serialize one non-env lane gesture ``("L", lane, op)`` -> ``[LANE][OP][params]``
    (DT is emitted by the caller, which owns the inter-event delta)."""
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


def read_lane_op(ids, i: int):
    """Inverse of :func:`emit_lane_op` from just past the LANE selector; returns
    ``(op, next_index)``."""
    kind = OPS_INV[ids[i] - OP_BASE]
    i += 1
    if kind == "NOTE":
        val, i = _read_s(ids, i)
        return ("NOTE", val), i
    if kind == "LOAD":
        val, i = _read_u(ids, i)
        return ("LOAD", val), i
    p, i = _read_u(ids, i)
    cnt, i = _read_u(ids, i)
    deltas = []
    for _ in range(p):
        d, i = _read_s(ids, i)
        deltas.append(d)
    return (kind, tuple(deltas), cnt), i


def is_digit_atom(tok: int) -> bool:
    return DIGIT_BASE <= tok < VOCAB_SIZE
