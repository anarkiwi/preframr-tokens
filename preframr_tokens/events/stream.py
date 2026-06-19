"""Canonical inline-event stream codec. ``encode(ow)`` settles the non-env lanes and
captures ctrl/AD/SR as the ORDERED write stream, emitting the inline event stream;
``decode`` rebuilds both as ordered ``(frame, reg, val)`` writes. Lossless against the
audio-faithful ``corrected_writes(ow)`` (settled non-env grid + ordered env writes);
only inaudible intra-frame non-env intermediates and env same-value rewrites drop."""

from __future__ import annotations

import numpy as np

from . import inline
from .oracle import (
    ENV_REGS,
    NUM_REGS,
    OrderedWrites,
    corrected_writes,
    env_writes,
    settled_grid,
)

EVENT_FORMAT_VERSION = 5
VOCAB_SIZE = inline.VOCAB_SIZE

_ENV_SET = frozenset(ENV_REGS)


def is_content_atom(tok: int) -> bool:
    """Content = the payload digits the model must predict (intervals, deltas,
    durations, values); structural = lane tags and op markers."""
    return inline.is_digit_atom(tok)


def single_speed(ow: OrderedWrites) -> bool:
    """One player call per frame: no register is written more than once in any
    frame in the ordered stream."""
    if len(ow) == 0:
        return True
    seen: set[tuple[int, int]] = set()
    for f, r in zip(ow.frame.tolist(), ow.reg.tolist()):
        key = (f, r)
        if key in seen:
            return False
        seen.add(key)
    return True


def canonical_writes(ow: OrderedWrites) -> list[tuple[int, int, int]]:
    """The audio-faithful byte-exact target the codec reproduces: per frame the
    settled non-env register changes (ascending register) interleaved with the
    ORDERED ctrl/AD/SR writes in source order. Only intra-frame non-env intermediates
    and env same-value rewrites are canonicalized away (both inaudible)."""
    return corrected_writes(ow)


def encode(ow: OrderedWrites, verify: bool = True) -> list[int]:
    """Ordered writes -> flat inline-event atom ids. ``verify`` self-checks the
    byte-exact round trip against the corrected target (settled non-env grid + ordered
    env writes) and raises on a miss."""
    grid = settled_grid(ow)
    ew = env_writes(ow)
    ids = inline.encode_target(grid, ew)
    if verify:
        if decode(ids) != corrected_writes(ow):
            raise ValueError("inline codec not byte-exact against corrected target")
    return ids


def _span(events) -> int:
    span = 1
    for sf, _sub, payload in events:
        if payload[0] == "L":
            op = payload[2]
            length = op[2] if op[0] in ("MOD", "RUN") else 1
        else:
            length = 1
        span = max(span, sf + length)
    return span


def _interleave(grid: np.ndarray, ew) -> list[tuple[int, int, int]]:
    """The corrected-target ordering of decoded ``(grid, env_writes)``: per frame the
    settled non-env changes (ascending register) then the ordered env writes."""
    nonenv_changes: dict[int, list[tuple[int, int]]] = {}
    prev = np.zeros(NUM_REGS, dtype=np.int64)
    for f in range(grid.shape[0]):
        row = grid[f]
        for r in range(NUM_REGS):
            if r not in _ENV_SET and row[r] != prev[r]:
                nonenv_changes.setdefault(f, []).append((r, int(row[r])))
        prev = row
    env_by_frame: dict[int, list[tuple[int, int]]] = {}
    for f, r, v in ew:
        env_by_frame.setdefault(f, []).append((r, v))
    out: list[tuple[int, int, int]] = []
    frames = set(nonenv_changes) | set(env_by_frame)
    nf = grid.shape[0]
    for f in range(max(frames) + 1 if frames else nf):
        for r, v in nonenv_changes.get(f, []):
            out.append((f, r, v))
        for r, v in env_by_frame.get(f, []):
            out.append((f, r, v))
    return out


def decode(tokens, extend: bool = False) -> list[tuple[int, int, int]]:
    """Flat inline-event atom ids -> ordered ``(frame, reg, val)`` writes (settled
    non-env changes per frame interleaved with the ordered env writes). The declared
    frame span is the last event's reach; ``extend`` is accepted for interface parity
    (the stream already carries its full span)."""
    del extend
    events = inline.ids_to_events(list(tokens))
    if not events:
        return []
    grid, ew = inline.decode_events(events, _span(events))
    return _interleave(grid, ew)


def roundtrip_ok(df) -> bool:
    """One-call smoke test: ``decode(encode(ow))`` reproduces ``canonical_writes``."""
    from .oracle import ordered_writes

    ow = ordered_writes(df)
    return decode(encode(ow)) == canonical_writes(ow)


def _skip_dt(tokens, i: int) -> int:
    """Index past a DT varint, or ``len + 1`` (a truncation sentinel) if the stream
    ends mid-varint."""
    n = len(tokens)
    lo = inline.DIGIT_BASE
    hi = inline.DIGIT_BASE + 16
    while i < n and lo <= tokens[i] < hi:
        i += 1
    return i + 1 if i < n else n + 1


def _skip_u(tokens, i: int) -> int:
    """Index past an unsigned varint, or ``len + 1`` if the stream ends mid-varint."""
    n = len(tokens)
    while i < n:
        terminal = tokens[i] >= inline.DIGIT_BASE + 16
        i += 1
        if terminal:
            return i
    return n + 1


def _decode_u(tokens, i: int) -> int:
    val = shift = 0
    n = len(tokens)
    while i < n:
        a = tokens[i] - inline.DIGIT_BASE
        i += 1
        if a < 16:
            val |= a << shift
            shift += 4
        else:
            val |= (a - 16) << shift
            break
    return val


def _skip_params(tokens, i: int) -> int:
    n = len(tokens)
    if not 0 <= i - 1 < n:
        return n + 1
    op = tokens[i - 1]
    if op in (inline.NOTE_OP, inline.LOAD_OP):
        return _skip_u(tokens, i)
    p = _decode_u(tokens, i)
    i = _skip_u(tokens, i)
    i = _skip_u(tokens, i)
    for _ in range(p):
        if i > n:
            return n + 1
        i = _skip_u(tokens, i)
    return i


def _event_end(tokens, start: int) -> int:
    n = len(tokens)
    i = _skip_dt(tokens, start)
    if i > n:
        return i
    if i >= n:
        return n + 1
    sel = tokens[i] - inline.LANE_BASE
    i += 1
    if 0 <= sel < inline.NUM_NONENV:
        if i >= n:
            return n + 1
        i += 1
        return _skip_params(tokens, i)
    return _skip_u(tokens, i)


def unit_starts(tokens) -> list[int]:
    """Grammar-unit start indices: each complete event ``[DT][SELECTOR][...]`` is one
    unit, so a start is every event head; a truncated trailing event is not a unit."""
    starts: list[int] = []
    i = 0
    n = len(tokens)
    while i < n:
        end = _event_end(tokens, i)
        if end > n or end <= i:
            break
        starts.append(i)
        i = end
    return starts


def strip_keyframes(tokens) -> list[int]:
    """No keyframe segments in the inline grammar -- identity (interface parity)."""
    return list(tokens)


def chunk_keyframe(atoms, upto: int) -> list[int]:
    """No keyframe conditioning in the inline grammar; chunks are plain prefixes."""
    del atoms, upto
    return []


def trim_to_decodable(tokens, min_keep: int = 0):
    """Trim a trailing partial event so the remaining prefix decodes cleanly."""
    del min_keep
    toks = list(tokens)
    starts = unit_starts(toks)
    if not starts:
        return [], []
    last = starts[-1]
    end = _event_end(toks, last)
    keep = toks[:end] if end <= len(toks) else toks[:last]
    return keep, decode(keep)


def decode_windowed(tokens, prior=None) -> list[tuple[int, int, int]]:
    """Decode a window of inline events (no prior-state seeding in this grammar)."""
    del prior
    keep, _ = trim_to_decodable(list(tokens))
    return decode(keep)


__all__ = [
    "EVENT_FORMAT_VERSION",
    "VOCAB_SIZE",
    "OrderedWrites",
    "canonical_writes",
    "chunk_keyframe",
    "decode",
    "decode_windowed",
    "encode",
    "is_content_atom",
    "roundtrip_ok",
    "single_speed",
    "strip_keyframes",
    "trim_to_decodable",
    "unit_starts",
]
