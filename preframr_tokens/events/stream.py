"""Canonical inline-event stream codec (the white-box decompiler representation).
``encode(ow)`` settles the ordered writes to the per-frame ``(n_frames, 25)`` grid
then emits the inline event stream; ``decode`` rebuilds the grid and emits ordered
``(frame, reg, val)`` writes. Lossless: ``decode(encode(ow))`` re-settles to
``canonical_writes(ow)`` (intra-frame order and same-value rewrites canonicalize away).
"""

from __future__ import annotations

import numpy as np

from . import inline
from .oracle import NUM_REGS, OrderedWrites, settled_grid

EVENT_FORMAT_VERSION = 4
VOCAB_SIZE = inline.VOCAB_SIZE


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


def _grid_to_writes(grid: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    prev = np.zeros(NUM_REGS, dtype=np.int64)
    n = grid.shape[0]
    for f in range(n):
        row = grid[f]
        for r in range(NUM_REGS):
            if row[r] != prev[r]:
                out.append((f, r, int(row[r])))
        prev = row
    return out


def canonical_writes(ow: OrderedWrites) -> list[tuple[int, int, int]]:
    """The byte-exact target the codec reproduces: per frame, every register whose
    settled value changed from the previous frame, ascending register. Intra-frame
    write order and same-value rewrites are canonicalized away."""
    return _grid_to_writes(settled_grid(ow))


def encode(ow: OrderedWrites, verify: bool = True) -> list[int]:
    """Ordered writes -> flat inline-event atom ids. ``verify`` self-checks the
    byte-exact round trip against the settled grid and raises on a miss."""
    grid = settled_grid(ow)
    ids = inline.encode_grid(grid)
    if verify:
        rec = inline.decode_grid(ids, grid.shape[0])
        if not np.array_equal(rec, grid):
            raise ValueError("inline codec not byte-exact against settled grid")
    return ids


def _span(events) -> int:
    span = 1
    for sf, _lane, op in events:
        length = op[2] if op[0] in ("MOD", "RUN") else 1
        span = max(span, sf + length)
    return span


def decode(tokens, extend: bool = False) -> list[tuple[int, int, int]]:
    """Flat inline-event atom ids -> ordered ``(frame, reg, val)`` writes. The
    declared frame span is the last event's reach; ``extend`` is accepted for
    interface parity (the stream already carries its full span)."""
    del extend
    events = inline.ids_to_events(list(tokens))
    if not events:
        return []
    grid = inline.decode_events(events, _span(events))
    return _grid_to_writes(grid)


def roundtrip_ok(df) -> bool:
    """One-call smoke test: ``decode(encode(ow))`` re-settles to ``canonical_writes``."""
    from .oracle import ordered_writes

    ow = ordered_writes(df)
    return decode(encode(ow)) == canonical_writes(ow)


def _skip_dt(tokens, i: int) -> int:
    n = len(tokens)
    lo = inline.DIGIT_BASE
    hi = inline.DIGIT_BASE + 16
    while i < n and lo <= tokens[i] < hi:
        i += 1
    return i + 1 if i < n else i


def _skip_u(tokens, i: int) -> int:
    n = len(tokens)
    while i < n:
        terminal = tokens[i] >= inline.DIGIT_BASE + 16
        i += 1
        if terminal:
            break
    return i


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
    if not 0 <= i - 1 < len(tokens):
        return i
    op = tokens[i - 1]
    if op in (inline.NOTE_OP, inline.LOAD_OP):
        return _skip_u(tokens, i)
    p = _decode_u(tokens, i)
    i = _skip_u(tokens, i)
    i = _skip_u(tokens, i)
    for _ in range(p):
        i = _skip_u(tokens, i)
    return i


def _event_end(tokens, start: int) -> int:
    i = _skip_dt(tokens, start)
    i += 2
    return _skip_params(tokens, i)


def unit_starts(tokens) -> list[int]:
    """Grammar-unit start indices: each event ``[DT][LANE][OP][params]`` is one
    unit, so a start is every event head."""
    starts: list[int] = []
    i = 0
    n = len(tokens)
    while i < n:
        starts.append(i)
        i = _event_end(tokens, i)
        if i <= starts[-1]:
            break
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
