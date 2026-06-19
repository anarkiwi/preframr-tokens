"""Canonical inline-event stream codec. ``encode(ow)`` settles the non-env lanes,
captures ctrl/AD/SR as a per-voice INSTRUMENT program, and runs an inline backward-only
sequencing LZ over the event stream; ``decode`` rebuilds ordered ``(frame, reg, val)``
writes. Lossless against ``corrected_writes(ow)``; the instrument + sequencing layers
live in :mod:`preframr_tokens.events.instrument` / :mod:`preframr_tokens.events.seqref`.
"""

from __future__ import annotations

import numpy as np

from . import inline, seqref
from .oracle import (
    ENV_REGS,
    NUM_REGS,
    OrderedWrites,
    corrected_writes,
    env_writes,
    settled_grid,
)

EVENT_FORMAT_VERSION = 6
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
    """Ordered writes -> flat inline-event atom ids (instrument program + sequencing
    refs). ``verify`` self-checks the byte-exact round trip against the corrected target
    (settled non-env grid + ordered env writes) and raises on a miss."""
    grid = settled_grid(ow)
    ew = env_writes(ow)
    ids = seqref.encode(grid, ew)
    if verify:
        if decode(ids) != corrected_writes(ow):
            raise ValueError("inline codec not byte-exact against corrected target")
    return ids


def _interleave(grid: np.ndarray, ew) -> list[tuple[int, int, int]]:
    """The corrected-target ordering of decoded ``(grid, env_writes)``: per frame the
    settled non-env changes (ascending register) then the env writes ordered stably by
    ``(voice, source index)`` (the per-voice fidelity axis -- see
    :func:`preframr_tokens.events.oracle.env_writes_by_voice`)."""
    from .oracle import env_writes_by_voice

    nonenv_changes: dict[int, list[tuple[int, int]]] = {}
    prev = np.zeros(NUM_REGS, dtype=np.int64)
    for f in range(grid.shape[0]):
        row = grid[f]
        for r in range(NUM_REGS):
            if r not in _ENV_SET and row[r] != prev[r]:
                nonenv_changes.setdefault(f, []).append((r, int(row[r])))
        prev = row
    env_by_frame = env_writes_by_voice(ew)
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
    non-env changes per frame interleaved with the per-voice ordered env writes). The
    frame span is the event stream's reach; ``extend`` is accepted for interface parity
    (the stream already carries its full span)."""
    del extend
    toks = list(tokens)
    if not toks:
        return []
    grid, ew = seqref.decode(toks)
    return _interleave(grid, ew)


def roundtrip_ok(df) -> bool:
    """One-call smoke test: ``decode(encode(ow))`` reproduces ``canonical_writes``."""
    from .oracle import ordered_writes

    ow = ordered_writes(df)
    return decode(encode(ow)) == canonical_writes(ow)


_DLO = inline.DIGIT_BASE
_DHI = inline.DIGIT_BASE + 16


def _skip_u(tokens, i: int) -> int:
    """Index past an unsigned varint, or ``len + 1`` if the stream ends mid-varint."""
    n = len(tokens)
    while i < n:
        terminal = tokens[i] >= _DHI
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


def _skip_lane_op(tokens, i: int) -> int:
    """Index past ``[OP][params]`` (the OP atom is at ``i``)."""
    n = len(tokens)
    if i >= n:
        return n + 1
    op = tokens[i]
    i += 1
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
    """The index past the complete grammar unit beginning at ``start`` (a literal event
    or a SEQREF copy), or ``> len`` if the unit is truncated. Mirrors the field reads of
    :func:`preframr_tokens.events.seqref.deserialize`."""
    n = len(tokens)
    i = start
    if i >= n:
        return n + 1
    head = tokens[i]
    if head == inline.SEQREF_OP:
        i = _skip_u(tokens, i + 1)
        return _skip_u(tokens, i)
    if inline.VOICE_BASE <= head < inline.VOICE_BASE + inline.NUM_VOICES:
        return i + 1
    if head == inline.LEAD_ITEM:
        count = _decode_u(tokens, i + 1)
        i = _skip_u(tokens, i + 1)
        for _ in range(count):
            for _field in range(3):
                if i > n:
                    return n + 1
                i = _skip_u(tokens, i)
        return i
    i = _skip_u(tokens, i)
    if i > n or i >= n:
        return n + 1
    sel = tokens[i]
    if inline.LANE_BASE <= sel < inline.LANE_BASE + inline.NUM_NONENV:
        return _skip_lane_op(tokens, i + 1)
    if sel == inline.RAW_ITEM:
        count = _decode_u(tokens, i + 1)
        i = _skip_u(tokens, i + 1)
        for _ in range(count):
            for _field in range(3):
                if i > n:
                    return n + 1
                i = _skip_u(tokens, i)
        return i
    if sel == inline.REF_ITEM:
        i = _skip_u(tokens, i + 1)
        return _skip_u(tokens, i)
    if sel == inline.DEF_ITEM:
        i = _skip_u(tokens, i + 1)
        hn = _decode_u(tokens, i)
        i = _skip_u(tokens, i)
        for _ in range(hn):
            for _field in range(3):
                if i > n:
                    return n + 1
                i = _skip_u(tokens, i)
        if i > n:
            return n + 1
        tn = _decode_u(tokens, i)
        i = _skip_u(tokens, i)
        for _ in range(tn):
            for _field in range(3):
                if i > n:
                    return n + 1
                i = _skip_u(tokens, i)
        return i
    return n + 1


def unit_starts(tokens) -> list[int]:
    """Grammar-unit start indices: each complete event (a lane gesture, an instrument
    item, a voice selector) or a SEQREF copy is one unit; a truncated trailing unit is
    not a start."""
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
