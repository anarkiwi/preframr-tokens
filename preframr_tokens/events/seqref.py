"""The full event stream + the inline backward-reference (sequencing) layer. The events
are the settled-lane gestures then the per-voice instrument program
(:mod:`preframr_tokens.events.instrument`). ``lz_parse`` is a greedy longest-match
backward LZ over them, emitting backward-only INLINE ``LIT``/``REF`` items with the
distance CAPPED at :data:`MAX_REF_DISTANCE` (the model's reachable context)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import inline, instrument
from .oracle import ENV_REGS

MAX_REF_DISTANCE = 2048
MIN_REF_LEN = 3
MAX_REF_LEN = 4096

_ENV_SET = frozenset(ENV_REGS)


def _opkey(op):
    if op[0] in ("NOTE", "LOAD"):
        return (op[0], int(op[1]))
    return (op[0], tuple(int(d) for d in op[1]), int(op[2]))


def nonenv_events(grid: np.ndarray, skip_lanes=()):
    """The settled non-env lane gestures as ``("NE", dt, lane, op)`` events, sorted by
    ``(frame, lane)`` with ``dt`` the global inter-event frame delta. ``skip_lanes``
    omits lanes folded into the instrument (synced PW)."""
    raw = inline.nonenv_lane_events(grid, skip_lanes=skip_lanes)
    out = []
    prev = 0
    for f, lid, payload in raw:
        out.append(("NE", f - prev, lid, _opkey(payload[2])))
        prev = f
    return out


def env_events(ew, pw_extra=()):
    """The per-voice instrument program as its event tuples (delegates to
    :func:`preframr_tokens.events.instrument.env_events`)."""
    return instrument.env_events(ew, pw_extra=pw_extra)


def tune_events(grid: np.ndarray, ew, skip_lanes=(), pw_extra=()):
    """The full event stream: non-env lane gestures then the instrument program (with any
    synced-PW lanes folded into the instrument via ``skip_lanes``/``pw_extra``)."""
    return nonenv_events(grid, skip_lanes=skip_lanes) + env_events(
        ew, pw_extra=pw_extra
    )


def _emit_literal(out, ev):
    """Append one literal event's atom bytes (matches the no-REF serialization)."""
    kind = ev[0]
    if kind == "NE":
        _, dt, lane, op = ev
        inline.emit_u(out, dt)
        inline.emit_lane_op(out, lane, op)
    elif kind == "VSEL":
        out.append(inline.VOICE_BASE + ev[1])
    elif kind == "LEAD":
        out.append(inline.LEAD_ITEM)
        lead = ev[1]
        inline.emit_u(out, len(lead))
        prevf = 0
        for f, r, val in lead:
            inline.emit_u(out, f - prevf)
            prevf = f
            inline.emit_u(out, r)
            inline.emit_u(out, val)
    elif kind == "RAW":
        _, dt, sig = ev
        inline.emit_u(out, dt)
        out.append(inline.RAW_ITEM)
        inline.emit_u(out, len(sig))
        pf = 0
        for off, r, val in sig:
            inline.emit_u(out, off - pf)
            pf = off
            inline.emit_u(out, r)
            inline.emit_u(out, val)
    elif kind == "REF":
        _, dt, iid, durtok = ev
        inline.emit_u(out, dt)
        out.append(inline.REF_ITEM)
        inline.emit_u(out, iid)
        inline.emit_u(out, durtok)
    elif kind == "DEF":
        _, dt, durtok, head, tail = ev
        inline.emit_u(out, dt)
        out.append(inline.DEF_ITEM)
        inline.emit_u(out, durtok)
        inline.emit_u(out, len(head))
        for off, r, val in head:
            inline.emit_u(out, off)
            inline.emit_u(out, r)
            inline.emit_u(out, val)
        inline.emit_u(out, len(tail))
        for off, r, val in tail:
            inline.emit_s(out, off)
            inline.emit_u(out, r)
            inline.emit_u(out, val)
    else:
        raise ValueError(kind)


def serialize_events(events) -> list[int]:
    """Literal-only serialization of the event stream (no sequencing refs)."""
    out: list[int] = []
    for ev in events:
        _emit_literal(out, ev)
    return out


def lz_parse(events, window: int = MAX_REF_DISTANCE, min_len: int = MIN_REF_LEN):
    """Greedy longest-match backward LZ over the event list. Returns a list of
    ``("LIT", event)`` and ``("REF", distance, length)`` items; ``distance`` is the
    number of events back to the start of the matched span (>= 1) and is capped at
    ``window``. Hash-chained on a rolling ``min_len``-gram for speed."""
    n = len(events)
    out: list[tuple] = []
    table: dict[tuple, list[int]] = defaultdict(list)
    i = 0
    k = min_len
    while i < n:
        best_len = 0
        best_pos = -1
        if i + k <= n:
            key = tuple(events[i : i + k])
            cands = table.get(key)
            if cands:
                lo = 0
                if window < i:
                    import bisect

                    lo = bisect.bisect_left(cands, i - window)
                for pos in reversed(cands[lo:]):
                    length = k
                    cap = min(MAX_REF_LEN, n - i)
                    while length < cap and events[pos + length] == events[i + length]:
                        length += 1
                    if length > best_len:
                        best_len = length
                        best_pos = pos
                        if length >= cap:
                            break
        if best_len >= k:
            out.append(("REF", i - best_pos, best_len))
            end = i + best_len
            j = i
            while j + k <= end:
                table[tuple(events[j : j + k])].append(j)
                j += 1
            i = end
        else:
            out.append(("LIT", events[i]))
            if i + k <= n:
                table[tuple(events[i : i + k])].append(i)
            i += 1
    return out


def lz_decode(parsed):
    """Reconstruct the event list from an LZ parse. Inverse of :func:`lz_parse`."""
    events: list[tuple] = []
    for item in parsed:
        if item[0] == "LIT":
            events.append(item[1])
        else:
            _, dist, length = item
            start = len(events) - dist
            for k in range(length):
                events.append(events[start + k])
    return events


def serialize_parsed(parsed) -> list[int]:
    """Flatten an LZ parse to atom ids: literals as their instrument-codec bytes, a
    ``REF`` as ``SEQREF_OP + LEB(distance) + LEB(length)``."""
    out: list[int] = []
    for item in parsed:
        if item[0] == "LIT":
            _emit_literal(out, item[1])
        else:
            _, dist, length = item
            out.append(inline.SEQREF_OP)
            inline.emit_u(out, dist)
            inline.emit_u(out, length)
    return out


def _read_literal(ids, i):
    """Read one literal event starting at ``i`` (just after any SEQREF dispatch);
    returns ``(event, next_index)``."""
    sel = ids[i]
    if inline.VOICE_BASE <= sel < inline.VOICE_BASE + inline.NUM_VOICES:
        return ("VSEL", sel - inline.VOICE_BASE), i + 1
    if sel == inline.LEAD_ITEM:
        i += 1
        count, i = inline.read_u(ids, i)
        lead = []
        prevf = 0
        for _ in range(count):
            d, i = inline.read_u(ids, i)
            prevf += d
            r, i = inline.read_u(ids, i)
            val, i = inline.read_u(ids, i)
            lead.append((prevf, r, val))
        return ("LEAD", tuple(lead)), i
    dt, i = inline.read_u(ids, i)
    sel = ids[i]
    if inline.LANE_BASE <= sel < inline.LANE_BASE + inline.NUM_NONENV:
        i += 1
        op, i = inline.read_lane_op(ids, i)
        return ("NE", dt, sel - inline.LANE_BASE, _opkey(op)), i
    if sel == inline.RAW_ITEM:
        i += 1
        count, i = inline.read_u(ids, i)
        sig = []
        pf = 0
        for _ in range(count):
            d, i = inline.read_u(ids, i)
            pf += d
            r, i = inline.read_u(ids, i)
            val, i = inline.read_u(ids, i)
            sig.append((pf, r, val))
        return ("RAW", dt, tuple(sig)), i
    if sel == inline.REF_ITEM:
        i += 1
        iid, i = inline.read_u(ids, i)
        durtok, i = inline.read_u(ids, i)
        return ("REF", dt, iid, durtok), i
    if sel == inline.DEF_ITEM:
        i += 1
        durtok, i = inline.read_u(ids, i)
        hn, i = inline.read_u(ids, i)
        head = []
        for _ in range(hn):
            off, i = inline.read_u(ids, i)
            r, i = inline.read_u(ids, i)
            val, i = inline.read_u(ids, i)
            head.append((off, r, val))
        tn, i = inline.read_u(ids, i)
        tail = []
        for _ in range(tn):
            off, i = inline.read_s(ids, i)
            r, i = inline.read_u(ids, i)
            val, i = inline.read_u(ids, i)
            tail.append((off, r, val))
        return ("DEF", dt, durtok, tuple(head), tuple(tail)), i
    raise ValueError(f"unexpected selector {sel} at {i}")


def deserialize(ids):
    """Flat atom ids (literals + SEQREF copies) -> the event list. Inverse of
    :func:`serialize_parsed` followed by :func:`lz_decode`."""
    events: list[tuple] = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == inline.SEQREF_OP:
            dist, i = inline.read_u(ids, i + 1)
            length, i = inline.read_u(ids, i)
            start = len(events) - dist
            for k in range(length):
                events.append(events[start + k])
        else:
            ev, i = _read_literal(ids, i)
            events.append(ev)
    return events


def _span_from_events(events) -> int:
    """The frame span the non-env lane events declare (for grid replay)."""
    span = 0
    prev = 0
    for ev in events:
        if ev[0] == "NE":
            _, dt, _lane, op = ev
            prev += dt
            span = max(span, prev + inline.lane_op_length(op))
    return span


def _events_to_lane_raw(events):
    """Reconstruct the ``(frame, lane, ("L", lane, op))`` lane events from the
    ``("NE", dt, lane, op)`` stream (un-deltaing dt)."""
    raw = []
    prev = 0
    for ev in events:
        if ev[0] == "NE":
            _, dt, lane, op = ev
            prev += dt
            raw.append((prev, lane, ("L", lane, op)))
    return raw


def encode(grid: np.ndarray, ew) -> list[int]:
    """``{settled non-env grid + ordered env writes}`` -> flat atom ids (instrument
    program + sequencing refs). A voice whose pulse-width is SYNCED to the instrument has
    its PW folded into its instrument items and its pw lane skipped in the settled half;
    free-running PW stays in its own lane."""
    synced = instrument.synced_pw_voices(grid, ew)
    skip = instrument.pw_skip_lanes(synced)
    pw_extra = instrument.pw_writes(grid, synced)
    events = tune_events(grid, ew, skip_lanes=skip, pw_extra=pw_extra)
    return serialize_parsed(lz_parse(events))


def decode(ids):
    """Flat atom ids -> ``(settled grid, ordered env writes)``. Folded synced-PW writes
    decoded out of the instrument items are written back into the settled grid; their
    lanes were skipped in the settled half, so the grid would otherwise be 0 there."""
    events = deserialize(ids)
    n_frames = _span_from_events(events)
    ew, pw = instrument.decode_env(events)
    for seq in (ew, pw):
        if seq:
            n_frames = max(n_frames, max(f for f, _r, _v in seq) + 1)
    skip = sorted({3 + instrument._voice_of(r) for _f, r, _v in pw})
    lane_raw = _events_to_lane_raw(events)
    grid = inline.lane_events_to_grid(
        lane_raw, n_frames, env_writes=ew, skip_lanes=skip
    )
    _apply_pw_writes(grid, pw, n_frames)
    return grid, ew


def _apply_pw_writes(grid, pw, n_frames):
    """Forward-fill folded PW pseudo-writes into the settled grid's pw regs."""
    by_reg: dict[int, list[tuple[int, int]]] = {}
    for f, r, val in pw:
        by_reg.setdefault(r, []).append((f, val))
    for r, seq in by_reg.items():
        seq.sort()
        cur = 0
        k = 0
        for f in range(n_frames):
            while k < len(seq) and seq[k][0] == f:
                cur = seq[k][1]
                k += 1
            grid[f, r] = cur


def is_content_atom(tok: int) -> bool:
    """Content atoms = the varint payload digits the model predicts."""
    return inline.is_digit_atom(tok)
