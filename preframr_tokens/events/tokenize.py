"""Serialize the event stream to/from a flat token-id stream: tokens are small ints
in disjoint per-field-family ranges, so the stream is self-delimiting and BPE-able, the only
"dictionary". Every field is a complete value over the escape-free zig-zag varint of
:mod:`preframr_tokens.events.varint` -- no ids, no escape. v0 serializes ``WRITE`` events as
``DT || REG || DELTA`` (DELTA = signed zig-zag change from the held value); stream order == write order.
"""

from __future__ import annotations

from . import varint
from .schema import Event, Kind, NUM_REGS

_VARINT_SPAN = 32

DT_BASE = 0
REG_BASE = DT_BASE + _VARINT_SPAN
DELTA_BASE = REG_BASE + NUM_REGS
VOCAB_SIZE = DELTA_BASE + _VARINT_SPAN


def _emit_varint_unsigned(out: list[int], base: int, value: int) -> None:
    for d in varint.encode_unsigned(value):
        out.append(base + d)


def _emit_varint_signed(out: list[int], base: int, value: int) -> None:
    for d in varint.encode_signed(value):
        out.append(base + d)


def _read_varint(tokens: list[int], pos: int, base: int) -> tuple[int, int]:
    """Read one varint whose digits live in ``[base, base+32)``; return ``(unsigned_value, next_pos)``."""
    shifted = []
    while True:
        if pos >= len(tokens):
            raise ValueError("truncated varint at end of token stream")
        tok = tokens[pos]
        if not (base <= tok < base + _VARINT_SPAN):
            raise ValueError(
                f"expected varint digit in [{base},{base+32}) at {pos}, got {tok}"
            )
        shifted.append(tok - base)
        pos += 1
        if not (shifted[-1] & varint.CONT):
            break
    value, _ = varint.decode_unsigned(shifted, 0)
    return value, pos


def to_tokens(events: list[Event]) -> list[int]:
    """Event stream -> flat token-id list (v0: WRITE events only)."""
    out: list[int] = []
    prev_frame = 0
    held = [0] * NUM_REGS
    for ev in events:
        if ev.kind != Kind.WRITE:
            raise ValueError(f"tokenize v0 cannot serialize {ev.kind.name}")
        reg = ev.fields["reg"]
        val = ev.fields["val"]
        _emit_varint_unsigned(out, DT_BASE, ev.frame - prev_frame)
        out.append(REG_BASE + reg)
        _emit_varint_signed(out, DELTA_BASE, val - held[reg])
        held[reg] = val
        prev_frame = ev.frame
    return out


def from_tokens(tokens: list[int]) -> list[Event]:
    """Flat token-id list -> event stream (inverse of :func:`to_tokens`)."""
    events: list[Event] = []
    pos = 0
    frame = 0
    held = [0] * NUM_REGS
    n = len(tokens)
    while pos < n:
        dt, pos = _read_varint(tokens, pos, DT_BASE)
        frame += dt
        if pos >= n or not (REG_BASE <= tokens[pos] < REG_BASE + NUM_REGS):
            raise ValueError(f"expected REG token at {pos}")
        reg = tokens[pos] - REG_BASE
        pos += 1
        delta_u, pos = _read_varint(tokens, pos, DELTA_BASE)
        val = held[reg] + varint.unzigzag(delta_u)
        held[reg] = val
        events.append(
            Event(kind=Kind.WRITE, frame=frame, fields={"reg": reg, "val": val})
        )
    return events
