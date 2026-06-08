"""Decoder: event stream -> ordered register-write stream (REDESIGN_optionB §7). Driver-replay: it expands
events to an ordered ``(frame, reg, value)`` list the encoder asserts equals the source oracle byte-for-
byte, including intra-frame order and same-register repeats. There is NO fall-back-to-literal guard --
every field is a complete encoding, so divergence is a bug and fails loudly. v0 mirrors the v0 encoder
(each ``WRITE`` -> one write); factored layers add expansions while this flat-list contract holds.
"""

from __future__ import annotations

from .schema import Event, Kind


def decode(events: list[Event]) -> list[tuple[int, int, int]]:
    """Event stream -> ordered ``(frame, reg, value)`` writes (the fidelity target, §7)."""
    out: list[tuple[int, int, int]] = []
    for ev in events:
        if ev.kind == Kind.WRITE:
            out.append((ev.frame, ev.fields["reg"], ev.fields["val"]))
        else:
            raise ValueError(f"decoder v0 cannot expand {ev.kind.name}")
    return out
