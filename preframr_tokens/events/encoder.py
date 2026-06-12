"""Encoder: ordered register-write stream -> event stream. Input is the fidelity
oracle :class:`~preframr_tokens.events.oracle.OrderedWrites` (not the settled grid); output is a
``list[Event]`` whose decode reproduces the oracle byte-for-byte in order. v0 here is the byte-exact
skeleton -- one ``WRITE`` event per source write -- the safety net the factored layers are guarded by, so
every refinement must keep the ordered-stream roundtrip green.
"""

from __future__ import annotations

from .oracle import OrderedWrites
from .schema import Event, Kind


def encode(ow: OrderedWrites) -> list[Event]:
    """Ordered write stream -> v0 event stream (one ``WRITE`` per write, source order preserved)."""
    events: list[Event] = []
    fr = ow.frame.tolist()
    rg = ow.reg.tolist()
    vl = ow.val.tolist()
    for f, r, v in zip(fr, rg, vl):
        events.append(
            Event(kind=Kind.WRITE, frame=int(f), fields={"reg": int(r), "val": int(v)})
        )
    return events
