"""Self-contained gesture cover + replay over the MDL primitive basis (REDESIGN_optionB §8.4-8.5): a
per-frame value series is covered losslessly by HOLD/POLY/PERIOD gestures via ``mdl_core.mdl_parse`` (the
kept MDL engine). Each gesture carries exactly the complete-value fields the decoder needs to replay it --
no codebook id, no per-tune bank. This module owns the replay arithmetic so the event model does not depend
on the retired ``codebook``/``mdl_gesture_pass`` machinery (§7.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preframr_tokens.macros import mdl_core
from .schema import Shape


@dataclass(frozen=True)
class Gesture:
    """One gesture run over ``[start, start+length)``. HOLD: ``params == (value,)``. POLY:
    ``params == (v0, d1, .., dN)``, the full initial forward-difference table (degree ``N``). PERIOD:
    ``params == (anchor, *cell)``, the start value then the ``period`` looped deltas."""

    shape: Shape
    start: int
    length: int
    params: tuple[int, ...]


def cover(series, wrap: bool = False) -> list[Gesture]:
    """Optimal lossless HOLD/POLY/PERIOD cover of ``series`` (``wrap`` for 16-bit register channels)."""
    s = np.asarray(series, dtype=np.int64)
    out: list[Gesture] = []
    for kind, i, j, param in mdl_core.mdl_parse(s, wrap=wrap):
        length = j - i
        if kind == "H":
            out.append(Gesture(Shape.HOLD, i, length, (int(param),)))
        elif kind == "D":
            N, _dN = param
            dt = mdl_core.difftable(s, i, N, wrap)
            out.append(Gesture(Shape.POLY, i, length, tuple(int(x) for x in dt)))
        elif kind == "P":
            cell = tuple(int(x) for x in param)
            out.append(Gesture(Shape.PERIOD, i, length, (int(s[i]), *cell)))
        else:
            raise ValueError(f"unknown mdl kind {kind!r}")
    return out


def replay_one(g: Gesture, wrap: bool = False) -> list[int]:
    """Replay a single gesture to its ``length`` values (inverse of the cover split)."""
    mask = 0xFFFF if wrap else None
    if g.shape == Shape.HOLD:
        return [g.params[0]] * g.length
    if g.shape == Shape.POLY:
        dt = [int(x) for x in g.params]
        deg = len(dt) - 1
        out = []
        for _ in range(g.length):
            out.append(dt[0])
            for k in range(deg):
                dt[k] += dt[k + 1]
            if mask is not None:
                dt[0] &= mask
        return out
    if g.shape == Shape.PERIOD:
        anchor = g.params[0]
        cell = g.params[1:]
        period = len(cell)
        cur = anchor
        out = []
        for k in range(g.length):
            if k:
                cur += cell[(k - 1) % period]
                if mask is not None:
                    cur &= mask
            out.append(cur)
        return out
    raise ValueError(f"unknown shape {g.shape!r}")


def replay(gestures: list[Gesture], n: int, wrap: bool = False) -> np.ndarray:
    """Replay a full cover back to its ``n``-frame series."""
    out = np.zeros(n, dtype=np.int64)
    for g in gestures:
        vals = replay_one(g, wrap)
        out[g.start : g.start + g.length] = vals
    return out
