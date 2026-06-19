"""The fidelity oracle: the exact ordered register-write stream -- the raw
dump rows ``(clock, irq, chipno, reg, val)`` filtered to ``chipno==0`` and sorted by ``clock``, NOT the
settled end-of-frame ``register_state``. That order is audibly significant (hard-restart, gate/ADSR order,
multi-speed sub-frame writes), so the decoder reproduces it write-for-write. A frame is the index into the
sorted unique ``irq`` values; multi-speed tunes write a reg more than once per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_REG = 24
NUM_REGS = 25

ENV_REGS: tuple[int, ...] = (4, 5, 6, 11, 12, 13, 18, 19, 20)
_ENV_SET = frozenset(ENV_REGS)


@dataclass(frozen=True)
class OrderedWrites:
    """The byte-exact ground truth for one tune (chipno 0): ``frame``/``reg``/``val`` are parallel
    ``int64`` arrays, one entry per register write, in exact source clock order. ``frame[k]`` is the dense
    frame index of write ``k``; within a frame writes keep source order (including same-reg repeats).
    ``irq`` is the sorted unique raw irq per frame index (decoder never needs it, kept for provenance).
    """

    frame: np.ndarray
    reg: np.ndarray
    val: np.ndarray
    n_frames: int
    irq: np.ndarray

    def __len__(self) -> int:
        return int(self.frame.shape[0])

    def triples(self) -> list[tuple[int, int, int]]:
        """The ordered ``(frame, reg, val)`` list -- the literal fidelity target."""
        return list(
            zip(
                self.frame.tolist(),
                self.reg.tolist(),
                self.val.tolist(),
            )
        )

    def by_frame(self) -> list[list[tuple[int, int]]]:
        """Per-frame ordered ``[(reg, val), ...]`` writes (length ``n_frames``); a frame with no writes is
        an empty list. This is the per-frame view the encoder/decoder reconcile against.
        """
        out: list[list[tuple[int, int]]] = [[] for _ in range(self.n_frames)]
        for f, r, v in zip(self.frame.tolist(), self.reg.tolist(), self.val.tolist()):
            out[f].append((r, v))
        return out


def ordered_writes(df: pd.DataFrame) -> OrderedWrites:
    """Build the :class:`OrderedWrites` oracle from a raw dump DataFrame: filter to ``chipno==0`` (v1 is
    single-SID), sort by ``clock`` (stable, so equal-clock rows keep source order), drop regs > 24,
    map each ``irq`` to a dense frame index, and mask values to a byte. The result is exactly the ordered
    write stream the decoder must reproduce."""
    if "chipno" in df.columns:
        df = df[df["chipno"] == 0]
    df = df[df["reg"] <= MAX_REG]
    if len(df) == 0:
        return OrderedWrites(
            frame=np.empty(0, dtype=np.int64),
            reg=np.empty(0, dtype=np.int64),
            val=np.empty(0, dtype=np.int64),
            n_frames=0,
            irq=np.empty(0, dtype=np.int64),
        )
    df = df.sort_values("clock", kind="stable")
    irq_raw = df["irq"].to_numpy(dtype=np.int64)
    uniq = np.unique(irq_raw)
    frame = np.searchsorted(uniq, irq_raw)
    reg = df["reg"].to_numpy(dtype=np.int64)
    val = df["val"].to_numpy(dtype=np.int64) & 0xFF
    return OrderedWrites(
        frame=frame.astype(np.int64),
        reg=reg,
        val=val,
        n_frames=int(uniq.shape[0]),
        irq=uniq.astype(np.int64),
    )


def writes_to_ordered(writes: list[tuple[int, int, int]]) -> OrderedWrites:
    """Build an :class:`OrderedWrites` directly from ordered ``(frame, reg, val)`` writes, keeping the
    frame indices AS GIVEN (``n_frames = max_frame + 1``). Unlike going through ``ordered_writes`` on a
    dump df -- which re-densifies via ``unique(irq)`` and so collapses empty/rest frames -- this preserves
    intermediate empty frames, the leading-rest off-by-one a re-canonicalised window must keep.
    """
    if not writes:
        return OrderedWrites(
            frame=np.empty(0, dtype=np.int64),
            reg=np.empty(0, dtype=np.int64),
            val=np.empty(0, dtype=np.int64),
            n_frames=0,
            irq=np.empty(0, dtype=np.int64),
        )
    frame = np.array([f for f, _, _ in writes], dtype=np.int64)
    reg = np.array([r for _, r, _ in writes], dtype=np.int64)
    val = np.array([v for _, _, v in writes], dtype=np.int64) & 0xFF
    n = int(frame.max()) + 1
    return OrderedWrites(
        frame=frame,
        reg=reg,
        val=val,
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def settled_grid(ow: OrderedWrites) -> np.ndarray:
    """Per-frame settled register state ``(n_frames, 25)`` (last write wins within a frame, forward-filled
    across frames from an all-zero start). This is the *secondary* musical view the gesture/note parse
    reads; it is NOT the fidelity target (intermediate and repeated writes are invisible here). Decoders
    are validated against :meth:`OrderedWrites.triples`, never against this grid.
    """
    grid = np.zeros((ow.n_frames, NUM_REGS), dtype=np.int64)
    cur = np.zeros(NUM_REGS, dtype=np.int64)
    fr = ow.frame.tolist()
    rg = ow.reg.tolist()
    vl = ow.val.tolist()
    k = 0
    n = len(fr)
    for f in range(ow.n_frames):
        while k < n and fr[k] == f:
            cur[rg[k]] = vl[k]
            k += 1
        grid[f] = cur
    return grid


def env_writes(ow: OrderedWrites) -> list[tuple[int, int, int]]:
    """The ORDERED ctrl/AD/SR write stream (regs in :data:`ENV_REGS`) in source clock
    order, de-duped consecutive same-reg-same-val no-ops only. This preserves the
    audibly load-bearing envelope/hard-restart write order that settling would erase.
    """
    out: list[tuple[int, int, int]] = []
    last: dict[int, int] = {}
    for f, r, v in zip(ow.frame.tolist(), ow.reg.tolist(), ow.val.tolist()):
        if r in _ENV_SET and last.get(r) != v:
            out.append((int(f), int(r), int(v)))
            last[r] = v
    return out


def corrected_writes(ow: OrderedWrites) -> list[tuple[int, int, int]]:
    """The audio-faithful fidelity target = the settled non-env grid writes (one
    ascending-register write per frame where a non-env reg's settled value changed)
    interleaved with the ORDERED env writes (kept in source order). This is exactly
    what the codec reproduces; only intra-frame non-env intermediates and env
    same-value rewrites (both inaudible) are dropped."""
    grid = settled_grid(ow)
    nonenv_changes: dict[int, list[tuple[int, int]]] = {}
    prev = np.zeros(NUM_REGS, dtype=np.int64)
    for f in range(grid.shape[0]):
        row = grid[f]
        for r in range(NUM_REGS):
            if r not in _ENV_SET and row[r] != prev[r]:
                nonenv_changes.setdefault(f, []).append((r, int(row[r])))
        prev = row
    env = env_writes(ow)
    env_by_frame: dict[int, list[tuple[int, int]]] = {}
    for f, r, v in env:
        env_by_frame.setdefault(f, []).append((r, v))
    out: list[tuple[int, int, int]] = []
    nf = grid.shape[0]
    frames = set(nonenv_changes) | set(env_by_frame)
    for f in range(max(frames) + 1 if frames else nf):
        for r, v in nonenv_changes.get(f, []):
            out.append((f, r, v))
        for r, v in env_by_frame.get(f, []):
            out.append((f, r, v))
    return out
