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
