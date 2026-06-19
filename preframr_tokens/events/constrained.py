"""Per-step grammar-validity mask over the inline-event alphabet for sampling-time
logit guarding: :class:`EventStreamState` is the flat-stream grammar as an
incremental token-at-a-time machine. ``valid_mask`` returns the legal next atoms,
``push`` advances (raising on an invalid atom), ``at_group_boundary`` flags a
completed event. Each event is ``[DT][LANE][OP][params]``; pure numpy, torch-free."""

from __future__ import annotations

import numpy as np

from . import inline

VOCAB_SIZE = inline.VOCAB_SIZE
_LO = inline.DIGIT_BASE
_HI = inline.DIGIT_BASE + 16
_ANY_DIGIT = list(range(inline.DIGIT_BASE, inline.DIGIT_BASE + 32))
_LANES = list(range(inline.LANE_BASE, inline.LANE_BASE + inline.NUM_LANES))
_OPS = [inline.NOTE_OP, inline.LOAD_OP, inline.MOD_OP, inline.RUN_OP]
_FIRST_ENV = inline.LANE_BASE + inline.NUM_NONENV


class EventStreamState:
    """Incremental grammar state for the inline-event alphabet. A fresh instance
    expects ``DT`` digits, then a LANE atom, an OP atom, and the op's params, then
    repeats. Mirrors the inline decoder's field reads exactly."""

    def __init__(self):
        self.phase = "DT"
        self.op = None
        self._sel = -1
        self._p = 0
        self._shift = 0
        self._deltas_left = 0
        self._gb = False

    @property
    def at_group_boundary(self) -> bool:
        return self._gb

    def valid_mask(self) -> np.ndarray:
        m = np.zeros(VOCAB_SIZE, dtype=bool)
        phase = self.phase
        if phase == "DT":
            for d in _ANY_DIGIT:
                m[d] = True
        elif phase == "LANE":
            for lane in _LANES:
                m[lane] = True
        elif phase == "OP":
            for op in _OPS:
                m[op] = True
        else:
            for d in _ANY_DIGIT:
                m[d] = True
        return m

    def push(self, tok: int) -> None:
        if not 0 <= tok < VOCAB_SIZE or not self.valid_mask()[tok]:
            raise ValueError(f"invalid atom {tok} for current grammar state")
        self._gb = False
        phase = self.phase
        if phase == "DT":
            if tok >= _HI:
                self.phase = "LANE"
        elif phase == "LANE":
            self._sel = tok - inline.LANE_BASE
            if self._sel < inline.NUM_NONENV:
                self.phase = "OP"
            else:
                self._p = 0
                self._shift = 0
                self.phase = "VAL"
        elif phase == "OP":
            self.op = tok
            self._begin_params(tok)
        else:
            self._feed_param(tok)

    def _begin_params(self, op: int) -> None:
        self._p = 0
        self._shift = 0
        self._deltas_left = 0
        self.phase = "VAL" if op in (inline.NOTE_OP, inline.LOAD_OP) else "P"

    def _feed_param(self, tok: int) -> None:
        terminal = tok >= _HI
        if self.phase == "VAL":
            if terminal:
                self._complete()
        elif self.phase == "P":
            digit = tok - (_HI if terminal else _LO)
            self._p |= digit << self._shift
            self._shift += 4
            if terminal:
                self.phase = "N"
        elif self.phase == "N":
            if terminal:
                self._deltas_left = self._p
                self.phase = "PVAL"
                if self._deltas_left == 0:
                    self._complete()
        else:
            if terminal:
                self._deltas_left -= 1
                if self._deltas_left <= 0:
                    self._complete()

    def _complete(self) -> None:
        self.phase = "DT"
        self.op = None
        self._gb = True


def valid_first_atoms() -> np.ndarray:
    return EventStreamState().valid_mask()


__all__ = ["EventStreamState", "VOCAB_SIZE", "valid_first_atoms"]
