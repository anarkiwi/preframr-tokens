"""Per-step grammar-validity mask over the inline-event alphabet for sampling-time logit
guarding: :class:`EventStreamState` is the flat-stream grammar as an incremental machine
(``valid_mask`` -> legal next atoms, ``push`` advances, ``at_group_boundary`` flags a
completed unit). A unit is a SEQREF copy, voice selector, LEAD seed, or DT-prefixed lane
gesture / RAW / instrument REF / DEF. Torch-free; mirrors the decoder's field reads."""

from __future__ import annotations

import numpy as np

from . import inline

VOCAB_SIZE = inline.VOCAB_SIZE
_HI = inline.DIGIT_BASE + 16
_ANY_DIGIT = list(range(inline.DIGIT_BASE, inline.DIGIT_BASE + 32))
_NONENV_LANES = list(range(inline.LANE_BASE, inline.LANE_BASE + inline.NUM_NONENV))
_VOICES = list(range(inline.VOICE_BASE, inline.VOICE_BASE + inline.NUM_VOICES))
_OPS = [inline.NOTE_OP, inline.LOAD_OP, inline.MOD_OP, inline.RUN_OP]
_START_SELECTORS = _NONENV_LANES + [inline.RAW_ITEM, inline.REF_ITEM, inline.DEF_ITEM]

_TRIPLE_U = ("rep", ("u", "u", "u"))
_TRIPLE_S = ("rep", ("s", "u", "u"))


class EventStreamState:
    """Incremental grammar state for the inline-event alphabet (see module docstring)."""

    def __init__(self):
        self.phase = "START"
        self._gb = False
        self._queue: list = []
        self._acc = 0
        self._shift = 0

    @property
    def at_group_boundary(self) -> bool:
        return self._gb

    def valid_mask(self) -> np.ndarray:
        m = np.zeros(VOCAB_SIZE, dtype=bool)
        if self.phase == "START":
            for d in _ANY_DIGIT:
                m[d] = True
            m[inline.SEQREF_OP] = True
            for v in _VOICES:
                m[v] = True
            m[inline.LEAD_ITEM] = True
        elif self.phase == "SELECTOR":
            for s in _START_SELECTORS:
                m[s] = True
        elif self.phase == "OP":
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
        if self.phase == "START":
            self._push_start(tok)
        elif self.phase == "DT":
            if tok >= _HI:
                self.phase = "SELECTOR"
        elif self.phase == "SELECTOR":
            self._push_selector(tok)
        elif self.phase == "OP":
            self._push_op(tok)
        else:
            self._push_field(tok)

    def _start(self, queue) -> None:
        """Begin a field run; a non-empty queue always needs at least one more atom (a
        plain field's varint, or a ``rep`` group's inline count)."""
        self._queue = list(queue)
        if not self._queue:
            self._complete()
        else:
            self.phase = "FIELD"
            self._acc = self._shift = 0

    def _push_start(self, tok: int) -> None:
        if tok == inline.SEQREF_OP:
            self._start(["u", "u"])
        elif tok in _VOICES:
            self._complete()
        elif tok == inline.LEAD_ITEM:
            self._start([_TRIPLE_U])
        else:
            self.phase = "SELECTOR" if tok >= _HI else "DT"

    def _push_selector(self, tok: int) -> None:
        if tok in _NONENV_LANES:
            self.phase = "OP"
        elif tok == inline.RAW_ITEM:
            self._start([_TRIPLE_U])
        elif tok == inline.REF_ITEM:
            self._start(["u", "u"])
        else:
            self._start(["u", _TRIPLE_U, _TRIPLE_S])

    def _push_op(self, tok: int) -> None:
        if tok in (inline.NOTE_OP, inline.LOAD_OP):
            self._start(["u"])
        else:
            self._start([("repn", ("s",))])

    def _push_field(self, tok: int) -> None:
        if tok < _HI:
            self._acc |= (tok - inline.DIGIT_BASE) << self._shift
            self._shift += 4
            return
        head = self._queue.pop(0)
        if isinstance(head, tuple):
            count = self._acc | ((tok - _HI) << self._shift)
            inner: list = []
            if head[0] == "repn":
                inner.append("u")
            for _ in range(count):
                inner.extend(head[1])
            self._queue[0:0] = inner
        self._acc = self._shift = 0
        if not self._queue:
            self._complete()

    def _complete(self) -> None:
        self.phase = "START"
        self._queue = []
        self._gb = True


def valid_first_atoms() -> np.ndarray:
    return EventStreamState().valid_mask()


__all__ = ["EventStreamState", "VOCAB_SIZE", "valid_first_atoms"]
