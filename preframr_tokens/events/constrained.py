"""Per-step grammar-validity mask over the 127-atom event alphabet for generation-time logit guarding:
:class:`EventStreamState` is :func:`stream._Decoder.parse` re-expressed as an incremental token-at-a-time
machine -- ``valid_mask`` returns the booleans of structurally-legal next atoms, ``push`` advances the
state (raising on an invalid atom), and ``at_group_boundary`` flags a completed frame group. Pure numpy,
torch-free; KEYFRAME is never valid (generation emits pure streams)."""

from __future__ import annotations

import numpy as np

from . import stream, varint
from .stream import (  # pylint: disable=unused-import
    FD_RAMP,
    FD_STEP,
    FLD_AD,
    FLD_CTRL,
    FLD_NOTE_ON,
    FLD_SR,
    G_RAMP,
    G_STEP,
    GLOBAL_REGS,
    KEYFRAME,
    NI_RAMP,
    NI_STEP,
    NIB_ART,
    NIB_ENV,
    NIB_WAVE,
    NOTE_TABLE,
    PW_RAMP,
    PW_STEP,
    REG_BASE,
    SHAPE_PERIOD,
    SHAPE_POLY,
    TICK,
    TUNING,
    VAR_BASE,
    VOCAB_SIZE,
    VOICE_BASE,
    _FLAG_HRAD,
    _FLAG_HRSR,
    _FLAG_OAD,
    _FLAG_OSR,
    _GO_NONE,
    _GO_VALUE,
)

_CONT = varint.CONT
_NDIGITS = 32
_EVENT_KINDS = tuple(range(NI_STEP, G_RAMP + 1))
_VOICE_EVENT_KINDS = tuple(range(NI_STEP, FLD_SR + 1))
_GLOBAL_EVENT_KINDS = (G_STEP, G_RAMP)
_GLOBAL_VOICE = 3
_GREG_TOKS = tuple(REG_BASE + r for r in GLOBAL_REGS)


def _kinds_for(voice):
    return _GLOBAL_EVENT_KINDS if voice == _GLOBAL_VOICE else _VOICE_EVENT_KINDS


def _ctrl_fields():
    return [["N", NIB_WAVE], ["N", NIB_ART]]


def _env_fields():
    return [["N", NIB_ENV], ["N", NIB_ENV]]


class EventStreamState:
    """Incremental grammar state for the event alphabet. Drive it with :meth:`valid_mask` /
    :meth:`push`; a fresh instance expects the frame-count header, then per-voice preamble headers,
    then frame groups. Mirrors the stream decoder's field reads exactly."""

    def __init__(self):
        self.stack = [["V", "u", "framecount", 0]]
        self.phase = "PRE"
        self.sub = "AWAIT_VOICE"
        self.last_voice = -1
        self.cur_voice = -1
        self.tick_by_voice = {}
        self._ptick = 1
        self._gb = False

    @property
    def at_group_boundary(self) -> bool:
        return self._gb

    def valid_mask(self) -> np.ndarray:
        m = np.zeros(VOCAB_SIZE, dtype=bool)
        if self.stack:
            top = self.stack[-1]
            kind = top[0]
            if kind == "V":
                if top[1] == "m":
                    m[VAR_BASE] = m[VAR_BASE + 1] = m[VAR_BASE + 2] = True
                else:
                    m[VAR_BASE : VAR_BASE + _NDIGITS] = True
            elif kind == "N":
                m[top[1] : top[1] + 16] = True
            elif kind == "SH":
                m[SHAPE_POLY] = m[SHAPE_PERIOD] = True
            elif kind == "GR":
                for tok in _GREG_TOKS:
                    m[tok] = True
            else:
                m[TUNING] = m[NOTE_TABLE] = m[TICK] = True
            return m
        if self.phase == "PRE":
            for w in range(3):
                m[VOICE_BASE + w] = True
            m[VAR_BASE : VAR_BASE + _NDIGITS] = True
        elif self.sub == "AWAIT_VOICE":
            for w in range(self.last_voice + 1, 4):
                m[VOICE_BASE + w] = True
        elif self.sub == "AFTER_VOICE":
            for k in _kinds_for(self.cur_voice):
                m[k] = True
        else:
            for k in _kinds_for(self.cur_voice):
                m[k] = True
            for w in range(self.cur_voice + 1, 4):
                m[VOICE_BASE + w] = True
            m[VAR_BASE : VAR_BASE + _NDIGITS] = True
        return m

    def push(self, tok: int) -> None:
        if not (0 <= tok < VOCAB_SIZE) or not self.valid_mask()[tok]:
            raise ValueError(f"invalid atom {tok} for current grammar state")
        if self.stack:
            top = self.stack[-1]
            if top[0] == "V":
                self._feed_var(tok)
            else:
                node = self.stack.pop()
                if node[0] == "HK":
                    self._open_header(tok)
                self._after_pop()
        else:
            self._group_push(tok)

    def _push(self, fields):
        for d in reversed(fields):
            self.stack.append(d)

    def _feed_var(self, tok):
        top = self.stack[-1]
        d = tok - VAR_BASE
        top[3] = top[3] * 16 + (d & 0xF)
        if (d & _CONT) and top[1] != "m":
            return
        kind, action, acc = top[1], top[2], top[3]
        value = varint.unzigzag(acc) if kind == "s" else acc
        self.stack.pop()
        self._do_action(action, value)
        self._after_pop()

    def _do_action(self, action, value):
        if action in (None, "framecount"):
            return
        if action == "dt":
            self.phase = "BODY"
            self.sub = "AWAIT_VOICE"
            self.last_voice = -1
            self._gb = False
        elif action == "tick_unit":
            self._ptick = value
        elif action == "tick_offset":
            self.tick_by_voice[self.cur_voice] = (self._ptick, value)
        elif action == "ntcount":
            self._push([["V", "s", None, 0] for _ in range(2 * value)])
        elif action == "deg":
            for _ in range(value):
                self.stack.insert(0, ["V", "s", None, 0])
        elif action == "flags":
            self._push(self._note_on_flag_fields(value))
        elif action == "mode":
            if value != _GO_NONE:
                self._push(self._note_on_dur_fields(value))

    def _note_on_flag_fields(self, flags):
        fields = []
        if flags & _FLAG_OAD:
            fields += _env_fields()
        if flags & _FLAG_OSR:
            fields += _env_fields()
        if flags & (_FLAG_HRAD | _FLAG_HRSR):
            fields += [["V", "u", None, 0]]
            if flags & _FLAG_HRAD:
                fields += _env_fields()
            if flags & _FLAG_HRSR:
                fields += _env_fields()
        fields += [["V", "m", "mode", 0]]
        return fields

    def _note_on_dur_fields(self, mode):
        tick = self.tick_by_voice.get(self.cur_voice, (1, 0))[0]
        if tick > 1:
            fields = [["V", "u", None, 0], ["V", "s", None, 0]]
        else:
            fields = [["V", "u", None, 0]]
        if mode == _GO_VALUE:
            fields += _ctrl_fields()
        return fields

    def _open_header(self, tok):
        if tok == TUNING:
            self._push([["V", "u", None, 0]])
        elif tok == NOTE_TABLE:
            self._push([["V", "u", "ntcount", 0]])
        else:
            self._push([["V", "u", "tick_unit", 0], ["V", "s", "tick_offset", 0]])

    def _event_fields(self, kind):
        if kind in (NI_STEP, FD_STEP):
            return [["V", "s", None, 0]]
        if kind in (NI_RAMP, FD_RAMP):
            return [
                ["SH"],
                ["V", "u", None, 0],
                ["V", "u", "deg", 0],
                ["V", "s", None, 0],
            ]
        if kind == PW_STEP:
            return [["N", NIB_ENV], ["V", "u", None, 0]]
        if kind == PW_RAMP:
            return [
                ["SH"],
                ["V", "u", None, 0],
                ["V", "u", "deg", 0],
                ["N", NIB_ENV],
                ["V", "u", None, 0],
            ]
        if kind == G_STEP:
            return [["GR"], ["V", "u", None, 0]]
        if kind == G_RAMP:
            return [
                ["GR"],
                ["SH"],
                ["V", "u", None, 0],
                ["V", "u", "deg", 0],
                ["V", "u", None, 0],
            ]
        if kind == FLD_CTRL:
            return _ctrl_fields()
        if kind in (FLD_AD, FLD_SR):
            return _env_fields()
        return _ctrl_fields() + [["V", "u", "flags", 0]]

    def _group_push(self, tok):
        if self.phase == "PRE":
            if VOICE_BASE <= tok < VOICE_BASE + 4:
                self.cur_voice = tok - VOICE_BASE
                self.stack.append(["HK"])
            else:
                self.stack.append(["V", "u", "dt", 0])
                self._feed_var(tok)
            return
        if self.sub == "AWAIT_VOICE":
            self.cur_voice = self.last_voice = tok - VOICE_BASE
            self.sub = "AFTER_VOICE"
            return
        if tok in _EVENT_KINDS:
            self.sub = "CONSUMING"
            self._gb = False
            self._push(self._event_fields(tok))
        elif VOICE_BASE <= tok < VOICE_BASE + 4:
            self.cur_voice = self.last_voice = tok - VOICE_BASE
            self.sub = "AFTER_VOICE"
            self._gb = False
        else:
            self._gb = False
            self.stack.append(["V", "u", "dt", 0])
            self._feed_var(tok)

    def _after_pop(self):
        if not self.stack and self.phase == "BODY" and self.sub == "CONSUMING":
            self.sub = "IN_FRAME"
            self._gb = True


__all__ = ["EventStreamState"]
