"""Event schema + SID register map (REDESIGN_optionB §3, §3.5): the model reads/writes a typed event
stream (a ``list[Event]``) distinct from the register-write df. Every field is a complete value over a
small fixed alphabet -- no ids, no DEF/REF, no escape -- so BPE over the serialized tokens is the only
"dictionary" (§2.3). Defines the kinds, the SID register layout, and the lane classification the
encoder/decoder share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

NUM_REGS = 25
MAX_REG = 24
VOICES = (0, 1, 2)

FREQ_LO = {0, 7, 14}
FREQ_HI = {1, 8, 15}
PW_LO = {2, 9, 16}
PW_HI = {3, 10, 17}
CTRL = {4, 11, 18}
AD = {5, 12, 19}
SR = {6, 13, 20}

CUTOFF_LO = 21
CUTOFF_HI = 22
RES_ROUTE = 23
MODE_VOL = 24
GLOBAL_REGS = {CUTOFF_LO, CUTOFF_HI, RES_ROUTE, MODE_VOL}

VOICE_OF = {}
for _v in VOICES:
    for _off in range(7):
        VOICE_OF[_off + 7 * _v] = _v

GLOBAL = 3


def freq_regs(v: int) -> tuple[int, int]:
    """``(lo, hi)`` freq register ids for voice ``v``."""
    return 7 * v, 7 * v + 1


def pw_regs(v: int) -> tuple[int, int]:
    """``(lo, hi)`` pulse-width register ids for voice ``v``."""
    return 7 * v + 2, 7 * v + 3


def ctrl_reg(v: int) -> int:
    return 7 * v + 4


def ad_reg(v: int) -> int:
    return 7 * v + 5


def sr_reg(v: int) -> int:
    return 7 * v + 6


def gate_on(ctrl_val: int) -> bool:
    """Gate bit (bit0) of a CTRL byte."""
    return bool(ctrl_val & 0x01)


class Kind(IntEnum):
    """The event-kind alphabet (one token, drives the grammar §7.1)."""

    NOTE_ON = 0
    NOTE_STEP = 1
    MOD_FREQ = 2
    MOD_PW = 3
    MOD_CTRL = 4
    MOD_CUTOFF = 5
    FILTER_CTL = 6
    MOD_VOL = 7
    TICK = 8
    TUNING = 9
    NOTE_TABLE = 10
    ORDER = 11
    WRITE = 12


class Shape(IntEnum):
    HOLD = 0
    POLY = 1
    PERIOD = 2


@dataclass
class Event:
    """One musical event: a ``Kind``, a ``VOICE`` tag (0..2 or :data:`GLOBAL`), the frame it occurs on
    (absolute; ``DT`` is derived at serialization), and kind-specific fields. Fields are plain Python
    ints / lists of ints -- every one a complete value over a small alphabet (§3). ``raw`` carries the
    v0 single-write payload ``(reg, value)`` until the factored layers subsume it.
    """

    kind: Kind
    frame: int
    voice: int = GLOBAL
    fields: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Event({self.kind.name}, f={self.frame}, v={self.voice}, {self.fields})"
