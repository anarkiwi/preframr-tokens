"""SID register map + gesture-shape enum for the event codec: the per-voice freq/PW/CTRL/AD/SR register
ids, the global-register set, and the ``Shape`` gesture basis (HOLD/POLY/PERIOD). Every field is a
complete value over a small fixed alphabet -- no ids, no DEF/REF, no escape -- so BPE over the serialized
tokens is the only dictionary. Imported by ``stream`` (register helpers) and ``gestures`` (Shape).
"""

from __future__ import annotations

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


class Shape(IntEnum):
    HOLD = 0
    POLY = 1
    PERIOD = 2
