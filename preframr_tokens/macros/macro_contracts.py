"""First-principles register/frame contracts per macro pass + a static reasoner that surfaces latent
interaction mismatches at test time. A pass's decode effect on a shared resource (a register's value
timeline, or the frame timeline) can violate a later pass's unstated assumption -- the class of bug
where a codebook pass replays a freq a later delta-encoder bases on. Passes declare their effects here
so the reasoner can flag the mismatches by construction, and a stream check verifies the code honours.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from preframr_tokens.stfconstants import (
    DIFF_OP,
    FLIP_OP,
    MODE_VOL_REG,
    SWEEP_OP,
    VOICE_REG_SIZE,
    VOICES,
)

__all__ = [
    "RegClass",
    "Effect",
    "FrameEffect",
    "MacroContract",
    "CONTRACTS",
    "PIPELINE_ORDER",
    "Mismatch",
    "interaction_mismatches",
    "KNOWN_MISMATCHES",
    "RELATIVE_OPS",
    "REPLAY_OPS",
    "reg_class",
]


class RegClass(Enum):
    """The SID register families a macro can write, voice-relative (FREQ/PWM/CTRL/AD/SR repeat per
    voice) plus the global FILTER cutoff/resonance. The unit a register-value contract reasons over.
    """

    FREQ = "freq"
    PWM = "pwm"
    CTRL = "ctrl"
    AD = "ad"
    SR = "sr"
    FILTER = "filter"


class Effect(Enum):
    """How a decode op sets a register: ABSOLUTE = a state-independent value (SET/preset/onset);
    RELATIVE = a function of the register's prior decode value (DIFF/FLIP) -- needs a stable base;
    REPLAY = regenerated from a codebook/trajectory (INSTR_REF/SWEEP/...) -- opaque to a delta encoder.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    REPLAY = "replay"


class FrameEffect(Enum):
    """How a pass touches the frame timeline: PRESERVES = never changes frame count/positions;
    MUTATES = may add/remove/reposition frames (consolidation); ANCHORED_REPLAY = emits multi-frame
    replays whose writes are pinned to absolute frame positions and so need that timeline stable.
    """

    PRESERVES = "preserves"
    MUTATES = "mutates"
    ANCHORED_REPLAY = "anchored_replay"


@dataclass(frozen=True)
class MacroContract:
    """One pass's interaction contract. ``writes`` is the (reg-class, effect) set it introduces at
    decode; ``leaves_surrounding`` are reg-classes it writes WITHOUT consuming the neighbouring writes
    (so a later delta-encoder can straddle its replay); ``base_barriers`` are producer pass names this
    pass's RELATIVE writes treat as a base barrier; ``can_empty_frames`` if it can leave a frame
    content-free (consolidation may then drop it)."""

    name: str
    writes: frozenset
    leaves_surrounding: frozenset
    base_barriers: frozenset
    frame_effect: FrameEffect
    can_empty_frames: bool


CONTRACTS = {
    c.name: c
    for c in (
        MacroContract(
            "frame_consolidation",
            frozenset(),
            frozenset(),
            frozenset(),
            FrameEffect.MUTATES,
            False,
        ),
    )
}

PIPELINE_ORDER = ("frame_consolidation",)


@dataclass(frozen=True)
class Mismatch:
    """A surfaced latent interaction: ``kind`` (relative_base | frame_anchor), the earlier producer
    and later consumer pass names, and the register class involved (None for a frame-timeline one).
    """

    kind: str
    earlier: str
    later: str
    reg_class: object


def interaction_mismatches():
    """Reason over PIPELINE_ORDER + CONTRACTS and return every latent mismatch. R1 (relative_base): a
    producer REPLAYs reg R and leaves surrounding writes, a later pass RELATIVE-writes R, and that pass
    does not barrier the producer. R3 (frame_anchor): a frame-anchored producer that can empty frames
    precedes a frame-mutating pass. Pure over the declarations -- no tune."""
    out = []
    order = [n for n in PIPELINE_ORDER if n in CONTRACTS]
    for i, pn in enumerate(order):
        prod = CONTRACTS[pn]
        for qn in order[i + 1 :]:
            cons = CONTRACTS[qn]
            for rc, eff in prod.writes:
                if (
                    eff is Effect.REPLAY
                    and rc in prod.leaves_surrounding
                    and (rc, Effect.RELATIVE) in cons.writes
                    and prod.name not in cons.base_barriers
                ):
                    out.append(Mismatch("relative_base", prod.name, cons.name, rc))
            if (
                prod.frame_effect is FrameEffect.ANCHORED_REPLAY
                and prod.can_empty_frames
                and cons.frame_effect is FrameEffect.MUTATES
            ):
                out.append(Mismatch("frame_anchor", prod.name, cons.name, None))
    return out


KNOWN_MISMATCHES = frozenset()


RELATIVE_OPS = frozenset({int(DIFF_OP), int(FLIP_OP)})
REPLAY_OPS = frozenset(
    {
        int(SWEEP_OP),
    }
)


def reg_class(reg):
    """Map a SID register (0-24) to its RegClass, or None for markers/MODE_VOL. Voice regs repeat
    every VOICE_REG_SIZE; 21/22/23 are the global filter, MODE_VOL is excluded (no contract).
    """
    reg = int(reg)
    if 0 <= reg < VOICES * VOICE_REG_SIZE:
        off = reg % VOICE_REG_SIZE
        return {
            0: RegClass.FREQ,
            2: RegClass.PWM,
            4: RegClass.CTRL,
            5: RegClass.AD,
            6: RegClass.SR,
        }.get(off)
    if reg in (21, 22, 23):
        return RegClass.FILTER
    if reg == MODE_VOL_REG:
        return None
    return None
