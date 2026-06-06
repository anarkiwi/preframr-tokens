"""First-principles register/frame contracts per macro pass + a static reasoner that surfaces latent
interaction mismatches at test time. A pass's decode effect on a shared resource (a register's value
timeline, or the frame timeline) can violate a later pass's unstated assumption -- the class of bug
where StampPass replays a freq a later delta-encoder bases on. Passes declare their effects here so the
reasoner can flag the mismatches by construction, and a stream check verifies the code honours them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from preframr_tokens.stfconstants import (
    DIFF_OP,
    FLIP_OP,
    FREQ_TRAJ_OP,
    GEN_TABLE_REF_OP,
    INSTR_REF_OP,
    MODE_VOL_REG,
    ORN_OP,
    SKEL_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    SWEEP_OP,
    TRACK_REF_OP,
    VOICE_REG_SIZE,
    VOICES,
    WAVETABLE_ONESHOT_OP,
    WAVETABLE_REF_OP,
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
    "relative_base_unsound",
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
    REPLAY = regenerated from a codebook/trajectory (STAMP_REF/SKEL/...) -- opaque to a delta encoder.
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


_FREQ, _PWM, _CTRL, _AD, _SR, _FILT = (
    RegClass.FREQ,
    RegClass.PWM,
    RegClass.CTRL,
    RegClass.AD,
    RegClass.SR,
    RegClass.FILTER,
)
_ABS, _REL, _RPL = Effect.ABSOLUTE, Effect.RELATIVE, Effect.REPLAY

CONTRACTS = {
    c.name: c
    for c in (
        MacroContract(
            "TrajectoryAnchorPass",
            frozenset(),
            frozenset(),
            frozenset(),
            FrameEffect.PRESERVES,
            False,
        ),
        MacroContract(
            "StampPass",
            frozenset(
                {(_FREQ, _RPL), (_PWM, _RPL), (_CTRL, _RPL), (_AD, _RPL), (_SR, _RPL)}
            ),
            frozenset({_FREQ, _PWM}),
            frozenset(),
            FrameEffect.ANCHORED_REPLAY,
            True,
        ),
        MacroContract(
            "SkeletonPass",
            frozenset({(_FREQ, _RPL)}),
            frozenset(),
            frozenset(),
            FrameEffect.ANCHORED_REPLAY,
            False,
        ),
        MacroContract(
            "WavetablePass",
            frozenset({(_FREQ, _RPL), (_PWM, _RPL)}),
            frozenset(),
            frozenset(),
            FrameEffect.ANCHORED_REPLAY,
            False,
        ),
        MacroContract(
            "FreqTrajectoryPass",
            frozenset({(_FREQ, _RPL)}),
            frozenset(),
            frozenset(),
            FrameEffect.PRESERVES,
            False,
        ),
        MacroContract(
            "PerRegBurstPass",
            frozenset({(_FREQ, _REL), (_PWM, _REL), (_FILT, _REL)}),
            frozenset(),
            frozenset({"StampPass"}),
            FrameEffect.PRESERVES,
            False,
        ),
        MacroContract(
            "InstrumentProgramPass",
            frozenset({(_CTRL, _RPL), (_AD, _RPL), (_SR, _RPL)}),
            frozenset(),
            frozenset(),
            FrameEffect.ANCHORED_REPLAY,
            True,
        ),
        MacroContract(
            "GeneratorPass",
            frozenset({(_FREQ, _RPL), (_PWM, _RPL), (_FILT, _RPL)}),
            frozenset(),
            frozenset(),
            FrameEffect.ANCHORED_REPLAY,
            True,
        ),
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

PIPELINE_ORDER = (
    "TrajectoryAnchorPass",
    "StampPass",
    "SkeletonPass",
    "WavetablePass",
    "FreqTrajectoryPass",
    "PerRegBurstPass",
    "InstrumentProgramPass",
    "GeneratorPass",
    "frame_consolidation",
)


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


KNOWN_MISMATCHES = frozenset(
    {
        Mismatch("frame_anchor", "StampPass", "frame_consolidation", None),
        Mismatch("frame_anchor", "InstrumentProgramPass", "frame_consolidation", None),
        Mismatch("frame_anchor", "GeneratorPass", "frame_consolidation", None),
    }
)


LOSSLESS_PASSES = frozenset({"StampPass"})


RELATIVE_OPS = frozenset({int(DIFF_OP), int(FLIP_OP)})
REPLAY_OPS = frozenset(
    {
        int(STAMP_REF_OP),
        int(STAMP_REL_REF_OP),
        int(SKEL_OP),
        int(ORN_OP),
        int(SWEEP_OP),
        int(FREQ_TRAJ_OP),
        int(TRACK_REF_OP),
        int(WAVETABLE_REF_OP),
        int(WAVETABLE_ONESHOT_OP),
        int(INSTR_REF_OP),
        int(GEN_TABLE_REF_OP),
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


def relative_base_unsound(df):
    """Walk an encoded token stream and return RELATIVE writes whose base crosses a REPLAY of the same
    register -- i.e. a delta op decoding against a value some codebook replay set, not the literal
    write its encoder saw. Empty iff every RELATIVE encoder honoured its barrier declaration. The code
    check backing the R1 contract: returns ``[(row_index, reg, replay_op)]``."""
    regs = df["reg"].to_numpy()
    ops = df["op"].to_numpy() if "op" in df.columns else None
    if ops is None:
        return []
    last_writer = {}
    bad = []
    for i in range(len(regs)):
        r = int(regs[i])
        if r < 0:
            continue
        op = int(ops[i])
        if op in RELATIVE_OPS:
            prev = last_writer.get(r)
            if prev is not None and prev in REPLAY_OPS:
                bad.append((i, r, prev))
            last_writer[r] = op
        elif op in REPLAY_OPS:
            last_writer[r] = op
        else:
            last_writer[r] = op
    return bad
