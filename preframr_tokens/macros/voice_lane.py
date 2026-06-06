"""Layer-3 voice-lane reorder core (AGENT_TASK_melody_skeleton.md §4B, superframe_voice_lane_design.md):
the frame-major <-> voice-major bijection. De-multiplex a frame-major block to voice-major lanes (each
voice contiguous, ordered accompaniment->melody) so a melody onset's own predecessor is positionally
local. Lossless by construction -- each event keeps its (frame, seq), so the inverse restores the exact
canonical frame-major order. The marker-agnostic heart the pipeline wiring drives."""

from __future__ import annotations

from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    RESERVED_REG_NEG124,
    VOICE_REG,
)

__all__ = [
    "lane_rank",
    "to_voice_major",
    "to_frame_major",
    "round_trips",
    "df_to_voice_major",
    "df_to_frame_major",
]

VLANE_REG = RESERVED_REG_NEG124
_VLANE_SUB_LANE = -2
_VLANE_SUB_FTAG = -3
_MARKER_REGS = frozenset({FRAME_REG, DELAY_REG, VOICE_REG})


def lane_rank(roles, voices):
    """Map ``{voice: role}`` to an emit rank per voice: bass(0) < mid(1) < lead(2) < unknown(3), so the
    accompaniment (its harmonic context) precedes the melody (melody-last). Ties break by voice index, so
    a voice with no role keeps a stable canonical slot."""
    order = {"bass": 0, "mid": 1, "lead": 2}
    return {v: (order.get(roles.get(v), 3), v) for v in voices}


def to_voice_major(events, ranks):
    """Frame-major events -> voice-major. Each event is ``(frame, voice, seq, payload)`` with ``seq`` its
    within-(frame,voice) order. Stable-sort by ``(lane_rank, frame, seq)`` so every voice's line is one
    contiguous lane, accompaniment lanes first. A pure permutation -- no event is added or dropped.
    """
    return sorted(events, key=lambda e: (ranks[e[1]], e[0], e[2]))


def to_frame_major(events):
    """Voice-major events -> the canonical frame-major order: stable-sort by ``(frame, voice, seq)``. The
    exact inverse of ``to_voice_major`` for any canonically-ordered input (the order the dumps render).
    """
    return sorted(events, key=lambda e: (e[0], e[1], e[2]))


def round_trips(events, ranks):
    """True iff ``to_frame_major(to_voice_major(events)) == events`` -- the bit-exact bijection guard for a
    canonically-ordered (frame, voice, seq) block."""
    return to_frame_major(to_voice_major(events, ranks)) == list(events)


def _annotate(records):
    """Per row, ``(is_marker, frame, slot)``: frame advances on FRAME(+1)/DELAY(+val); slot resets on
    FRAME and increments on each VOICE marker; content rows inherit the current (frame, slot). Prefix
    content before the first FRAME sits at (frame=-1, slot=0)."""
    frame = -1
    slot = 0
    ann = []
    for r in records:
        reg = int(r["reg"])
        if reg == FRAME_REG:
            frame += 1
            slot = 0
            ann.append((True, frame, slot))
        elif reg == DELAY_REG:
            frame += int(r["val"])
            ann.append((True, frame, slot))
        elif reg == VOICE_REG:
            slot += 1
            ann.append((True, frame, slot))
        else:
            ann.append((False, frame, slot))
    return ann


def _marker_row(template, val, subreg, diff):
    row = dict(template)
    row["reg"] = VLANE_REG
    row["val"] = val
    row["subreg"] = subreg
    row["diff"] = diff
    if "op" in row:
        row["op"] = 0
    return row


def df_to_voice_major(records):
    """Frame-major token records -> ``[marker skeleton verbatim] + [per-slot lanes]``. Each lane is a slot's
    content across frames (contiguous = de-multiplexed), led by a LANE marker, each frame-group preceded by
    a FTAG carrying its frame index. The skeleton (FRAME/DELAY/VOICE rows, verbatim) preserves the exact
    sval/DELAY bytes so the inverse regenerates the frame structure bit-exact."""
    ann = _annotate(records)
    skeleton = [r for r, (is_m, _f, _s) in zip(records, ann) if is_m]
    groups = {}
    slots = set()
    for r, (is_m, f, s) in zip(records, ann):
        if is_m:
            continue
        groups.setdefault((s, f), []).append(r)
        slots.add(s)
    template = records[0] if records else {}
    out = list(skeleton)
    for s in sorted(slots):
        out.append(_marker_row(template, s, _VLANE_SUB_LANE, 0))
        for gs, gf in sorted(k for k in groups if k[0] == s):
            out.append(_marker_row(template, 0, _VLANE_SUB_FTAG, gf))
            out.extend(groups[(gs, gf)])
    return out


def df_to_frame_major(records):
    """Inverse of ``df_to_voice_major``: replay the marker skeleton frame-major, slotting each (frame, slot)
    content group back at its position, to restore the exact original row order (bit-exact).
    """
    first_lane = next(
        (i for i, r in enumerate(records) if int(r["reg"]) == VLANE_REG), len(records)
    )
    skeleton = records[:first_lane]
    groups = {}
    cur_slot = None
    cur_frame = None
    for r in records[first_lane:]:
        if int(r["reg"]) == VLANE_REG:
            if int(r["subreg"]) == _VLANE_SUB_LANE:
                cur_slot = int(r["val"])
            elif int(r["subreg"]) == _VLANE_SUB_FTAG:
                cur_frame = int(r["diff"])
            continue
        groups.setdefault((cur_slot, cur_frame), []).append(r)
    out = []
    out.extend(groups.get((0, -1), []))
    frame = -1
    slot = 0
    for r in skeleton:
        reg = int(r["reg"])
        if reg == FRAME_REG:
            frame += 1
            slot = 0
        elif reg == DELAY_REG:
            frame += int(r["val"])
        elif reg == VOICE_REG:
            slot += 1
        out.append(r)
        out.extend(groups.get((slot, frame), []))
    return out
