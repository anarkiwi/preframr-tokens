"""Layer-3 voice-lane reorder core (AGENT_TASK_melody_skeleton.md §4B, superframe_voice_lane_design.md):
the frame-major <-> voice-major bijection. De-multiplex a frame-major block to voice-major lanes (each
voice contiguous, ordered accompaniment->melody) so a melody onset's own predecessor is positionally
local. Lossless by construction -- each event keeps its (frame, seq), so the inverse restores the exact
canonical frame-major order. The marker-agnostic heart the pipeline wiring drives."""

from __future__ import annotations

__all__ = ["lane_rank", "to_voice_major", "to_frame_major", "round_trips"]


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
