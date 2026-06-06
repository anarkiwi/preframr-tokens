"""Layer-3 voice-lane reorder core: the frame-major <-> voice-major bijection is bit-exact (the inverse
restores the canonical render order), de-multiplexes voices into contiguous lanes, and emits
accompaniment before melody (melody-last) by role rank."""

import itertools
import unittest

from preframr_tokens.macros.voice_lane import (
    lane_rank,
    round_trips,
    to_frame_major,
    to_voice_major,
)


def _canonical(frames, voices):
    """A canonical frame-major block: for each frame, each voice in index order, one event each."""
    out = []
    for f in range(frames):
        for v in range(voices):
            out.append((f, v, 0, f"f{f}v{v}"))
    return out


class TestVoiceLaneCore(unittest.TestCase):
    def test_round_trip_bit_exact(self):
        events = _canonical(5, 3)
        ranks = lane_rank({}, range(3))
        self.assertTrue(round_trips(events, ranks))

    def test_voice_major_is_contiguous_per_voice(self):
        events = _canonical(4, 3)
        ranks = lane_rank({}, range(3))
        vm = to_voice_major(events, ranks)
        voices_in_order = [e[1] for e in vm]
        self.assertEqual(voices_in_order, [0] * 4 + [1] * 4 + [2] * 4)

    def test_melody_last_ordering(self):
        events = _canonical(3, 3)
        roles = {0: "lead", 1: "bass", 2: "mid"}
        ranks = lane_rank(roles, range(3))
        vm = to_voice_major(events, ranks)
        lane_voice_order = [
            v
            for v, _g in sorted(
                {e[1]: ranks[e[1]] for e in vm}.items(), key=lambda kv: kv[1]
            )
        ]
        self.assertEqual(lane_voice_order, [1, 2, 0])
        first_lane = vm[0][1]
        last_lane = vm[-1][1]
        self.assertEqual(first_lane, 1, "bass (accompaniment) emitted first")
        self.assertEqual(last_lane, 0, "lead (melody) emitted last")

    def test_round_trip_with_roles_and_multi_seq(self):
        events = []
        for f in range(6):
            for v in range(3):
                for s in range(2):
                    events.append((f, v, s, (f, v, s)))
        roles = {0: "lead", 1: "bass", 2: "mid"}
        ranks = lane_rank(roles, range(3))
        self.assertTrue(round_trips(events, ranks))
        self.assertEqual(to_frame_major(to_voice_major(events, ranks)), events)

    def test_round_trip_all_role_permutations(self):
        events = _canonical(4, 3)
        for perm in itertools.permutations(["bass", "mid", "lead"]):
            roles = {v: perm[v] for v in range(3)}
            ranks = lane_rank(roles, range(3))
            self.assertTrue(round_trips(events, ranks), perm)

    def test_sparse_frames_preserved(self):
        events = [(0, 0, 0, "a"), (0, 2, 0, "b"), (3, 1, 0, "c"), (3, 2, 0, "d")]
        ranks = lane_rank({}, range(3))
        self.assertEqual(to_frame_major(to_voice_major(events, ranks)), events)


if __name__ == "__main__":
    unittest.main()
