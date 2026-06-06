"""Layer-3 voice-lane reorder core: the frame-major <-> voice-major bijection is bit-exact (the inverse
restores the canonical render order), de-multiplexes voices into contiguous lanes, and emits
accompaniment before melody (melody-last) by role rank."""

import glob
import itertools
import os
import unittest

from preframr_tokens.macros.voice_lane import (
    df_to_frame_major,
    df_to_voice_major,
    lane_rank,
    round_trips,
    to_frame_major,
    to_voice_major,
    VLANE_REG,
)
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, VOICE_REG

_HVSC = "/scratch/preframr/hvsc"


def _rec(reg, val, op=0, subreg=-1, diff=0):
    return {"reg": reg, "val": val, "op": op, "subreg": subreg, "diff": diff}


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


class TestVoiceLaneDf(unittest.TestCase):
    def test_synthetic_grammar_round_trips_and_demuxes(self):
        recs = [
            _rec(0, 116, op=84, subreg=0),
            _rec(FRAME_REG, 6),
            _rec(0, 1),
            _rec(VOICE_REG, 0),
            _rec(7, 2),
            _rec(FRAME_REG, 6),
            _rec(0, 3),
            _rec(VOICE_REG, 0),
            _rec(7, 4),
            _rec(DELAY_REG, 2),
            _rec(FRAME_REG, 6),
            _rec(0, 5),
            _rec(VOICE_REG, 0),
            _rec(7, 6),
        ]
        vm = df_to_voice_major(recs)
        self.assertEqual(df_to_frame_major(vm), recs)
        lane_markers = [r for r in vm if int(r["reg"]) == VLANE_REG]
        self.assertTrue(lane_markers, "no lane markers emitted")
        content_regs = [
            int(r["reg"])
            for r in vm
            if int(r["reg"]) >= 0 and int(r["reg"]) not in (FRAME_REG, VOICE_REG)
        ]
        self.assertEqual(
            content_regs, [0, 0, 0, 0, 7, 7, 7], "content not de-muxed by slot"
        )

    def test_prefix_content_before_first_frame(self):
        recs = [
            _rec(0, 9, op=84),
            _rec(FRAME_REG, 6),
            _rec(0, 1),
            _rec(VOICE_REG, 0),
            _rec(7, 2),
        ]
        self.assertEqual(df_to_frame_major(df_to_voice_major(recs)), recs)

    def test_corpus_round_trip_bit_exact(self):
        paths = sorted(
            glob.glob(os.path.join(_HVSC, "**", "*.dump.parquet"), recursive=True)
        )
        if not paths:
            self.skipTest("HVSC corpus unavailable")
        from preframr_tokens.reglogparser import RegLogParser
        from preframr_tokens.tokenizer_config import default_tokenizer_args

        args = default_tokenizer_args(
            generator_pass=True, instrument_program=True, melody_skeleton=True
        )
        checked = 0
        for path in paths[:: max(1, len(paths) // 40)][:20]:
            df = next(
                RegLogParser(args=args).parse(
                    path, max_perm=1, require_pq=False, reparse=True
                ),
                None,
            )
            if df is None:
                continue
            checked += 1
            recs = df.to_dict("records")
            self.assertEqual(
                df_to_frame_major(df_to_voice_major(recs)),
                recs,
                f"not bit-exact: {path}",
            )
        if checked == 0:
            self.skipTest("no corpus tunes parsed")


class TestVoiceLaneTransform(unittest.TestCase):
    def test_registered_and_is_a_flag(self):
        from preframr_tokens.macros.transform import (
            ensure_default_transforms_registered,
            get_transform_class,
        )
        from preframr_tokens.macros.flag_registry import macro_flag_names

        ensure_default_transforms_registered()
        cls = get_transform_class("voice_lane")
        self.assertEqual(cls.TIER, "bit_exact")
        self.assertIn("voice_block_order", cls.MUST_FOLLOW)
        self.assertIn("voice_lane", macro_flag_names())

    def test_default_off_is_identity(self):
        from preframr_tokens.macros.transform import get_transform_class
        from preframr_tokens.tokenizer_config import default_tokenizer_args
        import pandas as pd

        t = get_transform_class("voice_lane")()
        df = pd.DataFrame([_rec(FRAME_REG, 6), _rec(0, 1)])
        args = default_tokenizer_args()
        pd.testing.assert_frame_equal(t.forward(df, args=args), df)

    def test_round_trip_on_corpus_df(self):
        paths = sorted(
            glob.glob(os.path.join(_HVSC, "**", "*.dump.parquet"), recursive=True)
        )
        if not paths:
            self.skipTest("HVSC corpus unavailable")
        from preframr_tokens.macros.transform import get_transform_class
        from preframr_tokens.reglogparser import RegLogParser
        from preframr_tokens.tokenizer_config import default_tokenizer_args

        t = get_transform_class("voice_lane")()
        args = default_tokenizer_args(
            generator_pass=True,
            instrument_program=True,
            melody_skeleton=True,
            voice_lane=True,
        )
        checked = 0
        for path in paths[:: max(1, len(paths) // 20)][:6]:
            df = next(
                RegLogParser(args=args).parse(
                    path, max_perm=1, require_pq=False, reparse=True
                ),
                None,
            )
            if df is None:
                continue
            checked += 1
            self.assertTrue(t.round_trip_check(df, args=args))
            self.assertGreater(len(t.forward(df, args=args)), len(df))
        if checked == 0:
            self.skipTest("no corpus tunes parsed")


if __name__ == "__main__":
    unittest.main()
