"""VoiceTrajectoryTransform tests: registry, default-off no-op, forward annotation, change-only emission, inverse round-trip."""

import argparse
import unittest

import pandas as pd

from preframr_tokens.macros import (  # noqa: F401 register transforms
    transforms_audio_bit_exact,
    transforms_bit_exact,
    transforms_voice_trajectory,
)
from preframr_tokens.macros.transform import _REGISTRY
from preframr_tokens.macros.transforms_voice_trajectory import VoiceTrajectoryTransform
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FLIP_OP,
    FLIP2_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    SET_OP,
    TRANSPOSE_OP,
    VOICE_REG,
    VOICE_TRAJ_REG,
)


def _row(op, reg, val, subreg=-1, diff=0):
    return {
        "op": int(op),
        "reg": int(reg),
        "subreg": int(subreg),
        "val": int(val),
        "diff": int(diff),
    }


def _svt(voice_order):
    out = 0
    for slot, v in enumerate(voice_order):
        out |= (int(v) + 1) << (slot * 2)
    return int(out)


def _df_simple_voice_reg_only():
    return pd.DataFrame([_row(SET_OP, VOICE_REG, v) for v in (0, 1, 2)])


def _df_two_frames():
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(2):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            rows.append(_row(SET_OP, 0, 0x10 + f * 0x100 + v))
            rows.append(_row(SET_OP, 4, 0x41))
    return pd.DataFrame(rows)


def _df_many_frames(n_frames=12):
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(n_frames):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            rows.append(_row(SET_OP, 0, 0x20 + f * 10 + v))
            rows.append(_row(SET_OP, 4, 0x41))
    return pd.DataFrame(rows)


class TestVoiceTrajectoryRegistration(unittest.TestCase):
    def test_registered_under_voice_trajectory_name(self):
        self.assertIn("voice_trajectory", _REGISTRY)
        self.assertEqual(_REGISTRY["voice_trajectory"], VoiceTrajectoryTransform)

    def test_declared_position_constraints(self):
        cls = VoiceTrajectoryTransform
        self.assertIn("voice_block_order", cls.MUST_FOLLOW)
        self.assertTrue(cls.IDEMPOTENT)

    def test_default_window_is_8(self):
        instance = VoiceTrajectoryTransform()
        self.assertEqual(instance.params["window"], 8)

    def test_voice_traj_reg_constant_is_reserved(self):
        from preframr_tokens.stfconstants import (
            DELAY_REG,
            FRAME_REG,
            LOOP_OP_REG,
            SUPER_FRAME_REG,
            VOICE_REG,
        )

        reserved = {
            FRAME_REG,
            DELAY_REG,
            VOICE_REG,
            LOOP_OP_REG,
            SUPER_FRAME_REG,
            VOICE_TRAJ_REG,
        }
        self.assertEqual(
            len(reserved),
            6,
            msg=f"reserved-reg slots collide: {sorted(reserved)}",
        )


class TestDefaultOffIsNoOp(unittest.TestCase):
    def test_arg_disabled_passes_df_through(self):
        t = VoiceTrajectoryTransform()
        df = _df_two_frames()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=False))
        pd.testing.assert_frame_equal(out, df)

    def test_no_args_passes_df_through(self):
        t = VoiceTrajectoryTransform()
        df = _df_two_frames()
        out = t.forward(df, args=None)
        pd.testing.assert_frame_equal(out, df)

    def test_empty_df_passes_through(self):
        t = VoiceTrajectoryTransform()
        df = pd.DataFrame(columns=["op", "reg", "subreg", "val", "diff"])
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        pd.testing.assert_frame_equal(out, df)

    def test_no_voice_reg_rows_passes_through(self):
        t = VoiceTrajectoryTransform()
        df = pd.DataFrame([_row(SET_OP, 0, 1)])
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        pd.testing.assert_frame_equal(out, df)


class TestForwardInsertsTrajectoryRows(unittest.TestCase):
    def test_forward_emits_one_traj_row_per_voice_on_first_frame(self):
        t = VoiceTrajectoryTransform()
        df = _df_two_frames()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        self.assertGreater(len(out), len(df))
        traj_rows = out[out["reg"] == VOICE_TRAJ_REG]
        self.assertGreaterEqual(len(traj_rows), 3)
        self.assertTrue((traj_rows["val"] >= 0).all())
        self.assertTrue((traj_rows["val"] <= 255).all())

    def test_change_only_emission_skips_repeated_byte(self):
        t = VoiceTrajectoryTransform(window=4)
        df = _df_many_frames(n_frames=12)
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        traj_count = int((out["reg"] == VOICE_TRAJ_REG).sum())
        voice_count = int((df["reg"] == VOICE_REG).sum())
        self.assertLess(traj_count, voice_count)

    def test_traj_row_immediately_follows_voice_reg(self):
        t = VoiceTrajectoryTransform()
        df = _df_two_frames()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_pass=True)
        ).reset_index(drop=True)
        for i, reg in enumerate(out["reg"].tolist()):
            if int(reg) == int(VOICE_TRAJ_REG):
                self.assertGreater(i, 0)
                self.assertEqual(int(out.iloc[i - 1]["reg"]), int(VOICE_REG))


def _df_with_diff_freq(deltas_per_frame):
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f, delta in enumerate(deltas_per_frame):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            if v == 0:
                rows.append(_row(DIFF_OP, 0, delta & 0xFFFF))
            rows.append(_row(SET_OP, 4, 0x41))
    return pd.DataFrame(rows)


def _df_with_flip2_arp():
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(2):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            if v == 0:
                a, b = 5, -5
                packed = ((a & 0xFF) << 8) | (b & 0xFF)
                rows.append(_row(FLIP2_OP, 0, packed, subreg=6))
            rows.append(_row(SET_OP, 4, 0x41))
    return pd.DataFrame(rows)


def _df_with_transpose():
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(2):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            if v == 0:
                rows.append(_row(TRANSPOSE_OP, 0, 4, subreg=0b111))
            rows.append(_row(SET_OP, 4, 0x41))
    return pd.DataFrame(rows)


def _df_with_hard_restart():
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(2):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            if v == 0:
                packed = ((0x40 & 0xFF) << 8) | (0x41 & 0xFF)
                rows.append(_row(HARD_RESTART_OP, 4, packed))
    return pd.DataFrame(rows)


def _df_with_legato():
    voice_order = [0, 1, 2]
    svt = _svt(voice_order)
    rows = []
    for f in range(2):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        for v in voice_order:
            rows.append(_row(SET_OP, VOICE_REG, 0))
            if v == 0:
                rows.append(_row(LEGATO_OP_CLUSTER_2, 4, 0x04))
    return pd.DataFrame(rows)


class TestForwardReadsNonSetOps(unittest.TestCase):
    def test_diff_op_freq_lo_drives_pitch_direction(self):
        t = VoiceTrajectoryTransform(window=8)
        df = _df_with_diff_freq([10, 12, 14, 16, 18])
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        traj_rows = out[out["reg"] == VOICE_TRAJ_REG]
        voice0_bytes = []
        for i, reg in enumerate(out["reg"].tolist()):
            if int(reg) == int(VOICE_TRAJ_REG):
                prev = out.iloc[i - 1]
                if int(prev["reg"]) == int(VOICE_REG):
                    voice0_bytes.append(int(out.iloc[i]["val"]))
                    break
        self.assertTrue(voice0_bytes)
        pitch_dir = (voice0_bytes[0] >> 1) & 0b11
        self.assertEqual(pitch_dir, 2)

    def test_flip2_op_freq_lo_marks_arpeggiating(self):
        t = VoiceTrajectoryTransform(window=8)
        df = _df_with_flip2_arp()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        first_byte = None
        for i, reg in enumerate(out["reg"].tolist()):
            if int(reg) == int(VOICE_TRAJ_REG):
                prev = out.iloc[i - 1]
                if int(prev["reg"]) == int(VOICE_REG):
                    first_byte = int(out.iloc[i]["val"])
                    break
        self.assertIsNotNone(first_byte)
        arp_bit = (first_byte >> 3) & 0b1
        self.assertEqual(arp_bit, 1)

    def test_transpose_op_attributes_to_all_voices_in_mask(self):
        t = VoiceTrajectoryTransform(window=8)
        df = _df_with_transpose()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        voice_dirs = {}
        cur_v = None
        for i, reg in enumerate(out["reg"].tolist()):
            r = int(reg)
            if r == int(VOICE_REG):
                cur_v = int(out.iloc[i]["val"]) if int(out.iloc[i]["val"]) else cur_v
            elif r == int(VOICE_TRAJ_REG) and cur_v is not None:
                pitch_dir = (int(out.iloc[i]["val"]) >> 1) & 0b11
                voice_dirs.setdefault(cur_v, pitch_dir)
        self.assertTrue(all(d == 2 for d in voice_dirs.values()))

    def test_hard_restart_op_sets_gate_and_decodes_ctrl(self):
        t = VoiceTrajectoryTransform(window=8)
        df = _df_with_hard_restart()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        first_byte = None
        for i, reg in enumerate(out["reg"].tolist()):
            if int(reg) == int(VOICE_TRAJ_REG):
                prev = out.iloc[i - 1]
                if int(prev["reg"]) == int(VOICE_REG):
                    first_byte = int(out.iloc[i]["val"])
                    break
        self.assertIsNotNone(first_byte)
        gate_bit = first_byte & 0b1
        self.assertEqual(gate_bit, 1)

    def test_legato_op_sets_gate(self):
        t = VoiceTrajectoryTransform(window=8)
        df = _df_with_legato()
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        first_byte = None
        for i, reg in enumerate(out["reg"].tolist()):
            if int(reg) == int(VOICE_TRAJ_REG):
                prev = out.iloc[i - 1]
                if int(prev["reg"]) == int(VOICE_REG):
                    first_byte = int(out.iloc[i]["val"])
                    break
        self.assertIsNotNone(first_byte)
        gate_bit = first_byte & 0b1
        self.assertEqual(gate_bit, 1)


class TestInverseStripsTrajRows(unittest.TestCase):
    def test_inverse_drops_all_voice_traj_reg_rows_round_trip(self):
        t = VoiceTrajectoryTransform()
        df = _df_two_frames()
        forward = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True),
            df.reset_index(drop=True),
        )

    def test_inverse_passthrough_when_no_traj_rows(self):
        t = VoiceTrajectoryTransform()
        df = _df_simple_voice_reg_only()
        out = t.inverse(df)
        pd.testing.assert_frame_equal(out, df)


class TestReplaceVoiceRegMode(unittest.TestCase):
    def test_replace_drops_voice_reg_and_emits_traj_in_its_slot(self):
        t = VoiceTrajectoryTransform(window=4, replace_voice_reg=True)
        df = _df_two_frames()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_pass=True)
        ).reset_index(drop=True)
        self.assertEqual(int((out["reg"] == int(VOICE_REG)).sum()), 0)
        traj_rows = out[out["reg"] == int(VOICE_TRAJ_REG)]
        self.assertGreaterEqual(len(traj_rows), 3)

    def test_replace_mode_traj_rows_match_voice_reg_count(self):
        t = VoiceTrajectoryTransform(window=4, replace_voice_reg=True)
        df = _df_many_frames(n_frames=12)
        voice_reg_count = int((df["reg"] == int(VOICE_REG)).sum())
        out = t.forward(df, args=argparse.Namespace(voice_trajectory_pass=True))
        traj_count = int((out["reg"] == int(VOICE_TRAJ_REG)).sum())
        self.assertEqual(traj_count, voice_reg_count)


if __name__ == "__main__":
    unittest.main()
