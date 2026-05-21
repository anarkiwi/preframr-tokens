"""VoiceTrajectoryDistributedTransform tests: 4-bit per-voice VOICE_REG packing + 2-bit FRAME_REG bit-6/7 packing, no new rows, alphabet bounded."""

import argparse
import unittest

import pandas as pd

from preframr_tokens.macros import (  # noqa: F401 register transforms
    transforms_audio_bit_exact,
    transforms_bit_exact,
    transforms_voice_trajectory,
    transforms_voice_trajectory_distributed,
)
from preframr_tokens.macros.transform import _REGISTRY
from preframr_tokens.macros.transforms_voice_trajectory_distributed import (
    VoiceTrajectoryDistributedTransform,
)
from preframr_tokens.stfconstants import FRAME_REG, SET_OP, VOICE_REG


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


def _df_three_frames_voice0():
    svt = _svt([0])
    rows = []
    gates = [0x40, 0x41, 0x41]
    for f, gate in enumerate(gates):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        rows.append(_row(SET_OP, VOICE_REG, 0))
        rows.append(_row(SET_OP, 0, 0x10 + 0x10 * f))
        rows.append(_row(SET_OP, 4, gate))
    return pd.DataFrame(rows)


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("voice_trajectory_distributed", _REGISTRY)
        self.assertEqual(
            _REGISTRY["voice_trajectory_distributed"],
            VoiceTrajectoryDistributedTransform,
        )


class TestDefaultOff(unittest.TestCase):
    def test_disabled_is_no_op(self):
        t = VoiceTrajectoryDistributedTransform()
        df = _df_three_frames_voice0()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=False)
        )
        pd.testing.assert_frame_equal(out, df)


class TestRowCountUnchanged(unittest.TestCase):
    def test_distribution_adds_no_rows(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        self.assertEqual(len(out), len(df))


class TestFrameValPacks(unittest.TestCase):
    def test_frame_val_low_6_bits_preserve_svt(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        original_svt = int(df[df["reg"] == int(FRAME_REG)].iloc[0]["val"])
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        for _, row in out[out["reg"] == int(FRAME_REG)].iterrows():
            self.assertEqual(int(row["val"]) & 0x3F, original_svt & 0x3F)

    def test_gate_transition_frame_sets_bit_6(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        frame_vals = out[out["reg"] == int(FRAME_REG)]["val"].tolist()
        gate_bits = [(int(v) >> 6) & 0b1 for v in frame_vals]
        self.assertTrue(any(g == 1 for g in gate_bits))


class TestVoiceRegPacks(unittest.TestCase):
    def test_voice_reg_val_within_4_bits(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        for _, row in out[out["reg"] == int(VOICE_REG)].iterrows():
            self.assertLess(int(row["val"]), 16)

    def test_voice_reg_val_carries_gate_bit(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        out = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        voice_vals = out[out["reg"] == int(VOICE_REG)]["val"].tolist()
        gates_present = [int(v) & 0b1 for v in voice_vals]
        self.assertTrue(any(g == 1 for g in gates_present))


class TestInverseStrips(unittest.TestCase):
    def test_inverse_restores_svt_only_in_frame_reg(self):
        t = VoiceTrajectoryDistributedTransform(window=4)
        df = _df_three_frames_voice0()
        forward = t.forward(
            df, args=argparse.Namespace(voice_trajectory_distributed_pass=True)
        )
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True), df.reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
