"""SuperFrameTransform scaffold tests: registry presence, N=1 no-op, N>=2 not-implemented sentinel."""

import argparse
import unittest

import pandas as pd

from preframr_tokens.macros import (  # noqa: F401 register transforms
    transforms_audio_bit_exact,
    transforms_bit_exact,
    transforms_superframe,
)
from preframr_tokens.macros.transform import _REGISTRY
from preframr_tokens.macros.transforms_superframe import SuperFrameTransform
from preframr_tokens.stfconstants import FRAME_REG, SET_OP


def _tiny_frame_df(n_frames=8):
    rows = []
    for f in range(n_frames):
        rows.append(
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "diff": 19656}
        )
        rows.append({"op": SET_OP, "reg": 0, "subreg": -1, "val": 5 + f, "diff": 100})
    return pd.DataFrame(rows)


class TestSuperFrameRegistration(unittest.TestCase):
    def test_registered_under_super_frame_name(self):
        self.assertIn("super_frame", _REGISTRY)
        self.assertEqual(_REGISTRY["super_frame"], SuperFrameTransform)

    def test_declared_position_constraints(self):
        cls = SuperFrameTransform
        self.assertIn("voice_block_order", cls.MUST_FOLLOW)
        self.assertIn("add_voice_reg", cls.MUST_PRECEDE)
        self.assertFalse(cls.IDEMPOTENT)

    def test_default_n_frames_is_4(self):
        instance = SuperFrameTransform()
        self.assertEqual(instance.params["n_frames"], 4)


class TestNoOpAtN1(unittest.TestCase):
    def test_n1_is_identity(self):
        t = SuperFrameTransform(n_frames=1)
        df = _tiny_frame_df()
        out = t.forward(df, args=argparse.Namespace(super_frame_pass=True))
        pd.testing.assert_frame_equal(out, df)

    def test_disabled_is_identity_regardless_of_n(self):
        t = SuperFrameTransform(n_frames=4)
        df = _tiny_frame_df()
        out = t.forward(df, args=argparse.Namespace(super_frame_pass=False))
        pd.testing.assert_frame_equal(out, df)

    def test_inverse_passthrough_when_no_super_marker(self):
        t = SuperFrameTransform(n_frames=4)
        df = _tiny_frame_df()
        out = t.inverse(df)
        pd.testing.assert_frame_equal(out, df)


class TestN2PlusNotImplementedSentinel(unittest.TestCase):
    def test_forward_n4_raises_not_implemented(self):
        t = SuperFrameTransform(n_frames=4)
        df = _tiny_frame_df()
        with self.assertRaises(NotImplementedError) as cm:
            t.forward(df, args=argparse.Namespace(super_frame_pass=True))
        self.assertIn("superframe_mini_design.md", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
