"""Round-trip + voice-symmetry tests over every Transform subclass."""

import argparse
import unittest

import pandas as pd

from preframr_tokens.macros import (  # pylint: disable=unused-import
    transforms_audio_bit_exact,
    transforms_bit_exact,
)
from preframr_tokens.macros.transform import (
    _REGISTRY,
    collect_decomposing_op_codes,
    collect_op_loss_tiers,
    collect_substitutable_op_subregs,
    collect_substitutable_ops,
)
from preframr_tokens.stfconstants import (
    FRAME_REG,
    HARD_RESTART_OP,
    SET_OP,
    VOICE_CTRL_REG,
)


def _voice_ctrl_regs():
    return list(VOICE_CTRL_REG.values())


def _scaffold_ctrl_set_pair(ctrl_reg, a, b):
    return [
        {"op": SET_OP, "reg": ctrl_reg, "subreg": -1, "val": int(a), "diff": 100},
        {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "diff": 19656},
        {"op": SET_OP, "reg": ctrl_reg, "subreg": -1, "val": int(b), "diff": 100},
    ]


class TestSubstitutabilityRegistry(unittest.TestCase):
    def test_collects_substitutable_ops(self):
        ops = collect_substitutable_ops()
        self.assertIn(SET_OP, ops)
        self.assertNotIn(HARD_RESTART_OP, ops)

    def test_collects_substitutable_op_subregs(self):
        subregs = collect_substitutable_op_subregs()
        from preframr_tokens.stfconstants import (
            PATTERN_REPLAY_OP,
            PATTERN_REPLAY_SUBREG_DIST_HI,
        )

        self.assertIn(
            (int(PATTERN_REPLAY_OP), int(PATTERN_REPLAY_SUBREG_DIST_HI)), subregs
        )

    def test_collects_decomposing_op_codes(self):
        ops = collect_decomposing_op_codes()
        self.assertIn(HARD_RESTART_OP, ops)
        self.assertNotIn(SET_OP, ops)

    def test_collects_op_loss_tiers(self):
        """collect_op_loss_tiers is consumed by preframr/train/model/tier_map.py."""
        tiers = collect_op_loss_tiers()
        self.assertIsInstance(tiers, dict)
        self.assertTrue(tiers, "expected at least one op->tier mapping")
        self.assertEqual(tiers[int(HARD_RESTART_OP)], "structural")
        for op, tier in tiers.items():
            self.assertIsInstance(op, int)
            self.assertIn(tier, {"structural", "mid", "content", "zero"})


class TestTransformRegistry(unittest.TestCase):
    def test_registry_non_empty(self):
        self.assertGreater(len(_REGISTRY), 0)

    def test_all_transforms_declare_tier(self):
        for name, klass in _REGISTRY.items():
            self.assertIn(
                klass.TIER,
                ("bit_exact", "audio_bit_exact", "lossy"),
                msg=f"{name} TIER must be one of bit_exact/audio_bit_exact/lossy",
            )

    def test_op_codes_disjoint_across_transforms(self):
        seen = {}
        for name, klass in _REGISTRY.items():
            for op in klass.OP_CODES:
                self.assertNotIn(
                    int(op),
                    seen,
                    msg=f"op_code {op} owned by both {seen.get(int(op))} and {name}",
                )
                seen[int(op)] = name


class TestHardRestartRoundTrip(unittest.TestCase):
    def test_inverse_unpacks_single_row(self):
        from preframr_tokens.macros.transforms_bit_exact import HardRestartTransform

        t = HardRestartTransform()
        synth = pd.DataFrame(
            [
                {
                    "op": HARD_RESTART_OP,
                    "reg": 4,
                    "subreg": -1,
                    "val": 0x0841,
                    "diff": 100,
                    "description": 0,
                }
            ]
        )
        expanded = t.inverse(synth)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(int(expanded["val"].iloc[0]), 0x08)
        self.assertEqual(int(expanded["val"].iloc[1]), 0x41)
        for v in expanded["op"]:
            self.assertEqual(int(v), SET_OP)


class TestVoiceSymmetry(unittest.TestCase):
    def _voice_symmetry_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--hard-restart-pass", action="store_true", default=True)
        return ap.parse_args([])

    def _synthetic_voice_symmetric_df(self):
        pair_idx_a = (0x41, 0x40)
        rows = []
        for ctrl_reg in _voice_ctrl_regs():
            rows.append(
                {
                    "op": SET_OP,
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 1,
                    "diff": 19656,
                }
            )
            rows.append(
                {
                    "op": SET_OP,
                    "reg": int(ctrl_reg),
                    "subreg": -1,
                    "val": int(pair_idx_a[0]),
                    "diff": 100,
                }
            )
            rows.append(
                {
                    "op": SET_OP,
                    "reg": int(ctrl_reg),
                    "subreg": -1,
                    "val": int(pair_idx_a[1]),
                    "diff": 100,
                }
            )
        return pd.DataFrame(rows)

    def test_per_voice_transforms_produce_symmetric_output(self):
        args = self._voice_symmetry_args()
        df = self._synthetic_voice_symmetric_df()
        for name, klass in sorted(_REGISTRY.items()):
            if not klass.OPERATES_ON_VOICE_REGS:
                continue
            instance = klass()
            out = instance.forward(df.copy(), args=args)
            per_voice_op_counts = {}
            for ctrl_reg in _voice_ctrl_regs():
                sub = out[out["reg"] == int(ctrl_reg)]
                key = tuple(sorted(sub["op"].astype(int).tolist()))
                per_voice_op_counts[int(ctrl_reg)] = key
            uniq = set(per_voice_op_counts.values())
            self.assertEqual(
                len(uniq),
                1,
                msg=(
                    f"{name}: per-voice forward output is asymmetric: "
                    f"{per_voice_op_counts}. A transform that emits ops "
                    f"on one voice must emit the same pattern of ops on "
                    f"all voices."
                ),
            )


if __name__ == "__main__":
    unittest.main()
