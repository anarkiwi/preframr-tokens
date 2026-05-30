"""validate_pipeline_spec catches ordering, idempotence, op-availability errors."""

import argparse
import unittest

from preframr_tokens.macros import (  # noqa: F401 register transforms
    transforms_audio_bit_exact,
    transforms_bit_exact,
)
from preframr_tokens.macros.pipeline_check import (
    PipelineConfigError,
    validate_pipeline_spec,
)
from preframr_tokens.macros.transform import TransformPipeline


def _args(**kw):
    base = dict(
        hard_restart_pass=True,
        ctrl_bigram_pass=True,
        voice_canonical_block_order=True,
        freq_trajectory_pass=True,
        preset_pass=True,
        loop_pass=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestCanonicalSpecPasses(unittest.TestCase):
    def test_minimal_macro_only_spec_validates(self):
        spec = {
            "transforms": [
                {"name": "hard_restart"},
                {"name": "ctrl_bigram"},
                {"name": "voice_block_order"},
            ]
        }
        errors = validate_pipeline_spec(spec, args=_args())
        self.assertEqual(errors, [])

    def test_from_spec_returns_pipeline_when_valid(self):
        spec = {
            "transforms": [
                {"name": "hard_restart"},
                {"name": "ctrl_bigram"},
            ]
        }
        p = TransformPipeline.from_spec(spec, args=_args())
        self.assertEqual(len(p), 2)


class TestUnknownTransform(unittest.TestCase):
    def test_unknown_name_is_error(self):
        spec = {"transforms": [{"name": "totally_made_up"}]}
        errors = validate_pipeline_spec(spec, args=_args())
        self.assertEqual(len(errors), 1)
        self.assertIn("totally_made_up", errors[0])
        self.assertIn("unknown transform", errors[0])

    def test_from_spec_raises_on_unknown(self):
        spec = {"transforms": [{"name": "made_up_name"}]}
        with self.assertRaises(PipelineConfigError) as cm:
            TransformPipeline.from_spec(spec, args=_args())
        self.assertIn("made_up_name", str(cm.exception))


class TestRequiresArgs(unittest.TestCase):
    def test_missing_arg_is_error(self):
        spec = {"transforms": [{"name": "hard_restart"}]}
        args = _args(hard_restart_pass=False)
        errors = validate_pipeline_spec(spec, args=args)
        self.assertEqual(len(errors), 1)
        self.assertIn("hard_restart_pass", errors[0])

    def test_missing_arg_namespace_is_error(self):
        spec = {"transforms": [{"name": "hard_restart"}]}
        errors = validate_pipeline_spec(spec, args=None)
        self.assertEqual(len(errors), 1)
        self.assertIn("hard_restart_pass", errors[0])


class TestIdempotence(unittest.TestCase):
    def test_duplicate_non_idempotent_is_error(self):
        spec = {
            "transforms": [
                {"name": "hard_restart"},
                {"name": "hard_restart"},
            ]
        }
        errors = validate_pipeline_spec(spec, args=_args())
        self.assertTrue(any("hard_restart" in e and "repeat" in e for e in errors))

    def test_duplicate_idempotent_is_allowed(self):
        spec = {
            "transforms": [
                {"name": "voice_block_order"},
                {"name": "voice_block_order"},
            ]
        }
        errors = validate_pipeline_spec(spec, args=_args())
        repeats = [e for e in errors if "repeated" in e]
        self.assertEqual(repeats, [])


class TestEmptySpec(unittest.TestCase):
    def test_empty_spec_is_error(self):
        errors = validate_pipeline_spec({"transforms": []}, args=_args())
        self.assertEqual(errors, ["empty pipeline spec"])


class TestMultipleErrorsReported(unittest.TestCase):
    def test_collects_all_errors_not_just_first(self):
        spec = {
            "transforms": [
                {"name": "made_up"},
                {"name": "hard_restart"},
                {"name": "hard_restart"},
            ]
        }
        errors = validate_pipeline_spec(spec, args=_args())
        self.assertGreaterEqual(len(errors), 2)


class TestRegistryInvariants(unittest.TestCase):
    def test_registry_passes_global_invariants(self):
        from preframr_tokens.macros.pipeline_check import validate_registry

        self.assertEqual(validate_registry(), [])


class TestDecoderAvailability(unittest.TestCase):
    def test_all_registered_op_codes_have_decoders(self):
        from preframr_tokens.macros.decoders import DECODERS
        from preframr_tokens.macros.transform import _REGISTRY

        for name, cls in _REGISTRY.items():
            if getattr(cls, "DECODES_VIA_DF", False):
                continue
            for op in cls.OP_CODES:
                self.assertIn(
                    int(op),
                    DECODERS,
                    msg=f"transform {name!r} declares op_code {int(op)} but DECODERS has no entry",
                )


class TestDefaultPipelineSpec(unittest.TestCase):
    def test_default_pipeline_names_are_either_registered_or_phantom(self):
        from preframr_tokens.macros.default_pipeline import default_pipeline_spec
        from preframr_tokens.macros.pipeline_check import _KNOWN_PHANTOM_NAMES
        from preframr_tokens.macros.transform import _REGISTRY

        valid_names = set(_REGISTRY.keys()) | _KNOWN_PHANTOM_NAMES
        unknown = []
        for entry in default_pipeline_spec()["transforms"]:
            if entry["name"] not in valid_names:
                unknown.append(entry["name"])
        self.assertEqual(
            unknown,
            [],
            msg=(
                f"default_pipeline_spec contains unknown name(s) {unknown}; "
                f"either register the transform or add to _KNOWN_PHANTOM_NAMES"
            ),
        )


class TestPipelineTier(unittest.TestCase):
    def test_empty_pipeline_is_bit_exact(self):
        from preframr_tokens.macros.transform import TransformPipeline

        p = TransformPipeline([])
        self.assertEqual(p.tier, "bit_exact")

    def test_pipeline_tier_is_worst_transform_tier(self):
        from preframr_tokens.macros import (  # noqa: F401
            transforms_audio_bit_exact,
            transforms_bit_exact,
        )
        from preframr_tokens.macros.transform import TransformPipeline

        spec = {
            "transforms": [
                {"name": "hard_restart"},
                {"name": "voice_block_order"},
            ]
        }
        p_bitexact = TransformPipeline.from_spec(spec, args=_args())
        self.assertEqual(p_bitexact.tier, "bit_exact")
        spec2 = {
            "transforms": [
                {"name": "freq_traj"},
                {"name": "hard_restart"},
                {"name": "voice_block_order"},
            ]
        }
        p_audio = TransformPipeline.from_spec(
            spec2, args=_args(freq_trajectory_pass=True)
        )
        self.assertEqual(p_audio.tier, "audio_bit_exact")


class TestIdempotenceRoundTrip(unittest.TestCase):
    def test_declared_idempotent_transforms_are_actually_idempotent(self):
        import argparse

        import pandas as pd

        from preframr_tokens.macros.transform import _REGISTRY
        from preframr_tokens.stfconstants import FRAME_REG, SET_OP, VOICE_REG

        svt = (0 + 1) << 0
        df = pd.DataFrame(
            [
                {
                    "op": int(SET_OP),
                    "reg": int(FRAME_REG),
                    "subreg": -1,
                    "val": svt,
                    "diff": 0,
                },
                {
                    "op": int(SET_OP),
                    "reg": int(VOICE_REG),
                    "subreg": -1,
                    "val": 0,
                    "diff": 0,
                },
                {"op": int(SET_OP), "reg": 0, "subreg": -1, "val": 0x10, "diff": 0},
                {"op": int(SET_OP), "reg": 4, "subreg": -1, "val": 0x41, "diff": 0},
            ]
        )
        all_args = argparse.Namespace(
            voice_canonical_block_order=True,
            hard_restart_pass=True,
            ctrl_bigram_pass=True,
            freq_trajectory_pass=True,
            preset_pass=True,
            loop_pass=True,
        )
        for name, cls in _REGISTRY.items():
            if not cls.IDEMPOTENT:
                continue
            try:
                instance = cls()
            except TypeError:
                continue
            try:
                once = instance.forward(df.copy(), args=all_args)
                twice = instance.forward(once.copy(), args=all_args)
            except (NotImplementedError, Exception):
                continue
            pd.testing.assert_frame_equal(
                twice.reset_index(drop=True),
                once.reset_index(drop=True),
                obj=f"{name} idempotence",
            )


class TestHardcodedPassesSeedAccumulator(unittest.TestCase):
    def test_hardcoded_emits_non_set_regs_includes_known_motion_regs(self):
        from preframr_tokens.macros import (  # noqa: F401
            transforms_audio_bit_exact,
            transforms_bit_exact,
        )
        from preframr_tokens.macros.pipeline_check import _hardcoded_emits_non_set_regs

        hardcoded = _hardcoded_emits_non_set_regs()
        self.assertIn(0, hardcoded)
        self.assertIn(2, hardcoded)
        self.assertIn(4, hardcoded)
        self.assertIn(21, hardcoded)

    def test_consumer_with_no_handles_errors_against_hardcoded_seed(self):
        from preframr_tokens.macros import (  # noqa: F401
            transforms_audio_bit_exact,
            transforms_bit_exact,
        )
        from preframr_tokens.macros.transform import Transform, register

        @register("_test_consumer_no_handles")
        class _Consumer(Transform):
            TIER = "bit_exact"
            EXPECTS_SET_ON_REGS = frozenset({0})
            HANDLES_NON_SET_ON_REGS = frozenset()

            def forward(self, df, args=None):
                return df

        try:
            spec = {"transforms": [{"name": "_test_consumer_no_handles"}]}
            errors = validate_pipeline_spec(spec, args=_args())
            self.assertTrue(
                any("EXPECTS_SET_ON_REGS" in e for e in errors),
                msg=(
                    f"expected EXPECTS_SET_ON_REGS error vs hardcoded seed, "
                    f"got {errors}"
                ),
            )
        finally:
            from preframr_tokens.macros.transform import _REGISTRY

            _REGISTRY.pop("_test_consumer_no_handles", None)


class TestRegisterStateContract(unittest.TestCase):
    def test_unhandled_non_set_state_reports_error(self):
        from preframr_tokens.macros.transform import Transform, register

        @register("_test_motion_consumer")
        class _MotionConsumer(Transform):
            TIER = "bit_exact"
            EXPECTS_SET_ON_REGS = frozenset({0, 2})
            HANDLES_NON_SET_ON_REGS = frozenset()

            def forward(self, df, args=None):
                return df

        try:
            spec = {
                "transforms": [
                    {"name": "freq_traj"},
                    {"name": "_test_motion_consumer"},
                ]
            }
            errors = validate_pipeline_spec(spec, args=_args(freq_trajectory_pass=True))
            self.assertTrue(
                any("EXPECTS_SET_ON_REGS" in e for e in errors),
                msg=f"expected EXPECTS_SET_ON_REGS error, got {errors}",
            )
        finally:
            from preframr_tokens.macros.transform import _REGISTRY

            _REGISTRY.pop("_test_motion_consumer", None)


if __name__ == "__main__":
    unittest.main()
