"""flag_registry derives MACRO_FLAGS from the passes: surfaced flags present,
dead flags gone, and a drift guard that every boolean ``getattr(args, ...)`` a
pass reads is actually declared (so adding a pass-flag without a GATE_FLAGS /
REQUIRES_ARGS entry fails CI instead of silently missing from MACRO_FLAGS)."""

import pathlib
import re
import unittest

import preframr_tokens
from preframr_tokens.macros.flag_registry import macro_flag_names
from preframr_tokens.tokenizer_config import (
    MACRO_FLAGS,
    PARSER_DEFAULTS,
    REGISTERED_MACROS,
)

_BOOL_GETATTR = re.compile(
    r'getattr\(\s*args\s*,\s*"([a-z_]+)"\s*,\s*(?:True|False)\s*\)'
)


def _source_flags() -> set[str]:
    root = pathlib.Path(preframr_tokens.__file__).parent
    flags: set[str] = set()
    for path in root.rglob("*.py"):
        flags |= set(_BOOL_GETATTR.findall(path.read_text()))
    return flags - set(PARSER_DEFAULTS)


class TestFlagRegistry(unittest.TestCase):
    def test_macro_flags_is_sorted_derived(self):
        self.assertEqual(MACRO_FLAGS, tuple(sorted(macro_flag_names())))
        self.assertTrue(MACRO_FLAGS)

    def test_registered_subset(self):
        self.assertTrue(set(REGISTERED_MACROS).issubset(MACRO_FLAGS))

    def test_surfaced_flags_present(self):
        for flag in (
            "gate_slope_shift_pass",
            "voice_track_pass",
        ):
            self.assertIn(flag, MACRO_FLAGS, flag)

    def test_dead_flags_absent(self):
        for flag in (
            "mode_vol_flip_pass",
            "legato_pass_c3",
            "super_frame_pass",
            "voice_trajectory_pass",
            "voice_trajectory_distributed_pass",
            "set_to_diff_pass",
            "motif_pass",
            "freq_nudge_pass",
            "freq_onset_pass",
            "release_update_pass",
            "lonely_catch_all",
            "strict_lonely",
        ):
            self.assertNotIn(flag, MACRO_FLAGS, flag)

    def test_no_undeclared_gating_flag(self):
        undeclared = _source_flags() - set(MACRO_FLAGS)
        self.assertEqual(
            undeclared,
            set(),
            f"passes read these boolean args but no GATE_FLAGS / REQUIRES_ARGS "
            f"declares them, so they are missing from MACRO_FLAGS: {sorted(undeclared)}",
        )


if __name__ == "__main__":
    unittest.main()
