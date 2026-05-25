"""Tests for preframr_tokens.tokenizer_config: the one-source-of-truth args
builder has every pass flag, the presets parse, and RegLogParser accepts it."""

import unittest

from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import (
    MACRO_FLAGS,
    PARSER_DEFAULTS,
    default_tokenizer_args,
    named_config,
)


class TestTokenizerConfig(unittest.TestCase):
    def test_default_has_all_flags_and_params(self):
        ns = default_tokenizer_args()
        for flag in MACRO_FLAGS:
            self.assertIs(getattr(ns, flag), False, flag)
        for key in PARSER_DEFAULTS:
            self.assertTrue(hasattr(ns, key), key)

    def test_overrides_win(self):
        ns = default_tokenizer_args(cents=99, freq_trajectory_pass=True)
        self.assertEqual(ns.cents, 99)
        self.assertIs(ns.freq_trajectory_pass, True)

    def test_named_baseline_all_off(self):
        ns = named_config("baseline")
        self.assertTrue(all(getattr(ns, f) is False for f in MACRO_FLAGS))

    def test_named_full_macros_all_on(self):
        ns = named_config("full_macros")
        self.assertTrue(all(getattr(ns, f) is True for f in MACRO_FLAGS))

    def test_named_override(self):
        ns = named_config("full_macros", freq_trajectory_pass=False)
        self.assertIs(ns.freq_trajectory_pass, False)
        self.assertIs(ns.preset_pass, True)

    def test_named_unknown_raises(self):
        with self.assertRaises(KeyError):
            named_config("does_not_exist")

    def test_reglogparser_accepts_config(self):
        RegLogParser(args=default_tokenizer_args())
        RegLogParser(args=named_config("full_macros"))


if __name__ == "__main__":
    unittest.main()
