"""Tests for preframr_tokens.tokenizer_config: the one-source-of-truth args
builder has every pass flag, the presets parse, and RegLogParser accepts it."""

import unittest

from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import (
    MACRO_FLAGS,
    PARSER_DEFAULTS,
    REGISTERED_MACROS,
    default_pipeline_args,
    default_tokenizer_args,
    named_config,
)


class TestTokenizerConfig(unittest.TestCase):
    def test_default_builder_is_additive_all_off(self):
        ns = default_tokenizer_args()
        for flag in MACRO_FLAGS:
            self.assertIs(getattr(ns, flag), False, flag)
        for key in PARSER_DEFAULTS:
            self.assertTrue(hasattr(ns, key), key)

    def test_default_pipeline_is_the_generator_pipeline(self):
        ns = default_pipeline_args()
        on = {f for f in MACRO_FLAGS if getattr(ns, f) is True}
        self.assertEqual(on, set(REGISTERED_MACROS))
        for key in PARSER_DEFAULTS:
            self.assertTrue(hasattr(ns, key), key)

    def test_overrides_win(self):
        ns = default_tokenizer_args(cents=99, freq_trajectory_pass=True)
        self.assertEqual(ns.cents, 99)
        self.assertIs(ns.freq_trajectory_pass, True)

    def test_named_baseline_all_off(self):
        ns = named_config("baseline")
        self.assertTrue(all(getattr(ns, f) is False for f in MACRO_FLAGS))

    def test_named_full_macros_is_registered_set(self):
        ns = named_config("full_macros")
        on = {f for f in MACRO_FLAGS if getattr(ns, f) is True}
        self.assertEqual(on, set(REGISTERED_MACROS))
        self.assertTrue(set(REGISTERED_MACROS).issubset(MACRO_FLAGS))

    def test_named_override(self):
        ns = named_config("full_macros", generator_pass=False)
        self.assertIs(ns.generator_pass, False)
        self.assertIs(ns.instrument_program, True)

    def test_named_unknown_raises(self):
        with self.assertRaises(KeyError):
            named_config("does_not_exist")

    def test_reglogparser_accepts_config(self):
        RegLogParser(args=default_tokenizer_args())
        RegLogParser(args=named_config("full_macros"))


if __name__ == "__main__":
    unittest.main()
