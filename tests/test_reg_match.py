"""Tests for ``preframr_tokens.reg_match.reg_class`` and its agreement with
the boolean ``*_match`` predicates."""

from __future__ import annotations

import unittest

import pandas as pd

from preframr_tokens.reg_match import (
    ctrl_match,
    freq_match,
    pcm_match,
    reg_class,
)
from preframr_tokens.stfconstants import VOICES, VOICE_REG_SIZE


class TestRegClass(unittest.TestCase):
    def test_known_offsets_across_voices(self):
        for v in range(VOICES):
            base = v * VOICE_REG_SIZE
            self.assertEqual(reg_class(base + 0), ("FREQ", v))
            self.assertEqual(reg_class(base + 2), ("PW", v))
            self.assertEqual(reg_class(base + 4), ("CTRL", v))
            self.assertEqual(reg_class(base + 5), ("AD", v))
            self.assertEqual(reg_class(base + 6), ("SR", v))

    def test_unclassified_offset_is_none(self):
        self.assertIsNone(reg_class(1))
        self.assertIsNone(reg_class(3))

    def test_out_of_range_voice_is_none(self):
        self.assertIsNone(reg_class(VOICES * VOICE_REG_SIZE))

    def test_negative_is_none(self):
        self.assertIsNone(reg_class(-1))

    def test_accepts_numpy_int_like(self):
        self.assertEqual(reg_class(float(VOICE_REG_SIZE + 4)), ("CTRL", 1))

    def test_agrees_with_boolean_matchers(self):
        regs = list(range(VOICES * VOICE_REG_SIZE))
        df = pd.DataFrame({"reg": regs})
        for pred, kind in (
            (freq_match, "FREQ"),
            (pcm_match, "PW"),
            (ctrl_match, "CTRL"),
        ):
            mask = pred(df).to_numpy()
            for reg in regs:
                cls = reg_class(reg)
                selected = cls is not None and cls[0] == kind
                self.assertEqual(
                    bool(mask[reg]), selected, msg=f"reg={reg} kind={kind}"
                )


if __name__ == "__main__":
    unittest.main()
