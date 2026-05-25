"""Tests for ``preframr_tokens.reglogparser.read_initial_irq``."""

from __future__ import annotations

import unittest

import pandas as pd

from preframr_tokens.reglogparser import read_initial_irq
from preframr_tokens.stfconstants import DEFAULT_IRQ_CYCLES, DELAY_REG, FRAME_REG


class TestReadInitialIrq(unittest.TestCase):
    def test_picks_first_frame_diff(self):
        df = pd.DataFrame(
            [
                {"reg": 0, "diff": 1000},
                {"reg": FRAME_REG, "diff": 19656},
                {"reg": 1, "diff": 5},
                {"reg": FRAME_REG, "diff": 19700},
            ]
        )
        self.assertEqual(read_initial_irq(df), 19656)

    def test_skips_zero_first_frame_diff(self):
        df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "diff": 0},
                {"reg": 0, "diff": 5},
                {"reg": FRAME_REG, "diff": 19656},
            ]
        )
        self.assertEqual(read_initial_irq(df), 19656)

    def test_returns_default_when_all_frame_diffs_zero(self):
        df = pd.DataFrame(
            [{"reg": FRAME_REG, "diff": 0}, {"reg": FRAME_REG, "diff": 0}]
        )
        self.assertEqual(read_initial_irq(df), DEFAULT_IRQ_CYCLES)

    def test_returns_default_when_no_frame_rows(self):
        df = pd.DataFrame([{"reg": 0, "diff": 12}, {"reg": DELAY_REG, "diff": 8}])
        self.assertEqual(read_initial_irq(df), DEFAULT_IRQ_CYCLES)

    def test_caller_provided_default(self):
        df = pd.DataFrame([{"reg": 0, "diff": 12}])
        self.assertEqual(read_initial_irq(df, default=4242), 4242)

    def test_default_is_pal_irq(self):
        self.assertEqual(DEFAULT_IRQ_CYCLES, 19656)


if __name__ == "__main__":
    unittest.main()
