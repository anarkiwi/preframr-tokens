"""Tests for ``preframr_tokens.tier_classify``."""

from __future__ import annotations

import argparse
import unittest

import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FILTER_REG,
    FRAME_REG,
    MODE_VOL_REG,
    MODEL_PDTYPE,
    PAD_REG,
    SET_OP,
    VOICE_CTRL_REG,
)
from preframr_tokens.tier_classify import (
    CONTENT_TIER,
    build_vocab_tier_ids,
    build_vocab_tier_map,
    vocab_id_tier,
)


def _tiny_args():
    return argparse.Namespace(tkvocab=0, tkmodel=None, tokenizer="unigram")


def _tokens(rows):
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


def _rt(tokens_df):
    return RegTokenizer(_tiny_args(), tokens=tokens_df)


class TestVocabIdTier(unittest.TestCase):
    def _row(self, op=SET_OP, reg=0, subreg=-1, val=0):
        return {"op": op, "reg": reg, "subreg": subreg, "val": val, "n": 0}

    def test_frame_reg_is_structural(self):
        tokens = _tokens([self._row(reg=FRAME_REG, val=1)])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), "structural")

    def test_delay_reg_is_mid(self):
        tokens = _tokens([self._row(reg=DELAY_REG, val=2)])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), "mid")

    def test_filter_reg_is_zero(self):
        tokens = _tokens([self._row(reg=FILTER_REG)])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), "zero")

    def test_mode_vol_reg_is_zero(self):
        tokens = _tokens([self._row(reg=MODE_VOL_REG)])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), "zero")

    def test_voice_ctrl_reg_is_mid(self):
        ctrl_reg = next(iter(VOICE_CTRL_REG.values()))
        tokens = _tokens([self._row(reg=int(ctrl_reg))])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), "mid")

    def test_unmapped_reg_falls_through_to_content(self):
        tokens = _tokens([self._row(reg=0, val=5)])
        self.assertEqual(vocab_id_tier(0, _rt(tokens), tokens), CONTENT_TIER)


class TestBuildVocabTierIds(unittest.TestCase):
    def test_returns_int64_array_of_expected_length(self):
        tokens = _tokens(
            [
                {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "n": 0},
                {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 2, "n": 0},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 0},
            ]
        )
        arr = build_vocab_tier_ids(_rt(tokens), tokens, n_vocab=4)
        from preframr_tokens.stfconstants import LOSS_TIER_NAMES

        order = {n: i for i, n in enumerate(LOSS_TIER_NAMES)}
        self.assertEqual(arr.dtype.kind, "i")
        self.assertEqual(arr.shape, (4,))
        self.assertEqual(int(arr[1]), order["structural"])
        self.assertEqual(int(arr[2]), order["mid"])
        self.assertEqual(int(arr[3]), order["content"])

    def test_empty_tokens_defaults_to_content(self):
        order_content = 2
        arr = build_vocab_tier_ids(None, None, n_vocab=5)
        self.assertTrue((arr == order_content).all())


class TestBuildVocabTierMap(unittest.TestCase):
    def test_map_keys_are_vocab_ids_values_are_tier_names(self):
        tokens = _tokens(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 1, "n": 0},
                {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 2, "n": 0},
            ]
        )
        m = build_vocab_tier_map(_rt(tokens), tokens, n_vocab=2)
        self.assertEqual(m, {0: "structural", 1: "mid"})

    def test_empty_tokens_all_content(self):
        m = build_vocab_tier_map(None, None, n_vocab=3)
        self.assertEqual(m, {0: "content", 1: "content", 2: "content"})


class TestGeneratorOpsTiered(unittest.TestCase):
    """The generator/codebook ops are MacroPass-emitted (no Transform class), so they must be tiered via
    op_contracts.MACRO_OP_LOSS_TIERS -- not silently default to content, which would let the
    content_over_structural gate count codebook + tuning structure as content."""

    def test_every_generator_op_has_an_explicit_tier(self):
        from preframr_tokens.macros.flag_registry import ensure_passes_registered
        from preframr_tokens.macros.transform import collect_op_loss_tiers
        from preframr_tokens.stfconstants import (
            GEN_TABLE_DEF_OP,
            GEN_TABLE_END_OP,
            GEN_TABLE_REF_OP,
            GEN_TABLE_STEP_OP,
            GEN_TRI_OP,
            GEN_TUNING_OP,
            MELODY_INTERVAL_OP,
            SWEEP_OP,
        )

        ensure_passes_registered()
        tiers = collect_op_loss_tiers()
        for op in (
            SWEEP_OP,
            GEN_TRI_OP,
            GEN_TABLE_DEF_OP,
            GEN_TABLE_STEP_OP,
            GEN_TABLE_END_OP,
            GEN_TABLE_REF_OP,
            GEN_TUNING_OP,
            MELODY_INTERVAL_OP,
        ):
            self.assertIn(int(op), tiers, f"op {int(op)} silently defaults to content")
        self.assertEqual(tiers[int(GEN_TABLE_DEF_OP)], "structural")
        self.assertEqual(tiers[int(GEN_TUNING_OP)], "structural")
        self.assertEqual(tiers[int(MELODY_INTERVAL_OP)], "content")


if __name__ == "__main__":
    unittest.main()
