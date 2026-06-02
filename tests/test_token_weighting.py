"""Tests for ``preframr_tokens.token_weighting.vocab_frame_weights``."""

from __future__ import annotations

import argparse
import unittest

import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    DELAY_REG,
    DO_LOOP_OP,
    FRAME_REG,
    MODEL_PDTYPE,
    PAD_REG,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_LEN,
    SET_OP,
)
from preframr_tokens.token_weighting import vocab_frame_weights


def _tiny_args():
    return argparse.Namespace(tkvocab=0, tkmodel=None, tokenizer="unigram")


def _tokens(rows):
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


def _rt(tokens_df):
    return RegTokenizer(_tiny_args(), tokens=tokens_df)


class TestVocabFrameWeights(unittest.TestCase):
    def test_returns_float32_of_n_vocab(self):
        tokens = _tokens(
            [{"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0, "n": 0}]
        )
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=4)
        self.assertEqual(w.shape, (4,))
        self.assertEqual(w.dtype.name, "float32")

    def test_default_weight_is_one(self):
        tokens = _tokens([{"op": SET_OP, "reg": 0, "subreg": -1, "val": 5, "n": 0}])
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=1)
        self.assertEqual(float(w[0]), 1.0)

    def test_frame_reg_weighs_one(self):
        tokens = _tokens(
            [{"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0, "n": 0}]
        )
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=1)
        self.assertEqual(float(w[0]), 1.0)

    def test_delay_reg_weighs_val(self):
        tokens = _tokens(
            [{"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 7, "n": 0}]
        )
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=1)
        self.assertEqual(float(w[0]), 7.0)

    def test_pattern_replay_len_subreg_weighs_val(self):
        tokens = _tokens(
            [
                {
                    "op": PATTERN_REPLAY_OP,
                    "reg": 0,
                    "subreg": PATTERN_REPLAY_SUBREG_LEN,
                    "val": 4,
                    "n": 0,
                }
            ]
        )
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=1)
        self.assertEqual(float(w[0]), 4.0)

    def test_do_loop_subreg0_weighs_val(self):
        tokens = _tokens([{"op": DO_LOOP_OP, "reg": 0, "subreg": 0, "val": 12, "n": 0}])
        w = vocab_frame_weights(_rt(tokens), tokens, n_vocab=1)
        self.assertEqual(float(w[0]), 12.0)

    def test_empty_tokens_returns_ones(self):
        w = vocab_frame_weights(None, None, n_vocab=5)
        self.assertEqual(w.shape, (5,))
        self.assertTrue((w == 1.0).all())


if __name__ == "__main__":
    unittest.main()
