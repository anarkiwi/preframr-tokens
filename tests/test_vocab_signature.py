"""Tests for ``preframr_tokens.vocab_signature``.

These cover the load-bearing contract that the wrappers in
``tier_classify.py`` and ``token_weighting.py`` rely on but don't
themselves exercise: under Unigram sub-tokens, the tier is taken from
the FIRST atomic id, and the frame weight ACCUMULATES across atomic
ids.
"""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    MODEL_PDTYPE,
    PAD_REG,
    SET_OP,
)
from preframr_tokens.vocab_signature import CONTENT_TIER, VocabSignature


def _tiny_args():
    return argparse.Namespace(tkvocab=0, tkmodel=None, tokenizer="unigram")


def _tokens(rows):
    return pd.DataFrame(rows, dtype=MODEL_PDTYPE)


class _FakeTkModel:
    """Minimal tokenizer model: each sub-id maps to a tuple of atomic ids."""

    def __init__(self, rtok: RegTokenizer, sub_atomic_lists):
        # Position 0 is reserved for an <unk> sentinel; tests don't index it.
        self._sub_atomic_lists = [None] + list(sub_atomic_lists)
        self._sub_strs = [
            (
                rtok.encode_unicode(np.asarray(atomics, dtype=np.uint32))
                if atomics is not None
                else "<unk>"
            )
            for atomics in self._sub_atomic_lists
        ]

    def get_vocab_size(self):
        return len(self._sub_atomic_lists)

    def id_to_token(self, sub_id):
        if sub_id < 0 or sub_id >= len(self._sub_strs):
            return None
        return self._sub_strs[sub_id]

    def decode(self, encoded_ids):
        # Single id: return its unicode string. encoded_ids is a list-like.
        return "".join(self._sub_strs[int(i)] for i in encoded_ids)


def _rt_atomic(tokens_df: pd.DataFrame) -> RegTokenizer:
    return RegTokenizer(_tiny_args(), tokens=tokens_df)


def _rt_subtoken(tokens_df: pd.DataFrame, sub_atomic_lists) -> RegTokenizer:
    rt = RegTokenizer(_tiny_args(), tokens=tokens_df)
    rt.splitters = min(rt.splitters, int((tokens_df["reg"] == FRAME_REG).sum()))
    rt.tkmodel = _FakeTkModel(rt, sub_atomic_lists)
    return rt


class TestVocabSignatureAtomic(unittest.TestCase):
    def test_empty_tokens(self):
        sig = VocabSignature(_rt_atomic(_tokens([])), _tokens([]), n_vocab=3)
        np.testing.assert_array_equal(sig.frame_weights, np.ones(3, dtype=np.float32))
        # All tier_names default to CONTENT_TIER.
        self.assertEqual(set(sig.tier_names.values()), {CONTENT_TIER})

    def test_atomic_frame_and_delay_weights(self):
        # Vocab: [pad, FRAME, DELAY=5, content]. FRAME→+1.0, DELAY→+5, content→1.0 default.
        tokens = _tokens(
            [
                {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0, "n": 1},
                {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 5, "n": 2},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7, "n": 3},
            ]
        )
        sig = VocabSignature(_rt_atomic(tokens), tokens, n_vocab=4)
        # FRAME id contributes +1.0, DELAY id contributes +val=5, content stays at default 1.0.
        self.assertEqual(float(sig.frame_weights[1]), 1.0)
        self.assertEqual(float(sig.frame_weights[2]), 5.0)
        self.assertEqual(float(sig.frame_weights[3]), 1.0)


class TestVocabSignatureSubtoken(unittest.TestCase):
    def _atomic_vocab(self):
        # Index 0=pad, 1=FRAME, 2=DELAY(val=5), 3=content reg=0.
        return _tokens(
            [
                {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0, "n": 1},
                {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 5, "n": 2},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7, "n": 3},
            ]
        )

    def test_subtoken_first_atomic_tier_wins(self):
        """A sub-token decoding to [FRAME (structural), CONTENT_REG (content)]
        should be classified by the FIRST atomic id — structural."""
        tokens = self._atomic_vocab()
        # sub-id 1 → [FRAME, content]; sub-id 2 → [content, FRAME] (reversed).
        rt = _rt_subtoken(tokens, sub_atomic_lists=[[1, 3], [3, 1]])
        sig = VocabSignature(rt, tokens, n_vocab=rt.tkmodel.get_vocab_size())
        # sub-id 0 is the <unk> sentinel; sub-id 1 starts with FRAME → "structural".
        self.assertEqual(sig.tier_names[1], "structural")
        # sub-id 2 starts with content reg → CONTENT_TIER.
        self.assertEqual(sig.tier_names[2], CONTENT_TIER)

    def test_subtoken_weights_accumulate_across_atomics(self):
        """A sub-token containing FRAME and DELAY atomics should sum the
        two contributions (FRAME +1.0, DELAY +val=5 → total 6.0)."""
        tokens = self._atomic_vocab()
        rt = _rt_subtoken(tokens, sub_atomic_lists=[[1, 2], [2, 1]])
        sig = VocabSignature(rt, tokens, n_vocab=rt.tkmodel.get_vocab_size())
        self.assertEqual(float(sig.frame_weights[1]), 6.0)
        self.assertEqual(float(sig.frame_weights[2]), 6.0)

    def test_subtoken_only_content_atomics_default_weight(self):
        tokens = self._atomic_vocab()
        rt = _rt_subtoken(tokens, sub_atomic_lists=[[3, 3]])
        sig = VocabSignature(rt, tokens, n_vocab=rt.tkmodel.get_vocab_size())
        # No FRAME / DELAY / weighted-op contributions → stays at default 1.0.
        self.assertEqual(float(sig.frame_weights[1]), 1.0)


class TestVocabSignatureConsistency(unittest.TestCase):
    def test_tier_ids_and_tier_names_agree(self):
        tokens = _tokens(
            [
                {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0, "n": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0, "n": 1},
                {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 1, "n": 2},
                {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7, "n": 3},
            ]
        )
        sig = VocabSignature(_rt_atomic(tokens), tokens, n_vocab=4)
        for vid, name in sig.tier_names.items():
            self.assertEqual(sig.tier_order[int(sig.tier_ids[vid])], name)


if __name__ == "__main__":
    unittest.main()
