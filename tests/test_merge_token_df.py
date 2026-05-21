"""Unit tests for ``RegTokenizer.merge_token_df`` substitution behaviour."""

import unittest

import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    DIFF_OP,
    FRAME_REG,
    LOOP_OP_REG,
    MODEL_PDTYPE,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    SET_OP,
)


class FakeArgs:
    def __init__(self):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = 128
        self.tkvocab = 0
        self.tkmodel = None
        self.max_files = 1
        self.diffq = 64
        self.tokenizer = "bpe"


def _tokens(rows):
    """Build a tokens df from a list of (op, reg, subreg, val) tuples."""
    return pd.DataFrame(
        [
            {
                "op": op,
                "reg": reg,
                "subreg": subreg,
                "val": val,
                "n": n,
                "count": 1,
            }
            for n, (op, reg, subreg, val) in enumerate(rows)
        ],
        dtype=MODEL_PDTYPE,
    )


def _df(rows, frame_diff=19000):
    """Build an input df. Always prefixes a FRAME_REG row so
    ``_merged_and_missing`` has its irq pivot."""
    out = [
        {
            "op": SET_OP,
            "reg": FRAME_REG,
            "subreg": -1,
            "val": 0,
            "diff": frame_diff,
            "description": 0,
        }
    ]
    for op, reg, subreg, val in rows:
        out.append(
            {
                "op": op,
                "reg": reg,
                "subreg": subreg,
                "val": val,
                "diff": 32,
                "description": 0,
            }
        )
    return pd.DataFrame(out, dtype=MODEL_PDTYPE)


class TestMergeTokenDfPassThrough(unittest.TestCase):
    def test_complete_alphabet_passes_through(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
                (SET_OP, 1, -1, 7),
            ]
        )
        df = _df([(SET_OP, 1, -1, 5), (SET_OP, 1, -1, 7)])
        result = loader.merge_token_df(tokens, df)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertFalse(result["n"].isna().any())


class TestSetOpSubstitution(unittest.TestCase):
    """SET op has a continuous val space; nearest-val substitution within
    the same (op, reg, subreg) is allowed."""

    def test_substitutes_nearest_val_in_same_op_reg_subreg(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
                (SET_OP, 1, -1, 9),
            ]
        )
        df = _df([(SET_OP, 1, -1, 7)])
        result = loader.merge_token_df(tokens, df)
        self.assertIsNotNone(result)
        self.assertFalse(result["n"].isna().any())
        substituted_val = int(
            result[(result["reg"] == 1) & (result["op"] == SET_OP)]["val"].iloc[0]
        )
        self.assertIn(substituted_val, (5, 9))

    def test_does_not_substitute_across_op_boundary(self):
        """A row with op=DIFF must not be substituted from SET tokens at
        the same reg, even if the nearest val is shared."""
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 1, -1, 5),
            ]
        )
        df = _df([(DIFF_OP, 1, -1, 5)])
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)


class TestMacroOpsRefuseSubstitution(unittest.TestCase):
    """Macro ops carry categorical val (back-ref distance, palette slot,
    program length, etc.); near-val substitution would silently corrupt
    the encoding. The substitution path raises KeyError instead."""

    def _assert_refuses(self, missing_row):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, missing_row[1], -1, 5),
                (SET_OP, missing_row[1], -1, 9),
            ]
        )
        df = _df([missing_row])
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)

    def test_back_ref_op_refuses(self):
        self._assert_refuses((BACK_REF_OP, LOOP_OP_REG, -1, 1234))

    def test_pattern_replay_op_refuses(self):
        self._assert_refuses((PATTERN_REPLAY_OP, LOOP_OP_REG, 2, 5678))

    def test_pattern_overlay_op_refuses(self):
        self._assert_refuses((PATTERN_OVERLAY_OP, LOOP_OP_REG, 0, 4096))


class TestSubstitutionRespectsSubreg(unittest.TestCase):
    """Two SET rows on the same reg with different subregs are distinct
    tokens; a missing (op, reg, subreg=A) must not substitute from
    (op, reg, subreg=B)."""

    def test_subreg_zero_does_not_match_subreg_one(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (SET_OP, 4, 1, 5),
            ]
        )
        df = _df([(SET_OP, 4, 0, 5)])
        with self.assertRaises(KeyError):
            loader.merge_token_df(tokens, df)


if __name__ == "__main__":
    unittest.main()
