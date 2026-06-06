"""Regression (2026-06-03 codebook-id-snap): codebook ids are define->ref pointers,
not magnitudes -- ``project_df`` and ``merge_token_df`` must not value-snap them (that
silently rebinds the reference), and ``make_tokens`` densely enumerates the id ordinal
range so an eval tune's tune-local id is always in the alphabet (overflow raises loud).
"""

import unittest

import numpy as np
import pandas as pd

from preframr_tokens.alphabet_projection import project_df
from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    FRAME_REG,
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_ID,
    INSTR_DEF_OP,
    MODEL_PDTYPE,
    SET_OP,
)


class FakeArgs:
    def __init__(self):
        self.reglogs = ""
        self.seq_len = 128
        self.tkvocab = 0
        self.tkmodel = None
        self.tokenizer = "bpe"


def _tokens(rows):
    return pd.DataFrame(
        [
            {"op": op, "reg": reg, "subreg": subreg, "val": val, "n": n, "count": 1}
            for n, (op, reg, subreg, val) in enumerate(rows)
        ],
        dtype=MODEL_PDTYPE,
    )


def _df(rows):
    out = [
        {
            "op": SET_OP,
            "reg": FRAME_REG,
            "subreg": -1,
            "val": 0,
            "diff": 19000,
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


class TestProjectDfSkipsCodebookIds(unittest.TestCase):
    def test_codebook_id_passes_through_content_snaps(self):
        table = {
            (INSTR_DEF_OP, 4, -1): np.array([0, 1, 2], dtype=np.int64),
            (SET_OP, 21, -1): np.array([64], dtype=np.int64),
        }
        df = pd.DataFrame(
            {
                "op": [INSTR_DEF_OP, SET_OP],
                "reg": [4, 21],
                "subreg": [-1, -1],
                "val": [5, 50],
            }
        )
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 5)
        self.assertEqual(int(out["val"].iloc[1]), 64)

    def test_rel_ref_id_subreg_skips_but_base_subreg_snaps(self):
        table = {
            (GEN_TABLE_REF_OP, 0, GEN_TABLE_REF_SUBREG_ID): np.array(
                [0, 1], dtype=np.int64
            ),
            (GEN_TABLE_REF_OP, 0, 2): np.array([100], dtype=np.int64),
        }
        df = pd.DataFrame(
            {
                "op": [GEN_TABLE_REF_OP, GEN_TABLE_REF_OP],
                "reg": [0, 0],
                "subreg": [GEN_TABLE_REF_SUBREG_ID, 2],
                "val": [7, 50],
            }
        )
        out = project_df(df, table)
        self.assertEqual(int(out["val"].iloc[0]), 7)
        self.assertEqual(int(out["val"].iloc[1]), 100)


class TestMakeTokensCodebookIdCoverage(unittest.TestCase):
    def test_enumerates_id_range_leaves_content(self):
        tk = RegTokenizer(FakeArgs(), tokens=None)
        tokens = pd.DataFrame(
            [
                {
                    "op": INSTR_DEF_OP,
                    "reg": 4,
                    "subreg": -1,
                    "val": 0,
                    "count": 1,
                },
                {
                    "op": INSTR_DEF_OP,
                    "reg": 4,
                    "subreg": -1,
                    "val": 1,
                    "count": 1,
                },
                {
                    "op": INSTR_DEF_OP,
                    "reg": 4,
                    "subreg": -1,
                    "val": 5,
                    "count": 1,
                },
                {"op": SET_OP, "reg": 21, "subreg": -1, "val": 64, "count": 1},
            ]
        )
        out = tk._add_codebook_id_coverage(tokens)
        ids = sorted(
            int(v) for o, v in zip(out["op"], out["val"]) if int(o) == INSTR_DEF_OP
        )
        self.assertEqual(ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(sum(1 for o in out["op"] if int(o) == SET_OP), 1)


class TestMergeRaisesOnOovCodebookId(unittest.TestCase):
    def test_oov_codebook_id_raises_not_snaps(self):
        tk = RegTokenizer(FakeArgs(), tokens=None)
        tokens = _tokens(
            [
                (SET_OP, FRAME_REG, -1, 0),
                (INSTR_DEF_OP, 4, -1, 0),
                (INSTR_DEF_OP, 4, -1, 1),
            ]
        )
        df = _df([(INSTR_DEF_OP, 4, -1, 99)])
        with self.assertRaises(KeyError):
            tk.merge_token_df(tokens, df)


if __name__ == "__main__":
    unittest.main()
