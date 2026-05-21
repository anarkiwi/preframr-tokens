"""Coverage tests for ``RegTokenizer`` helpers not covered by the
existing ``test_regtokenizer.py`` make_tokens path."""

import unittest

import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import FRAME_REG, MODEL_PDTYPE, SET_OP


class FakeArgs:
    tkvocab = 0
    tokenizer = "unigram"
    tkmodel = None


class TestRegMaxAndWidths(unittest.TestCase):
    def test_get_reg_max_picks_max_per_reg(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame(
            [
                {"reg": 0, "val": 5},
                {"reg": 0, "val": 200},
                {"reg": 1, "val": 7},
            ]
        )
        out = loader.get_reg_max(df, {})
        self.assertEqual(out[0], 200)
        self.assertEqual(out[1], 7)

    def test_get_reg_max_keeps_existing_max(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame([{"reg": 0, "val": 5}])
        out = loader.get_reg_max(df, {0: 999})
        self.assertEqual(out[0], 999)

    def test_get_reg_width_from_max(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        widths = loader.get_reg_width_from_max({0: 10, 1: 256, 2: 2**16, 3: 2**24})
        self.assertEqual(widths[0], 1)
        self.assertEqual(widths[1], 2)
        self.assertEqual(widths[2], 3)
        self.assertEqual(widths[3], 4)


class TestTokenMetadataNoTkmodel(unittest.TestCase):
    def test_metadata_format(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        loader.tokens = pd.DataFrame(
            [
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 5},
                {"op": SET_OP, "reg": 2, "subreg": -1, "val": 9},
            ],
            dtype=MODEL_PDTYPE,
        )
        meta = loader.token_metadata()
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta[0], "0 1 -1 5")


class TestEncodeDecodeNoTkmodel(unittest.TestCase):
    def test_pass_through(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        import numpy as np

        seq = np.array([1, 2, 3, 4], dtype=np.int64)
        self.assertTrue((loader.encode(seq) == seq).all())
        self.assertTrue((loader.decode(seq) == seq).all())


class TestMergeTokensWrapper(unittest.TestCase):
    def test_merge_tokens_iterates_dfs(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        tokens = pd.DataFrame(
            [
                {
                    "op": 0,
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 1,
                    "count": 1,
                    "n": 0,
                },
                {
                    "op": 0,
                    "reg": 1,
                    "subreg": -1,
                    "val": 5,
                    "count": 1,
                    "n": 1,
                },
            ],
            dtype=MODEL_PDTYPE,
        )

        def _df():
            return pd.DataFrame(
                [
                    {
                        "op": 0,
                        "reg": FRAME_REG,
                        "subreg": -1,
                        "val": 1,
                        "diff": 19656,
                    },
                    {"op": 0, "reg": 1, "subreg": -1, "val": 5, "diff": 32},
                ],
                dtype=MODEL_PDTYPE,
            )

        result = loader.merge_tokens(tokens, [_df(), _df()])
        self.assertEqual(len(result), 2)
        for merged in result:
            self.assertIn("n", merged.columns)


class TestAccumulateAutoCrunch(unittest.TestCase):
    def test_auto_crunches_at_threshold(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        df = pd.DataFrame(
            [{"op": SET_OP, "reg": 1, "subreg": -1, "val": 1}], dtype=MODEL_PDTYPE
        )
        for i in range(70):
            loader.accumulate_tokens(df.copy(), f"file{i}")
        self.assertGreater(len(loader.frame_tokens), 0)
        self.assertLess(len(loader.frame_tokens), 70)


def _alphabet_keys(tokens):
    """Set of (op, reg, subreg, val) tuples in the token alphabet,
    excluding the synthetic pad row at idx 0 (PAD_REG=-1)."""
    out = set()
    for row in tokens.itertuples():
        if int(row.reg) < 0 and int(row.op) == 0 and int(row.val) == 0:
            continue
        sr = int(row.subreg) if pd.notna(row.subreg) else -1
        out.add((int(row.op), int(row.reg), sr, int(row.val)))
    return out


def _block_token_keys(blocks):
    """Union of (op, reg, subreg, val) tuples present in any block."""
    out = set()
    for block in blocks:
        for row in block.itertuples():
            sr = int(row.subreg) if pd.notna(row.subreg) else -1
            out.add((int(row.op), int(row.reg), sr, int(row.val)))
    return out


class TestAlphabetSongVsBlockCoverage(unittest.TestCase):
    """Token alphabet collected by ``accumulate_tokens`` over the
    full-song df + emitted blocks must not contain entries that don't
    appear in any block.
    """

    def test_no_dead_alphabet_entries_when_blocks_cover_song(self):
        """When every distinct (op, reg, subreg, val) in the song-df
        also appears in some block, the alphabet has no dead entries.
        Synthetic happy-path fixture pinning the property.
        """
        song_df = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 100},
                {"op": SET_OP, "reg": 2, "subreg": -1, "val": 200},
                {"op": SET_OP, "reg": 3, "subreg": -1, "val": 300},
            ],
            dtype=MODEL_PDTYPE,
        )
        block1 = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 100},
                {"op": SET_OP, "reg": 2, "subreg": -1, "val": 200},
            ],
            dtype=MODEL_PDTYPE,
        )
        block2 = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": 3, "subreg": -1, "val": 300},
            ],
            dtype=MODEL_PDTYPE,
        )
        loader = RegTokenizer(FakeArgs(), tokens=None)
        loader.accumulate_tokens(song_df, "song")
        for block in (block1, block2):
            loader.accumulate_tokens(block, "song")
        tokens = loader.make_tokens()
        alphabet = _alphabet_keys(tokens)
        block_union = _block_token_keys([block1, block2])
        dead = alphabet - block_union
        self.assertEqual(
            dead,
            set(),
            f"alphabet has dead entries (in song-df but not in any "
            f"emitted block): {sorted(dead)}",
        )

    def test_property_check_detects_song_only_token(self):
        """Symmetric regression case: a token introduced via the song-
        df accumulation but absent from every block IS a dead alphabet
        entry. This pins the property check itself -- without this
        sanity case, a vacuous alphabet (or a buggy `_alphabet_keys`)
        would silently pass the happy-path test above.
        """
        song_df = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 100},
                {"op": SET_OP, "reg": 99, "subreg": -1, "val": 999},
            ],
            dtype=MODEL_PDTYPE,
        )
        block1 = pd.DataFrame(
            [
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": 1, "subreg": -1, "val": 100},
            ],
            dtype=MODEL_PDTYPE,
        )
        loader = RegTokenizer(FakeArgs(), tokens=None)
        loader.accumulate_tokens(song_df, "song")
        loader.accumulate_tokens(block1, "song")
        tokens = loader.make_tokens()
        alphabet = _alphabet_keys(tokens)
        block_union = _block_token_keys([block1])
        dead = alphabet - block_union
        self.assertIn((SET_OP, 99, -1, 999), dead)


if __name__ == "__main__":
    unittest.main()
