"""Verify the constrained Unigram trainer eliminates straddling
sub-tokens at PATTERN_REPLAY / PATTERN_OVERLAY boundaries.
"""

import os
import string
import tempfile
import unittest

import numpy as np
import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    FRAME_REG,
    LOOP_OP_REG,
    MODEL_PDTYPE,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    SET_OP,
    UNICODE_BASE,
)
from preframr_tokens.train_worker import (
    _build_unigram_pre_tokenizer,
    _isolation_char_class,
)

SPLITCHS = [ord(i) for i in string.punctuation]


def _atomic_id_from_char(ch, splitters):
    c = ord(ch)
    if c in SPLITCHS:
        idx = SPLITCHS.index(c)
        if idx < splitters:
            return idx
    if c >= 0xE000:
        c -= 0x800
    return c - UNICODE_BASE


def _classify_subtoken(atomic_ids, op_by_id, subreg_by_id):
    pending = 0
    saw_pr_short = False
    saw_po_orphan = False
    for aid in atomic_ids:
        op = op_by_id.get(aid)
        if op is None:
            continue
        if op == PATTERN_REPLAY_OP:
            if pending > 0:
                saw_pr_short = True
            pending = max(int(subreg_by_id.get(aid, 0)), 0)
        elif op == PATTERN_OVERLAY_OP:
            if pending == 0:
                saw_po_orphan = True
            else:
                pending -= 1
        else:
            if pending > 0:
                saw_pr_short = True
                pending = 0
    if pending > 0:
        saw_pr_short = True
    return saw_pr_short or saw_po_orphan


def _count_straddles(tkmodel_path, tokens_df, splitters):
    import json

    with open(tkmodel_path) as f:
        tk = json.load(f)
    op_by_id = {int(r.n): int(r.op) for r in tokens_df.itertuples()}
    subreg_by_id = {int(r.n): int(r.subreg) for r in tokens_df.itertuples()}
    n_multi = 0
    n_straddle = 0
    for tok_str, _logp in tk["model"]["vocab"]:
        if tok_str.startswith("<") and tok_str.endswith(">"):
            continue
        atomic_ids = [_atomic_id_from_char(ch, splitters) for ch in tok_str]
        if len(atomic_ids) <= 1:
            continue
        n_multi += 1
        if _classify_subtoken(atomic_ids, op_by_id, subreg_by_id):
            n_straddle += 1
    return n_multi, n_straddle


class FakeArgs:
    def __init__(self, tkmodel, tkvocab=64, tokenizer="unigram"):
        self.tokenizer = tokenizer
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.diffq = 64


def _build_synthetic_tokens():
    """Atomic alphabet: pad, several FRAME_REG variants (so splitters
    saturates the 32-punct range, matching production), SET rows,
    PATTERN_REPLAY (2 overlays), and two PATTERN_OVERLAY rows.
    """
    rows = [
        {"op": 0, "reg": PAD_REG, "subreg": -1, "val": 0, "count": 0, "n": 0},
    ]
    n = 1
    for v in range(4):
        rows.append(
            {
                "op": 0,
                "reg": FRAME_REG,
                "subreg": -1,
                "val": v + 1,
                "count": 100,
                "n": n,
            }
        )
        n += 1
    set_start = n
    for v in range(8):
        rows.append(
            {
                "op": SET_OP,
                "reg": 4 + (v % 3),
                "subreg": -1,
                "val": v + 1,
                "count": 50,
                "n": n,
            }
        )
        n += 1
    pr_id = n
    rows.append(
        {
            "op": PATTERN_REPLAY_OP,
            "reg": LOOP_OP_REG,
            "subreg": 2,
            "val": 0x101,
            "count": 50,
            "n": n,
        }
    )
    n += 1
    ov_ids = (n, n + 1)
    rows.extend(
        [
            {
                "op": PATTERN_OVERLAY_OP,
                "reg": LOOP_OP_REG,
                "subreg": 0,
                "val": 0x10001,
                "count": 50,
                "n": n,
            },
            {
                "op": PATTERN_OVERLAY_OP,
                "reg": LOOP_OP_REG,
                "subreg": 1,
                "val": 0x10002,
                "count": 50,
                "n": n + 1,
            },
        ]
    )
    df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
    df.attrs["pr_id"] = pr_id
    df.attrs["ov_ids"] = ov_ids
    df.attrs["set_ids"] = list(range(set_start, pr_id))
    df.attrs["frame_ids"] = list(range(1, set_start))
    return df


class TestUnigramIsolation(unittest.TestCase):
    """End-to-end check that ``train_worker``'s isolation hook prevents
    the unigram trainer from learning sub-tokens that straddle a
    PATTERN_REPLAY / PATTERN_OVERLAY macro boundary. We bypass
    ``RegTokenizer.train_tokenizer`` (which asserts an exact vocab
    size) and drive ``train_worker`` directly so the test stays
    """

    def _train_and_audit(self, isolation_override=None):
        import zstandard as zstd

        from preframr_tokens.train_worker import train_worker

        tokens_df = _build_synthetic_tokens()
        pr_id = tokens_df.attrs["pr_id"]
        ov1, ov2 = tokens_df.attrs["ov_ids"]
        set_ids = tokens_df.attrs["set_ids"]
        frame_ids = tokens_df.attrs["frame_ids"]
        rng = np.random.default_rng(0)
        seq = []
        for _ in range(1500):
            seq.append(int(rng.choice(frame_ids)))
            seq.append(int(rng.choice(set_ids)))
            seq.extend([pr_id, ov1, ov2])
            seq.append(int(rng.choice(set_ids)))
            seq.extend([pr_id, ov1, ov2])
            seq.append(int(rng.choice(set_ids)))
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(tkmodel=os.path.join(tmpdir, "tk.json"))
            loader = RegTokenizer(args, tokens=tokens_df)
            n_arr = np.array(seq, dtype=np.uint32)
            n_frame = int((tokens_df["reg"] == FRAME_REG).sum())
            loader.splitters = min(loader.splitters, n_frame)
            encoded = loader.encode_unicode(n_arr)
            uni_path = os.path.join(tmpdir, "syn.0.uni.zst")
            with zstd.open(uni_path, "w") as f:
                f.write(encoded)
            isolation_chars = (
                isolation_override
                if isolation_override is not None
                else loader._isolation_chars_for_ops(
                    [PATTERN_REPLAY_OP, PATTERN_OVERLAY_OP]
                )
            )
            train_worker(
                "unigram",
                32,
                args.tkmodel,
                [uni_path],
                [],
                isolation_chars,
            )
            return _count_straddles(args.tkmodel, tokens_df, loader.splitters)

    def test_constrained_eliminates_straddles(self):
        n_multi, n_straddle = self._train_and_audit()
        self.assertGreater(n_multi, 0, "trainer learned no multi-atom sub-tokens")
        self.assertEqual(
            n_straddle,
            0,
            f"expected zero straddles with isolation, got {n_straddle}/{n_multi}",
        )


class TestPreTokenizerHelper(unittest.TestCase):
    def test_no_isolation_chars_falls_back_to_punctuation(self):
        from tokenizers import pre_tokenizers as pt

        pre_tok = _build_unigram_pre_tokenizer("")
        self.assertIsInstance(pre_tok, pt.Punctuation)

    def test_char_class_collapses_runs(self):
        cls = _isolation_char_class("ABCDE")
        self.assertEqual(cls, "[A-E]")
        run = "".join(chr(0x400 + i) for i in range(1000))
        cls = _isolation_char_class(run)
        self.assertEqual(cls.count("-"), 1)

    def test_char_class_handles_50k_chars_in_few_runs(self):
        from tokenizers import Regex, pre_tokenizers as pt

        chars = "".join(chr(0xE000 + i) for i in range(50000))
        cls = _isolation_char_class(chars)
        self.assertIsNotNone(cls)
        self.assertEqual(cls.count("-"), 1)
        pt.Split(pattern=Regex(cls), behavior="isolated", invert=False)

    def test_isolation_chars_yield_sequence(self):
        from tokenizers import pre_tokenizers as pt

        pre_tok = _build_unigram_pre_tokenizer("Ԁԁ")
        self.assertIsInstance(pre_tok, pt.Sequence)
        out = pre_tok.pre_tokenize_str("aԀbԁc")
        chars_isolated = [t for t, _span in out]
        self.assertIn("Ԁ", chars_isolated)
        self.assertIn("ԁ", chars_isolated)

    def test_isolation_blocks_unigram_merge_attempts(self):
        tokens_df = _build_synthetic_tokens()
        loader = RegTokenizer(FakeArgs(tkmodel=None), tokens=tokens_df)
        loader.splitters = min(
            loader.splitters, int((tokens_df["reg"] == FRAME_REG).sum())
        )
        isolation_chars = loader._isolation_chars_for_ops(
            [PATTERN_REPLAY_OP, PATTERN_OVERLAY_OP]
        )
        self.assertGreater(len(isolation_chars), 0)
        pre_tok = _build_unigram_pre_tokenizer(isolation_chars)
        pr_id = tokens_df.attrs["pr_id"]
        ov1, ov2 = tokens_df.attrs["ov_ids"]
        set_ids = tokens_df.attrs["set_ids"]
        seq = [
            int(set_ids[0]),
            pr_id,
            ov1,
            ov2,
            int(set_ids[1]),
            pr_id,
            ov1,
            ov2,
        ]
        n_arr = np.array(seq, dtype=np.uint32)
        encoded = loader.encode_unicode(n_arr)
        out = pre_tok.pre_tokenize_str(encoded)
        isolation_set = set(isolation_chars)
        for chunk, _span in out:
            if any(ch in isolation_set for ch in chunk):
                self.assertEqual(
                    len(chunk),
                    1,
                    f"PR/OV char must be isolated; got pre-token {chunk!r}",
                )


if __name__ == "__main__":
    unittest.main()
