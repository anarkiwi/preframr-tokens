"""Smoke tests for ``preframr_tokens.corpus.Corpus``. Comprehensive coverage of the load/parse/tokenize pipeline lives in the main `preframr` repo's RegDataset tests (corpus fixtures + parser fixtures live there); here we verify the module imports cleanly, the constructor sets initial state, and the SeqMeta + parse_eval_reglogs exports work."""

import logging
import unittest

from preframr_tokens.blocks import (
    LEGACY_EVAL_SUBSET_NAME,
    SeqMeta,
    parse_eval_reglogs,
)
from preframr_tokens.corpus import Corpus


class _StubArgs:
    """Minimal args namespace that Corpus.__init__ reads from."""

    def __init__(self):
        self.seq_len = 128
        self.tkvocab = 0
        self.max_perm = 1


class TestCorpusInit(unittest.TestCase):
    def test_constructor_sets_initial_state(self):
        args = _StubArgs()
        logger = logging.getLogger("test-corpus")
        c = Corpus(args, logger)
        self.assertEqual(c.reg_widths, {})
        self.assertEqual(c.n_vocab, 0)
        self.assertEqual(c.n_words, 0)
        self.assertEqual(c.val_subset_names, [])
        self.assertIsNone(c._tokenize_meta)  # pylint: disable=protected-access
        self.assertIsNotNone(c.tokenizer)


class TestSeqMeta(unittest.TestCase):
    def test_dataclass_fields(self):
        m = SeqMeta(irq=19656, df_file="/tmp/foo.dump.parquet", i=2)
        self.assertEqual(m.irq, 19656)
        self.assertEqual(m.df_file, "/tmp/foo.dump.parquet")
        self.assertEqual(m.i, 2)
        self.assertIsNone(m.l)
        self.assertIsNone(m.npy_path)


class TestParseEvalReglogs(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_eval_reglogs(""), {})
        self.assertEqual(parse_eval_reglogs(None), {})

    def test_legacy_single_glob(self):
        out = parse_eval_reglogs("/tmp/eval/*.dump.parquet")
        self.assertEqual(out, {LEGACY_EVAL_SUBSET_NAME: "/tmp/eval/*.dump.parquet"})

    def test_multi_subset(self):
        out = parse_eval_reglogs("eval_a=/a/*.pq;eval_b=/b/*.pq")
        self.assertEqual(list(out.keys()), ["eval_a", "eval_b"])
        self.assertEqual(out["eval_a"], "/a/*.pq")
        self.assertEqual(out["eval_b"], "/b/*.pq")

    def test_duplicate_subset_raises(self):
        with self.assertRaises(ValueError):
            parse_eval_reglogs("eval_a=/a/*.pq;eval_a=/b/*.pq")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            parse_eval_reglogs("=/a/*.pq")

    def test_forbidden_char_raises(self):
        with self.assertRaises(ValueError):
            parse_eval_reglogs("eval/a=/a/*.pq")


if __name__ == "__main__":
    unittest.main()
