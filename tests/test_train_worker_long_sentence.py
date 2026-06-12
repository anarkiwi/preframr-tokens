"""Guard the in-library RUST_MIN_STACK fix: a long single sentence (~40k atoms) reproduced the
unigram-trainer recursive-Rc::drop SIGSEGV, which ``train_worker`` now prevents by raising the Rust
thread stack before training starts."""

import os
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset as events_dataset
from preframr_tokens.stfconstants import DUMP_SUFFIX


class TestTrainWorkerLongSentence(unittest.TestCase):
    def test_long_sentence_trains_without_segfault(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                tokenizer="unigram",
                tkvocab=160,
                tkmodel=os.path.join(tmpdir, "tk.json"),
            )
            tok = events_dataset.make_tokenizer(args)
            rng = np.random.default_rng(0)
            ids = rng.integers(1, 128, size=40_000, dtype=np.int64)
            df = pd.DataFrame({"n": ids})
            df_file = os.path.join(tmpdir, "long" + DUMP_SUFFIX)
            tok.train_tokenizer([(df_file, df, 0)])
            self.assertTrue(os.path.exists(args.tkmodel))
            self.assertGreater(os.path.getsize(args.tkmodel), 0)


if __name__ == "__main__":
    unittest.main()
