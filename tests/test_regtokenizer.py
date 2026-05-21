import os
import random
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import MODEL_PDTYPE, UNICODE_BASE


class FakeArgs:
    def __init__(self, seq_len=128, tkvocab=0, diffq=64, tkmodel=None, tokenizer="bpe"):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.tkvocab = tkvocab
        self.tkmodel = tkmodel
        self.max_files = 1
        self.diffq = diffq
        self.tokenizer = tokenizer


class TestRegTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        max_n = 256
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=2048,
                tkvocab=(max_n * 2),
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
            )
            loader = RegTokenizer(args, tokens=None)
            x = []
            for _ in range(100):
                x.extend([random.randint(0, max_n - 1) for _ in range(args.seq_len)])
            df = pd.DataFrame(x, dtype=pd.UInt16Dtype(), columns=["n"])
            df.loc[(df["n"] == 0) & (df["n"].shift() == 0), "n"] = 1

            for tokenizer in ("bpe", "unigram"):
                args.tokenizer = tokenizer
                loader = RegTokenizer(args, tokens=None)
                loader.train_tokenizer([(f"{tmpdir}/tune.dump.parquet", df, 1)])
                orig = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9], dtype=np.uint16)
                encoded = loader.encode(orig)
                decoded = loader.decode(encoded)
                self.assertTrue(
                    np.array_equal(orig, decoded), (tokenizer, orig, decoded)
                )

    def test_unicode(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        x = np.array([65, 0, 66, 67, 68, 69], dtype=np.uint32)
        y = loader.decode_unicode(loader.encode_unicode(x))
        self.assertTrue(np.array_equal(x, y), (x, y))
        for _ in range(10):
            x = [j for j in range(65536 - UNICODE_BASE)]
            random.shuffle(x)
            x = np.array(x, dtype=np.uint32)
            y = loader.decode_unicode(loader.encode_unicode(x))
            self.assertTrue(np.array_equal(x, y), (x, y))
        surr_ids = list(range(0xD800 - UNICODE_BASE - 16, 0xE000 - UNICODE_BASE + 16))
        x = np.array(surr_ids, dtype=np.uint32)
        encoded = loader.encode_unicode(x)
        self.assertEqual(encoded.encode("utf-8").decode("utf-8"), encoded)
        y = loader.decode_unicode(encoded)
        self.assertTrue(np.array_equal(x, y), (x, y))
        x = np.array([100000, 200000, 500000, 1000000], dtype=np.uint32)
        encoded = loader.encode_unicode(x)
        self.assertEqual(encoded.encode("utf-8").decode("utf-8"), encoded)
        y = loader.decode_unicode(encoded)
        self.assertTrue(np.array_equal(x, y), (x, y))

    def test_initial_alphabet_keeps_every_atom_char_in_vocab(self):
        """Regression for the encode/decode silent-atom-loss bug."""
        max_n = 64
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=2048,
                tkvocab=256,
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
                tokenizer="unigram",
            )
            loader = RegTokenizer(args, tokens=None)
            tokens_rows = [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0}
            ]
            for i in range(1, max_n):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": 1,
                        "subreg": -1,
                        "val": i,
                        "count": max_n - i,
                        "n": i,
                    }
                )
            loader.tokens = pd.DataFrame(tokens_rows, dtype=MODEL_PDTYPE)

            seq = []
            for _ in range(200):
                seq.extend(range(max_n))
                random.shuffle(seq)
            df = pd.DataFrame(seq, dtype=pd.UInt16Dtype(), columns=["n"])
            loader.train_tokenizer([(f"{tmpdir}/tune.dump.parquet", df, 1)])

            tk = loader.tkmodel
            vocab = tk.get_vocab()
            for atom_id in range(max_n):
                ch = loader.encode_unicode(np.array([atom_id], dtype=np.uint32))
                self.assertIn(
                    ch,
                    vocab,
                    f"atom {atom_id} char {ch!r} pruned from tkmodel vocab "
                    "-- encode/decode would silently drop this atom",
                )
                arr = np.array([atom_id], dtype=np.uint32)
                self.assertTrue(
                    np.array_equal(loader.decode(loader.encode(arr)), arr),
                    (atom_id, ch),
                )

    def test_validate_encoding_catches_pruned_atom(self):
        """``validate_encoding`` is the parse-time gate that fails the
        build if any atom would be silently dropped at encode time.
        Construct a tkmodel whose vocab is missing one atom char, then
        confirm validate_encoding raises rather than letting the
        corruption land in .blocks.npy.
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        max_n = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=64,
                tkvocab=max_n + 1,
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
                tokenizer="unigram",
            )
            loader = RegTokenizer(args, tokens=None)
            tokens_rows = [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0}
            ]
            for i in range(1, max_n):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": 1,
                        "subreg": -1,
                        "val": i,
                        "count": 1,
                        "n": i,
                    }
                )
            loader.tokens = pd.DataFrame(tokens_rows, dtype=MODEL_PDTYPE)

            kept = [
                loader.encode_unicode(np.array([i], dtype=np.uint32))
                for i in range(max_n)
                if i != 1
            ]
            tk = Tokenizer(models.Unigram())
            tk.normalizer = None
            tk.pre_tokenizer = pre_tokenizers.Punctuation()
            trainer = trainers.UnigramTrainer(
                vocab_size=max_n,
                special_tokens=["<unk>"],
                unk_token="<unk>",
                initial_alphabet=kept,
            )
            tk.train_from_iterator(["".join(kept) * 4], trainer=trainer)
            tk.save(args.tkmodel)
            loader.tkmodel = Tokenizer.from_file(args.tkmodel)

            seq = np.array([0, 1, 2, 3, 1, 4], dtype=np.int64)
            with self.assertRaises(AssertionError):
                loader.validate_encoding("test_file", seq)

    def test_load_in_fresh_instance_round_trips_against_trainer(self):
        """Cross-instance round-trip: simulate the production write/read
        boundary where ``stftokenize.py`` trains the tokenizer in one
        process and ``train.py`` / ``predict.py`` load it in another.
        """
        max_n = 80
        n_frame_variants = 7
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=2048,
                tkvocab=256,
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
                tokenizer="unigram",
            )
            trainer = RegTokenizer(args, tokens=None)
            tokens_rows = [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0}
            ]
            for i in range(1, n_frame_variants + 1):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": -128,
                        "subreg": -1,
                        "val": i,
                        "count": 100 - i,
                        "n": i,
                    }
                )
            for i in range(n_frame_variants + 1, max_n):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": 1,
                        "subreg": -1,
                        "val": i,
                        "count": max_n - i,
                        "n": i,
                    }
                )
            trainer.tokens = pd.DataFrame(tokens_rows, dtype=MODEL_PDTYPE)

            seq = []
            for _ in range(200):
                seq.extend(range(max_n))
                random.shuffle(seq)
            df = pd.DataFrame(seq, dtype=pd.UInt16Dtype(), columns=["n"])
            trainer.train_tokenizer([(f"{tmpdir}/tune.dump.parquet", df, 1)])
            self.assertEqual(
                trainer.splitters,
                n_frame_variants,
                "train_tokenizer should clamp splitters to frame_tokens",
            )

            tokens_csv = os.path.join(str(tmpdir), "tokens.csv")
            trainer.tokens.to_csv(tokens_csv, index=False)

            loader = RegTokenizer(args, tokens=None)
            self.assertEqual(
                loader.splitters,
                32,
                "fresh instance should default to SPLITTERS=32",
            )
            with open(args.tkmodel) as fh:
                tkmodel_str = fh.read()
            loaded_tokens = pd.read_csv(tokens_csv)
            loader.load(tkmodel_str, loaded_tokens)
            self.assertEqual(
                loader.splitters,
                n_frame_variants,
                "load() must re-derive splitters from the tokens table -- "
                "otherwise encode/decode at the read side uses a different "
                "atom-id -> char mapping than the trained tkmodel was built "
                "around",
            )

            sample_atoms = np.array([1, 2, 7, 8, 15, 30, 50, 70], dtype=np.uint32)
            trainer_encoded = trainer.encode(sample_atoms)
            loader_encoded = loader.encode(sample_atoms)
            self.assertTrue(
                np.array_equal(trainer_encoded, loader_encoded),
                f"trainer and loader produced different super-tokens for "
                f"{sample_atoms.tolist()}: trainer={trainer_encoded.tolist()} "
                f"loader={loader_encoded.tolist()}",
            )

            loader_decoded = loader.decode(loader_encoded)
            self.assertTrue(
                np.array_equal(sample_atoms, loader_decoded[: len(sample_atoms)]),
                f"loader.decode(loader.encode(atoms)) != atoms: "
                f"in={sample_atoms.tolist()} out={loader_decoded.tolist()}",
            )

    def test_load_snapshot_matches_train_tokenizer_snapshot(self):
        """Property test: ``load(saved_tkmodel, tokens)`` reproduces the
        same ``RegTokenizer.__dict__`` snapshot that ``train_tokenizer``
        leaves on the trainer instance.
        """
        max_n = 80
        n_frame_variants = 7
        with tempfile.TemporaryDirectory() as tmpdir:
            args = FakeArgs(
                seq_len=2048,
                tkvocab=256,
                tkmodel=os.path.join(str(tmpdir), "tk.model"),
                tokenizer="unigram",
            )
            tokens_rows = [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0}
            ]
            for i in range(1, n_frame_variants + 1):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": -128,
                        "subreg": -1,
                        "val": i,
                        "count": 100 - i,
                        "n": i,
                    }
                )
            for i in range(n_frame_variants + 1, max_n):
                tokens_rows.append(
                    {
                        "op": 0,
                        "reg": 1,
                        "subreg": -1,
                        "val": i,
                        "count": max_n - i,
                        "n": i,
                    }
                )
            tokens = pd.DataFrame(tokens_rows, dtype=MODEL_PDTYPE)

            trainer = RegTokenizer(args, tokens=tokens)
            seq = []
            for _ in range(200):
                seq.extend(range(max_n))
                random.shuffle(seq)
            df = pd.DataFrame(seq, dtype=pd.UInt16Dtype(), columns=["n"])
            trainer.train_tokenizer([(f"{tmpdir}/tune.dump.parquet", df, 1)])

            with open(args.tkmodel) as fh:
                tkmodel_str = fh.read()

            loader = RegTokenizer(args, tokens=None)
            loader.load(tkmodel_str, tokens)

            trainer_snap = self._snapshot(trainer)
            loader_snap = self._snapshot(loader)
            self.assertEqual(
                trainer_snap.keys(),
                loader_snap.keys(),
                "RegTokenizer attribute set diverged between trainer and "
                "loader -- a new mutable field was added to one path "
                "and not the other",
            )
            for key in trainer_snap:
                self.assertEqual(
                    trainer_snap[key],
                    loader_snap[key],
                    f"RegTokenizer.{key} diverges across train/load: "
                    f"trainer={trainer_snap[key]!r} loader={loader_snap[key]!r}. "
                    f"If this is a new mutable attribute, mutate it in "
                    f"BOTH train_tokenizer and load (or make it a derived "
                    f"property of the loaded artifacts).",
                )

    @staticmethod
    def _snapshot(rt):
        """Return a deeply-comparable representation of mutable state."""
        snap = {}
        for k, v in rt.__dict__.items():
            if k in ("args", "logger"):
                continue
            snap[k] = TestRegTokenizer._normalize_value(v)
        return snap

    @staticmethod
    def _normalize_value(v):
        from tokenizers import Tokenizer

        if isinstance(v, Tokenizer):
            return ("tokenizer", v.to_str())
        if isinstance(v, pd.DataFrame):
            return (
                "df",
                tuple(v.shape),
                tuple(v.columns.tolist()),
                v.to_dict(orient="records"),
            )
        if isinstance(v, list):
            return [TestRegTokenizer._normalize_value(x) for x in v]
        if isinstance(v, tuple):
            return tuple(TestRegTokenizer._normalize_value(x) for x in v)
        if isinstance(v, np.ndarray):
            return ("ndarray", v.tolist())
        return v

    def test_make_tokens(self):
        loader = RegTokenizer(FakeArgs(), tokens=None)
        test_df = pd.DataFrame(
            [
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        tokens_df = pd.DataFrame(
            [
                {"op": 0, "reg": -1, "subreg": -1, "val": 0, "count": 0, "n": 0},
                {"op": 0, "reg": 1, "subreg": -1, "val": 1, "count": 3, "n": 1},
                {"op": 0, "reg": 1, "subreg": -1, "val": 2, "count": 10, "n": 2},
                {"op": 0, "reg": 1, "subreg": -1, "val": 3, "count": 1, "n": 3},
                {"op": 0, "reg": 2, "subreg": -1, "val": 1, "count": 3, "n": 4},
                {"op": 0, "reg": 2, "subreg": -1, "val": 3, "count": 1, "n": 5},
                {"op": 0, "reg": 3, "subreg": -1, "val": 1, "count": 3, "n": 6},
                {"op": 0, "reg": 3, "subreg": -1, "val": 3, "count": 1, "n": 7},
                {"op": 0, "reg": 4, "subreg": -1, "val": 1, "count": 3, "n": 8},
                {"op": 0, "reg": 4, "subreg": -1, "val": 3, "count": 1, "n": 9},
                {"op": 0, "reg": 5, "subreg": -1, "val": 1, "count": 3, "n": 10},
                {"op": 0, "reg": 5, "subreg": -1, "val": 3, "count": 1, "n": 11},
            ],
            dtype=MODEL_PDTYPE,
        )
        for _ in range(5):
            loader.accumulate_tokens(test_df, "test")
            test_df.loc[test_df["val"] != 2, "reg"] += 1
        result_df = loader.make_tokens().astype(MODEL_PDTYPE)
        self.assertTrue(
            tokens_df.equals(result_df), result_df.to_dict(orient="records")
        )
