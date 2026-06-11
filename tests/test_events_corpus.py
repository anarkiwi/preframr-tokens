"""Events-native corpus dataset-build guards (REDESIGN_optionB §7.1, step 3): Corpus.preload globs raw
dumps, sets the fixed event alphabet, writes per-dump .blocks.npy + df-map + reg-widths, and (with tkvocab)
trains a BPE over the event token streams. The whole-tune block stream decodes byte-exact both ways.
"""

import glob
import logging
import os
import tempfile
import types

import numpy as np
import pandas as pd

from preframr_tokens.corpus import Corpus
from preframr_tokens.events import dataset as events_dataset
from preframr_tokens.events import oracle, stream


def _synth_dump(path, seed):
    writes = []
    for f in range(120):
        fr = 0x1240 + seed * 16 + (6 * ((f % 4) - 2) if f >= 8 else 0)
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, (fr >> 8) & 0xFF),
            (f, 4, 0x41 if 4 <= f < 60 else 0x40),
            (f, 5, 8),
            (f, 6, 0xA9),
            (f, 2, (64 + f) & 0xFF),
            (f, 3, 0),
        ]
    writes.sort(key=lambda t: t[0])
    pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    ).to_parquet(path)


def _args(d, **kw):
    base = dict(
        reglogs=os.path.join(d, "*.dump.parquet"),
        max_files=10,
        require_pq=False,
        seq_len=64,
        tkvocab=0,
        tokenizer="unigram",
        dataset_csv=os.path.join(d, "ds.csv.zst"),
        df_map_csv=os.path.join(d, "dfmap.csv"),
        token_csv=os.path.join(d, "tokens.csv"),
        tkmodel=os.path.join(d, "tk.json"),
        write_blocks=True,
        eval_reglogs="",
        max_perm=99,
        meta_irq_lo=0,
        meta_irq_hi=0,
        meta_exclude_digi=False,
        meta_require=False,
        min_song_tokens=0,
        reglog=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _reassemble(blocks_path):
    flat = [int(x) for row in np.load(blocks_path) for x in row]
    while flat and flat[-1] == 0:
        flat.pop()
    return flat


def test_preload_writes_events_blocks_and_decodes_byte_exact():
    with tempfile.TemporaryDirectory() as d:
        for i in range(2):
            _synth_dump(os.path.join(d, f"song{i}.dump.parquet"), i)
        c = Corpus(_args(d, tkvocab=0), logging.getLogger("t"))
        c.preload()
        assert len(c.tokenizer.tokens) == events_dataset.events_alphabet().shape[0]
        cols = set(pd.read_csv(c.args.df_map_csv).columns)
        assert {"dump_file", "kind", "irq", "n_rotations"} <= cols
        blocks = sorted(glob.glob(os.path.join(d, "*.blocks.npy")))
        assert len(blocks) == 2
        for bp in blocks:
            src = bp.replace(".0.blocks.npy", ".dump.parquet")
            ow = oracle.ordered_writes(pd.read_parquet(src))
            flat = _reassemble(bp)
            assert events_dataset.ids_to_writes(flat) == stream.canonical_writes(ow)
        assert len(list(c.iter_block_seqs())) == 2


def test_preload_trains_bpe_over_events_and_round_trips():
    with tempfile.TemporaryDirectory() as d:
        for i in range(3):
            _synth_dump(os.path.join(d, f"song{i}.dump.parquet"), i)
        c = Corpus(
            _args(d, seq_len=128, tkvocab=stream.VOCAB_SIZE + 8, dataset_csv=None),
            logging.getLogger("t"),
        )
        c.preload()
        assert c.tokenizer.tkmodel is not None
        for bp in sorted(glob.glob(os.path.join(d, "*.blocks.npy"))):
            src = bp.replace(".0.blocks.npy", ".dump.parquet")
            ow = oracle.ordered_writes(pd.read_parquet(src))
            bpe = np.asarray(_reassemble(bp), dtype=np.uint32)
            nspace = list(c.tokenizer.decode(bpe))
            assert events_dataset.ids_to_writes(nspace) == stream.canonical_writes(ow)
