"""Event-boundary-respecting dictionary: the separator-injected ``.uni`` words sit exactly at
grammar-unit starts, the unigram vocab never welds across a unit boundary, and the whole path stays
byte-exact (space-stripped uni text round-trips, decode reproduces the atom ids)."""

import logging
import os
import types

import numpy as np
import pandas as pd

from preframr_tokens.corpus import Corpus
from preframr_tokens.events import dataset
from preframr_tokens.macros import pitch_grid


def _args(tkvocab=0):
    return types.SimpleNamespace(tokenizer="unigram", tkvocab=tkvocab)


def _corpus_args(d, **kw):
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


def _trained_corpus(tmp_path, tkvocab):
    """Train a tiny unigram dictionary through the real events preload (the segmenter is wired in
    ``Corpus.preload``); returns ``(corpus, dump_paths)``."""
    paths = []
    for k in range(5):
        p = str(tmp_path / f"song{k}.dump.parquet")
        _synth_df(96, seed=k).to_parquet(p)
        paths.append(p)
    c = Corpus(_corpus_args(str(tmp_path), tkvocab=tkvocab), logging.getLogger("t"))
    c.preload()
    return c, paths


def _synth_df(n_frames=64, seed=7):
    """A 3-voice dump (per-voice note walk + gate toggles) so headers, DT runs, voice markers and
    events all recur -- a unigram would weld across unit boundaries without segmentation.
    """
    rng = np.random.default_rng(seed)
    writes = []
    for f in range(n_frames):
        for v in range(3):
            base = 7 * v
            nt = 38 + v * 6 + int(rng.integers(0, 10))
            fr = pitch_grid.note_freq_at(nt, 0.0)
            writes.append((f, base + 0, fr & 0xFF))
            writes.append((f, base + 1, (fr >> 8) & 0xFF))
            writes.append((f, base + 4, 0x41 if (f + v) % 5 else 0x40))
            writes.append((f, base + 5, 0x08))
            writes.append((f, base + 6, 0xA9))
    writes.sort(key=lambda t: t[0])
    return pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )


def _segmented_tokenizer():
    tk = dataset.make_tokenizer(_args())
    tk.unit_segmenter = dataset.unit_starts
    return tk


def test_uni_text_spaces_sit_at_unit_starts():
    ids = dataset.dump_token_ids(_synth_df())
    tk = _segmented_tokenizer()
    text = tk._uni_text(np.asarray(ids))  # pylint: disable=protected-access
    starts = dataset.unit_starts(ids)
    words = text.split(" ")
    assert len(words) == len(starts), "one word per grammar unit"
    assert " " not in tk.encode_unicode(np.asarray(ids)), "space is never an atom char"
    pos = 0
    rebuilt_starts = []
    for w in words:
        rebuilt_starts.append(pos)
        pos += len(w)
    assert rebuilt_starts == starts, "word boundaries fall exactly at unit starts"


def test_space_stripped_uni_text_roundtrips_ids():
    ids = dataset.dump_token_ids(_synth_df())
    tk = _segmented_tokenizer()
    text = tk._uni_text(np.asarray(ids))  # pylint: disable=protected-access
    decoded = tk.decode_unicode(text.replace(" ", ""))
    assert list(decoded) == list(
        ids
    ), "stripping the injected spaces recovers the atom ids"


def test_no_segmenter_leaves_text_space_free():
    ids = dataset.dump_token_ids(_synth_df())
    tk = dataset.make_tokenizer(_args())
    text = tk._uni_text(np.asarray(ids))  # pylint: disable=protected-access
    assert " " not in text, "without a segmenter the uni text is the bare encoding"
    assert text == tk.encode_unicode(np.asarray(ids))


def test_events_preload_sets_segmenter_and_trains(tmp_path):
    """Corpus.preload wires the events segmenter unconditionally and trains the dictionary."""
    c, _paths = _trained_corpus(tmp_path, tkvocab=160)
    assert c.tokenizer.unit_segmenter is dataset.unit_starts
    assert c.tokenizer.tkmodel is not None
    assert c.tokenizer.tkmodel.get_vocab_size() == 160
