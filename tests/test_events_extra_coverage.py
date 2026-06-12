"""Direct coverage for the events primitives the deleted v0/factored tests used to exercise
incidentally: the pipeline window/block helpers, oracle empties, the dataset atom-cache + stride paths,
and the stream codec's conditioning/roundtrip edge helpers."""

import os
import types

import numpy as np
import pandas as pd
import pytest
import zstandard as zstd

from preframr_tokens.events import dataset, oracle, pipeline, stream
from preframr_tokens.macros import pitch_grid
from preframr_tokens.stfconstants import DUMP_SUFFIX

_COLS = ["clock", "irq", "chipno", "reg", "val"]


def _multi_voice_ids(k):
    rng = np.random.default_rng(k)
    writes = []
    for f in range(40):
        for v in range(3):
            base = 7 * v
            nt = 40 + v * 5 + int(rng.integers(0, 12))
            fr = pitch_grid.note_freq_at(nt, 0.0)
            writes += [
                (f, base, fr & 0xFF),
                (f, base + 1, (fr >> 8) & 0xFF),
                (f, base + 4, 0x41 if (f + v) % 5 else 0x40),
                (f, base + 5, 8),
                (f, base + 6, 0xA9),
            ]
    writes.sort(key=lambda t: t[0])
    df = pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )
    return dataset.dump_token_ids(df)


def _synth_df(n=24):
    writes = []
    for f in range(n):
        fr = 0x1240 + (6 * ((f % 4) - 2) if f >= 8 else 0)
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, (fr >> 8) & 0xFF),
            (f, 4, 0x41 if 4 <= f < 18 else 0x40),
            (f, 5, 8),
            (f, 6, 0xA9),
        ]
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


def _empty_df():
    return pd.DataFrame({c: np.empty(0, dtype=np.int64) for c in _COLS})


def test_oracle_empty_dump_is_writeless():
    ow = oracle.ordered_writes(_empty_df())
    assert ow.n_frames == 0 and len(ow) == 0
    assert oracle.ordered_writes(_synth_df(6).assign(chipno=1)).n_frames == 0


def test_pipeline_blocks_and_array_round_trip():
    df = _synth_df(24)
    ow = oracle.ordered_writes(df)
    blocks = pipeline.dump_blocks(df, frames_per_block=8)
    assert blocks
    for w, toks in zip(pipeline.iter_windows(ow, 8), blocks):
        assert pipeline.block_writes(toks) == stream.canonical_writes(w)
    assert pipeline.block_array(df, block_size=64, frames_per_block=8).shape[1] == 64
    assert pipeline.block_array(df, block_size=8, frames_per_block=8).shape[1] == 8
    assert pipeline.block_array(_empty_df(), 64, 8).shape == (0, 64)
    assert pipeline.atoms() == list(range(pipeline.VOCAB_SIZE))


def test_pipeline_window_stride_advances():
    ow = oracle.ordered_writes(_synth_df(24))
    windows = list(pipeline.iter_windows(ow, frames_per_block=8, stride=8))
    assert len(windows) == 3 and windows[0].n_frames == 8


def test_stream_chunk_keyframe_and_strip_edges():
    assert stream.chunk_keyframe([], 0) == []
    assert stream.chunk_keyframe([stream.VAR_BASE], 1) == []
    toks = stream.encode(oracle.ordered_writes(_synth_df(24)))
    kf = stream.chunk_keyframe(toks, upto=len(toks) // 2)
    assert kf and kf[0] == stream.KEYFRAME
    assert stream.strip_keyframes(kf + toks) == toks
    assert stream.roundtrip_ok(_synth_df(12))


def test_dataset_atom_cache_corruption_falls_back(tmp_path):
    df = _synth_df(16)
    df_file = str(tmp_path / "song.1.dump.parquet")
    df.to_parquet(df_file)
    ids = dataset.dump_token_ids(df, df_file)
    cache = list(tmp_path.glob("*.atoms.zst"))[0]
    with zstd.open(str(cache), "wb") as fh:
        fh.write(b"abc")
    assert dataset.dump_token_ids(df, df_file) == ids


def test_dataset_encode_block_array_explicit_stride():
    df = _synth_df(40)
    tk = dataset.make_tokenizer(types.SimpleNamespace(tokenizer="unigram", tkvocab=0))
    arr = dataset.encode_block_array(tk, df, block_size=64, stride=48)
    assert arr.shape[1] == 64 and arr.shape[0] >= 1
    flat = [int(x) for row in arr for x in row]
    while flat and flat[-1] == 0:
        flat.pop()
    assert dataset.ids_to_writes(flat) == stream.canonical_writes(
        oracle.ordered_writes(df)
    )


def test_regtokenizer_token_metadata_with_and_without_tkmodel(tmp_path):
    tk0 = dataset.make_tokenizer(types.SimpleNamespace(tokenizer="unigram", tkvocab=0))
    assert len(tk0.token_metadata()) == len(tk0.tokens)
    streams = [_multi_voice_ids(k) for k in range(16)]
    args = types.SimpleNamespace(
        tokenizer="unigram", tkvocab=160, tkmodel=str(tmp_path / "tk.json")
    )
    tok = dataset.make_tokenizer(args)
    dfs = [
        (
            str(tmp_path / f"s{i}{DUMP_SUFFIX}"),
            pd.DataFrame({"n": np.asarray(s, dtype=np.int64)}),
            0,
        )
        for i, s in enumerate(streams)
    ]
    tok.train_tokenizer(dfs)
    assert len(tok.token_metadata()) == 160


def test_dataset_atom_cache_write_failure_is_silent(tmp_path):
    df = _synth_df(12)
    df_file = str(tmp_path / "song.1.dump.parquet")
    df.to_parquet(df_file)
    cache_path = dataset._atom_cache_path(df_file)
    os.mkdir(cache_path)
    expected = dataset.dump_token_ids(df)
    assert dataset.dump_token_ids(df, df_file) == expected
    assert os.path.isdir(cache_path)


def test_stream_decode_rejects_trailing_after_empty():
    with pytest.raises(ValueError):
        stream.decode([stream.VAR_BASE, stream.VOICE_BASE])
