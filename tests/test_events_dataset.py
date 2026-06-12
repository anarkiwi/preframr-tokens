"""Events-native tokenizer/alphabet guards: the fixed event-atom alphabet,
the PAD-reserving n-space offset, and byte-exact block<->ids round-trips through the reused RegTokenizer
layer.
"""

import os
import tempfile
import types

import numpy as np
import pandas as pd

from preframr_tokens import corpus
from preframr_tokens.events import dataset, oracle, pipeline, stream
from preframr_tokens.macros import pitch_grid
from preframr_tokens.stfconstants import DUMP_SUFFIX


def _args():
    return types.SimpleNamespace(tokenizer="unigram", tkvocab=0)


def _multi_voice_df(k):
    """A 3-voice synthetic dump (each voice gated with a per-stream note walk) so VOICE_BASE atoms
    recur and a unigram would weld across them without boundary isolation."""
    rng = np.random.default_rng(k)
    writes = []
    for f in range(48):
        for v in range(3):
            base = 7 * v
            nt = 40 + v * 5 + int(rng.integers(0, 12))
            fr = pitch_grid.note_freq_at(nt, 0.0)
            writes.append((f, base + 0, fr & 0xFF))
            writes.append((f, base + 1, (fr >> 8) & 0xFF))
            writes.append((f, base + 4, 0x41 if (f + v) % 6 else 0x40))
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


def _synth_df():
    writes = []
    for f in range(48):
        fr = 0x1240 + (6 * ((f % 4) - 2) if f >= 8 else 0)
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, (fr >> 8) & 0xFF),
            (f, 4, 0x41 if 4 <= f < 30 else 0x40),
            (f, 5, 8),
            (f, 6, 0xA9),
            (f, 2, 64 + f),
            (f, 3, 0),
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


def test_events_alphabet_and_splitters():
    alpha = dataset.events_alphabet()
    assert len(alpha) == pipeline.VOCAB_SIZE + 1, "PAD row + one row per atom"
    assert int(alpha["n"].iloc[0]) == dataset.PAD_ID
    tk = dataset.make_tokenizer(_args())
    assert (
        tk.splitters == 0
    ), "no FRAME_REG in the event alphabet -> no reserved splitters"


def test_block_ids_are_pad_safe_and_canonical():
    df = _synth_df()
    ow = oracle.ordered_writes(df)
    for w in pipeline.iter_windows(ow, 16):
        ids = dataset.block_to_ids(w)
        assert (
            min(ids) >= 1 and max(ids) <= pipeline.VOCAB_SIZE
        ), "n-space avoids PAD (0)"
        assert dataset.ids_to_writes(ids) == stream.canonical_writes(
            w
        ), "block ids decode to the window's canonical writes"


def test_unicode_serialize_roundtrip():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    ids = dataset.dump_block_ids(df, frames_per_block=16)[0]
    uni = tk.encode_unicode(np.array(ids, dtype=np.uint32))
    assert len(uni) == len(ids)
    assert list(tk.decode_unicode(uni)) == ids


def test_encode_block_array_chunks_whole_tune_byte_exact():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    block_size = 32
    arr = dataset.encode_block_array(tk, df, block_size)
    assert arr.dtype == np.int32 and arr.shape[1] == block_size
    assert int(arr.max()) <= pipeline.VOCAB_SIZE
    flat = [int(x) for row in arr for x in row]
    while flat and flat[-1] == 0:
        flat.pop()
    stripped = stream.strip_keyframes([n - 1 for n in flat if n])
    assert stripped == [
        n - 1 for n in dataset.dump_token_ids(df)
    ], "chunks (minus KEYFRAME prefixes) reassemble the whole-tune stream"
    assert dataset.ids_to_writes(flat) == stream.canonical_writes(
        oracle.ordered_writes(df)
    )


def test_dump_token_ids_atom_cache_roundtrip(tmp_path):
    """``df_file`` populates a codec-version-keyed ``.atoms.zst`` sidecar and a second call reuses it,
    ignoring the df it is handed -- the tkvocab-independent encode runs once."""
    df = _synth_df()
    df_file = str(tmp_path / "song.1.dump.parquet")
    df.to_parquet(df_file)
    expected = dataset.dump_token_ids(df)
    first = dataset.dump_token_ids(df, df_file)
    assert first == expected
    caches = list(tmp_path.glob("*.atoms.zst"))
    assert len(caches) == 1, "atom stream cached next to the dump"
    reused = dataset.dump_token_ids(_synth_df().iloc[:7], df_file)
    assert reused == expected, "second call reuses the cache, not the new df"


def test_encode_block_array_uses_atom_cache(tmp_path):
    """``encode_block_array(df_file=...)`` populates the same atom cache and stays byte-identical to the
    uncached path."""
    df = _synth_df()
    df_file = str(tmp_path / "song.1.dump.parquet")
    df.to_parquet(df_file)
    tk = dataset.make_tokenizer(_args())
    arr_cached = dataset.encode_block_array(tk, df, 32, df_file=df_file)
    assert list(tmp_path.glob("*.atoms.zst")), "block encode populates the atom cache"
    arr_plain = dataset.encode_block_array(tk, df, 32)
    assert np.array_equal(arr_cached, arr_plain), "cached path matches direct encode"


def test_block_worker_writes_blocks_byte_identical(tmp_path):
    """The ProcessPool block-worker funcs (run in-process here) rebuild the tokenizer from its serialized
    state and write a tune's ``.0.blocks.npy`` byte-identical to a direct ``encode_block_array``.
    """
    df = _synth_df()
    df_file = str(tmp_path / "song.1.dump.parquet")
    df.to_parquet(df_file)
    args = types.SimpleNamespace(tokenizer="unigram", tkvocab=0)
    tk = dataset.make_tokenizer(args)
    block_size = 33
    corpus._init_block_worker(args, tk.tokens, "", block_size)
    corpus._encode_block_worker(df_file)
    out = df_file.replace(".dump.parquet", ".0.blocks.npy")
    assert os.path.exists(out), "worker wrote the block array"
    saved = np.load(out)
    direct = dataset.encode_block_array(tk, df, block_size, df_file=df_file)
    assert np.array_equal(saved, direct), "worker output matches a direct encode"


def test_alen_cache_shared_across_calls_matches():
    """A shared ``alen_cache`` (the parallel block pass reuses one per worker) yields the same array as the
    default per-call cache."""
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    shared: dict = {}
    a = dataset.encode_block_array(tk, df, 128, alen_cache=shared)
    b = dataset.encode_block_array(tk, df, 128)
    assert np.array_equal(a, b) and shared, "shared cache populated + result unchanged"


def test_keyframe_prefixes_make_chunks_self_interpreting():
    """With a roomy block size, every chunk after the first is led by a KEYFRAME conditioning
    segment carrying the tune's tick/tuning headers + per-voice state; segments strip away for
    decode (the canonical stream stays redundancy-free)."""
    frames = []
    for rep in range(12):
        d = _synth_df()
        d["irq"] = d["irq"] + rep * 60
        d["clock"] = d["clock"] + rep * 100000
        frames.append(d)
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    tk = dataset.make_tokenizer(_args())
    arr = dataset.encode_block_array(tk, df, 256)
    kf_n = stream.KEYFRAME + 1
    assert arr.shape[0] >= 2, "synth tune must span multiple chunks"
    for row in arr[1:]:
        ids = [int(x) for x in row if x]
        assert ids and ids[0] == kf_n, "chunk must open with a KEYFRAME bracket"
        seg = ids[1 : ids.index(kf_n, 1)]
        atoms = [n - 1 for n in seg]
        assert stream.TUNING in atoms or stream.TICK in atoms or stream.NI_STEP in atoms
    assert stream.strip_keyframes([int(x) - 1 for row in arr for x in row if x]) == [
        n - 1 for n in dataset.dump_token_ids(df)
    ]


def _train_isolation(streams, isolate):
    """Train a unigram over the n-space streams with/without boundary isolation; return its vocab."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = types.SimpleNamespace(
            tokenizer="unigram",
            tkvocab=160,
            tkmodel=os.path.join(tmpdir, "tk.json"),
        )
        tok = dataset.make_tokenizer(args)
        if isolate:
            tok.isolation_ns = dataset.BOUNDARY_ISOLATION_NS
        dfs = [
            (
                os.path.join(tmpdir, f"s{i}{DUMP_SUFFIX}"),
                pd.DataFrame({"n": np.asarray(s, dtype=np.int64)}),
                0,
            )
            for i, s in enumerate(streams)
        ]
        tok.train_tokenizer(dfs)
        return dict(tok.tkmodel.get_vocab()), tok


def _boundary_crossing_pieces(vocab, tok):
    """Pieces (vocab keys, skipping <unk>) decoding to >1 atom that include a BOUNDARY_ISOLATION_NS id."""
    bset = set(dataset.BOUNDARY_ISOLATION_NS)
    crossing = []
    for piece in vocab:
        if piece == "<unk>":
            continue
        ids = [int(x) for x in tok.decode_unicode(piece)]
        if len(ids) > 1 and any(i in bset for i in ids):
            crossing.append(ids)
    return crossing


def test_boundary_isolation_keeps_voice_atoms_unmerged():
    """With bpe_isolate_boundaries on, no unigram merge welds across a VOICE/KEYFRAME boundary: every
    vocab piece carrying a BOUNDARY_ISOLATION_NS id is a single atom. Isolation-off is the A/B control
    (only required to train green and to show welds occur), proving the isolation made the difference.
    """
    streams = [dataset.dump_token_ids(_multi_voice_df(k)) for k in range(20)]
    vocab_off, _tok_off = _train_isolation(streams, isolate=False)
    vocab_on, tok_on = _train_isolation(streams, isolate=True)
    assert (
        _boundary_crossing_pieces(vocab_on, tok_on) == []
    ), "isolation must leave no voice/KEYFRAME-crossing merges"
    bset = set(dataset.BOUNDARY_ISOLATION_NS)
    on_has_boundary = any(
        int(x) in bset
        for piece in vocab_on
        if piece != "<unk>"
        for x in tok_on.decode_unicode(piece)
    )
    assert on_has_boundary, "boundary atoms must appear (else the check is vacuous)"
