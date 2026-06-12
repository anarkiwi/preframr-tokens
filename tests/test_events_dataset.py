"""Events-native tokenizer/alphabet guards: the fixed event-atom alphabet,
the PAD-reserving n-space offset, and byte-exact block<->ids round-trips through the reused RegTokenizer
layer.
"""

import types

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset, oracle, pipeline, stream


def _args():
    return types.SimpleNamespace(tokenizer="unigram", tkvocab=0)


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


def test_keyframe_prefixes_make_chunks_self_interpreting():
    """With a roomy block size, every chunk after the first is led by a KEYFRAME conditioning
    segment carrying the tune's tick/tuning headers + per-voice state; segments strip away for
    decode (the canonical stream stays redundancy-free)."""
    frames = []
    for rep in range(6):
        d = _synth_df()
        d["irq"] = d["irq"] + rep * 60
        d["clock"] = d["clock"] + rep * 100000
        frames.append(d)
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    tk = dataset.make_tokenizer(_args())
    arr = dataset.encode_block_array(tk, df, 128)
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
