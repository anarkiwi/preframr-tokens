"""Events-native tokenizer/alphabet guards (REDESIGN_optionB §7.1, step 2): the fixed 68-atom alphabet, the
PAD-reserving n-space offset, and byte-exact block<->ids round-trips through the reused RegTokenizer layer.
"""

import types

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset, oracle, pipeline


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


def test_block_ids_are_pad_safe_and_byte_exact():
    df = _synth_df()
    ow = oracle.ordered_writes(df)
    for w in pipeline.iter_windows(ow, 16):
        ids = dataset.block_to_ids(w)
        assert (
            min(ids) >= 1 and max(ids) <= pipeline.VOCAB_SIZE
        ), "n-space avoids PAD (0)"
        assert dataset.ids_to_writes(ids) == w.triples(), "block ids decode byte-exact"


def test_unicode_serialize_roundtrip():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    ids = dataset.dump_block_ids(df, frames_per_block=16)[0]
    uni = tk.encode_unicode(np.array(ids, dtype=np.uint32))
    assert len(uni) == len(ids)
    assert list(tk.decode_unicode(uni)) == ids


def test_encode_block_array_passthrough_shape():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    block_size = 257
    arr = dataset.encode_block_array(tk, df, block_size, frames_per_block=16)
    assert arr.dtype == np.int32 and arr.shape[1] == block_size
    assert arr.shape[0] == len(
        list(pipeline.iter_windows(oracle.ordered_writes(df), 16))
    )
    assert int(arr.max()) <= pipeline.VOCAB_SIZE
