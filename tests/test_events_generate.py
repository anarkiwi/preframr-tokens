"""Events generation-decode guards (REDESIGN_optionB §7.1, step 4): a generated token stream decodes back
to the exact ordered writes, and the render-ready dump DataFrame carries the right per-frame timing.
"""

import types

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset, generate, oracle, stream


def _args():
    return types.SimpleNamespace(tokenizer="unigram", tkvocab=0)


def _synth_df():
    writes = []
    for f in range(60):
        fr = 0x1240 + (6 * ((f % 4) - 2) if f >= 8 else 0)
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, (fr >> 8) & 0xFF),
            (f, 4, 0x41 if 4 <= f < 30 else 0x40),
            (f, 5, 8),
            (f, 6, 0xA9),
            (f, 2, (64 + f) & 0xFF),
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


def test_generated_tokens_decode_canonical():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    ids = dataset.dump_token_ids(df)
    writes = generate.tokens_to_writes(tk, ids)
    assert writes == stream.canonical_writes(oracle.ordered_writes(df))


def test_generated_tokens_tolerate_trailing_pad():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    ids = dataset.dump_token_ids(df) + [0, 0, 0]
    assert generate.tokens_to_writes(tk, ids) == stream.canonical_writes(
        oracle.ordered_writes(df)
    )


def test_render_df_carries_frame_timing():
    df = _synth_df()
    tk = dataset.make_tokenizer(_args())
    out = generate.tokens_to_dump_df(tk, dataset.dump_token_ids(df))
    assert list(out.columns) == ["clock", "irq", "chipno", "reg", "val"]
    src = stream.canonical_writes(oracle.ordered_writes(df))
    assert out["irq"].tolist() == [f for f, _, _ in src], "irq is the per-write frame"
    assert out["reg"].tolist() == [r for _, r, _ in src]
    assert out["val"].tolist() == [v for _, _, v in src]
    assert out["clock"].is_monotonic_increasing
