"""Events generation-decode guards: a generated token stream decodes back
to the exact ordered writes, and the render-ready dump DataFrame carries the right per-frame timing.
"""

import types

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset, generate, oracle, stream, varint
from preframr_tokens.macros import pitch_grid


def _args():
    return types.SimpleNamespace(tokenizer="unigram", tkvocab=0)


def _ow_4frame():
    """A 4-frame voice-0 tune whose non-arithmetic notes stay one event-group per frame (no single-
    ramp collapse), so the last group is at frame 3 and a DT=1 continuation lands at frame 4.
    """
    writes = []
    for f, nt in enumerate((49, 56, 50, 58)):
        fr = pitch_grid.note_freq_at(nt, 0.0)
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF), (f, 4, 0x41)]
    writes.sort(key=lambda t: t[0])
    return oracle.OrderedWrites(
        frame=np.array([w[0] for w in writes], dtype=np.int64),
        reg=np.array([w[1] for w in writes], dtype=np.int64),
        val=np.array([w[2] for w in writes], dtype=np.int64),
        n_frames=4,
        irq=np.arange(4, dtype=np.int64),
    )


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


def test_decode_extend_replays_past_declared_frame_count():
    """decode/generate default truncate at the declared frame count; extend=True replays a
    hand-appended continuation group (DT=1, voice-0 NI step +2), adding writes at the new frame.
    """
    ow = _ow_4frame()
    toks = stream.encode(ow)
    extra = (
        [stream.VAR_BASE + d for d in varint.encode_unsigned(1)]
        + [stream.VOICE_BASE + 0, stream.NI_STEP]
        + [stream.VAR_BASE + d for d in varint.encode_signed(2)]
    )
    ext = toks + extra
    assert stream.decode(ext) == stream.decode(
        toks
    ), "default truncates the extra group"
    assert any(
        f == 4 for f, _, _ in stream.decode(ext, extend=True)
    ), "extend replays the appended frame-4 group"
    tk = dataset.make_tokenizer(_args())
    nspace = [a + 1 for a in ext]
    assert generate.tokens_to_writes(tk, nspace) == stream.decode(toks)
    assert any(
        f == 4 for f, _, _ in generate.tokens_to_writes(tk, nspace, extend=True)
    ), "generate extend pass-through reaches frame 4"
