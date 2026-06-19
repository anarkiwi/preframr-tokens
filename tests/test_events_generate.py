"""Events generation-decode guards: a generated token stream decodes back
to the exact ordered writes, and the render-ready dump DataFrame carries the right per-frame timing.
"""

import types

import numpy as np
import pandas as pd

from preframr_tokens.events import dataset, generate, oracle, stream
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


def test_decode_reflects_inline_continuation():
    """The inline grammar is continuable: any valid event appended to a stream
    extends the song without a separate frame-count or an extend flag. Appending a
    later-frame NOTE event to a freq lane adds writes at the new frame."""
    from preframr_tokens.events import inline

    ow = _ow_4frame()
    toks = stream.encode(ow)
    base_writes = stream.decode(toks)
    base_span = max(f for f, _, _ in base_writes)
    lane = inline.NONENV_LANES.index(("freq", (0, 1)))
    extra = inline.events_to_ids([(4, lane, ("L", lane, ("NOTE", 64)))])
    ext = toks + extra
    ext_writes = stream.decode(ext)
    assert (
        max(f for f, _, _ in ext_writes) > base_span
    ), "appended event extends the song"
    tk = dataset.make_tokenizer(_args())
    nspace = [a + 1 for a in ext]
    gen = generate.tokens_to_writes(tk, nspace)
    assert max(f for f, _, _ in gen) > base_span


def test_recanon_idempotent_and_write_preserving():
    ids = [a + 1 for a in stream.encode(_synth_ow())]
    rc1 = generate.recanon(ids)
    rc2 = generate.recanon(rc1)
    assert rc1 == rc2
    assert dataset.ids_to_writes(rc1) == dataset.ids_to_writes(ids)


def test_recanon_trim_handles_partial_tail():
    from preframr_tokens.events import inline

    ids = [a + 1 for a in stream.encode(_synth_ow())] + [inline.DIGIT_BASE + 1]
    rc = generate.recanon(ids, trim=True)
    assert dataset.ids_to_writes(rc)


def test_tokens_to_dump_df_render_ready():
    ids = [a + 1 for a in stream.encode(_synth_ow())]
    tk = dataset.make_tokenizer(_args())
    df = generate.tokens_to_dump_df(tk, ids)
    assert list(df.columns) == ["clock", "irq", "chipno", "reg", "val"]
    assert (df["chipno"] == 0).all()


def _synth_ow():
    writes = []
    for f in range(40):
        if f >= 2:
            writes.append((f, 0, 0x40))
        if f >= 2:
            writes.append((f, 4, 0x11))
        if 10 <= f < 30:
            v = (0x1000 + (f - 10) * 4) & 0xFFFF
            writes.append((f, 0, v & 0xFF))
            writes.append((f, 1, v >> 8))
        if f >= 20:
            writes.append((f, 21, 7))
    writes.sort(key=lambda t: t[0])
    return oracle.writes_to_ordered(writes)
