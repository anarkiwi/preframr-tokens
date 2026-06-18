"""Byte-exact contract for the inline-event codec: every lane kind and every op
(NOTE/MOD relative pitch, LOAD/RUN delta-run) round-trips a settled register grid
exactly, the flat-atom serialization is self-delimiting, the alphabet has no SET op
and no escape, and any prefix of the token stream is a valid, decodable song."""

import numpy as np
import pytest

from preframr_tokens.events import inline, stream
from preframr_tokens.events.oracle import settled_grid, writes_to_ordered


def _grid_from_writes(writes, n_frames):
    g = np.zeros((n_frames, inline.NUM_REGS), dtype=np.int64)
    cur = np.zeros(inline.NUM_REGS, dtype=np.int64)
    k = 0
    for f in range(n_frames):
        while k < len(writes) and writes[k][0] == f:
            cur[writes[k][1]] = writes[k][2]
            k += 1
        g[f] = cur
    return g


def _rand_grid(seed, n_frames=200):
    rng = np.random.default_rng(seed)
    g = np.zeros((n_frames, inline.NUM_REGS), dtype=np.int64)
    for v in range(3):
        b = 7 * v
        freq = 0
        pw = 0
        for f in range(n_frames):
            if rng.random() < 0.1:
                freq = int(rng.integers(0, 0x10000))
            elif rng.random() < 0.2:
                freq = (freq + int(rng.integers(-32, 33))) & 0xFFFF
            g[f, b + 0] = freq & 0xFF
            g[f, b + 1] = freq >> 8
            if rng.random() < 0.1:
                pw = int(rng.integers(0, 0x1000))
            g[f, b + 2] = pw & 0xFF
            g[f, b + 3] = pw >> 8
            g[f, b + 4] = (
                int(rng.integers(0, 256))
                if rng.random() < 0.15
                else (g[f - 1, b + 4] if f else 0)
            )
            g[f, b + 5] = g[f - 1, b + 5] if f else 0
            g[f, b + 6] = g[f - 1, b + 6] if f else 0
    for r in (21, 22, 23, 24):
        val = 0
        for f in range(n_frames):
            if rng.random() < 0.1:
                val = int(rng.integers(0, 256))
            g[f, r] = val
    return g


def test_vocab_has_no_set_op_and_no_escape():
    """The alphabet is LANE + {NOTE, LOAD, MOD, RUN} + digits -- no SET op, no escape
    token, no id range."""
    assert inline.VOCAB_SIZE == inline.NUM_LANES + 4 + 32
    op_names = {inline.OPS_INV[i] for i in range(4)}
    assert op_names == {"NOTE", "LOAD", "MOD", "RUN"}
    assert "SET" not in op_names


@pytest.mark.parametrize("seed", range(8))
def test_random_grid_round_trips_byte_exact(seed):
    g = _rand_grid(seed)
    ids = inline.encode_grid(g)
    rec = inline.decode_grid(ids, len(g))
    assert np.array_equal(rec, g)
    assert max(ids) < inline.VOCAB_SIZE


@pytest.mark.parametrize("seed", range(4))
def test_stream_decode_equals_canonical(seed):
    g = _rand_grid(seed)
    ow = writes_to_ordered(stream._grid_to_writes(g))
    assert np.array_equal(settled_grid(ow), g)
    ids = stream.encode(ow, verify=True)
    assert stream.decode(ids) == stream.canonical_writes(ow)


def test_each_op_kind_present():
    """A grid with a jump, a held note, a steady sweep, and a periodic vibrato uses
    LOAD, NOTE, RUN/MOD respectively."""
    g = np.zeros((60, inline.NUM_REGS), dtype=np.int64)
    g[2:, 0] = 0x34
    g[2:, 1] = 0x12
    for f in range(10, 30):
        val = (0x1000 + (f - 10) * 7) & 0xFFFF
        g[f, 0] = val & 0xFF
        g[f, 1] = val >> 8
    g[30:, 0] = 0x1000 & 0xFF
    g[30:, 1] = (0x1000 >> 8) & 0xFF
    for f in range(40, 60):
        g[f, 24] = 0x0F if f % 2 else 0x00
    events = inline.encode_events(g)
    ops = {op[0] for _f, _l, op in events}
    assert ops & {"NOTE", "MOD"}
    assert "RUN" in ops or "LOAD" in ops
    assert np.array_equal(inline.decode_grid(inline.events_to_ids(events), len(g)), g)


def test_any_prefix_is_decodable():
    """Inline streaming: every unit-boundary prefix of the token stream decodes to
    the song so far (no preamble, continuable)."""
    g = _rand_grid(1, n_frames=120)
    ids = inline.encode_grid(g)
    starts = stream.unit_starts(ids)
    assert starts and starts[0] == 0
    for s in starts:
        writes = stream.decode(ids[:s]) if s else []
        assert isinstance(writes, list)
    assert stream.decode(ids) == stream.canonical_writes(
        writes_to_ordered(stream._grid_to_writes(g))
    )


def test_serialization_is_self_delimiting():
    """events_to_ids / ids_to_events invert exactly with no separator atom."""
    g = _rand_grid(2, n_frames=80)
    events = inline.encode_events(g)
    ids = inline.events_to_ids(events)
    assert inline.ids_to_events(ids) == events


def test_single_speed_false_on_repeated_reg():
    from preframr_tokens.events.oracle import OrderedWrites

    ow = OrderedWrites(
        frame=np.array([0, 0], dtype=np.int64),
        reg=np.array([0, 0], dtype=np.int64),
        val=np.array([1, 2], dtype=np.int64),
        n_frames=1,
        irq=np.array([0], dtype=np.int64),
    )
    assert not stream.single_speed(ow)


def test_trim_to_decodable_drops_partial_tail():
    g = _rand_grid(3, n_frames=60)
    ids = inline.encode_grid(g)
    keep, writes = stream.trim_to_decodable(ids + [inline.DIGIT_BASE])
    assert keep == ids
    assert writes == stream.decode(ids)
    assert stream.decode_windowed(ids + [inline.DIGIT_BASE]) == stream.decode(ids)


def test_empty_inputs():
    assert stream.decode([]) == []
    assert inline.encode_grid(np.zeros((0, inline.NUM_REGS), dtype=np.int64)) == []
    assert stream.strip_keyframes([1, 2]) == [1, 2]
    assert stream.chunk_keyframe([1], 1) == []


def test_roundtrip_ok_smoke():
    import pandas as pd

    g = _rand_grid(0, n_frames=30)
    writes = stream._grid_to_writes(g)
    df = pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )
    assert stream.roundtrip_ok(df)


def test_single_speed_empty_is_true():
    from preframr_tokens.events.oracle import OrderedWrites

    empty = OrderedWrites(
        frame=np.empty(0, np.int64),
        reg=np.empty(0, np.int64),
        val=np.empty(0, np.int64),
        n_frames=0,
        irq=np.empty(0, np.int64),
    )
    assert stream.single_speed(empty)
    assert stream.unit_starts([]) == []
    assert stream.trim_to_decodable([]) == ([], [])


def test_encode_verify_raises_on_corrupt_codec(monkeypatch):
    from preframr_tokens.events import inline as inline_mod
    from preframr_tokens.events.oracle import writes_to_ordered

    g = _rand_grid(0, n_frames=20)
    ow = writes_to_ordered(stream._grid_to_writes(g))
    monkeypatch.setattr(
        inline_mod, "decode_grid", lambda ids, n: np.zeros((n, 25), dtype=np.int64)
    )
    with pytest.raises(ValueError):
        stream.encode(ow, verify=True)


def test_oracle_empty_and_writes_to_ordered():
    import pandas as pd

    from preframr_tokens.events.oracle import ordered_writes, writes_to_ordered

    empty = ordered_writes(
        pd.DataFrame(
            {"clock": [], "irq": [], "chipno": [], "reg": [], "val": []}
        ).astype(np.int64)
    )
    assert empty.n_frames == 0
    assert writes_to_ordered([]).n_frames == 0
    ow = writes_to_ordered([(0, 0, 5), (1, 0, 6)])
    assert ow.n_frames == 2


def test_dump_meta_without_clock_column_uses_irq_frames():
    import pandas as pd

    from preframr_tokens.dump_meta import _build_meta_from_raw

    df = pd.DataFrame(
        {
            "irq": np.repeat(np.arange(1, 11), 3),
            "chipno": 0,
            "reg": np.tile([0, 1, 4], 10),
            "val": 1,
        }
    )
    meta = _build_meta_from_raw(None, df)
    assert meta["is_digi"] is False
    assert meta["n_frames"] == 10
