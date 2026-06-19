"""Byte-exact contract for the corrected inline-event codec: settled non-env lanes
round-trip as NOTE/MOD pitch and LOAD/RUN delta-runs, while ctrl/AD/SR round-trip as
the ORDERED write stream (including an intra-frame gate toggle settling would erase);
the flat-atom serialization is self-delimiting with no SET op and no escape, and any
prefix of the token stream is a valid decodable song."""

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import inline, stream
from preframr_tokens.events.oracle import (
    OrderedWrites,
    corrected_writes,
    env_writes,
    ordered_writes,
    writes_to_ordered,
)


def _df(writes):
    writes = sorted(writes, key=lambda t: t[0])
    return pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )


def _ow_from_grid(grid):
    """Ordered writes from a settled grid: each frame emits, in ascending register,
    every reg whose value changed from the previous frame (env and non-env alike).
    A grid has no intra-frame order, so this is the canonical single-write-per-frame
    stream -- used to exercise the settled-lane half of the codec."""
    writes = []
    prev = np.zeros(grid.shape[1], dtype=np.int64)
    for f in range(grid.shape[0]):
        for r in range(grid.shape[1]):
            if grid[f, r] != prev[r]:
                writes.append((f, r, int(grid[f, r])))
        prev = grid[f]
    if not writes:
        return OrderedWrites(
            frame=np.empty(0, np.int64),
            reg=np.empty(0, np.int64),
            val=np.empty(0, np.int64),
            n_frames=grid.shape[0],
            irq=np.arange(grid.shape[0], dtype=np.int64),
        )
    return OrderedWrites(
        frame=np.array([w[0] for w in writes], dtype=np.int64),
        reg=np.array([w[1] for w in writes], dtype=np.int64),
        val=np.array([w[2] for w in writes], dtype=np.int64),
        n_frames=grid.shape[0],
        irq=np.arange(grid.shape[0], dtype=np.int64),
    )


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
    """The alphabet is SELECTOR + {NOTE, LOAD, MOD, RUN} + digits -- no SET op, no
    escape token, no id range. Selectors split into non-env lanes + env regs."""
    assert inline.VOCAB_SIZE == inline.NUM_LANES + 4 + 32
    assert inline.NUM_LANES == inline.NUM_NONENV + inline.NUM_ENV
    op_names = {inline.OPS_INV[i] for i in range(4)}
    assert op_names == {"NOTE", "LOAD", "MOD", "RUN"}
    assert "SET" not in op_names


def test_intra_frame_gate_toggle_round_trips_as_ordered_writes():
    """The audio-load-bearing case: a gate-off then gate-on in ONE frame (ctrl reg
    written 0x40 then 0x41). Settling would keep only 0x41; the corrected codec keeps
    BOTH writes in order."""
    writes = [
        (0, 0, 0x34),
        (0, 1, 0x12),
        (0, 4, 0x40),
        (0, 4, 0x41),
        (0, 5, 0x08),
        (0, 6, 0xA9),
        (1, 0, 0x40),
        (1, 4, 0x41),
    ]
    ow = ordered_writes(_df(writes))
    assert (0, 4, 0x40) in env_writes(ow) and (0, 4, 0x41) in env_writes(ow)
    ids = stream.encode(ow)
    dec = stream.decode(ids)
    assert dec == corrected_writes(ow)
    ctrl0 = [w for w in dec if w[0] == 0 and w[1] == 4]
    assert ctrl0 == [(0, 4, 0x40), (0, 4, 0x41)]


def test_intra_frame_hard_restart_sequence_preserved():
    """A hard-restart ADSR sequence written within one frame (AD then SR then ctrl,
    in that exact order) round-trips write-for-write."""
    writes = [
        (0, 0, 0x10),
        (1, 5, 0x00),
        (1, 6, 0x00),
        (1, 4, 0x08),
        (1, 5, 0x0A),
        (1, 6, 0xF9),
        (1, 4, 0x11),
    ]
    ow = ordered_writes(_df(writes))
    ids = stream.encode(ow)
    dec = stream.decode(ids)
    assert dec == corrected_writes(ow)
    frame1_env = [w for w in dec if w[0] == 1 and w[1] in inline.ENV_REGS]
    assert frame1_env == [
        (1, 5, 0x00),
        (1, 6, 0x00),
        (1, 4, 0x08),
        (1, 5, 0x0A),
        (1, 6, 0xF9),
        (1, 4, 0x11),
    ]


def test_env_same_value_noops_deduped():
    """Consecutive same-reg-same-val env writes (inaudible) collapse to one."""
    writes = [(0, 4, 0x41), (1, 4, 0x41), (2, 4, 0x41), (3, 4, 0x40)]
    ow = ordered_writes(_df(writes))
    assert env_writes(ow) == [(0, 4, 0x41), (3, 4, 0x40)]


@pytest.mark.parametrize("seed", range(8))
def test_random_grid_round_trips_byte_exact(seed):
    g = _rand_grid(seed)
    ow = _ow_from_grid(g)
    ids = stream.encode(ow)
    assert stream.decode(ids) == corrected_writes(ow)
    assert max(ids) < inline.VOCAB_SIZE


@pytest.mark.parametrize("seed", range(4))
def test_stream_decode_equals_canonical(seed):
    g = _rand_grid(seed)
    ow = _ow_from_grid(g)
    ids = stream.encode(ow, verify=True)
    assert stream.decode(ids) == stream.canonical_writes(ow)


def test_each_op_kind_present():
    """A grid with a jump, a steady sweep, and a periodic vibrato uses LOAD/NOTE and
    RUN/MOD; the env reg uses ordered WRITE events."""
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
    g[5:, 4] = 0x41
    ow = _ow_from_grid(g)
    events = inline.encode_events(g[:, :], env_writes(ow))
    lane_ops = {op[2][0] for _f, _s, op in events if op[0] == "L"}
    assert lane_ops & {"NOTE", "MOD"}
    assert "RUN" in lane_ops or "LOAD" in lane_ops
    assert any(op[0] == "W" for _f, _s, op in events)


def test_any_prefix_is_decodable():
    """Inline streaming: every unit-boundary prefix of the token stream decodes (no
    preamble, continuable)."""
    g = _rand_grid(1, n_frames=120)
    ow = _ow_from_grid(g)
    ids = stream.encode(ow)
    starts = stream.unit_starts(ids)
    assert starts and starts[0] == 0
    for s in starts:
        writes = stream.decode(ids[:s]) if s else []
        assert isinstance(writes, list)
    assert stream.decode(ids) == corrected_writes(ow)


def test_serialization_is_self_delimiting():
    """events_to_ids / ids_to_events invert exactly with no separator atom."""
    g = _rand_grid(2, n_frames=80)
    ow = _ow_from_grid(g)
    events = inline.encode_events(g, env_writes(ow))
    ids = inline.events_to_ids(events)
    assert inline.ids_to_events(ids) == events


def test_single_speed_false_on_repeated_reg():
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
    ow = _ow_from_grid(g)
    ids = stream.encode(ow)
    keep, writes = stream.trim_to_decodable(ids + [inline.DIGIT_BASE])
    assert keep == ids
    assert writes == stream.decode(ids)
    assert stream.decode_windowed(ids + [inline.DIGIT_BASE]) == stream.decode(ids)


def test_empty_inputs():
    assert stream.decode([]) == []
    assert (
        inline.encode_target(np.zeros((0, inline.NUM_REGS), dtype=np.int64), []) == []
    )
    assert stream.strip_keyframes([1, 2]) == [1, 2]
    assert stream.chunk_keyframe([1], 1) == []


def test_roundtrip_ok_smoke():
    g = _rand_grid(0, n_frames=30)
    ow = _ow_from_grid(g)
    df = _df([(int(f), int(r), int(v)) for f, r, v in ow.triples()])
    assert stream.roundtrip_ok(df)


def test_single_speed_empty_is_true():
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

    g = _rand_grid(0, n_frames=20)
    ow = _ow_from_grid(g)
    monkeypatch.setattr(
        inline_mod,
        "decode_events",
        lambda events, n: (np.zeros((n, 25), dtype=np.int64), []),
    )
    with pytest.raises(ValueError):
        stream.encode(ow, verify=True)


def test_oracle_empty_and_writes_to_ordered():
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
