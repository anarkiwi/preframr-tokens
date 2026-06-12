"""Acceptance guards for the event primitives that survived the v0/factored-codec removal: the
escape-free varint codec, the lossless gesture basis, the ordered-write oracle, and the SID register-map
helpers. The stream codec's own byte-exact roundtrip lives in test_events_roundtrip / test_events_stream.
"""

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import gestures, oracle, schema, varint
from preframr_tokens.events.gestures import Gesture


def test_no_escape_every_value_decodes_over_one_alphabet():
    """Every numeric field is the complete escape-free zig-zag/varint over ONE digit alphabet -- a rare
    large value is just more digits of the same alphabet, never a different path."""
    for v in list(range(-300, 301)) + [-70000, 65535, 1 << 20]:
        assert varint.unzigzag(varint.decode_unsigned(varint.encode_signed(v))[0]) == v


def test_gesture_basis_is_lossless_for_all_shapes():
    """Every gesture shape (HOLD/POLY/PERIOD) replays its source series exactly (the lossless cover)."""
    rng = np.random.default_rng(0)
    for s in (
        np.full(30, 7),
        np.arange(30) * 3 + 2,
        np.cumsum(rng.integers(-4, 5, 40)) + 100,
    ):
        s = s.astype(np.int64)
        assert (gestures.replay(gestures.cover(s), len(s)) == s).all()


def test_ordered_writes_oracle_chip0_clock_sorted():
    """ordered_writes keeps chipno==0, drops regs>24, clock-sorts (stable), and maps irq to dense
    frames; triples() is the byte-exact target and by_frame groups per dense frame."""
    df = pd.DataFrame(
        {
            "clock": [2, 1, 5, 6, 4],
            "irq": [100, 100, 100, 200, 200],
            "chipno": [0, 0, 1, 0, 0],
            "reg": [1, 0, 4, 30, 0],
            "val": [0x22, 0x11, 0x99, 0x88, 0x33],
        }
    )
    ow = oracle.ordered_writes(df)
    assert ow.n_frames == 2
    assert ow.triples() == [(0, 0, 0x11), (0, 1, 0x22), (1, 0, 0x33)]
    assert ow.by_frame() == [[(0, 0x11), (1, 0x22)], [(0, 0x33)]]


def test_schema_register_map_helpers():
    """The voice register-map helpers address the per-voice SID layout and the gate bit; Shape is the
    gesture-basis enum imported by gestures."""
    assert schema.freq_regs(1) == (7, 8)
    assert schema.pw_regs(2) == (16, 17)
    assert (schema.ctrl_reg(0), schema.ad_reg(0), schema.sr_reg(0)) == (4, 5, 6)
    assert schema.gate_on(0x41) and not schema.gate_on(0x40)
    assert (schema.Shape.HOLD, schema.Shape.POLY, schema.Shape.PERIOD) == (0, 1, 2)


def test_replay_one_covers_poly_wrap_and_period():
    """replay_one reconstructs a 16-bit-wrapping POLY and a PERIOD gesture and rejects an unknown
    shape (the per-gesture inverse of the cover split, exercised without the full cover/parse).
    """
    poly = Gesture(schema.Shape.POLY, 0, 3, (0xFFFF, 2))
    assert gestures.replay_one(poly, wrap=True) == [0xFFFF, 1, 3]
    period = Gesture(schema.Shape.PERIOD, 0, 5, (100, 2, -1))
    assert gestures.replay_one(period) == [100, 102, 101, 103, 102]
    period_wrap = Gesture(schema.Shape.PERIOD, 0, 3, (0xFFFF, 2))
    assert gestures.replay_one(period_wrap, wrap=True) == [0xFFFF, 1, 3]
    with pytest.raises(ValueError):
        gestures.replay_one(Gesture(99, 0, 1, (0,)))


def test_varint_rejects_negative_and_truncated():
    """The unsigned codec rejects a negative input and a continue-bit-terminated truncation (no escape,
    no silent tolerance)."""
    with pytest.raises(ValueError):
        varint.encode_unsigned(-1)
    with pytest.raises(ValueError):
        varint.decode_unsigned([varint.CONT | 1])
