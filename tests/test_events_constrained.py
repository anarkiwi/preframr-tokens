"""Grammar-mask contract for the inline-event alphabet: a fresh EventStreamState drives
a real encoded stream atom by atom (every atom legal at its step), rejects out-of-grammar
atoms, and flags exactly one boundary per serialized unit (lane gestures, instrument
items, voice selectors, SEQREF copies)."""

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import inline, seqref, stream
from preframr_tokens.events.constrained import EventStreamState, valid_first_atoms
from preframr_tokens.events.oracle import (
    env_writes,
    ordered_writes,
    settled_grid,
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


def _looping_df(n_notes=20, period=8):
    writes = []
    phrase = [0x1200, 0x1300, 0x1400, 0x1500]
    for i in range(n_notes):
        f = i * period
        fr = phrase[i % len(phrase)]
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, fr >> 8),
            (f, 4, 0x41),
            (f, 5, 0x0A),
            (f + period - 1, 4, 0x40),
            (f, 9, (64 + i) & 0xFF),
            (f, 24, i % 16),
        ]
    return _df(writes)


def test_fresh_state_expects_unit_heads():
    """At a unit boundary the legal atoms are a unit head (SEQREF/voice/LEAD) or a DT
    digit, never an OP or a non-env lane selector."""
    m = valid_first_atoms()
    assert m.shape[0] == inline.VOCAB_SIZE
    assert all(m[d] for d in range(inline.DIGIT_BASE, inline.VOCAB_SIZE))
    assert m[inline.SEQREF_OP]
    assert m[inline.VOICE_BASE] and m[inline.LEAD_ITEM]
    assert not m[inline.NOTE_OP]
    assert not m[inline.LANE_BASE + 0]


def test_drives_full_stream_with_boundaries():
    """Every atom of a real encoded stream is legal at its step, and the number of
    boundaries equals the number of serialized units (literals + SEQREF copies)."""
    ow = ordered_writes(_looping_df())
    ids = stream.encode(ow)
    parsed = seqref.lz_parse(seqref.tune_events(settled_grid(ow), env_writes(ow)))
    st = EventStreamState()
    boundaries = 0
    for tok in ids:
        assert st.valid_mask()[tok], (tok, st.phase)
        st.push(tok)
        if st.at_group_boundary:
            boundaries += 1
    assert st.phase == "START"
    assert boundaries == len(parsed)


def test_rejects_out_of_grammar_atom():
    st = EventStreamState()
    st.push(inline.DIGIT_BASE + 16)
    with pytest.raises(ValueError):
        st.push(inline.NOTE_OP)


def test_dt_then_lane_then_op_phase_sequence():
    st = EventStreamState()
    st.push(inline.DIGIT_BASE + 0)
    st.push(inline.DIGIT_BASE + 16)
    m = st.valid_mask()
    assert all(m[inline.LANE_BASE + lane] for lane in range(inline.NUM_NONENV))
    assert m[inline.RAW_ITEM] and m[inline.REF_ITEM] and m[inline.DEF_ITEM]
    assert not m[inline.NOTE_OP]
    st.push(inline.LANE_BASE + 0)
    m = st.valid_mask()
    assert m[inline.NOTE_OP] and m[inline.LOAD_OP]
    assert m[inline.MOD_OP] and m[inline.RUN_OP]
    assert not m[inline.LANE_BASE + 0]


@pytest.mark.parametrize("op", ["NOTE", "LOAD"])
def test_single_value_ops_complete_after_one_varint(op):
    st = EventStreamState()
    ids = seqref.serialize_events([("NE", 3, 1, (op, 7))])
    for tok in ids[:-1]:
        st.push(tok)
        assert not st.at_group_boundary
    st.push(ids[-1])
    assert st.at_group_boundary


@pytest.mark.parametrize("op", ["MOD", "RUN"])
def test_run_ops_complete_after_all_deltas(op):
    st = EventStreamState()
    lane = 0 if op == "MOD" else 5
    ids = seqref.serialize_events([("NE", 0, lane, (op, (1, -2, 3), 9))])
    for tok in ids[:-1]:
        st.push(tok)
        assert not st.at_group_boundary
    st.push(ids[-1])
    assert st.at_group_boundary


def test_voice_selector_is_a_complete_unit():
    st = EventStreamState()
    st.push(inline.VOICE_BASE + 1)
    assert st.at_group_boundary
    assert st.phase == "START"


def test_seqref_completes_after_distance_and_length():
    st = EventStreamState()
    ids = seqref.serialize_parsed([("REF", 5, 9)])
    for tok in ids[:-1]:
        st.push(tok)
        assert not st.at_group_boundary
    st.push(ids[-1])
    assert st.at_group_boundary


def test_def_item_completes_after_head_and_tail():
    st = EventStreamState()
    ev = ("DEF", 2, 16, ((0, 4, 0x41), (0, 5, 0x0A)), ((-1, 4, 0x40),))
    ids = seqref.serialize_events([ev])
    for tok in ids[:-1]:
        st.push(tok)
        assert not st.at_group_boundary
    st.push(ids[-1])
    assert st.at_group_boundary
