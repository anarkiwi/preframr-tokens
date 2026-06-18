"""Grammar-mask contract for the inline-event alphabet: a fresh EventStreamState
drives a real encoded stream atom by atom (every atom legal at its step), rejects
out-of-grammar atoms, flags event boundaries, and its valid_mask exactly matches
the stream decoder's field reads for DT / LANE / OP / param phases."""

import numpy as np
import pytest

from preframr_tokens.events import inline
from preframr_tokens.events.constrained import EventStreamState, valid_first_atoms


def _stream():
    g = np.zeros((40, inline.NUM_REGS), dtype=np.int64)
    g[2:, 0] = 0x34
    g[2:, 1] = 0x12
    for f in range(10, 30):
        val = (0x1000 + (f - 10) * 5) & 0xFFFF
        g[f, 0] = val & 0xFF
        g[f, 1] = val >> 8
    for f in range(30, 40):
        g[f, 24] = 0x0F if f % 2 else 0x00
    g[5:, 4] = 0x41
    return inline.encode_grid(g)


def test_fresh_state_expects_dt_digits_only():
    m = valid_first_atoms()
    assert m.shape[0] == inline.VOCAB_SIZE
    assert all(m[d] for d in range(inline.DIGIT_BASE, inline.VOCAB_SIZE))
    assert not any(m[lane] for lane in range(inline.NUM_LANES))
    assert not m[inline.NOTE_OP]


def test_drives_full_stream_with_boundaries():
    ids = _stream()
    st = EventStreamState()
    boundaries = 0
    for tok in ids:
        assert st.valid_mask()[tok], (tok, st.phase)
        st.push(tok)
        if st.at_group_boundary:
            boundaries += 1
    assert boundaries == len(inline.ids_to_events(ids))


def test_rejects_out_of_grammar_atom():
    st = EventStreamState()
    st.push(inline.DIGIT_BASE + 16)
    with pytest.raises(ValueError):
        st.push(inline.NOTE_OP)


def test_lane_then_op_phase_sequence():
    st = EventStreamState()
    st.push(inline.DIGIT_BASE + 16)
    m = st.valid_mask()
    assert all(m[inline.LANE_BASE + lane] for lane in range(inline.NUM_LANES))
    assert not m[inline.NOTE_OP]
    st.push(inline.LANE_BASE + 0)
    m = st.valid_mask()
    assert m[inline.NOTE_OP] and m[inline.LOAD_OP]
    assert m[inline.MOD_OP] and m[inline.RUN_OP]
    assert not m[inline.LANE_BASE + 0]


@pytest.mark.parametrize("op", ["NOTE", "LOAD"])
def test_single_value_ops_complete_after_one_varint(op):
    st = EventStreamState()
    ids = inline.events_to_ids([(3, 1, (op, 7))])
    for tok in ids:
        assert st.valid_mask()[tok]
        st.push(tok)
    assert st.at_group_boundary


@pytest.mark.parametrize("op", ["MOD", "RUN"])
def test_run_ops_complete_after_all_deltas(op):
    st = EventStreamState()
    ids = inline.events_to_ids([(0, 0 if op == "MOD" else 5, (op, (1, -2, 3), 9))])
    for tok in ids[:-1]:
        st.push(tok)
        assert not st.at_group_boundary
    st.push(ids[-1])
    assert st.at_group_boundary
