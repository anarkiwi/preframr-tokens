"""Contract for the sequencing backward-reference (LZ-over-events) layer: a repeated
event span is copied by a backward-only SEQREF (distance capped at
:data:`preframr_tokens.events.seqref.MAX_REF_DISTANCE`), the parse round-trips the event
list exactly, the serialized literal-only stream equals the no-ref serialization, and
the whole codec stays byte-exact through the real parse output."""

import numpy as np
import pandas as pd

from preframr_tokens.events import inline, seqref, stream
from preframr_tokens.events.oracle import (
    corrected_writes,
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


def _looping_df(n_notes=24, period=8):
    """A voice that loops the same 4-note phrase, so the event stream has a long
    repeated span the LZ can copy backward."""
    writes = []
    phrase = [(0x1200, 0x41), (0x1300, 0x41), (0x1400, 0x41), (0x1500, 0x41)]
    for i in range(n_notes):
        f = i * period
        fr, ctrl = phrase[i % len(phrase)]
        writes += [
            (f, 0, fr & 0xFF),
            (f, 1, fr >> 8),
            (f, 4, ctrl),
            (f, 5, 0x0A),
            (f + period - 1, 4, 0x40),
        ]
    return _df(writes)


def test_backward_ref_copies_repeated_span():
    """A looping phrase makes the LZ emit at least one backward REF, and the parse
    round-trips the event list exactly."""
    df = _looping_df()
    ow = ordered_writes(df)
    events = seqref.tune_events(settled_grid(ow), env_writes(ow))
    parsed = seqref.lz_parse(events)
    assert any(it[0] == "REF" for it in parsed)
    assert seqref.lz_decode(parsed) == events


def test_ref_distance_capped():
    """No emitted reference exceeds the distance cap."""
    df = _looping_df(n_notes=40)
    ow = ordered_writes(df)
    events = seqref.tune_events(settled_grid(ow), env_writes(ow))
    parsed = seqref.lz_parse(events, window=seqref.MAX_REF_DISTANCE)
    assert all(it[1] <= seqref.MAX_REF_DISTANCE for it in parsed if it[0] == "REF")


def test_literal_serialization_matches_no_ref():
    """The literal-only serialization equals serializing every event as a literal (the
    REF op is purely additive over the same byte stream)."""
    df = _looping_df()
    ow = ordered_writes(df)
    events = seqref.tune_events(settled_grid(ow), env_writes(ow))
    lit = seqref.serialize_events(events)
    all_lit = seqref.serialize_parsed([("LIT", ev) for ev in events])
    assert lit == all_lit


def test_seqref_roundtrip_is_byte_exact():
    """The full encode (instrument + sequencing) decodes byte-exact to the corrected
    target on a real parse output."""
    df = _looping_df()
    ow = ordered_writes(df)
    ids = stream.encode(ow)
    assert stream.decode(ids) == corrected_writes(ow)
    assert max(ids) < inline.VOCAB_SIZE


def test_seqref_marker_present_in_stream():
    df = _looping_df()
    ids = stream.encode(ordered_writes(df))
    assert inline.SEQREF_OP in ids
