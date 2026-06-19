"""Byte-exact contract for the per-voice INSTRUMENT program (ctrl/AD/SR env writes):
note instances split into onset-anchored HEAD + note-end-anchored TAIL, DEFined inline
and REFerenced backward at a different duration, with a RAW fallback that keeps the
stream byte-exact. Tests go through the real ``ordered_writes`` parse output, not
synthetic event tuples."""

import numpy as np
import pandas as pd

from preframr_tokens.events import instrument
from preframr_tokens.events.oracle import env_writes, ordered_writes


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


def _byvoice_frame(writes):
    """Per (frame, voice) ordered ``(reg, val)`` -- the per-voice fidelity axis."""
    voice = instrument.VOICE
    out: dict[tuple[int, int], list] = {}
    for f, r, v in writes:
        out.setdefault((f, voice[r]), []).append((r, v))
    return out


def _roundtrip_ok(ew):
    events = instrument.env_events(ew)
    dec_ew, _pw = instrument.decode_env(events)
    return _byvoice_frame(dec_ew) == _byvoice_frame(ew)


def test_def_then_backward_ref_at_different_duration():
    """Identical-shape notes of DIFFERENT durations: the first DEFs the instrument, a
    later same-shape note REFs it backward (duration-abstraction is the point). The last
    note is anchorless (no tail), so a non-terminal middle note carries the REF."""
    plan = ((0, 8), (8, 16), (24, 8), (32, 8))
    writes = []
    for on, dur in plan:
        writes.append((on, 4, 0x41))
        writes.append((on, 5, 0x0A))
        writes.append((on + dur - 1, 6, 0x00))
        writes.append((on + dur - 1, 4, 0x40))
    ow = ordered_writes(_df(writes))
    ew = env_writes(ow)
    events = instrument.encode_voice(ew, 0)
    kinds = [e[0] for e in events if e[0] in ("DEF", "REF")]
    assert kinds[0] == "DEF"
    assert "REF" in kinds
    assert _roundtrip_ok(ew)


def test_raw_fallback_keeps_byte_exact():
    """A note whose writes do not fit the head/tail model falls back to RAW and still
    round-trips byte-exact."""
    writes = [
        (0, 4, 0x41),
        (3, 6, 0x7F),
        (5, 5, 0x22),
        (7, 4, 0x40),
        (8, 4, 0x41),
        (9, 4, 0x40),
    ]
    ow = ordered_writes(_df(writes))
    ew = env_writes(ow)
    assert _roundtrip_ok(ew)


def test_lead_seed_before_first_onset():
    """Env writes before the first gate-on are emitted as a LEAD seed and replayed."""
    writes = [(0, 5, 0x09), (0, 6, 0x00), (2, 4, 0x41), (6, 4, 0x40)]
    ow = ordered_writes(_df(writes))
    ew = env_writes(ow)
    events = instrument.encode_voice(ew, 0)
    assert any(e[0] == "LEAD" for e in events)
    assert _roundtrip_ok(ew)


def test_synced_pw_folds_and_stays_byte_exact():
    """A voice whose PW resets to the same trajectory at each onset is detected as
    SYNCED, its PW folds into the instrument, and the full codec stays byte-exact while
    that voice's pw lane carries no NE gestures."""
    from preframr_tokens.events import stream
    from preframr_tokens.events.oracle import (
        corrected_writes,
        ordered_writes,
        settled_grid,
    )

    period = 10
    writes = []
    for i in range(12):
        on = i * period
        fr = 0x1200 + 0x80 * (i % 3)
        writes += [(on, 0, fr & 0xFF), (on, 1, fr >> 8), (on, 4, 0x41), (on, 5, 0x0A)]
        for k in range(period):
            pw = 0x400 + k * 0x20
            writes += [(on + k, 2, pw & 0xFF), (on + k, 3, (pw >> 8) & 0xF)]
        writes.append((on + period - 1, 4, 0x40))
    ow = ordered_writes(_df(writes))
    assert 0 in instrument.synced_pw_voices(settled_grid(ow), env_writes(ow))
    ids = stream.encode(ow)
    assert stream.decode(ids) == corrected_writes(ow)


def test_free_running_pw_stays_in_lane():
    """A PW whose per-onset trajectory does NOT repeat (each note a different shape) is
    NOT synced, so it is not folded -- and the codec is byte-exact either way."""
    import random

    from preframr_tokens.events import stream
    from preframr_tokens.events.oracle import (
        corrected_writes,
        ordered_writes,
        settled_grid,
    )

    rng = random.Random(7)
    writes = []
    for i in range(10):
        on = i * 16
        writes += [(on, 4, 0x41), (on, 0, 0x12), (on, 1, 0x12)]
        for k in range(15):
            pw = rng.randint(0, 0xFFF)
            writes += [(on + k, 2, pw & 0xFF), (on + k, 3, (pw >> 8) & 0xF)]
        writes.append((on + 15, 4, 0x40))
    ow = ordered_writes(_df(writes))
    assert 0 not in instrument.synced_pw_voices(settled_grid(ow), env_writes(ow))
    assert stream.decode(stream.encode(ow)) == corrected_writes(ow)


def test_split_head_tail_anchors_release_to_note_end():
    """The release block (gate-off near the note end) anchors to the end (negative
    offset), so the same instrument reproduces at any duration."""
    sig = [(0, 4, 0x41), (0, 5, 0x0A), (15, 6, 0x00), (17, 4, 0x40)]
    head, tail = instrument.split_head_tail(sig, on=0, dur=18)
    assert (0, 4, 0x41) in head and (0, 5, 0x0A) in head
    assert any(off < 0 and r == 4 for off, r, _ in tail)
    assert instrument._replay(head, tail, 0, 18) == sig
