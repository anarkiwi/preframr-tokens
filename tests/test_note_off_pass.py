"""NoteOffPass unit + round-trip tests on synthetic dfs (no HVSC). A ctrl write whose gate bit falls
1->0 is re-labelled as a NOTE_OFF atom that decodes byte-identically; a gate-on, a gate-already-low
write, and the first (no prior gate-on) clear are left alone; note_off OFF is a no-op.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.note_off_pass import NoteOffPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import FRAME_REG, NOTE_OFF_OP, NOTE_ON_OP, SET_OP

_IRQ = 19656
_CTRL = CTRL_REGS_BY_VOICE[0]


def _args(**over):
    cfg = dict(note_off=True)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _row(reg, val, op=SET_OP, subreg=-1):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": _IRQ,
        "op": int(op),
        "subreg": int(subreg),
        "irq": _IRQ,
        "description": 0,
    }


class _Builder:
    def __init__(self):
        self.rows = []

    def write(self, reg, val):
        self.rows.append(_row(FRAME_REG, 0))
        self.rows.append(_row(reg, val))
        return self

    def df(self):
        self.rows.append(_row(FRAME_REG, 0))
        return pd.DataFrame(self.rows)


def _roundtrip_exact(raw, args):
    enc = NoteOffPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "note-off replay must be byte-exact"
    return enc


def test_gate_fall_is_tagged_and_roundtrips():
    raw = _Builder().write(_CTRL, 0x41).write(_CTRL, 0x40).df()
    enc = _roundtrip_exact(raw, _args())
    assert (
        int((enc["op"] == NOTE_OFF_OP).sum()) == 1
    ), "the gate-clear write is a NOTE_OFF"
    assert int(((enc["op"] == NOTE_OFF_OP) & (enc["val"] == 0x40)).sum()) == 1


def test_full_gate_off_to_zero_tagged():
    raw = _Builder().write(_CTRL, 0x41).write(_CTRL, 0x00).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == NOTE_OFF_OP).sum()) == 1


def test_gate_on_not_tagged():
    raw = _Builder().write(_CTRL, 0x40).write(_CTRL, 0x41).df()
    enc = NoteOffPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == NOTE_OFF_OP).sum()) == 0
    ), "a gate-on rise is not a note-off"


def test_gate_rise_tagged_note_on():
    raw = _Builder().write(_CTRL, 0x41).write(_CTRL, 0x40).write(_CTRL, 0x41).df()
    enc = _roundtrip_exact(raw, _args(note_on=True))
    assert (
        int((enc["op"] == NOTE_ON_OP).sum()) == 1
    ), "the gate-rise after a fall is a NOTE_ON"
    assert (
        int((enc["op"] == NOTE_OFF_OP).sum()) == 1
    ), "the gate-fall is still a NOTE_OFF"


def test_first_clear_without_prior_gate_on_not_tagged():
    raw = _Builder().write(_CTRL, 0x40).write(_CTRL, 0x00).df()
    enc = NoteOffPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == NOTE_OFF_OP).sum()) == 0
    ), "no prior gate-on -> not a release"


def test_multiple_notes_each_tagged():
    raw = (
        _Builder()
        .write(_CTRL, 0x41)
        .write(_CTRL, 0x40)
        .write(_CTRL, 0x41)
        .write(_CTRL, 0x40)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == NOTE_OFF_OP).sum()) == 2


def test_note_off_off_is_noop():
    raw = _Builder().write(_CTRL, 0x41).write(_CTRL, 0x40).df()
    out = NoteOffPass().apply(raw.copy(), args=_args(note_off=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
