"""PreGateFreqPass unit tests on synthetic dfs. A freq written before a voice's first gate-on is
dropped (when the first note sets its own freq) or relocated into the gate-on frame (when it does not);
either way the AUDIBLE region -- the register_state from the gate-on frame onward -- is byte-identical,
while only the silent pre-gate frames change. OFF is a no-op."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.pre_gate_freq_pass import PreGateFreqPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE, FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import FRAME_REG, SET_OP

_C0 = int(CTRL_REGS_BY_VOICE[0])
_F0 = int(FREQ_REGS_BY_VOICE[0])


def _args(**over):
    cfg = dict(pre_gate_freq=True)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _row(reg, val, op=SET_OP, subreg=-1):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": 32,
        "op": int(op),
        "subreg": int(subreg),
        "irq": 100,
        "description": 0,
    }


def _audible_region_preserved(raw, enc, first_on_frame):
    """register_state from the gate-on frame onward (row first_on+1 = after that frame) is identical."""
    rs = register_state(raw)
    es = register_state(enc)
    tail = first_on_frame + 1
    assert rs[tail:].shape == es[tail:].shape, (rs.shape, es.shape)
    assert np.array_equal(
        rs[tail:], es[tail:]
    ), "audible region (gate-on onward) must be preserved"


def test_drop_when_gate_on_frame_has_its_own_freq():
    raw = pd.DataFrame(
        [
            _row(FRAME_REG, 0),
            _row(_F0, 450),
            _row(FRAME_REG, 0),
            _row(_F0, 800),
            _row(_C0, 0x41),
            _row(FRAME_REG, 0),
        ]
    )
    enc = PreGateFreqPass().apply(raw.copy(), args=_args())
    pre = enc[(enc["reg"] == _F0) & (enc["val"] == 450)]
    assert len(pre) == 0, "pre-gate freq dropped (the note sets its own freq)"
    assert int((enc["reg"] == _F0).sum()) == 1, "the note's own freq write remains"
    _audible_region_preserved(raw, enc, first_on_frame=1)


def test_relocate_when_gate_on_frame_has_no_freq():
    raw = pd.DataFrame(
        [
            _row(FRAME_REG, 0),
            _row(_F0, 450),
            _row(FRAME_REG, 0),
            _row(_C0, 0x41),
            _row(FRAME_REG, 0),
        ]
    )
    enc = PreGateFreqPass().apply(raw.copy(), args=_args())
    assert int((enc["reg"] == _F0).sum()) == 1, "freq relocated, not duplicated"
    frames = (enc["reg"].isin({FRAME_REG})).cumsum()
    freq_frame = int(frames[(enc["reg"] == _F0)].iloc[0])
    gate_frame = int(frames[(enc["reg"] == _C0)].iloc[0])
    assert freq_frame == gate_frame, "freq moved into the gate-on frame"
    _audible_region_preserved(raw, enc, first_on_frame=1)


def test_no_gate_on_leaves_freq_literal():
    raw = pd.DataFrame([_row(FRAME_REG, 0), _row(_F0, 450), _row(FRAME_REG, 0)])
    enc = PreGateFreqPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["reg"] == _F0).sum()) == 1
    ), "no gate-on -> nothing is pre-gate -> kept"


def test_off_is_noop():
    raw = pd.DataFrame(
        [_row(FRAME_REG, 0), _row(_F0, 450), _row(FRAME_REG, 0), _row(_C0, 0x41)]
    )
    out = PreGateFreqPass().apply(raw.copy(), args=_args(pre_gate_freq=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
