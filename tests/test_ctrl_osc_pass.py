"""CtrlOscPass unit + round-trip tests on synthetic dfs (no HVSC). A per-frame ctrl oscillation of
>= CTRL_OSC_MIN_LEN consecutive frames cycling through P distinct bytes drains to one byte-exact
CTRL_OSC atom; replay is the exact cycle; a constant run (period 1), a short run, and a run broken by a
held frame are left alone; ctrl_osc OFF is a no-op.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.ctrl_osc_pass import CtrlOscPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import CTRL_OSC_OP, FRAME_REG, SET_OP

_IRQ = 19656
_CTRL_REG = CTRL_REGS_BY_VOICE[0]


def _args(**over):
    cfg = dict(ctrl_osc=True)
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
    """Per-frame ctrl write stream (every frame writes ctrl, the way a gate/waveform oscillation pokes
    it)."""

    def __init__(self):
        self.rows = []

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def cycle(self, vals, n, reg=_CTRL_REG):
        for k in range(n):
            self._frame()
            self.rows.append(_row(reg, vals[k % len(vals)]))
        return self

    def hold(self, n=2):
        for _ in range(n):
            self._frame()
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


def _roundtrip_exact(raw, args):
    enc = CtrlOscPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "ctrl-osc replay must be byte-exact"
    return enc


def test_two_state_oscillation_drains_and_roundtrips():
    raw = _Builder().cycle([0x41, 0x40], 12).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == CTRL_OSC_OP).sum()) > 0, "one CTRL_OSC descriptor"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _CTRL_REG)).sum()) == 0


def test_three_state_oscillation_roundtrips():
    raw = _Builder().cycle([0x21, 0x41, 0x11], 12).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == CTRL_OSC_OP).sum()) > 0


def test_long_oscillation_roundtrips():
    """A long run drains to one atom with a large LEN; the per-frame drain stays register-exact."""
    raw = _Builder().cycle([0x41, 0x40], 40).df()
    _roundtrip_exact(raw, _args())


def test_constant_run_left_alone():
    raw = _Builder().cycle([0x41], 12).df()
    enc = CtrlOscPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == CTRL_OSC_OP).sum()) == 0
    ), "period-1 hold is not an oscillation"


def test_short_run_left_alone():
    raw = _Builder().cycle([0x41, 0x40], 3).df()
    enc = CtrlOscPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == CTRL_OSC_OP).sum()) == 0
    ), "below CTRL_OSC_MIN_LEN -> not mined"


def test_held_frame_roundtrips():
    """A multi-frame hold mid-oscillation (ffilled across the gap) still drains register-exactly."""
    raw = _Builder().cycle([0x41, 0x40], 4).hold(3).cycle([0x41, 0x40], 4).df()
    _roundtrip_exact(raw, _args())


def test_ctrl_osc_off_is_noop():
    raw = _Builder().cycle([0x41, 0x40], 12).df()
    out = CtrlOscPass().apply(raw.copy(), args=_args(ctrl_osc=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
