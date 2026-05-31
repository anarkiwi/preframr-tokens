"""SweepPass unit + round-trip tests on synthetic dfs (no HVSC). A constant-raw-freq-delta run of
>= SWEEP_MIN_LEN consecutive frames drains to one byte-exact SWEEP atom; replay is the exact ramp; a
short run (< MIN) and a non-constant-delta run are left alone; sweep_pass OFF is a no-op.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.sweep_pass import SweepPass
from preframr_tokens.stfconstants import FRAME_REG, SET_OP, SWEEP_OP

_IRQ = 19656
_FREQ_REG = 0


def _args(**over):
    cfg = dict(sweep_pass=True, skeleton_pass=False, stamp_pass=False)
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
    """Per-frame freq write stream (every frame writes freq, the way a sweep engine pokes it)."""

    def __init__(self):
        self.rows = []

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def ramp(self, start, delta, n, reg=_FREQ_REG):
        for k in range(n):
            self._frame()
            self.rows.append(_row(reg, (start + k * delta) & 0xFFFF))
        return self

    def hold(self, reg, val, n=2):
        for _ in range(n):
            self._frame()
        self.rows.append(_row(reg, val))
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


def _roundtrip_exact(raw, args):
    enc = SweepPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "sweep replay must be byte-exact"
    return enc


def test_descending_sweep_drains_and_roundtrips():
    raw = _Builder().ramp(40000, -624, 15).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == SWEEP_OP).sum()) == 5, "one 5-atom SWEEP descriptor"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0


def test_ascending_sweep_roundtrips():
    raw = _Builder().ramp(1000, 37, 8).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == SWEEP_OP).sum()) == 5


def test_short_run_left_alone():
    raw = _Builder().ramp(1000, 50, 3).df()
    enc = SweepPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == SWEEP_OP).sum()) == 0, "below SWEEP_MIN_LEN -> not a sweep"


def test_non_constant_delta_left_alone():
    raw = pd.DataFrame(
        [_row(FRAME_REG, 0), _row(_FREQ_REG, 1000)]
        + [
            r
            for k in (1100, 1250, 1450, 1700)
            for r in (_row(FRAME_REG, 0), _row(_FREQ_REG, k))
        ]
        + [_row(FRAME_REG, 0)]
    )
    enc = SweepPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == SWEEP_OP).sum()) == 0
    ), "accelerating (non-constant) delta is not a sweep"


def test_sweep_off_is_noop():
    raw = _Builder().ramp(40000, -624, 15).df()
    out = SweepPass().apply(raw.copy(), args=_args(sweep_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
