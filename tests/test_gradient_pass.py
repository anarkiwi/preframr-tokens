"""GradientPass unit + round-trip tests on synthetic dfs (no HVSC). A held-step value curve of
>= GRADIENT_MIN_STAGES writes on a modulation reg (each value held until the next write) drains to
GRADIENT atoms whose per-frame replay is register-exact; a two-step run is left alone; gradient OFF is
a no-op. Real-parse losslessness is exercised by test_full_pipeline_fidelity."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.gradient_pass import GradientPass
from preframr_tokens.macros.state import AD_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    FRAME_REG,
    GRADIENT_OP,
    MODE_VOL_REG,
    SET_OP,
)

_IRQ = 19656


def _args(**over):
    cfg = dict(modevol_gradient=True, env_gradient=True, filter_gradient=True)
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
    """Held-step automation: each step writes a value then holds it for ``hold`` frames before the next
    write (the way a volume fade / filter gradient pokes the reg sparsely)."""

    def __init__(self):
        self.rows = []

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def step(self, val, hold, reg=MODE_VOL_REG):
        self._frame()
        self.rows.append(_row(reg, val))
        for _ in range(hold - 1):
            self._frame()
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


def _roundtrip_exact(raw, args):
    enc = GradientPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "gradient replay must be register-exact"
    return enc


def test_held_step_curve_drains_and_roundtrips():
    raw = _Builder().step(13, 5).step(11, 5).step(10, 5).step(9, 5).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == GRADIENT_OP).sum()) > 0, "one GRADIENT descriptor"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == MODE_VOL_REG)).sum()) == 0


def test_long_curve_tiles_and_roundtrips():
    """A curve longer than GRADIENT_MAX_STAGES tiles into several atoms; the replay stays exact."""
    b = _Builder()
    for k in range(14):
        b.step(15 - k, 4)
    _roundtrip_exact(b.df(), _args())


def test_varied_holds_roundtrip():
    raw = _Builder().step(12, 3).step(8, 20).step(4, 7).step(1, 11).df()
    _roundtrip_exact(raw, _args())


def test_env_reg_curve_roundtrips():
    reg = AD_REGS_BY_VOICE[0]
    raw = (
        _Builder()
        .step(0x12, 6, reg=reg)
        .step(0x34, 6, reg=reg)
        .step(0x56, 6, reg=reg)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == GRADIENT_OP).sum()) > 0


def test_two_step_run_left_alone():
    raw = _Builder().step(13, 5).step(11, 5).df()
    enc = GradientPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == GRADIENT_OP).sum()) == 0
    ), "below GRADIENT_MIN_STAGES -> not mined"


def test_gradient_off_is_noop():
    raw = _Builder().step(13, 5).step(11, 5).step(10, 5).df()
    out = GradientPass().apply(
        raw.copy(),
        args=_args(modevol_gradient=False, env_gradient=False, filter_gradient=False),
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
