"""GlobalOscPass unit + round-trip tests on synthetic dfs (no HVSC). A period-P per-frame cycle on a
global filter / mode-vol reg (21/22/23/24) drains to GLOBAL_OSC atoms whose per-frame replay is
register-exact; a sub-2-period run and a period-1 held value are left alone; osc OFF is a no-op.
Real-parse losslessness is exercised by the test_sid_frame_diff frame-exact matrix."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.global_osc_pass import GlobalOscPass
from preframr_tokens.stfconstants import (
    FILTER_REG,
    FRAME_REG,
    GLOBAL_OSC_OP,
    MODE_VOL_REG,
    SET_OP,
)

_IRQ = 19656


def _args(**over):
    cfg = dict(global_osc=True)
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


def _cycle_df(reg, cycle, frames):
    """Write ``cycle[k % len(cycle)]`` to ``reg`` on each of ``frames`` consecutive song frames."""
    rows = []
    for k in range(frames):
        rows.append(_row(FRAME_REG, 0))
        rows.append(_row(reg, cycle[k % len(cycle)]))
    rows.append(_row(FRAME_REG, 0))
    return pd.DataFrame(rows)


def _roundtrip_exact(raw, args):
    enc = GlobalOscPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "global-osc replay must be register-exact"
    return enc


def test_p2_modevol_wobble_drains_and_roundtrips():
    raw = _cycle_df(MODE_VOL_REG, [15, 0], 8)
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == GLOBAL_OSC_OP).sum()) > 0, "one GLOBAL_OSC descriptor"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == MODE_VOL_REG)).sum()) == 0


def test_p3_resfilt_cycle_drains_and_roundtrips():
    raw = _cycle_df(FILTER_REG, [0xF1, 0xF2, 0xF4], 12)
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == GLOBAL_OSC_OP).sum()) > 0


def test_long_cycle_tiles_and_roundtrips():
    """A run longer than GLOBAL_OSC_MAX_SPAN tiles into several atoms; the replay stays exact."""
    _roundtrip_exact(_cycle_df(MODE_VOL_REG, [10, 5], 60), _args())


def test_short_run_left_alone():
    raw = _cycle_df(MODE_VOL_REG, [15, 0], 3)
    enc = GlobalOscPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == GLOBAL_OSC_OP).sum()) == 0, "below 2*period -> not mined"


def test_period_one_held_not_mined():
    raw = _cycle_df(MODE_VOL_REG, [15], 8)
    enc = GlobalOscPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == GLOBAL_OSC_OP).sum()) == 0, "held value is not oscillation"


def test_osc_off_is_noop():
    raw = _cycle_df(MODE_VOL_REG, [15, 0], 8)
    out = GlobalOscPass().apply(raw.copy(), args=_args(global_osc=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
