"""StampPass unit + round-trip tests on synthetic dfs (no HVSC). A recurring exact (freq,ctrl)
write-series >=STAMP_MINREP times drains to an inline STAMP_DEF + per-hit STAMP_REF; replay is
byte-exact (the exact write-series, not the floor); a non-recurring series is left alone;
stamp_pass OFF is a no-op."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import midi_to_fn
from preframr_tokens.macros.stamp_pass import StampPass, classify_char
from preframr_tokens.stfconstants import (
    FRAME_REG,
    SET_OP,
    STAMP_CHAR_HAT,
    STAMP_DEF_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
)

_IRQ = 19656
_FREQ_REG = 0
_CTRL_REG = 4


def _args(**over):
    cfg = dict(stamp_pass=True, skeleton_pass=False, freq_trajectory_pass=False)
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
    """Per-frame freq/ctrl write stream (squeezed: only-on-change), the way the parser feeds
    the pass. A drum hit is a short (freq,ctrl) write-series; ``hit`` stamps one."""

    def __init__(self):
        self.rows = []
        self._fn = None
        self._ctrl = None

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def hit(self, series):
        for fn, ctrl in series:
            self._frame()
            if ctrl != self._ctrl:
                self.rows.append(_row(_CTRL_REG, ctrl))
                self._ctrl = ctrl
            if fn != self._fn:
                self.rows.append(_row(_FREQ_REG, fn))
                self._fn = fn
        return self

    def gap(self, n=3):
        for _ in range(n):
            self._frame()
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


_HAT = [(2000, 0x81), (2000, 0x80), (2000, 0x80)]
_KICK = [
    (midi_to_fn(60), 0x41),
    (midi_to_fn(48), 0x41),
    (midi_to_fn(40), 0x41),
    (midi_to_fn(36), 0x40),
]


def _roundtrip_exact(raw, args):
    enc = StampPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "stamp replay must be byte-exact"
    return enc


def test_recurring_hat_drains_to_stamp_and_roundtrips():
    raw = _Builder().hit(_HAT).gap().hit(_HAT).gap().hit(_HAT).gap().hit(_HAT).df()
    enc = _roundtrip_exact(raw, _args())
    assert (
        int((enc["op"] == STAMP_DEF_OP).sum()) == 1
    ), "one inline def for the recurring hat"
    assert int((enc["op"] == STAMP_REF_OP).sum()) == 4, "one ref per hit"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0


def test_two_distinct_drums_get_two_defs():
    raw = (
        _Builder()
        .hit(_HAT)
        .gap()
        .hit(_KICK)
        .gap()
        .hit(_HAT)
        .gap()
        .hit(_KICK)
        .gap()
        .hit(_HAT)
        .gap()
        .hit(_KICK)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == STAMP_DEF_OP).sum()) == 2
    assert int((enc["op"] == STAMP_REF_OP).sum()) == 6


def test_non_recurring_series_left_alone():
    raw = _Builder().hit(_HAT).gap().hit(_KICK).df()
    enc = StampPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == STAMP_DEF_OP).sum()) == 0
    assert int((enc["op"] == STAMP_REF_OP).sum()) == 0


def test_stamp_off_is_noop():
    raw = _Builder().hit(_HAT).gap().hit(_HAT).gap().hit(_HAT).df()
    out = StampPass().apply(raw.copy(), args=_args(stamp_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )


def test_classify_hat():
    fns = [2000, 2000, 2000]
    ctrls = [0x81, 0x80, 0x80]
    assert classify_char(fns, ctrls) == STAMP_CHAR_HAT


def _gesture(base):
    """A pitched effect: gate-on sweep up then gate-off, identical freq-DELTA + ctrl shape at any
    base. Each base is unique (so ABS never reaches MINREP), but the transpose-shape recurs.
    """
    return [(base, 0x41), (base + 100, 0x41), (base + 50, 0x40)]


def test_transposed_gesture_drains_to_rel_stamp_and_roundtrips():
    raw = (
        _Builder()
        .hit(_gesture(1000))
        .gap()
        .hit(_gesture(2000))
        .gap()
        .hit(_gesture(3000))
        .gap()
        .hit(_gesture(4000))
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == STAMP_DEF_OP).sum()) == 1, "one REL def for the shape"
    assert (
        int((enc["op"] == STAMP_REF_OP).sum()) == 0
    ), "no exact-ABS hits (each base unique)"
    assert (
        int((enc["op"] == STAMP_REL_REF_OP).sum()) == 12
    ), "3 atoms (id + base hi/lo) per hit x 4 hits"
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0


def test_two_transposed_gestures_left_alone():
    raw = _Builder().hit(_gesture(1000)).gap().hit(_gesture(2000)).df()
    enc = StampPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == STAMP_DEF_OP).sum()) == 0, "below MINREP -> not stamped"
    assert int((enc["op"] == STAMP_REL_REF_OP).sum()) == 0


def test_exact_repeat_prefers_abs_over_rel():
    raw = (
        _Builder()
        .hit(_gesture(1000))
        .gap()
        .hit(_gesture(1000))
        .gap()
        .hit(_gesture(1000))
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == STAMP_REF_OP).sum()) == 3, "same base -> exact ABS"
    assert int((enc["op"] == STAMP_REL_REF_OP).sum()) == 0
