"""InitPass unit + round-trip tests on synthetic dfs (no HVSC). Pre-first-note-on SETs on the
single-byte value regs become INIT_OP byte-identically; writes at/after the note-on and on combined
regs are untouched; init_preamble OFF is a no-op. Real-parse losslessness is exercised by
test_full_pipeline_fidelity."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.init_pass import InitPass
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
)
from preframr_tokens.stfconstants import (
    FRAME_REG,
    INIT_OP,
    MODE_VOL_REG,
    SET_OP,
)

_IRQ = 19656
_CTRL = CTRL_REGS_BY_VOICE[0]
_AD = AD_REGS_BY_VOICE[0]
_SR = SR_REGS_BY_VOICE[0]


def _args(**over):
    cfg = dict(init_preamble=True)
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

    def frame(self, *writes):
        self.rows.append(_row(FRAME_REG, 0))
        for reg, val in writes:
            self.rows.append(_row(reg, val))
        return self

    def df(self):
        return pd.DataFrame(self.rows)


def _roundtrip(raw, args):
    enc = InitPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "init relabel must be register-exact"
    return enc


def _preamble():
    return (
        _Builder()
        .frame((_AD, 0x12), (_SR, 0x34), (MODE_VOL_REG, 0x0F))
        .frame((_CTRL, 0x10))
        .frame((MODE_VOL_REG, 0x0E))
        .frame((_CTRL, 0x11), (_AD, 0x56))
        .frame((_AD, 0x78))
        .frame()
        .df()
    )


def test_preamble_relabelled_and_roundtrips():
    enc = _roundtrip(_preamble(), _args())
    assert int((enc["op"] == INIT_OP).sum()) > 0, "pre-note value SETs become INIT_OP"


def test_no_value_set_before_noteon_remains():
    enc = InitPass().apply(_preamble(), args=_args())
    gate_on_row = enc.index[enc["reg"] == _CTRL][1]
    pre = enc.iloc[:gate_on_row]
    val_sets = pre[
        pre["reg"].isin([_AD, _SR, _CTRL, MODE_VOL_REG]) & (pre["op"] == SET_OP)
    ]
    assert len(val_sets) == 0, "no pre-note value SET should remain"


def test_postnote_writes_untouched():
    enc = InitPass().apply(_preamble(), args=_args())
    post = enc[(enc["reg"] == _AD) & (enc["val"] == 0x78)]
    assert int((post["op"] == SET_OP).sum()) == 1


def test_no_noteon_means_noop_when_intro_too_long():
    b = _Builder()
    for _ in range(80):
        b.frame((MODE_VOL_REG, 0x0F))
    enc = InitPass().apply(b.df(), args=_args())
    assert int((enc["op"] == INIT_OP).sum()) == 0


def test_init_off_is_noop():
    raw = _preamble()
    out = InitPass().apply(raw.copy(), args=_args(init_preamble=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
