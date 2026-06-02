"""PatchPass unit + round-trip tests on synthetic dfs (no HVSC). A full (AD,SR) envelope load
recurring >= PATCH_MINREP times drains to an inline PATCH_DEF + per-reuse PATCH_SET; replay is
byte-exact (the exact AD/SR writes); a single-use load and a partial (AD-only / SR-only) write are
left alone; patch_pass OFF is a no-op."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.patch_pass import PatchPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_STEP_OP,
    SET_OP,
)

_IRQ = 19656
_FREQ_REG = 0
_AD_REG = 5
_SR_REG = 6


def _args(**over):
    cfg = dict(patch_pass=True, skeleton_pass=False, stamp_pass=False)
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
    """Per-frame register write stream (squeezed: only-on-change), the way the parser feeds the
    pass. ``load`` writes a full (AD,SR) envelope on a voice at one frame (an instrument load).
    """

    def __init__(self):
        self.rows = []

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def load(self, ad, sr, freq_reg=_FREQ_REG):
        self._frame()
        self.rows.append(_row(freq_reg + _AD_REG, ad))
        self.rows.append(_row(freq_reg + _SR_REG, sr))
        return self

    def write(self, reg, val):
        self._frame()
        self.rows.append(_row(reg, val))
        return self

    def gap(self, n=2):
        for _ in range(n):
            self._frame()
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


def _roundtrip_exact(raw, args):
    enc = PatchPass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "patch replay must be byte-exact"
    return enc


def test_recurring_patch_drains_to_def_and_roundtrips():
    raw = _Builder().load(0x09, 0x00).gap().load(0x09, 0x00).gap().load(0x09, 0x00).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == PATCH_DEF_OP).sum()) == 1, "one inline def for the patch"
    assert int((enc["op"] == PATCH_STEP_OP).sum()) == 2, "AD + SR steps in the def"
    assert int((enc["op"] == PATCH_SET_OP).sum()) == 2, "one set per reuse"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"].isin([_AD_REG, _SR_REG]))).sum()) == 0
    )


def test_two_distinct_patches_get_two_defs():
    raw = (
        _Builder()
        .load(0x09, 0x00)
        .gap()
        .load(0xA5, 0xF0)
        .gap()
        .load(0x09, 0x00)
        .gap()
        .load(0xA5, 0xF0)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == PATCH_DEF_OP).sum()) == 2
    assert int((enc["op"] == PATCH_SET_OP).sum()) == 2


def test_patch_codebook_is_per_voice():
    """Per-voice instrument codebook: the same envelope used once in each of three voices is NOT shared
    into one def. Cross-voice sharing was dropped because it let a reuse in one voice sort ahead of its
    def in another under the voice-major _norm_pr_order; single-use-per-voice stays literal, byte-exact.
    """
    raw = (
        _Builder()
        .load(0x09, 0x00, freq_reg=0)
        .gap()
        .load(0x09, 0x00, freq_reg=7)
        .gap()
        .load(0x09, 0x00, freq_reg=14)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == PATCH_DEF_OP).sum()) == 0
    assert int((enc["op"] == PATCH_SET_OP).sum()) == 0


def test_single_use_load_left_alone():
    raw = _Builder().load(0x09, 0x00).gap().load(0xA5, 0xF0).df()
    enc = PatchPass().apply(raw.copy(), args=_args())
    assert int((enc["op"] == PATCH_DEF_OP).sum()) == 0
    assert int((enc["op"] == PATCH_SET_OP).sum()) == 0


def test_partial_write_not_claimed():
    raw = (
        _Builder()
        .write(_AD_REG, 0x09)
        .gap()
        .write(_AD_REG, 0x09)
        .gap()
        .write(_AD_REG, 0x09)
        .df()
    )
    enc = PatchPass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == PATCH_DEF_OP).sum()) == 0
    ), "AD-only is not a full envelope load"
    assert int((enc["op"] == PATCH_SET_OP).sum()) == 0


def test_patch_off_is_noop():
    raw = _Builder().load(0x09, 0x00).gap().load(0x09, 0x00).gap().load(0x09, 0x00).df()
    out = PatchPass().apply(raw.copy(), args=_args(patch_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )


def test_redefine_rebinds_id():
    raw = (
        _Builder()
        .load(0x09, 0x00)
        .gap()
        .load(0x09, 0x00)
        .gap()
        .load(0xA5, 0xF0)
        .gap()
        .load(0xA5, 0xF0)
        .gap()
        .load(0x09, 0x00)
        .df()
    )
    _roundtrip_exact(raw, _args())
