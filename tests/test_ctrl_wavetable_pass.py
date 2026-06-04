"""CtrlWavetablePass unit + round-trip tests on synthetic dfs (no HVSC). A ctrl byte recurring
>= CTRL_WT_MINREP times on a voice drains to one CTRL_WT_DEF + per-reuse CTRL_WT_SET that decodes
byte-identically; a once-only ctrl byte and a different voice are left literal; ctrl_wavetable OFF is a
no-op.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.ctrl_wavetable_pass import (
    CtrlWavetablePass,
    CtrlWavetableNibblePass,
)
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
)
from preframr_tokens.stfconstants import (
    CTRL_WT_DEF_OP,
    CTRL_WT_SET_OP,
    CTRL_WT_SUBREG_ID_NIB0,
    CTRL_WT_SUBREG_ID_NIB1,
    FILTER_REG,
    FRAME_REG,
    MODE_VOL_REG,
    SET_OP,
)

_IRQ = 19656
_C0 = CTRL_REGS_BY_VOICE[0]
_C1 = CTRL_REGS_BY_VOICE[1]
_SR0 = SR_REGS_BY_VOICE[0]
_AD0 = AD_REGS_BY_VOICE[0]
_RES = int(FILTER_REG)
_MV = int(MODE_VOL_REG)
_F0 = FREQ_REGS_BY_VOICE[0]


def _args(**over):
    cfg = dict(ctrl_wavetable=True)
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
    enc = CtrlWavetablePass().apply(raw.copy(), args=args)
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "ctrl-wt replay must be byte-exact"
    return enc


def test_recurring_ctrl_byte_drains_to_def_plus_refs():
    raw = (
        _Builder()
        .write(_C0, 0x41)
        .write(_C0, 0x81)
        .write(_C0, 0x41)
        .write(_C0, 0x41)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1
    ), "one DEF for the recurring 0x41"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 2, "two reuses after the def"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _C0) & (enc["val"] == 0x41)).sum())
        == 0
    )


def test_two_recurring_values_two_ids():
    raw = (
        _Builder()
        .write(_C0, 0x41)
        .write(_C0, 0x09)
        .write(_C0, 0x41)
        .write(_C0, 0x09)
        .df()
    )
    enc = _roundtrip_exact(raw, _args())
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 2
    ), "one def per distinct recurring value"


def test_once_only_value_left_literal():
    raw = _Builder().write(_C0, 0x41).write(_C0, 0x81).df()
    enc = CtrlWavetablePass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 0
    ), "below CTRL_WT_MINREP -> literal"


def test_cross_voice_shared_instrument_interned():
    """The same byte used once on each of two voices (below per-voice MINREP) interns to ONE shared
    cross-voice id -- a DEF on the earliest occurrence, a CTRL_WT_SET on the other voice's reg.
    """
    raw = _Builder().write(_C0, 0x41).write(_C1, 0x41).df()
    enc = _roundtrip_exact(raw, _args())
    assert int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1, "one shared cross-voice def"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 1, "ref on the other voice"


def test_cross_voice_same_frame_left_literal():
    """Two voices writing the shared byte in the SAME frame are left literal -- the cross-voice DEF
    cannot be ordered before a same-frame ref under voice-major norm, so the guard skips it.
    """
    rows = [_row(FRAME_REG, 0), _row(_C0, 0x41), _row(_C1, 0x41), _row(FRAME_REG, 0)]
    raw = pd.DataFrame(rows)
    enc = CtrlWavetablePass().apply(raw.copy(), args=_args())
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 0
    ), "same-frame cross-voice not interned"


def test_env_wavetable_mines_adsr_when_enabled():
    raw = (
        _Builder()
        .write(_SR0, 0x00)
        .write(_AD0, 0xFF)
        .write(_SR0, 0x00)
        .write(_AD0, 0xFF)
        .df()
    )
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, env_wavetable=True))
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 2
    ), "one def per recurring AD/SR byte"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 2


def test_env_wavetable_off_leaves_adsr_literal():
    raw = _Builder().write(_SR0, 0x00).write(_SR0, 0x00).df()
    enc = CtrlWavetablePass().apply(raw.copy(), args=_args(env_wavetable=False))
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 0
    ), "ctrl_wavetable does not mine AD/SR"


def test_filter_wavetable_mines_resonance():
    raw = _Builder().write(_RES, 0xF3).write(_RES, 0x10).write(_RES, 0xF3).df()
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, filter_wavetable=True))
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1
    ), "recurring RES_FILT byte interned"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 1


def test_modevol_wavetable_mines_mode_vol():
    raw = (
        _Builder()
        .write(_MV, 0x1F)
        .write(_MV, 0x10)
        .write(_MV, 0x1F)
        .write(_MV, 0x10)
        .df()
    )
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, modevol_wavetable=True))
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 2
    ), "one def per recurring mode/vol byte"


def test_freq_wavetable_mines_recurring_pitch():
    """A recurring 16-bit freq word (post combine_regs) interns to a CTRL_WT codebook entry."""
    raw = _Builder().write(_F0, 5611).write(_F0, 10000).write(_F0, 5611).df()
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, freq_wavetable=True))
    assert int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1, "recurring freq word interned"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 1


def test_onset_instrument_held_envelope_drains():
    """An AD value written ONCE but held across >= MINREP gate-rise onsets is the note's instrument:
    onset_instrument emits a DEF at the setup write + a CTRL_WT_SET at each reusing onset, draining the
    one envelope write byte-exactly (the refs re-emit the held value)."""
    rows = [
        _row(FRAME_REG, 0),
        _row(_AD0, 0xFF),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
        _row(_C0, 0x40),
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
        _row(_C0, 0x40),
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
    ]
    raw = pd.DataFrame(rows)
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, onset_instrument=True))
    assert int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1, "the held envelope is a DEF"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _AD0) & (enc["val"] == 0xFF)).sum())
        == 0
    ), "the one AD setup write is drained"


def test_onset_def_once_only_drains_to_lone_def():
    """A single-reg instrument value written ONCE at/after the first note-on drains to a lone
    CTRL_WT_DEF (no reuse) -- the define names the per-tune instrument value, the STEP re-emits it
    byte-exactly."""
    rows = [
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
        _row(_SR0, 0x30),
        _row(FRAME_REG, 0),
    ]
    raw = pd.DataFrame(rows)
    enc = _roundtrip_exact(raw, _args(ctrl_wavetable=False, onset_def=True))
    assert (
        int(((enc["op"] == CTRL_WT_DEF_OP) & (enc["reg"] == _SR0)).sum()) == 1
    ), "the singleton becomes one define"
    assert (
        int(((enc["op"] == CTRL_WT_SET_OP) & (enc["reg"] == _SR0)).sum()) == 0
    ), "no reuse -> no ref"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _SR0)).sum()) == 0
    ), "raw SET drained"


def test_onset_def_preonset_preamble_left_for_init():
    """A write BEFORE the first gate-rise is the driver init preamble (InitPass's province): onset_def
    does not claim it."""
    rows = [
        _row(FRAME_REG, 0),
        _row(_MV, 0x0F),
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
    ]
    raw = pd.DataFrame(rows)
    enc = CtrlWavetablePass().apply(
        raw.copy(), args=_args(ctrl_wavetable=False, onset_def=True)
    )
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _MV)).sum()) == 1
    ), "pre-onset MV stays literal"


def test_onset_def_same_frame_multiwrite_left_for_hard_restart():
    """Two writes to the same reg in one frame are a hard-restart multiload, not a single onset value:
    onset_def (one-write-per-frame) leaves them for HardRestartPass."""
    rows = [
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
        _row(_SR0, 0x08),
        _row(_SR0, 0x30),
        _row(FRAME_REG, 0),
    ]
    raw = pd.DataFrame(rows)
    enc = CtrlWavetablePass().apply(
        raw.copy(), args=_args(ctrl_wavetable=False, onset_def=True)
    )
    assert (
        int(((enc["op"] == CTRL_WT_DEF_OP) & (enc["reg"] == _SR0)).sum()) == 0
    ), "double-load is not a define"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _SR0)).sum()) == 2
    ), "both SR writes stay literal"


def test_onset_def_off_is_noop():
    rows = [
        _row(FRAME_REG, 0),
        _row(_C0, 0x41),
        _row(FRAME_REG, 0),
        _row(_SR0, 0x30),
        _row(FRAME_REG, 0),
    ]
    raw = pd.DataFrame(rows)
    out = CtrlWavetablePass().apply(
        raw.copy(), args=_args(ctrl_wavetable=False, onset_def=False)
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )


def _nib_args(on=True):
    return SimpleNamespace(nibble_wavetable=on)


def _nib_roundtrip(raw, on=True):
    enc = CtrlWavetableNibblePass().apply(raw.copy(), args=_nib_args(on))
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es), "nibble-codebook replay must be byte-exact"
    return enc


def test_nibble_recurring_interns_to_def_plus_ref():
    """A recurring (reg, subreg-0, val) nibble lane the full-byte CTRL_WT misses interns to one DEF
    (lane on subreg NIB0) + a per-reuse SET, byte-exact."""
    rows = [
        _row(FRAME_REG, 0),
        _row(_SR0, 2, subreg=0),
        _row(FRAME_REG, 0),
        _row(_SR0, 2, subreg=0),
        _row(FRAME_REG, 0),
    ]
    enc = _nib_roundtrip(pd.DataFrame(rows))
    assert (
        int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1
    ), "one DEF for the recurring nibble"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 1, "one reuse"
    defs = enc[enc["op"] == CTRL_WT_DEF_OP]
    assert (
        int(defs.iloc[0]["subreg"]) == CTRL_WT_SUBREG_ID_NIB0
    ), "lane 0 on the DEF subreg"


def test_nibble_singleton_define_on_first():
    """A once-only high-nibble (subreg 1) write is still drained -- a lone define-on-first DEF (lane on
    subreg NIB1), no reuse, byte-exact; the raw nibble SET is consumed."""
    rows = [_row(FRAME_REG, 0), _row(_SR0, 5, subreg=1), _row(FRAME_REG, 0)]
    enc = _nib_roundtrip(pd.DataFrame(rows))
    assert int((enc["op"] == CTRL_WT_DEF_OP).sum()) == 1, "lone define-on-first DEF"
    assert int((enc["op"] == CTRL_WT_SET_OP).sum()) == 0, "no reuse -> no ref"
    defs = enc[enc["op"] == CTRL_WT_DEF_OP]
    assert (
        int(defs.iloc[0]["subreg"]) == CTRL_WT_SUBREG_ID_NIB1
    ), "lane 1 on the DEF subreg"
    assert (
        int(((enc["op"] == SET_OP) & (enc["reg"] == _SR0) & (enc["subreg"] == 1)).sum())
        == 0
    ), "raw nibble SET drained"


def test_nibble_off_is_noop():
    rows = [_row(FRAME_REG, 0), _row(_SR0, 2, subreg=0), _row(FRAME_REG, 0)]
    raw = pd.DataFrame(rows)
    out = CtrlWavetableNibblePass().apply(raw.copy(), args=_nib_args(False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )


def test_both_flags_off_is_noop():
    raw = _Builder().write(_C0, 0x41).write(_C0, 0x41).df()
    out = CtrlWavetablePass().apply(
        raw.copy(), args=_args(ctrl_wavetable=False, env_wavetable=False)
    )
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
