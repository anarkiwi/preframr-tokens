"""WavetablePass unit + round-trip tests. A recurring note-relative offset program across skeleton
RESID notes drains to an inline WAVETABLE_DEF + per-note WAVETABLE_REF codebook; replay is byte-identical
to the content-floor RESID it replaces (isolation oracle: register_state OFF == ON); a structured
non-recurring program emits an inline one-shot; wavetable_pass OFF is a no-op."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT, SkeletonPass
from preframr_tokens.macros.wavetable import factorise, program_key, unroll
from preframr_tokens.macros.wavetable_pass import WavetablePass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    SET_OP,
    WAVETABLE_DEF_OP,
    WAVETABLE_REF_OP,
)

_IRQ = 19656
_FREQ_REG = 0
_CTRL_REG = 4


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
    """Per-frame freq/ctrl stream (only-on-change) the parser feeds the skeleton; ``note`` plays a
    gate-on onset at ``base`` then one freq per note-relative offset in ``prog`` (held one frame).
    """

    def __init__(self):
        self.rows = []

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def note(self, base, prog, ctrl_on=0x41, ctrl_off=0x40):
        self._frame()
        self.rows.append(_row(_CTRL_REG, ctrl_off))
        self._frame()
        self.rows.append(_row(_CTRL_REG, ctrl_on))
        self.rows.append(_row(_FREQ_REG, int(LUT[base])))
        for off in prog:
            self._frame()
            self.rows.append(_row(_FREQ_REG, int(LUT[base + off])))
        return self

    def end(self, n=6):
        for _ in range(n):
            self._frame()
        return self

    def df(self):
        return pd.DataFrame(self.rows)


def _skel(df, held_arp=True):
    args = SimpleNamespace(skeleton_pass=True, held_arp=held_arp)
    return SkeletonPass().apply(df, args)


def _wt(df, on=True):
    return WavetablePass().apply(df, SimpleNamespace(wavetable_pass=on))


def _resid_count(df):
    return int(
        (
            (df["op"] == ORN_OP)
            & (df["subreg"] == ORN_SUBREG_TYPE)
            & (df["val"] == ORN_TYPE_RESID)
        ).sum()
    )


def _build(prog, bases):
    b = _Builder()
    for base in bases:
        b.note(base, prog)
    return b.end().df()


def test_factorise_roundtrips_and_classifies():
    for core, has_body in (
        ([5, 0, 12, 0, 12, 0, 12], True),
        ([0, 0, 12, 12, 24, 24] * 3, True),
        ([0, 12, 0, 12, 0, 12], True),
        ([0, 5, 7, 3, 9, 1], False),
        ([26, 5, 9, 14], False),
    ):
        steps, loop = factorise(core)
        assert unroll(steps, loop, len(core)) == core, (core, steps, loop)
        assert (loop < len(steps)) == has_body, (core, steps, loop)


def test_program_key_length_invariant():
    short = factorise([5, 0, 12, 0, 12])
    long = factorise([5, 0, 12, 0, 12, 0, 12])
    assert program_key(*short) == program_key(*long)
    assert unroll(*long, 5) == [5, 0, 12, 0, 12]


def test_unroll_lead_byte_exact():
    steps, loop = factorise([0, 7, 0, 7, 0, 7])
    assert unroll(steps, loop, 8, lead=[-3, -1]) == [-3, -1, 0, 7, 0, 7, 0, 7]


def test_recurring_resid_drains_to_codebook():
    prog = [26, 5, 9, 14]
    sk = _skel(_build(prog, [40, 60, 50]))
    assert _resid_count(sk) >= 3, _resid_count(sk)
    on = _wt(sk)
    assert np.array_equal(register_state(sk), register_state(on))
    assert int((on["op"] == WAVETABLE_REF_OP).sum()) >= 3 * 4
    assert int((on["op"] == WAVETABLE_DEF_OP).sum()) == 1
    assert _resid_count(on) < _resid_count(sk)


def test_codebook_bounded():
    prog = [26, 5, 9, 14]
    on = _wt(_skel(_build(prog, [40, 60, 50, 44, 70])))
    assert int((on["op"] == WAVETABLE_DEF_OP).sum()) == 1


def test_inline_oneshot_structured():
    prog = [26, 0, 7, 0, 7, 0, 7]
    sk = _skel(_build(prog, [48]))
    assert _resid_count(sk) >= 1
    on = _wt(sk)
    assert np.array_equal(register_state(sk), register_state(on))
    assert int((on["op"] == WAVETABLE_DEF_OP).sum()) == 1
    assert int((on["op"] == WAVETABLE_REF_OP).sum()) >= 1


def test_pass_off_is_noop():
    sk = _skel(_build([26, 5, 9, 14], [40, 60, 50]))
    off = _wt(sk, on=False)
    assert off.equals(sk)


def test_nonrecurring_flat_oneshot_stays_resid():
    sk = _skel(_build([26, 5, 9, 14], [40]))
    before = _resid_count(sk)
    on = _wt(sk)
    assert _resid_count(on) == before
    assert int((on["op"] == WAVETABLE_DEF_OP).sum()) == 0
