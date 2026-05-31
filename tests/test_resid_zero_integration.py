"""Capstone integration guard for the RESID->0 stack (W1-W4 + the W3 backstop): with every gate on
(zero_plain/wt_short/wt_oneshot/slide_wide over wavetable_pass + held_arp) a tune carrying one of each
surviving residue class (ZERO, recurring SHORT, recurring noise-onset, wide SLIDE ramp, unique FLAT,
no-pitched-core noise) drains to zero ``ORN_TYPE_RESID`` in the deployed stream through the full
``RegLogParser.parse``, byte-exact vs the same config gates-OFF. Stamp/Sweep OFF so each class survives.
"""

import os
import tempfile

import numpy as np

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_RESID

_NOISE = 0x81
_GATES = dict(zero_plain=True, wt_short=True, wt_oneshot=True, slide_wide=True)


def _args(gates_on):
    cfg = dict(skeleton_pass=True, held_arp=True, wavetable_pass=True)
    if gates_on:
        cfg.update(_GATES)
    return parse_args(**cfg)


def _noise_onset(b, base):
    b.frame().ctrl(0x40).ctrl(0x41).freq(LUT[base])
    b.frame().ctrl(_NOISE).freq(LUT[base + 33])
    b.frame().ctrl(0x41).freq(LUT[base])


def _noise_core(b, base):
    b.frame().ctrl(0x40).ctrl(_NOISE).freq(LUT[base])
    for off in (20, 30, 18):
        b.frame().ctrl(_NOISE).freq(LUT[base + off])


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    b.note([LUT[55], 0, LUT[55], LUT[55]])
    for base in (40, 52, 47):
        b.note([LUT[base], LUT[base + 31]])
        b.note([LUT[20]] * 4)
    for base in (65, 70, 67):
        _noise_onset(b, base)
        b.note([LUT[22]] * 4)
    b.note([LUT[36]] + [LUT[36 + off] for off in range(1, 41)])
    b.note([LUT[24]] * 4)
    b.note([LUT[58]] + [LUT[58 + off] for off in (26, 5, 9, 14)])
    b.note([LUT[27]] * 4)
    _noise_core(b, 50)
    b.note([LUT[33]] * 5)
    return write_dump(b, path)


def _parse(path, gates_on):
    return next(
        RegLogParser(args=_args(gates_on)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _resid_count(df):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    return int(
        ((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == ORN_TYPE_RESID)).sum()
    )


def test_all_gates_drain_resid_to_zero_byte_exact():
    """Every residue class drains: deployed RESID is positive with the gates off and exactly zero with
    them on, and the decoded register state is identical (the isolation oracle)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "resid_zero.dump.parquet"))
        off = _parse(path, gates_on=False)
        on = _parse(path, gates_on=True)

    assert off is not None and on is not None
    assert _resid_count(off) > 0
    assert _resid_count(on) == 0
    assert np.array_equal(register_state(off), register_state(on))
