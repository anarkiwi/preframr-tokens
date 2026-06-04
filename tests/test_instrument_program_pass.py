"""InstrumentProgramPass tests through the full RegLogParser.parse (the deployed token stream, not a
hand-built df): a voice's per-frame (ctrl, AD, SR) program drains to an inline INSTR_DEF + INSTR_REF,
byte-exact vs the flag OFF, with zero ctrl/AD/SR raw-SET residual; define-on-first emits a DEF per
unique program; and the register_state guard falls back to the unclaimed stream on a forced mis-decode.
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.instrument_program_pass import InstrumentProgramPass
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    FRAME_REG,
    INSTR_DEF_OP,
    INSTR_REF_OP,
    SET_OP,
)

_TIMBRE = (4, 5, 6, 11, 12, 13, 18, 19, 20)
_IRQ = 19656


def _note(b, walk, base, ad, sr):
    """One note: AD/SR load + gate-on hard-restart pair + a per-frame ctrl walk + a short held-freq
    tail (the freq content keeps the frames alive, as a real driver does)."""
    b.frame()
    b.adsr(ad, sr)
    b.ctrl(0x40)
    b.ctrl(0x41)
    b.freq(base)
    for i, c in enumerate(walk):
        b.frame()
        b.ctrl(c)
        b.freq(base + 200 * (i + 1))
    for i in range(3):
        b.frame()
        b.freq(base + 1000 + 80 * i)


def _bank_dump(path):
    """A small instrument bank reused across notes: instrument A (3x) and B (3x) plus a unique C (once)
    -- so the codebook emits shared DEFs for A/B and a define-on-first DEF for C."""
    b = DumpBuilder()
    for _ in range(3):
        _note(b, [0x11, 0x41], 4000, 0x09, 0x00)
        _note(b, [0x21, 0x41], 5000, 0xA5, 0xF0)
    _note(b, [0x81, 0x41], 6000, 0x47, 0x88)
    return write_dump(b, path)


def _distinct_dump(path):
    """Every note a DIFFERENT (ctrl, AD, SR) program -- so each span is unique and drains via
    define-on-first (minrep=1), one DEF apiece."""
    b = DumpBuilder()
    for k in range(6):
        _note(b, [0x11 + k, 0x41], 4000 + 300 * k, 0x09 + k, 0x10 * k)
    return write_dump(b, path)


def _parse(path, on):
    return next(
        RegLogParser(args=parse_args(instrument_program=on)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _resid(df):
    ops = df["op"].to_numpy()
    regs = df["reg"].to_numpy()
    return int(
        sum(
            1
            for k in range(len(df))
            if int(ops[k]) == SET_OP and int(regs[k]) in _TIMBRE
        )
    )


def test_flag_is_byte_exact_and_drains_residual():
    """The flag never changes the rendered register stream (byte-exact vs OFF) yet drains the ctrl/AD/SR
    raw-SET residual to zero (and the residual was non-zero OFF, proving the pass did the work).
    """
    with tempfile.TemporaryDirectory() as t:
        p = _bank_dump(os.path.join(t, "bank.dump.parquet"))
        off = _parse(p, False)
        on = _parse(p, True)
    assert off is not None and on is not None
    assert _resid(off) > 0
    assert _resid(on) == 0
    ro, rn = register_state(off), register_state(on)
    assert ro.shape == rn.shape
    assert np.array_equal(ro, rn)


def test_recurring_instrument_shares_defs():
    """A reused instrument bank interns to fewer DEFs than REFs (each reuse is a single-row REF)."""
    with tempfile.TemporaryDirectory() as t:
        p = _bank_dump(os.path.join(t, "bank.dump.parquet"))
        on = _parse(p, True)
    assert on is not None
    defs = int((on["op"] == INSTR_DEF_OP).sum())
    refs = int((on["op"] == INSTR_REF_OP).sum())
    assert defs >= 1
    assert refs > defs


def test_define_on_first_unique_programs():
    """Every span unique: minrep=1 still emits a lone DEF per program, no ctrl/AD/SR SET survives, and
    the render is byte-exact."""
    with tempfile.TemporaryDirectory() as t:
        p = _distinct_dump(os.path.join(t, "distinct.dump.parquet"))
        off = _parse(p, False)
        on = _parse(p, True)
    assert off is not None and on is not None
    assert int((on["op"] == INSTR_DEF_OP).sum()) >= 1
    assert _resid(on) == 0
    assert np.array_equal(register_state(off), register_state(on))


def _inline_df():
    """A parser-stage row stream (actual voice regs, op column present, gate-on retriggers) the way the
    inline loop feeds the pass -- a recurring (ctrl, AD, SR) program on voice 0."""
    rows = []

    def add(reg, val, diff):
        rows.append(
            {
                "reg": int(reg),
                "val": int(val),
                "diff": int(diff),
                "op": int(SET_OP),
                "subreg": -1,
                "irq": _IRQ,
                "description": 0,
            }
        )

    for _ in range(3):
        add(FRAME_REG, 0, _IRQ)
        add(5, 0x09, 32)
        add(6, 0x00, 32)
        add(4, 0x41, 32)
        add(FRAME_REG, 0, _IRQ)
        add(4, 0x11, 32)
        add(FRAME_REG, 0, _IRQ)
        add(4, 0x41, 32)
    add(FRAME_REG, 0, _IRQ)
    return pd.DataFrame(rows)


def test_lossless_guard_falls_back(monkeypatch):
    """When the replay would diverge, the register_state guard drops the whole claim and returns the
    unclaimed stream unchanged -- so enabling the flag can never change the render."""
    df = _inline_df()
    args = SimpleNamespace(instrument_program=True)
    claimed = InstrumentProgramPass().apply(df.copy(), args=args)
    assert int((claimed["op"] == INSTR_DEF_OP).sum()) >= 1

    monkeypatch.setattr(
        InstrumentProgramPass,
        "_instr_is_lossless",
        staticmethod(lambda before, after: False),
    )
    fell_back = InstrumentProgramPass().apply(df.copy(), args=args)
    assert int((fell_back["op"] == INSTR_DEF_OP).sum()) == 0
    pd.testing.assert_frame_equal(
        fell_back.reset_index(drop=True), df.reset_index(drop=True)
    )
