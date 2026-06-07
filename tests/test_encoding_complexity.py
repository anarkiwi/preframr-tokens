"""Unit tests for the encoding-explosion guard (synthetic; corpus-fixture tests are separate)."""
import pandas as pd
import pytest

from preframr_tokens.encoding_complexity import (
    EncodingExplosion,
    check_no_explosion,
    encoded_freq_complexity,
    input_freq_complexity,
)
from preframr_tokens.stfconstants import (
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_ID,
    GEN_TABLE_REF_SUBREG_BASE_NOTE,
    GEN_TABLE_REF_SUBREG_RESID_LO,
    GEN_TABLE_REF_SUBREG_RESID_HI,
    GEN_TABLE_REF_SUBREG_LEN_HI,
    GEN_TABLE_REF_SUBREG_LEN_LO,
)


def _reg_row(irq, reg, val):
    return {"clock": irq, "irq": irq, "chipno": 0, "reg": reg, "val": val}


def _two_note_input():
    """Voice 0 holds two on-grid notes (C5=4455-ish, +12). Low structural complexity."""
    rows = []
    for irq, fw in enumerate([4455, 4455, 8910, 8910] * 4, start=1):
        rows.append(_reg_row(irq, 4, 1))  # gate on
        rows.append(_reg_row(irq, 0, fw & 0xFF))
        rows.append(_reg_row(irq, 1, (fw >> 8) & 0xFF))
    return pd.DataFrame(rows)


def _ref_rows(cb, resid_seq):
    """A note-table REF on voice 0 (reg 0) carrying a residual payload."""
    rows = [(0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_ID, cb),
            (0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_BASE_NOTE, 49)]
    for r in resid_seq:
        rows.append((0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_RESID_LO, r & 0xFF))
        rows.append((0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_RESID_HI, (r >> 8) & 0xFF))
    rows.append((0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_LEN_HI, 0))
    rows.append((0, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_LEN_LO, 4))
    return rows


def _tokens(payloads, cbs=None):
    rows = []
    for i, p in enumerate(payloads):
        rows.extend(_ref_rows(i if cbs is None else cbs[i], p))
    return pd.DataFrame(rows, columns=["reg", "op", "subreg", "val"])


def test_input_complexity_counts_notes_not_vibrato_samples():
    inp = input_freq_complexity(_two_note_input())
    assert inp[0] == 2  # two distinct on-grid notes, no off-grid levels


def test_explosion_raises_when_residual_payloads_exceed_input():
    reg = _two_note_input()
    # one shape, but 8 distinct residual payloads -> minted vocabulary >> input alphabet
    tok = _tokens([(10 * i, 20 * i) for i in range(1, 9)])
    enc = encoded_freq_complexity(tok)
    assert enc[0] > input_freq_complexity(reg)[0]
    with pytest.raises(EncodingExplosion):
        check_no_explosion(reg, tok, tune="synthetic")


def test_no_explosion_when_shapes_reuse_one_gesture():
    reg = _two_note_input()
    # one shape (shared cb), residuals all identical -> 2 keys <= 2 input
    tok = _tokens([(5, 5)] * 6, cbs=[0] * 6)
    rep = check_no_explosion(reg, tok, tune="synthetic", raise_on_explosion=False)
    assert not rep["explosions"]
