"""Golden regression gate for the codebook-unification refactor: decode a corpus exercising every
inline-codebook family and REF variant with the registry-driven CodebookDecoder machine and assert the
expand_ops output matches a frozen golden captured from the legacy per-family decoders before they were
deleted (Step 6). This permanently pins the machine to byte-identical pre-refactor decode output.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from preframr_tokens.macros.codebook import CODEBOOK_FAMILIES
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.instrument_program_pass import InstrumentProgramPass
from preframr_tokens.macros.generator_pass import GeneratorPass
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    FRAME_REG,
    GESTURE_DEF_OP,
    GESTURE_END_OP,
    GESTURE_KIND_HOLD,
    GESTURE_KIND_PERIOD,
    GESTURE_KIND_POLY,
    GESTURE_REF_OP,
    GESTURE_REF_SUBREG_ANCHOR_HI,
    GESTURE_REF_SUBREG_ANCHOR_LO,
    GESTURE_REF_SUBREG_D1_HI,
    GESTURE_REF_SUBREG_D1_LO,
    GESTURE_REF_SUBREG_D2_HI,
    GESTURE_REF_SUBREG_D2_LO,
    GESTURE_REF_SUBREG_ID,
    GESTURE_REF_SUBREG_LEN_HI,
    GESTURE_REF_SUBREG_LEN_LO,
    GESTURE_STEP_OP,
    GESTURE_SUBREG_CELL_HI,
    GESTURE_SUBREG_CELL_LO,
    GESTURE_SUBREG_DEGREE,
    GESTURE_SUBREG_KIND,
    INSTR_OFF_CTRL,
    INSTR_REF_OP,
    SET_OP,
)

_IRQ = 19656
_CODEBOOK_OPS = sorted({op for fam in CODEBOOK_FAMILIES.values() for op in fam.ops})
_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "codebook_machine_golden.json"
_GOLDEN = json.loads(_GOLDEN_PATH.read_text())
_GOLDEN_COLS = ("reg", "val", "diff", "description")


def _cols(df):
    """expand_ops output as plain ``{col: [int]}`` -- the golden's serialisable, dtype-agnostic form."""
    return {c: [int(x) for x in df[c].tolist()] for c in _GOLDEN_COLS}


def _decode(df, seed=None):
    return expand_ops(df.copy(), codebook_seed=seed)


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


def _args(**over):
    from types import SimpleNamespace

    return SimpleNamespace(**over)


class _Builder:
    """Minimal per-frame register-write stream builder (only-on-change), shared by the corpus."""

    def __init__(self):
        self.rows = []

    def frame(self):
        self.rows.append(_row(FRAME_REG, 0))
        return self

    def write(self, reg, val, op=SET_OP, subreg=-1):
        self.rows.append(_row(reg, val, op, subreg))
        return self

    def df(self):
        return pd.DataFrame(self.rows)


def _instrument_streams():
    c0 = int(CTRL_REGS_BY_VOICE[0])

    def build(notes):
        b = _Builder()
        for walk, ad, sr in notes:
            b.frame().write(c0 + 1, ad).write(c0 + 2, sr).write(c0, 0x41)
            for c in walk:
                b.frame().write(c0, c)
        b.frame()
        return b.df()

    note = ([0x11, 0x41], 0x09, 0x00)
    out = {}
    out["instrument_def_ref"] = InstrumentProgramPass().apply(
        build([note, note, note]), args=_args(instrument_program=True)
    )
    return out


def _generator_streams():
    """A freq arp (period-3 TABLE) reused across two transposed notes -- exercises the GEN_TABLE
    codebook DEF/STEP/END/REF + the GEN_TUNING head atom."""

    def build(arps):
        b = _Builder()
        for arp in arps:
            for i in range(9):
                b.frame().write(0, int(arp[i % 3]))
        for _ in range(4):
            b.frame()
        return b.df()

    arp_a = [int(LUT[60]), int(LUT[64]), int(LUT[67])]
    arp_b = [int(LUT[62]), int(LUT[66]), int(LUT[69])]
    out = {}
    out["generator_table"] = GeneratorPass().apply(
        build([arp_a, arp_b]), args=_args(generator_pass=True)
    )
    return out


def _gesture_def(b, idv, kind, degree, cells):
    """One gesture DEF/STEP/END defining a reusable shape into the dictionary (reg 0, voice-relative)."""
    b.write(0, idv, op=GESTURE_DEF_OP)
    b.write(0, kind, op=GESTURE_STEP_OP, subreg=GESTURE_SUBREG_KIND)
    b.write(0, degree, op=GESTURE_STEP_OP, subreg=GESTURE_SUBREG_DEGREE)
    for c in cells:
        b.write(0, c & 0xFF, op=GESTURE_STEP_OP, subreg=GESTURE_SUBREG_CELL_LO)
        b.write(0, (c >> 8) & 0xFF, op=GESTURE_STEP_OP, subreg=GESTURE_SUBREG_CELL_HI)
    b.write(0, idv, op=GESTURE_END_OP)


def _gesture_ref(b, reg, idv, anchor, d1, d2, length):
    """One fixed-layout gesture REF replaying shape ``idv`` with per-instance anchor + diffs + length."""
    a, d1u, d2u, ln = anchor & 0xFFFF, d1 & 0xFFFF, d2 & 0xFFFF, length & 0xFFFF
    b.write(reg, idv, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_ID)
    b.write(reg, a & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_ANCHOR_LO)
    b.write(
        reg, (a >> 8) & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_ANCHOR_HI
    )
    b.write(reg, d1u & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_D1_LO)
    b.write(reg, (d1u >> 8) & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_D1_HI)
    b.write(reg, d2u & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_D2_LO)
    b.write(reg, (d2u >> 8) & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_D2_HI)
    b.write(reg, (ln >> 8) & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_LEN_HI)
    b.write(reg, ln & 0xFF, op=GESTURE_REF_OP, subreg=GESTURE_REF_SUBREG_LEN_LO)


def _gesture_streams():
    """One stream exercising every gesture op and all three shape kinds: a HOLD, a POLY(1) ramp, and a
    PERIOD(2) cell, each defined once then replayed on a scalar reg. Pins the unified _GestureCodec
    decode (DEF/STEP/END/REF -> per-frame writes) byte-exactly."""
    b = _Builder()
    b.frame()
    _gesture_def(b, 0, GESTURE_KIND_HOLD, 0, ())
    _gesture_def(b, 1, GESTURE_KIND_POLY, 1, (5,))
    _gesture_def(b, 2, GESTURE_KIND_PERIOD, 2, (3, -3))
    _gesture_ref(b, 23, 0, 50, 0, 0, 3)
    for _ in range(3):
        b.frame()
    _gesture_ref(b, 2, 1, 100, 0, 0, 4)
    for _ in range(4):
        b.frame()
    _gesture_ref(b, 9, 2, 200, 0, 0, 5)
    for _ in range(5):
        b.frame()
    b.frame()
    return {"gesture_shapes": b.df()}


def _dead_ref_stream():
    """An INSTR_REF to an id that was never defined: every decoder silently drops it (DEAD_REF_POLICY)."""
    c0 = int(CTRL_REGS_BY_VOICE[0])
    return _Builder().frame().write(c0, 99, op=INSTR_REF_OP).frame().df()


def _corpus():
    streams = {}
    streams.update(_instrument_streams())
    streams.update(_generator_streams())
    streams.update(_gesture_streams())
    streams["dead_ref"] = _dead_ref_stream()
    return streams


_CORPUS = _corpus()


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_machine_matches_golden(name):
    assert _cols(_decode(_CORPUS[name])) == _GOLDEN[name]


def test_corpus_covers_every_codebook_op():
    """The corpus must actually exercise every codebook op, else the golden proves nothing for it."""
    seen = set()
    for df in _CORPUS.values():
        if "op" in df.columns:
            seen.update(int(o) for o in df["op"].dropna().unique())
    missing = set(_CODEBOOK_OPS) - seen
    assert (
        not missing
    ), f"corpus exercises no stream with codebook ops {sorted(missing)}"


def test_golden_covers_corpus():
    """Every corpus stream (plus the seed case) has a golden entry and vice-versa -- no silent drift."""
    assert set(_GOLDEN) == set(_CORPUS) | {"seed_materialized"}


def test_seed_materialized_ref_matches_golden():
    """A REF whose DEF preceded the window resolves from the codebook_seed snapshot, byte-identical to
    the pre-refactor decoder."""
    c0 = int(CTRL_REGS_BY_VOICE[0])
    df = _Builder().frame().write(c0, 7, op=INSTR_REF_OP).frame().df()
    seed = {
        "instrument_table": {7: [[(INSTR_OFF_CTRL, 0x41), (INSTR_OFF_CTRL + 1, 0x09)]]}
    }
    assert _cols(_decode(df, seed=seed)) == _GOLDEN["seed_materialized"]
