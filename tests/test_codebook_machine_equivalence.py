"""Differential oracle gate for the codebook-unification refactor: decode a corpus exercising every
inline-codebook family and REF variant twice -- once with the legacy per-family decoders, once with the
registry-driven CodebookDecoder machine -- and assert byte-identical expand_ops output. The legacy
decoders are byte-exact and heavily tested, so they are the oracle the new machine must reproduce.
"""

import contextlib

import pandas as pd
import pytest

from preframr_tokens.macros import decoders as dec_mod
from preframr_tokens.macros.codebook import CODEBOOK_FAMILIES, codebook_decoders
from preframr_tokens.macros.ctrl_wavetable_pass import CtrlWavetablePass
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.macros.patch_pass import PatchPass
from preframr_tokens.macros.skeleton_pass import LUT, SkeletonPass
from preframr_tokens.macros.stamp_pass import StampPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.macros.wavetable_pass import WavetablePass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    STAMP_REF_OP,
    WAVETABLE_ONESHOT_OP,
    WT_ONESHOT_SUBREG_END,
    WT_ONESHOT_SUBREG_LEN_HI,
    WT_ONESHOT_SUBREG_LEN_LO,
    WT_ONESHOT_SUBREG_OFFSET,
)
from preframr_tokens.macros.skeleton_pass import midi_to_fn

_IRQ = 19656
_CODEBOOK_OPS = sorted({op for fam in CODEBOOK_FAMILIES.values() for op in fam.ops})


def _legacy_codebook_map():
    """The legacy per-family decoder registration, rebuilt explicitly so the oracle stays the legacy
    classes regardless of what decoders.DECODERS currently holds (it flips to the machine in Step 3).
    """
    stamp, patch = dec_mod.StampDecoder(), dec_mod.PatchDecoder()
    wt, ctrl_wt = dec_mod.WavetableDecoder(), dec_mod.CtrlWtDecoder()
    by_name = {"stamp": stamp, "patch": patch, "wavetable": wt, "ctrl_wt": ctrl_wt}
    return {
        op: by_name[fam.name] for fam in CODEBOOK_FAMILIES.values() for op in fam.ops
    }


@contextlib.contextmanager
def _decoders(codebook_map):
    """Swap the codebook ops in the shared DECODERS dict in place (decode/walker hold the same dict)."""
    saved = {op: DECODERS.get(op) for op in _CODEBOOK_OPS}
    try:
        DECODERS.update(codebook_map)
        yield
    finally:
        for op, dec in saved.items():
            if dec is None:
                DECODERS.pop(op, None)
            else:
                DECODERS[op] = dec


def _decode(df, codebook_map, seed=None):
    with _decoders(codebook_map):
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


def _stamp_streams():
    _FREQ, _CTRL = 0, 4
    hat = [(2000, 0x81), (2000, 0x80), (2000, 0x80)]
    kick = [
        (midi_to_fn(60), 0x41),
        (midi_to_fn(48), 0x41),
        (midi_to_fn(40), 0x41),
        (midi_to_fn(36), 0x40),
    ]

    def gesture(base):
        return [(base, 0x41), (base + 100, 0x41), (base + 50, 0x40)]

    def build(hits, gap=3):
        b = _Builder()
        last_fn = last_ctrl = None
        for series in hits:
            for fn, ctrl in series:
                b.frame()
                if ctrl != last_ctrl:
                    b.write(_CTRL, ctrl)
                    last_ctrl = ctrl
                if fn != last_fn:
                    b.write(_FREQ, fn)
                    last_fn = fn
            for _ in range(gap):
                b.frame()
        b.frame()
        return b.df()

    out = {}
    out["stamp_abs"] = StampPass().apply(build([hat] * 4), args=_args(stamp_pass=True))
    out["stamp_two_defs"] = StampPass().apply(
        build([hat, kick, hat, kick, hat, kick]), args=_args(stamp_pass=True)
    )
    out["stamp_rel"] = StampPass().apply(
        build([gesture(b) for b in (1000, 2000, 3000, 4000)]),
        args=_args(stamp_pass=True),
    )
    return out


def _patch_streams():
    _AD, _SR = 5, 6

    def load(loads, gap=2):
        b = _Builder()
        for ad, sr, freq_reg in loads:
            b.frame().write(freq_reg + _AD, ad).write(freq_reg + _SR, sr)
            for _ in range(gap):
                b.frame()
        b.frame()
        return b.df()

    out = {}
    out["patch_def_set"] = PatchPass().apply(
        load([(0x09, 0x00, 0)] * 3), args=_args(patch_pass=True)
    )
    out["patch_two"] = PatchPass().apply(
        load([(0x09, 0x00, 0), (0xA5, 0xF0, 0), (0x09, 0x00, 0), (0xA5, 0xF0, 0)]),
        args=_args(patch_pass=True),
    )
    out["patch_rebind"] = PatchPass().apply(
        load(
            [
                (0x09, 0x00, 0),
                (0x09, 0x00, 0),
                (0xA5, 0xF0, 0),
                (0xA5, 0xF0, 0),
                (0x09, 0x00, 0),
            ]
        ),
        args=_args(patch_pass=True),
    )
    return out


def _wavetable_streams():
    _FREQ, _CTRL = 0, 4

    def build(prog, bases):
        b = _Builder()
        for base in bases:
            b.frame().write(_CTRL, 0x40)
            b.frame().write(_CTRL, 0x41).write(_FREQ, int(LUT[base]))
            for off in prog:
                b.frame().write(_FREQ, int(LUT[base + off]))
        for _ in range(6):
            b.frame()
        return b.df()

    def skel(df):
        return SkeletonPass().apply(df, _args(skeleton_pass=True, held_arp=True))

    out = {}
    out["wt_def_ref"] = WavetablePass().apply(
        skel(build([26, 5, 9, 14], [40, 60, 50])), _args(wavetable_pass=True)
    )
    out["wt_oneshot"] = WavetablePass().apply(
        skel(build([26, 5, 9, 14], [40])), _args(wavetable_pass=True, wt_oneshot=True)
    )
    return out


def _ctrl_wt_streams():
    c0 = int(CTRL_REGS_BY_VOICE[0])
    c1 = int(CTRL_REGS_BY_VOICE[1])

    def build(writes):
        b = _Builder()
        for reg, val in writes:
            b.frame().write(reg, val)
        b.frame()
        return b.df()

    out = {}
    out["ctrl_wt_def_set"] = CtrlWavetablePass().apply(
        build([(c0, 0x41), (c0, 0x81), (c0, 0x41), (c0, 0x41)]),
        args=_args(ctrl_wavetable=True),
    )
    out["ctrl_wt_cross_voice"] = CtrlWavetablePass().apply(
        build([(c0, 0x41), (c1, 0x41)]), args=_args(ctrl_wavetable=True)
    )
    return out


def _dead_ref_stream():
    """A STAMP_REF to an id that was never defined: every decoder silently drops it (DEAD_REF_POLICY)."""
    return _Builder().frame().write(0, 99, op=STAMP_REF_OP).frame().df()


def _oneshot_handbuilt_stream():
    """A self-contained WAVETABLE one-shot after an absolute SKEL note (exercises ONESHOT replay)."""
    b = _Builder()
    b.frame().write(0, 48, op=SKEL_OP, subreg=SKEL_SUBREG_ABS)
    b.frame().write(0, 0, op=WAVETABLE_ONESHOT_OP, subreg=WT_ONESHOT_SUBREG_LEN_HI)
    b.write(0, 3, op=WAVETABLE_ONESHOT_OP, subreg=WT_ONESHOT_SUBREG_LEN_LO)
    for off in (0, 7, 12):
        b.write(0, off, op=WAVETABLE_ONESHOT_OP, subreg=WT_ONESHOT_SUBREG_OFFSET)
    b.write(0, 0, op=WAVETABLE_ONESHOT_OP, subreg=WT_ONESHOT_SUBREG_END)
    for _ in range(6):
        b.frame()
    return b.df()


def _corpus():
    streams = {}
    streams.update(_stamp_streams())
    streams.update(_patch_streams())
    streams.update(_wavetable_streams())
    streams.update(_ctrl_wt_streams())
    streams["dead_ref"] = _dead_ref_stream()
    streams["oneshot_handbuilt"] = _oneshot_handbuilt_stream()
    return streams


_CORPUS = _corpus()
_LEGACY_MAP = _legacy_codebook_map()
_NEW_MAP = codebook_decoders()


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_machine_matches_legacy(name):
    df = _CORPUS[name]
    legacy = _decode(df, _LEGACY_MAP)
    new = _decode(df, _NEW_MAP)
    pd.testing.assert_frame_equal(legacy, new)


def test_corpus_covers_every_codebook_op():
    """The corpus must actually exercise every codebook op, else equivalence proves nothing for it."""
    seen = set()
    for df in _CORPUS.values():
        if "op" in df.columns:
            seen.update(int(o) for o in df["op"].dropna().unique())
    missing = set(_CODEBOOK_OPS) - seen
    assert (
        not missing
    ), f"corpus exercises no stream with codebook ops {sorted(missing)}"


def test_seed_materialized_ref_matches_legacy():
    """A REF whose DEF preceded the window resolves from the codebook_seed snapshot; both machines
    must seed and replay it identically."""
    df = _Builder().frame().write(0, 7, op=STAMP_REF_OP).frame().df()
    seed = {"stamp_table": {7: [[(0, 1234), (4, 0x41)]]}}
    legacy = _decode(df, _LEGACY_MAP, seed=seed)
    new = _decode(df, _NEW_MAP, seed=seed)
    pd.testing.assert_frame_equal(legacy, new)
