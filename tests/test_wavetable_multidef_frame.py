"""Regression: two codebook DEFs in one frame must stay byte-exact. The within-frame row sort
(`_norm_pr_order`) grouped all DEFs then STEPs then ENDs, shattering two same-frame `DEF..STEP*..END`
blocks so the decoder replayed garbage freqs (~2.5% of corpus tunes diverged); collapsing each family's
STEP/END to its DEF op keeps blocks contiguous. Two voices share a recurring wide-jump RESID program so
two WAVETABLE_DEFs land in one frame; asserts register_state OFF==ON through the full parse.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import FRAME_REG, WAVETABLE_DEF_OP
from tests.parse_probes import parse_args

_PHI = 985248
_FRAME = _PHI // 50
_PROG = [26, 5, 9, 14]
_BASES = [40, 52, 47, 60]


def _build_two_voice_dump(path):
    rows = []
    clock = [0]

    def w(reg, val):
        rows.append((clock[0], clock[0], 0, int(reg), int(val) & 0xFF))

    def frame():
        clock[0] += _FRAME

    for v in (0, 1):
        b = v * 7
        w(b + 5, 0x00)
        w(b + 6, 0xF0)
        w(b + 2, 0x00)
        w(b + 3, 0x08)
    for base in _BASES:
        for v in (0, 1):
            b = v * 7
            fn = LUT[base]
            if v == 0:
                frame()
            w(b + 4, 0x40)
            w(b + 4, 0x41)
            w(b + 0, fn & 0xFF)
            w(b + 1, (fn >> 8) & 0xFF)
        for off in _PROG:
            frame()
            for v in (0, 1):
                b = v * 7
                fn = LUT[base + off]
                w(b + 0, fn & 0xFF)
                w(b + 1, (fn >> 8) & 0xFF)
        for v in (0, 1):
            b = v * 7
            fn = LUT[base]
            frame()
            w(b + 0, fn & 0xFF)
            w(b + 1, (fn >> 8) & 0xFF)
            frame()
            w(b + 0, fn & 0xFF)
            w(b + 1, (fn >> 8) & 0xFF)
    frame()
    pd.DataFrame(rows, columns=["clock", "irq", "chipno", "reg", "val"]).to_parquet(
        path
    )
    return path


def _parse(path, wavetable_pass):
    a = parse_args(
        skeleton_pass=True,
        trajectory_anchor_pass=True,
        held_arp=True,
        wavetable_pass=wavetable_pass,
    )
    return next(
        RegLogParser(args=a).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def _max_defs_per_frame(df):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    frame = (df["reg"].to_numpy() == FRAME_REG).cumsum()
    headers = frame[(op == WAVETABLE_DEF_OP) & (sub == -1)]
    if len(headers) == 0:
        return 0
    _, counts = np.unique(headers, return_counts=True)
    return int(counts.max())


def test_two_wavetable_defs_in_one_frame_stay_byte_exact():
    """Two voices sharing a recurring RESID program put two WAVETABLE_DEFs in one frame; the codebook
    must stay byte-exact (the within-frame op-sort previously shattered the blocks)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_two_voice_dump(os.path.join(tmp, "wt_multidef.dump.parquet"))
        off = _parse(path, wavetable_pass=False)
        on = _parse(path, wavetable_pass=True)

    assert off is not None and on is not None
    assert _max_defs_per_frame(on) >= 2, "fixture did not put two DEFs in one frame"
    assert np.array_equal(register_state(off), register_state(on))
