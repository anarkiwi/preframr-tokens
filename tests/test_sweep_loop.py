"""Parse-level guard for W5.3 (SWEEP loop-period): a looping freq-domain arp (constant -delta/frame
reset every P, SoundMonitor) that the linear run-finder shatters at each reset drains to looping SWEEP
atoms carrying a period, byte-exact through the FULL ``RegLogParser.parse``. ``sweep_loop`` OFF still
drains it (as per-period linear SWEEPs), so both configs decode identically. Each atom spans at most
SWEEP_MAX_SPAN frames so the inter-atom DELAY stays in ``_cap_delay``'s exact range.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.macros.sweep_pass import SweepPass
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    FRAME_REG,
    SET_OP,
    SWEEP_OP,
    SWEEP_SUBREG_PERIOD,
)

_IRQ = 19656
_FREG = 0
_PATTERN = [40000 + k * (-8500) for _cyc in range(4) for k in range(5)]


def _args(sweep_loop):
    return parse_args(skeleton_pass=True, sweep_pass=True, sweep_loop=sweep_loop)


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[72]] * 4)
    b.note(_PATTERN)
    b.note([LUT[36]] * 4)
    return write_dump(b, path)


def _parse(path, sweep_loop):
    return next(
        RegLogParser(args=_args(sweep_loop)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _sweep_subreg_count(df, subreg):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    return int(((op == SWEEP_OP) & (sub == subreg)).sum())


def test_loop_sweep_drains_byte_exact_through_parse():
    """Through ``RegLogParser.parse`` the looping sweep drains to period-carrying SWEEP atoms; OFF
    decodes the same stream as ON (both byte-exact), and ON emits a PERIOD subreg the linear path
    never does."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "sweep_loop.dump.parquet"))
        off = _parse(path, sweep_loop=False)
        on = _parse(path, sweep_loop=True)

    assert off is not None and on is not None
    assert _sweep_subreg_count(off, SWEEP_SUBREG_PERIOD) == 0
    assert _sweep_subreg_count(on, SWEEP_SUBREG_PERIOD) >= 1
    assert _sweep_subreg_count(on, 0) < _sweep_subreg_count(off, 0)
    ro = register_state(off)
    rn = register_state(on)
    assert ro.shape == rn.shape
    assert np.array_equal(ro, rn)


def _row(reg, val):
    return {
        "reg": reg,
        "val": val,
        "diff": _IRQ,
        "op": int(SET_OP),
        "subreg": -1,
        "irq": _IRQ,
        "description": 0,
    }


def test_long_run_chunks_survive_cap_delay():
    """A single long constant-delta run (> SWEEP_MAX_SPAN) splits into re-anchored chunk atoms so the
    inter-chunk DELAY never exceeds ``_cap_delay``'s exact range; replay stays byte-exact (regression
    for the latent base-SWEEP coarsening bug a single long atom hit)."""
    rows = []
    for k in range(22):
        rows.append(_row(FRAME_REG, 0))
        rows.append(_row(_FREG, (40000 - 700 * k) & 0xFFFF))
    rows.append(_row(FRAME_REG, 0))
    raw = pd.DataFrame(rows)
    args = parse_args(sweep_pass=True, skeleton_pass=False, stamp_pass=False)
    enc = SweepPass().apply(raw.copy(), args=args)
    ri = register_state(raw)
    ro = register_state(enc)
    assert _sweep_subreg_count(enc, 0) >= 2
    assert ri.shape == ro.shape
    assert np.array_equal(ri, ro)


def test_sweep_loop_default_matches_explicit_off():
    """The default args namespace (no ``sweep_loop`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "sweep_loop_default.dump.parquet"))
        explicit_off = _parse(path, sweep_loop=False)
        default_args = _args(sweep_loop=False)
        delattr(default_args, "sweep_loop")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
