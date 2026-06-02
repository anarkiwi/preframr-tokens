"""Parse-level guard for the PW / filter SWEEP targets: a per-frame PW ramp and a global
filter-cutoff ramp through the FULL ``RegLogParser.parse`` (``parse_audit='raise'``) drain to byte-exact
SWEEP atoms under ``pw_sweep`` / ``filter_sweep`` (default/OFF emits none); a gate-on retrigger mid-ramp
must NOT segment the run (PW/filter persist across notes). Byte-exactness is checked by expanding the
DELAYs to per-frame markers and asserting the decoded ramp matches the input (trailing note keeps it).
"""

import os
import tempfile

import numpy as np
import pandas as pd

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
    FC_PRESET_OP,
    FRAME_REG,
    PWM_PRESET_OP,
    SWEEP_OP,
)

_HELD = LUT[48]
_PW_REG = 2
_START, _DELTA, _N = 0x1000, 0x100, 10
_PW_START, _PW_DELTA = 0x200, 0x40


def _args(**over):
    cfg = dict(
        skeleton_pass=True,
        trajectory_anchor_pass=True,
        stamp_pass=True,
        sweep_pass=True,
        patch_pass=True,
        held_arp=True,
        wavetable_pass=True,
        parse_audit="raise",
    )
    cfg.update(over)
    return parse_args(**cfg)


def _parse(path, args):
    return next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def _op_count(df, op):
    return int((df["op"].to_numpy() == op).sum())


def _expand_delays(df):
    """Re-expand each DELAY into ``val`` FRAME markers so ``register_state`` snaps one row per real
    frame (the SWEEP drain is one value per frame tick) -- the granularity needed to compare a
    consolidated sweep stream against the per-frame input ramp."""
    rows = []
    for r in df.to_dict("records"):
        if int(r["reg"]) == DELAY_REG:
            for _ in range(int(r["val"])):
                rows.append({**r, "reg": int(FRAME_REG), "val": 0})
        else:
            rows.append(r)
    return pd.DataFrame(rows)


def _per_frame(df, reg):
    return register_state(_expand_delays(df))[:, reg]


def _contains_subseq(seq, sub):
    seq = list(int(x) for x in seq)
    sub = list(int(x) for x in sub)
    return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


def _build_pw_dump(path):
    """A held note whose per-frame PW climbs by a constant delta, with a gate-on retrigger mid-ramp
    (the PW sweep must stay one run across the note boundary -- note_aligned=False), then a trailing
    held note so the ramp's frames survive the trailing-empty-frame strip."""
    b = DumpBuilder().adsr()
    for k in range(_N):
        b.frame()
        if k in (0, 5):
            b.gate_on()
        b.freq(_HELD)
        b.pw(_PW_START + k * _PW_DELTA)
    b.note([LUT[55]] * 4)
    return write_dump(b, path)


def _build_fc_dump(path):
    """A held note whose per-frame global filter cutoff climbs by a constant delta (one filter,
    global), with a gate-on retrigger mid-ramp, then a trailing held note."""
    b = DumpBuilder().adsr().pw(0x800)
    for k in range(_N):
        b.frame()
        if k in (0, 5):
            b.gate_on()
        b.freq(_HELD)
        b.fc(_START + k * _DELTA)
    b.note([LUT[55]] * 4)
    return write_dump(b, path)


def test_pw_sweep_drains_through_real_parse():
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_pw_dump(os.path.join(tmp, "pw_sweep.dump.parquet"))
        off = _parse(path, _args(pw_sweep=False))
        on = _parse(path, _args(pw_sweep=True))

    assert off is not None and on is not None
    assert _op_count(off, SWEEP_OP) == 0, "no SWEEP without pw_sweep"
    assert _op_count(on, SWEEP_OP) >= 1, "PW ramp did not drain to SWEEP"
    assert _op_count(on, PWM_PRESET_OP) < _op_count(off, PWM_PRESET_OP)
    ramp = [_PW_START + k * _PW_DELTA for k in range(_N)]
    assert _contains_subseq(_per_frame(on, _PW_REG), ramp), _per_frame(on, _PW_REG)


def test_filter_sweep_drains_through_real_parse():
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_fc_dump(os.path.join(tmp, "fc_sweep.dump.parquet"))
        off = _parse(path, _args(filter_sweep=False))
        on = _parse(path, _args(filter_sweep=True))

    assert off is not None and on is not None
    assert _op_count(off, SWEEP_OP) == 0, "no SWEEP without filter_sweep"
    assert _op_count(on, SWEEP_OP) >= 1, "filter cutoff ramp did not drain to SWEEP"
    assert _op_count(on, FC_PRESET_OP) < _op_count(off, FC_PRESET_OP)
    ramp = [_START + k * _DELTA for k in range(_N)]
    assert _contains_subseq(_per_frame(on, FC_LO_REG), ramp), _per_frame(on, FC_LO_REG)


def test_pw_filter_sweep_default_matches_explicit_off():
    """The default args namespace (no pw_sweep/filter_sweep attr) parses identically to explicit OFF,
    proving the new sub-flags cannot perturb the default golden stream."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_pw_dump(os.path.join(tmp, "pw_default.dump.parquet"))
        explicit_off = _parse(path, _args(pw_sweep=False, filter_sweep=False))
        default_args = _args(pw_sweep=False, filter_sweep=False)
        delattr(default_args, "pw_sweep")
        delattr(default_args, "filter_sweep")
        default = _parse(path, default_args)
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
