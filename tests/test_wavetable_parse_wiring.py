"""Parse-level regression guard for the WAVETABLE codebook pass: a recurring note-relative
RESID program driven through the FULL ``RegLogParser.parse`` must drain to a WAVETABLE_DEF +
per-note WAVETABLE_REF codebook, byte-exact, while default/OFF stays a no-op. The direct-apply
unit tests stayed green when the pass was registered only in ``FREQ_BLOCK_PASSES`` and omitted
from the parallel hand-listed parse sequence, so the pass was dead for every real parse.
"""

import os
import tempfile

import numpy as np

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    WAVETABLE_DEF_OP,
    WAVETABLE_REF_OP,
)

_PROG = [26, 5, 9, 14]
_BASES = (40, 52, 47, 60)


def _args(wavetable_pass):
    return parse_args(
        skeleton_pass=True,
        trajectory_anchor_pass=True,
        held_arp=True,
        wavetable_pass=wavetable_pass,
    )


def _build_dump(path):
    """A wide-jump non-periodic program the skeleton escapes to RESID, repeated at several bases
    (recurs >= WT_MINREP) between plain notes the parser keeps."""
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    for base in _BASES:
        b.note([LUT[base]] + [LUT[base + off] for off in _PROG])
        b.note([LUT[base]] * 4)
    b.note([LUT[48]] * 5)
    return write_dump(b, path)


def _parse(path, wavetable_pass):
    return next(
        RegLogParser(args=_args(wavetable_pass)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _op_count(df, op):
    return int((df["op"].to_numpy() == op).sum())


def _resid_count(df):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    return int(
        ((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == ORN_TYPE_RESID)).sum()
    )


def test_wavetable_pass_fires_through_real_parse():
    """Through ``RegLogParser.parse`` the recurring RESID drains to a codebook byte-exactly while
    OFF keeps the RESID and emits no codebook ops; regression for the parse-sequence omission.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wavetable_wiring.dump.parquet"))
        off = _parse(path, wavetable_pass=False)
        on = _parse(path, wavetable_pass=True)

    assert off is not None and on is not None
    assert "op" in off.columns and "op" in on.columns

    resid_off = _resid_count(off)
    assert resid_off >= len(_BASES), resid_off
    assert _op_count(off, WAVETABLE_DEF_OP) == 0
    assert _op_count(off, WAVETABLE_REF_OP) == 0

    assert (
        _op_count(on, WAVETABLE_DEF_OP) >= 1
    ), "WavetablePass did not fire through parse()"
    assert _op_count(on, WAVETABLE_REF_OP) >= len(_BASES)
    assert _resid_count(on) < resid_off

    assert np.array_equal(register_state(off), register_state(on))


def test_wavetable_default_matches_explicit_off():
    """The default args namespace (no ``wavetable_pass`` attr) parses identically to explicit OFF,
    proving the wiring addition cannot perturb the default golden stream."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wavetable_default.dump.parquet"))
        explicit_off = _parse(path, wavetable_pass=False)
        default_args = _args(wavetable_pass=False)
        delattr(default_args, "wavetable_pass")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))


def test_wavetable_pass_imported_in_parse_module():
    """Guard the import wiring so a refactor that drops ``WavetablePass`` from the parse module is
    caught at collection time."""
    import preframr_tokens.reglogparser as rlp

    assert hasattr(rlp, "WavetablePass")
