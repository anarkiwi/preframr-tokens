"""Parse-level guard for W1 (ZERO -> PLAIN): a note held at its base whose freq only moves on
unresolvable (silent/noise) frames the content floor snaps to 0 is an all-offset-0 RESID note.
Through the FULL ``RegLogParser.parse`` the ``zero_plain`` gate rewrites it to ``ORN_TYPE_PLAIN``
byte-exactly (the content floor already replays base), while default/OFF keeps the RESID escape.
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
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
)


def _args(zero_plain):
    return parse_args(
        skeleton_pass=True,
        trajectory_anchor_pass=True,
        held_arp=True,
        zero_plain=zero_plain,
    )


def _build_dump(path):
    """Three gated notes, the middle one held at its base but with an unresolvable (freq=0)
    frame the floor snaps to 0 -> an all-offset-0 RESID note (the ZERO survivor class).
    """
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    b.note([LUT[55], 0, LUT[55], LUT[55]])
    b.note([LUT[48]] * 5)
    return write_dump(b, path)


def _parse(path, zero_plain):
    return next(
        RegLogParser(args=_args(zero_plain)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _type_count(df, orn_type):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    return int(((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == orn_type)).sum())


def test_zero_plain_rewrites_all_zero_resid():
    """Through ``RegLogParser.parse`` the all-offset-0 RESID drains to PLAIN byte-exactly while
    OFF keeps it RESID."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "zero_plain.dump.parquet"))
        off = _parse(path, zero_plain=False)
        on = _parse(path, zero_plain=True)

    assert off is not None and on is not None
    resid_off = _type_count(off, ORN_TYPE_RESID)
    assert resid_off >= 1, resid_off
    assert _type_count(on, ORN_TYPE_RESID) < resid_off
    assert _type_count(on, ORN_TYPE_PLAIN) > _type_count(off, ORN_TYPE_PLAIN)
    assert np.array_equal(register_state(off), register_state(on))


def test_zero_plain_default_matches_explicit_off():
    """The default args namespace (no ``zero_plain`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "zero_plain_default.dump.parquet"))
        explicit_off = _parse(path, zero_plain=False)
        default_args = _args(zero_plain=False)
        delattr(default_args, "zero_plain")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
