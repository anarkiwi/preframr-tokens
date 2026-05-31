"""Parse-level guard for W5.1 (exact-landing SLIDE2): a monotone ramp with a constant per-frame delta
the rate-only SLIDE cannot express (e.g. +2/frame ``[2,4,6,8]``) is routed to ``ORN_TYPE_SLIDE2``
(target + duration) instead of leaking to RESID, both narrow and wider than the offset limit. Driven
through the FULL ``RegLogParser.parse``; ``slide_landing`` OFF (even with ``slide_wide`` on) keeps it
RESID, byte-exact. Sweep is OFF (it would claim the raw-freq ramp pre-skeleton).
"""

import os
import tempfile

import numpy as np

from tests.parse_probes import (
    DumpBuilder,
    inline_note_signature,
    parse_args,
    write_dump,
)
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE2,
)


def _args(slide_landing, slide_wide=False):
    return parse_args(
        skeleton_pass=True,
        held_arp=True,
        slide_wide=slide_wide,
        slide_landing=slide_landing,
    )


def _ramp(b, base, span, step):
    b.note([LUT[base]] + [LUT[base + step * k] for k in range(1, span + 1)])


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    _ramp(b, 40, 4, 2)
    b.note([LUT[30]] * 4)
    _ramp(b, 36, 13, 2)
    b.note([LUT[33]] * 5)
    return write_dump(b, path)


def _parse(path, slide_landing, slide_wide=False):
    return next(
        RegLogParser(args=_args(slide_landing, slide_wide)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _type_count(df, orn_type):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    return int(((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == orn_type)).sum())


def test_slide_landing_routes_constant_delta_ramp():
    """Through ``RegLogParser.parse`` a +2/frame ramp (narrow and wide) drains to SLIDE2 byte-exactly,
    while OFF -- even with the rate-only wide SLIDE on -- keeps it RESID."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "slide_landing.dump.parquet"))
        off = _parse(path, slide_landing=False, slide_wide=True)
        on = _parse(path, slide_landing=True, slide_wide=True)

    assert off is not None and on is not None
    assert _type_count(off, ORN_TYPE_RESID) >= 2
    assert _type_count(off, ORN_TYPE_SLIDE2) == 0
    assert _type_count(on, ORN_TYPE_SLIDE2) >= 2
    assert _type_count(on, ORN_TYPE_RESID) < _type_count(off, ORN_TYPE_RESID)
    assert np.array_equal(register_state(off), register_state(on))


def test_slide_landing_is_provenance_invariant():
    """The same constant-delta ramp shape at two bases encodes to the same SLIDE2 ornament tokens:
    identical (orn_type, params) regardless of the base the SKEL atom carries."""
    with tempfile.TemporaryDirectory() as tmp:
        b = DumpBuilder().adsr().pw(0x800)
        b.note([LUT[60]] * 5)
        _ramp(b, 40, 6, 2)
        b.note([LUT[30]] * 4)
        _ramp(b, 52, 6, 2)
        b.note([LUT[33]] * 5)
        path = write_dump(b, os.path.join(tmp, "slide_landing_prov.dump.parquet"))
        sigs = inline_note_signature(path, _args(slide_landing=True))
    slides = [(orn, offs) for _skel, orn, offs in sigs if orn == ORN_TYPE_SLIDE2]
    assert len(slides) >= 2
    assert len(set(slides)) == 1, slides


def test_slide_landing_default_matches_explicit_off():
    """The default args namespace (no ``slide_landing`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "slide_landing_default.dump.parquet"))
        explicit_off = _parse(path, slide_landing=False)
        default_args = _args(slide_landing=False)
        delattr(default_args, "slide_landing")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
