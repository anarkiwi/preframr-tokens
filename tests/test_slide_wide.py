"""Parse-level guard for W4 (route SWEEP/monotone residue to SLIDE): a wide monotone note-relative
ramp the rate-only SLIDE reproduces exactly is routed to ``ORN_TYPE_SLIDE`` instead of leaking to
RESID -- a genuine glissando becomes a real primitive (not a one-shot dump) and is provenance-invariant
(same shape -> same SLIDE tokens at any base). Through the FULL ``RegLogParser.parse``; ``slide_wide``
OFF keeps it RESID, byte-exact. Sweep is OFF (it would claim the raw-freq ramp pre-skeleton).
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
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
)

_SPAN = 40


def _args(slide_wide):
    return parse_args(skeleton_pass=True, held_arp=True, slide_wide=slide_wide)


def _ramp(b, base, span):
    b.note([LUT[base]] + [LUT[base + off] for off in range(1, span + 1)])


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    _ramp(b, 36, _SPAN)
    b.note([LUT[30]] * 4)
    _ramp(b, 48, _SPAN)
    b.note([LUT[33]] * 5)
    return write_dump(b, path)


def _parse(path, slide_wide):
    return next(
        RegLogParser(args=_args(slide_wide)).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


def _type_count(df, orn_type):
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    return int(((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == orn_type)).sum())


def test_slide_wide_routes_wide_ramp_to_slide():
    """Through ``RegLogParser.parse`` a wide monotone ramp drains to SLIDE byte-exactly while OFF
    keeps it RESID."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "slide_wide.dump.parquet"))
        off = _parse(path, slide_wide=False)
        on = _parse(path, slide_wide=True)

    assert off is not None and on is not None
    assert _type_count(off, ORN_TYPE_RESID) >= 2
    assert _type_count(off, ORN_TYPE_SLIDE) == 0
    assert _type_count(on, ORN_TYPE_SLIDE) >= 2
    assert _type_count(on, ORN_TYPE_RESID) < _type_count(off, ORN_TYPE_RESID)
    assert np.array_equal(register_state(off), register_state(on))


def test_slide_wide_is_provenance_invariant():
    """The same ramp shape at two bases encodes to the same SLIDE ornament tokens (#11.4): identical
    (orn_type, params) regardless of the base the SKEL atom carries."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "slide_wide_prov.dump.parquet"))
        sigs = inline_note_signature(path, _args(slide_wide=True))
    slides = [(orn, offs) for _skel, orn, offs in sigs if orn == ORN_TYPE_SLIDE]
    assert len(slides) >= 2
    assert len(set(slides)) == 1, slides


def test_slide_wide_default_matches_explicit_off():
    """The default args namespace (no ``slide_wide`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "slide_wide_default.dump.parquet"))
        explicit_off = _parse(path, slide_wide=False)
        default_args = _args(slide_wide=False)
        delattr(default_args, "slide_wide")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
