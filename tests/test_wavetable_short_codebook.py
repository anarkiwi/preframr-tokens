"""Parse-level guard for W2 (short literal-tuple WAVETABLE codebook): residue at/below the length
threshold keys on its verbatim offset tuple (no factorise/onset-strip/pitched-core), so RECUR/SHORT
survivors the factorise key misses (length-1 and noise-onset transients) drain to a DEF + N REFs
through the FULL ``RegLogParser.parse``; ``wt_short`` OFF keeps them RESID, byte-exact. Stamp/Sweep
are OFF (their ABS/raw-delta recurrence would eat these synthetic semitone-relative spans first).
"""

import os
import tempfile

import numpy as np

from tests.parse_probes import DumpBuilder, parse_args, write_dump
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_RESID,
    WAVETABLE_DEF_OP,
    WAVETABLE_REF_OP,
)

_NOISE = 0x81
_SEP_BASE = 30


def _args(wt_short):
    return parse_args(
        skeleton_pass=True,
        held_arp=True,
        wavetable_pass=True,
        wt_short=wt_short,
    )


def _noise_onset(b, base):
    """A [33, 0] transient whose +33 onset frame is noise (non-pitched): the old onset-strip drops
    its core below _MIN_CORE -> RESID; the literal key keeps the verbatim [33, 0]."""
    b.frame().ctrl(0x40).ctrl(0x41).freq(LUT[base])
    b.frame().ctrl(_NOISE).freq(LUT[base + 33])
    b.frame().ctrl(0x41).freq(LUT[base])


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    for base in (40, 52, 47):
        b.note([LUT[base], LUT[base + 31]])
        b.note([LUT[_SEP_BASE]] * 4)
    for base in (65, 70, 67):
        _noise_onset(b, base)
        b.note([LUT[_SEP_BASE]] * 4)
    b.note([LUT[55]] * 5)
    return write_dump(b, path)


def _parse(path, wt_short):
    return next(
        RegLogParser(args=_args(wt_short)).parse(
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


def test_short_codebook_drains_recurring_transients():
    """Through ``RegLogParser.parse`` the recurring length-1 and noise-onset short residue drains
    to a codebook byte-exactly, while OFF keeps them RESID with no codebook ids."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_short.dump.parquet"))
        off = _parse(path, wt_short=False)
        on = _parse(path, wt_short=True)

    assert off is not None and on is not None
    resid_off = _resid_count(off)
    assert resid_off >= 6, resid_off
    assert _op_count(off, WAVETABLE_DEF_OP) == 0
    assert _op_count(on, WAVETABLE_DEF_OP) >= 1
    assert _op_count(on, WAVETABLE_REF_OP) >= 6
    assert _resid_count(on) < resid_off
    assert np.array_equal(register_state(off), register_state(on))


def test_short_codebook_drains_to_zero_resid():
    """All six recurring short transients drain: the deployed stream has no RESID left."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_short_zero.dump.parquet"))
        on = _parse(path, wt_short=True)
    assert _resid_count(on) == 0


def test_short_codebook_bounded_vocab():
    """The literal codebook only mints ids for tuples that recur: the two distinct recurring tuples
    ([31] and the noise [33, 0]) yield exactly two DEFs, not one per note."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_short_vocab.dump.parquet"))
        on = _parse(path, wt_short=True)
    assert _op_count(on, WAVETABLE_DEF_OP) == 2


def test_short_codebook_default_matches_explicit_off():
    """The default args namespace (no ``wt_short`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_short_default.dump.parquet"))
        explicit_off = _parse(path, wt_short=False)
        default_args = _args(wt_short=False)
        delattr(default_args, "wt_short")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
