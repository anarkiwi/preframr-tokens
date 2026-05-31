"""Parse-level guard for W3 (inline one-shot, the RESID=0 backstop): every surviving RESID note that
matched no codebook id emits a self-contained ``WAVETABLE_ONESHOT_OP`` (offsets stored verbatim, no
codebook id, no pitched-core gate), so ``ORN_TYPE_RESID`` cannot reach the deployed stream. Driven
through the FULL ``RegLogParser.parse``; ``wt_oneshot`` OFF keeps the FLAT/no-core residue as RESID,
byte-exact both ways. Stamp/Sweep are OFF so the synthetic residue reaches the wavetable proposer.
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
    WAVETABLE_ONESHOT_OP,
)

_NOISE = 0x81
_FLAT = (26, 5, 9, 14)


def _args(wt_oneshot):
    return parse_args(
        skeleton_pass=True,
        held_arp=True,
        wavetable_pass=True,
        wt_short=True,
        wt_oneshot=wt_oneshot,
    )


def _noise_core(b, base):
    """An all-noise note (every frame non-pitched, wide-jumping): the onset-strip consumes the whole
    core, so the codebook key drops it -> RESID; the one-shot stores its offsets verbatim regardless.
    """
    b.frame().ctrl(0x40).ctrl(_NOISE).freq(LUT[base])
    for off in (20, 30, 18):
        b.frame().ctrl(_NOISE).freq(LUT[base + off])


def _build_dump(path):
    b = DumpBuilder().adsr().pw(0x800)
    b.note([LUT[60]] * 5)
    b.note([LUT[60]] + [LUT[60 + off] for off in _FLAT])
    b.note([LUT[30]] * 4)
    _noise_core(b, 50)
    b.note([LUT[33]] * 5)
    return write_dump(b, path)


def _parse(path, wt_oneshot):
    return next(
        RegLogParser(args=_args(wt_oneshot)).parse(
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


def test_oneshot_drains_all_resid_to_zero():
    """Through ``RegLogParser.parse`` every FLAT / no-pitched-core RESID note emits an inline one-shot
    so the deployed stream has zero RESID, byte-exact, while OFF keeps them RESID with no one-shots.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_oneshot.dump.parquet"))
        off = _parse(path, wt_oneshot=False)
        on = _parse(path, wt_oneshot=True)

    assert off is not None and on is not None
    assert _resid_count(off) >= 2
    assert _op_count(off, WAVETABLE_ONESHOT_OP) == 0

    assert _resid_count(on) == 0
    assert _op_count(on, WAVETABLE_ONESHOT_OP) >= 1
    assert np.array_equal(register_state(off), register_state(on))


def test_oneshot_contract_is_self_contained_atom():
    """W6: the one-shot is contracted as a self-contained ATOM (no out-of-window materialization),
    carrying no codebook id -- it is absent from CODEBOOK_SPECS, unlike the WAVETABLE_REF path.
    """
    from preframr_tokens.macros.op_contracts import (
        CODEBOOK_SPECS,
        OP_CONTRACTS,
        MaskRole,
    )

    contract = OP_CONTRACTS.get(WAVETABLE_ONESHOT_OP)
    assert contract is not None
    assert contract.role == MaskRole.ATOM
    assert WAVETABLE_ONESHOT_OP not in CODEBOOK_SPECS


def test_oneshot_default_matches_explicit_off():
    """The default args namespace (no ``wt_oneshot`` attr) parses identically to explicit OFF."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _build_dump(os.path.join(tmp, "wt_oneshot_default.dump.parquet"))
        explicit_off = _parse(path, wt_oneshot=False)
        default_args = _args(wt_oneshot=False)
        delattr(default_args, "wt_oneshot")
        default = next(
            RegLogParser(args=default_args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )
    assert default is not None and explicit_off is not None
    assert default.reset_index(drop=True).equals(explicit_off.reset_index(drop=True))
