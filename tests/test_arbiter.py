"""Arbiter unit tests: a single claim is byte-identical to a direct splice; overlapping claims
are resolved to a write-partition by score/priority/source-order; coverage is preserved
(unclaimed writes survive)."""

import pandas as pd

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import _splice_rows


def _df(n):
    return pd.DataFrame(
        {
            "reg": list(range(n)),
            "val": [v * 10 for v in range(n)],
            "diff": [0] * n,
            "op": [0] * n,
            "subreg": [-1] * n,
            "irq": [0] * n,
            "description": [0] * n,
        }
    )


def _tok(reg, val, pos):
    return {
        "reg": reg,
        "val": val,
        "diff": 0,
        "op": 99,
        "subreg": 0,
        "irq": 0,
        "description": 0,
        "__pos": pos,
    }


def test_single_claim_matches_splice():
    df = _df(6)
    drop_idx = [2, 3]
    new_rows = [_tok(200, 1, 2)]
    direct = _splice_rows(df, drop_idx, new_rows)
    viaarb = arbitrate(df, [Claim(writes=(2, 3), tokens=[_tok(200, 1, 2)])])
    pd.testing.assert_frame_equal(direct, viaarb)


def test_empty_claim_is_noop():
    df = _df(4)
    out = arbitrate(df, [Claim(writes=(1,), tokens=[])])
    pd.testing.assert_frame_equal(df, out)


def test_no_claims_returns_df():
    df = _df(4)
    pd.testing.assert_frame_equal(df, arbitrate(df, []))


def test_higher_score_wins_overlap():
    df = _df(6)
    lo = Claim(writes=(2, 3), tokens=[_tok(100, 1, 2)], score=(1, 0, 0), priority=0)
    hi = Claim(writes=(3, 4), tokens=[_tok(200, 1, 3)], score=(2, 0, 0), priority=9)
    out = arbitrate(df, [lo, hi])
    assert 200 in out["reg"].tolist()
    assert 100 not in out["reg"].tolist()
    assert 2 in out["reg"].tolist()


def test_priority_breaks_score_tie():
    df = _df(6)
    a = Claim(writes=(2,), tokens=[_tok(100, 1, 2)], score=(1, 0, 0), priority=0)
    b = Claim(writes=(2,), tokens=[_tok(200, 1, 2)], score=(1, 0, 0), priority=5)
    out = arbitrate(df, [b, a])
    assert 100 in out["reg"].tolist()
    assert 200 not in out["reg"].tolist()


def test_nonoverlapping_claims_both_applied():
    df = _df(8)
    a = Claim(writes=(1, 2), tokens=[_tok(100, 1, 1)])
    b = Claim(writes=(5, 6), tokens=[_tok(200, 1, 5)])
    out = arbitrate(df, [a, b])
    regs = out["reg"].tolist()
    assert 100 in regs and 200 in regs


def test_determinism_independent_of_input_order():
    df = _df(8)
    claims = [
        Claim(writes=(1,), tokens=[_tok(100, 1, 1)], score=(2, 0, 0), priority=1),
        Claim(writes=(1,), tokens=[_tok(200, 1, 1)], score=(2, 0, 0), priority=0),
        Claim(writes=(4,), tokens=[_tok(300, 1, 4)], score=(1, 0, 0), priority=2),
    ]
    out1 = arbitrate(df, claims)
    out2 = arbitrate(df, list(reversed(claims)))
    pd.testing.assert_frame_equal(out1, out2)
