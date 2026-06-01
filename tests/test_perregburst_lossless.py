"""Adversarial round-trip property test for the PerRegBurst freq delta encoder. PerRegBurst re-encodes
per-frame freq SETs as DIFF/FLIP runs; decoding must reproduce the input exactly (the RESID=0 goal).
The FLIP detector over-matches some same-magnitude non-alternating delta runs (and DIFF mis-bases on
others), so it is currently lossy for ~1/5 of random small-delta walks -- xfail-strict until fixed.
"""

import unittest

import pandas as pd
import pytest

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.stfconstants import FRAME_REG, SET_OP, _MIN_DIFF
from preframr_tokens.tokenizer_config import default_tokenizer_args

_ARGS = default_tokenizer_args(freq_trajectory_pass=False)
_REG = 7


def _build(values):
    rows = []
    for v in values:
        rows.append(
            {"reg": int(FRAME_REG), "val": 0, "op": int(SET_OP), "diff": int(_MIN_DIFF)}
        )
        if v is not None:
            rows.append(
                {"reg": _REG, "val": int(v), "op": int(SET_OP), "diff": int(_MIN_DIFF)}
            )
    return pd.DataFrame(rows)


def _bad_frames(values):
    df = _build(values)
    out = PerRegBurstPass().apply(df.copy(), args=_ARGS)
    gt = register_state(df)[:, _REG]
    dec = register_state(out)[:, _REG]
    n = min(len(gt), len(dec))
    return [(i, int(gt[i]), int(dec[i])) for i in range(n) if gt[i] != dec[i]]


def _lcg(seed, n):
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x)
    return out


class TestPerRegBurstFreqLossless(unittest.TestCase):
    @pytest.mark.xfail(
        strict=True,
        reason="PerRegBurst._add_change_reg is still lossy on a few % of small-delta walks. The FLIP "
        "detector now requires genuine alternation (val == -neighbour), which fixed most cases; the "
        "residual is the begin/end short-run handling (2-row flip runs leave an orphan begin) plus a "
        "DIFF mis-base on some runs. Needs a lossless run-detection re-write; then drop this xfail.",
    )
    def test_random_small_delta_walks_round_trip(self):
        fails = []
        for trial in range(300):
            r = _lcg(12345 + trial * 7, 14)
            cur = 100 + (r[0] % 40)
            seq = [cur]
            for j in range(1, 12):
                cur = max(1, min(250, cur + [0, 1, -1, 2, -2][r[j] % 5]))
                seq.append(cur if (r[j] % 7) else None)
            if _bad_frames(seq):
                fails.append(seq)
        self.assertEqual(
            fails, [], f"{len(fails)}/300 freq walks lossy; e.g. {fails[:2]}"
        )

    def test_diff_only_round_trips_lossless(self):
        """The lossless floor: with FLIP disabled (DIFF-only), every walk must round-trip -- the
        fallback a correct encoder uses, and proof the lossiness is the FLIP/run detection, not DIFF.
        """
        bad = []
        for trial in range(150):
            r = _lcg(999 + trial * 13, 14)
            cur = 100 + (r[0] % 40)
            seq = [cur]
            for j in range(1, 12):
                cur = max(1, min(250, cur + [0, 1, -1, 2, -2][r[j] % 5]))
                seq.append(cur if (r[j] % 7) else None)
            df = _build(seq)
            from preframr_tokens.stfconstants import DIFF_OP

            out = PerRegBurstPass(opcodes=[int(DIFF_OP)]).apply(df.copy(), args=_ARGS)
            gt = register_state(df)[:, _REG]
            dec = register_state(out)[:, _REG]
            n = min(len(gt), len(dec))
            if any(gt[i] != dec[i] for i in range(n)):
                bad.append(seq)
        self.assertEqual(bad, [], f"DIFF-only lossy on {len(bad)} walks: {bad[:2]}")


if __name__ == "__main__":
    unittest.main()
