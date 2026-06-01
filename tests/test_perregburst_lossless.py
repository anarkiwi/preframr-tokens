"""Adversarial round-trip property test for the PerRegBurst freq delta encoder. PerRegBurst re-encodes
per-frame freq SETs as DIFF/FLIP runs; decoding must reproduce the input exactly (the RESID=0 goal).
After the lossless run-detection rewrite (FLIP fires only on genuine >=3 alternation), random
small-delta walks round-trip exactly -- this guards that, with DIFF-only as the lossless floor.
"""

import unittest

import pandas as pd

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


class TestPerRegBurstFallbackTool(unittest.TestCase):
    """The opt-in per-register lossless fallback: off by default, and when on it re-encodes only the
    registers the delta pass could not round-trip (a rare pval mis-base) as literal SET.
    """

    def test_fallback_off_by_default_on_demand_when_set(self):
        from preframr_tokens.macros.per_reg_burst import _fallback_enabled

        self.assertFalse(_fallback_enabled(None))

        class _Args:
            perreg_lossless_fallback = True

        self.assertTrue(_fallback_enabled(_Args()))

    def test_lossy_regs_flags_only_diverging_registers(self):
        from preframr_tokens.macros.per_reg_burst import _lossy_regs

        clean = _build([100, 110, 105])
        diverged = _build([100, 120, 105])
        self.assertEqual(_lossy_regs(clean, clean, had_op=True), frozenset())
        self.assertEqual(_lossy_regs(clean, diverged, had_op=True), frozenset({_REG}))


if __name__ == "__main__":
    unittest.main()
