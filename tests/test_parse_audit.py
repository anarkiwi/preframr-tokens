"""Tests for the toggleable parse-time consistency audit. Verifies it is a no-op when off, stays
silent on a lossless transform, and pinpoints the transform that breaks losslessness, the elapsed-frame
budget, or reference integrity -- the bookkeeping that catches inconsistencies wholesale.
"""

import unittest

import pandas as pd

from preframr_tokens.parse_audit import PassAudit
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, SET_OP, _MIN_DIFF


def _stream(freqs, delays=None):
    """A minimal decodable stream: one FRAME (or DELAY) marker per frame plus a reg-0 SET per freq."""
    rows = []
    delays = delays or [None] * len(freqs)
    for v, d in zip(freqs, delays):
        if d is None:
            rows.append(
                {
                    "reg": int(FRAME_REG),
                    "val": 0,
                    "op": int(SET_OP),
                    "diff": int(_MIN_DIFF),
                }
            )
        else:
            rows.append(
                {
                    "reg": int(DELAY_REG),
                    "val": int(d),
                    "op": int(SET_OP),
                    "diff": int(_MIN_DIFF),
                }
            )
        if v is not None:
            rows.append(
                {"reg": 0, "val": int(v), "op": int(SET_OP), "diff": int(_MIN_DIFF)}
            )
    return pd.DataFrame(rows)


class TestPassAudit(unittest.TestCase):
    def test_off_is_noop(self):
        audit = PassAudit(None)
        audit.start(_stream([100, 110]))
        audit.after(_stream([100, 999]), "PerRegBurstPass")

    def test_lossless_pass_preserving_state_is_silent(self):
        audit = PassAudit("raise")
        audit.start(_stream([100, 110, 105]))
        audit.after(_stream([100, 110, 105]), "PerRegBurstPass")

    def test_lossless_pass_changing_state_raises(self):
        audit = PassAudit("raise")
        audit.start(_stream([100, 110, 105]))
        with self.assertRaises(AssertionError):
            audit.after(_stream([100, 120, 105]), "PerRegBurstPass")

    def test_elapsed_frame_budget_change_raises(self):
        audit = PassAudit("raise")
        audit.start(_stream([100, None], delays=[None, 5]))
        with self.assertRaises(AssertionError):
            audit.after(_stream([100, None], delays=[None, 3]), "PerRegBurstPass")

    def test_lossy_pass_rebaselines(self):
        """SkeletonPass is lossy by design (RESID-snap) -- the audit must re-baseline on it, not raise,
        and then hold the next lossless pass to the new state."""
        audit = PassAudit("raise")
        audit.start(_stream([100, 110]))
        audit.after(_stream([100, 120]), "SkeletonPass")
        audit.after(_stream([100, 120]), "PerRegBurstPass")
        with self.assertRaises(AssertionError):
            audit.after(_stream([100, 130]), "PerRegBurstPass")

    def test_ref_check_runs_after_every_pass(self):
        """validate_back_refs is loop-aware (counts DO_LOOP-expanded frames), so the audit runs the ref
        check after EVERY pass -- including the post-rotation passes where LoopPass mints back-refs -- so
        a malformed loop ref is pinpointed immediately, not only via the losslessness check.
        """
        import preframr_tokens.macros.validators as validators

        audit = PassAudit("raise")
        audit.start(_stream([100, 110]))
        calls = []
        original = validators.validate_stream
        validators.validate_stream = lambda df: calls.append(1)
        try:
            audit.after(_stream([100, 110]), "run_passes")
            audit.after(_stream([100, 110]), "_add_voice_reg")
            audit.after(_stream([100, 110]), "LoopPass")
        finally:
            validators.validate_stream = original
        self.assertEqual(calls, [1, 1, 1], "ref check should run after every pass")


if __name__ == "__main__":
    unittest.main()
