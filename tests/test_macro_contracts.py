"""Macro interaction contracts: first-principles register/frame requirements declared per pass, a
static reasoner that surfaces latent mismatches by construction, and a stream check that verifies the
code honours the declarations. Guards the class of bug where one pass's decode effect (a codebook pass
replaying a freq) violates a later pass's encode assumption (PerRegBurst delta-basing on it).
"""

import unittest

import pandas as pd

from preframr_tokens.macros import FREQ_BLOCK_PASSES
from preframr_tokens.macros.macro_contracts import (
    CONTRACTS,
    KNOWN_MISMATCHES,
    PIPELINE_ORDER,
    interaction_mismatches,
)
from preframr_tokens.macros.op_contracts import CODEBOOK_SPECS
from preframr_tokens.reglogparser import (
    assert_elapsed_frames,
    elapsed_frames,
)
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG


class TestContractCompleteness(unittest.TestCase):
    def test_every_pipeline_name_has_a_contract(self):
        """Every name the reasoner orders over must declare a contract, and every freq-block pass
        class must be in the pipeline -- a new freq pass then forces a contract (or fails here)
        rather than silently entering the interaction matrix unmodelled."""
        for name in PIPELINE_ORDER:
            self.assertIn(name, CONTRACTS, name)
        for macro_pass in FREQ_BLOCK_PASSES:
            self.assertIn(
                type(macro_pass).__name__, PIPELINE_ORDER, type(macro_pass).__name__
            )

    def test_no_codebook_ref_ops_remain(self):
        """The inline-codebook families are retired (the events codec owns those channels), so the
        derived spec table carries no REF ops."""
        ref_ops = {int(op) for op, spec in CODEBOOK_SPECS.items() if spec.kind == "ref"}
        self.assertEqual(ref_ops, set())


class TestStaticReasoner(unittest.TestCase):
    def test_interaction_mismatches_match_known(self):
        """The reasoner's surfaced mismatches must equal the documented KNOWN set exactly. A NEW
        latent mismatch (a new replay/relative pair without a barrier, or an anchored replay before a
        frame mutator) goes red here; resolving a known one without removing it from KNOWN also goes
        red. R1 (relative_base) must be empty.
        """
        found = set(interaction_mismatches())
        self.assertEqual(
            found,
            set(KNOWN_MISMATCHES),
            f"unexpected: {found - set(KNOWN_MISMATCHES)}; "
            f"stale: {set(KNOWN_MISMATCHES) - found}",
        )
        self.assertFalse(
            [m for m in found if m.kind == "relative_base"],
            "a relative-base mismatch is unbarriered",
        )


class TestElapsedFrameConservation(unittest.TestCase):
    def test_elapsed_frames_counts_frame_and_delay_ticks(self):
        """elapsed_frames is one tick per FRAME marker plus each DELAY's value (non-markers ignored)
        -- the decoded frame budget every lossless macro/transform must conserve."""
        df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0},
                {"reg": 7, "val": 9000},
                {"reg": DELAY_REG, "val": 5},
                {"reg": FRAME_REG, "val": 0},
            ]
        )
        self.assertEqual(elapsed_frames(df), 1 + 5 + 1)

    def test_assert_elapsed_frames_raises_where_the_budget_changes(self):
        """The conservation law RegLogParser.parse enforces after each lossless transform: a changed
        frame budget raises, pinpointing the transform; an unchanged one is silent."""
        df = pd.DataFrame([{"reg": FRAME_REG, "val": 0}])
        assert_elapsed_frames(df, 1, "noop")
        with self.assertRaises(AssertionError):
            assert_elapsed_frames(df, 2, "shrank")


if __name__ == "__main__":
    unittest.main()
