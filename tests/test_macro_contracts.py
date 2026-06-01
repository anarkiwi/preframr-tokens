"""Macro interaction contracts: first-principles register/frame requirements declared per pass, a
static reasoner that surfaces latent mismatches by construction, and a stream check that verifies the
code honours the declarations. Guards the class of bug where one pass's decode effect (StampPass
replaying a freq) violates a later pass's encode assumption (PerRegBurst delta-basing on it).
"""

import unittest

import pandas as pd

from tests.sid_fixtures import FixtureUnavailable, grid_runner_dumps

from preframr_tokens.macros import FREQ_BLOCK_PASSES
from preframr_tokens.macros.macro_contracts import (
    CONTRACTS,
    KNOWN_MISMATCHES,
    PIPELINE_ORDER,
    REPLAY_OPS,
    interaction_mismatches,
    relative_base_unsound,
)
from preframr_tokens.macros.op_contracts import CODEBOOK_SPECS
from preframr_tokens.reglogparser import (
    RegLogParser,
    assert_elapsed_frames,
    elapsed_frames,
    remove_voice_reg,
)
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG
from preframr_tokens.tokenizer_config import default_tokenizer_args


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

    def test_every_codebook_ref_op_is_a_replay(self):
        """Every inline-codebook REF op (the ops that replay a stored register series at decode) must
        be classified REPLAY, so a delta encoder's barrier logic and the stream check account for it.
        A new codebook REF op fails here until it is added -- the latent-replay drift the bug was.
        """
        ref_ops = {int(op) for op, spec in CODEBOOK_SPECS.items() if spec.kind == "ref"}
        self.assertTrue(ref_ops)
        self.assertTrue(ref_ops <= REPLAY_OPS, ref_ops - REPLAY_OPS)


class TestStaticReasoner(unittest.TestCase):
    def test_interaction_mismatches_match_known(self):
        """The reasoner's surfaced mismatches must equal the documented KNOWN set exactly. A NEW
        latent mismatch (a new replay/relative pair without a barrier, or an anchored replay before a
        frame mutator) goes red here; resolving a known one without removing it from KNOWN also goes
        red. R1 (relative_base) must be empty -- PerRegBurst declares its StampPass barrier.
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


class TestRelativeBaseSoundnessOnStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _head, wide = grid_runner_dumps()
        except FixtureUnavailable as err:
            raise unittest.SkipTest(f"Grid Runner dump unavailable: {err}")
        cls.dump = str(wide)

    def test_stamp_stream_has_no_relative_write_across_a_replay(self):
        """The code backing the R1 contract: parse Grid Runner with stamp_pass and assert no DIFF/FLIP
        decodes against a register a STAMP_REF replayed (the 46-vs-86 bug). Empty iff PerRegBurst's
        barrier actually fires -- catches a code regression the static reasoner (declarations only)
        cannot."""
        parser = RegLogParser(args=default_tokenizer_args(stamp_pass=True))
        xdf = next(parser.parse(self.dump, max_perm=1, require_pq=False, reparse=True))
        df, _ = remove_voice_reg(xdf.copy(), {})
        bad = relative_base_unsound(df)
        self.assertEqual(bad, [], f"relative writes straddle a replay: {bad[:5]}")


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
