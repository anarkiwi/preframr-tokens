"""Codebook REF liveness mask (RESID_ZERO_PHASE3 §4 B2/B3): the constrained-decode mask forbids an
inline-codebook STAMP/PATCH REF whose id is not live (defined + committed), a DEF stashes the pending id
and a COMMIT makes it live, a later DEF rebinds it, and an out-of-window DEF can be seeded via
init_codebook_ids -- so the DEF->REF backref can no longer silently vanish at inference.
"""

import unittest

import pandas as pd

from preframr_tokens.constrained_decode import StreamState, precompute_vocab_arrays
from preframr_tokens.macros.validators import (
    codebook_live_ids,
    validate_codebook_refs,
    validate_stream,
)
from preframr_tokens.stfconstants import (
    FRAME_REG,
    PAD_REG,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_STEP_OP,
    PATCH_SUBREG_AD,
    PATCH_SUBREG_SR,
    SET_OP,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_REF_OP,
    WAVETABLE_DEF_OP,
    WAVETABLE_END_OP,
    WAVETABLE_REF_OP,
    WT_REF_SUBREG_ID,
)

PAD, FRAME, STAMP_DEF3, STAMP_END3, STAMP_REF3, STAMP_REF7 = 0, 1, 2, 3, 4, 5
PATCH_DEF5, PATCH_AD, PATCH_SR, PATCH_SET5 = 6, 7, 8, 9
WT_DEF9, WT_END9, WT_REF9 = 10, 11, 12

_ROWS = [
    {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
    {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11},
    {"op": STAMP_DEF_OP, "reg": 0, "subreg": -1, "val": 3},
    {"op": STAMP_END_OP, "reg": 0, "subreg": -1, "val": 3},
    {"op": STAMP_REF_OP, "reg": 0, "subreg": -1, "val": 3},
    {"op": STAMP_REF_OP, "reg": 0, "subreg": -1, "val": 7},
    {"op": PATCH_DEF_OP, "reg": 0, "subreg": -1, "val": 5},
    {"op": PATCH_STEP_OP, "reg": 0, "subreg": PATCH_SUBREG_AD, "val": 30},
    {"op": PATCH_STEP_OP, "reg": 0, "subreg": PATCH_SUBREG_SR, "val": 40},
    {"op": PATCH_SET_OP, "reg": 0, "subreg": -1, "val": 5},
    {"op": WAVETABLE_DEF_OP, "reg": 0, "subreg": -1, "val": 9},
    {"op": WAVETABLE_END_OP, "reg": 0, "subreg": -1, "val": 9},
    {"op": WAVETABLE_REF_OP, "reg": 0, "subreg": WT_REF_SUBREG_ID, "val": 9},
]


def _arrays():
    return precompute_vocab_arrays(pd.DataFrame(_ROWS))


def _state(**kw):
    defaults = {"init_frame_count": 5, "irq": 100, "disable_resource_masks": True}
    defaults.update(kw)
    return StreamState(_arrays(), **defaults)


def _masked(state, token):
    return bool(state.compute_invalid_mask()[token])


class TestCodebookMask(unittest.TestCase):
    def test_ref_to_undefined_id_is_masked(self):
        state = _state()
        self.assertTrue(_masked(state, STAMP_REF3))
        self.assertTrue(_masked(state, STAMP_REF7))
        self.assertTrue(_masked(state, PATCH_SET5))

    def test_def_always_allowed(self):
        state = _state()
        self.assertFalse(_masked(state, STAMP_DEF3))
        self.assertFalse(_masked(state, PATCH_DEF5))

    def test_def_then_commit_makes_ref_legal(self):
        state = _state()
        state.update(STAMP_DEF3)
        self.assertTrue(_masked(state, STAMP_REF3), "ref illegal before commit")
        state.update(STAMP_END3)
        self.assertFalse(_masked(state, STAMP_REF3), "ref legal after commit")
        self.assertTrue(_masked(state, STAMP_REF7), "other id still illegal")

    def test_patch_commits_on_sr_step(self):
        state = _state()
        state.update(PATCH_DEF5)
        state.update(PATCH_AD)
        self.assertTrue(_masked(state, PATCH_SET5))
        state.update(PATCH_SR)
        self.assertFalse(_masked(state, PATCH_SET5))

    def test_seeded_live_id_makes_ref_legal(self):
        state = _state(init_codebook_ids={0: {3}})
        self.assertFalse(_masked(state, STAMP_REF3), "seeded id legal from start")
        self.assertTrue(_masked(state, STAMP_REF7))

    def test_rebind_pending_def(self):
        state = _state()
        state.update(STAMP_DEF3)
        state.update(STAMP_END3)
        self.assertFalse(_masked(state, STAMP_REF3))
        self.assertEqual(state.codebook_live[0], {3})


def _df(*token_ids):
    return pd.DataFrame([_ROWS[t] for t in token_ids])


class TestWavetableCodebookMask(unittest.TestCase):
    def test_wavetable_ref_liveness(self):
        state = _state()
        self.assertTrue(_masked(state, WT_REF9), "ref illegal before def+commit")
        state.update(WT_DEF9)
        self.assertTrue(_masked(state, WT_REF9), "ref illegal before commit")
        state.update(WT_END9)
        self.assertFalse(_masked(state, WT_REF9), "ref legal after WAVETABLE_END")

    def test_wavetable_validation(self):
        self.assertTrue(validate_codebook_refs(_df(WT_DEF9, WT_END9, WT_REF9)))
        with self.assertRaises(AssertionError):
            validate_codebook_refs(_df(WT_REF9))

    def test_wavetable_live_ids_table_index(self):
        live = codebook_live_ids(_df(WT_DEF9, WT_END9))
        self.assertEqual(live[2], {9}, "wavetable is table index 2")


class TestCodebookValidation(unittest.TestCase):
    def test_accepts_def_commit_ref(self):
        self.assertTrue(validate_codebook_refs(_df(STAMP_DEF3, STAMP_END3, STAMP_REF3)))
        self.assertTrue(
            validate_stream(_df(STAMP_DEF3, STAMP_END3, STAMP_REF3, STAMP_REF3))
        )

    def test_rejects_ref_to_undefined(self):
        with self.assertRaises(AssertionError):
            validate_codebook_refs(_df(STAMP_REF3))
        with self.assertRaises(AssertionError):
            validate_stream(_df(STAMP_REF3))

    def test_rejects_ref_before_commit(self):
        with self.assertRaises(AssertionError):
            validate_codebook_refs(_df(STAMP_DEF3, STAMP_REF3))

    def test_patch_def_commit_set(self):
        self.assertTrue(
            validate_codebook_refs(_df(PATCH_DEF5, PATCH_AD, PATCH_SR, PATCH_SET5))
        )
        with self.assertRaises(AssertionError):
            validate_codebook_refs(_df(PATCH_DEF5, PATCH_AD, PATCH_SET5))

    def test_seeded_live_id_accepts_ref(self):
        self.assertTrue(validate_codebook_refs(_df(STAMP_REF3), live_ids={0: {3}}))
        self.assertTrue(validate_stream(_df(STAMP_REF3), live_ids={0: {3}}))


class TestMaterialization(unittest.TestCase):
    def test_live_ids_from_prior_context(self):
        live = codebook_live_ids(
            _df(STAMP_DEF3, STAMP_END3, PATCH_DEF5, PATCH_AD, PATCH_SR)
        )
        self.assertEqual(live[0], {3})
        self.assertEqual(live[1], {5})

    def test_uncommitted_def_is_not_live(self):
        self.assertEqual(codebook_live_ids(_df(STAMP_DEF3))[0], set())

    def test_materialized_window_ref_becomes_legal(self):
        live = codebook_live_ids(_df(STAMP_DEF3, STAMP_END3))
        window = _df(STAMP_REF3)
        with self.assertRaises(AssertionError):
            validate_stream(window)
        self.assertTrue(validate_stream(window, live_ids=live))
        state = StreamState(
            _arrays(),
            init_frame_count=5,
            irq=100,
            disable_resource_masks=True,
            init_codebook_ids=live,
        )
        self.assertFalse(_masked(state, STAMP_REF3))


class TestDecoderSnapshotSeed(unittest.TestCase):
    def test_seed_renders_out_of_window_ref(self):
        from preframr_tokens.macros.decode import expand_ops

        rows = [
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11, "diff": 100},
            {"op": STAMP_REF_OP, "reg": 0, "subreg": -1, "val": 3, "diff": 100},
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11, "diff": 100},
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11, "diff": 100},
        ]
        df = pd.DataFrame(rows)
        unseeded = expand_ops(df)
        seeded = expand_ops(df, codebook_seed={"stamp_table": {3: [{0: 100}]}})
        self.assertEqual(int((unseeded["reg"] == 0).sum()), 0, "ref drops without seed")
        seeded_reg0 = seeded[seeded["reg"] == 0]
        self.assertGreater(len(seeded_reg0), 0, "ref renders with seed")
        self.assertEqual(int(seeded_reg0["val"].iloc[0]), 100)


if __name__ == "__main__":
    unittest.main()
