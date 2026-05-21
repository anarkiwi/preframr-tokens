"""Unit tests for ``preframr_tokens.constrained_decode``. Torch-free: mask checks call ``state._compute_invalid()`` (returns numpy bool); the torch glue at the masked_fill boundary is exercised in the main repo's predict tests."""

import unittest

import numpy as np
import pandas as pd

from preframr_tokens.constrained_decode import (
    StreamState,
    _frame_marker_count,
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
)
from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    DELAY_REG,
    FRAME_REG,
    MIN_DIFF,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    SET_OP,
    VOICE_REG,
)

VOCAB_ROWS = [
    {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
    {"op": SET_OP, "reg": 0, "subreg": -1, "val": 7},
    {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 11},
    {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 5},
    {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 2},
    {"op": SET_OP, "reg": VOICE_REG, "subreg": -1, "val": 0},
    {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_DIST_HI, "val": 0},
    {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_DIST_LO, "val": 1},
    {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_DIST_LO, "val": 5},
    {"op": BACK_REF_OP, "reg": -125, "subreg": BACK_REF_SUBREG_LEN, "val": 1},
    {
        "op": PATTERN_REPLAY_OP,
        "reg": -125,
        "subreg": PATTERN_REPLAY_SUBREG_DIST_HI,
        "val": 0,
    },
    {
        "op": PATTERN_REPLAY_OP,
        "reg": -125,
        "subreg": PATTERN_REPLAY_SUBREG_DIST_LO,
        "val": 2,
    },
    {
        "op": PATTERN_REPLAY_OP,
        "reg": -125,
        "subreg": PATTERN_REPLAY_SUBREG_LEN,
        "val": 1,
    },
    {
        "op": PATTERN_REPLAY_OP,
        "reg": -125,
        "subreg": PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
        "val": 2,
    },
    {
        "op": PATTERN_OVERLAY_OP,
        "reg": -125,
        "subreg": PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
        "val": 0,
    },
    {
        "op": PATTERN_OVERLAY_OP,
        "reg": -125,
        "subreg": PATTERN_OVERLAY_SUBREG_TARGET_REG,
        "val": 4,
    },
    {
        "op": PATTERN_OVERLAY_OP,
        "reg": -125,
        "subreg": PATTERN_OVERLAY_SUBREG_NEW_VAL,
        "val": 99,
    },
]

PAD = 0
SET_R0 = 1
FRAME11 = 2
FRAME5 = 3
DELAY = 4
VOICE = 5
BR_HI = 6
BR_LO1 = 7
BR_LO5 = 8
BR_LEN = 9
PR_HI = 10
PR_LO = 11
PR_LEN = 12
PR_OV = 13
PO_FO = 14
PO_TR = 15
PO_NV = 16


def _build_vocab():
    return pd.DataFrame(VOCAB_ROWS)


def _mask_bool(state, n_vocab):
    invalid = state._compute_invalid()  # pylint: disable=protected-access
    return [bool(invalid[i]) for i in range(n_vocab)]


class TestFrameMarkerCount(unittest.TestCase):
    def test_counts_frame_and_delay(self):
        is_frame_marker = np.array([False, True, True], dtype=bool)
        ids = [0, 1, 0, 2, 1]
        self.assertEqual(_frame_marker_count(ids, is_frame_marker), 3)

    def test_empty(self):
        self.assertEqual(_frame_marker_count([], np.zeros(1, dtype=bool)), 0)


class TestPrecomputeVocabArrays(unittest.TestCase):
    def test_keys_present(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df)
        for k in (
            "is_frame_marker",
            "is_delay_reg",
            "is_pad",
            "is_real_reg",
            "is_back_ref_dist_hi",
            "is_back_ref_dist_lo",
            "is_back_ref_len",
            "is_pattern_replay_dist_hi",
            "is_pattern_replay_dist_lo",
            "is_pattern_replay_len",
            "is_pattern_replay_ov_count",
            "is_dist_hi_row",
            "is_dist_lo_row",
            "is_pair_intermediate",
            "is_pattern_overlay",
            "is_frame_reg_strict",
            "is_voice_reg",
            "frame_sval",
            "dist_hi_val",
            "dist_lo_val",
            "overlay_count",
        ):
            self.assertIn(k, arrs, k)
        self.assertEqual(arrs["n_vocab"], len(df))

    def test_per_subreg_flags(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df)
        self.assertTrue(bool(arrs["is_back_ref_dist_hi"][BR_HI].item()))
        self.assertTrue(bool(arrs["is_back_ref_dist_lo"][BR_LO1].item()))
        self.assertTrue(bool(arrs["is_back_ref_dist_lo"][BR_LO5].item()))
        self.assertTrue(bool(arrs["is_back_ref_len"][BR_LEN].item()))
        self.assertTrue(bool(arrs["is_pattern_replay_dist_hi"][PR_HI].item()))
        self.assertTrue(bool(arrs["is_pattern_replay_dist_lo"][PR_LO].item()))
        self.assertTrue(bool(arrs["is_pattern_replay_len"][PR_LEN].item()))
        self.assertTrue(bool(arrs["is_pattern_replay_ov_count"][PR_OV].item()))
        self.assertTrue(bool(arrs["is_dist_hi_row"][BR_HI].item()))
        self.assertTrue(bool(arrs["is_dist_hi_row"][PR_HI].item()))
        self.assertFalse(bool(arrs["is_dist_hi_row"][BR_LO1].item()))
        self.assertTrue(bool(arrs["is_dist_lo_row"][BR_LO1].item()))
        self.assertTrue(bool(arrs["is_dist_lo_row"][BR_LO5].item()))
        self.assertTrue(bool(arrs["is_dist_lo_row"][PR_LO].item()))
        self.assertTrue(bool(arrs["is_pair_intermediate"][BR_LO1].item()))
        self.assertTrue(bool(arrs["is_pair_intermediate"][BR_LEN].item()))
        self.assertTrue(bool(arrs["is_pair_intermediate"][PR_LEN].item()))
        self.assertTrue(bool(arrs["is_pair_intermediate"][PR_OV].item()))
        self.assertFalse(bool(arrs["is_pair_intermediate"][BR_HI].item()))

    def test_dist_hi_and_lo_val_extraction(self):
        df = _build_vocab()
        arrs = precompute_vocab_arrays(df)
        self.assertEqual(int(arrs["dist_hi_val"][BR_HI].item()), 0)
        self.assertEqual(int(arrs["dist_lo_val"][BR_LO1].item()), 1)
        self.assertEqual(int(arrs["dist_lo_val"][BR_LO5].item()), 5)
        self.assertEqual(int(arrs["dist_lo_val"][PR_LO].item()), 2)
        self.assertEqual(int(arrs["overlay_count"][PR_OV].item()), 2)


class TestStreamStateMasking(unittest.TestCase):
    def setUp(self):
        self.df = _build_vocab()
        self.arrs = precompute_vocab_arrays(self.df)
        self.n = self.arrs["n_vocab"]

    def _state(self, **kw):
        defaults = dict(
            init_frame_count=10,
            irq=19656,
            init_budget=19656,
        )
        defaults.update(kw)
        return StreamState(self.arrs, **defaults)

    def test_pad_always_masked(self):
        m = _mask_bool(self._state(), self.n)
        self.assertTrue(m[PAD])

    def test_delay_reg_masked_at_top_level(self):
        m = _mask_bool(self._state(), self.n)
        self.assertTrue(m[DELAY])

    def test_pattern_overlay_orphan_masked_at_top_level(self):
        m = _mask_bool(self._state(), self.n)
        for i in (PO_FO, PO_TR, PO_NV):
            self.assertTrue(m[i])

    def test_pair_intermediate_orphan_masked_at_top_level(self):
        m = _mask_bool(self._state(), self.n)
        self.assertTrue(m[BR_LO1])
        self.assertTrue(m[BR_LEN])
        self.assertTrue(m[PR_LO])
        self.assertTrue(m[PR_LEN])
        self.assertTrue(m[PR_OV])

    def test_back_ref_distance_check_too_far(self):
        m = _mask_bool(self._state(init_frame_count=0), self.n)
        self.assertTrue(m[BR_HI])

    def test_back_ref_within_bounds(self):
        m = _mask_bool(self._state(init_frame_count=5), self.n)
        self.assertFalse(m[BR_HI])

    def test_pending_dist_lo_admits_only_lo_in_range(self):
        state = self._state(init_frame_count=10)
        state.pending_back_ref_dist_lo = True
        state.current_dist_hi = 0
        m = _mask_bool(state, self.n)
        self.assertFalse(m[BR_LO1])
        self.assertFalse(m[BR_LO5])
        for i in range(self.n):
            if i not in (BR_LO1, BR_LO5):
                self.assertTrue(m[i])

    def test_pending_dist_lo_too_far_masked(self):
        state = self._state(init_frame_count=3)
        state.pending_back_ref_dist_lo = True
        state.current_dist_hi = 0
        m = _mask_bool(state, self.n)
        self.assertFalse(m[BR_LO1])
        self.assertTrue(m[BR_LO5])

    def test_pending_back_ref_len_admits_only_len_row(self):
        state = self._state()
        state.pending_back_ref_len = True
        m = _mask_bool(state, self.n)
        self.assertFalse(m[BR_LEN])
        for i in range(self.n):
            if i != BR_LEN:
                self.assertTrue(m[i])

    def test_pending_pr_len_admits_only_pr_len(self):
        state = self._state()
        state.pending_pr_len = True
        m = _mask_bool(state, self.n)
        self.assertFalse(m[PR_LEN])
        for i in range(self.n):
            if i != PR_LEN:
                self.assertTrue(m[i])

    def test_pending_pr_ov_count_admits_only_pr_ov_count(self):
        state = self._state()
        state.pending_pr_ov_count = True
        m = _mask_bool(state, self.n)
        self.assertFalse(m[PR_OV])

    def test_pattern_overlay_inside_block_walks_slots(self):
        state = self._state()
        state.pending_overlays = 2
        m = _mask_bool(state, self.n)
        self.assertFalse(m[PO_FO])
        state.pending_overlay_slot = 1
        m = _mask_bool(state, self.n)
        self.assertFalse(m[PO_TR])
        state.pending_overlay_slot = 2
        m = _mask_bool(state, self.n)
        self.assertFalse(m[PO_NV])

    def test_diff_budget_exhaustion(self):
        state = self._state(init_budget=MIN_DIFF - 1)
        m = _mask_bool(state, self.n)
        self.assertTrue(m[SET_R0])
        self.assertFalse(m[FRAME11])

    def test_remaining_steps_blocks_pair_open_when_too_few_steps(self):
        m = _mask_bool(self._state(remaining_steps=2, init_frame_count=10), self.n)
        self.assertTrue(m[BR_HI])
        m2 = _mask_bool(self._state(remaining_steps=3, init_frame_count=10), self.n)
        self.assertTrue(m2[PR_HI])

    def test_all_masked_safety_valve(self):
        m = _mask_bool(self._state(init_frame_count=0, init_budget=0), self.n)
        self.assertIn(False, m)


class TestStreamStateUpdate(unittest.TestCase):
    def setUp(self):
        self.df = _build_vocab()
        self.arrs = precompute_vocab_arrays(self.df)

    def test_frame_marker_increments_count_resets_budget(self):
        state = StreamState(self.arrs, init_frame_count=0, irq=100, init_budget=10)
        state.update(FRAME11)
        self.assertEqual(state.frame_count, 1)
        self.assertEqual(state.frame_budget, 100)

    def test_real_reg_charges_budget(self):
        state = StreamState(self.arrs, init_frame_count=0, irq=100, init_budget=64)
        state.update(SET_R0)
        self.assertEqual(state.frame_budget, 64 - MIN_DIFF)

    def test_voice_rotation_tracking(self):
        state = StreamState(self.arrs, init_frame_count=0, irq=100)
        state.update(FRAME11)
        self.assertEqual(state.current_sval, 11)
        self.assertEqual(state.current_fn, 0)
        state.update(VOICE)
        self.assertEqual(state.current_fn, 1)

    def test_back_ref_triple_state_machine(self):
        state = StreamState(self.arrs, init_frame_count=10, irq=100)
        state.update(BR_HI)
        self.assertTrue(state.pending_back_ref_dist_lo)
        self.assertEqual(state.current_dist_hi, 0)
        state.update(BR_LO1)
        self.assertFalse(state.pending_back_ref_dist_lo)
        self.assertTrue(state.pending_back_ref_len)
        state.update(BR_LEN)
        self.assertFalse(state.pending_back_ref_len)

    def test_pattern_replay_quad_state_machine(self):
        state = StreamState(self.arrs, init_frame_count=10, irq=100)
        state.update(PR_HI)
        self.assertTrue(state.pending_pr_dist_lo)
        state.update(PR_LO)
        self.assertFalse(state.pending_pr_dist_lo)
        self.assertTrue(state.pending_pr_len)
        state.update(PR_LEN)
        self.assertFalse(state.pending_pr_len)
        self.assertTrue(state.pending_pr_ov_count)
        state.update(PR_OV)
        self.assertFalse(state.pending_pr_ov_count)
        self.assertEqual(state.pending_overlays, 2)

    def test_overlay_consumed(self):
        state = StreamState(self.arrs, init_frame_count=10, irq=100)
        state.update(PR_HI)
        state.update(PR_LO)
        state.update(PR_LEN)
        state.update(PR_OV)
        self.assertEqual(state.pending_overlays, 2)
        state.update(PO_FO)
        state.update(PO_TR)
        state.update(PO_NV)
        self.assertEqual(state.pending_overlays, 1)
        state.update(PO_FO)
        state.update(PO_TR)
        state.update(PO_NV)
        self.assertEqual(state.pending_overlays, 0)


class _FakeTkModel:
    def __init__(self, sub_strings):
        self._strs = ["<unk>"] + list(sub_strings)

    def get_vocab_size(self):
        return len(self._strs)

    def id_to_token(self, sub_id):
        if sub_id < 0 or sub_id >= len(self._strs):
            return None
        return self._strs[sub_id]


def _fake_regtokenizer(tokens_df, sub_id_atomic_lists):
    class _Args:
        tkvocab = 0
        tkmodel = None
        tokenizer = "unigram"

    rtok = RegTokenizer(_Args(), tokens=tokens_df)
    rtok.splitters = min(rtok.splitters, int((tokens_df["reg"] == FRAME_REG).sum()))
    sub_strs = []
    for atomic_ids in sub_id_atomic_lists:
        arr = np.asarray(atomic_ids, dtype=np.uint32)
        sub_strs.append(rtok.encode_unicode(arr))
    rtok.tkmodel = _FakeTkModel(sub_strs)
    return rtok


class TestPrecomputeSubtokenArrays(unittest.TestCase):
    def test_singleton_macro_flags(self):
        tokens = _build_vocab()
        sub_atomics = [
            [BR_HI],
            [BR_LO1],
            [BR_LEN],
            [PR_HI],
            [PR_LO],
            [PR_LEN],
            [PR_OV],
            [PO_FO],
            [PO_TR],
            [PO_NV],
        ]
        rtok = _fake_regtokenizer(tokens, sub_atomics)
        arrs = precompute_subtoken_arrays(tokens, rtok)
        self.assertEqual(arrs["n_vocab"], 11)
        self.assertTrue(bool(arrs["is_pad"][0].item()))
        self.assertTrue(bool(arrs["is_singleton_back_ref_dist_hi"][1].item()))
        self.assertTrue(bool(arrs["is_singleton_back_ref_dist_lo"][2].item()))
        self.assertTrue(bool(arrs["is_singleton_back_ref_len"][3].item()))
        self.assertTrue(bool(arrs["is_singleton_pr_dist_hi"][4].item()))
        self.assertTrue(bool(arrs["is_singleton_pr_dist_lo"][5].item()))
        self.assertTrue(bool(arrs["is_singleton_pr_len"][6].item()))
        self.assertTrue(bool(arrs["is_singleton_pr_ov_count"][7].item()))

    def test_br_hi_then_lo_subtoken(self):
        tokens = _build_vocab()
        rtok = _fake_regtokenizer(tokens, [[BR_HI, BR_LO1]])
        arrs = precompute_subtoken_arrays(tokens, rtok)
        self.assertTrue(bool(arrs["is_singleton_back_ref_dist_hi"][1].item()))
        self.assertTrue(bool(arrs["extends_to_back_ref_lo_consumed"][1]))
        self.assertFalse(bool(arrs["extends_to_back_ref_len_consumed"][1]))

    def test_br_complete_subtoken(self):
        tokens = _build_vocab()
        rtok = _fake_regtokenizer(tokens, [[BR_HI, BR_LO1, BR_LEN]])
        arrs = precompute_subtoken_arrays(tokens, rtok)
        self.assertTrue(bool(arrs["is_singleton_back_ref_dist_hi"][1].item()))
        self.assertTrue(bool(arrs["extends_to_back_ref_lo_consumed"][1]))
        self.assertTrue(bool(arrs["extends_to_back_ref_len_consumed"][1]))

    def test_pr_complete_subtoken(self):
        tokens = _build_vocab()
        rtok = _fake_regtokenizer(tokens, [[PR_HI, PR_LO, PR_LEN, PR_OV]])
        arrs = precompute_subtoken_arrays(tokens, rtok)
        self.assertTrue(bool(arrs["is_singleton_pr_dist_hi"][1].item()))
        self.assertTrue(bool(arrs["extends_to_pr_lo_consumed"][1]))
        self.assertTrue(bool(arrs["extends_to_pr_len_consumed"][1]))
        self.assertTrue(bool(arrs["extends_to_pr_ov_count_consumed"][1]))
        self.assertEqual(int(arrs["overlay_count"][1].item()), 2)

    def test_malformed_macro_flag(self):
        tokens = _build_vocab()
        rtok = _fake_regtokenizer(tokens, [[SET_R0, BR_LO1]])
        arrs = precompute_subtoken_arrays(tokens, rtok)
        self.assertTrue(bool(arrs["is_malformed_macro"][1].item()))


class TestStreamStateSubtokenMode(unittest.TestCase):
    def _arrays(self, sub_atomics):
        tokens = _build_vocab()
        rtok = _fake_regtokenizer(tokens, sub_atomics)
        return precompute_subtoken_arrays(tokens, rtok)

    def test_pad_masked(self):
        arrs = self._arrays([[BR_HI], [SET_R0]])
        state = StreamState(arrs, init_frame_count=0, irq=10000)
        invalid = state._compute_invalid()  # pylint: disable=protected-access
        self.assertTrue(bool(invalid[0]))

    def test_back_ref_triple_state_machine(self):
        arrs = self._arrays([[BR_HI], [BR_LO1], [BR_LEN]])
        state = StreamState(arrs, init_frame_count=10, irq=10000)
        state.update(1)
        self.assertTrue(state.pending_back_ref_dist_lo)
        state.update(2)
        self.assertFalse(state.pending_back_ref_dist_lo)
        self.assertTrue(state.pending_back_ref_len)
        state.update(3)
        self.assertFalse(state.pending_back_ref_len)

    def test_br_hi_then_lo_advances_to_len_pending(self):
        arrs = self._arrays([[BR_HI, BR_LO1], [BR_LEN]])
        state = StreamState(arrs, init_frame_count=10, irq=10000)
        state.update(1)
        self.assertFalse(state.pending_back_ref_dist_lo)
        self.assertTrue(state.pending_back_ref_len)
        state.update(2)
        self.assertFalse(state.pending_back_ref_len)

    def test_br_complete_advances_to_idle(self):
        arrs = self._arrays([[BR_HI, BR_LO1, BR_LEN]])
        state = StreamState(arrs, init_frame_count=10, irq=10000)
        state.update(1)
        self.assertFalse(state.pending_back_ref_dist_lo)
        self.assertFalse(state.pending_back_ref_len)

    def test_pr_complete_advances_to_overlays(self):
        arrs = self._arrays([[PR_HI, PR_LO, PR_LEN, PR_OV]])
        state = StreamState(arrs, init_frame_count=10, irq=10000)
        state.update(1)
        self.assertEqual(state.pending_overlays, 2)

    def test_packed_full_distance_masked_when_out_of_range(self):
        arrs = self._arrays([[BR_HI, BR_LO5, BR_LEN]])
        self.assertEqual(int(arrs["full_distance"][1]), 5)
        state = StreamState(arrs, init_frame_count=3, irq=10000)
        mask = _mask_bool(state, arrs["n_vocab"])
        self.assertTrue(mask[1])

    def test_packed_full_distance_admitted_when_in_range(self):
        arrs = self._arrays([[BR_HI, BR_LO5, BR_LEN]])
        state = StreamState(arrs, init_frame_count=5, irq=10000)
        mask = _mask_bool(state, arrs["n_vocab"])
        self.assertFalse(mask[1])

    def test_subtoken_pending_dist_lo_caps_full_distance(self):
        arrs = self._arrays([[BR_HI], [BR_LO5], [BR_LEN]])
        state = StreamState(arrs, init_frame_count=3, irq=10000)
        state.update(1)
        self.assertTrue(state.pending_back_ref_dist_lo)
        self.assertEqual(int(arrs["dist_lo_val"][2]), 5)
        mask = _mask_bool(state, arrs["n_vocab"])
        self.assertTrue(mask[2])

    def test_subtoken_pending_dist_lo_admits_in_range(self):
        arrs = self._arrays([[BR_HI], [BR_LO1], [BR_LEN]])
        state = StreamState(arrs, init_frame_count=3, irq=10000)
        state.update(1)
        self.assertEqual(int(arrs["dist_lo_val"][2]), 1)
        mask = _mask_bool(state, arrs["n_vocab"])
        self.assertFalse(mask[2])


class TestModuleTorchFree(unittest.TestCase):
    def test_constrained_decode_imports_without_torch(self):
        """Pin the load-bearing property that `preframr_tokens.constrained_decode` has no module-level torch import. Lets a future move into a torch-free package proceed unblocked."""
        import importlib
        import sys

        sys.modules.pop("preframr_tokens.constrained_decode", None)
        torch_was_present = sys.modules.pop("torch", None)
        try:
            importlib.import_module("preframr_tokens.constrained_decode")
            self.assertNotIn("torch", sys.modules)
        finally:
            if torch_was_present is not None:
                sys.modules["torch"] = torch_was_present


if __name__ == "__main__":
    unittest.main()
