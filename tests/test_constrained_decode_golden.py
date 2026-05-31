"""Golden-master regression lock for constrained decode (RESID_ZERO_PHASE3 §4 B0/B1): freezes the CURRENT
precompute arrays, per-position mask, and validator verdicts on a corpus of streams (atomic + Unigram
sub-token, valid + corrupted) to a committed JSON and asserts the live code reproduces them byte-for-byte
-- a regression lock today, the equivalence oracle once the OpContract registry replaces the hand-written
StreamState / precompute / validators. Regenerate with PREFRAMR_REGEN_GOLDEN=1 (review the diff first).
"""

import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_tokens.constrained_decode import (
    StreamState,
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
)
from preframr_tokens.macros.validators import (
    validate_back_refs,
    validate_pattern_overlays,
    validate_stream,
)
from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    DELAY_REG,
    FRAME_REG,
    PAD_REG,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    SET_OP,
    VOICE_REG,
)

_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "constrained_decode_golden.json"

_VOCAB_ROWS = [
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

PAD, SET_R0, FRAME11, FRAME5, DELAY, VOICE = 0, 1, 2, 3, 4, 5
BR_HI, BR_LO1, BR_LO5, BR_LEN = 6, 7, 8, 9
PR_HI, PR_LO, PR_LEN, PR_OV = 10, 11, 12, 13
PO_FO, PO_TR, PO_NV = 14, 15, 16

_ATOMIC_STREAMS = {
    "free": (
        {"init_frame_count": 5, "irq": 100, "init_budget": 100},
        [SET_R0, FRAME11, SET_R0, VOICE, SET_R0],
    ),
    "backref_ok": ({"init_frame_count": 10, "irq": 100}, [BR_HI, BR_LO5, BR_LEN]),
    "preplay_overlay": (
        {"init_frame_count": 10, "irq": 100},
        [PR_HI, PR_LO, PR_LEN, PR_OV, PO_FO, PO_TR, PO_NV, PO_FO, PO_TR, PO_NV],
    ),
    "budget_exhaust": (
        {"init_frame_count": 1, "irq": 100, "init_budget": 64},
        [SET_R0, SET_R0, SET_R0],
    ),
    "remaining_steps": (
        {"init_frame_count": 10, "irq": 100, "remaining_steps": 3},
        [SET_R0],
    ),
    "delay_top": ({"init_frame_count": 5, "irq": 100}, [FRAME11]),
}

_SUB_ATOMICS = [
    [SET_R0],
    [FRAME11],
    [BR_HI, BR_LO1, BR_LEN],
    [PR_HI, PR_LO, PR_LEN, PR_OV],
    [PO_FO, PO_TR, PO_NV],
    [BR_HI],
    [BR_LO1],
    [VOICE],
]

_SUBTOKEN_STREAMS = {
    "sub_free": ({"init_frame_count": 10, "irq": 100}, [1, 2, 1]),
    "sub_macro": ({"init_frame_count": 10, "irq": 100}, [3]),
    "sub_pr_overlay": ({"init_frame_count": 10, "irq": 100}, [4, 5, 5]),
    "sub_pending": ({"init_frame_count": 10, "irq": 100}, [6, 7]),
}

_VALIDATOR_STREAMS = {
    "v_backref_ok": [
        FRAME11,
        FRAME11,
        FRAME11,
        FRAME11,
        FRAME11,
        BR_HI,
        BR_LO5,
        BR_LEN,
    ],
    "v_backref_bad_distance": [BR_HI, BR_LO5, BR_LEN],
    "v_backref_truncated": [BR_HI, BR_LO5],
    "v_overlay_ok": [
        PR_HI,
        PR_LO,
        PR_LEN,
        PR_OV,
        PO_FO,
        PO_TR,
        PO_NV,
        PO_FO,
        PO_TR,
        PO_NV,
    ],
    "v_overlay_orphan": [PO_FO],
    "v_overlay_truncated": [PR_HI, PR_LO, PR_LEN, PR_OV, PO_FO, PO_TR, PO_NV],
}


class _FakeTkModel:
    def __init__(self, sub_strings):
        self._strs = ["<unk>"] + list(sub_strings)

    def get_vocab_size(self):
        return len(self._strs)

    def id_to_token(self, sub_id):
        if sub_id < 0 or sub_id >= len(self._strs):
            return None
        return self._strs[sub_id]


def _vocab_df():
    return pd.DataFrame(_VOCAB_ROWS)


def _fake_regtokenizer(tokens_df, sub_id_atomic_lists):
    class _Args:
        tkvocab = 0
        tkmodel = None
        tokenizer = "unigram"

    rtok = RegTokenizer(_Args(), tokens=tokens_df)
    rtok.splitters = min(rtok.splitters, int((tokens_df["reg"] == FRAME_REG).sum()))
    sub_strs = [
        rtok.encode_unicode(np.asarray(a, dtype=np.uint32)) for a in sub_id_atomic_lists
    ]
    rtok.tkmodel = _FakeTkModel(sub_strs)
    return rtok


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _arrays_to_json(arrays):
    return {key: _jsonable(arrays[key]) for key in sorted(arrays)}


def _mask_trace(arrays, init, token_ids):
    state = StreamState(arrays, **init)
    trace = []
    for tok in token_ids:
        trace.append([int(b) for b in state.compute_invalid_mask().tolist()])
        state.update(int(tok))
    return trace


def _verdict(fn, df):
    try:
        fn(df)
        return False
    except AssertionError:
        return True


def _compute_golden():
    vocab = _vocab_df()
    atomic_arrays = precompute_vocab_arrays(vocab)
    rtok = _fake_regtokenizer(vocab, _SUB_ATOMICS)
    subtoken_arrays = precompute_subtoken_arrays(vocab, rtok)
    golden = {
        "precompute_vocab_arrays": _arrays_to_json(atomic_arrays),
        "precompute_subtoken_arrays": _arrays_to_json(subtoken_arrays),
        "atomic_masks": {
            name: _mask_trace(atomic_arrays, init, ids)
            for name, (init, ids) in _ATOMIC_STREAMS.items()
        },
        "subtoken_masks": {
            name: _mask_trace(subtoken_arrays, init, ids)
            for name, (init, ids) in _SUBTOKEN_STREAMS.items()
        },
        "validator_verdicts": {
            name: {
                "back_refs_reject": _verdict(
                    validate_back_refs, vocab.iloc[ids].reset_index(drop=True)
                ),
                "pattern_overlays_reject": _verdict(
                    validate_pattern_overlays, vocab.iloc[ids].reset_index(drop=True)
                ),
                "stream_reject": _verdict(
                    validate_stream, vocab.iloc[ids].reset_index(drop=True)
                ),
            }
            for name, ids in _VALIDATOR_STREAMS.items()
        },
    }
    return golden


def _maybe_regen(golden):
    if os.environ.get("PREFRAMR_REGEN_GOLDEN") != "1":
        return
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


class TestMaskDecodeForwardInvariant(unittest.TestCase):
    def test_mask_never_forbids_structurally_valid_continuation(self):
        vocab = _vocab_df()
        arrays = precompute_vocab_arrays(vocab)
        for name in ("free", "backref_ok", "preplay_overlay"):
            init, ids = _ATOMIC_STREAMS[name]
            kwargs = dict(init)
            kwargs["disable_resource_masks"] = True
            state = StreamState(arrays, **kwargs)
            for tok in ids:
                mask = state.compute_invalid_mask()
                self.assertFalse(
                    bool(mask[tok]),
                    f"{name}: mask forbids decoder-valid token {tok}",
                )
                state.update(int(tok))


class TestConstrainedDecodeGolden(unittest.TestCase):
    def test_current_code_matches_frozen_golden(self):
        golden = _compute_golden()
        _maybe_regen(golden)
        self.assertTrue(
            _GOLDEN_PATH.exists(),
            f"missing golden fixture; regenerate with PREFRAMR_REGEN_GOLDEN=1 ({_GOLDEN_PATH})",
        )
        frozen = json.loads(_GOLDEN_PATH.read_text())
        self.assertEqual(set(golden), set(frozen), "golden top-level sections drifted")
        for section in sorted(golden):
            self.assertEqual(
                golden[section], frozen[section], f"golden section '{section}' drifted"
            )


if __name__ == "__main__":
    unittest.main()
