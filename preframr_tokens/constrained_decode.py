"""Sampling-time logit guard for predict.py."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from preframr_tokens.stfconstants import (
    BACK_REF_DIST_HI_SHIFT,
    BACK_REF_DIST_LO_MASK,
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    DELAY_REG,
    FRAME_REG,
    MAX_REG,
    _MIN_DIFF,
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
from preframr_tokens.macros.op_contracts import (
    STRUCTURAL_SUBREGS,
    STRUCTURAL_VALUE_ARRAYS,
)
from preframr_tokens.utils import to_int64_arrays

__all__ = [
    "frame_marker_count",
    "tail_charge_for_prompt",
    "precompute_vocab_arrays",
    "precompute_subtoken_arrays",
    "PendingSlot",
    "StreamState",
    "VocabArrays",
]


class OverlaySlot(IntEnum):
    """Which of the 3 atomic slots a pattern-overlay row fills, indexed in emission order. Values match the corresponding ``PATTERN_OVERLAY_SUBREG_*`` constants so comparisons against raw subreg ints remain valid."""

    FRAME_OFFSET = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
    TARGET_REG = PATTERN_OVERLAY_SUBREG_TARGET_REG
    NEW_VAL = PATTERN_OVERLAY_SUBREG_NEW_VAL


class MacroShape(IntEnum):
    """Identifies the structural shape of a Unigram sub-token's atomic-id decomposition. Used by ``precompute_subtoken_arrays`` to drive a downstream per-shape flag-setting switch. ``MALFORMED`` covers any sub-token whose macro structure doesn't match one of the known shapes."""

    NONE = 0
    MALFORMED = 1
    SINGLETON_BACK_REF_DIST_HI = 2
    SINGLETON_BACK_REF_DIST_LO = 3
    SINGLETON_BACK_REF_LEN = 4
    SINGLETON_PR_DIST_HI = 5
    SINGLETON_PR_DIST_LO = 6
    SINGLETON_PR_LEN = 7
    SINGLETON_PR_OV_COUNT = 8
    SINGLETON_OVERLAY_FRAME_OFFSET = 9
    SINGLETON_OVERLAY_TARGET_REG = 10
    SINGLETON_OVERLAY_NEW_VAL = 11
    BR_HI_THEN_LO = 12
    BR_COMPLETE = 13
    BR_LO_THEN_LEN = 14
    PR_HI_THEN_LO = 15
    PR_HI_THROUGH_LEN = 16
    PR_COMPLETE = 17
    PR_LO_THEN_LEN = 18
    PR_LO_THROUGH_OV_COUNT = 19
    BR_LEN_WITH_TAIL = 20
    PR_LEN_THEN_OV_COUNT = 21
    OV_TARGET_THEN_NEW_VAL = 22


_SHAPES_WITH_DIST_LO_FIRST = frozenset(
    {
        MacroShape.SINGLETON_BACK_REF_DIST_LO,
        MacroShape.SINGLETON_PR_DIST_LO,
        MacroShape.BR_LO_THEN_LEN,
        MacroShape.PR_LO_THEN_LEN,
        MacroShape.PR_LO_THROUGH_OV_COUNT,
    }
)

_SHAPES_WITH_DIST_LO_SECOND = frozenset(
    {
        MacroShape.BR_HI_THEN_LO,
        MacroShape.BR_COMPLETE,
        MacroShape.PR_HI_THEN_LO,
        MacroShape.PR_HI_THROUGH_LEN,
        MacroShape.PR_COMPLETE,
    }
)


@dataclass(frozen=True)
class _FrameAggregates:
    """Per-sub-token aggregates over the atomic decomposition. Computed once per sub-token by ``_walk_frame_aggregates``."""

    frame_advance: int
    charge_first_segment: int
    charge_last_segment: int
    sets_sval: bool
    final_sval: int
    fn_delta: int
    fn_after_last_strict: int
    contains_delay: bool


def _walk_frame_aggregates(
    atomic_ids,
    *,
    val_a,
    is_macro_a,
    is_frame_marker_a,
    is_delay_a,
    is_frame_strict_a,
    is_voice_reg_a,
    is_real_reg_a,
):
    """Walk ``atomic_ids`` and aggregate per-sub-token frame-time state. Pure function over the per-atom bool/int arrays. Used by ``precompute_subtoken_arrays``; exposed for direct unit tests."""
    local_frame = 0
    first_seg_charge = 0
    last_seg_charge = 0
    first_seg_done = False
    local_sets_sval = False
    local_final_sval = 0
    local_fn_delta = 0
    local_fn_after_strict = 0
    local_contains_delay = False
    for aid in atomic_ids:
        aid = int(aid)
        if is_macro_a[aid]:
            continue
        if is_frame_marker_a[aid]:
            if not first_seg_done:
                first_seg_charge = last_seg_charge
                first_seg_done = True
            local_frame += 1
            last_seg_charge = 0
            if is_delay_a[aid]:
                local_contains_delay = True
            if is_frame_strict_a[aid]:
                local_sets_sval = True
                local_final_sval = int(val_a[aid]) & 0x3F
                local_fn_after_strict = 0
            continue
        if is_voice_reg_a[aid]:
            if local_sets_sval:
                local_fn_after_strict += 1
            else:
                local_fn_delta += 1
            continue
        if is_real_reg_a[aid]:
            last_seg_charge += _MIN_DIFF
    if not first_seg_done:
        first_seg_charge = last_seg_charge
    return _FrameAggregates(
        frame_advance=local_frame,
        charge_first_segment=first_seg_charge,
        charge_last_segment=last_seg_charge,
        sets_sval=local_sets_sval,
        final_sval=local_final_sval,
        fn_delta=local_fn_delta,
        fn_after_last_strict=local_fn_after_strict,
        contains_delay=local_contains_delay,
    )


@dataclass(frozen=True)
class _ShapeRule:
    """One match-rule for ``_classify_macro_shape``. Matched when the head atom's ``(op, subreg)`` keys into ``_HEAD_RULES`` and the trailing atoms (atoms 1..n-1) have ``(op, subreg)`` equal to ``trailing[k]``. ``val_indices`` lists which ``atomic_ids`` positions contribute their ``val_a`` value to the result tuple as extras (after the shape tag)."""

    trailing: tuple[tuple[int, int], ...]
    shape: "MacroShape"
    val_indices: tuple[int, ...] = ()


_BR_LO = (BACK_REF_OP, BACK_REF_SUBREG_DIST_LO)
_BR_LEN = (BACK_REF_OP, BACK_REF_SUBREG_LEN)
_PR_LO = (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_LO)
_PR_LEN = (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_LEN)
_PR_OVC = (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_OVERLAY_COUNT)
_OV_NEW = (PATTERN_OVERLAY_OP, PATTERN_OVERLAY_SUBREG_NEW_VAL)


_HEAD_RULES: dict[tuple[int, int], tuple[_ShapeRule, ...]] = {
    (BACK_REF_OP, BACK_REF_SUBREG_DIST_HI): (
        _ShapeRule((), MacroShape.SINGLETON_BACK_REF_DIST_HI, (0,)),
        _ShapeRule((_BR_LO,), MacroShape.BR_HI_THEN_LO, (0,)),
        _ShapeRule((_BR_LO, _BR_LEN), MacroShape.BR_COMPLETE, (0,)),
    ),
    (BACK_REF_OP, BACK_REF_SUBREG_DIST_LO): (
        _ShapeRule((), MacroShape.SINGLETON_BACK_REF_DIST_LO, (0,)),
        _ShapeRule((_BR_LEN,), MacroShape.BR_LO_THEN_LEN),
    ),
    (BACK_REF_OP, BACK_REF_SUBREG_LEN): (
        _ShapeRule((), MacroShape.SINGLETON_BACK_REF_LEN, (0,)),
    ),
    (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI): (
        _ShapeRule((), MacroShape.SINGLETON_PR_DIST_HI, (0,)),
        _ShapeRule((_PR_LO,), MacroShape.PR_HI_THEN_LO, (0,)),
        _ShapeRule((_PR_LO, _PR_LEN), MacroShape.PR_HI_THROUGH_LEN, (0,)),
        _ShapeRule((_PR_LO, _PR_LEN, _PR_OVC), MacroShape.PR_COMPLETE, (0, 3)),
    ),
    (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_LO): (
        _ShapeRule((), MacroShape.SINGLETON_PR_DIST_LO, (0,)),
        _ShapeRule((_PR_LEN,), MacroShape.PR_LO_THEN_LEN),
        _ShapeRule((_PR_LEN, _PR_OVC), MacroShape.PR_LO_THROUGH_OV_COUNT, (2,)),
    ),
    (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_LEN): (
        _ShapeRule((), MacroShape.SINGLETON_PR_LEN, (0,)),
        _ShapeRule((_PR_OVC,), MacroShape.PR_LEN_THEN_OV_COUNT, (1,)),
    ),
    (PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_OVERLAY_COUNT): (
        _ShapeRule((), MacroShape.SINGLETON_PR_OV_COUNT, (0,)),
    ),
    (PATTERN_OVERLAY_OP, PATTERN_OVERLAY_SUBREG_FRAME_OFFSET): (
        _ShapeRule((), MacroShape.SINGLETON_OVERLAY_FRAME_OFFSET, (0,)),
    ),
    (PATTERN_OVERLAY_OP, PATTERN_OVERLAY_SUBREG_TARGET_REG): (
        _ShapeRule((), MacroShape.SINGLETON_OVERLAY_TARGET_REG, (0,)),
        _ShapeRule((_OV_NEW,), MacroShape.OV_TARGET_THEN_NEW_VAL),
    ),
    (PATTERN_OVERLAY_OP, PATTERN_OVERLAY_SUBREG_NEW_VAL): (
        _ShapeRule((), MacroShape.SINGLETON_OVERLAY_NEW_VAL, (0,)),
    ),
}


_SHAPE_HANDLERS: dict[
    "MacroShape", tuple[tuple[str, ...], tuple[tuple[str, int], ...]]
] = {
    MacroShape.MALFORMED: (("is_malformed_macro",), ()),
    MacroShape.SINGLETON_BACK_REF_DIST_HI: (
        ("is_singleton_back_ref_dist_hi",),
        (("distance_hi", 1),),
    ),
    MacroShape.SINGLETON_BACK_REF_DIST_LO: (
        ("is_singleton_back_ref_dist_lo", "consumes_back_ref_dist_lo_gate"),
        (),
    ),
    MacroShape.SINGLETON_BACK_REF_LEN: (
        ("is_singleton_back_ref_len", "consumes_back_ref_len_gate"),
        (),
    ),
    MacroShape.SINGLETON_PR_DIST_HI: (
        ("is_singleton_pr_dist_hi",),
        (("distance_hi", 1),),
    ),
    MacroShape.SINGLETON_PR_DIST_LO: (
        ("is_singleton_pr_dist_lo", "consumes_pr_dist_lo_gate"),
        (),
    ),
    MacroShape.SINGLETON_PR_LEN: (
        ("is_singleton_pr_len", "consumes_pr_len_gate"),
        (),
    ),
    MacroShape.SINGLETON_PR_OV_COUNT: (
        ("is_singleton_pr_ov_count", "consumes_pr_ov_count_gate"),
        (("overlay_count", 1), ("pending_overlays_delta", 1)),
    ),
    MacroShape.SINGLETON_OVERLAY_FRAME_OFFSET: (
        (
            "is_singleton_pattern_overlay",
            "is_singleton_pattern_overlay_frame_offset",
            "consumes_overlay_slot_0_gate",
        ),
        (),
    ),
    MacroShape.SINGLETON_OVERLAY_TARGET_REG: (
        (
            "is_singleton_pattern_overlay",
            "is_singleton_pattern_overlay_target_reg",
            "consumes_overlay_slot_1_gate",
        ),
        (),
    ),
    MacroShape.SINGLETON_OVERLAY_NEW_VAL: (
        (
            "is_singleton_pattern_overlay",
            "is_singleton_pattern_overlay_new_val",
            "consumes_overlay_slot_2_gate",
        ),
        (),
    ),
    MacroShape.BR_HI_THEN_LO: (
        ("is_singleton_back_ref_dist_hi", "extends_to_back_ref_lo_consumed"),
        (("distance_hi", 1),),
    ),
    MacroShape.BR_COMPLETE: (
        (
            "is_singleton_back_ref_dist_hi",
            "extends_to_back_ref_lo_consumed",
            "extends_to_back_ref_len_consumed",
        ),
        (("distance_hi", 1),),
    ),
    MacroShape.BR_LO_THEN_LEN: (
        ("consumes_back_ref_dist_lo_gate", "extends_to_back_ref_len_consumed"),
        (),
    ),
    MacroShape.PR_HI_THEN_LO: (
        ("is_singleton_pr_dist_hi", "extends_to_pr_lo_consumed"),
        (("distance_hi", 1),),
    ),
    MacroShape.PR_HI_THROUGH_LEN: (
        (
            "is_singleton_pr_dist_hi",
            "extends_to_pr_lo_consumed",
            "extends_to_pr_len_consumed",
        ),
        (("distance_hi", 1),),
    ),
    MacroShape.PR_COMPLETE: (
        (
            "is_singleton_pr_dist_hi",
            "extends_to_pr_lo_consumed",
            "extends_to_pr_len_consumed",
            "extends_to_pr_ov_count_consumed",
        ),
        (("distance_hi", 1), ("overlay_count", 2), ("pending_overlays_delta", 2)),
    ),
    MacroShape.PR_LO_THEN_LEN: (
        ("consumes_pr_dist_lo_gate", "extends_to_pr_len_consumed"),
        (),
    ),
    MacroShape.PR_LO_THROUGH_OV_COUNT: (
        (
            "consumes_pr_dist_lo_gate",
            "extends_to_pr_len_consumed",
            "extends_to_pr_ov_count_consumed",
        ),
        (("overlay_count", 1), ("pending_overlays_delta", 1)),
    ),
    MacroShape.BR_LEN_WITH_TAIL: (("consumes_back_ref_len_gate",), ()),
    MacroShape.PR_LEN_THEN_OV_COUNT: (
        ("consumes_pr_len_gate", "extends_to_pr_ov_count_consumed"),
        (("overlay_count", 1), ("pending_overlays_delta", 1)),
    ),
    MacroShape.OV_TARGET_THEN_NEW_VAL: (
        ("consumes_overlay_slot_1_gate", "extends_to_overlay_completed"),
        (),
    ),
}


def _classify_macro_shape(atomic_ids, op_a, subreg_a, val_a, is_macro_a):
    """Classify a sub-token's macro shape against ``_HEAD_RULES``; returns ``(MacroShape, *extras)``. ``BR_LEN_WITH_TAIL`` is the lone irregular shape, handled out-of-table."""
    n = atomic_ids.size
    if n == 0:
        return (MacroShape.NONE,)
    first_macro_idx = -1
    for k in range(n):
        if is_macro_a[int(atomic_ids[k])]:
            first_macro_idx = k
            break
    if first_macro_idx == -1:
        return (MacroShape.NONE,)
    if first_macro_idx > 0:
        return (MacroShape.MALFORMED,)
    first = int(atomic_ids[0])
    head = (int(op_a[first]), int(subreg_a[first]))
    if head == (BACK_REF_OP, BACK_REF_SUBREG_LEN) and n >= 2:
        for k in range(1, n):
            if is_macro_a[int(atomic_ids[k])]:
                return (MacroShape.MALFORMED,)
        return (MacroShape.BR_LEN_WITH_TAIL, int(val_a[first]))
    rules = _HEAD_RULES.get(head)
    if rules is None:
        return (MacroShape.MALFORMED,)
    target_trailing_len = n - 1
    for rule in rules:
        if len(rule.trailing) != target_trailing_len:
            continue
        matched = True
        for k, (want_op, want_sr) in enumerate(rule.trailing):
            atom = int(atomic_ids[k + 1])
            if int(op_a[atom]) != want_op or int(subreg_a[atom]) != want_sr:
                matched = False
                break
        if matched:
            extras = tuple(int(val_a[int(atomic_ids[i])]) for i in rule.val_indices)
            return (rule.shape, *extras)
    return (MacroShape.MALFORMED,)


class VocabArrays(dict):
    """Per-vocab-id arrays bundle returned by ``precompute_vocab_arrays`` and ``precompute_subtoken_arrays``. Subclasses ``dict`` so external consumers can keep indexing by string key (``a["is_real_reg"]``); also supports attribute access (``a.is_real_reg``) for in-module readability. Shape is documented in the precompute functions."""

    __slots__ = ()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def frame_marker_count(token_ids, is_frame_marker):
    """Number of frame-marker tokens in ``token_ids`` (a 1-D iterable of
    vocab indices)."""
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size == 0:
        return 0
    return int(is_frame_marker[arr].sum().item())


def tail_charge_for_prompt(prompt_ids, vocab_arrays) -> int:
    """Cycles consumed by real-reg writes between the last frame marker in ``prompt_ids`` and prompt end. Used by predictors to seed the in-frame ``frame_budget`` when resuming sampling so the first generated token doesn't double-charge. Returns 0 if the prompt has no frame markers (treat as fresh frame)."""
    arr = np.asarray(prompt_ids, dtype=np.int64)
    if arr.size == 0:
        return 0
    is_frame_marker = vocab_arrays["is_frame_marker"]
    is_real_reg = vocab_arrays["is_real_reg"]
    marker_positions = np.nonzero(is_frame_marker[arr])[0]
    if marker_positions.size == 0:
        return 0
    tail = arr[int(marker_positions[-1]) + 1 :]
    return int(is_real_reg[tail].sum() * _MIN_DIFF)


def _structural_arrays(op, subreg, val, n):
    """Build the per-vocab BACK_REF / PATTERN_REPLAY / PATTERN_OVERLAY classification + value arrays by
    iterating ``STRUCTURAL_SUBREGS`` -- one source of truth for which (op, subreg) sets which flag and
    scatters its ``val``, replacing the hand-listed ``op == BACK_REF_OP & subreg == ...`` chain.
    """
    flags = {
        sf.flag: np.zeros(n, dtype=np.bool_)
        for specs in STRUCTURAL_SUBREGS.values()
        for sf in specs
    }
    values = {name: np.zeros(n, dtype=np.int64) for name in STRUCTURAL_VALUE_ARRAYS}
    for op_code, specs in STRUCTURAL_SUBREGS.items():
        op_match = op == op_code
        for sf in specs:
            mask = op_match & (subreg == sf.subreg)
            flags[sf.flag] = mask
            if sf.value_array is not None:
                values[sf.value_array][mask] = val[mask]
    return flags, values


def precompute_vocab_arrays(tokens_df):
    """Per-vocab-id numpy arrays for the per-step mask. Sized by the atomic alphabet -- correct when the model emits atomic ids (``tkvocab=0``). For Unigram (``tkvocab > 0``) the model emits sub-token ids and ``StreamState`` would index out of bounds; use ``precompute_subtoken_arrays`` instead."""
    n = len(tokens_df)
    op, reg, subreg, val = to_int64_arrays(
        tokens_df,
        "op",
        "reg",
        "subreg",
        "val",
        fillna={"op": SET_OP, "subreg": -1},
    )

    is_frame_marker = np.isin(reg, [FRAME_REG, DELAY_REG])
    is_frame_reg_strict = reg == FRAME_REG
    is_voice_reg = reg == VOICE_REG
    is_delay_reg = reg == DELAY_REG
    is_pad = reg == PAD_REG
    is_real_reg = (reg >= 0) & (reg <= MAX_REG)
    is_slope_term_hi = np.zeros(n, dtype=np.bool_)
    is_slope_term_lo = np.zeros(n, dtype=np.bool_)
    is_slope_runtime = np.zeros(n, dtype=np.bool_)
    flags, values = _structural_arrays(op, subreg, val, n)
    is_back_ref_dist_hi = flags["is_back_ref_dist_hi"]
    is_back_ref_dist_lo = flags["is_back_ref_dist_lo"]
    is_back_ref_len = flags["is_back_ref_len"]
    is_pattern_replay_dist_hi = flags["is_pattern_replay_dist_hi"]
    is_pattern_replay_dist_lo = flags["is_pattern_replay_dist_lo"]
    is_pattern_replay_len = flags["is_pattern_replay_len"]
    is_pattern_replay_ov_count = flags["is_pattern_replay_ov_count"]
    is_dist_hi_row = is_back_ref_dist_hi | is_pattern_replay_dist_hi
    is_dist_lo_row = is_back_ref_dist_lo | is_pattern_replay_dist_lo
    is_pair_intermediate = (
        is_back_ref_dist_lo
        | is_back_ref_len
        | is_pattern_replay_dist_lo
        | is_pattern_replay_len
        | is_pattern_replay_ov_count
        | is_slope_term_lo
        | is_slope_runtime
    )
    is_pattern_overlay = op == PATTERN_OVERLAY_OP
    is_pattern_overlay_frame_offset = flags["is_pattern_overlay_frame_offset"]
    is_pattern_overlay_target_reg = flags["is_pattern_overlay_target_reg"]
    is_pattern_overlay_new_val = flags["is_pattern_overlay_new_val"]
    dist_hi_val = values["dist_hi_val"]
    dist_lo_val = values["dist_lo_val"]
    length = values["length"]
    overlay_count = values["overlay_count"]

    frame_sval = np.zeros(n, dtype=np.int64)
    frame_sval[is_frame_reg_strict] = val[is_frame_reg_strict] & 0x3F

    return VocabArrays(
        {
            "n_vocab": n,
            "subtoken_mode": False,
            "is_frame_marker": is_frame_marker,
            "is_delay_reg": is_delay_reg.astype(np.bool_),
            "is_pad": is_pad.astype(np.bool_),
            "is_real_reg": is_real_reg.astype(np.bool_),
            "is_back_ref_dist_hi": is_back_ref_dist_hi.astype(np.bool_),
            "is_back_ref_dist_lo": is_back_ref_dist_lo.astype(np.bool_),
            "is_back_ref_len": is_back_ref_len.astype(np.bool_),
            "is_slope_term_hi": is_slope_term_hi.astype(np.bool_),
            "is_slope_term_lo": is_slope_term_lo.astype(np.bool_),
            "is_slope_runtime": is_slope_runtime.astype(np.bool_),
            "is_pattern_replay_dist_hi": is_pattern_replay_dist_hi.astype(np.bool_),
            "is_pattern_replay_dist_lo": is_pattern_replay_dist_lo.astype(np.bool_),
            "is_pattern_replay_len": is_pattern_replay_len.astype(np.bool_),
            "is_pattern_replay_ov_count": is_pattern_replay_ov_count.astype(np.bool_),
            "is_dist_hi_row": is_dist_hi_row.astype(np.bool_),
            "is_dist_lo_row": is_dist_lo_row.astype(np.bool_),
            "is_pair_intermediate": is_pair_intermediate.astype(np.bool_),
            "is_pattern_overlay": is_pattern_overlay.astype(np.bool_),
            "is_pattern_overlay_frame_offset": is_pattern_overlay_frame_offset.astype(
                np.bool_
            ),
            "is_pattern_overlay_target_reg": is_pattern_overlay_target_reg.astype(
                np.bool_
            ),
            "is_pattern_overlay_new_val": is_pattern_overlay_new_val.astype(np.bool_),
            "is_frame_reg_strict": is_frame_reg_strict.astype(np.bool_),
            "is_voice_reg": is_voice_reg.astype(np.bool_),
            "frame_sval": frame_sval,
            "dist_hi_val": dist_hi_val,
            "dist_lo_val": dist_lo_val,
            "length": length,
            "overlay_count": overlay_count,
        }
    )


def precompute_subtoken_arrays(tokens_df, regtokenizer, pad_id=0):
    """Per-sub-token numpy arrays for the per-step mask under Unigram."""
    tkmodel = regtokenizer.tkmodel
    if tkmodel is None:
        raise ValueError("precompute_subtoken_arrays requires a trained tkmodel")
    n_sub = tkmodel.get_vocab_size()
    n_atomic = len(tokens_df)
    op_a, reg_a, subreg_a, val_a = to_int64_arrays(
        tokens_df,
        "op",
        "reg",
        "subreg",
        "val",
        fillna={"op": SET_OP, "subreg": -1},
    )
    is_frame_marker_a = np.isin(reg_a, [FRAME_REG, DELAY_REG])
    is_frame_strict_a = reg_a == FRAME_REG
    is_voice_reg_a = reg_a == VOICE_REG
    is_delay_a = reg_a == DELAY_REG
    is_real_reg_a = (reg_a >= 0) & (reg_a <= MAX_REG)
    is_back_ref_a = op_a == BACK_REF_OP
    is_pr_a = op_a == PATTERN_REPLAY_OP
    is_overlay_a = op_a == PATTERN_OVERLAY_OP
    is_macro_a = is_back_ref_a | is_pr_a | is_overlay_a

    is_pad = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_back_ref_dist_hi = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_back_ref_dist_lo = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_back_ref_len = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pr_dist_hi = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pr_dist_lo = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pr_len = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pr_ov_count = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pattern_overlay = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pattern_overlay_frame_offset = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pattern_overlay_target_reg = np.zeros(n_sub, dtype=np.bool_)
    is_singleton_pattern_overlay_new_val = np.zeros(n_sub, dtype=np.bool_)
    consumes_back_ref_dist_lo_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_back_ref_len_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_pr_dist_lo_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_pr_len_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_pr_ov_count_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_overlay_slot_0_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_overlay_slot_1_gate = np.zeros(n_sub, dtype=np.bool_)
    consumes_overlay_slot_2_gate = np.zeros(n_sub, dtype=np.bool_)
    extends_to_back_ref_lo_consumed = np.zeros(n_sub, dtype=np.bool_)
    extends_to_back_ref_len_consumed = np.zeros(n_sub, dtype=np.bool_)
    extends_to_pr_lo_consumed = np.zeros(n_sub, dtype=np.bool_)
    extends_to_pr_len_consumed = np.zeros(n_sub, dtype=np.bool_)
    extends_to_pr_ov_count_consumed = np.zeros(n_sub, dtype=np.bool_)
    extends_to_overlay_completed = np.zeros(n_sub, dtype=np.bool_)
    is_malformed_macro = np.zeros(n_sub, dtype=np.bool_)
    contains_delay = np.zeros(n_sub, dtype=np.bool_)
    distance_hi = np.zeros(n_sub, dtype=np.int64)
    dist_lo_val = np.zeros(n_sub, dtype=np.int64)
    full_distance = np.zeros(n_sub, dtype=np.int64)
    overlay_count = np.zeros(n_sub, dtype=np.int64)
    frame_advance = np.zeros(n_sub, dtype=np.int64)
    charge_first_segment = np.zeros(n_sub, dtype=np.int64)
    charge_last_segment = np.zeros(n_sub, dtype=np.int64)
    sets_sval = np.zeros(n_sub, dtype=np.bool_)
    final_sval = np.zeros(n_sub, dtype=np.int64)
    fn_delta = np.zeros(n_sub, dtype=np.int64)
    fn_after_last_strict = np.zeros(n_sub, dtype=np.int64)
    pending_overlays_delta = np.zeros(n_sub, dtype=np.int64)

    arrays_by_name = {
        "is_malformed_macro": is_malformed_macro,
        "is_singleton_back_ref_dist_hi": is_singleton_back_ref_dist_hi,
        "is_singleton_back_ref_dist_lo": is_singleton_back_ref_dist_lo,
        "is_singleton_back_ref_len": is_singleton_back_ref_len,
        "is_singleton_pr_dist_hi": is_singleton_pr_dist_hi,
        "is_singleton_pr_dist_lo": is_singleton_pr_dist_lo,
        "is_singleton_pr_len": is_singleton_pr_len,
        "is_singleton_pr_ov_count": is_singleton_pr_ov_count,
        "is_singleton_pattern_overlay": is_singleton_pattern_overlay,
        "is_singleton_pattern_overlay_frame_offset": is_singleton_pattern_overlay_frame_offset,
        "is_singleton_pattern_overlay_target_reg": is_singleton_pattern_overlay_target_reg,
        "is_singleton_pattern_overlay_new_val": is_singleton_pattern_overlay_new_val,
        "consumes_back_ref_dist_lo_gate": consumes_back_ref_dist_lo_gate,
        "consumes_back_ref_len_gate": consumes_back_ref_len_gate,
        "consumes_pr_dist_lo_gate": consumes_pr_dist_lo_gate,
        "consumes_pr_len_gate": consumes_pr_len_gate,
        "consumes_pr_ov_count_gate": consumes_pr_ov_count_gate,
        "consumes_overlay_slot_0_gate": consumes_overlay_slot_0_gate,
        "consumes_overlay_slot_1_gate": consumes_overlay_slot_1_gate,
        "consumes_overlay_slot_2_gate": consumes_overlay_slot_2_gate,
        "extends_to_back_ref_lo_consumed": extends_to_back_ref_lo_consumed,
        "extends_to_back_ref_len_consumed": extends_to_back_ref_len_consumed,
        "extends_to_pr_lo_consumed": extends_to_pr_lo_consumed,
        "extends_to_pr_len_consumed": extends_to_pr_len_consumed,
        "extends_to_pr_ov_count_consumed": extends_to_pr_ov_count_consumed,
        "extends_to_overlay_completed": extends_to_overlay_completed,
        "distance_hi": distance_hi,
        "overlay_count": overlay_count,
        "pending_overlays_delta": pending_overlays_delta,
    }

    for sub_id in range(n_sub):
        s = tkmodel.id_to_token(sub_id)
        if s is None:
            continue
        if s.startswith("<") and s.endswith(">"):
            if sub_id == pad_id:
                is_pad[sub_id] = True
            continue
        atomic_ids = regtokenizer.decode_unicode(s, dtype=np.int64)
        if atomic_ids.size == 0:
            continue
        atomic_ids = atomic_ids[(atomic_ids >= 0) & (atomic_ids < n_atomic)]
        if atomic_ids.size == 0:
            continue
        shape = _classify_macro_shape(atomic_ids, op_a, subreg_a, val_a, is_macro_a)
        tag = shape[0]
        handler = _SHAPE_HANDLERS.get(tag)
        if handler is not None:
            bool_flags, int_assigns = handler
            for name in bool_flags:
                arrays_by_name[name][sub_id] = True
            for name, idx in int_assigns:
                arrays_by_name[name][sub_id] = shape[idx]

        if tag in _SHAPES_WITH_DIST_LO_FIRST:
            dist_lo_val[sub_id] = int(val_a[int(atomic_ids[0])]) & BACK_REF_DIST_LO_MASK
        if tag in _SHAPES_WITH_DIST_LO_SECOND:
            lo_byte = int(val_a[int(atomic_ids[1])]) & BACK_REF_DIST_LO_MASK
            full_distance[sub_id] = (
                int(distance_hi[sub_id]) << BACK_REF_DIST_HI_SHIFT
            ) | lo_byte

        agg = _walk_frame_aggregates(
            atomic_ids,
            val_a=val_a,
            is_macro_a=is_macro_a,
            is_frame_marker_a=is_frame_marker_a,
            is_delay_a=is_delay_a,
            is_frame_strict_a=is_frame_strict_a,
            is_voice_reg_a=is_voice_reg_a,
            is_real_reg_a=is_real_reg_a,
        )
        frame_advance[sub_id] = agg.frame_advance
        charge_first_segment[sub_id] = agg.charge_first_segment
        charge_last_segment[sub_id] = agg.charge_last_segment
        sets_sval[sub_id] = agg.sets_sval
        final_sval[sub_id] = agg.final_sval
        fn_delta[sub_id] = agg.fn_delta
        fn_after_last_strict[sub_id] = agg.fn_after_last_strict
        if agg.contains_delay:
            contains_delay[sub_id] = True

    is_singleton_dist_hi = is_singleton_back_ref_dist_hi | is_singleton_pr_dist_hi
    is_singleton_pair_intermediate = (
        is_singleton_back_ref_dist_lo
        | is_singleton_back_ref_len
        | is_singleton_pr_dist_lo
        | is_singleton_pr_len
        | is_singleton_pr_ov_count
    )

    return VocabArrays(
        {
            "n_vocab": n_sub,
            "subtoken_mode": True,
            "is_frame_marker": (frame_advance > 0),
            "is_pad": is_pad,
            "is_singleton_back_ref_dist_hi": is_singleton_back_ref_dist_hi,
            "is_singleton_back_ref_dist_lo": is_singleton_back_ref_dist_lo,
            "is_singleton_back_ref_len": is_singleton_back_ref_len,
            "is_singleton_pr_dist_hi": is_singleton_pr_dist_hi,
            "is_singleton_pr_dist_lo": is_singleton_pr_dist_lo,
            "is_singleton_pr_len": is_singleton_pr_len,
            "is_singleton_pr_ov_count": is_singleton_pr_ov_count,
            "is_singleton_dist_hi": is_singleton_dist_hi,
            "is_singleton_pair_intermediate": is_singleton_pair_intermediate,
            "is_singleton_pattern_overlay": is_singleton_pattern_overlay,
            "is_singleton_pattern_overlay_frame_offset": (
                is_singleton_pattern_overlay_frame_offset
            ),
            "is_singleton_pattern_overlay_target_reg": (
                is_singleton_pattern_overlay_target_reg
            ),
            "is_singleton_pattern_overlay_new_val": is_singleton_pattern_overlay_new_val,
            "is_malformed_macro": is_malformed_macro,
            "consumes_back_ref_len_gate": consumes_back_ref_len_gate,
            "consumes_pr_len_gate": consumes_pr_len_gate,
            "consumes_back_ref_dist_lo_gate": consumes_back_ref_dist_lo_gate,
            "consumes_pr_dist_lo_gate": consumes_pr_dist_lo_gate,
            "consumes_pr_ov_count_gate": consumes_pr_ov_count_gate,
            "consumes_overlay_slot_0_gate": consumes_overlay_slot_0_gate,
            "consumes_overlay_slot_1_gate": consumes_overlay_slot_1_gate,
            "consumes_overlay_slot_2_gate": consumes_overlay_slot_2_gate,
            "extends_to_back_ref_lo_consumed": extends_to_back_ref_lo_consumed,
            "extends_to_back_ref_len_consumed": extends_to_back_ref_len_consumed,
            "extends_to_pr_lo_consumed": extends_to_pr_lo_consumed,
            "extends_to_pr_len_consumed": extends_to_pr_len_consumed,
            "extends_to_pr_ov_count_consumed": extends_to_pr_ov_count_consumed,
            "extends_to_overlay_completed": extends_to_overlay_completed,
            "contains_delay": contains_delay,
            "distance_hi": distance_hi,
            "dist_lo_val": dist_lo_val,
            "full_distance": full_distance,
            "overlay_count": overlay_count,
            "frame_advance": frame_advance,
            "charge_first_segment": charge_first_segment,
            "charge_last_segment": charge_last_segment,
            "sets_sval": sets_sval,
            "final_sval": final_sval,
            "fn_delta": fn_delta,
            "fn_after_last_strict": fn_after_last_strict,
            "pending_overlays_delta": pending_overlays_delta,
        }
    )


class PendingSlot(IntEnum):
    """Which structural slot the next token must fill. The 7 macro mid-walk states are mutually exclusive — the previous implementation tracked them as 7 independent booleans, which made the invariant implicit and the dispatch table long. Overlay-in-flight is tracked separately via ``pending_overlays`` because it's a counter, not a single-shot slot."""

    NONE = 0
    BACK_REF_DIST_LO = 1
    BACK_REF_LEN = 2
    PR_DIST_LO = 3
    PR_LEN = 4
    PR_OV_COUNT = 5
    SLOPE_TERM_LO = 6
    SLOPE_RUNTIME = 7


_ATOMIC_SLOT_GATE = {
    PendingSlot.BACK_REF_DIST_LO: ("is_back_ref_dist_lo", "distance"),
    PendingSlot.PR_DIST_LO: ("is_pattern_replay_dist_lo", "distance"),
    PendingSlot.BACK_REF_LEN: ("is_back_ref_len", None),
    PendingSlot.PR_LEN: ("is_pattern_replay_len", None),
    PendingSlot.PR_OV_COUNT: ("is_pattern_replay_ov_count", "overlay_cap"),
    PendingSlot.SLOPE_TERM_LO: ("is_slope_term_lo", None),
    PendingSlot.SLOPE_RUNTIME: ("is_slope_runtime", None),
}

_SUBTOKEN_SLOT_GATE = {
    PendingSlot.BACK_REF_DIST_LO: ("consumes_back_ref_dist_lo_gate", "distance"),
    PendingSlot.PR_DIST_LO: ("consumes_pr_dist_lo_gate", "distance"),
    PendingSlot.BACK_REF_LEN: ("consumes_back_ref_len_gate", None),
    PendingSlot.PR_LEN: ("consumes_pr_len_gate", None),
    PendingSlot.PR_OV_COUNT: ("consumes_pr_ov_count_gate", "overlay_cap"),
}

_ATOMIC_OVERLAY_GATES = (
    "is_pattern_overlay_frame_offset",
    "is_pattern_overlay_target_reg",
    "is_pattern_overlay_new_val",
)

_SUBTOKEN_OVERLAY_GATES = (
    "consumes_overlay_slot_0_gate",
    "consumes_overlay_slot_1_gate",
    "consumes_overlay_slot_2_gate",
)

_OVERLAY_SLOT_INDEX = {
    OverlaySlot.FRAME_OFFSET: 0,
    OverlaySlot.TARGET_REG: 1,
    OverlaySlot.NEW_VAL: 2,
}

_ATOMIC_SLOT_TRANSITION = {
    PendingSlot.BACK_REF_DIST_LO: (
        "is_back_ref_dist_lo",
        PendingSlot.BACK_REF_LEN,
        None,
    ),
    PendingSlot.PR_DIST_LO: ("is_pattern_replay_dist_lo", PendingSlot.PR_LEN, None),
    PendingSlot.BACK_REF_LEN: ("is_back_ref_len", PendingSlot.NONE, None),
    PendingSlot.PR_LEN: ("is_pattern_replay_len", PendingSlot.PR_OV_COUNT, None),
    PendingSlot.PR_OV_COUNT: (
        "is_pattern_replay_ov_count",
        PendingSlot.NONE,
        "seed_overlays",
    ),
    PendingSlot.SLOPE_TERM_LO: ("is_slope_term_lo", PendingSlot.SLOPE_RUNTIME, None),
    PendingSlot.SLOPE_RUNTIME: ("is_slope_runtime", PendingSlot.NONE, None),
}

_ATOMIC_NEW_PENDING = (
    ("is_back_ref_dist_hi", PendingSlot.BACK_REF_DIST_LO),
    ("is_pattern_replay_dist_hi", PendingSlot.PR_DIST_LO),
    ("is_slope_term_hi", PendingSlot.SLOPE_TERM_LO),
)


def _make_slot_property(slot: PendingSlot):
    def fget(self) -> bool:
        return self.pending_slot == slot

    def fset(self, value: bool) -> None:
        if value:
            self.pending_slot = slot
        elif self.pending_slot == slot:
            self.pending_slot = PendingSlot.NONE

    return property(fget, fset)


class StreamState:
    """Per-step structural-validity tracker."""

    pending_back_ref_dist_lo = _make_slot_property(PendingSlot.BACK_REF_DIST_LO)
    pending_back_ref_len = _make_slot_property(PendingSlot.BACK_REF_LEN)
    pending_pr_dist_lo = _make_slot_property(PendingSlot.PR_DIST_LO)
    pending_pr_len = _make_slot_property(PendingSlot.PR_LEN)
    pending_pr_ov_count = _make_slot_property(PendingSlot.PR_OV_COUNT)
    pending_slope_term_lo = _make_slot_property(PendingSlot.SLOPE_TERM_LO)
    pending_slope_runtime = _make_slot_property(PendingSlot.SLOPE_RUNTIME)

    def __init__(
        self,
        vocab_arrays,
        init_frame_count,
        irq,
        init_budget=None,
        init_sval=0,
        init_fn=0,
        remaining_steps=None,
        logger=None,
        disable_resource_masks=False,
    ):
        self.arrays = vocab_arrays
        self.frame_count = int(init_frame_count)
        self.pending_overlays = 0
        self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
        self.pending_slot: PendingSlot = PendingSlot.NONE
        self.current_dist_hi = 0
        self.irq = int(irq)
        self.frame_budget = int(init_budget) if init_budget is not None else int(irq)
        self.current_sval = int(init_sval)
        self.current_fn = int(init_fn)
        self.remaining_steps = remaining_steps
        a = vocab_arrays
        self.subtoken_mode = bool(a.get("subtoken_mode", False))
        self.disable_resource_masks = disable_resource_masks
        self.logger = logger
        self._stuck_warned = False
        if self.subtoken_mode:
            self._slot_gate = _SUBTOKEN_SLOT_GATE
            self._overlay_gate = _SUBTOKEN_OVERLAY_GATES
        else:
            self._slot_gate = _ATOMIC_SLOT_GATE
            self._overlay_gate = _ATOMIC_OVERLAY_GATES

    def mask_logits(self, logits):
        """Set logits of structurally-invalid tokens to -inf. Computes the invalid mask in numpy then applies it to ``logits`` via a single ``masked_fill`` (torch is imported lazily so the rest of this module stays torch-free)."""
        import torch  # pylint: disable=import-outside-toplevel

        invalid_np = self.compute_invalid_mask()
        invalid = torch.from_numpy(invalid_np).to(logits.device)
        return logits.masked_fill(invalid, float("-inf"))

    def compute_invalid_mask(self):
        """Per-vocab-id bool numpy array; True for tokens that would violate structural invariants at the current state. Pure numpy; consumers in torch land call ``mask_logits`` instead."""
        a = self.arrays
        invalid = np.zeros(a["n_vocab"], dtype=np.bool_)
        invalid |= a["is_pad"]
        if self.subtoken_mode:
            invalid |= a["is_malformed_macro"]
        if self.pending_slot != PendingSlot.NONE:
            self._apply_pending_slot_mask(invalid, a)
        elif self.pending_overlays > 0:
            self._apply_overlay_slot_mask(invalid, a)
        elif self.subtoken_mode:
            self._apply_subtoken_free_mask(invalid, a)
        else:
            self._apply_atomic_free_mask(invalid, a)
        return self._unstick(invalid, a["is_frame_marker"])

    def update(self, token_id):
        """Advance state with the just-sampled token."""
        if self.subtoken_mode:
            self._update_subtoken(int(token_id))
        else:
            self._update_atomic(int(token_id))

    def _apply_pending_slot_mask(self, invalid, a):
        gate_key, check = self._slot_gate[self.pending_slot]
        gate = a[gate_key]
        invalid |= ~gate
        if check == "distance":
            full_dist = (self.current_dist_hi << BACK_REF_DIST_HI_SHIFT) + a[
                "dist_lo_val"
            ]
            invalid |= gate & (full_dist > self.frame_count)
        elif check == "overlay_cap" and self.remaining_steps is not None:
            cap = max((self.remaining_steps - 1) // 3, 0)
            invalid |= gate & (a["overlay_count"] > cap)

    def _apply_overlay_slot_mask(self, invalid, a):
        idx = _OVERLAY_SLOT_INDEX.get(self.pending_overlay_slot, 2)
        invalid |= ~a[self._overlay_gate[idx]]

    def _apply_atomic_free_mask(self, invalid, a):
        invalid |= a["is_pattern_overlay"]
        invalid |= a["is_pair_intermediate"]
        if self.frame_count <= 0:
            invalid |= a["is_dist_hi_row"]
        else:
            hi_max = self.frame_count >> BACK_REF_DIST_HI_SHIFT
            invalid |= a["is_dist_hi_row"] & (a["dist_hi_val"] > hi_max)
        if self.remaining_steps is not None:
            if self.remaining_steps < 3:
                invalid |= a["is_back_ref_dist_hi"]
            if self.remaining_steps < 4:
                invalid |= a["is_pattern_replay_dist_hi"]
        if not self.disable_resource_masks:
            invalid |= a["is_delay_reg"]
            if self.frame_budget < _MIN_DIFF:
                invalid |= a["is_real_reg"]

    def _apply_subtoken_free_mask(self, invalid, a):
        """Sub-token-aware mask for the free-choice branch: each entry summarizes the aggregate effect of a Unigram sub-token's atomic-id decomposition. Voice-dependent masks (GATE_REPLAY / PLAY_INSTRUMENT palettes) are skipped here — the safety net catches palette violations post-decode."""
        invalid |= a["is_singleton_pattern_overlay"]
        invalid |= a["is_singleton_pair_intermediate"]
        invalid |= (
            a["consumes_back_ref_dist_lo_gate"] & ~a["is_singleton_back_ref_dist_lo"]
        )
        invalid |= a["consumes_pr_dist_lo_gate"] & ~a["is_singleton_pr_dist_lo"]
        invalid |= a["consumes_back_ref_len_gate"] & ~a["is_singleton_back_ref_len"]
        invalid |= a["consumes_pr_len_gate"] & ~a["is_singleton_pr_len"]
        invalid |= (
            a["consumes_overlay_slot_1_gate"]
            & ~a["is_singleton_pattern_overlay_target_reg"]
        )
        if self.frame_count <= 0:
            invalid |= a["is_singleton_dist_hi"]
        else:
            hi_max = self.frame_count >> BACK_REF_DIST_HI_SHIFT
            invalid |= a["is_singleton_dist_hi"] & (a["distance_hi"] > hi_max)
            invalid |= (a["full_distance"] > 0) & (
                a["full_distance"] > self.frame_count
            )
        if self.remaining_steps is not None:
            if self.remaining_steps < 3:
                invalid |= a["is_singleton_back_ref_dist_hi"]
            if self.remaining_steps < 4:
                invalid |= a["is_singleton_pr_dist_hi"]
        if not self.disable_resource_masks:
            invalid |= a["contains_delay"]
            invalid |= a["charge_first_segment"] > self.frame_budget

    def _unstick(self, invalid, frame_marker):
        if invalid.all():
            if self.logger is not None and not self._stuck_warned:
                self.logger.warning(
                    "constrained_decode: all tokens masked at frame=%u, "
                    "pending_overlays=%u; falling back to a frame-advance token",
                    self.frame_count,
                    self.pending_overlays,
                )
                self._stuck_warned = True
            invalid = invalid.copy()
            frame_idxs = np.flatnonzero(frame_marker)
            if frame_idxs.size:
                invalid[int(frame_idxs[0])] = False
        return invalid

    def _advance_overlay_slot(self):
        if self.pending_overlay_slot == OverlaySlot.NEW_VAL:
            self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
            self.pending_overlays -= 1
        else:
            self.pending_overlay_slot = OverlaySlot(self.pending_overlay_slot + 1)

    def _update_atomic(self, token_id):
        a = self.arrays
        if self.remaining_steps is not None:
            self.remaining_steps -= 1
        if bool(a["is_frame_marker"][token_id].item()):
            self.frame_count += 1
            self.frame_budget = self.irq
        elif bool(a["is_real_reg"][token_id].item()):
            self.frame_budget -= _MIN_DIFF
        if a["is_frame_reg_strict"][token_id]:
            self.current_sval = int(a["frame_sval"][token_id])
            self.current_fn = 0
        elif a["is_voice_reg"][token_id]:
            self.current_fn += 1
        if self.pending_slot != PendingSlot.NONE:
            assert_key, next_slot, action = _ATOMIC_SLOT_TRANSITION[self.pending_slot]
            assert a[assert_key][token_id], (
                f"pending {self.pending_slot.name} but token {token_id} does not "
                f"match {assert_key}"
            )
            self.pending_slot = next_slot
            if action == "seed_overlays":
                self.pending_overlays = int(a["overlay_count"][token_id])
                self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
        elif self.pending_overlays > 0:
            self._advance_overlay_slot()
        else:
            for gate_key, next_slot in _ATOMIC_NEW_PENDING:
                if a[gate_key][token_id]:
                    self.pending_slot = next_slot
                    if next_slot in (
                        PendingSlot.BACK_REF_DIST_LO,
                        PendingSlot.PR_DIST_LO,
                    ):
                        self.current_dist_hi = int(a["dist_hi_val"][token_id])
                    break

    def _update_subtoken(self, sub_id):
        a = self.arrays
        if self.remaining_steps is not None:
            self.remaining_steps -= 1
        fa = int(a["frame_advance"][sub_id])
        self.frame_count += fa
        if fa > 0:
            self.frame_budget = self.irq - int(a["charge_last_segment"][sub_id])
        else:
            self.frame_budget -= int(a["charge_first_segment"][sub_id])
        if bool(a["sets_sval"][sub_id]):
            self.current_sval = int(a["final_sval"][sub_id])
            self.current_fn = int(a["fn_after_last_strict"][sub_id])
        else:
            self.current_fn += int(a["fn_delta"][sub_id])
        if self.pending_back_ref_dist_lo:
            self.pending_back_ref_dist_lo = False
            if not bool(a["extends_to_back_ref_len_consumed"][sub_id]):
                self.pending_back_ref_len = True
        elif self.pending_pr_dist_lo:
            self.pending_pr_dist_lo = False
            if a["extends_to_pr_ov_count_consumed"][sub_id]:
                self.pending_overlays += int(a["overlay_count"][sub_id])
                self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
            elif bool(a["extends_to_pr_len_consumed"][sub_id]):
                pass
            else:
                self.pending_pr_len = True
        elif self.pending_back_ref_len:
            self.pending_back_ref_len = False
        elif self.pending_pr_len:
            self.pending_pr_len = False
            if a["extends_to_pr_ov_count_consumed"][sub_id]:
                self.pending_overlays += int(a["overlay_count"][sub_id])
                self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
            else:
                self.pending_pr_ov_count = True
        elif self.pending_pr_ov_count:
            self.pending_pr_ov_count = False
            self.pending_overlays += int(a["overlay_count"][sub_id])
            self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
        elif self.pending_overlays > 0:
            if a["extends_to_overlay_completed"][sub_id]:
                self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
                self.pending_overlays -= 1
            else:
                self._advance_overlay_slot()
        elif a["is_singleton_back_ref_dist_hi"][sub_id]:
            self.current_dist_hi = int(a["distance_hi"][sub_id])
            if bool(a["extends_to_back_ref_len_consumed"][sub_id]):
                pass
            elif bool(a["extends_to_back_ref_lo_consumed"][sub_id]):
                self.pending_back_ref_len = True
            else:
                self.pending_back_ref_dist_lo = True
        elif a["is_singleton_pr_dist_hi"][sub_id]:
            self.current_dist_hi = int(a["distance_hi"][sub_id])
            if bool(a["extends_to_pr_ov_count_consumed"][sub_id]):
                self.pending_overlays += int(a["overlay_count"][sub_id])
                self.pending_overlay_slot = OverlaySlot.FRAME_OFFSET
            elif bool(a["extends_to_pr_len_consumed"][sub_id]):
                self.pending_pr_ov_count = True
            elif bool(a["extends_to_pr_lo_consumed"][sub_id]):
                self.pending_pr_len = True
            else:
                self.pending_pr_dist_lo = True
