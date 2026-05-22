"""Sampling-time logit guard for predict.py."""

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
    SLOPE_OPS,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
    VOICE_REG,
)


def _frame_marker_count(token_ids, is_frame_marker):
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
    return int(is_real_reg[tail].sum() * MIN_DIFF)


def precompute_vocab_arrays(tokens_df):
    """Per-vocab-id numpy arrays for the per-step mask. Sized by the atomic alphabet -- correct when the model emits atomic ids (``tkvocab=0``). For Unigram (``tkvocab > 0``) the model emits sub-token ids and ``StreamState`` would index out of bounds; use ``precompute_subtoken_arrays`` instead."""
    n = len(tokens_df)
    op = tokens_df["op"].fillna(SET_OP).astype(np.int64).to_numpy()
    reg = tokens_df["reg"].astype(np.int64).to_numpy()
    subreg = tokens_df["subreg"].fillna(-1).astype(np.int64).to_numpy()
    val = tokens_df["val"].astype(np.int64).to_numpy()

    is_frame_marker = np.isin(reg, [FRAME_REG, DELAY_REG])
    is_frame_reg_strict = reg == FRAME_REG
    is_voice_reg = reg == VOICE_REG
    is_delay_reg = reg == DELAY_REG
    is_pad = reg == PAD_REG
    is_real_reg = (reg >= 0) & (reg <= MAX_REG)
    is_back_ref = op == BACK_REF_OP
    is_pattern_replay = op == PATTERN_REPLAY_OP
    is_slope = np.isin(op, np.asarray(SLOPE_OPS, dtype=np.int64))
    is_slope_term_hi = is_slope & (subreg == SLOPE_SUBREG_TERMINAL_HI)
    is_slope_term_lo = is_slope & (subreg == SLOPE_SUBREG_TERMINAL_LO)
    is_slope_runtime = is_slope & (subreg == SLOPE_SUBREG_RUNTIME)
    is_back_ref_dist_hi = is_back_ref & (subreg == BACK_REF_SUBREG_DIST_HI)
    is_back_ref_dist_lo = is_back_ref & (subreg == BACK_REF_SUBREG_DIST_LO)
    is_back_ref_len = is_back_ref & (subreg == BACK_REF_SUBREG_LEN)
    is_pattern_replay_dist_hi = is_pattern_replay & (
        subreg == PATTERN_REPLAY_SUBREG_DIST_HI
    )
    is_pattern_replay_dist_lo = is_pattern_replay & (
        subreg == PATTERN_REPLAY_SUBREG_DIST_LO
    )
    is_pattern_replay_len = is_pattern_replay & (subreg == PATTERN_REPLAY_SUBREG_LEN)
    is_pattern_replay_ov_count = is_pattern_replay & (
        subreg == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
    )
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
    is_pattern_overlay_frame_offset = is_pattern_overlay & (
        subreg == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
    )
    is_pattern_overlay_target_reg = is_pattern_overlay & (
        subreg == PATTERN_OVERLAY_SUBREG_TARGET_REG
    )
    is_pattern_overlay_new_val = is_pattern_overlay & (
        subreg == PATTERN_OVERLAY_SUBREG_NEW_VAL
    )
    dist_hi_val = np.zeros(n, dtype=np.int64)
    dist_hi_val[is_dist_hi_row] = val[is_dist_hi_row]
    dist_lo_val = np.zeros(n, dtype=np.int64)
    dist_lo_val[is_dist_lo_row] = val[is_dist_lo_row]

    length = np.zeros(n, dtype=np.int64)
    length[is_back_ref_len | is_pattern_replay_len] = val[
        is_back_ref_len | is_pattern_replay_len
    ]
    overlay_count = np.zeros(n, dtype=np.int64)
    overlay_count[is_pattern_replay_ov_count] = val[is_pattern_replay_ov_count]

    frame_sval = np.zeros(n, dtype=np.int64)
    frame_sval[is_frame_reg_strict] = val[is_frame_reg_strict] & 0x3F

    return {
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
        "is_pattern_overlay_target_reg": is_pattern_overlay_target_reg.astype(np.bool_),
        "is_pattern_overlay_new_val": is_pattern_overlay_new_val.astype(np.bool_),
        "is_frame_reg_strict": is_frame_reg_strict.astype(np.bool_),
        "is_voice_reg": is_voice_reg.astype(np.bool_),
        "frame_sval": frame_sval,
        "dist_hi_val": dist_hi_val,
        "dist_lo_val": dist_lo_val,
        "length": length,
        "overlay_count": overlay_count,
    }


def precompute_subtoken_arrays(tokens_df, regtokenizer, pad_id=0):
    """Per-sub-token numpy arrays for the per-step mask under Unigram."""
    tkmodel = regtokenizer.tkmodel
    if tkmodel is None:
        raise ValueError("precompute_subtoken_arrays requires a trained tkmodel")
    n_sub = tkmodel.get_vocab_size()
    n_atomic = len(tokens_df)
    op_a = tokens_df["op"].fillna(SET_OP).astype(np.int64).to_numpy()
    reg_a = tokens_df["reg"].astype(np.int64).to_numpy()
    subreg_a = tokens_df["subreg"].fillna(-1).astype(np.int64).to_numpy()
    val_a = tokens_df["val"].astype(np.int64).to_numpy()
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

    def _classify_macro_shape(atomic_ids):
        """Classify a multi-atom sub-token's macro structure."""
        n = atomic_ids.size
        if n == 0:
            return (None,)
        first_macro_idx = -1
        for k in range(n):
            if is_macro_a[int(atomic_ids[k])]:
                first_macro_idx = k
                break
        if first_macro_idx == -1:
            return (None,)
        if first_macro_idx > 0:
            return ("malformed",)
        first = int(atomic_ids[0])
        first_op = int(op_a[first])
        first_sr = int(subreg_a[first])
        first_val = int(val_a[first])
        if n == 1:
            if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_DIST_HI:
                return ("singleton_back_ref_dist_hi", first_val)
            if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_DIST_LO:
                return ("singleton_back_ref_dist_lo", first_val)
            if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_LEN:
                return ("singleton_back_ref_len", first_val)
            if first_op == PATTERN_REPLAY_OP:
                if first_sr == PATTERN_REPLAY_SUBREG_DIST_HI:
                    return ("singleton_pr_dist_hi", first_val)
                if first_sr == PATTERN_REPLAY_SUBREG_DIST_LO:
                    return ("singleton_pr_dist_lo", first_val)
                if first_sr == PATTERN_REPLAY_SUBREG_LEN:
                    return ("singleton_pr_len", first_val)
                if first_sr == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT:
                    return ("singleton_pr_ov_count", first_val)
            if first_op == PATTERN_OVERLAY_OP:
                if first_sr == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                    return ("singleton_overlay_frame_offset", first_val)
                if first_sr == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                    return ("singleton_overlay_target_reg", first_val)
                if first_sr == PATTERN_OVERLAY_SUBREG_NEW_VAL:
                    return ("singleton_overlay_new_val", first_val)
            return ("malformed",)
        if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_DIST_HI:
            if n == 2:
                second = int(atomic_ids[1])
                if (
                    int(op_a[second]) == BACK_REF_OP
                    and int(subreg_a[second]) == BACK_REF_SUBREG_DIST_LO
                ):
                    return ("br_hi_then_lo", first_val)
            if n == 3:
                second = int(atomic_ids[1])
                third = int(atomic_ids[2])
                if (
                    int(op_a[second]) == BACK_REF_OP
                    and int(subreg_a[second]) == BACK_REF_SUBREG_DIST_LO
                    and int(op_a[third]) == BACK_REF_OP
                    and int(subreg_a[third]) == BACK_REF_SUBREG_LEN
                ):
                    return ("br_complete", first_val)
            return ("malformed",)
        if first_op == PATTERN_REPLAY_OP and first_sr == PATTERN_REPLAY_SUBREG_DIST_HI:
            if n == 2:
                second = int(atomic_ids[1])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_DIST_LO
                ):
                    return ("pr_hi_then_lo", first_val)
            if n == 3:
                second = int(atomic_ids[1])
                third = int(atomic_ids[2])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_DIST_LO
                    and int(op_a[third]) == PATTERN_REPLAY_OP
                    and int(subreg_a[third]) == PATTERN_REPLAY_SUBREG_LEN
                ):
                    return ("pr_hi_through_len", first_val)
            if n == 4:
                second = int(atomic_ids[1])
                third = int(atomic_ids[2])
                fourth = int(atomic_ids[3])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_DIST_LO
                    and int(op_a[third]) == PATTERN_REPLAY_OP
                    and int(subreg_a[third]) == PATTERN_REPLAY_SUBREG_LEN
                    and int(op_a[fourth]) == PATTERN_REPLAY_OP
                    and int(subreg_a[fourth]) == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
                ):
                    return ("pr_complete", first_val, int(val_a[fourth]))
            return ("malformed",)
        if (
            first_op == PATTERN_OVERLAY_OP
            and first_sr == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
        ):
            return ("malformed",)
        if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_DIST_LO:
            if n == 2:
                second = int(atomic_ids[1])
                if (
                    int(op_a[second]) == BACK_REF_OP
                    and int(subreg_a[second]) == BACK_REF_SUBREG_LEN
                ):
                    return ("br_lo_then_len",)
            return ("malformed",)
        if first_op == PATTERN_REPLAY_OP and first_sr == PATTERN_REPLAY_SUBREG_DIST_LO:
            if n == 2:
                second = int(atomic_ids[1])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_LEN
                ):
                    return ("pr_lo_then_len",)
            if n == 3:
                second = int(atomic_ids[1])
                third = int(atomic_ids[2])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_LEN
                    and int(op_a[third]) == PATTERN_REPLAY_OP
                    and int(subreg_a[third]) == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
                ):
                    return ("pr_lo_through_ov_count", int(val_a[third]))
            return ("malformed",)
        if first_op == BACK_REF_OP and first_sr == BACK_REF_SUBREG_LEN:
            for k in range(1, n):
                if is_macro_a[int(atomic_ids[k])]:
                    return ("malformed",)
            return ("br_len_with_tail", first_val)
        if first_op == PATTERN_REPLAY_OP and first_sr == PATTERN_REPLAY_SUBREG_LEN:
            if n == 2:
                second = int(atomic_ids[1])
                if (
                    int(op_a[second]) == PATTERN_REPLAY_OP
                    and int(subreg_a[second]) == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
                ):
                    return ("pr_len_then_ov_count", int(val_a[second]))
            return ("malformed",)
        if (
            first_op == PATTERN_REPLAY_OP
            and first_sr == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
        ):
            return ("malformed",)
        if (
            first_op == PATTERN_OVERLAY_OP
            and first_sr == PATTERN_OVERLAY_SUBREG_TARGET_REG
        ):
            if n != 2:
                return ("malformed",)
            second = int(atomic_ids[1])
            if (
                int(op_a[second]) == PATTERN_OVERLAY_OP
                and int(subreg_a[second]) == PATTERN_OVERLAY_SUBREG_NEW_VAL
            ):
                return ("ov_target_then_new_val",)
            return ("malformed",)
        if (
            first_op == PATTERN_OVERLAY_OP
            and first_sr == PATTERN_OVERLAY_SUBREG_NEW_VAL
        ):
            return ("malformed",)
        return ("malformed",)

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
        shape = _classify_macro_shape(atomic_ids)
        tag = shape[0]
        if tag == "malformed":
            is_malformed_macro[sub_id] = True
        elif tag == "singleton_back_ref_dist_hi":
            is_singleton_back_ref_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
        elif tag == "singleton_back_ref_dist_lo":
            is_singleton_back_ref_dist_lo[sub_id] = True
            consumes_back_ref_dist_lo_gate[sub_id] = True
        elif tag == "singleton_back_ref_len":
            is_singleton_back_ref_len[sub_id] = True
            consumes_back_ref_len_gate[sub_id] = True
        elif tag == "singleton_pr_dist_hi":
            is_singleton_pr_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
        elif tag == "singleton_pr_dist_lo":
            is_singleton_pr_dist_lo[sub_id] = True
            consumes_pr_dist_lo_gate[sub_id] = True
        elif tag == "singleton_pr_len":
            is_singleton_pr_len[sub_id] = True
            consumes_pr_len_gate[sub_id] = True
        elif tag == "singleton_pr_ov_count":
            is_singleton_pr_ov_count[sub_id] = True
            consumes_pr_ov_count_gate[sub_id] = True
            overlay_count[sub_id] = shape[1]
            pending_overlays_delta[sub_id] = shape[1]
        elif tag == "singleton_overlay_frame_offset":
            is_singleton_pattern_overlay[sub_id] = True
            is_singleton_pattern_overlay_frame_offset[sub_id] = True
            consumes_overlay_slot_0_gate[sub_id] = True
        elif tag == "singleton_overlay_target_reg":
            is_singleton_pattern_overlay[sub_id] = True
            is_singleton_pattern_overlay_target_reg[sub_id] = True
            consumes_overlay_slot_1_gate[sub_id] = True
        elif tag == "singleton_overlay_new_val":
            is_singleton_pattern_overlay[sub_id] = True
            is_singleton_pattern_overlay_new_val[sub_id] = True
            consumes_overlay_slot_2_gate[sub_id] = True
        elif tag == "br_hi_then_lo":
            is_singleton_back_ref_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
            extends_to_back_ref_lo_consumed[sub_id] = True
        elif tag == "br_complete":
            is_singleton_back_ref_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
            extends_to_back_ref_lo_consumed[sub_id] = True
            extends_to_back_ref_len_consumed[sub_id] = True
        elif tag == "br_lo_then_len":
            consumes_back_ref_dist_lo_gate[sub_id] = True
            extends_to_back_ref_len_consumed[sub_id] = True
        elif tag == "pr_hi_then_lo":
            is_singleton_pr_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
            extends_to_pr_lo_consumed[sub_id] = True
        elif tag == "pr_hi_through_len":
            is_singleton_pr_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
            extends_to_pr_lo_consumed[sub_id] = True
            extends_to_pr_len_consumed[sub_id] = True
        elif tag == "pr_complete":
            is_singleton_pr_dist_hi[sub_id] = True
            distance_hi[sub_id] = shape[1]
            overlay_count[sub_id] = shape[2]
            pending_overlays_delta[sub_id] = shape[2]
            extends_to_pr_lo_consumed[sub_id] = True
            extends_to_pr_len_consumed[sub_id] = True
            extends_to_pr_ov_count_consumed[sub_id] = True
        elif tag == "pr_lo_then_len":
            consumes_pr_dist_lo_gate[sub_id] = True
            extends_to_pr_len_consumed[sub_id] = True
        elif tag == "pr_lo_through_ov_count":
            consumes_pr_dist_lo_gate[sub_id] = True
            extends_to_pr_len_consumed[sub_id] = True
            extends_to_pr_ov_count_consumed[sub_id] = True
            overlay_count[sub_id] = shape[1]
            pending_overlays_delta[sub_id] = shape[1]
        elif tag == "br_len_with_tail":
            consumes_back_ref_len_gate[sub_id] = True
        elif tag == "pr_len_then_ov_count":
            _, ov_count_val = shape
            consumes_pr_len_gate[sub_id] = True
            extends_to_pr_ov_count_consumed[sub_id] = True
            overlay_count[sub_id] = ov_count_val
            pending_overlays_delta[sub_id] = ov_count_val
        elif tag == "ov_target_then_new_val":
            consumes_overlay_slot_1_gate[sub_id] = True
            extends_to_overlay_completed[sub_id] = True

        if tag in (
            "singleton_back_ref_dist_lo",
            "singleton_pr_dist_lo",
            "br_lo_then_len",
            "pr_lo_then_len",
            "pr_lo_through_ov_count",
        ):
            dist_lo_val[sub_id] = int(val_a[int(atomic_ids[0])]) & BACK_REF_DIST_LO_MASK
        if tag in (
            "br_hi_then_lo",
            "br_complete",
            "pr_hi_then_lo",
            "pr_hi_through_len",
            "pr_complete",
        ):
            lo_byte = int(val_a[int(atomic_ids[1])]) & BACK_REF_DIST_LO_MASK
            full_distance[sub_id] = (
                int(distance_hi[sub_id]) << BACK_REF_DIST_HI_SHIFT
            ) | lo_byte

        local_frame = 0
        first_seg_charge = 0
        last_seg_charge = 0
        first_seg_done = False
        local_sets_sval = False
        local_final_sval = 0
        local_fn_delta = 0
        local_fn_after_strict = 0
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
                    contains_delay[sub_id] = True
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
                last_seg_charge += MIN_DIFF
        if not first_seg_done:
            first_seg_charge = last_seg_charge
        frame_advance[sub_id] = local_frame
        charge_first_segment[sub_id] = first_seg_charge
        charge_last_segment[sub_id] = last_seg_charge
        sets_sval[sub_id] = local_sets_sval
        final_sval[sub_id] = local_final_sval
        fn_delta[sub_id] = local_fn_delta
        fn_after_last_strict[sub_id] = local_fn_after_strict

    is_singleton_dist_hi = is_singleton_back_ref_dist_hi | is_singleton_pr_dist_hi
    is_singleton_pair_intermediate = (
        is_singleton_back_ref_dist_lo
        | is_singleton_back_ref_len
        | is_singleton_pr_dist_lo
        | is_singleton_pr_len
        | is_singleton_pr_ov_count
    )

    return {
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


class StreamState:
    """Per-step structural-validity tracker."""

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
        self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
        self.pending_back_ref_dist_lo = False
        self.pending_back_ref_len = False
        self.pending_pr_dist_lo = False
        self.pending_pr_len = False
        self.pending_pr_ov_count = False
        self.pending_slope_term_lo = False
        self.pending_slope_runtime = False
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

    def mask_logits(self, logits):
        """Set logits of structurally-invalid tokens to -inf. Computes the invalid mask in numpy then applies it to ``logits`` via a single ``masked_fill`` (torch is imported lazily so the rest of this module stays torch-free)."""
        import torch  # pylint: disable=import-outside-toplevel

        invalid_np = self._compute_invalid()
        invalid = torch.from_numpy(invalid_np).to(logits.device)
        return logits.masked_fill(invalid, float("-inf"))

    def _compute_invalid(self):
        if self.subtoken_mode:
            invalid = self._compute_invalid_subtoken()
        else:
            invalid = self._compute_invalid_atomic()
        return self._unstick(invalid, self.arrays["is_frame_marker"])

    def update(self, token_id):
        """Advance state with the just-sampled token."""
        if self.subtoken_mode:
            self._update_subtoken(int(token_id))
        else:
            self._update_atomic(int(token_id))

    def _compute_invalid_atomic(self):
        a = self.arrays
        invalid = np.zeros(a["n_vocab"], dtype=np.bool_)

        invalid |= a["is_pad"]
        if self.pending_back_ref_dist_lo:
            invalid |= ~a["is_back_ref_dist_lo"]
            full_dist = (self.current_dist_hi << BACK_REF_DIST_HI_SHIFT) + a[
                "dist_lo_val"
            ]
            too_far = full_dist > self.frame_count
            invalid |= a["is_back_ref_dist_lo"] & too_far
        elif self.pending_pr_dist_lo:
            invalid |= ~a["is_pattern_replay_dist_lo"]
            full_dist = (self.current_dist_hi << BACK_REF_DIST_HI_SHIFT) + a[
                "dist_lo_val"
            ]
            too_far = full_dist > self.frame_count
            invalid |= a["is_pattern_replay_dist_lo"] & too_far
        elif self.pending_back_ref_len:
            invalid |= ~a["is_back_ref_len"]
        elif self.pending_pr_len:
            invalid |= ~a["is_pattern_replay_len"]
        elif self.pending_pr_ov_count:
            invalid |= ~a["is_pattern_replay_ov_count"]
            if self.remaining_steps is not None:
                cap = max((self.remaining_steps - 1) // 3, 0)
                ov_too_long = a["is_pattern_replay_ov_count"] & (
                    a["overlay_count"] > cap
                )
                invalid |= ov_too_long
        elif self.pending_slope_term_lo:
            invalid |= ~a["is_slope_term_lo"]
        elif self.pending_slope_runtime:
            invalid |= ~a["is_slope_runtime"]
        elif self.pending_overlays > 0:
            if self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                invalid |= ~a["is_pattern_overlay_frame_offset"]
            elif self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                invalid |= ~a["is_pattern_overlay_target_reg"]
            else:
                invalid |= ~a["is_pattern_overlay_new_val"]
        else:
            invalid |= a["is_pattern_overlay"]
            invalid |= a["is_pair_intermediate"]
            if self.frame_count <= 0:
                invalid |= a["is_dist_hi_row"]
            else:
                hi_max = self.frame_count >> BACK_REF_DIST_HI_SHIFT
                too_far_hi = a["dist_hi_val"] > hi_max
                invalid |= a["is_dist_hi_row"] & too_far_hi
            if self.remaining_steps is not None:
                if self.remaining_steps < 3:
                    invalid |= a["is_back_ref_dist_hi"]
                if self.remaining_steps < 4:
                    invalid |= a["is_pattern_replay_dist_hi"]
            if not self.disable_resource_masks:
                invalid |= a["is_delay_reg"]
                if self.frame_budget < MIN_DIFF:
                    invalid |= a["is_real_reg"]
        return invalid

    def _compute_invalid_subtoken(self):
        """Sub-token-aware mask: each entry summarizes the aggregate effect of a Unigram sub-token's atomic-id decomposition. Voice-dependent masks (GATE_REPLAY palette / PLAY_INSTRUMENT palette) are skipped here -- the safety net catches palette violations post-decode."""
        a = self.arrays
        invalid = np.zeros(a["n_vocab"], dtype=np.bool_)
        invalid |= a["is_pad"]
        invalid |= a["is_malformed_macro"]
        if self.pending_back_ref_dist_lo:
            invalid |= ~a["consumes_back_ref_dist_lo_gate"]
            full_dist = (self.current_dist_hi << BACK_REF_DIST_HI_SHIFT) + a[
                "dist_lo_val"
            ]
            too_far = full_dist > self.frame_count
            invalid |= a["consumes_back_ref_dist_lo_gate"] & too_far
        elif self.pending_pr_dist_lo:
            invalid |= ~a["consumes_pr_dist_lo_gate"]
            full_dist = (self.current_dist_hi << BACK_REF_DIST_HI_SHIFT) + a[
                "dist_lo_val"
            ]
            too_far = full_dist > self.frame_count
            invalid |= a["consumes_pr_dist_lo_gate"] & too_far
        elif self.pending_back_ref_len:
            invalid |= ~a["consumes_back_ref_len_gate"]
        elif self.pending_pr_len:
            invalid |= ~a["consumes_pr_len_gate"]
        elif self.pending_pr_ov_count:
            invalid |= ~a["consumes_pr_ov_count_gate"]
            if self.remaining_steps is not None:
                cap = max((self.remaining_steps - 1) // 3, 0)
                ov_too_long = a["consumes_pr_ov_count_gate"] & (
                    a["overlay_count"] > cap
                )
                invalid |= ov_too_long
        elif self.pending_overlays > 0:
            if self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                invalid |= ~a["consumes_overlay_slot_0_gate"]
            elif self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                invalid |= ~a["consumes_overlay_slot_1_gate"]
            else:
                invalid |= ~a["consumes_overlay_slot_2_gate"]
        else:
            invalid |= a["is_singleton_pattern_overlay"]
            invalid |= a["is_singleton_pair_intermediate"]
            invalid |= (
                a["consumes_back_ref_dist_lo_gate"]
                & ~a["is_singleton_back_ref_dist_lo"]
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
                too_far_hi = a["distance_hi"] > hi_max
                invalid |= a["is_singleton_dist_hi"] & too_far_hi
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
        return invalid

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

    def _update_atomic(self, token_id):
        a = self.arrays
        if self.remaining_steps is not None:
            self.remaining_steps -= 1
        if bool(a["is_frame_marker"][token_id].item()):
            self.frame_count += 1
            self.frame_budget = self.irq
        elif bool(a["is_real_reg"][token_id].item()):
            self.frame_budget -= MIN_DIFF
        if a["is_frame_reg_strict"][token_id]:
            self.current_sval = int(a["frame_sval"][token_id])
            self.current_fn = 0
        elif a["is_voice_reg"][token_id]:
            self.current_fn += 1
        if self.pending_back_ref_dist_lo:
            assert a["is_back_ref_dist_lo"][token_id], (
                f"pending_back_ref_dist_lo but token {token_id} is not a "
                f"BACK_REF DIST_LO row"
            )
            self.pending_back_ref_dist_lo = False
            self.pending_back_ref_len = True
        elif self.pending_pr_dist_lo:
            assert a["is_pattern_replay_dist_lo"][token_id], (
                f"pending_pr_dist_lo but token {token_id} is not a "
                f"PATTERN_REPLAY DIST_LO row"
            )
            self.pending_pr_dist_lo = False
            self.pending_pr_len = True
        elif self.pending_back_ref_len:
            assert a["is_back_ref_len"][token_id], (
                f"pending_back_ref_len but token {token_id} is not a "
                f"BACK_REF length row"
            )
            self.pending_back_ref_len = False
        elif self.pending_pr_len:
            assert a["is_pattern_replay_len"][token_id], (
                f"pending_pr_len but token {token_id} is not a "
                f"PATTERN_REPLAY length row"
            )
            self.pending_pr_len = False
            self.pending_pr_ov_count = True
        elif self.pending_pr_ov_count:
            assert a["is_pattern_replay_ov_count"][token_id], (
                f"pending_pr_ov_count but token {token_id} is not a "
                f"PATTERN_REPLAY overlay_count row"
            )
            self.pending_pr_ov_count = False
            self.pending_overlays = int(a["overlay_count"][token_id])
            self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
        elif self.pending_overlays > 0:
            if self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_TARGET_REG
            elif self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_NEW_VAL
            else:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
                self.pending_overlays -= 1
        elif self.pending_slope_term_lo:
            assert a["is_slope_term_lo"][token_id], token_id
            self.pending_slope_term_lo = False
            self.pending_slope_runtime = True
        elif self.pending_slope_runtime:
            assert a["is_slope_runtime"][token_id], token_id
            self.pending_slope_runtime = False
        elif a["is_back_ref_dist_hi"][token_id]:
            self.pending_back_ref_dist_lo = True
            self.current_dist_hi = int(a["dist_hi_val"][token_id])
        elif a["is_pattern_replay_dist_hi"][token_id]:
            self.pending_pr_dist_lo = True
            self.current_dist_hi = int(a["dist_hi_val"][token_id])
        elif a["is_slope_term_hi"][token_id]:
            self.pending_slope_term_lo = True

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
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
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
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
            else:
                self.pending_pr_ov_count = True
        elif self.pending_pr_ov_count:
            self.pending_pr_ov_count = False
            self.pending_overlays += int(a["overlay_count"][sub_id])
            self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
        elif self.pending_overlays > 0:
            if a["extends_to_overlay_completed"][sub_id]:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
                self.pending_overlays -= 1
            elif self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_TARGET_REG
            elif self.pending_overlay_slot == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_NEW_VAL
            else:
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
                self.pending_overlays -= 1
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
                self.pending_overlay_slot = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
            elif bool(a["extends_to_pr_len_consumed"][sub_id]):
                self.pending_pr_ov_count = True
            elif bool(a["extends_to_pr_lo_consumed"][sub_id]):
                self.pending_pr_len = True
            else:
                self.pending_pr_dist_lo = True
