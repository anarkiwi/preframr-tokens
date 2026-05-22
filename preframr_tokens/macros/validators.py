"""Stream-level validators for encoded macro DataFrames."""

__all__ = ["validate_pattern_overlays", "validate_back_refs"]

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from preframr_tokens.macros.roles import (
    DISTANCE_PAIR_OPS,
    DistancePairSpec,
    distance_pair_role,
)
from preframr_tokens.macros.state import _FRAME_MARKER_REGS
from preframr_tokens.stfconstants import (
    BACK_REF_DIST_HI_SHIFT,
    BACK_REF_OP,
    DO_LOOP_OP,
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
)


@dataclass
class _DistancePairState:
    output_frame_count: int = 0
    pending_dist_op: Optional[int] = None
    pending_dist_idx: Optional[int] = None
    pending_dist_hi: Optional[int] = None


def _step_distance_pair(idx, row, sr, op, spec: DistancePairSpec, state):
    """Advance one row through the distance-pair state machine. Mutates state."""
    label = spec.label
    role = distance_pair_role(op, sr)
    if role == "dist_hi":
        assert state.pending_dist_op is None, (
            f"row {idx}: pending {state.pending_dist_op} at row "
            f"{state.pending_dist_idx} not closed before new {label} DIST_HI"
        )
        state.pending_dist_op = op
        state.pending_dist_idx = idx
        state.pending_dist_hi = int(row["val"])
        return
    if role == "dist_lo":
        assert state.pending_dist_op == op and state.pending_dist_hi is not None, (
            f"row {idx}: {label} DIST_LO without preceding DIST_HI "
            f"(pending_dist_op={state.pending_dist_op})"
        )
        distance = (state.pending_dist_hi << BACK_REF_DIST_HI_SHIFT) | int(row["val"])
        target = state.output_frame_count - distance
        long_label = "PATTERN_REPLAY" if op == PATTERN_REPLAY_OP else label
        assert target >= 0, (
            f"row {idx}: {long_label} distance={distance} reaches "
            f"before frame 0 (output_frame_count={state.output_frame_count})"
        )
        state.pending_dist_hi = None
        return
    if role == "len":
        assert state.pending_dist_op == op and state.pending_dist_hi is None, (
            f"row {idx}: {label} length without a complete DIST pair "
            f"(pending_dist_op={state.pending_dist_op}, "
            f"hi={state.pending_dist_hi})"
        )
        state.output_frame_count += int(row["val"])
        state.pending_dist_op = None
        state.pending_dist_idx = None
        return
    if role == "ov_count":
        return
    allowed = sorted({spec.dist_hi, spec.dist_lo, spec.length, *spec.extra_subregs})
    raise AssertionError(
        f"row {idx}: {label if op == BACK_REF_OP else 'PATTERN_REPLAY'} "
        f"subreg={sr} not in {{{', '.join(str(s) for s in allowed)}}}"
    )


def validate_pattern_overlays(df):
    """Walk ``df`` and verify PATTERN_REPLAY triples are followed by exactly
    ``overlay_count`` PATTERN_OVERLAY triples -- the same pairing rule
    ``expand_loops`` enforces with an opaque ``orphan PATTERN_OVERLAY_OP``
    AssertionError. Catching it up here gives the predict-side safety
    net a clear error and lets it short-circuit before we hand the
    """
    if "op" not in df.columns:
        return True
    if not df["op"].isin([PATTERN_REPLAY_OP, PATTERN_OVERLAY_OP]).any():
        return True
    pending = 0
    pr_idx = None
    head_expected = PATTERN_REPLAY_SUBREG_DIST_HI
    ov_slot_expected = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
    for idx, row in df.iterrows():
        op_raw = row["op"]
        op = int(op_raw) if not pd.isna(op_raw) else SET_OP
        if op == PATTERN_REPLAY_OP:
            assert pending == 0, (
                f"row {idx}: PATTERN_REPLAY at row {pr_idx} expected "
                f"{pending} more overlays but a new PATTERN_REPLAY "
                f"started instead"
            )
            sr_raw = row.get("subreg", -1)
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            assert sr == head_expected, (
                f"row {idx}: PATTERN_REPLAY subreg={sr} out of order "
                f"(expected subreg={head_expected})"
            )
            if sr == PATTERN_REPLAY_SUBREG_DIST_HI:
                pr_idx = idx
                head_expected = PATTERN_REPLAY_SUBREG_DIST_LO
            elif sr == PATTERN_REPLAY_SUBREG_DIST_LO:
                head_expected = PATTERN_REPLAY_SUBREG_LEN
            elif sr == PATTERN_REPLAY_SUBREG_LEN:
                head_expected = PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
            else:
                count = int(row["val"])
                assert count >= 0, (
                    f"row {idx}: PATTERN_REPLAY overlay_count={count} " f"must be >= 0"
                )
                pending = count
                head_expected = PATTERN_REPLAY_SUBREG_DIST_HI
                ov_slot_expected = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
            continue
        assert head_expected == PATTERN_REPLAY_SUBREG_DIST_HI, (
            f"row {idx}: PATTERN_REPLAY quad at row {pr_idx} interrupted "
            f"with op={op} before subreg={head_expected} row arrived"
        )
        if op == PATTERN_OVERLAY_OP:
            assert pending > 0, (
                f"row {idx}: orphan PATTERN_OVERLAY_OP "
                f"(no PATTERN_REPLAY with pending overlays)"
            )
            sr_raw = row.get("subreg", -1)
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            assert sr == ov_slot_expected, (
                f"row {idx}: PATTERN_OVERLAY subreg={sr} out of order "
                f"(expected subreg={ov_slot_expected})"
            )
            if sr == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET:
                ov_slot_expected = PATTERN_OVERLAY_SUBREG_TARGET_REG
            elif sr == PATTERN_OVERLAY_SUBREG_TARGET_REG:
                ov_slot_expected = PATTERN_OVERLAY_SUBREG_NEW_VAL
            else:
                ov_slot_expected = PATTERN_OVERLAY_SUBREG_FRAME_OFFSET
                pending -= 1
            continue
        assert pending == 0, (
            f"row {idx}: PATTERN_REPLAY at row {pr_idx} interrupted "
            f"with op={op} before its {pending} remaining overlays"
        )
    assert pending == 0, (
        f"PATTERN_REPLAY at row {pr_idx} unfinished at end of df: "
        f"{pending} overlays missing"
    )
    assert head_expected == PATTERN_REPLAY_SUBREG_DIST_HI, (
        f"PATTERN_REPLAY at row {pr_idx} unfinished at end of df: "
        f"awaiting subreg={head_expected} row"
    )
    assert ov_slot_expected == PATTERN_OVERLAY_SUBREG_FRAME_OFFSET, (
        f"PATTERN_OVERLAY triple unfinished at end of df: "
        f"awaiting subreg={ov_slot_expected} row"
    )
    return True


def validate_back_refs(df, prompt_frame_count=0):
    """Walk ``df`` and verify every BACK_REF / PATTERN_REPLAY resolves
    within bounds.
    """
    if "op" not in df.columns:
        return True
    state = _DistancePairState(output_frame_count=prompt_frame_count)
    for idx, row in df.iterrows():
        op = int(row["op"]) if not pd.isna(row["op"]) else SET_OP
        spec = DISTANCE_PAIR_OPS.get(op)
        if spec is not None:
            sr_raw = row.get("subreg", -1)
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            _step_distance_pair(idx, row, sr, op, spec, state)
            continue
        assert state.pending_dist_op is None, (
            f"row {idx}: distance row at row {state.pending_dist_idx} "
            f"(op={state.pending_dist_op}) interrupted with op={op}"
        )
        if op == PATTERN_OVERLAY_OP:
            continue
        if op == DO_LOOP_OP:
            continue
        if row["reg"] in _FRAME_MARKER_REGS:
            state.output_frame_count += 1
    assert state.pending_dist_op is None, (
        f"distance row at row {state.pending_dist_idx} "
        f"(op={state.pending_dist_op}) unfinished at end of df"
    )
    return True
