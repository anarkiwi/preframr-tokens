"""Stream-level validators for encoded macro DataFrames."""

import pandas as pd

from preframr_tokens.macros.state import _FRAME_MARKER_REGS
from preframr_tokens.stfconstants import (
    BACK_REF_DIST_HI_SHIFT,
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
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
    output_frame_count = prompt_frame_count
    pending_dist_op = None
    pending_dist_idx = None
    pending_dist_hi = None
    for idx, row in df.iterrows():
        op = int(row["op"]) if not pd.isna(row["op"]) else SET_OP
        if op == BACK_REF_OP:
            sr_raw = row.get("subreg", -1)
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            if sr == BACK_REF_SUBREG_DIST_HI:
                assert pending_dist_op is None, (
                    f"row {idx}: pending {pending_dist_op} at row "
                    f"{pending_dist_idx} not closed before new BACK_REF DIST_HI"
                )
                pending_dist_op = BACK_REF_OP
                pending_dist_idx = idx
                pending_dist_hi = int(row["val"])
                continue
            if sr == BACK_REF_SUBREG_DIST_LO:
                assert pending_dist_op == BACK_REF_OP and pending_dist_hi is not None, (
                    f"row {idx}: BACK_REF DIST_LO without preceding DIST_HI "
                    f"(pending_dist_op={pending_dist_op})"
                )
                distance = (pending_dist_hi << BACK_REF_DIST_HI_SHIFT) | int(row["val"])
                target = output_frame_count - distance
                assert target >= 0, (
                    f"row {idx}: BACK_REF distance={distance} reaches before "
                    f"frame 0 (output_frame_count={output_frame_count})"
                )
                pending_dist_hi = None
                continue
            if sr == BACK_REF_SUBREG_LEN:
                assert pending_dist_op == BACK_REF_OP and pending_dist_hi is None, (
                    f"row {idx}: BACK_REF length without a complete DIST pair "
                    f"(pending_dist_op={pending_dist_op}, hi={pending_dist_hi})"
                )
                length = int(row["val"])
                output_frame_count += length
                pending_dist_op = None
                pending_dist_idx = None
                continue
            raise AssertionError(
                f"row {idx}: BACK_REF subreg={sr} not in "
                f"{{{BACK_REF_SUBREG_DIST_HI}, {BACK_REF_SUBREG_DIST_LO}, "
                f"{BACK_REF_SUBREG_LEN}}}"
            )
        if op == PATTERN_REPLAY_OP:
            sr_raw = row.get("subreg", -1)
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            if sr == PATTERN_REPLAY_SUBREG_DIST_HI:
                assert pending_dist_op is None, (
                    f"row {idx}: pending {pending_dist_op} at row "
                    f"{pending_dist_idx} not closed before new PR DIST_HI"
                )
                pending_dist_op = PATTERN_REPLAY_OP
                pending_dist_idx = idx
                pending_dist_hi = int(row["val"])
                continue
            if sr == PATTERN_REPLAY_SUBREG_DIST_LO:
                assert (
                    pending_dist_op == PATTERN_REPLAY_OP and pending_dist_hi is not None
                ), (
                    f"row {idx}: PR DIST_LO without preceding DIST_HI "
                    f"(pending_dist_op={pending_dist_op})"
                )
                distance = (pending_dist_hi << BACK_REF_DIST_HI_SHIFT) | int(row["val"])
                target = output_frame_count - distance
                assert target >= 0, (
                    f"row {idx}: PATTERN_REPLAY distance={distance} reaches "
                    f"before frame 0 (output_frame_count={output_frame_count})"
                )
                pending_dist_hi = None
                continue
            if sr == PATTERN_REPLAY_SUBREG_LEN:
                assert (
                    pending_dist_op == PATTERN_REPLAY_OP and pending_dist_hi is None
                ), (
                    f"row {idx}: PR length without a complete DIST pair "
                    f"(pending_dist_op={pending_dist_op}, hi={pending_dist_hi})"
                )
                length = int(row["val"])
                output_frame_count += length
                pending_dist_op = None
                pending_dist_idx = None
                continue
            if sr == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT:
                continue
            raise AssertionError(
                f"row {idx}: PATTERN_REPLAY subreg={sr} not in "
                f"{{{PATTERN_REPLAY_SUBREG_DIST_HI}, "
                f"{PATTERN_REPLAY_SUBREG_DIST_LO}, "
                f"{PATTERN_REPLAY_SUBREG_LEN}, "
                f"{PATTERN_REPLAY_SUBREG_OVERLAY_COUNT}}}"
            )
        assert pending_dist_op is None, (
            f"row {idx}: distance row at row {pending_dist_idx} "
            f"(op={pending_dist_op}) interrupted with op={op}"
        )
        if op == PATTERN_OVERLAY_OP:
            continue
        if op == DO_LOOP_OP:
            continue
        if row["reg"] in _FRAME_MARKER_REGS:
            output_frame_count += 1
    assert pending_dist_op is None, (
        f"distance row at row {pending_dist_idx} (op={pending_dist_op}) "
        f"unfinished at end of df"
    )
    return True
