"""PATTERN_REPLAY / DO_LOOP machinery."""

import logging
from collections import defaultdict

import pandas as pd

__all__ = ["expand_loops", "OVERLAY_BODY_FREQ_DELTA", "OVERLAY_BODY_FREQ_DELTA_BIN"]

_logger = logging.getLogger(__name__)

from preframr_tokens.macros.state import _FRAME_MARKER_REGS, FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    BACK_REF_DIST_HI_SHIFT,
    BACK_REF_DIST_LO_MASK,
    DO_LOOP_OP,
    LOOP_OP_REG,
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


def _pattern_replay_rows(distance, length, overlay_count, diff_default, irq_default):
    assert 1 <= length <= 255, length
    assert distance >= 1, distance
    assert 0 <= overlay_count, overlay_count
    dist_hi = (int(distance) >> BACK_REF_DIST_HI_SHIFT) & BACK_REF_DIST_LO_MASK
    dist_lo = int(distance) & BACK_REF_DIST_LO_MASK
    rows = [
        {
            "reg": int(LOOP_OP_REG),
            "val": dist_hi,
            "diff": diff_default,
            "op": int(PATTERN_REPLAY_OP),
            "subreg": int(PATTERN_REPLAY_SUBREG_DIST_HI),
            "irq": irq_default,
            "description": 0,
        },
        {
            "reg": int(LOOP_OP_REG),
            "val": dist_lo,
            "diff": diff_default,
            "op": int(PATTERN_REPLAY_OP),
            "subreg": int(PATTERN_REPLAY_SUBREG_DIST_LO),
            "irq": irq_default,
            "description": 0,
        },
        {
            "reg": int(LOOP_OP_REG),
            "val": int(length),
            "diff": diff_default,
            "op": int(PATTERN_REPLAY_OP),
            "subreg": int(PATTERN_REPLAY_SUBREG_LEN),
            "irq": irq_default,
            "description": 0,
        },
    ]
    if int(overlay_count) > 0:
        rows.append(
            {
                "reg": int(LOOP_OP_REG),
                "val": int(overlay_count),
                "diff": diff_default,
                "op": int(PATTERN_REPLAY_OP),
                "subreg": int(PATTERN_REPLAY_SUBREG_OVERLAY_COUNT),
                "irq": irq_default,
                "description": 0,
            }
        )
    return rows


OVERLAY_BODY_FREQ_DELTA = 0xFE

OVERLAY_BODY_FREQ_DELTA_BIN = 16


def _bin_body_freq_delta(delta):
    """Quantize a body-wide freq delta to OVERLAY_BODY_FREQ_DELTA_BIN
    cents. Round-to-nearest with ties going to even (numpy default);
    keeps the sign of the input. The encoder applies this before
    emitting the OV new_val row so the train alphabet's val space is
    bounded by ``ceil(range / bin) + 1`` distinct entries."""
    bin_w = int(OVERLAY_BODY_FREQ_DELTA_BIN)
    if bin_w <= 1:
        return int(delta)
    d = int(delta)
    if d >= 0:
        return ((d + bin_w // 2) // bin_w) * bin_w
    return -(((-d) + bin_w // 2) // bin_w) * bin_w


def _pattern_overlay_rows(frame_offset, target_reg, new_val, diff_default, irq_default):
    """Emit one PATTERN_OVERLAY as a triple of atomic rows."""
    return [
        {
            "reg": int(LOOP_OP_REG),
            "val": int(frame_offset),
            "diff": diff_default,
            "op": int(PATTERN_OVERLAY_OP),
            "subreg": int(PATTERN_OVERLAY_SUBREG_FRAME_OFFSET),
            "irq": irq_default,
            "description": 0,
        },
        {
            "reg": int(LOOP_OP_REG),
            "val": int(target_reg),
            "diff": diff_default,
            "op": int(PATTERN_OVERLAY_OP),
            "subreg": int(PATTERN_OVERLAY_SUBREG_TARGET_REG),
            "irq": irq_default,
            "description": 0,
        },
        {
            "reg": int(LOOP_OP_REG),
            "val": int(new_val),
            "diff": diff_default,
            "op": int(PATTERN_OVERLAY_OP),
            "subreg": int(PATTERN_OVERLAY_SUBREG_NEW_VAL),
            "irq": irq_default,
            "description": 0,
        },
    ]


MULTI_ROW_MACRO_EMITTERS = (
    (_pattern_replay_rows, PATTERN_REPLAY_OP, PATTERN_REPLAY_SUBREG_DIST_HI),
    (_pattern_overlay_rows, PATTERN_OVERLAY_OP, PATTERN_OVERLAY_SUBREG_FRAME_OFFSET),
)

MULTI_ROW_MACRO_HEAD_OPS = tuple((op, sr) for _fn, op, sr in MULTI_ROW_MACRO_EMITTERS)

EXTRA_ISOLATION_HEAD_OPS: tuple[tuple[int, int], ...] = ()


_FREQ_REGS_VOICED = frozenset(FREQ_REGS_BY_VOICE)


def _is_frame_marker_row(row):
    return row[0] in _FRAME_MARKER_REGS


def expand_loops(df):
    """Materialize PATTERN_REPLAY and DO_LOOP rows into literal frame copies."""
    if "op" not in df.columns:
        return df
    has_loops = df["op"].isin([DO_LOOP_OP, PATTERN_REPLAY_OP]).any()
    if not has_loops:
        return df

    cols = list(df.columns)
    out = []
    output_frame_starts = []
    do_stack = []
    orphans = defaultdict(int)

    col_arrays = {c: df[c].to_numpy() for c in cols}
    op_arr = col_arrays["op"]
    val_arr = col_arrays["val"]
    subreg_arr = col_arrays["subreg"] if "subreg" in cols else None

    def _row_to_dict(i):
        return {c: col_arrays[c][i] for c in cols}

    def append_row(row_dict):
        out.append(row_dict)
        if row_dict["reg"] in _FRAME_MARKER_REGS:
            output_frame_starts.append(len(out) - 1)

    n = len(df)
    i = 0
    while i < n:
        op_raw = op_arr[i]
        op = int(op_raw) if not pd.isna(op_raw) else SET_OP
        if op == DO_LOOP_OP:
            subreg_raw = subreg_arr[i] if subreg_arr is not None else -1
            subreg = int(subreg_raw) if not pd.isna(subreg_raw) else -1
            if subreg == 0:
                n_iter = int(val_arr[i])
                assert n_iter >= 1, n_iter
                do_stack.append([i + 1, n_iter - 1])
                i += 1
                continue
            if do_stack and do_stack[-1][1] > 0:
                body_start, remaining = do_stack[-1]
                do_stack[-1][1] = remaining - 1
                i = body_start
            else:
                if do_stack:
                    do_stack.pop()
                i += 1
            continue
        if op == PATTERN_REPLAY_OP:
            sr_raw = subreg_arr[i] if subreg_arr is not None else -1
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            if sr != PATTERN_REPLAY_SUBREG_DIST_HI:
                orphans["pr_continuation_without_dist_hi"] += 1
                i += 1
                continue
            dist_hi = int(val_arr[i])
            if i + 2 >= n:
                orphans["pr_dist_hi_at_eof"] += 1
                i += 1
                continue
            lo_op_raw = op_arr[i + 1]
            lo_op = int(lo_op_raw) if not pd.isna(lo_op_raw) else SET_OP
            lo_sr_raw = subreg_arr[i + 1] if subreg_arr is not None else -1
            lo_sr = int(lo_sr_raw) if not pd.isna(lo_sr_raw) else -1
            if not (
                lo_op == PATTERN_REPLAY_OP and lo_sr == PATTERN_REPLAY_SUBREG_DIST_LO
            ):
                orphans["pr_dist_hi_no_lo_partner"] += 1
                i += 1
                continue
            dist_lo = int(val_arr[i + 1])
            distance = (dist_hi << BACK_REF_DIST_HI_SHIFT) | dist_lo
            len_op_raw = op_arr[i + 2]
            len_op = int(len_op_raw) if not pd.isna(len_op_raw) else SET_OP
            len_sr_raw = subreg_arr[i + 2] if subreg_arr is not None else -1
            len_sr = int(len_sr_raw) if not pd.isna(len_sr_raw) else -1
            if not (
                len_op == PATTERN_REPLAY_OP and len_sr == PATTERN_REPLAY_SUBREG_LEN
            ):
                orphans["pr_dist_no_len_partner"] += 1
                i += 1
                continue
            length = int(val_arr[i + 2])
            num_overlays = 0
            head_rows = 3
            if i + 3 < n:
                ov_op_raw = op_arr[i + 3]
                ov_op_field = int(ov_op_raw) if not pd.isna(ov_op_raw) else SET_OP
                ov_sr_raw = subreg_arr[i + 3] if subreg_arr is not None else -1
                ov_sr_field = int(ov_sr_raw) if not pd.isna(ov_sr_raw) else -1
                if (
                    ov_op_field == PATTERN_REPLAY_OP
                    and ov_sr_field == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT
                ):
                    num_overlays = int(val_arr[i + 3])
                    if num_overlays < 0:
                        orphans["pr_negative_overlay_count"] += 1
                        i += 1
                        continue
                    head_rows = 4
            cur_frame = len(output_frame_starts)
            target = cur_frame - distance
            assert target >= 0, (
                f"PATTERN_REPLAY target frame {target} reaches before output "
                f"start (cur_frame={cur_frame}, distance={distance})"
            )
            assert target + length <= cur_frame, (
                f"PATTERN_REPLAY target range [{target},{target+length}) "
                f"overlaps present frame {cur_frame}"
            )
            overlays = []
            body_freq_delta = 0
            for k in range(num_overlays):
                base_idx = i + head_rows + k * 3
                fo_idx, tr_idx, nv_idx = base_idx, base_idx + 1, base_idx + 2
                for slot_idx, expected_sr, slot_label in (
                    (fo_idx, PATTERN_OVERLAY_SUBREG_FRAME_OFFSET, "frame_offset"),
                    (tr_idx, PATTERN_OVERLAY_SUBREG_TARGET_REG, "target_reg"),
                    (nv_idx, PATTERN_OVERLAY_SUBREG_NEW_VAL, "new_val"),
                ):
                    op_raw = op_arr[slot_idx]
                    op_v = int(op_raw) if not pd.isna(op_raw) else SET_OP
                    sr_raw = subreg_arr[slot_idx] if subreg_arr is not None else -1
                    sr_v = int(sr_raw) if not pd.isna(sr_raw) else -1
                    assert op_v == PATTERN_OVERLAY_OP and sr_v == expected_sr, (
                        f"PATTERN_REPLAY at row {i} overlay {k}: expected "
                        f"{slot_label} row at {slot_idx} (op="
                        f"{PATTERN_OVERLAY_OP} subreg={expected_sr}), got "
                        f"op={op_v} subreg={sr_v}"
                    )
                frame_offset = int(val_arr[fo_idx])
                target_reg = int(val_arr[tr_idx])
                new_val = int(val_arr[nv_idx])
                if frame_offset < 0 and target_reg == OVERLAY_BODY_FREQ_DELTA:
                    body_freq_delta = new_val
                    continue
                overlays.append((frame_offset, target_reg, new_val))
            ov_by_frame = defaultdict(list)
            for fo, r, v in overlays:
                ov_by_frame[fo].append((r, v))
            for f in range(target, target + length):
                src_lo = output_frame_starts[f]
                src_hi = (
                    output_frame_starts[f + 1]
                    if f + 1 < len(output_frame_starts)
                    else len(out)
                )
                snapshot = list(out[src_lo:src_hi])
                for snap_row in snapshot:
                    new_row = dict(snap_row)
                    if (
                        body_freq_delta
                        and int(new_row.get("reg", -1)) in _FREQ_REGS_VOICED
                        and int(new_row.get("op", SET_OP)) == SET_OP
                        and int(new_row.get("subreg", -1)) == -1
                    ):
                        new_row["val"] = int(new_row["val"]) + body_freq_delta
                    append_row(new_row)
                frame_offset = f - target
                for r, v in ov_by_frame.get(frame_offset, ()):
                    template = dict(snapshot[0]) if snapshot else {}
                    template.update(
                        {
                            "reg": int(r),
                            "val": int(v),
                            "op": int(SET_OP),
                            "subreg": -1,
                        }
                    )
                    append_row(template)
            i += head_rows + num_overlays * 3
            continue
        if op == PATTERN_OVERLAY_OP:
            raise AssertionError(f"orphan PATTERN_OVERLAY_OP at row {i}")
        append_row(_row_to_dict(i))
        i += 1

    if orphans:
        _logger.warning(
            "expand_loops orphans dropped: %s (n_rows=%d)",
            dict(orphans),
            n,
        )
    if not out:
        return df.iloc[0:0]
    expanded = pd.DataFrame(out, columns=cols)
    for col, dt in df.dtypes.items():
        try:
            expanded[col] = expanded[col].astype(dt)
        except (TypeError, ValueError):
            pass
    expanded = expanded.reset_index(drop=True)
    if orphans:
        expanded.attrs["_orphans"] = dict(orphans)
    return expanded
