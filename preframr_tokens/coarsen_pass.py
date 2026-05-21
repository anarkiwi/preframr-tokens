"""PatternCoarseningPass — selective materialisation of short-range
BR / PR invocations.
"""

from collections import defaultdict

import pandas as pd

from preframr_tokens.stfconstants import (
    BACK_REF_DIST_HI_SHIFT,
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    DELAY_REG,
    DO_LOOP_OP,
    FRAME_REG,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    SET_OP,
    VOICE_REG_SIZE,
    VOICES,
)

DEFAULT_MIN_COARSE_LEN = 16

_FRAME_MARKER_REGS = {FRAME_REG, DELAY_REG}

_FREQ_REGS_VOICED = frozenset(v * VOICE_REG_SIZE + 0 for v in range(VOICES))


def _materialise_back_ref(out, output_frame_starts, val_arr, i, append_row):
    """Materialise the BR (HI, LO, LEN) triple at rows ``i..i+2`` inline."""
    dist_hi = int(val_arr[i])
    dist_lo = int(val_arr[i + 1])
    distance = (dist_hi << BACK_REF_DIST_HI_SHIFT) | dist_lo
    length = int(val_arr[i + 2])
    cur_frame = len(output_frame_starts)
    target = cur_frame - distance
    if target < 0 or target + length > cur_frame:
        return None
    for f in range(target, target + length):
        src_lo = output_frame_starts[f]
        src_hi = (
            output_frame_starts[f + 1] if f + 1 < len(output_frame_starts) else len(out)
        )
        for snap_row in list(out[src_lo:src_hi]):
            append_row(dict(snap_row))
    return i + 3


def _materialise_pattern_replay(out, output_frame_starts, val_arr, i, append_row):
    """Materialise the PR (HI, LO, LEN, OV_COUNT, [overlays]) block at row ``i``."""
    from preframr_tokens.macros import OVERLAY_BODY_FREQ_DELTA

    dist_hi = int(val_arr[i])
    dist_lo = int(val_arr[i + 1])
    distance = (dist_hi << BACK_REF_DIST_HI_SHIFT) | dist_lo
    length = int(val_arr[i + 2])
    num_overlays = int(val_arr[i + 3])
    head_rows = 4
    cur_frame = len(output_frame_starts)
    target = cur_frame - distance
    if target < 0 or target + length > cur_frame:
        return None

    overlays = []
    body_freq_delta = 0
    for k in range(num_overlays):
        base_idx = i + head_rows + k * 3
        fo = int(val_arr[base_idx])
        tr = int(val_arr[base_idx + 1])
        nv = int(val_arr[base_idx + 2])
        if fo < 0 and tr == OVERLAY_BODY_FREQ_DELTA:
            body_freq_delta = nv
            continue
        overlays.append((fo, tr, nv))

    ov_by_frame = defaultdict(list)
    for fo, r, v in overlays:
        ov_by_frame[fo].append((r, v))

    for f in range(target, target + length):
        src_lo = output_frame_starts[f]
        src_hi = (
            output_frame_starts[f + 1] if f + 1 < len(output_frame_starts) else len(out)
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
                {"reg": int(r), "val": int(v), "op": int(SET_OP), "subreg": -1}
            )
            append_row(template)
    return i + head_rows + num_overlays * 3


def coarsen_pass(df, min_coarse_len=DEFAULT_MIN_COARSE_LEN):
    """Selectively materialise short-range BR / PR invocations in df."""
    if "op" not in df.columns:
        return df
    has_loops = df["op"].isin([BACK_REF_OP, PATTERN_REPLAY_OP, DO_LOOP_OP]).any()
    if not has_loops:
        return df

    cols = list(df.columns)
    col_arrays = {c: df[c].to_numpy() for c in cols}
    op_arr = col_arrays["op"]
    val_arr = col_arrays["val"]
    subreg_arr = col_arrays["subreg"] if "subreg" in cols else None
    n = len(df)

    out = []
    output_frame_starts = []

    def _row_to_dict(i):
        return {c: col_arrays[c][i] for c in cols}

    def append_row(row_dict):
        out.append(row_dict)
        if row_dict["reg"] in _FRAME_MARKER_REGS:
            output_frame_starts.append(len(out) - 1)

    i = 0
    while i < n:
        op_raw = op_arr[i]
        op = int(op_raw) if not pd.isna(op_raw) else SET_OP

        if op == BACK_REF_OP:
            sr_raw = subreg_arr[i] if subreg_arr is not None else -1
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            assert sr == BACK_REF_SUBREG_DIST_HI, (i, sr)
            length = int(val_arr[i + 2])
            if length < min_coarse_len:
                ni = _materialise_back_ref(
                    out, output_frame_starts, val_arr, i, append_row
                )
                if ni is not None:
                    i = ni
                    continue
            append_row(_row_to_dict(i))
            append_row(_row_to_dict(i + 1))
            append_row(_row_to_dict(i + 2))
            i += 3
            continue

        if op == PATTERN_REPLAY_OP:
            sr_raw = subreg_arr[i] if subreg_arr is not None else -1
            sr = int(sr_raw) if not pd.isna(sr_raw) else -1
            assert sr == PATTERN_REPLAY_SUBREG_DIST_HI, (i, sr)
            length = int(val_arr[i + 2])
            num_overlays = int(val_arr[i + 3])
            head_and_overlays = 4 + num_overlays * 3
            if length < min_coarse_len:
                ni = _materialise_pattern_replay(
                    out,
                    output_frame_starts,
                    val_arr,
                    i,
                    append_row,
                )
                if ni is not None:
                    i = ni
                    continue
            for j in range(i, i + head_and_overlays):
                append_row(_row_to_dict(j))
            i += head_and_overlays
            continue

        if op == DO_LOOP_OP:
            append_row(_row_to_dict(i))
            i += 1
            continue

        append_row(_row_to_dict(i))
        i += 1

    expanded = pd.DataFrame(out, columns=cols)
    for col, dt in df.dtypes.items():
        try:
            expanded[col] = expanded[col].astype(dt)
        except (TypeError, ValueError):
            pass
    expanded = expanded.reset_index(drop=True)
    if df.attrs:
        expanded.attrs.update(df.attrs)
    return expanded


class CoarsenPass:
    """Wrapper to fit ``run_post_norm_pre_voice_passes`` interface."""

    # pylint: disable=unused-argument
    def apply(self, df, args=None):
        if args is None:
            return df
        if not getattr(args, "coarsen_pass", False):
            return df
        min_len = getattr(args, "coarsen_min_len", DEFAULT_MIN_COARSE_LEN)
        return coarsen_pass(df, min_coarse_len=min_len)
