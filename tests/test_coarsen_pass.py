"""Unit tests for preframr.coarsen_pass."""

import argparse

import pandas as pd

from preframr_tokens.coarsen_pass import (
    CoarsenPass,
    coarsen_pass,
)
from preframr_tokens.macros import OVERLAY_BODY_FREQ_DELTA, expand_loops
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    FRAME_REG,
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


def _row(*, op, reg, val, subreg=-1, diff=32):
    return {"op": op, "reg": reg, "val": val, "subreg": subreg, "diff": diff}


def _br(distance, length, diff=32):
    return [
        _row(
            op=BACK_REF_OP,
            reg=LOOP_OP_REG,
            val=(int(distance) >> 8) & 0xFF,
            subreg=BACK_REF_SUBREG_DIST_HI,
            diff=diff,
        ),
        _row(
            op=BACK_REF_OP,
            reg=LOOP_OP_REG,
            val=int(distance) & 0xFF,
            subreg=BACK_REF_SUBREG_DIST_LO,
            diff=diff,
        ),
        _row(
            op=BACK_REF_OP,
            reg=LOOP_OP_REG,
            val=int(length),
            subreg=BACK_REF_SUBREG_LEN,
            diff=diff,
        ),
    ]


def _pr(distance, length, ov_count=0, diff=32):
    return [
        _row(
            op=PATTERN_REPLAY_OP,
            reg=LOOP_OP_REG,
            val=(int(distance) >> 8) & 0xFF,
            subreg=PATTERN_REPLAY_SUBREG_DIST_HI,
            diff=diff,
        ),
        _row(
            op=PATTERN_REPLAY_OP,
            reg=LOOP_OP_REG,
            val=int(distance) & 0xFF,
            subreg=PATTERN_REPLAY_SUBREG_DIST_LO,
            diff=diff,
        ),
        _row(
            op=PATTERN_REPLAY_OP,
            reg=LOOP_OP_REG,
            val=int(length),
            subreg=PATTERN_REPLAY_SUBREG_LEN,
            diff=diff,
        ),
        _row(
            op=PATTERN_REPLAY_OP,
            reg=LOOP_OP_REG,
            val=int(ov_count),
            subreg=PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
            diff=diff,
        ),
    ]


def _frame_marker():
    return _row(op=SET_OP, reg=FRAME_REG, val=0, subreg=-1, diff=19656)


def _set(reg, val, subreg=-1):
    return _row(op=SET_OP, reg=reg, val=val, subreg=subreg)


def _df(rows):
    df = pd.DataFrame(rows)
    df["op"] = df["op"].astype("UInt8")
    df["reg"] = df["reg"].astype("Int8")
    df["val"] = df["val"].astype("Int32")
    df["subreg"] = df["subreg"].astype("int64")
    df["diff"] = df["diff"].astype("UInt16")
    return df


def test_no_op_when_no_macros():
    rows = [
        _frame_marker(),
        _set(0, 100),
        _set(4, 0x21),
        _frame_marker(),
        _set(4, 0x20),
    ]
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    pd.testing.assert_frame_equal(df, out)


def test_default_off_via_class_adapter():
    """CoarsenPass.apply with no args => no-op (args is None)."""
    rows = [
        _frame_marker(),
        _set(0, 100),
        *_br(distance=1, length=1),
    ]
    df = _df(rows)
    out = CoarsenPass().apply(df, args=None)
    pd.testing.assert_frame_equal(df, out)


def test_off_when_arg_false():
    """CoarsenPass with args.coarsen_pass=False => no-op."""
    rows = [
        _frame_marker(),
        _set(0, 100),
        *_br(distance=1, length=1),
    ]
    df = _df(rows)
    args = argparse.Namespace(coarsen_pass=False)
    out = CoarsenPass().apply(df, args=args)
    pd.testing.assert_frame_equal(df, out)


def test_short_back_ref_materialised():
    """A 2-frame BR (length=2) with min_coarse_len=16 should
    materialise inline. Result should match what expand_loops would
    produce for the same input.
    """
    rows = [
        _frame_marker(),
        _set(0, 100),
        _set(4, 0x21),
        _frame_marker(),
        _set(4, 0x20),
        *_br(distance=2, length=2),
    ]
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    expected = expand_loops(df.copy()).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


def test_long_back_ref_preserved():
    """A 16-frame BR with min_coarse_len=16 should be kept as-is
    (BR triple survives unchanged).
    """
    rows = [_frame_marker()]
    for _ in range(15):
        rows.append(_frame_marker())
    rows.extend(_br(distance=16, length=16))
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    assert len(out) == len(df)
    assert out.iloc[-3]["op"] == BACK_REF_OP
    assert out.iloc[-3]["subreg"] == BACK_REF_SUBREG_DIST_HI
    assert out.iloc[-2]["op"] == BACK_REF_OP
    assert out.iloc[-2]["subreg"] == BACK_REF_SUBREG_DIST_LO
    assert out.iloc[-1]["op"] == BACK_REF_OP
    assert out.iloc[-1]["subreg"] == BACK_REF_SUBREG_LEN


def test_short_pattern_replay_materialised():
    """PR(dist=2, len=2, ov_count=0) with min_coarse_len=16 → materialise."""
    rows = [
        _frame_marker(),
        _set(0, 100),
        _frame_marker(),
        _set(4, 0x20),
        *_pr(distance=2, length=2, ov_count=0),
    ]
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    expected = expand_loops(df.copy()).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


def test_short_pattern_replay_with_body_freq_delta():
    """Body-wide freq delta overlay should apply during materialisation."""
    rows = [
        _frame_marker(),
        _set(0, 100),
        _frame_marker(),
        _set(4, 0x20),
        *_pr(distance=2, length=2, ov_count=1),
        _row(
            op=PATTERN_OVERLAY_OP,
            reg=LOOP_OP_REG,
            val=-1,
            subreg=PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
        ),
        _row(
            op=PATTERN_OVERLAY_OP,
            reg=LOOP_OP_REG,
            val=OVERLAY_BODY_FREQ_DELTA,
            subreg=PATTERN_OVERLAY_SUBREG_TARGET_REG,
        ),
        _row(
            op=PATTERN_OVERLAY_OP,
            reg=LOOP_OP_REG,
            val=200,
            subreg=PATTERN_OVERLAY_SUBREG_NEW_VAL,
        ),
    ]
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    expected = expand_loops(df.copy()).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


def test_long_pattern_replay_preserved():
    """PR(len=16) with min_coarse_len=16 → preserved (head quad survives)."""
    rows = []
    for _ in range(16):
        rows.append(_frame_marker())
    rows.extend(_pr(distance=16, length=16, ov_count=0))
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    assert len(out) == len(df)
    assert out.iloc[-4]["op"] == PATTERN_REPLAY_OP
    assert out.iloc[-4]["subreg"] == PATTERN_REPLAY_SUBREG_DIST_HI
    assert out.iloc[-1]["op"] == PATTERN_REPLAY_OP
    assert out.iloc[-1]["subreg"] == PATTERN_REPLAY_SUBREG_OVERLAY_COUNT


def test_mixed_short_and_long_brs():
    """A short BR and a long BR in the same df → short materialises,
    long survives."""
    rows = [_frame_marker()]
    for _ in range(20):
        rows.append(_frame_marker())
    rows.extend(_br(distance=2, length=2))
    rows.extend(_br(distance=20, length=16))
    df = _df(rows)
    out = coarsen_pass(df, min_coarse_len=16)
    br_rows = out[out["op"] == BACK_REF_OP]
    assert len(br_rows) == 3
    assert int(br_rows.iloc[2]["val"]) == 16
