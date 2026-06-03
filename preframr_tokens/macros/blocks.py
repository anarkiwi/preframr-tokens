"""Self-contained block extraction for training and inference paths."""

__all__ = [
    "expand_to_literal_form",
    "self_contain_slice",
    "iter_self_contained_row_blocks",
]

import logging

import numpy as np

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.stfconstants import (
    DELAY_REG,
    DESCRIPTION_PDTYPE,
    DIFF_PDTYPE,
    FRAME_REG,
    IRQ_PDTYPE,
    OP_PDTYPE,
    REG_PDTYPE,
    SET_OP,
    SUBREG_PDTYPE,
    VAL_PDTYPE,
)

_logger = logging.getLogger(__name__)

_FAST_INT_COLS = ("reg", "val", "diff", "irq", "op", "subreg", "description")
_CANONICAL_DTYPES = {
    "reg": REG_PDTYPE,
    "val": VAL_PDTYPE,
    "diff": DIFF_PDTYPE,
    "irq": IRQ_PDTYPE,
    "op": OP_PDTYPE,
    "subreg": SUBREG_PDTYPE,
    "description": DESCRIPTION_PDTYPE,
}


def _to_fast_int(df):
    """Cast the always-populated int columns to plain int64 numpy. The per-block re-fire passes
    never see NA in these columns (corpus NA audit: 3270/3270 block-pipeline dfs NA-free), so the
    canonical nullable Int* dtypes are pure per-cell boxing (maybe_box_native / masked.__iter__) on
    every to_dict / itertuples in the pass loop; canonical dtypes are restored at the block boundary
    so the emitted block stays byte-identical for tokenisation."""
    cast = {c: np.int64 for c in _FAST_INT_COLS if c in df.columns}
    return df.astype(cast) if cast else df


def _to_canonical_int(df):
    """Restore the canonical nullable Int* dtypes on the columns present."""
    cast = {c: dt for c, dt in _CANONICAL_DTYPES.items() if c in df.columns}
    return df.astype(cast) if cast else df


def expand_to_literal_form(df):
    """Fully expand all macros in ``df`` to literal SET rows."""
    saved_attrs = df.attrs
    df.attrs = {}
    try:
        df_in = df.copy()
    finally:
        df.attrs = saved_attrs
    if "description" not in df_in.columns:
        df_in["description"] = 0
    literal = expand_ops(df_in, strict=False)
    if "op" not in literal.columns:
        literal["op"] = int(SET_OP)
    else:
        literal["op"] = literal["op"].fillna(int(SET_OP)).astype(int)
    if "subreg" not in literal.columns:
        literal["subreg"] = -1
    return literal.reset_index(drop=True)


def self_contain_slice(df, slice_lo_frame, slice_hi_frame, args=None):
    """Materialise a single ``[slice_lo_frame, slice_hi_frame)`` slice
    into a self-contained row DataFrame.
    """
    from preframr_tokens.macros import run_passes

    literal = expand_to_literal_form(df)
    is_marker = literal["reg"].isin({FRAME_REG, DELAY_REG})
    marker_idx = literal.index[is_marker].tolist()
    if slice_lo_frame >= len(marker_idx):
        return literal.iloc[0:0].reset_index(drop=True).copy()
    row_lo = int(marker_idx[slice_lo_frame])
    row_hi = (
        int(marker_idx[slice_hi_frame])
        if slice_hi_frame < len(marker_idx)
        else len(literal)
    )
    slice_df = literal.iloc[row_lo:row_hi].reset_index(drop=True).copy()
    if args is None:
        return slice_df
    return run_passes(slice_df, args=args)


def iter_self_contained_row_blocks(df, frames_per_block, args=None, stride=None):
    """Yield row-DataFrames each covering ``frames_per_block`` logical
    frame slots (= FRAME_REG/DELAY_REG markers) of ``df``. Every block
    has its out-of-block references (PATTERN_REPLAY_OP, GATE_REPLAY_OP,
    PLAY_INSTRUMENT_OP, DO_LOOP_OP) rewritten to literals so the block
    can be tokenized and decoded standalone.
    """
    from preframr_tokens.macros import (
        run_block_refire_passes,
        run_post_norm_pre_voice_passes,
    )

    if stride is None or stride < 1:
        stride = frames_per_block

    if "op" not in df.columns:
        is_marker = df["reg"].isin({FRAME_REG, DELAY_REG})
        marker_idx = df.index[is_marker].tolist()
        if not marker_idx:
            yield df.reset_index(drop=True).copy()
            return
        n_frames = len(marker_idx)
        for lo in range(0, n_frames, stride):
            hi = min(lo + frames_per_block, n_frames)
            row_lo = marker_idx[lo]
            row_hi = marker_idx[hi] if hi < n_frames else len(df)
            yield df.iloc[row_lo:row_hi].reset_index(drop=True).copy()
        return

    is_marker = df["reg"].isin({FRAME_REG, DELAY_REG})
    marker_count = int(is_marker.sum())
    if marker_count == 0:
        yield df.reset_index(drop=True).copy()
        return

    literal = expand_to_literal_form(df)
    literal.attrs.clear()
    literal = _to_fast_int(literal)
    lit_is_marker = literal["reg"].isin({FRAME_REG, DELAY_REG})
    marker_idx = literal.index[lit_is_marker].tolist()
    n_lit_frames = len(marker_idx)

    from preframr_tokens.reglogparser import RegLogParser

    consolidator = RegLogParser(args)
    for lo_frame in range(0, marker_count, stride):
        if lo_frame >= n_lit_frames:
            break
        hi_frame = min(lo_frame + frames_per_block, n_lit_frames)
        row_lo = int(marker_idx[lo_frame])
        row_hi = int(marker_idx[hi_frame]) if hi_frame < n_lit_frames else len(literal)
        slice_df = literal.iloc[row_lo:row_hi].reset_index(drop=True).copy()
        if slice_df.empty:
            continue
        if args is not None:
            block = run_block_refire_passes(slice_df, args=args)
            block = consolidator._norm_pr_order(block)
            block = run_post_norm_pre_voice_passes(block, args=args)
        else:
            block = slice_df
        if block.empty:
            continue
        block.attrs.clear()
        try:
            block = consolidator._consolidate_frames(block)
        except Exception as e:  # pylint: disable=broad-except
            _logger.warning(
                "_consolidate_frames failed (block rows=%d, lo_frame=%d, "
                "hi_frame=%d); block will ship with inflated FRAME_REG "
                "spam: %s",
                len(block),
                lo_frame,
                hi_frame,
                e,
            )
        yield _to_canonical_int(block)
