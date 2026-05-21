"""Macro op infrastructure."""

from collections import defaultdict

import numpy as np
import pandas as pd

from preframr_tokens.macros.blocks import (
    expand_to_literal_form,
    iter_self_contained_row_blocks,
    self_contain_slice,
)
from preframr_tokens.macros.decoders import (
    DECODERS,
    DiffDecoder,
    Flip2Decoder,
    FlipDecoder,
    MacroDecoder,
    SetDecoder,
    SubregFlushDecoder,
    TransposeDecoder,
)
from preframr_tokens.macros.loop_pass import (
    LoopPass,
    _musical_fingerprint,
    _per_frame_state_walk,
)
from preframr_tokens.macros.passes import (
    DedupSetPass,
    Flip2Pass,
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    TransposePass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.local_macros import (
    CtrlBigramPass,
)
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.loops import (
    OVERLAY_BODY_FREQ_DELTA,
    OVERLAY_BODY_FREQ_DELTA_BIN,
    _back_ref_rows,
    _bin_body_freq_delta,
    _FREQ_REGS_VOICED,
    _is_frame_marker_row,
    _pattern_overlay_rows,
    _pattern_replay_rows,
    expand_loops,
)
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    DecodeState,
    FREQ_REGS_BY_VOICE,
    GATE_REGS_BY_VOICE,
    PWM_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
    SUBREG_REGS,
    _BUNDLE_REGS_FLAT,
    _build_decode_state,
    _build_last_diff,
    _df_arrays_and_frames,
    _FastRow,
    _fastrow_from_arrs,
    _FRAME_MARKER_REGS,
    _frame_arrays,
    _GATE_REG_TO_VOICE,
    _PER_VOICE_SUBREG_BASES,
)
from preframr_tokens.macros.validators import (
    validate_back_refs,
    validate_pattern_overlays,
)
from preframr_tokens.macros.walker import FrameWalker
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    DELAY_REG,
    DIFF_OP,
    DO_LOOP_OP,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    FC_LO_REG,
    FILTER_REG,
    FLIP2_OP,
    FLIP_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    LOOP_OP_REG,
    MIN_DIFF,
    MODE_VOL_REG,
    SET_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICES,
    VOICE_REG_SIZE,
)

PASSES = [
    PresetPass(),
    GateSlopeShiftPass(),
    Flip2Pass(),
    TransposePass(),
    DedupSetPass(),
    DedupSetPass(),
    HardRestartPass(),
    LegatoPerClusterPass(),
    CtrlBigramPass(),
    SubregPass(),
]


POST_NORM_PRE_VOICE_PASSES = [
    VoiceBlockOrderPass(),
    LoopPass(),
]


def _maybe_append_coarsen_pass():
    """Coarsen pass is wired in lazily to avoid the circular-import risk
    if coarsen_pass.py ever needs anything from macros.py at module top
    level. Called once on first import of this module via the bottom
    of the file (after PASSES / POST_NORM_PRE_VOICE_PASSES are defined).
    """
    from preframr_tokens.coarsen_pass import CoarsenPass

    POST_NORM_PRE_VOICE_PASSES.append(CoarsenPass())


def run_post_norm_pre_voice_passes(df, args=None):
    """Apply passes that need post-norm row order but pre-voice-rotation
    regs.
    """
    for macro_pass in POST_NORM_PRE_VOICE_PASSES:
        df = macro_pass.apply(df, args=args)
    return df


def run_passes(df, args=None):
    """Apply every PRE-norm-order ``MacroPass`` in order."""
    for macro_pass in PASSES:
        df = macro_pass.apply(df, args=args)
    return df


_maybe_append_coarsen_pass()
