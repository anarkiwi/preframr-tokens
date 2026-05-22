"""Macro op infrastructure."""

from preframr_tokens.macros.blocks import (
    iter_self_contained_row_blocks,
    self_contain_slice,
)
from preframr_tokens.macros.loop_pass import LoopPass
from preframr_tokens.macros.passes import (
    DedupSetPass,
    Flip2Pass,
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    TransposePass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.local_macros import CtrlBigramPass
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.loops import (
    OVERLAY_BODY_FREQ_DELTA,
    OVERLAY_BODY_FREQ_DELTA_BIN,
    _bin_body_freq_delta,
    expand_loops,
)
from preframr_tokens.macros.state import (
    DecodeState,
    _build_decode_state,
    _build_last_diff,
)
from preframr_tokens.macros.validators import (
    validate_back_refs,
    validate_pattern_overlays,
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
