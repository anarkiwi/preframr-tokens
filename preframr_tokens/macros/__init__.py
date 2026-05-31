"""Macro op infrastructure."""

from preframr_tokens.coarsen_pass import CoarsenPass
from preframr_tokens.macros.blocks import (
    iter_self_contained_row_blocks,
    self_contain_slice,
)
from preframr_tokens.macros.loop_pass import LoopPass
from preframr_tokens.macros.passes import (
    DedupSetPass,
    HardRestartPass,
    LegatoPerClusterPass,
    SubregPass,
    TransposePass,
    VoiceBlockOrderPass,
)
from preframr_tokens.macros.ctrl_triple_pass import CtrlTriplePass
from preframr_tokens.macros.freq_onset_pass import FreqOnsetPass
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.macros.local_macros import CtrlBigramPass
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.release_update_pass import ReleaseUpdatePass
from preframr_tokens.macros.patch_pass import PatchPass
from preframr_tokens.macros.skeleton_pass import SkeletonPass
from preframr_tokens.macros.stamp_pass import StampPass
from preframr_tokens.macros.sweep_pass import SweepPass
from preframr_tokens.macros.trajectory_anchor import TrajectoryAnchorPass
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
    validate_stream,
)

FREQ_BLOCK_PASSES = [
    TrajectoryAnchorPass(),
    StampPass(),
    SweepPass(),
    SkeletonPass(),
    FreqTrajectoryPass(),
    FreqOnsetPass(),
    PerRegBurstPass(),
    PatchPass(),
    ReleaseUpdatePass(),
]


PASSES = [
    PresetPass(),
    GateSlopeShiftPass(),
    TransposePass(),
    DedupSetPass(),
    DedupSetPass(),
    HardRestartPass(),
    LegatoPerClusterPass(),
    CtrlTriplePass(),
    CtrlBigramPass(),
    SubregPass(),
]


def run_freq_block_passes(df, args=None):
    """Freq-encoder passes (TrajectoryAnchor / FreqTrajectory / FreqOnset / PerRegBurst /
    ReleaseUpdate) that produce op45/47/48/49 atoms from literal SETs. Run once at the
    start of each self-contained block (after ``expand_to_literal_form`` decompiles
    everything) and once at parse-time before rotation -- kept out of ``PASSES`` so
    they don't re-fire on already-encoded atoms inside the rotation loop."""
    for macro_pass in FREQ_BLOCK_PASSES:
        df = macro_pass.apply(df, args=args)
    return df


POST_NORM_PRE_VOICE_PASSES = [
    VoiceBlockOrderPass(),
    LoopPass(),
    CoarsenPass(),
]


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
