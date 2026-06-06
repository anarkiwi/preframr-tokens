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
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.macros.instrument_program_pass import InstrumentProgramPass
from preframr_tokens.macros.generator_pass import GeneratorPass
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
    codebook_live_ids,
    validate_back_refs,
    validate_codebook_refs,
    validate_pattern_overlays,
    validate_stream,
)

FREQ_BLOCK_PASSES = [
    PerRegBurstPass(),
    InstrumentProgramPass(),
    GeneratorPass(),
]


PASSES = [
    PresetPass(),
    GateSlopeShiftPass(),
    TransposePass(),
    DedupSetPass(),
    DedupSetPass(),
    HardRestartPass(),
    LegatoPerClusterPass(),
    SubregPass(),
]


def run_freq_block_passes(df, args=None):
    """Freq-encoder passes (PerRegBurst / InstrumentProgram / Generator) that encode
    literal SETs. Run once at the start of each self-contained block (after
    ``expand_to_literal_form`` decompiles everything) and once at parse-time before rotation
    -- kept out of ``PASSES`` so they don't re-fire on already-encoded atoms inside the
    rotation loop."""
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


def block_refire_passes():
    """The ordered MacroPasses re-fired on every self-contained block after ``expand_to_literal_form``
    decompiles it -- the single source of truth shared by ``blocks.iter_self_contained_row_blocks`` and
    the ``op_contracts`` block-decoder contract. Any reference/ID-emitting pass (LoopPass, codebooks)
    MUST be here, else the window enters the tokenizer with that reference expanded to literal and the
    model never sees it (``test_block_refire_contract``)."""
    return tuple(FREQ_BLOCK_PASSES) + tuple(PASSES)


def run_block_refire_passes(df, args=None):
    """Apply ``block_refire_passes`` in order on one self-contained block slice."""
    for macro_pass in block_refire_passes():
        df = macro_pass.apply(df, args=args)
    return df


def block_refire_pass_names():
    """Class names of every pass re-fired per self-contained window -- the freq + pre-norm chain
    (``run_block_refire_passes``) PLUS the post-norm pre-voice passes (``run_post_norm_pre_voice_passes``,
    incl. ``LoopPass``) the builder applies after a norm. The set the block-decoder contract requires
    every reference-op producer (op_contracts.reference_op_producers) to be in."""
    return {p.__class__.__name__ for p in block_refire_passes()} | {
        p.__class__.__name__ for p in POST_NORM_PRE_VOICE_PASSES
    }
