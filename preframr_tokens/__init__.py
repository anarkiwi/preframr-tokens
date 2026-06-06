"""Public API for preframr-tokens: import the ``__all__`` names from this
package root (e.g. ``from preframr_tokens import RegLogParser``). The
``stfconstants`` and ``engine_fingerprint`` submodules are also public, stable
namespaces; every other ``preframr_tokens.*`` submodule path is internal and
may move between releases."""

from preframr_tokens.stfconstants import (
    DEFAULT_IRQ_CYCLES,
    DUMP_SUFFIX,
    LOSS_TIER_NAMES,
    MODEL_PDTYPE,
    PAD_ID,
)
from preframr_tokens.utils import to_int64_arrays
from preframr_tokens.reg_match import reg_class
from preframr_tokens.palette_io import dump_palettes_attrs, load_palettes_attrs
from preframr_tokens.macros.roles import (
    DISTANCE_PAIR_OPS,
    DistancePairSpec,
    distance_pair_role,
    frame_weight_role,
)
from preframr_tokens.audit_primitives import (
    detect_tail_cycle,
    distinct_n,
    op_atom_profile,
    register_state,
    tier_accuracy,
)
from preframr_tokens.tokenizer_config import (
    default_tokenizer_args,
    named_config,
)
from preframr_tokens.vocab_signature import CONTENT_TIER, VocabSignature
from preframr_tokens.tier_classify import (
    build_vocab_tier_ids,
    build_vocab_tier_map,
    vocab_id_tier,
)
from preframr_tokens.token_weighting import vocab_frame_weights
from preframr_tokens.constrained_decode import (
    PendingSlot,
    StreamState,
    VocabArrays,
    frame_marker_count,
    precompute_subtoken_arrays,
    precompute_vocab_arrays,
    tail_charge_for_prompt,
)
from preframr_tokens.macros.transform import (
    PassBackedTransform,
    PipelineEntry,
    RowExpandingTransform,
    Transform,
    TransformPipeline,
    ensure_default_transforms_registered,
    get_transform_class,
    register,
)
from preframr_tokens.macros import (
    codebook_live_ids,
    validate_back_refs,
    validate_codebook_refs,
    validate_pattern_overlays,
    validate_stream,
)
from preframr_tokens.macros.op_contracts import op_name_by_id, op_name_tiers
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.reglogparser import (
    RegLogParser,
    combine_reg,
    prepare_df_for_audio,
    read_initial_irq,
    remove_voice_reg,
)
from preframr_tokens.blocks import (
    LEGACY_EVAL_SUBSET_NAME,
    iter_voiced_blocks,
    reg_widths_path,
    self_contained_prompt_df,
)
from preframr_tokens.corpus import Corpus, TokenizeMeta
from preframr_tokens.parse_runner import parse_corpus

__all__ = [
    "RegLogParser",
    "RegTokenizer",
    "Corpus",
    "TokenizeMeta",
    "StreamState",
    "PendingSlot",
    "VocabArrays",
    "VocabSignature",
    "Transform",
    "TransformPipeline",
    "PipelineEntry",
    "PassBackedTransform",
    "RowExpandingTransform",
    "DistancePairSpec",
    "PerRegBurstPass",
    "register",
    "get_transform_class",
    "ensure_default_transforms_registered",
    "parse_corpus",
    "prepare_df_for_audio",
    "remove_voice_reg",
    "read_initial_irq",
    "combine_reg",
    "reg_class",
    "to_int64_arrays",
    "frame_marker_count",
    "tail_charge_for_prompt",
    "precompute_vocab_arrays",
    "precompute_subtoken_arrays",
    "iter_voiced_blocks",
    "reg_widths_path",
    "self_contained_prompt_df",
    "vocab_id_tier",
    "build_vocab_tier_ids",
    "build_vocab_tier_map",
    "vocab_frame_weights",
    "tier_accuracy",
    "detect_tail_cycle",
    "distinct_n",
    "register_state",
    "op_atom_profile",
    "default_tokenizer_args",
    "named_config",
    "distance_pair_role",
    "frame_weight_role",
    "validate_back_refs",
    "validate_codebook_refs",
    "validate_pattern_overlays",
    "validate_stream",
    "codebook_live_ids",
    "load_palettes_attrs",
    "dump_palettes_attrs",
    "PAD_ID",
    "MODEL_PDTYPE",
    "DUMP_SUFFIX",
    "LEGACY_EVAL_SUBSET_NAME",
    "DEFAULT_IRQ_CYCLES",
    "LOSS_TIER_NAMES",
    "DISTANCE_PAIR_OPS",
    "CONTENT_TIER",
    "op_name_by_id",
    "op_name_tiers",
]
