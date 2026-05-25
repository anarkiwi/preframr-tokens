"""Torch-free source of truth for a tokenizer/parser args namespace: parser
params + every macro-pass flag, so ``RegLogParser`` / the macro passes read
defaults from one place. ``full_macros`` is the production-registered subset
(``REGISTERED_MACROS``), NOT every flag -- the rest are experimental/refuted and
corrupt FRAME svt when combined."""

from __future__ import annotations

from types import SimpleNamespace

__all__ = [
    "PARSER_DEFAULTS",
    "MACRO_FLAGS",
    "REGISTERED_MACROS",
    "default_tokenizer_args",
    "named_config",
    "NAMED_CONFIGS",
]

PARSER_DEFAULTS = {
    "cents": 50,
    "exclude_list": None,
    "min_irq": int(1.5e4),
    "max_irq": int(2.5e4),
    "min_song_tokens": 0,
    "diffq": 4,
    "loop_lookahead": 3,
    "coarsen_min_len": 16,
    "voice_trajectory_window": 8,
    "pipeline_spec": "",
    "meta_exclude_digi": False,
    "meta_irq_lo": 0,
    "meta_irq_hi": 0,
    "meta_require": False,
}

MACRO_FLAGS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c3",
    "legato_pass_c4",
    "legato_pass_c7",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "fuzzy_loop_pass",
    "fuzzy_fp_adsr",
    "coarsen_pass",
    "mode_vol_flip_pass",
    "voice_trajectory_pass",
    "voice_trajectory_distributed_pass",
    "set_to_diff_pass",
    "freq_nudge_pass",
    "release_update_pass",
    "ctrl_triple_pass",
    "lonely_catch_all",
)


def default_tokenizer_args(**overrides) -> SimpleNamespace:
    """Namespace with the real parser params plus every macro-pass flag present
    and off (the no-macro baseline); ``overrides`` win so callers enable only the
    passes they want."""
    cfg = dict(PARSER_DEFAULTS)
    for flag in MACRO_FLAGS:
        cfg[flag] = False
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


REGISTERED_MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "freq_nudge_pass",
    "release_update_pass",
    "ctrl_triple_pass",
    "lonely_catch_all",
)

NAMED_CONFIGS = {
    "baseline": {},
    "full_macros": {flag: True for flag in REGISTERED_MACROS},
}


def named_config(name: str, **overrides) -> SimpleNamespace:
    """Build a preset args namespace by name (``baseline`` / ``full_macros``);
    ``overrides`` are applied on top of the preset."""
    if name not in NAMED_CONFIGS:
        raise KeyError(f"unknown config {name!r}; known: {sorted(NAMED_CONFIGS)}")
    cfg = dict(NAMED_CONFIGS[name])
    cfg.update(overrides)
    return default_tokenizer_args(**cfg)
