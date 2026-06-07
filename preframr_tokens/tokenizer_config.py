"""Torch-free source of truth for a tokenizer/parser args namespace: parser
params + every macro-pass flag. ``MACRO_FLAGS`` is derived from the passes
(``flag_registry.macro_flag_names``) not hand-listed, so flags can't drift.
``full_macros`` is the production-registered subset (``REGISTERED_MACROS``), NOT
every flag -- the rest are experimental/refuted and corrupt FRAME svt combined."""

from __future__ import annotations

from types import SimpleNamespace

__all__ = [
    "PARSER_DEFAULTS",
    "MACRO_FLAGS",
    "REGISTERED_MACROS",
    "DEFAULT_PIPELINE",
    "default_tokenizer_args",
    "default_pipeline_args",
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
    "pipeline_spec": "",
    "meta_exclude_digi": False,
    "meta_irq_lo": 0,
    "meta_irq_hi": 0,
    "meta_require": False,
}

_MACRO_FLAGS_CACHE: tuple[str, ...] | None = None


def _macro_flags() -> tuple[str, ...]:
    """Sorted tuple of every macro-pass gating flag, collected from the passes.
    Memoized: the (pandas-touching) macro import only happens on first use."""
    global _MACRO_FLAGS_CACHE  # pylint: disable=global-statement
    if _MACRO_FLAGS_CACHE is None:
        # pylint: disable=import-outside-toplevel
        from preframr_tokens.macros.flag_registry import macro_flag_names

        _MACRO_FLAGS_CACHE = tuple(sorted(macro_flag_names()))
    return _MACRO_FLAGS_CACHE


def __getattr__(name):
    if name == "MACRO_FLAGS":
        return _macro_flags()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_tokenizer_args(**overrides) -> SimpleNamespace:
    """The additive low-level builder: real parser params plus every macro-pass flag present and
    OFF, so an omitted flag means OFF and ``overrides`` enable only the passes a caller names. The
    DEFAULT PARSE PIPELINE the runner uses is ``default_pipeline_args()`` (the generator pipeline),
    NOT this bare builder -- keeping omission-means-off intact for every partial-flags caller.
    """
    cfg = dict(PARSER_DEFAULTS)
    for flag in _macro_flags():
        cfg[flag] = False
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


REGISTERED_MACROS = (
    "generator_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "loop_pass",
    "loop_transposed",
    "instrument_program",
    "melody_skeleton",
    "universal_pitch",
    "table_resid_split",
)

NAMED_CONFIGS = ("baseline", "full_macros")

DEFAULT_PIPELINE = "full_macros"


def named_config(name: str, **overrides) -> SimpleNamespace:
    """Build a preset args namespace by name (``baseline`` = additive all-off /
    ``full_macros`` = the generator pipeline = ``REGISTERED_MACROS``); ``overrides`` win.
    """
    if name == "baseline":
        cfg = {}
    elif name == "full_macros":
        cfg = {flag: True for flag in REGISTERED_MACROS}
    else:
        raise KeyError(f"unknown config {name!r}; known: {sorted(NAMED_CONFIGS)}")
    cfg.update(overrides)
    return default_tokenizer_args(**cfg)


def default_pipeline_args(**overrides) -> SimpleNamespace:
    """The runner's DEFAULT parse pipeline: the corrected generator pipeline
    (``named_config(DEFAULT_PIPELINE)`` = ``REGISTERED_MACROS``). Callers that want the parse to
    behave as the framework default does should build args from this, not the additive bare builder.
    """
    return named_config(DEFAULT_PIPELINE, **overrides)
