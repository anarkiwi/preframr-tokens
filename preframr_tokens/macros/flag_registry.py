"""Single source of truth for macro-pass gating flags: each gated ``MacroPass``
declares the args it reads via ``GATE_FLAGS`` (each ``Transform`` via
``REQUIRES_ARGS``); ``macro_flag_names`` glob-imports the ``macros`` package and
unions the declarations, so ``tokenizer_config.MACRO_FLAGS`` derives from the
passes rather than a hand-maintained copy that drifts."""

from __future__ import annotations

__all__ = ["macro_flag_names", "ensure_passes_registered"]


def ensure_passes_registered() -> None:
    """Import every module that may define a gated pass/transform so their
    classes are present in the ``MacroPass`` / ``Transform`` subclass trees.
    Idempotent (imports are cached)."""
    # pylint: disable=import-outside-toplevel
    import importlib
    import pkgutil

    import preframr_tokens.macros as macros_pkg

    for mod in pkgutil.iter_modules(macros_pkg.__path__, macros_pkg.__name__ + "."):
        importlib.import_module(mod.name)
    importlib.import_module("preframr_tokens.coarsen_pass")
    from preframr_tokens.macros.transform_registry import (
        ensure_default_transforms_registered,
    )

    ensure_default_transforms_registered()


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def macro_flag_names() -> set[str]:
    """The set of argparse flag names that any registered gated pass/transform
    reads to decide whether (or how) to run."""
    ensure_passes_registered()
    # pylint: disable=import-outside-toplevel
    from preframr_tokens.macros.passes_base import MacroPass
    from preframr_tokens.macros.transform import Transform

    flags: set[str] = set()
    for base in (MacroPass, Transform):
        for cls in _all_subclasses(base):
            flags |= set(getattr(cls, "GATE_FLAGS", frozenset()) or frozenset())
            flags |= set(getattr(cls, "REQUIRES_ARGS", frozenset()) or frozenset())
    return flags
