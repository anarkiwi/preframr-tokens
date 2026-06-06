"""Single source of truth for macro-pass gating flags: each gated ``MacroPass``
declares the args it reads via ``GATE_FLAGS`` (each ``Transform`` via
``REQUIRES_ARGS``); ``macro_flag_names`` glob-imports the ``macros`` package and
unions the declarations, so ``tokenizer_config.MACRO_FLAGS`` derives from the
passes rather than a hand-maintained copy that drifts."""

from __future__ import annotations

__all__ = [
    "macro_flag_names",
    "ensure_passes_registered",
    "resolve_flags",
    "valid_combo",
    "minimal_configs",
    "FLAG_REQUIRES",
    "FLAG_CONFLICTS",
]

FLAG_REQUIRES: dict[str, frozenset[str]] = {
    "melody_skeleton": frozenset({"generator_pass"}),
    "universal_pitch": frozenset({"melody_skeleton", "generator_pass"}),
}
FLAG_CONFLICTS: dict[str, frozenset[str]] = {}


def resolve_flags(flags):
    """Expand a set of requested flags with their transitive REQUIRES. Raises ValueError if the closure
    is internally inconsistent (a required flag conflicts with a requested one)."""
    out = set(flags)
    changed = True
    while changed:
        changed = False
        for f in list(out):
            need = FLAG_REQUIRES.get(f, frozenset())
            if not need <= out:
                out |= need
                changed = True
    conflict = _conflict_pair(out)
    if conflict:
        raise ValueError(
            f"incompatible pipeline: {conflict[0]} conflicts with {conflict[1]}"
        )
    return out


def _conflict_pair(flags):
    """Return a conflicting ``(a, b)`` pair from ``flags`` (symmetric over FLAG_CONFLICTS), else None."""
    for a, against in FLAG_CONFLICTS.items():
        if a in flags:
            for b in against:
                if b in flags:
                    return (a, b)
    for a in flags:
        for b, against in FLAG_CONFLICTS.items():
            if a in against and b in flags:
                return (b, a)
    return None


def valid_combo(flags):
    """True if ``flags`` (already resolved) has no conflicting pair."""
    return _conflict_pair(set(flags)) is None


def minimal_configs():
    """One minimal VALID config per gated flag: the flag plus its transitive REQUIRES, conflicts dropped.
    Used by the combinatorial frame-diff gate so every registered pass is exercised at least once and a
    newly-added transform is picked up automatically."""
    configs = {}
    for f in sorted(macro_flag_names()):
        try:
            configs[f] = frozenset(resolve_flags({f}))
        except ValueError:
            continue
    return configs


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
