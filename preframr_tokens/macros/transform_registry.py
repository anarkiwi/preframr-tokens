"""Pipeline-spec primitives shared between ``transform.py`` and ``pipeline_check.py``. Lives in its own module so both can import without forming a cycle: ``transform_registry`` has no in-package dependencies, ``transform.py`` and ``pipeline_check.py`` both import from here, and ``transform.py`` can top-level import ``validate_pipeline_spec`` from ``pipeline_check.py`` without circular load."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PipelineEntry",
    "PipelineConfigError",
    "register",
    "ensure_default_transforms_registered",
]

# Populated at ``Transform`` subclass-registration time. Module-private to
# transform_registry; consumers should use ``get_transform_class`` /
# ``collect_op_loss_tiers`` / etc. from ``transform.py``.
_REGISTRY: dict[str, type] = {}

_DEFAULTS_REGISTERED = False


def register(name: str):
    """Decorator binding a ``Transform`` subclass into ``_REGISTRY`` under ``name``. Raises ``ValueError`` if ``name`` is already taken."""

    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"transform name {name!r} already registered")
        cls.NAME = name
        _REGISTRY[name] = cls
        return cls

    return deco


def ensure_default_transforms_registered() -> None:
    """Import the default-transform modules once so their ``@register(...)`` side effects populate ``_REGISTRY``. Idempotent; safe to call anywhere ``collect_op_loss_tiers`` / ``collect_substitutable_ops`` / ``get_transform_class`` / ``validate_pipeline_spec`` is reachable."""
    global _DEFAULTS_REGISTERED  # pylint: disable=global-statement
    if _DEFAULTS_REGISTERED:
        return
    # pylint: disable=import-outside-toplevel,unused-import
    from preframr_tokens.macros import (
        transforms_audio_bit_exact,  # noqa: F401
        transforms_bit_exact,  # noqa: F401
        transforms_parser_stubs,  # noqa: F401
    )

    _DEFAULTS_REGISTERED = True


@dataclass
class PipelineEntry:
    """One ``Transform`` invocation in a pipeline spec: a registered ``name`` plus optional ``params`` dict forwarded to the Transform constructor."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class PipelineConfigError(ValueError):
    """Raised by ``TransformPipeline.from_spec`` when ``validate_pipeline_spec`` returns non-empty errors. Carries the error list under ``.errors``."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("pipeline config errors:\n  - " + "\n  - ".join(errors))


def _normalize_spec(spec: Any) -> list[PipelineEntry]:
    """Coerce a pipeline spec (json string, dict with ``transforms`` key, list of names or dicts) into a list of ``PipelineEntry``."""
    if isinstance(spec, str):
        spec = json.loads(spec)
    if isinstance(spec, dict):
        if "transforms" not in spec:
            raise ValueError(
                "pipeline spec dict must have a 'transforms' key with a list"
            )
        raw = spec["transforms"]
    elif isinstance(spec, list):
        raw = spec
    else:
        raise TypeError(f"unsupported pipeline spec type {type(spec)}")
    out: list[PipelineEntry] = []
    for item in raw:
        if isinstance(item, str):
            out.append(PipelineEntry(name=item))
        elif isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError(f"pipeline spec entry missing 'name': {item}")
            out.append(
                PipelineEntry(name=name, params=dict(item.get("params", {})))
            )
        elif isinstance(item, PipelineEntry):
            out.append(item)
        else:
            raise TypeError(f"unsupported pipeline spec entry type {type(item)}")
    return out
