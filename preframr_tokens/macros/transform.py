"""Transform ABC + TransformPipeline + registry for the parse pipeline."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Optional

import pandas as pd

_REGISTRY: dict[str, type["Transform"]] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"transform name {name!r} already registered")
        cls.NAME = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_transform_class(name: str) -> type["Transform"]:
    if name not in _REGISTRY:
        raise KeyError(f"no transform registered as {name!r}")
    return _REGISTRY[name]


class Transform(ABC):
    """One step in the parse pipeline. Forward + inverse both operate on the entire df."""

    NAME: ClassVar[str] = ""
    TIER: ClassVar[str] = "audio_bit_exact"
    OP_CODES: ClassVar[frozenset[int]] = frozenset()
    OPERATES_ON_VOICE_REGS: ClassVar[bool] = False
    SUBSTITUTABLE_OPS: ClassVar[frozenset[int]] = frozenset()
    SUBSTITUTABLE_OP_SUBREGS: ClassVar[frozenset[tuple[int, int]]] = frozenset()
    DECOMPOSES_TO_ATOMS: ClassVar[bool] = False
    LOSS_TIER: ClassVar[str] = "content"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}
    LOSSY_TOLERANCE: ClassVar[float] = 0.0
    REQUIRES_OPS: ClassVar[frozenset[int]] = frozenset()
    REQUIRES_REGS: ClassVar[frozenset[int]] = frozenset()
    PROVIDES_OPS: ClassVar[frozenset[int]] = frozenset()
    CONSUMES_OPS: ClassVar[frozenset[int]] = frozenset()
    IDEMPOTENT: ClassVar[bool] = False
    MUST_FOLLOW: ClassVar[frozenset[str]] = frozenset()
    MUST_PRECEDE: ClassVar[frozenset[str]] = frozenset()
    POSITION: ClassVar[Optional[str]] = None
    REQUIRES_ARGS: ClassVar[frozenset[str]] = frozenset()
    EMITS_NON_SET_REGS: ClassVar[frozenset[int]] = frozenset()
    EXPECTS_SET_ON_REGS: ClassVar[frozenset[int]] = frozenset()
    HANDLES_NON_SET_ON_REGS: ClassVar[frozenset[int]] = frozenset()
    PARAM_VALIDATORS: ClassVar[dict[str, Any]] = {}
    DECODES_VIA_DF: ClassVar[bool] = False

    def __init__(self, **params):
        merged = {**self.DEFAULT_PARAMS, **params}
        unknown = set(params) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise ValueError(
                f"{self.NAME}: unknown params {sorted(unknown)}; "
                f"declared {sorted(self.DEFAULT_PARAMS)}"
            )
        for key, value in merged.items():
            validator = self.PARAM_VALIDATORS.get(key)
            if validator is not None and not validator(value):
                raise ValueError(
                    f"{self.NAME}: param {key!r}={value!r} failed PARAM_VALIDATORS check"
                )
        self.params = merged

    @abstractmethod
    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame: ...

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        return df

    def round_trip_check(self, df: pd.DataFrame, args=None):
        forward = self.forward(df, args=args)
        recovered = self.inverse(forward, args=args)
        if self.TIER == "bit_exact":
            return _assert_frame_equal(df, recovered, label=self.NAME)
        from preframr_audio.fidelity import assert_dfs_render_equivalent

        return assert_dfs_render_equivalent(
            df,
            recovered,
            args=args,
            tmp_path=None,
            label_a=f"{self.NAME}_orig",
            label_b=f"{self.NAME}_round_trip",
            tolerance=self.LOSSY_TOLERANCE,
        )


def _assert_frame_equal(a: pd.DataFrame, b: pd.DataFrame, label: str):
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_like=False,
        obj=label,
    )
    return True


@dataclass
class PipelineEntry:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class TransformPipeline:
    """Ordered list of Transforms; forward applies in order, inverse in reverse."""

    def __init__(self, transforms: Iterable[Transform]):
        self._transforms: list[Transform] = list(transforms)
        self._by_op: dict[int, Transform] = {}
        for t in self._transforms:
            for op in t.OP_CODES:
                self._by_op[int(op)] = t

    @classmethod
    def from_spec(
        cls, spec: Any, args=None, validate: bool = True
    ) -> "TransformPipeline":
        if validate:
            from preframr_tokens.macros.pipeline_check import (
                PipelineConfigError,
                validate_pipeline_spec,
            )

            errors = validate_pipeline_spec(spec, args=args)
            if errors:
                raise PipelineConfigError(errors)
        entries = _normalize_spec(spec)
        instances = []
        for entry in entries:
            klass = get_transform_class(entry.name)
            instances.append(klass(**entry.params))
        return cls(instances)

    def to_spec(self) -> dict[str, Any]:
        return {
            "transforms": [
                {"name": t.NAME, **({"params": dict(t.params)} if t.params else {})}
                for t in self._transforms
            ]
        }

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        for t in self._transforms:
            df = t.forward(df, args=args)
        return df

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        for t in reversed(self._transforms):
            df = t.inverse(df, args=args)
        return df

    def by_op_code(self, op: int) -> Optional[Transform]:
        return self._by_op.get(int(op))

    @property
    def tier(self) -> str:
        if not self._transforms:
            return "bit_exact"
        order = {"bit_exact": 0, "audio_bit_exact": 1, "lossy": 2}
        worst = max(order.get(t.TIER, len(order)) for t in self._transforms)
        for k, v in order.items():
            if v == worst:
                return k
        return "lossy"

    def __iter__(self):
        return iter(self._transforms)

    def __len__(self):
        return len(self._transforms)


def _normalize_spec(spec: Any) -> list[PipelineEntry]:
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
            out.append(PipelineEntry(name=name, params=dict(item.get("params", {}))))
        elif isinstance(item, PipelineEntry):
            out.append(item)
        else:
            raise TypeError(f"unsupported pipeline spec entry type {type(item)}")
    return out


def collect_substitutable_ops() -> frozenset[int]:
    out: set[int] = set()
    for klass in _REGISTRY.values():
        out.update(int(o) for o in klass.SUBSTITUTABLE_OPS)
    return frozenset(out)


def collect_substitutable_op_subregs() -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for klass in _REGISTRY.values():
        out.update((int(o), int(s)) for o, s in klass.SUBSTITUTABLE_OP_SUBREGS)
    return frozenset(out)


def collect_decomposing_op_codes() -> frozenset[int]:
    out: set[int] = set()
    for klass in _REGISTRY.values():
        if klass.DECOMPOSES_TO_ATOMS:
            out.update(int(o) for o in klass.OP_CODES)
    return frozenset(out)


_LOSS_TIER_NAMES = ("structural", "mid", "content", "zero")


def collect_op_loss_tiers() -> dict[int, str]:
    out: dict[int, str] = {}
    for klass in _REGISTRY.values():
        tier = klass.LOSS_TIER
        if tier not in _LOSS_TIER_NAMES:
            raise ValueError(
                f"{klass.__name__}.LOSS_TIER={tier!r} not in {_LOSS_TIER_NAMES}"
            )
        for op in klass.OP_CODES:
            out[int(op)] = tier
    return out
