"""Transform ABC + TransformPipeline + registry for the parse pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, Optional

import pandas as pd

from preframr_tokens.macros.pipeline_check import validate_pipeline_spec
from preframr_tokens.macros.transform_registry import (
    PipelineConfigError,
    PipelineEntry,
    _REGISTRY,
    _normalize_spec,
    ensure_default_transforms_registered,
    register,
)
from preframr_tokens.stfconstants import LOSS_TIER_NAMES

__all__ = [
    "Transform",
    "TransformPipeline",
    "PipelineEntry",
    "PipelineConfigError",
    "PassBackedTransform",
    "RowExpandingTransform",
    "register",
    "get_transform_class",
    "ensure_default_transforms_registered",
    "collect_substitutable_ops",
    "collect_substitutable_op_subregs",
    "collect_decomposing_op_codes",
    "collect_op_loss_tiers",
]


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


def _row_to_dict(row, columns):
    return {c: getattr(row, c) for c in columns}


def _expand_op_rows(df: pd.DataFrame, op_codes, expand_fn) -> pd.DataFrame:
    """Decompose rows whose ``op`` is in ``op_codes`` via ``expand_fn(row)``; pass other rows through unchanged. Returns a new DataFrame with original dtypes preserved."""
    if "op" not in df.columns or df.empty:
        return df
    out_rows = []
    for row in df.itertuples(index=False):
        if int(getattr(row, "op")) in op_codes:
            out_rows.extend(expand_fn(row))
        else:
            out_rows.append(_row_to_dict(row, df.columns))
    if not out_rows:
        return df.iloc[0:0]
    return pd.DataFrame(out_rows, columns=df.columns).astype(df.dtypes.to_dict())


class PassBackedTransform(Transform):
    """Base for Transforms whose ``forward()`` is a single ``MacroPass.apply`` and (optionally) whose ``expand_atom()`` delegates to a per-row ``Decoder.expand``. Subclasses set ``PASS_CLASS`` (required) and ``DECODER_CLASS`` (optional)."""

    PASS_CLASS: ClassVar[Optional[type]] = None
    DECODER_CLASS: ClassVar[Optional[type]] = None

    def __init__(self, **params):
        super().__init__(**params)
        assert (
            self.PASS_CLASS is not None
        ), f"{type(self).__name__}: PASS_CLASS must be set on subclass"
        self._impl = self.PASS_CLASS()
        if self.DECODER_CLASS is not None:
            self._decoder = self.DECODER_CLASS()

    def forward(self, df, args=None):
        return self._impl.apply(df, args=args)

    def expand_atom(self, row, state):
        return self._decoder.expand(row, state)


class RowExpandingTransform(PassBackedTransform):
    """``PassBackedTransform`` whose ``inverse()`` decomposes ``OP_CODES`` rows back into SET atoms via the subclass's ``_expand_row(row)`` staticmethod."""

    def inverse(self, df, args=None):
        return _expand_op_rows(df, self.OP_CODES, self._expand_row)

    @staticmethod
    def _expand_row(row):
        raise NotImplementedError


def collect_op_loss_tiers() -> dict[int, str]:
    """Op -> loss tier over both declaration surfaces: the ``Transform`` classes (their ``LOSS_TIER``) and
    the MacroPass-emitted generator/codebook ops (``op_contracts.MACRO_OP_LOSS_TIERS``, since those carry
    no Transform class). A Transform tier wins on the (non-existent) overlap. Lazy import keeps op_contracts
    -> transform the only edge."""
    out: dict[int, str] = {}
    from preframr_tokens.macros.op_contracts import MACRO_OP_LOSS_TIERS

    for op, tier in MACRO_OP_LOSS_TIERS.items():
        if tier not in LOSS_TIER_NAMES:
            raise ValueError(
                f"MACRO_OP_LOSS_TIERS[{op}]={tier!r} not in {LOSS_TIER_NAMES}"
            )
        out[int(op)] = tier
    for klass in _REGISTRY.values():
        tier = klass.LOSS_TIER
        if tier not in LOSS_TIER_NAMES:
            raise ValueError(
                f"{klass.__name__}.LOSS_TIER={tier!r} not in {LOSS_TIER_NAMES}"
            )
        for op in klass.OP_CODES:
            out[int(op)] = tier
    return out
