"""Static checker for pipeline_spec: dependency, position, idempotence, op-availability, register-shape, decoder availability, param validation, registry-wide invariants."""

from __future__ import annotations

from typing import Any

from preframr_tokens.macros.transform_registry import (
    PipelineConfigError,
    _REGISTRY,
    _normalize_spec,
    ensure_default_transforms_registered,
)
from preframr_tokens.stfconstants import DIFF_OP, FLIP_OP, SET_OP

__all__ = ["PipelineConfigError", "validate_pipeline_spec"]

_PRIMITIVE_OPS_ALWAYS_AVAILABLE = frozenset({int(SET_OP), int(DIFF_OP), int(FLIP_OP)})

_KNOWN_PHANTOM_NAMES = frozenset(
    {
        "add_voice_reg",
        "squeeze_changes",
        "combine_regs",
        "quantize_freq_to_cents",
        "simplify_ctrl",
        "simplify_pcm",
        "add_frame_reg",
        "filter",
        "squeeze_frame_regs",
        "consolidate_frames",
        "cap_delay",
        "rotate_voice_augment",
        "norm_pr_order",
    }
)


_HARDCODED_PRE_NORM_TRANSFORM_NAMES = frozenset(
    {
        "slope",
        "flip2",
        "transpose",
        "dedup_set",
        "hard_restart",
        "legato_per_cluster",
        "subreg_flush",
        "voice_block_order",
        "loop",
    }
)


def _hardcoded_emits_non_set_regs() -> set[int]:
    """Worst-case union of EMITS_NON_SET_REGS across the hardcoded pre-norm and post-norm-pre-voice passes that run in reglogparser regardless of pipeline_spec. Computed dynamically from the registered Transform classes so adding a new declaration on an existing transform automatically flows through."""
    out: set[int] = set()
    for name in _HARDCODED_PRE_NORM_TRANSFORM_NAMES:
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        out |= set(int(r) for r in cls.EMITS_NON_SET_REGS)
    return out


def validate_pipeline_spec(spec: Any, args=None) -> list[str]:
    """Return a list of human-readable errors. Empty list = spec is valid."""
    ensure_default_transforms_registered()
    entries = _normalize_spec(spec)
    errors: list[str] = []
    if not entries:
        return ["empty pipeline spec"]
    for i, entry in enumerate(entries):
        if entry.name not in _REGISTRY:
            errors.append(f"#{i} '{entry.name}': unknown transform name")
    valid = [(i, e) for i, e in enumerate(entries) if e.name in _REGISTRY]
    pos = {e.name: i for i, e in valid}
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        if cls.POSITION == "first" and i != 0:
            errors.append(f"#{i} '{entry.name}': POSITION='first' but appears at #{i}")
        if cls.POSITION == "last" and i != len(entries) - 1:
            errors.append(
                f"#{i} '{entry.name}': POSITION='last' but appears at #{i} "
                f"(spec has {len(entries)} entries)"
            )
        for predecessor in cls.MUST_FOLLOW:
            if predecessor not in pos:
                errors.append(
                    f"#{i} '{entry.name}': MUST_FOLLOW='{predecessor}' "
                    f"absent from spec"
                )
            elif pos[predecessor] > i:
                errors.append(
                    f"#{i} '{entry.name}': MUST_FOLLOW='{predecessor}' "
                    f"appears later at #{pos[predecessor]}"
                )
        for successor in cls.MUST_PRECEDE:
            if successor in pos and pos[successor] < i:
                errors.append(
                    f"#{i} '{entry.name}': MUST_PRECEDE='{successor}' "
                    f"appears earlier at #{pos[successor]}"
                )
        for arg_name in cls.REQUIRES_ARGS:
            if args is None or not getattr(args, arg_name, False):
                errors.append(
                    f"#{i} '{entry.name}': REQUIRES_ARGS includes "
                    f"'{arg_name}' but args.{arg_name} is unset/false"
                )
    seen: dict[str, int] = {}
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        if entry.name in seen and not cls.IDEMPOTENT:
            errors.append(
                f"#{i} '{entry.name}': repeated (also at "
                f"#{seen[entry.name]}); transform is not declared IDEMPOTENT"
            )
        seen.setdefault(entry.name, i)
    available_ops: set[int] = set(_PRIMITIVE_OPS_ALWAYS_AVAILABLE)
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        required = set(int(o) for o in cls.REQUIRES_OPS)
        missing = required - available_ops
        if missing:
            errors.append(
                f"#{i} '{entry.name}': REQUIRES_OPS={sorted(missing)} "
                f"not provided by any earlier transform"
            )
        provides = cls.PROVIDES_OPS or cls.OP_CODES
        available_ops |= set(int(o) for o in provides)
        available_ops -= set(int(o) for o in cls.CONSUMES_OPS)
    non_set_regs_so_far: set[int] = _hardcoded_emits_non_set_regs()
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        expects = set(int(r) for r in cls.EXPECTS_SET_ON_REGS)
        handles = set(int(r) for r in cls.HANDLES_NON_SET_ON_REGS)
        unhandled = (expects & non_set_regs_so_far) - handles
        if unhandled:
            errors.append(
                f"#{i} '{entry.name}': EXPECTS_SET_ON_REGS={sorted(unhandled)} "
                f"but upstream transforms emit non-SET there; add to "
                f"HANDLES_NON_SET_ON_REGS or reorder pipeline"
            )
        non_set_regs_so_far |= set(int(r) for r in cls.EMITS_NON_SET_REGS)
    try:
        from preframr_tokens.macros.decoders import DECODERS

        decoder_ops = set(DECODERS.keys())
    except Exception:
        decoder_ops = None
    if decoder_ops is not None:
        for i, entry in valid:
            cls = _REGISTRY[entry.name]
            if getattr(cls, "DECODES_VIA_DF", False):
                continue
            missing = set(int(o) for o in cls.OP_CODES) - decoder_ops
            if missing:
                errors.append(
                    f"#{i} '{entry.name}': OP_CODES {sorted(missing)} have no "
                    f"DECODERS entry; audio render would fail at runtime"
                )
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        substitutable = set(int(o) for o in cls.SUBSTITUTABLE_OPS)
        own = set(int(o) for o in cls.OP_CODES)
        stray = substitutable - own
        if stray:
            errors.append(
                f"#{i} '{entry.name}': SUBSTITUTABLE_OPS {sorted(stray)} are "
                f"not in this transform's OP_CODES; substitutability declarations "
                f"must be a subset of OP_CODES"
            )
        sub_pairs = set((int(o), int(s)) for o, s in cls.SUBSTITUTABLE_OP_SUBREGS)
        stray_pairs = {(o, s) for (o, s) in sub_pairs if o not in own}
        if stray_pairs:
            errors.append(
                f"#{i} '{entry.name}': SUBSTITUTABLE_OP_SUBREGS contains "
                f"(op, subreg) pairs whose op is not in this transform's "
                f"OP_CODES: {sorted(stray_pairs)}"
            )
    args_satisfied: dict[int, set[str]] = {}
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        unsatisfied_args = {
            a
            for a in cls.REQUIRES_ARGS
            if not (args is not None and getattr(args, a, False))
        }
        args_satisfied[i] = unsatisfied_args
    provided_by: dict[int, int] = {}
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        provides = cls.PROVIDES_OPS or cls.OP_CODES
        for op in provides:
            provided_by.setdefault(int(op), i)
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        for op in cls.REQUIRES_OPS:
            op_int = int(op)
            provider_i = provided_by.get(op_int)
            if provider_i is None:
                continue
            provider_unsat = args_satisfied.get(provider_i, set())
            if provider_unsat:
                provider_name = valid[
                    [j for j, (idx, _) in enumerate(valid) if idx == provider_i][0]
                ][1].name
                errors.append(
                    f"#{i} '{entry.name}': REQUIRES_OPS includes {op_int} "
                    f"provided by '{provider_name}', but provider's "
                    f"REQUIRES_ARGS {sorted(provider_unsat)} are not all set "
                    f"in args; provider will be a no-op"
                )
    for i, entry in valid:
        cls = _REGISTRY[entry.name]
        validators = getattr(cls, "PARAM_VALIDATORS", {}) or {}
        for key, value in entry.params.items():
            v = validators.get(key)
            if v is not None:
                try:
                    ok = bool(v(value))
                except Exception as exc:
                    ok = False
                    errors.append(
                        f"#{i} '{entry.name}': PARAM_VALIDATORS[{key!r}] raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not ok:
                    errors.append(
                        f"#{i} '{entry.name}': param {key!r}={value!r} failed "
                        f"PARAM_VALIDATORS check"
                    )
    return errors


def validate_registry() -> list[str]:
    """Walk every registered transform and check global invariants: op-code uniqueness, loss-tier uniqueness per op, MUST_FOLLOW / MUST_PRECEDE referent existence, inverse coverage for bit-exact tiers, no-op transforms cannot claim PROVIDES_OPS, DECOMPOSES_TO_ATOMS requires non-trivial inverse, and registered class methods have the right signature. Returns a list of human-readable errors."""
    import inspect

    from preframr_tokens.macros.transform import Transform as _BaseTransform

    errors: list[str] = []
    op_to_owner: dict[int, str] = {}
    op_to_loss_tier: dict[int, tuple[str, str]] = {}
    for name, cls in _REGISTRY.items():
        for op in cls.OP_CODES:
            op_int = int(op)
            if op_int in op_to_owner and op_to_owner[op_int] != name:
                errors.append(
                    f"op code {op_int} declared by both "
                    f"'{op_to_owner[op_int]}' and '{name}'"
                )
            else:
                op_to_owner[op_int] = name
            tier = cls.LOSS_TIER
            if op_int in op_to_loss_tier:
                prev_tier, prev_owner = op_to_loss_tier[op_int]
                if prev_tier != tier:
                    errors.append(
                        f"op code {op_int} loss-tier mismatch: '{prev_owner}' "
                        f"declares LOSS_TIER={prev_tier!r}, '{name}' declares "
                        f"LOSS_TIER={tier!r}"
                    )
            else:
                op_to_loss_tier[op_int] = (tier, name)
    valid_names = set(_REGISTRY.keys()) | _KNOWN_PHANTOM_NAMES
    for name, cls in _REGISTRY.items():
        for ref in cls.MUST_FOLLOW:
            if ref not in valid_names:
                errors.append(
                    f"'{name}': MUST_FOLLOW={ref!r} not in registry "
                    f"and not a known phantom name"
                )
        for ref in cls.MUST_PRECEDE:
            if ref not in valid_names:
                errors.append(
                    f"'{name}': MUST_PRECEDE={ref!r} not in registry "
                    f"and not a known phantom name"
                )
    for name, cls in _REGISTRY.items():
        if not cls.OP_CODES and cls.PROVIDES_OPS:
            errors.append(
                f"'{name}': OP_CODES is empty but PROVIDES_OPS={sorted(int(o) for o in cls.PROVIDES_OPS)} "
                f"is non-empty; a transform with no op codes cannot provide ops"
            )
        if cls.DECOMPOSES_TO_ATOMS:
            if cls.inverse is _BaseTransform.inverse:
                errors.append(
                    f"'{name}': DECOMPOSES_TO_ATOMS=True but inverse() is the "
                    f"base no-op; a decomposing transform must implement inverse"
                )
            if not cls.OP_CODES:
                errors.append(
                    f"'{name}': DECOMPOSES_TO_ATOMS=True but OP_CODES is empty"
                )
        try:
            forward_sig = inspect.signature(cls.forward)
            params = list(forward_sig.parameters)
            if params[:2] != ["self", "df"] or "args" not in params:
                errors.append(
                    f"'{name}': forward() signature {params} does not match "
                    f"expected (self, df, args=None)"
                )
        except (TypeError, ValueError) as exc:
            errors.append(f"'{name}': forward() inspect.signature failed: {exc}")
        try:
            inverse_sig = inspect.signature(cls.inverse)
            params = list(inverse_sig.parameters)
            if params[:2] != ["self", "df"] or "args" not in params:
                errors.append(
                    f"'{name}': inverse() signature {params} does not match "
                    f"expected (self, df, args=None)"
                )
        except (TypeError, ValueError) as exc:
            errors.append(f"'{name}': inverse() inspect.signature failed: {exc}")
    return errors
