"""Single op-contract registry for constrained decode (RESID_ZERO_PHASE3 §4 B0): one ``OpContract`` per
op the model can emit -- the atom ops in ``DECODERS`` plus the loop ops ``expand_loops`` consumes -- so
the sampling mask, the stream validators, and the precompute arrays dispatch on one source of truth
instead of three hand-kept copies. Each op declares its ``MaskRole`` (how constrained decode treats it);
the completeness test goes red if any emittable op lacks a contract (codebook-family drift, caught).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import preframr_tokens.stfconstants as _stfconstants
from preframr_tokens.macros.codebook import codebook_spec_tuples
from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.stfconstants import (
    GEN_TRI_OP,
    GEN_TUNING_OP,
    MELODY_INTERVAL_OP,
    NOTE_INTERVAL_OP,
    DIFF_OP,
    DO_LOOP_OP,
    FLIP_OP,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    SET_OP,
    SUBREG_FLUSH_OP,
    SWEEP_OP,
    TRANSPOSE_OP,
)

__all__ = [
    "MaskRole",
    "OpContract",
    "OP_CONTRACTS",
    "LOOP_OPS",
    "StructuralSubreg",
    "STRUCTURAL_SUBREGS",
    "STRUCTURAL_VALUE_ARRAYS",
    "CodebookSpec",
    "CODEBOOK_SPECS",
    "CODEBOOK_TABLES",
    "OP_PRODUCER",
    "MACRO_OP_LOSS_TIERS",
    "non_atom_ops",
    "reference_ops",
    "reference_op_producers",
    "contract_emit_ops",
    "missing_contracts",
    "op_name_by_id",
    "op_name_tiers",
]


class MaskRole(Enum):
    """How constrained decode treats an op. ATOM = self-contained single emission, transparent to the
    structural mask/validator (only its reg classification matters); DISTANCE_PAIR / OVERLAY / LOOP_CTRL
    are the loop-op state machines; CODEBOOK_DEF/STEP/END/REF are the inline-redefinable dictionary ops
    whose REF is legal iff its id is live (enforced once B2 wires the live id-sets into AbsState).
    """

    ATOM = "atom"
    DISTANCE_PAIR = "distance_pair"
    OVERLAY = "overlay"
    LOOP_CTRL = "loop_ctrl"
    CODEBOOK_DEF = "codebook_def"
    CODEBOOK_STEP = "codebook_step"
    CODEBOOK_END = "codebook_end"
    CODEBOOK_REF = "codebook_ref"


@dataclass(frozen=True)
class OpContract:
    """One op's constrained-decode contract: its ``op_code`` and ``role``. The shape / legal_next /
    update logic the mask and validator replay is dispatched from ``role`` (filled in as each consumer
    is moved onto the registry); the registry's job today is to enumerate every emittable op exactly
    once so a missing implementation fails the completeness test rather than silently shipping.
    """

    op_code: int
    role: MaskRole


LOOP_OPS = (PATTERN_REPLAY_OP, PATTERN_OVERLAY_OP, DO_LOOP_OP)

_CONTRACT_LIST = (
    OpContract(SET_OP, MaskRole.ATOM),
    OpContract(DIFF_OP, MaskRole.ATOM),
    OpContract(FLIP_OP, MaskRole.ATOM),
    OpContract(TRANSPOSE_OP, MaskRole.ATOM),
    OpContract(SUBREG_FLUSH_OP, MaskRole.ATOM),
    OpContract(HARD_RESTART_OP, MaskRole.ATOM),
    OpContract(LEGATO_OP_CLUSTER_2, MaskRole.ATOM),
    OpContract(LEGATO_OP_CLUSTER_3, MaskRole.ATOM),
    OpContract(LEGATO_OP_CLUSTER_4, MaskRole.ATOM),
    OpContract(LEGATO_OP_CLUSTER_7, MaskRole.ATOM),
    OpContract(SWEEP_OP, MaskRole.ATOM),
    OpContract(GEN_TRI_OP, MaskRole.ATOM),
    OpContract(GEN_TUNING_OP, MaskRole.ATOM),
    OpContract(MELODY_INTERVAL_OP, MaskRole.ATOM),
    OpContract(NOTE_INTERVAL_OP, MaskRole.ATOM),
    OpContract(PATTERN_REPLAY_OP, MaskRole.DISTANCE_PAIR),
    OpContract(PATTERN_OVERLAY_OP, MaskRole.OVERLAY),
    OpContract(DO_LOOP_OP, MaskRole.LOOP_CTRL),
)

OP_CONTRACTS: dict[int, OpContract] = {int(c.op_code): c for c in _CONTRACT_LIST}


MACRO_OP_LOSS_TIERS: dict[int, str] = {
    int(SWEEP_OP): "content",
    int(GEN_TRI_OP): "content",
    int(MELODY_INTERVAL_OP): "content",
    int(NOTE_INTERVAL_OP): "content",
    int(GEN_TUNING_OP): "structural",
}
"""Loss tier for the generator/codebook ops, which are MacroPass-emitted (not ``Transform`` classes) so
``collect_op_loss_tiers`` cannot read a ``LOSS_TIER`` off them. Value-bearing atoms (the freq/pitch
trajectories + the codebook DEF body STEP) are ``content``; the codebook markers/pointers (DEF/END/REF)
and the per-voice tuning config (GEN_TUNING) are ``structural`` scaffolding. Merged into
``collect_op_loss_tiers`` so the per-tier loss / ``content_over_structural`` gate stops counting the
codebook + tuning structure as content."""


@dataclass(frozen=True)
class StructuralSubreg:
    """One row-slot of a structural loop op for the per-vocab precompute: the ``subreg`` that keys it,
    the boolean ``flag`` array it sets in ``precompute_vocab_arrays``, and the optional ``value_array``
    its ``val`` is scattered into. Lets the precompute build the PATTERN_REPLAY / PATTERN_OVERLAY
    classification arrays by iterating the registry instead of hand-listing (op, subreg) in three files.
    """

    subreg: int
    flag: str
    value_array: str | None = None
    consumes_gate: str | None = None


STRUCTURAL_SUBREGS: dict[int, tuple[StructuralSubreg, ...]] = {
    PATTERN_REPLAY_OP: (
        StructuralSubreg(
            PATTERN_REPLAY_SUBREG_DIST_HI, "is_pattern_replay_dist_hi", "dist_hi_val"
        ),
        StructuralSubreg(
            PATTERN_REPLAY_SUBREG_DIST_LO,
            "is_pattern_replay_dist_lo",
            "dist_lo_val",
            "consumes_pr_dist_lo_gate",
        ),
        StructuralSubreg(
            PATTERN_REPLAY_SUBREG_LEN,
            "is_pattern_replay_len",
            "length",
            "consumes_pr_len_gate",
        ),
        StructuralSubreg(
            PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
            "is_pattern_replay_ov_count",
            "overlay_count",
            "consumes_pr_ov_count_gate",
        ),
    ),
    PATTERN_OVERLAY_OP: (
        StructuralSubreg(
            PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
            "is_pattern_overlay_frame_offset",
            None,
            "consumes_overlay_slot_0_gate",
        ),
        StructuralSubreg(
            PATTERN_OVERLAY_SUBREG_TARGET_REG,
            "is_pattern_overlay_target_reg",
            None,
            "consumes_overlay_slot_1_gate",
        ),
        StructuralSubreg(
            PATTERN_OVERLAY_SUBREG_NEW_VAL,
            "is_pattern_overlay_new_val",
            None,
            "consumes_overlay_slot_2_gate",
        ),
    ),
}

STRUCTURAL_VALUE_ARRAYS: tuple[str, ...] = (
    "dist_hi_val",
    "dist_lo_val",
    "length",
    "overlay_count",
)

CODEBOOK_TABLES: tuple[str, ...] = ("gesture",)


@dataclass(frozen=True)
class CodebookSpec:
    """How constrained decode treats one inline-codebook op (RESID_ZERO_PHASE3 §4 B2): which ``table``
    its id lives in, its ``kind`` (``def`` opens an id, ``commit`` makes the pending id live, ``ref``
    replays an id and is legal iff that id is live), and the ``subreg`` carrying the id (a ``ref``) or
    triggering the commit (``None`` = the op's own row / always)."""

    op_code: int
    table: str
    kind: str
    subreg: int | None = None


CODEBOOK_SPECS: dict[int, CodebookSpec] = {
    op: CodebookSpec(op, table, kind, subreg)
    for op, (table, kind, subreg) in codebook_spec_tuples().items()
}


OP_PRODUCER: dict[int, str] = {
    PATTERN_REPLAY_OP: "LoopPass",
    PATTERN_OVERLAY_OP: "LoopPass",
    DO_LOOP_OP: "LoopPass",
}

_REFERENCE_ROLES = frozenset(
    {
        MaskRole.DISTANCE_PAIR,
        MaskRole.OVERLAY,
        MaskRole.LOOP_CTRL,
        MaskRole.CODEBOOK_REF,
    }
)


def non_atom_ops() -> set[int]:
    """Every op whose role is not ATOM: the multi-row structural ops (inline-codebook DEF/STEP/END/REF
    and the loop distance-pair / overlay / loop-ctrl ops). Their rows are a contiguous, order-load-bearing
    sequence -- a reorder that splits a DEF..END group or moves a REF before its DEF breaks reference
    integrity (``VoiceBlockOrderPass`` must leave frames containing one of these unreordered).
    """
    return {op for op, c in OP_CONTRACTS.items() if c.role is not MaskRole.ATOM}


def reference_ops() -> set[int]:
    """Ops whose decode resolves against earlier stream state (a back-ref distance, a loop, or an
    inline-codebook id): the DISTANCE_PAIR / OVERLAY / LOOP_CTRL / CODEBOOK_REF roles. These are the ops
    the per-block re-fire MUST re-emit so the model sees the reference, not its literal expansion.
    """
    return {op for op, c in OP_CONTRACTS.items() if c.role in _REFERENCE_ROLES}


def reference_op_producers() -> set[str]:
    """The MacroPass class names that emit a reference op (the producers the block-decoder contract
    requires in the per-block re-fire). Raises KeyError via ``OP_PRODUCER`` if a reference op has no
    declared producer -- a deliberate bite so a new ref op can't be added without wiring its pass.
    """
    return {OP_PRODUCER[op] for op in reference_ops()}


def contract_emit_ops() -> set[int]:
    """Every op the model can emit and the constrained decoder must therefore contract for: the atom
    decoders plus the loop ops. Derived programmatically (not a hand list) so a new decoder auto-extends
    the required set and the completeness test forces its contract."""
    return {int(op) for op in DECODERS} | {int(op) for op in LOOP_OPS}


def missing_contracts(emit_ops: set[int] | None = None) -> set[int]:
    """Emittable ops with no ``OpContract`` -- the completeness test asserts this is empty, and that a
    dummy op added to the emit set surfaces here (the registry bites)."""
    ops = contract_emit_ops() if emit_ops is None else emit_ops
    return set(ops) - set(OP_CONTRACTS)


def op_name_by_id() -> dict[int, str]:
    """Canonical ``{op_int: NAME}`` map -- tokens' single source of truth for op naming, so a consumer
    reads it here instead of re-deriving names by ``dir()``-scanning ``stfconstants`` (which couples it to
    raw constant *names* a rename silently changes). Scans every non-negative-int ``*_OP`` constant once;
    ``NAME`` is the constant name with the ``_OP`` token removed (``SET_OP`` -> ``"SET"``,
    ``LEGATO_OP_CLUSTER_2`` -> ``"LEGATO_CLUSTER_2"``). ``op_name_tiers`` joins this with op->tier.
    """
    out: dict[int, str] = {}
    for name in dir(_stfconstants):
        if "_OP" not in name:
            continue
        val = getattr(_stfconstants, name)
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            continue
        out[int(val)] = name.replace("_OP", "")
    return out


def op_name_tiers() -> dict[int, tuple[str, str]]:
    """``{op_int: (name, tier)}`` joining ``op_name_by_id`` (op->name) with
    ``collect_op_loss_tiers`` (op->tier) over the union of both maps -- name defaults to ``""`` for an op
    that carries a tier but no ``*_OP`` constant, tier to ``""`` for a named op with no declared tier.
    """
    # pylint: disable=import-outside-toplevel
    from preframr_tokens.macros.transform import collect_op_loss_tiers

    names = op_name_by_id()
    tiers = collect_op_loss_tiers()
    return {
        op: (names.get(op, ""), tiers.get(op, "")) for op in set(names) | set(tiers)
    }
