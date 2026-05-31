"""Single op-contract registry for constrained decode (RESID_ZERO_PHASE3 §4 B0): one ``OpContract`` per
op the model can emit -- the atom ops in ``DECODERS`` plus the loop ops ``expand_loops`` consumes -- so
the sampling mask, the stream validators, and the precompute arrays dispatch on one source of truth
instead of three hand-kept copies. Each op declares its ``MaskRole`` (how constrained decode treats it);
the completeness test goes red if any emittable op lacks a contract (the STAMP/PATCH drift, caught).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_DIST_HI,
    BACK_REF_SUBREG_DIST_LO,
    BACK_REF_SUBREG_LEN,
    CTRL_BIGRAM_OP,
    CTRL_TRIPLE_OP,
    CTRL_UPDATE_OP,
    DIFF_OP,
    DO_LOOP_OP,
    FC_PRESET_OP,
    FLIP_OP,
    FREQ_NUDGE_OP,
    FREQ_ONSET_OP,
    FREQ_TRAJ_OP,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_STEP_OP,
    PATTERN_OVERLAY_OP,
    PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
    PATTERN_OVERLAY_SUBREG_NEW_VAL,
    PATTERN_OVERLAY_SUBREG_TARGET_REG,
    PATTERN_REPLAY_OP,
    PATTERN_REPLAY_SUBREG_DIST_HI,
    PATTERN_REPLAY_SUBREG_DIST_LO,
    PATTERN_REPLAY_SUBREG_LEN,
    PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    PWM_PRESET_OP,
    PWM_PRESET_SHIFTED_OP,
    PWM_SUSTAIN_OP,
    RELEASE_UPDATE_OP,
    SET_OP,
    SKEL_OP,
    ORN_OP,
    PATCH_SUBREG_SR,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    STAMP_REL_SUBREG_ID,
    STAMP_STEP_OP,
    SUBREG_FLUSH_OP,
    SWEEP_OP,
    TRACK_REF_OP,
    TRANSPOSE_OP,
    WAVETABLE_SUSTAIN_OP,
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
    "contract_emit_ops",
    "missing_contracts",
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


LOOP_OPS = (BACK_REF_OP, PATTERN_REPLAY_OP, PATTERN_OVERLAY_OP, DO_LOOP_OP)

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
    OpContract(PWM_PRESET_OP, MaskRole.ATOM),
    OpContract(FC_PRESET_OP, MaskRole.ATOM),
    OpContract(PWM_PRESET_SHIFTED_OP, MaskRole.ATOM),
    OpContract(CTRL_BIGRAM_OP, MaskRole.ATOM),
    OpContract(PWM_SUSTAIN_OP, MaskRole.ATOM),
    OpContract(WAVETABLE_SUSTAIN_OP, MaskRole.ATOM),
    OpContract(FREQ_TRAJ_OP, MaskRole.ATOM),
    OpContract(TRACK_REF_OP, MaskRole.ATOM),
    OpContract(FREQ_NUDGE_OP, MaskRole.ATOM),
    OpContract(FREQ_ONSET_OP, MaskRole.ATOM),
    OpContract(RELEASE_UPDATE_OP, MaskRole.ATOM),
    OpContract(CTRL_TRIPLE_OP, MaskRole.ATOM),
    OpContract(CTRL_UPDATE_OP, MaskRole.ATOM),
    OpContract(SKEL_OP, MaskRole.ATOM),
    OpContract(ORN_OP, MaskRole.ATOM),
    OpContract(SWEEP_OP, MaskRole.ATOM),
    OpContract(STAMP_DEF_OP, MaskRole.CODEBOOK_DEF),
    OpContract(STAMP_STEP_OP, MaskRole.CODEBOOK_STEP),
    OpContract(STAMP_END_OP, MaskRole.CODEBOOK_END),
    OpContract(STAMP_REF_OP, MaskRole.CODEBOOK_REF),
    OpContract(STAMP_REL_REF_OP, MaskRole.CODEBOOK_REF),
    OpContract(PATCH_DEF_OP, MaskRole.CODEBOOK_DEF),
    OpContract(PATCH_STEP_OP, MaskRole.CODEBOOK_STEP),
    OpContract(PATCH_SET_OP, MaskRole.CODEBOOK_REF),
    OpContract(BACK_REF_OP, MaskRole.DISTANCE_PAIR),
    OpContract(PATTERN_REPLAY_OP, MaskRole.DISTANCE_PAIR),
    OpContract(PATTERN_OVERLAY_OP, MaskRole.OVERLAY),
    OpContract(DO_LOOP_OP, MaskRole.LOOP_CTRL),
)

OP_CONTRACTS: dict[int, OpContract] = {int(c.op_code): c for c in _CONTRACT_LIST}


@dataclass(frozen=True)
class StructuralSubreg:
    """One row-slot of a structural loop op for the per-vocab precompute: the ``subreg`` that keys it,
    the boolean ``flag`` array it sets in ``precompute_vocab_arrays``, and the optional ``value_array``
    its ``val`` is scattered into. Lets the precompute build the BACK_REF / PATTERN_REPLAY / PATTERN_OVERLAY
    classification arrays by iterating the registry instead of hand-listing (op, subreg) in three files.
    """

    subreg: int
    flag: str
    value_array: str | None = None
    consumes_gate: str | None = None


STRUCTURAL_SUBREGS: dict[int, tuple[StructuralSubreg, ...]] = {
    BACK_REF_OP: (
        StructuralSubreg(BACK_REF_SUBREG_DIST_HI, "is_back_ref_dist_hi", "dist_hi_val"),
        StructuralSubreg(
            BACK_REF_SUBREG_DIST_LO,
            "is_back_ref_dist_lo",
            "dist_lo_val",
            "consumes_back_ref_dist_lo_gate",
        ),
        StructuralSubreg(
            BACK_REF_SUBREG_LEN,
            "is_back_ref_len",
            "length",
            "consumes_back_ref_len_gate",
        ),
    ),
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

CODEBOOK_TABLES: tuple[str, ...] = ("stamp", "patch")


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
    STAMP_DEF_OP: CodebookSpec(STAMP_DEF_OP, "stamp", "def"),
    STAMP_END_OP: CodebookSpec(STAMP_END_OP, "stamp", "commit"),
    STAMP_REF_OP: CodebookSpec(STAMP_REF_OP, "stamp", "ref"),
    STAMP_REL_REF_OP: CodebookSpec(
        STAMP_REL_REF_OP, "stamp", "ref", STAMP_REL_SUBREG_ID
    ),
    PATCH_DEF_OP: CodebookSpec(PATCH_DEF_OP, "patch", "def"),
    PATCH_STEP_OP: CodebookSpec(PATCH_STEP_OP, "patch", "commit", PATCH_SUBREG_SR),
    PATCH_SET_OP: CodebookSpec(PATCH_SET_OP, "patch", "ref"),
}


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
