"""Single source of truth for the inline-codebook families (STAMP, PATCH, WAVETABLE, CTRL_WT).
Each shares one machine: DEF opens a pending entry, STEP(s) accumulate its payload, a COMMIT
(an END op or a STEP at a terminal subreg) makes the id live in an ``id -> entry`` table, and
REF(s) replay a live id. This leaf (stfconstants only) declares that as ``CodebookFamily``
records and derives the ``op_contracts.CODEBOOK_SPECS`` table the registry test pins equal.
"""

from __future__ import annotations

from dataclasses import dataclass

from preframr_tokens.stfconstants import (
    CTRL_WT_DEF_OP,
    CTRL_WT_SET_OP,
    CTRL_WT_STEP_OP,
    CTRL_WT_SUBREG_VAL,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_STEP_OP,
    PATCH_SUBREG_SR,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    STAMP_REL_SUBREG_ID,
    STAMP_STEP_OP,
    WAVETABLE_DEF_OP,
    WAVETABLE_END_OP,
    WAVETABLE_ONESHOT_OP,
    WAVETABLE_REF_OP,
    WAVETABLE_STEP_OP,
    WT_REF_SUBREG_ID,
)

__all__ = [
    "CODEBOOK_TABLE_NAMES",
    "RefSpec",
    "CodebookFamily",
    "CODEBOOK_FAMILIES",
    "family_by_name",
    "family_for_op",
    "codebook_spec_tuples",
]

CODEBOOK_TABLE_NAMES: tuple[str, ...] = ("stamp", "patch", "wavetable", "ctrl_wt")


@dataclass(frozen=True)
class RefSpec:
    """One REF op of a family. ``id_subreg`` is the subreg carrying the codebook id (a
    multi-row ref like WAVETABLE_REF/STAMP_REL_REF) or ``None`` when the op's own row carries
    it (a single-row ref like PATCH_SET/CTRL_WT_SET/STAMP_REF). ``table_less`` marks a ref that
    carries its payload inline and looks up no id (WAVETABLE_ONESHOT); such ops are not
    liveness-tracked and stay out of the spec table."""

    op: int
    id_subreg: int | None = None
    table_less: bool = False


@dataclass(frozen=True)
class CodebookFamily:
    """Complete declaration of one inline-codebook family. ``def_op``/``commit_op``/
    ``commit_subreg``/``refs`` derive the per-op ``CodebookSpec`` consumed by validation and the
    legality mask; ``step_ops`` and ``def_emits`` additionally drive the unified decoder.
    ``commit_subreg`` is ``None`` when ``commit_op`` is a dedicated END op, else the terminal
    STEP subreg that triggers the commit (PATCH/CTRL_WT)."""

    name: str
    def_op: int
    commit_op: int
    refs: tuple[RefSpec, ...]
    commit_subreg: int | None = None
    step_ops: tuple[int, ...] = ()
    def_emits: bool = False

    @property
    def table_index(self) -> int:
        return CODEBOOK_TABLE_NAMES.index(self.name)

    @property
    def ops(self) -> frozenset[int]:
        """Every op code this family owns: def, steps, commit, and refs (incl. table-less)."""
        out = {self.def_op, self.commit_op, *self.step_ops}
        out.update(r.op for r in self.refs)
        return frozenset(out)

    def spec_tuples(self) -> dict[int, tuple[str, str, int | None]]:
        """``op -> (table, kind, subreg)`` for the liveness-tracked ops, matching the legacy
        ``op_contracts.CODEBOOK_SPECS`` shape. Non-commit STEP ops and table-less refs are
        omitted (they look up no id), exactly as in the legacy literal."""
        out: dict[int, tuple[str, str, int | None]] = {
            self.def_op: (self.name, "def", None),
            self.commit_op: (self.name, "commit", self.commit_subreg),
        }
        for r in self.refs:
            if not r.table_less:
                out[r.op] = (self.name, "ref", r.id_subreg)
        return out


CODEBOOK_FAMILIES: dict[str, CodebookFamily] = {
    "stamp": CodebookFamily(
        name="stamp",
        def_op=STAMP_DEF_OP,
        step_ops=(STAMP_STEP_OP,),
        commit_op=STAMP_END_OP,
        commit_subreg=None,
        refs=(
            RefSpec(STAMP_REF_OP),
            RefSpec(STAMP_REL_REF_OP, id_subreg=STAMP_REL_SUBREG_ID),
        ),
        def_emits=False,
    ),
    "patch": CodebookFamily(
        name="patch",
        def_op=PATCH_DEF_OP,
        step_ops=(PATCH_STEP_OP,),
        commit_op=PATCH_STEP_OP,
        commit_subreg=PATCH_SUBREG_SR,
        refs=(RefSpec(PATCH_SET_OP),),
        def_emits=True,
    ),
    "wavetable": CodebookFamily(
        name="wavetable",
        def_op=WAVETABLE_DEF_OP,
        step_ops=(WAVETABLE_STEP_OP,),
        commit_op=WAVETABLE_END_OP,
        commit_subreg=None,
        refs=(
            RefSpec(WAVETABLE_REF_OP, id_subreg=WT_REF_SUBREG_ID),
            RefSpec(WAVETABLE_ONESHOT_OP, table_less=True),
        ),
        def_emits=False,
    ),
    "ctrl_wt": CodebookFamily(
        name="ctrl_wt",
        def_op=CTRL_WT_DEF_OP,
        step_ops=(CTRL_WT_STEP_OP,),
        commit_op=CTRL_WT_STEP_OP,
        commit_subreg=CTRL_WT_SUBREG_VAL,
        refs=(RefSpec(CTRL_WT_SET_OP),),
        def_emits=True,
    ),
}


def family_by_name(name: str) -> CodebookFamily:
    return CODEBOOK_FAMILIES[name]


_OP_TO_FAMILY: dict[int, CodebookFamily] = {
    op: fam for fam in CODEBOOK_FAMILIES.values() for op in fam.ops
}


def family_for_op(op: int) -> CodebookFamily | None:
    return _OP_TO_FAMILY.get(int(op))


def codebook_spec_tuples() -> dict[int, tuple[str, str, int | None]]:
    """Union of every family's ``spec_tuples`` -- the registry-derived ``op -> (table, kind,
    subreg)`` equivalent of ``op_contracts.CODEBOOK_SPECS``, asserted equal by the registry test.
    """
    out: dict[int, tuple[str, str, int | None]] = {}
    for fam in CODEBOOK_FAMILIES.values():
        out.update(fam.spec_tuples())
    return out
