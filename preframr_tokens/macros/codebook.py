"""Single source of truth for the inline-codebook macro families (STAMP, PATCH,
WAVETABLE, CTRL_WT).

Every inline codebook shares one abstract machine: a DEF op opens a pending entry, STEP
op(s) accumulate its payload, a COMMIT (a dedicated END op, or a STEP at a terminal
subreg) makes the id live in an ``id -> entry`` table, and REF op(s) replay a live id.
The lifecycle, the id table, the mid-song seed/materialization, and the multi-frame replay
drain are identical across families; only the *payload codec* (how STEP rows serialize
into an entry, how a REF replays it into register writes) and the *replay target* differ.

This module is the registry of that structure. It is a **leaf** (depends only on
``stfconstants``) so the decoder dispatch, the validators, and the constrained-decode mask
can all build on it without an import cycle. Today it derives the per-op
``CodebookSpec`` table that ``op_contracts.CODEBOOK_SPECS`` exposes; the decode-side hooks
(``step_ops`` / replay) are declared here too so a single ``CodebookFamily`` registration
is enough to teach decode, validation, and the legality mask about a family.

To add a new codebook family: append its table name to ``CODEBOOK_TABLE_NAMES`` and
register one ``CodebookFamily`` in ``CODEBOOK_FAMILIES``. ``tests/test_codebook_registry.py``
and ``tests/test_codebook_consistency.py`` enforce that the registration is complete and
agrees with the legacy spec.
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

# Ordered table names; the index is the table id used by the seed snapshot, the validator,
# the constrained-decode mask, and ``DecodeState``. Append-only (never reorder: the index is
# persisted in mask seeds). Mirrors ``op_contracts.CODEBOOK_TABLES``.
CODEBOOK_TABLE_NAMES: tuple[str, ...] = ("stamp", "patch", "wavetable", "ctrl_wt")


@dataclass(frozen=True)
class RefSpec:
    """One REF op of a family. ``id_subreg`` is the subreg carrying the codebook id (a
    multi-row ref like WAVETABLE_REF / STAMP_REL_REF), or ``None`` when the op's own row
    carries the id (a single-row ref like PATCH_SET / CTRL_WT_SET / STAMP_REF).
    ``table_less`` marks a ref that carries its payload inline and looks up no id
    (WAVETABLE_ONESHOT); such ops are not liveness-tracked and stay out of the spec table.
    """

    op: int
    id_subreg: int | None = None
    table_less: bool = False


@dataclass(frozen=True)
class CodebookFamily:
    """The complete declaration of one inline-codebook family.

    Structural fields (``def_op`` / ``commit_op`` / ``commit_subreg`` / ``refs``) derive the
    per-op ``CodebookSpec`` consumed by validation and the legality mask. ``step_ops`` and the
    commit semantics additionally drive the unified decoder (added incrementally; see
    ``agents.md``). ``commit_subreg`` is ``None`` when ``commit_op`` is a dedicated END op,
    else it is the terminal STEP subreg that triggers the commit (PATCH/CTRL_WT).
    """

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
        """Every op code this family owns (def, steps, commit, refs incl. table-less)."""
        out = {self.def_op, self.commit_op, *self.step_ops}
        out.update(r.op for r in self.refs)
        return frozenset(out)

    def spec_tuples(self) -> dict[int, tuple[str, str, int | None]]:
        """``op -> (table, kind, subreg)`` for the liveness-tracked ops, matching the legacy
        ``op_contracts.CODEBOOK_SPECS`` shape. STEP ops that are not the commit are not
        liveness-relevant and are omitted (as in the legacy literal); table-less refs are
        omitted (they look up no id)."""
        out: dict[int, tuple[str, str, int | None]] = {
            self.def_op: (self.name, "def", None),
            self.commit_op: (self.name, "commit", self.commit_subreg),
        }
        for r in self.refs:
            if r.table_less:
                continue
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
    """The union of every family's ``spec_tuples`` — the registry-derived equivalent of
    ``op_contracts.CODEBOOK_SPECS`` in ``op -> (table, kind, subreg)`` form. The registry
    test asserts this equals the legacy literal so the two cannot drift."""
    out: dict[int, tuple[str, str, int | None]] = {}
    for fam in CODEBOOK_FAMILIES.values():
        out.update(fam.spec_tuples())
    return out
