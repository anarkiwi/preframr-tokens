"""Inline-codebook family registry. The instrument / generator / gesture families are all retired -- the
events codec (``preframr_tokens.events``) now owns the freq/ctrl/adsr/pw/filter channels -- so the registry
is empty. It is kept as the (no-op) seam that ``op_contracts`` and ``decoders`` derive their codebook tables
from, so a future family can be declared in one place without re-wiring those consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CODEBOOK_TABLE_NAMES",
    "RefSpec",
    "CodebookFamily",
    "CODEBOOK_FAMILIES",
    "family_by_name",
    "family_for_op",
    "codebook_spec_tuples",
    "codebook_decoders",
    "DEAD_REF_POLICY",
]

DEAD_REF_POLICY = "drop"

CODEBOOK_TABLE_NAMES: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefSpec:
    """One REF op of a family: ``id_subreg`` carries the codebook id (or ``None`` when the op's own row
    does); ``table_less`` marks a ref that inlines its payload and looks up no id."""

    op: int
    id_subreg: int | None = None
    table_less: bool = False


@dataclass(frozen=True)
class CodebookFamily:
    """Declaration of one inline-codebook family: ``def_op``/``commit_op``/``commit_subreg``/``refs``
    derive the per-op ``CodebookSpec``; ``step_ops``/``def_emits`` drive the decoder. ``commit_subreg`` is
    ``None`` for a dedicated END op, else the terminal STEP subreg that commits."""

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
        """``op -> (table, kind, subreg)`` for the liveness-tracked ops (non-commit STEPs and table-less
        refs are omitted), matching the ``op_contracts.CODEBOOK_SPECS`` shape."""
        out: dict[int, tuple[str, str, int | None]] = {
            self.def_op: (self.name, "def", None),
            self.commit_op: (self.name, "commit", self.commit_subreg),
        }
        for r in self.refs:
            if not r.table_less:
                out[r.op] = (self.name, "ref", r.id_subreg)
        return out


CODEBOOK_FAMILIES: dict[str, CodebookFamily] = {}


def family_by_name(name: str) -> CodebookFamily:
    return CODEBOOK_FAMILIES[name]


_OP_TO_FAMILY: dict[int, CodebookFamily] = {
    op: fam for fam in CODEBOOK_FAMILIES.values() for op in fam.ops
}


def family_for_op(op: int) -> CodebookFamily | None:
    return _OP_TO_FAMILY.get(int(op))


def codebook_spec_tuples() -> dict[int, tuple[str, str, int | None]]:
    """Union of every family's ``spec_tuples`` (empty: no families) -- the registry-derived equivalent of
    ``op_contracts.CODEBOOK_SPECS``."""
    out: dict[int, tuple[str, str, int | None]] = {}
    for fam in CODEBOOK_FAMILIES.values():
        out.update(fam.spec_tuples())
    return out


def codebook_decoders() -> dict[int, object]:
    """``op -> decoder`` for every codebook op (empty: no families), merged into ``decoders.DECODERS``."""
    return {}
