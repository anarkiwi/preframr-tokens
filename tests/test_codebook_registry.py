"""The CodebookFamily registry (preframr_tokens.macros.codebook) is the single source of
truth for the inline-codebook families. These tests pin it against the legacy hand-written
``op_contracts.CODEBOOK_SPECS`` / ``CODEBOOK_TABLES`` so the two cannot drift, and enforce
that every spec'd op belongs to a registered family (the guard that catches a codebook op
added upstream without a family registration)."""

from preframr_tokens.macros import codebook
from preframr_tokens.macros.op_contracts import CODEBOOK_SPECS, CODEBOOK_TABLES


def test_table_names_match_legacy():
    assert codebook.CODEBOOK_TABLE_NAMES == CODEBOOK_TABLES


def test_derived_specs_match_legacy_literal():
    """codebook.codebook_spec_tuples() reproduces op_contracts.CODEBOOK_SPECS exactly."""
    legacy = {
        op: (spec.table, spec.kind, spec.subreg) for op, spec in CODEBOOK_SPECS.items()
    }
    derived = codebook.codebook_spec_tuples()
    assert derived == legacy, (
        f"registry-derived specs diverge from legacy literal:\n"
        f"  only in legacy: {set(legacy) - set(derived)}\n"
        f"  only in derived: {set(derived) - set(legacy)}\n"
        f"  value mismatches: "
        f"{ {op: (legacy[op], derived[op]) for op in set(legacy) & set(derived) if legacy[op] != derived[op]} }"
    )


def test_every_legacy_spec_op_has_a_family():
    """Resilience guard: every op in CODEBOOK_SPECS resolves to a registered family. If an
    upstream change adds a codebook op without registering a CodebookFamily, this fails with
    the offending ops."""
    orphan = {op for op in CODEBOOK_SPECS if codebook.family_for_op(op) is None}
    assert not orphan, (
        f"ops in CODEBOOK_SPECS with no CodebookFamily — register one in "
        f"preframr_tokens/macros/codebook.py: {sorted(orphan)}"
    )


def test_family_ops_are_disjoint():
    """No op code is claimed by two families."""
    seen: dict[int, str] = {}
    for fam in codebook.CODEBOOK_FAMILIES.values():
        for op in fam.ops:
            assert op not in seen, f"op {op} claimed by both {seen[op]} and {fam.name}"
            seen[op] = fam.name


def test_table_index_is_stable():
    for i, name in enumerate(codebook.CODEBOOK_TABLE_NAMES):
        assert codebook.family_by_name(name).table_index == i
