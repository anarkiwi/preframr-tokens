"""Completeness tests for the OpContract registry (RESID_ZERO_PHASE3 §4 B0): every op the model can emit
-- the DECODERS atom ops plus the loop ops -- must have exactly one OpContract, so a decoder added
without a constrained-decode contract fails here at unit-test time rather than silently shipping (the
codebook-family drift the registry exists to prevent). The registry bites: a dummy emittable op surfaces as
missing."""

import unittest

from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.macros.op_contracts import (
    LOOP_OPS,
    OP_CONTRACTS,
    STRUCTURAL_SUBREGS,
    STRUCTURAL_VALUE_ARRAYS,
    MaskRole,
    contract_emit_ops,
    missing_contracts,
    op_name_by_id,
    op_name_tiers,
)
from preframr_tokens.macros.transform import collect_op_loss_tiers
from preframr_tokens.stfconstants import (
    LEGATO_OP_CLUSTER_2,
    LOSS_TIER_NAMES,
    PATTERN_REPLAY_OP,
    SET_OP,
    SWEEP_OP,
)

_STRUCTURAL_ROLES = {MaskRole.DISTANCE_PAIR, MaskRole.OVERLAY}


class TestOpContractCompleteness(unittest.TestCase):
    def test_every_emittable_op_has_a_contract(self):
        self.assertEqual(
            missing_contracts(), set(), "emittable ops without an OpContract"
        )

    def test_registry_superset_of_decoders_and_loop_ops(self):
        self.assertTrue(set(OP_CONTRACTS) >= {int(op) for op in DECODERS})
        self.assertTrue(set(OP_CONTRACTS) >= {int(op) for op in LOOP_OPS})

    def test_no_orphan_contracts(self):
        self.assertEqual(
            set(OP_CONTRACTS) - contract_emit_ops(),
            set(),
            "contracts for ops that cannot be emitted (dead registry entries)",
        )

    def test_contract_op_code_matches_key(self):
        for op_code, contract in OP_CONTRACTS.items():
            self.assertEqual(op_code, contract.op_code)
            self.assertIsInstance(contract.role, MaskRole)

    def test_completeness_check_bites_on_missing_contract(self):
        dummy_op = max(contract_emit_ops()) + 1
        self.assertNotIn(dummy_op, OP_CONTRACTS)
        augmented = contract_emit_ops() | {dummy_op}
        self.assertIn(dummy_op, missing_contracts(augmented))


class TestStructuralRegistryConsistency(unittest.TestCase):
    def test_structural_subregs_exactly_cover_structural_role_ops(self):
        structural_role_ops = {
            op for op, c in OP_CONTRACTS.items() if c.role in _STRUCTURAL_ROLES
        }
        self.assertEqual(set(STRUCTURAL_SUBREGS), structural_role_ops)

    def test_structural_flag_names_unique(self):
        flags = [sf.flag for specs in STRUCTURAL_SUBREGS.values() for sf in specs]
        self.assertEqual(len(flags), len(set(flags)), "duplicate structural flag name")

    def test_structural_value_arrays_are_declared(self):
        for specs in STRUCTURAL_SUBREGS.values():
            for sf in specs:
                if sf.value_array is not None:
                    self.assertIn(sf.value_array, STRUCTURAL_VALUE_ARRAYS)


class TestOpNameApi(unittest.TestCase):
    """The canonical op->name API (PW/filter SWEEP brief Part B): tokens owns op->name (the constant
    name with the ``_OP`` token removed) so a downstream consumer reads it here instead of re-deriving
    names by ``dir()``-scanning stfconstants."""

    def test_known_op_names(self):
        names = op_name_by_id()
        self.assertEqual(names[SET_OP], "SET")
        self.assertEqual(names[PATTERN_REPLAY_OP], "PATTERN_REPLAY")
        self.assertEqual(names[SWEEP_OP], "SWEEP")
        self.assertEqual(names[LEGATO_OP_CLUSTER_2], "LEGATO_CLUSTER_2")

    def test_every_contract_op_has_a_name(self):
        names = op_name_by_id()
        missing = [op for op in OP_CONTRACTS if op not in names]
        self.assertEqual(missing, [], "OP_CONTRACTS ops without a name")

    def test_op_name_tiers_joins_names_and_tiers(self):
        joined = op_name_tiers()
        tiers = collect_op_loss_tiers()
        names = op_name_by_id()
        self.assertEqual(set(joined), set(names) | set(tiers))
        for op, tier in tiers.items():
            self.assertEqual(joined[op], (names.get(op, ""), tier))
            self.assertTrue(tier == "" or tier in LOSS_TIER_NAMES)


if __name__ == "__main__":
    unittest.main()
