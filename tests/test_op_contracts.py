"""Completeness tests for the OpContract registry (RESID_ZERO_PHASE3 §4 B0): every op the model can emit
-- the DECODERS atom ops plus the loop ops -- must have exactly one OpContract, so a decoder added
without a constrained-decode contract fails here at unit-test time rather than silently shipping (the
STAMP/PATCH drift the registry exists to prevent). The registry bites: a dummy emittable op surfaces as
missing."""

import unittest

from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.macros.op_contracts import (
    LOOP_OPS,
    OP_CONTRACTS,
    MaskRole,
    contract_emit_ops,
    missing_contracts,
)


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


if __name__ == "__main__":
    unittest.main()
