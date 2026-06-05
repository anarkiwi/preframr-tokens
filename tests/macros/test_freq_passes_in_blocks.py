"""Regression guard: ``iter_self_contained_row_blocks`` calls ``run_freq_block_passes``
after ``expand_to_literal_form`` so FREQ_TRAJ atoms re-appear in the encoded stream.
Without this, every freq macro produced by ``RegLogParser.parse`` is silently
decompiled to literal SETs before tokenization and the model never trains on op45
(the bug that zeroed melody from mini A/Bs)."""

from __future__ import annotations

import unittest

import pandas as pd

from preframr_tokens import macros
from preframr_tokens.macros.blocks import iter_self_contained_row_blocks
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.macros.trajectory_anchor import TrajectoryAnchorPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_TRAJ_OP,
    SET_OP,
)


class FakeArgs:
    seq_len = 64
    cents = 50
    instrument_pass = False
    hard_restart_pass = False
    legato_pass_c2 = False
    legato_pass_c3 = False
    legato_pass_c4 = False
    legato_pass_c7 = False
    fuzzy_loop_pass = False
    preset_pass = True
    freq_trajectory_pass = True
    trajectory_anchor_pass = False
    freq_v0_interval = False


def _frame_row(diff=19656):
    return {
        "reg": FRAME_REG,
        "val": 0,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _set_row(reg, val, diff=32):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _ramp_df(reg=0, start=100, n=12):
    """Literal-form SET ramp on a freq reg -- exact pattern FreqTrajectoryPass
    detects and replaces with op45 atoms."""
    rows = []
    for i in range(n):
        rows.append(_frame_row())
        rows.append(_set_row(reg, start + i))
    rows.append(_frame_row())
    return pd.DataFrame(rows)


class TestFreqBlockPassesContract(unittest.TestCase):
    def test_required_freq_passes_present_in_freq_block_passes(self):
        present = {type(p) for p in macros.FREQ_BLOCK_PASSES}
        required = {
            TrajectoryAnchorPass,
            FreqTrajectoryPass,
            PerRegBurstPass,
        }
        missing = required - present
        self.assertFalse(
            missing,
            "FREQ_BLOCK_PASSES must include every freq-encoder pass so they "
            "re-fire inside iter_self_contained_row_blocks after "
            f"expand_to_literal_form decompiles them; missing: "
            f"{sorted(c.__name__ for c in missing)}",
        )

    def test_freq_block_passes_not_in_passes_list(self):
        passes_types = {type(p) for p in macros.PASSES}
        freq_types = {type(p) for p in macros.FREQ_BLOCK_PASSES}
        overlap = passes_types & freq_types
        self.assertFalse(
            overlap,
            "Freq passes must NOT also live in PASSES -- run_passes fires inside the "
            "rotation loop AFTER parse() already ran the freq passes once, and "
            "double-firing breaks lossless-roundtrip fidelity. They belong in "
            f"FREQ_BLOCK_PASSES only; overlap: {sorted(c.__name__ for c in overlap)}",
        )


class TestIterSelfContainedRowBlocksPreservesFreqOps(unittest.TestCase):
    def test_literal_freq_ramp_yields_freq_traj_atoms(self):
        df = _ramp_df(reg=0, start=100, n=12)
        args = FakeArgs()
        seen_freq_traj = 0
        seen_blocks = 0
        for block in iter_self_contained_row_blocks(df, 8, args=args):
            if block.empty:
                continue
            seen_blocks += 1
            seen_freq_traj += int((block["op"] == FREQ_TRAJ_OP).sum())
        self.assertGreater(seen_blocks, 0, "expected at least one non-empty block")
        self.assertGreater(
            seen_freq_traj,
            0,
            "iter_self_contained_row_blocks lost every FREQ_TRAJ atom -- "
            "run_freq_block_passes is not being called after expand_to_literal_form. "
            "This is the bug that silently zeroed melody from training streams.",
        )


if __name__ == "__main__":
    unittest.main()
