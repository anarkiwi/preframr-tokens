"""Tests for PresetPass."""

import unittest

import pandas as pd

from preframr_tokens.macros.preset_pass import (
    PresetPass,
    _preset_id_for,
    _snap,
)
from preframr_tokens.stfconstants import (
    FC_PRESET_OP,
    FC_PRESET_TABLE,
    FC_PRESET_VAL_TO_ID,
    FRAME_REG,
    MODEL_PDTYPE,
    PRESET_OPS,
    PRESET_REG_GRID,
    PRESET_REG_TO_OP,
    PWM_PRESET_OP,
    PWM_PRESET_TABLE,
    PWM_PRESET_VAL_TO_ID,
    SET_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame():
    return {
        "reg": FRAME_REG,
        "val": 0,
        "diff": 19000,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _row(reg, val, diff=32, op=SET_OP, subreg=-1):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": op,
        "subreg": subreg,
        "description": 0,
    }


class TestSnap(unittest.TestCase):
    def test_snap_grid_128(self):
        self.assertEqual(_snap(0, 128), 0)
        self.assertEqual(_snap(128, 128), 128)
        self.assertEqual(_snap(127, 128), 128)
        self.assertEqual(_snap(64, 128), 128)
        self.assertEqual(_snap(63, 128), 0)
        self.assertEqual(_snap(2048, 128), 2048)
        self.assertEqual(_snap(2049, 128), 2048)
        self.assertEqual(_snap(2050, 128), 2048)

    def test_snap_grid_256(self):
        self.assertEqual(_snap(0, 256), 0)
        self.assertEqual(_snap(8192, 256), 8192)
        self.assertEqual(_snap(8200, 256), 8192)
        self.assertEqual(_snap(8064, 256), 8192)

    def test_snap_negative(self):
        self.assertEqual(_snap(-128, 128), -128)
        self.assertEqual(_snap(-2050, 128), -2048)
        self.assertEqual(_snap(-8200, 256), -8192)


class TestPresetIdFor(unittest.TestCase):
    def test_pwm_preset_id(self):
        for reg in (2, 9, 16):
            self.assertEqual(_preset_id_for(reg, 0), 0)
            self.assertEqual(_preset_id_for(reg, 64), 1)
            self.assertEqual(_preset_id_for(reg, 2048), 32)
            self.assertEqual(_preset_id_for(reg, 4032), 63)

    def test_fc_preset_id(self):
        self.assertEqual(_preset_id_for(21, 0), 0)
        self.assertEqual(_preset_id_for(21, 256), 1)
        self.assertEqual(_preset_id_for(21, 8192), 32)

    def test_pwm_preset_id_caps(self):
        huge = PRESET_REG_GRID[2] * (len(PWM_PRESET_TABLE) + 5)
        for reg in (2, 9, 16):
            self.assertEqual(_preset_id_for(reg, huge), len(PWM_PRESET_TABLE) - 1)

    def test_fc_preset_id_caps(self):
        huge = PRESET_REG_GRID[21] * (len(FC_PRESET_TABLE) + 5)
        self.assertEqual(_preset_id_for(21, huge), len(FC_PRESET_TABLE) - 1)

    def test_negative_clamps_to_zero(self):
        self.assertEqual(_preset_id_for(2, -100), 0)
        self.assertEqual(_preset_id_for(9, -100), 0)
        self.assertEqual(_preset_id_for(16, -100), 0)
        self.assertEqual(_preset_id_for(21, -100), 0)


class TestPresetTableInverses(unittest.TestCase):
    def test_pwm_preset_table_inverse_consistency(self):
        for i, v in enumerate(PWM_PRESET_TABLE):
            self.assertEqual(PWM_PRESET_VAL_TO_ID[v], i)

    def test_fc_preset_table_inverse_consistency(self):
        for i, v in enumerate(FC_PRESET_TABLE):
            self.assertEqual(FC_PRESET_VAL_TO_ID[v], i)

    def test_pwm_preset_size(self):
        self.assertEqual(len(PWM_PRESET_TABLE), 64)

    def test_fc_preset_size(self):
        self.assertEqual(len(FC_PRESET_TABLE), 256)


class TestPresetOpMaps(unittest.TestCase):
    def test_inverse_consistency(self):
        for _reg, op in PRESET_REG_TO_OP.items():
            self.assertIn(op, PRESET_OPS)
        self.assertEqual(PRESET_REG_TO_OP[2], PWM_PRESET_OP)
        self.assertEqual(PRESET_REG_TO_OP[9], PWM_PRESET_OP)
        self.assertEqual(PRESET_REG_TO_OP[16], PWM_PRESET_OP)
        self.assertEqual(PRESET_REG_TO_OP[21], FC_PRESET_OP)


class TestPresetPass(unittest.TestCase):
    def _apply(self, df, **flags):
        flags.setdefault("preset_pass", True)
        return PresetPass().apply(df, args=FakeArgs(**flags))

    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame(
            [_frame(), _row(2, 2048), _frame(), _row(21, 8192)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df, preset_pass=False)
        self.assertTrue(result.equals(df))

    def test_pwm_preset_on_grid_zero_drift(self):
        df = pd.DataFrame(
            [_frame(), _row(2, 2048)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        presets = result[result["op"] == PWM_PRESET_OP]
        self.assertEqual(len(presets), 1)
        preset_id = int(presets.iloc[0]["val"])
        self.assertEqual(PWM_PRESET_TABLE[preset_id], 2048)

    def test_pwm_preset_off_grid_snaps(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 2050),
                _frame(),
                _row(2, 2049),
                _frame(),
                _row(2, 2070),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        presets = result[result["op"] == PWM_PRESET_OP]
        self.assertEqual(len(presets), 3)
        for _, row in presets.iterrows():
            preset_id = int(row["val"])
            self.assertEqual(PWM_PRESET_TABLE[preset_id], 2048)

    def test_fc_preset_on_grid_zero_drift(self):
        df = pd.DataFrame(
            [_frame(), _row(21, 8192)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        presets = result[result["op"] == FC_PRESET_OP]
        self.assertEqual(len(presets), 1)
        preset_id = int(presets.iloc[0]["val"])
        self.assertEqual(FC_PRESET_TABLE[preset_id], 8192)

    def test_fc_preset_off_grid_snaps(self):
        df = pd.DataFrame(
            [_frame(), _row(21, 8200)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        presets = result[result["op"] == FC_PRESET_OP]
        self.assertEqual(len(presets), 1)
        preset_id = int(presets.iloc[0]["val"])
        self.assertEqual(FC_PRESET_TABLE[preset_id], 8192)

    def test_excluded_regs_untouched(self):
        df = pd.DataFrame(
            [_frame(), _row(0, 12345), _frame(), _row(4, 65)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        any_preset = result["op"].isin(PRESET_OPS).any()
        self.assertFalse(bool(any_preset))

    def test_subreg_rows_untouched(self):
        df = pd.DataFrame(
            [_frame(), _row(2, 5, subreg=0), _frame(), _row(21, 7, subreg=1)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        any_preset = result["op"].isin(PRESET_OPS).any()
        self.assertFalse(bool(any_preset))

    def test_non_set_ops_untouched(self):
        df = pd.DataFrame(
            [_frame(), _row(2, 128, op=1)],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        any_preset = result["op"].isin(PRESET_OPS).any()
        self.assertFalse(bool(any_preset))

    def test_pwm_preset_covers_all_voices(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 2048),
                _frame(),
                _row(9, 1024),
                _frame(),
                _row(16, 256),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        presets = result[result["op"] == PWM_PRESET_OP]
        self.assertEqual(len(presets), 3)
        regs = sorted(int(r) for r in presets["reg"])
        self.assertEqual(regs, [2, 9, 16])

    def test_aggressive_snap_no_residual_set(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 100),
                _frame(),
                _row(2, 2048),
                _frame(),
                _row(9, 500),
                _frame(),
                _row(16, 1500),
                _frame(),
                _row(21, 200),
                _frame(),
                _row(21, 65000),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        for reg in (2, 9, 16, 21):
            residual = result[
                (result["reg"] == reg)
                & (result["op"] == SET_OP)
                & (result["subreg"] == -1)
            ]
            self.assertEqual(len(residual), 0, f"residual SET on reg={reg}")


if __name__ == "__main__":
    unittest.main()
