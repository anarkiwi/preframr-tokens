import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass
from preframr_tokens.macros.preset_pass import PresetPass
from preframr_tokens.macros.slope_pass import SlopePass
from preframr_tokens.stfconstants import (
    BASE_TO_SHIFTED_OP,
    DELAY_REG,
    FC_PRESET_OP,
    FRAME_REG,
    MODEL_PDTYPE,
    PWM_PRESET_OP,
    SET_OP,
    SLOPE_FC_LO_OP,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame(diff=19000):
    return {
        "reg": FRAME_REG,
        "val": 1,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


def _delay(val=1, diff=19000):
    return {
        "reg": DELAY_REG,
        "val": val,
        "diff": diff,
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


def _sid_writes(df):
    keep = df["reg"] >= 0
    cols = ["reg", "val", "diff"]
    return df.loc[keep, cols].reset_index(drop=True)


class TestGateSlopeShiftPassAudioInvariant(unittest.TestCase):
    def _expand_through_pass(self, baseline, run_slope=False, run_preset=False):
        df = baseline.copy()
        if run_slope:
            df = SlopePass().apply(df, args=FakeArgs(slope_pass=True))
        if run_preset:
            df = PresetPass().apply(df, args=FakeArgs(preset_pass=True))
        baseline_writes = _sid_writes(expand_ops(df, strict=False))
        shifted = GateSlopeShiftPass().apply(
            df.copy(), args=FakeArgs(gate_slope_shift_pass=True)
        )
        shifted_writes = _sid_writes(expand_ops(shifted, strict=False))
        return df, shifted, baseline_writes, shifted_writes

    def test_preset_pwm_gate_lonely_frame(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _frame(),
                _row(2, 0x400),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        _df, shifted, base_w, shift_w = self._expand_through_pass(
            baseline, run_preset=True
        )
        shifted_present = (shifted["op"] == BASE_TO_SHIFTED_OP[PWM_PRESET_OP]).any()
        self.assertTrue(shifted_present, "expected PWM_PRESET_SHIFTED_OP")
        pd.testing.assert_frame_equal(base_w, shift_w)

    def test_preset_fc_gate_lonely_frame(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _frame(),
                _row(21, 0x2000),
                _frame(),
                _row(0, 50),
            ],
            dtype=MODEL_PDTYPE,
        )
        _df, _shifted, base_w, shift_w = self._expand_through_pass(
            baseline, run_preset=True
        )
        pd.testing.assert_frame_equal(base_w, shift_w)

    def test_slope_3row_gate_lonely_frame(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _row(0, 10),
                _frame(),
                _row(4, 0x10),
                _row(0, 11),
                _frame(),
                _row(0, 12),
                _frame(),
                _row(0, 13),
                _frame(),
                _row(0, 14),
                _frame(),
                _row(4, 0x21),
            ],
            dtype=MODEL_PDTYPE,
        )
        _df, _shifted, base_w, shift_w = self._expand_through_pass(
            baseline, run_slope=True
        )
        pd.testing.assert_frame_equal(base_w, shift_w)

    def _shifted_vs_equivalent(self, delay_val):
        n_empty = delay_val
        shifted = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _row(2, 8, op=BASE_TO_SHIFTED_OP[PWM_PRESET_OP]),
                _delay(val=delay_val),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        equivalent_rows = [
            _frame(),
            _row(4, 0x11),
            _frame(),
            _row(4, 0x10),
        ]
        for _ in range(n_empty - 1):
            equivalent_rows.append(_frame())
        equivalent_rows.extend(
            [
                _frame(),
                _row(2, 8, op=PWM_PRESET_OP),
                _frame(),
                _row(0, 100),
            ]
        )
        equivalent = pd.DataFrame(equivalent_rows, dtype=MODEL_PDTYPE)
        s_writes = _sid_writes(expand_ops(shifted, strict=False))
        e_writes = _sid_writes(expand_ops(equivalent, strict=False))
        pd.testing.assert_frame_equal(s_writes, e_writes)

    def test_shifted_drained_at_delay_val_1(self):
        self._shifted_vs_equivalent(delay_val=1)

    def test_shifted_drained_at_delay_val_2(self):
        self._shifted_vs_equivalent(delay_val=2)

    def test_shifted_drained_at_delay_val_4(self):
        self._shifted_vs_equivalent(delay_val=4)

    def test_no_shift_when_no_gate_change(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(2, 0x400),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        df = PresetPass().apply(baseline.copy(), args=FakeArgs(preset_pass=True))
        shifted = GateSlopeShiftPass().apply(
            df.copy(), args=FakeArgs(gate_slope_shift_pass=True)
        )
        pd.testing.assert_frame_equal(df, shifted)

    def test_no_shift_when_dst_preceded_by_delay(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _delay(val=2),
                _frame(),
                _row(2, 0x400),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        df = PresetPass().apply(baseline.copy(), args=FakeArgs(preset_pass=True))
        shifted = GateSlopeShiftPass().apply(
            df.copy(), args=FakeArgs(gate_slope_shift_pass=True)
        )
        shifted_pwm = shifted[shifted["op"] == BASE_TO_SHIFTED_OP[PWM_PRESET_OP]]
        self.assertEqual(len(shifted_pwm), 0)

    def test_shifted_op_codes_emitted(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _frame(),
                _row(2, 0x400),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        df = PresetPass().apply(baseline.copy(), args=FakeArgs(preset_pass=True))
        shifted = GateSlopeShiftPass().apply(
            df, args=FakeArgs(gate_slope_shift_pass=True)
        )
        shifted_pwm = shifted[shifted["op"] == BASE_TO_SHIFTED_OP[PWM_PRESET_OP]]
        self.assertEqual(len(shifted_pwm), 1)

    def test_no_crash_on_fc_preset_body(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _frame(),
                _row(21, 0, op=FC_PRESET_OP),
                _frame(),
                _row(0, 100),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = GateSlopeShiftPass().apply(
            df.copy(), args=FakeArgs(gate_slope_shift_pass=True)
        )
        self.assertEqual(int((out["op"] == FC_PRESET_OP).sum()), 1)

    def test_no_crash_on_slope_fc_lo_body(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x11),
                _frame(),
                _row(4, 0x10),
                _frame(),
                _row(21, 0x10, op=SLOPE_FC_LO_OP, subreg=0),
                _row(21, 0x20, op=SLOPE_FC_LO_OP, subreg=1),
                _row(21, 4, op=SLOPE_FC_LO_OP, subreg=2),
                _frame(),
                _row(0, 50),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = GateSlopeShiftPass().apply(
            df.copy(), args=FakeArgs(gate_slope_shift_pass=True)
        )
        self.assertEqual(int((out["op"] == SLOPE_FC_LO_OP).sum()), 3)


if __name__ == "__main__":
    unittest.main()
