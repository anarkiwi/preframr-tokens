"""Tests for SlopePass."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.slope_pass import (
    SlopePass,
    _detect_runs,
    _quantise_terminal,
    _split_runtime,
    quantise_slope_runtime,
)
from preframr_tokens.stfconstants import (
    FRAME_REG,
    MODEL_PDTYPE,
    SET_OP,
    SLOPE_FC_LO_OP,
    SLOPE_FREQ_LO_OP,
    SLOPE_MAX_RUNTIME,
    SLOPE_OPS,
    SLOPE_PW_LO_OP,
    SLOPE_REG_TO_OP,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
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


def _row(reg, val, diff=32):
    return {
        "reg": reg,
        "val": val,
        "diff": diff,
        "op": SET_OP,
        "subreg": -1,
        "description": 0,
    }


class TestQuantiseSlopeRuntime(unittest.TestCase):
    def test_low_range_exact(self):
        for n in range(1, 17):
            self.assertEqual(quantise_slope_runtime(n), n)

    def test_buckets_snap(self):
        self.assertEqual(quantise_slope_runtime(17), 16)
        self.assertEqual(quantise_slope_runtime(24), 32)
        self.assertEqual(quantise_slope_runtime(32), 32)
        self.assertEqual(quantise_slope_runtime(33), 32)
        self.assertEqual(quantise_slope_runtime(48), 64)
        self.assertEqual(quantise_slope_runtime(64), 64)
        self.assertEqual(quantise_slope_runtime(128), 128)
        self.assertEqual(quantise_slope_runtime(256), 256)

    def test_above_max_returns_none(self):
        self.assertIsNone(quantise_slope_runtime(257))
        self.assertIsNone(quantise_slope_runtime(512))

    def test_zero_or_negative(self):
        self.assertIsNone(quantise_slope_runtime(0))
        self.assertIsNone(quantise_slope_runtime(-5))


class TestSplitRuntime(unittest.TestCase):
    def test_no_split_under_max(self):
        self.assertEqual(_split_runtime(256), [256])
        self.assertEqual(_split_runtime(100), [100])

    def test_even_split(self):
        chunks = _split_runtime(512)
        self.assertEqual(sum(chunks), 512)
        self.assertTrue(all(c <= SLOPE_MAX_RUNTIME for c in chunks))
        self.assertEqual(chunks, [256, 256])

    def test_uneven_split(self):
        chunks = _split_runtime(513)
        self.assertEqual(sum(chunks), 513)
        self.assertTrue(all(c <= SLOPE_MAX_RUNTIME for c in chunks))


class TestDetectRuns(unittest.TestCase):
    def test_constant_run(self):
        runs = _detect_runs([5, 5, 5, 5, 5])
        self.assertEqual(runs, [(0, 5, 0)])

    def test_linear_up(self):
        runs = _detect_runs([1, 2, 3, 4, 5])
        self.assertEqual(runs, [(0, 5, 1)])

    def test_tolerance_pm1(self):
        runs = _detect_runs([1, 3, 4, 7, 9])
        self.assertEqual(len(runs), 1)
        ofs, run_len, slope = runs[0]
        self.assertEqual(ofs, 0)
        self.assertEqual(run_len, 5)
        self.assertEqual(slope, 2)

    def test_run_below_min_len_rejected(self):
        self.assertEqual(_detect_runs([1, 2]), [])
        self.assertEqual(_detect_runs([1, 2, 3]), [])
        self.assertEqual(_detect_runs([1, 2, 3, 4]), [])


class TestSlopeOpMaps(unittest.TestCase):
    def test_freq_pw_per_voice_share_op(self):
        for v in range(3):
            self.assertEqual(SLOPE_REG_TO_OP[v * 7], SLOPE_FREQ_LO_OP)
            self.assertEqual(SLOPE_REG_TO_OP[v * 7 + 2], SLOPE_PW_LO_OP)
        self.assertEqual(SLOPE_REG_TO_OP[21], SLOPE_FC_LO_OP)
        for op in SLOPE_OPS:
            self.assertIn(op, set(SLOPE_REG_TO_OP.values()))

    def test_excludes_ctrl_adsr_filter_mode(self):
        for reg in (4, 5, 6, 11, 12, 13, 18, 19, 20, 23, 24):
            self.assertNotIn(reg, SLOPE_REG_TO_OP)


class TestQuantiseTerminal(unittest.TestCase):
    def test_freq_lo_no_snap(self):
        self.assertEqual(_quantise_terminal(0, 12345), 12345)

    def test_pw_lo_snaps_to_32(self):
        self.assertEqual(_quantise_terminal(2, 2050), 2048)
        self.assertEqual(_quantise_terminal(2, 2049), 2048)
        self.assertEqual(_quantise_terminal(2, 2080), 2080)
        self.assertEqual(_quantise_terminal(2, 16), 32)
        self.assertEqual(_quantise_terminal(2, 15), 0)

    def test_fc_lo_snaps_to_256(self):
        self.assertEqual(_quantise_terminal(21, 8200), 8192)
        self.assertEqual(_quantise_terminal(21, 8000), 7936)
        self.assertEqual(_quantise_terminal(21, 8064), 8192)

    def test_negative_snaps(self):
        self.assertEqual(_quantise_terminal(2, -2050), -2048)
        self.assertEqual(_quantise_terminal(21, -8200), -8192)


class TestSlopePass(unittest.TestCase):
    def _apply(self, df, **flags):
        flags.setdefault("slope_pass", True)
        return SlopePass().apply(df, args=FakeArgs(**flags))

    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 1),
                _frame(),
                _row(0, 2),
                _frame(),
                _row(0, 3),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df, slope_pass=False)
        self.assertTrue(result.equals(df))

    def test_simple_linear_emits_slope(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 10),
                _frame(),
                _row(0, 11),
                _frame(),
                _row(0, 12),
                _frame(),
                _row(0, 13),
                _frame(),
                _row(0, 14),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_FREQ_LO_OP]
        self.assertEqual(len(slope), 3)
        subregs = list(slope["subreg"])
        self.assertEqual(
            subregs,
            [
                SLOPE_SUBREG_TERMINAL_HI,
                SLOPE_SUBREG_TERMINAL_LO,
                SLOPE_SUBREG_RUNTIME,
            ],
        )
        hi_row = slope.iloc[0]
        lo_row = slope.iloc[1]
        terminal = (int(hi_row["val"]) << 8) | int(lo_row["val"])
        self.assertEqual(terminal, 14)
        runtime = int(slope.iloc[2]["val"])
        self.assertEqual(runtime, 4)
        sets_for_reg0 = result[
            (result["reg"] == 0) & (result["op"] == SET_OP) & (result["subreg"] == -1)
        ]
        self.assertEqual(len(sets_for_reg0), 1)
        self.assertEqual(int(sets_for_reg0.iloc[0]["val"]), 10)

    def test_constant_run_emits_slope_zero(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 5),
                _frame(),
                _row(0, 5),
                _frame(),
                _row(0, 5),
                _frame(),
                _row(0, 5),
                _frame(),
                _row(0, 5),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_FREQ_LO_OP]
        self.assertEqual(len(slope), 3)
        hi = int(slope.iloc[0]["val"])
        lo = int(slope.iloc[1]["val"])
        terminal = (hi << 8) | lo
        self.assertEqual(terminal, 5)

    def test_short_run_left_alone(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 1),
                _frame(),
                _row(0, 2),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        if "op" in result.columns:
            any_slope = result["op"].isin(SLOPE_OPS).any()
            self.assertFalse(bool(any_slope))

    def test_chained_slope(self):
        rows = [_frame(), _row(0, 0)]
        for k in range(1, 600):
            rows.append(_frame())
            rows.append(_row(0, k))
        df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        result = self._apply(df)
        slope_runtime_rows = result[
            (result["op"] == SLOPE_FREQ_LO_OP)
            & (result["subreg"] == SLOPE_SUBREG_RUNTIME)
        ]
        self.assertGreaterEqual(len(slope_runtime_rows), 2)

    def test_no_op_safety_no_slopes(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 1),
                _frame(),
                _row(0, 100),
                _frame(),
                _row(0, 5),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        if "op" in result.columns:
            any_slope = result["op"].isin(SLOPE_OPS).any()
            self.assertFalse(bool(any_slope))

    def test_excluded_reg_ctrl_does_not_slope(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 10),
                _frame(),
                _row(4, 11),
                _frame(),
                _row(4, 12),
                _frame(),
                _row(4, 13),
                _frame(),
                _row(4, 14),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        if "op" in result.columns:
            any_slope = result["op"].isin(SLOPE_OPS).any()
            self.assertFalse(bool(any_slope))

    def test_non_eligible_reg_does_not_slope(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(1, 10),
                _frame(),
                _row(1, 11),
                _frame(),
                _row(1, 12),
                _frame(),
                _row(1, 13),
                _frame(),
                _row(1, 14),
            ],
            dtype=MODEL_PDTYPE,
        )
        result = self._apply(df)
        if "op" in result.columns:
            any_slope = result["op"].isin(SLOPE_OPS).any()
            self.assertFalse(bool(any_slope))

    def test_pw_lo_terminal_quantises_to_32(self):
        rows = [_frame(), _row(2, 0)]
        for k in (512, 1024, 1536, 2048):
            rows.append(_frame())
            rows.append(_row(2, k))
        df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_PW_LO_OP]
        self.assertEqual(len(slope), 3)
        hi = int(slope.iloc[0]["val"])
        lo = int(slope.iloc[1]["val"])
        terminal = (hi << 8) | lo
        self.assertEqual(terminal, 2048)
        self.assertEqual(terminal % 32, 0)

    def test_pw_lo_terminal_quantises_off_grid_value(self):
        rows = [_frame(), _row(2, 0)]
        for k in (10, 20, 30, 40):
            rows.append(_frame())
            rows.append(_row(2, k))
        df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_PW_LO_OP]
        self.assertEqual(len(slope), 3)
        hi = int(slope.iloc[0]["val"])
        lo = int(slope.iloc[1]["val"])
        terminal = (hi << 8) | lo
        self.assertEqual(terminal, 32)

    def test_fc_lo_terminal_quantises_to_256(self):
        rows = [_frame(), _row(21, 0)]
        for k in (2048, 4096, 6144, 8192):
            rows.append(_frame())
            rows.append(_row(21, k))
        df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_FC_LO_OP]
        self.assertEqual(len(slope), 3)
        hi = int(slope.iloc[0]["val"])
        lo = int(slope.iloc[1]["val"])
        terminal = (hi << 8) | lo
        self.assertEqual(terminal, 8192)
        self.assertEqual(terminal % 256, 0)

    def test_fc_lo_terminal_quantises_off_grid_value(self):
        rows = [_frame(), _row(21, 0)]
        for k in (100, 200, 300, 400):
            rows.append(_frame())
            rows.append(_row(21, k))
        df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        result = self._apply(df)
        slope = result[result["op"] == SLOPE_FC_LO_OP]
        self.assertEqual(len(slope), 3)
        hi = int(slope.iloc[0]["val"])
        lo = int(slope.iloc[1]["val"])
        terminal = (hi << 8) | lo
        self.assertEqual(terminal, 512)


class TestSlopeRoundTrip(unittest.TestCase):
    def test_decoder_emits_per_frame_writes(self):
        baseline = pd.DataFrame(
            [
                _frame(),
                _row(0, 10),
                _frame(),
                _row(0, 11),
                _frame(),
                _row(0, 12),
                _frame(),
                _row(0, 13),
                _frame(),
                _row(0, 14),
            ]
        )
        encoded = SlopePass().apply(baseline.copy(), args=FakeArgs(slope_pass=True))
        slope = encoded[encoded["op"] == SLOPE_FREQ_LO_OP]
        self.assertEqual(len(slope), 3)
        baseline_writes = expand_ops(baseline, strict=False).reset_index(drop=True)
        encoded_writes = expand_ops(encoded, strict=False).reset_index(drop=True)
        b_reg0 = baseline_writes[baseline_writes["reg"] == 0]["val"].tolist()
        e_reg0 = encoded_writes[encoded_writes["reg"] == 0]["val"].tolist()
        self.assertEqual(b_reg0, e_reg0)


if __name__ == "__main__":
    unittest.main()
