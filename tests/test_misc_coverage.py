"""Small coverage-tightening tests for branches missed by the topical suites."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preframr_tokens.audit_primitives import detect_tail_cycle
from preframr_tokens.dump_meta import DumpMeta, meta_path_for, read_meta
from preframr_tokens.stfconstants import DUMP_SUFFIX


class TestDetectTailCycleSkipsImpossiblePeriods(unittest.TestCase):
    def test_period_skipped_when_repeats_dont_fit(self):
        """Force the ``continue`` branch where ``tail_window < period * min_repeats``."""
        tokens = [1] * 4 + [1, 2, 1, 2]
        out = detect_tail_cycle(tokens, tail_window=8, max_period=4, min_repeats=3)
        self.assertIsNone(out)


class TestDumpMetaProperties(unittest.TestCase):
    def test_per_frame_max_properties(self):
        meta = DumpMeta(
            dump_path="/tmp/x.dump.parquet",
            fields={
                "vol_changes_per_frame_max": 7,
                "ctrl_changes_per_frame_max": 11,
                "freq_writes_per_frame_max": 13,
            },
        )
        self.assertEqual(meta.vol_changes_per_frame_max, 7)
        self.assertEqual(meta.ctrl_changes_per_frame_max, 11)
        self.assertEqual(meta.freq_writes_per_frame_max, 13)

    def test_per_frame_max_defaults_to_zero(self):
        meta = DumpMeta(dump_path="/tmp/x.dump.parquet", fields={})
        self.assertEqual(meta.vol_changes_per_frame_max, 0)
        self.assertEqual(meta.ctrl_changes_per_frame_max, 0)


class TestReadMetaCorruptParquet(unittest.TestCase):
    def test_corrupt_parquet_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            dump_path = Path(td) / f"x{DUMP_SUFFIX}"
            mp = meta_path_for(dump_path)
            mp.write_bytes(b"not a parquet")
            self.assertIsNone(read_meta(dump_path))


class TestVoiceOfShiftable(unittest.TestCase):
    def test_fc_preset_returns_none(self):
        from preframr_tokens.macros.gate_slope_shift_pass import _voice_of_shiftable
        from preframr_tokens.stfconstants import FC_PRESET_OP, SLOPE_FREQ_LO_OP

        self.assertIsNone(_voice_of_shiftable(0, FC_PRESET_OP))
        self.assertEqual(_voice_of_shiftable(0, SLOPE_FREQ_LO_OP), 0)
        self.assertEqual(_voice_of_shiftable(7, SLOPE_FREQ_LO_OP), 1)

    def test_out_of_voice_range_returns_none(self):
        from preframr_tokens.macros.gate_slope_shift_pass import _voice_of_shiftable
        from preframr_tokens.stfconstants import SLOPE_FREQ_LO_OP

        self.assertIsNone(_voice_of_shiftable(99, SLOPE_FREQ_LO_OP))


class TestGateSlopeShiftPassEarlyExits(unittest.TestCase):
    def _pass(self, **flags):
        from preframr_tokens.macros.gate_slope_shift_pass import GateSlopeShiftPass

        return GateSlopeShiftPass(), _FakeArgs(**flags)

    def test_empty_df_returns_unchanged(self):
        gp, args = self._pass(gate_slope_shift_pass=True)
        empty = pd.DataFrame()
        self.assertTrue(gp.apply(empty, args=args) is empty)

    def test_flag_disabled_returns_unchanged(self):
        gp, args = self._pass(gate_slope_shift_pass=False)
        df = pd.DataFrame([{"reg": 0, "val": 1, "op": 0, "diff": 32}])
        out = gp.apply(df, args=args)
        self.assertTrue(out.equals(df))

    def test_no_frame_markers_returns_unchanged(self):
        gp, args = self._pass(gate_slope_shift_pass=True)
        df = pd.DataFrame([{"reg": 0, "val": 1, "op": 0, "diff": 32}])
        out = gp.apply(df, args=args)
        self.assertTrue(out.equals(df.reset_index(drop=True).copy()))


class _FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


class TestDecodedCtrlVal(unittest.TestCase):
    """Direct unit tests for ``transforms_set_to_diff._decoded_ctrl_val`` decoder dispatch."""

    def test_set_op_returns_low_byte(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import SET_OP

        self.assertEqual(_decoded_ctrl_val(int(SET_OP), 0x123), 0x23)

    def test_hard_restart_returns_low_byte(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import HARD_RESTART_OP

        self.assertEqual(_decoded_ctrl_val(int(HARD_RESTART_OP), 0xFE), 0xFE)

    def test_legato_nibble_op_shifts_nibble(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import LEGATO_OP_CLUSTER_2

        self.assertEqual(_decoded_ctrl_val(int(LEGATO_OP_CLUSTER_2), 0x05), 0x50)

    def test_legato_cluster_7_returns_low_byte(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import LEGATO_OP_CLUSTER_7

        self.assertEqual(_decoded_ctrl_val(int(LEGATO_OP_CLUSTER_7), 0xAB), 0xAB)

    def test_ctrl_bigram_op_returns_cur_byte(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import CTRL_BIGRAM_OP, CTRL_BIGRAM_TABLE

        _prev, cur = CTRL_BIGRAM_TABLE[0]
        self.assertEqual(_decoded_ctrl_val(int(CTRL_BIGRAM_OP), 0), int(cur))

    def test_ctrl_bigram_out_of_range_returns_none(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val
        from preframr_tokens.stfconstants import CTRL_BIGRAM_OP, CTRL_BIGRAM_TABLE

        self.assertIsNone(
            _decoded_ctrl_val(int(CTRL_BIGRAM_OP), len(CTRL_BIGRAM_TABLE) + 100)
        )

    def test_unknown_op_returns_none(self):
        from preframr_tokens.macros.transforms_set_to_diff import _decoded_ctrl_val

        self.assertIsNone(_decoded_ctrl_val(999, 0xAA))


class TestSnapGroup(unittest.TestCase):
    def test_empty_sorted_vals_passes_through(self):
        from preframr_tokens.alphabet_projection import _snap_group
        import numpy as np

        vals = np.array([1, 2, 3], dtype=np.int64)
        out = _snap_group(np.array([], dtype=np.int64), vals)
        np.testing.assert_array_equal(out, vals)

    def test_empty_vals_passes_through(self):
        from preframr_tokens.alphabet_projection import _snap_group
        import numpy as np

        vals = np.array([], dtype=np.int64)
        out = _snap_group(np.array([1, 2, 3], dtype=np.int64), vals)
        np.testing.assert_array_equal(out, vals)


if __name__ == "__main__":
    unittest.main()
