"""Smoke tests for ``preframr_tokens.constrained_decode``. Comprehensive coverage lives in the main `preframr` repo (`tests/predict/test_constrained_decode.py`) where torch is available; here we only verify the module imports torch-free and the precompute helpers return the expected numpy structure."""

import importlib
import sys
import unittest

import numpy as np
import pandas as pd

from preframr_tokens.constrained_decode import (
    StreamState,
    _frame_marker_count,
    precompute_vocab_arrays,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    PAD_REG,
    SET_OP,
)


class TestModuleTorchFree(unittest.TestCase):
    def test_constrained_decode_imports_without_torch(self):
        sys.modules.pop("preframr_tokens.constrained_decode", None)
        torch_was_present = sys.modules.pop("torch", None)
        try:
            importlib.import_module("preframr_tokens.constrained_decode")
            self.assertNotIn("torch", sys.modules)
        finally:
            if torch_was_present is not None:
                sys.modules["torch"] = torch_was_present


class TestPrecomputeVocabArrays(unittest.TestCase):
    def _df(self):
        rows = [
            {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
            {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
            {"op": SET_OP, "reg": DELAY_REG, "subreg": -1, "val": 1},
            {"op": SET_OP, "reg": 0, "subreg": -1, "val": 42},
        ]
        return pd.DataFrame(rows)

    def test_returns_numpy_dict_no_torch_tensors(self):
        arrs = precompute_vocab_arrays(self._df())
        for key, val in arrs.items():
            if isinstance(val, np.ndarray):
                self.assertEqual(val.shape[0], arrs["n_vocab"])
            else:
                self.assertIn(key, ("n_vocab", "subtoken_mode"))

    def test_is_pad_marks_pad_reg(self):
        arrs = precompute_vocab_arrays(self._df())
        self.assertTrue(bool(arrs["is_pad"][0]))
        self.assertFalse(bool(arrs["is_pad"][1]))

    def test_is_frame_marker_covers_frame_and_delay(self):
        arrs = precompute_vocab_arrays(self._df())
        self.assertTrue(bool(arrs["is_frame_marker"][1]))
        self.assertTrue(bool(arrs["is_frame_marker"][2]))
        self.assertFalse(bool(arrs["is_frame_marker"][3]))


class TestFrameMarkerCount(unittest.TestCase):
    def test_counts_only_frame_markers(self):
        is_frame_marker = np.array([False, True, True, False])
        self.assertEqual(_frame_marker_count([0, 1, 2, 3], is_frame_marker), 2)
        self.assertEqual(_frame_marker_count([], is_frame_marker), 0)


class TestStreamStateConstructsTorchFree(unittest.TestCase):
    def test_atomic_state_constructs(self):
        df = pd.DataFrame(
            [
                {"op": SET_OP, "reg": PAD_REG, "subreg": -1, "val": 0},
                {"op": SET_OP, "reg": FRAME_REG, "subreg": -1, "val": 0},
            ]
        )
        arrs = precompute_vocab_arrays(df)
        state = StreamState(arrs, init_frame_count=0, irq=10000)
        invalid = state._compute_invalid()  # pylint: disable=protected-access
        self.assertEqual(invalid.dtype, np.bool_)
        self.assertEqual(invalid.shape, (arrs["n_vocab"],))


if __name__ == "__main__":
    unittest.main()
