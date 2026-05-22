"""Tests for ``preframr.macros``."""

import unittest

import pandas as pd

from preframr_tokens.macros import (
    DecodeState,
    Flip2Pass,
    HardRestartPass,
    LoopPass,
    OVERLAY_BODY_FREQ_DELTA,
    OVERLAY_BODY_FREQ_DELTA_BIN,
    SubregPass,
    TransposePass,
    _bin_body_freq_delta,
    _build_last_diff,
    expand_loops,
    iter_self_contained_row_blocks,
    run_passes,
    validate_back_refs,
    validate_pattern_overlays,
)
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    DIFF_OP,
    DO_LOOP_OP,
    FLIP2_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    _MIN_DIFF,
    PATTERN_OVERLAY_OP,
    PATTERN_REPLAY_OP,
    LOOP_OP_REG,
    MODEL_PDTYPE,
    SET_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICE_REG_SIZE,
)


class FakeArgs:
    """Args fixture; pass kwargs become attributes."""

    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _frame(diff=19000):
    return {
        "reg": FRAME_REG,
        "subreg": -1,
        "val": 0,
        "diff": diff,
        "op": SET_OP,
        "description": 0,
    }


def _row(reg, val, op=SET_OP, diff=32, subreg=-1):
    return {
        "reg": reg,
        "subreg": subreg,
        "val": val,
        "diff": diff,
        "op": op,
        "description": 0,
    }


def _expand(df):
    """Run the dispatcher and return the expanded SID write df."""
    return expand_ops(df, strict=False).reset_index(drop=True).astype(MODEL_PDTYPE)


def _assert_round_trip(_test, baseline_df, encoded_df):
    """Both forms must expand to byte-identical SID write streams."""
    pd.testing.assert_frame_equal(
        _expand(baseline_df.copy()),
        _expand(encoded_df.copy()),
        check_dtype=False,
    )


class TestSubregPass(unittest.TestCase):
    def test_lo_only_change_becomes_subreg_0(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x40, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        rows = result[(result["reg"] == 4) & (result["op"] == SET_OP)]
        self.assertEqual(len(rows), 2)
        subregs = rows["subreg"].tolist()
        vals = rows["val"].tolist()
        self.assertEqual(subregs, [1, 0])
        self.assertEqual(vals, [4, 1])

    def test_both_nibbles_change_kept_as_full_byte_set(self):
        df = pd.DataFrame([_frame(), _row(4, 0x41, op=SET_OP)])
        result = SubregPass().apply(df)
        sub_rows = result[
            (result["reg"] == 4) & (result["op"] == SET_OP) & (result["subreg"] != -1)
        ]
        self.assertEqual(len(sub_rows), 0)
        full_rows = result[
            (result["reg"] == 4) & (result["op"] == SET_OP) & (result["subreg"] == -1)
        ]
        self.assertEqual(len(full_rows), 1)
        self.assertEqual(int(full_rows["val"].iloc[0]), 0x41)

    def test_no_change_left_alone(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x41, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        sub01 = result[(result["reg"] == 4) & (result["subreg"].isin([0, 1]))]
        self.assertEqual(len(sub01), 0)
        full = result[(result["reg"] == 4) & (result["subreg"] == -1)]
        self.assertEqual(len(full), 2)

    def test_unaffected_regs_pass_through(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 100, op=SET_OP),
                _row(7, 200, op=SET_OP),
            ]
        )
        result = SubregPass().apply(df)
        self.assertEqual(len(result[result["subreg"] != -1]), 0)

    def test_round_trip_lone_nibbles(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x40, op=SET_OP),
                _frame(),
                _row(4, 0x41, op=SET_OP),
                _frame(),
                _row(4, 0x40, op=SET_OP),
                _frame(),
                _row(4, 0xC0, op=SET_OP),
                _frame(),
                _row(4, 0xC1, op=SET_OP),
            ]
        )
        encoded = SubregPass().apply(df.copy())
        _assert_round_trip(self, df, encoded)

    def test_round_trip_mixed_lone_and_paired(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(5, 0x00, op=SET_OP),
                _frame(),
                _row(5, 0x35, op=SET_OP),
                _frame(),
                _row(5, 0x36, op=SET_OP),
                _frame(),
                _row(5, 0x86, op=SET_OP),
            ]
        )
        encoded = SubregPass().apply(df.copy())
        _assert_round_trip(self, df, encoded)

    def test_case_3_inserts_flush_to_preserve_intermediate(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),
                _row(4, 0x65, op=SET_OP),
            ]
        )
        encoded = SubregPass().apply(df.copy())
        flushes = encoded[encoded["op"] == SUBREG_FLUSH_OP]
        self.assertEqual(len(flushes), 1)
        _assert_round_trip(self, df, encoded)

    def test_no_flush_for_both_nib_split(self):
        df = pd.DataFrame([_frame(), _row(4, 0x35, op=SET_OP)])
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)

    def test_no_flush_when_decoder_naturally_flushes(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),
                _row(4, 0x07, op=SET_OP),
            ]
        )
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)

    def test_no_flush_across_different_regs(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 0x05, op=SET_OP),
                _row(5, 0x03, op=SET_OP),
                _row(4, 0x65, op=SET_OP),
            ]
        )
        encoded = SubregPass().apply(df.copy())
        self.assertEqual(int((encoded["op"] == SUBREG_FLUSH_OP).sum()), 0)
        _assert_round_trip(self, df, encoded)


class TestTransposePass(unittest.TestCase):
    def test_encode_two_voices_same_delta(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),
                _row(7, 24, op=DIFF_OP),
                _row(14, -8, op=DIFF_OP),
            ]
        )
        result = TransposePass().apply(df, args=FakeArgs(transpose_pass=True))
        trans = result[result["op"] == TRANSPOSE_OP]
        self.assertEqual(len(trans), 1)
        self.assertEqual(int(trans.iloc[0]["val"]), 24)
        self.assertEqual(int(trans.iloc[0]["subreg"]), 0b011)
        v2 = result[(result["reg"] == 14) & (result["op"] == DIFF_OP)]
        self.assertEqual(len(v2), 1)

    def test_no_collapse_when_only_one_voice(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),
                _row(7, 8, op=DIFF_OP),
            ]
        )
        result = TransposePass().apply(df, args=FakeArgs(transpose_pass=True))
        self.assertEqual(len(result[result["op"] == TRANSPOSE_OP]), 0)

    def test_round_trip(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 24, op=DIFF_OP),
                _row(7, 24, op=DIFF_OP),
                _row(14, 24, op=DIFF_OP),
            ]
        )
        encoded = TransposePass().apply(df.copy(), args=FakeArgs(transpose_pass=True))
        _assert_round_trip(self, df, encoded)


class TestFlip2Pass(unittest.TestCase):
    def test_encode_asymmetric_run(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
            ]
        )
        result = Flip2Pass().apply(df, args=FakeArgs(flip2_pass=True))
        flips = result[result["op"] == FLIP2_OP]
        self.assertEqual(len(flips), 1)
        self.assertEqual(int(flips.iloc[0]["subreg"]), 4)

    def test_skips_symmetric(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -5, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -5, op=DIFF_OP),
            ]
        )
        result = Flip2Pass().apply(df, args=FakeArgs(flip2_pass=True))
        self.assertEqual(len(result[result["op"] == FLIP2_OP]), 0)

    def test_round_trip(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
                _frame(),
                _row(2, 5, op=DIFF_OP),
                _frame(),
                _row(2, -3, op=DIFF_OP),
            ]
        )
        encoded = Flip2Pass().apply(df.copy(), args=FakeArgs(flip2_pass=True))
        _assert_round_trip(self, df, encoded)


class TestHardRestartPass(unittest.TestCase):
    CTRL_V0 = 4
    CTRL_V1 = 11

    def _pair_df(self, a, b, ctrl_reg=CTRL_V0):
        return pd.DataFrame(
            [
                _frame(),
                _row(ctrl_reg, a, op=SET_OP),
                _row(ctrl_reg, b, op=SET_OP),
            ]
        )

    def test_encode_galway_test_pulse(self):
        df = self._pair_df(a=0x08, b=0x11)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        ops = result[result["op"] == HARD_RESTART_OP]
        self.assertEqual(len(ops), 1)
        self.assertEqual(int(ops.iloc[0]["reg"]), self.CTRL_V0)
        self.assertEqual(int(ops.iloc[0]["val"]), (0x08 << 8) | 0x11)
        leftover = result[(result["reg"] == self.CTRL_V0) & (result["op"] == SET_OP)]
        self.assertEqual(len(leftover), 0)

    def test_encode_test_plus_gate_flavor(self):
        df = self._pair_df(a=0x09, b=0x21)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        ops = result[result["op"] == HARD_RESTART_OP]
        self.assertEqual(len(ops), 1)
        self.assertEqual(int(ops.iloc[0]["val"]), (0x09 << 8) | 0x21)

    def test_encode_hubbard_gate_clear(self):
        df = self._pair_df(a=0x40, b=0x41)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        ops = result[result["op"] == HARD_RESTART_OP]
        self.assertEqual(len(ops), 1)
        self.assertEqual(int(ops.iloc[0]["val"]), (0x40 << 8) | 0x41)

    def test_skips_non_restart_pair(self):
        df = self._pair_df(a=0x11, b=0x21)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 0)

    def test_skips_ctrl_second_without_gate(self):
        df = self._pair_df(a=0x08, b=0x10)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 0)

    def test_skips_ctrl_second_with_test_bit(self):
        df = self._pair_df(a=0x08, b=0x19)
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 0)

    def test_skips_cross_frame_pair(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _frame(),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
            ]
        )
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 0)

    def test_off_by_default(self):
        df = self._pair_df(a=0x08, b=0x11)
        result = HardRestartPass().apply(df, args=FakeArgs())
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 0)
        leftover = result[(result["reg"] == self.CTRL_V0) & (result["op"] == SET_OP)]
        self.assertEqual(len(leftover), 2)

    def test_multiple_pairs_same_voice(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(self.CTRL_V0, 0x21, op=SET_OP),
            ]
        )
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        ops = result[result["op"] == HARD_RESTART_OP]
        self.assertEqual(len(ops), 2)
        leftover = result[(result["reg"] == self.CTRL_V0) & (result["op"] == SET_OP)]
        self.assertEqual(len(leftover), 0)

    def test_other_voice_unaffected(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
                _row(self.CTRL_V1, 0x21, op=SET_OP),
            ]
        )
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 1)
        v1_sets = result[(result["reg"] == self.CTRL_V1) & (result["op"] == SET_OP)]
        self.assertEqual(len(v1_sets), 1)

    def test_three_ctrl_sets_pairs_first_two_only(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
                _row(self.CTRL_V0, 0x10, op=SET_OP),
            ]
        )
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 1)
        leftover = result[(result["reg"] == self.CTRL_V0) & (result["op"] == SET_OP)]
        self.assertEqual(len(leftover), 1)
        self.assertEqual(int(leftover.iloc[0]["val"]), 0x10)

    def test_intervening_freq_set_ok(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(0, 0xAB, op=SET_OP),
                _row(1, 0xCD, op=SET_OP),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
            ]
        )
        result = HardRestartPass().apply(df, args=FakeArgs(hard_restart_pass=True))
        self.assertEqual(len(result[result["op"] == HARD_RESTART_OP]), 1)

    def test_round_trip_galway(self):
        df = self._pair_df(a=0x08, b=0x11)
        encoded = HardRestartPass().apply(
            df.copy(), args=FakeArgs(hard_restart_pass=True)
        )
        _assert_round_trip(self, df, encoded)

    def test_round_trip_test_plus_gate(self):
        df = self._pair_df(a=0x09, b=0x21)
        encoded = HardRestartPass().apply(
            df.copy(), args=FakeArgs(hard_restart_pass=True)
        )
        _assert_round_trip(self, df, encoded)

    def test_round_trip_hubbard(self):
        df = self._pair_df(a=0x40, b=0x41)
        encoded = HardRestartPass().apply(
            df.copy(), args=FakeArgs(hard_restart_pass=True)
        )
        _assert_round_trip(self, df, encoded)

    def test_round_trip_multiple_pairs(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(self.CTRL_V0, 0x08, op=SET_OP),
                _row(self.CTRL_V0, 0x11, op=SET_OP),
                _frame(),
                _row(self.CTRL_V0, 0x40, op=SET_OP),
                _row(self.CTRL_V0, 0x41, op=SET_OP),
            ]
        )
        encoded = HardRestartPass().apply(
            df.copy(), args=FakeArgs(hard_restart_pass=True)
        )
        _assert_round_trip(self, df, encoded)


class TestDecodeState(unittest.TestCase):
    def test_tick_frame_consumes_pending_diffs(self):
        state = DecodeState(frame_diff=19000)
        state.last_diff[2] = 32
        state.pending_diffs[2].extend([32, 32])
        writes_a = state.tick_frame()
        self.assertEqual(len(writes_a), 1)
        self.assertEqual(writes_a[0], (2, 32, 32))
        writes_b = state.tick_frame()
        self.assertEqual(len(writes_b), 1)
        self.assertEqual(writes_b[0], (2, 64, 32))
        self.assertEqual(state.tick_frame(), [])

    def test_tick_frame_independent_pending_diffs_per_reg(self):
        state = DecodeState(frame_diff=19000)
        state.last_diff[2] = 32
        state.last_diff[9] = 32
        state.pending_diffs[2].extend([10, 10])
        state.pending_diffs[9].extend([5])
        writes = state.tick_frame()
        regs = sorted(w[0] for w in writes)
        self.assertEqual(regs, [2, 9])


def _back_ref_row(distance, length, diff=32):
    """Build the triple-row encoding of one BACK_REF: DIST_HI, DIST_LO, LEN."""
    from preframr_tokens.stfconstants import (
        BACK_REF_SUBREG_DIST_HI,
        BACK_REF_SUBREG_DIST_LO,
        BACK_REF_SUBREG_LEN,
    )

    return [
        {
            "reg": LOOP_OP_REG,
            "subreg": BACK_REF_SUBREG_DIST_HI,
            "val": (int(distance) >> 8) & 0xFF,
            "diff": diff,
            "op": BACK_REF_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": BACK_REF_SUBREG_DIST_LO,
            "val": int(distance) & 0xFF,
            "diff": diff,
            "op": BACK_REF_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": BACK_REF_SUBREG_LEN,
            "val": int(length),
            "diff": diff,
            "op": BACK_REF_OP,
            "description": 0,
        },
    ]


def _do_loop_begin_row(n, diff=32):
    return {
        "reg": LOOP_OP_REG,
        "subreg": 0,
        "val": n,
        "diff": diff,
        "op": DO_LOOP_OP,
        "description": 0,
    }


def _do_loop_end_row(diff=32):
    return {
        "reg": LOOP_OP_REG,
        "subreg": 1,
        "val": 0,
        "diff": diff,
        "op": DO_LOOP_OP,
        "description": 0,
    }


def _pattern_replay_row(distance, length, num_overlays=0, diff=32):
    """Build the quad-row encoding of PATTERN_REPLAY: HI, LO, LEN, OV_COUNT."""
    from preframr_tokens.stfconstants import (
        PATTERN_REPLAY_SUBREG_DIST_HI,
        PATTERN_REPLAY_SUBREG_DIST_LO,
        PATTERN_REPLAY_SUBREG_LEN,
        PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    )

    return [
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_DIST_HI,
            "val": (int(distance) >> 8) & 0xFF,
            "diff": diff,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_DIST_LO,
            "val": int(distance) & 0xFF,
            "diff": diff,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_LEN,
            "val": int(length),
            "diff": diff,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
            "val": int(num_overlays),
            "diff": diff,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
    ]


def _pattern_overlay_row(frame_offset, target_reg, new_val, diff=32):
    """Triple-row PATTERN_OVERLAY (frame_offset, target_reg, new_val).
    Caller splats with ``*_pattern_overlay_row(...)``.
    """
    from preframr_tokens.stfconstants import (
        PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
        PATTERN_OVERLAY_SUBREG_TARGET_REG,
        PATTERN_OVERLAY_SUBREG_NEW_VAL,
    )

    return [
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
            "val": int(frame_offset),
            "diff": diff,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_TARGET_REG,
            "val": int(target_reg),
            "diff": diff,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_NEW_VAL,
            "val": int(new_val),
            "diff": diff,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
    ]


class TestBuildDecodeState(unittest.TestCase):
    """Coverage for ``_build_decode_state`` -- the consolidated
    DecodeState constructor that replaced 5 open-coded sites in the
    pass apply methods."""

    def test_returns_none_without_frame_reg(self):
        from preframr_tokens.macros import _build_decode_state

        df = pd.DataFrame([_row(4, 8, op=SET_OP)])
        self.assertIsNone(_build_decode_state(df))

    def test_seeds_frame_diff_from_first_frame(self):
        from preframr_tokens.macros import _build_decode_state

        df = pd.DataFrame([_frame(diff=20000), _row(4, 8, op=SET_OP)])
        state = _build_decode_state(df)
        self.assertEqual(state.frame_diff, 20000)

    def test_seeds_last_diff_per_reg(self):
        from preframr_tokens.macros import _build_decode_state

        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP, diff=24),
                _row(5, 10, op=SET_OP, diff=48),
            ]
        )
        state = _build_decode_state(df)
        self.assertEqual(state.last_diff[4], 24)
        self.assertEqual(state.last_diff[5], 48)


class TestBuildLastDiffFallback(unittest.TestCase):
    """Documents `_build_last_diff`'s _MIN_DIFF fallback for regs whose
    only writes come from macro decoders (PLAY_INSTRUMENT body, INTERVAL
    mirror) rather than top-level SETs.
    """

    def test_first_set_diff_seeds_last_diff(self):
        """First SET row on a reg defines `last_diff[reg]`; subsequent
        SETs and other ops do NOT override the seed."""
        df = pd.DataFrame(
            [
                _row(1, 0xAA, op=SET_OP, diff=200),
                _row(1, 0xBB, op=SET_OP, diff=50),
                _row(1, 0xCC, op=DIFF_OP, diff=75),
            ]
        )
        last_diff = _build_last_diff(df)
        self.assertEqual(last_diff[1], 200)

    def test_decoder_only_reg_falls_back_to_min_diff(self):
        df = pd.DataFrame(
            [
                _row(0, 0x10, op=FLIP2_OP, diff=50, subreg=4),
            ]
        )
        last_diff = _build_last_diff(df)
        self.assertEqual(last_diff[0], _MIN_DIFF)

    def test_diff_op_alone_does_not_seed(self):
        """DIFF rows on a reg without any SET also fall back to
        _MIN_DIFF -- only SET rows seed the lookup. This is by design
        (the seed represents the reg's initial-write cycle count, which
        only an explicit SET establishes)."""
        df = pd.DataFrame(
            [
                _row(2, 0x40, op=DIFF_OP, diff=200),
            ]
        )
        last_diff = _build_last_diff(df)
        self.assertEqual(last_diff[2], _MIN_DIFF)

    def test_decode_state_diff_for_unseeded_reg_is_min_diff(self):
        from preframr_tokens.macros import _build_decode_state

        df = pd.DataFrame(
            [
                _frame(diff=19000),
                _row(4, 0x40, op=SET_OP, diff=200),
            ]
        )
        state = _build_decode_state(df)
        self.assertEqual(state.diff_for(4), 200)
        self.assertEqual(state.diff_for(99), _MIN_DIFF)


class TestOverlayBodyFreqDeltaCents(unittest.TestCase):
    """OVERLAY_BODY_FREQ_DELTA quantization vs. ``--cents`` quantization."""

    def test_bin_is_zero_within_half_bin(self):
        """Inputs in [-bin_w/2, bin_w/2) round to 0."""
        bin_w = OVERLAY_BODY_FREQ_DELTA_BIN
        for d in range(-bin_w // 2 + 1, bin_w // 2):
            self.assertEqual(_bin_body_freq_delta(d), 0, f"d={d}")

    def test_bin_returns_multiple_of_bin_width(self):
        """Every output is a multiple of OVERLAY_BODY_FREQ_DELTA_BIN."""
        bin_w = OVERLAY_BODY_FREQ_DELTA_BIN
        for d in range(-200, 201):
            out = _bin_body_freq_delta(d)
            self.assertEqual(
                out % bin_w,
                0,
                f"_bin_body_freq_delta({d}) = {out}, not a multiple of {bin_w}",
            )

    def test_bin_is_symmetric_around_zero(self):
        """``_bin(d) == -_bin(-d)`` so positive and negative deltas
        round symmetrically (no asymmetric drift on transposed-down
        loops vs transposed-up)."""
        for d in range(1, 200):
            self.assertEqual(
                _bin_body_freq_delta(d),
                -_bin_body_freq_delta(-d),
                f"asymmetric rounding at d={d}",
            )

    def test_bin_drift_bounded_by_half_bin(self):
        """``|_bin(d) - d| <= bin_w / 2`` for every input. This is the
        documented "microtonally invisible" bound -- the maximum
        deviation between the represented delta and the true delta."""
        bin_w = OVERLAY_BODY_FREQ_DELTA_BIN
        half = bin_w // 2
        for d in range(-200, 201):
            self.assertLessEqual(
                abs(_bin_body_freq_delta(d) - d),
                half,
                f"_bin_body_freq_delta({d}) drift exceeds bin_w/2={half}",
            )

    def test_decoder_adds_bin_quantized_delta_integer_exact(self):
        """End-to-end: a PATTERN_REPLAY with a body-wide freq-delta
        overlay produces replayed val = original_val + bin_delta,
        integer-exact. No off-grid leakage from the bin / cents
        mismatch -- the operation is pure integer arithmetic."""
        cases = [
            (100, 16),
            (100, -16),
            (100, 32),
            (100, 48),
            (1, 16),
            (200, -32),
        ]
        for src_val, bin_delta in cases:
            with self.subTest(src_val=src_val, bin_delta=bin_delta):
                df = pd.DataFrame(
                    [
                        _frame(),
                        _row(0, src_val, op=SET_OP),
                        _frame(),
                        *_pattern_replay_row(distance=2, length=1, num_overlays=1),
                        *_pattern_overlay_row(
                            frame_offset=-1,
                            target_reg=OVERLAY_BODY_FREQ_DELTA,
                            new_val=bin_delta,
                        ),
                    ]
                )
                out = expand_loops(df)
                freq_writes = out[(out["reg"] == 0) & (out["op"] == SET_OP)]
                vals = sorted(int(v) for v in freq_writes["val"].tolist())
                self.assertEqual(
                    vals,
                    sorted([src_val, src_val + bin_delta]),
                    f"src={src_val} delta={bin_delta} produced vals {vals}",
                )

    def test_encoder_quantizes_arbitrary_delta_to_bin(self):
        """The encoder applies `_bin_body_freq_delta` to the raw delta
        before emitting. The decoder receives the already-binned
        delta. Verifies the encode-side rounding the decoder relies
        on: any input delta produces an emit-able new_val that is a
        multiple of OVERLAY_BODY_FREQ_DELTA_BIN.
        """
        bin_w = OVERLAY_BODY_FREQ_DELTA_BIN
        for raw in (5, 15, 17, 25, 100, -7, -8, -25):
            new_val = _bin_body_freq_delta(raw)
            self.assertEqual(new_val % bin_w, 0, raw)


class TestExpandLoops(unittest.TestCase):
    def test_no_loops_passthrough(self):
        df = pd.DataFrame([_frame(), _row(4, 8, op=SET_OP)])
        out = expand_loops(df)
        self.assertEqual(len(out), len(df))

    def test_back_ref_copies_frames(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                *_back_ref_row(distance=2, length=2),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 8)
        self.assertEqual(int(out.iloc[5]["val"]), 8)
        self.assertEqual(int(out.iloc[7]["val"]), 10)

    def test_do_loop_unrolls(self):
        df = pd.DataFrame(
            [
                _do_loop_begin_row(3),
                _frame(),
                _row(4, 8, op=SET_OP),
                _do_loop_end_row(),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 6)

    def test_do_loop_nested(self):
        df = pd.DataFrame(
            [
                _do_loop_begin_row(2),
                _do_loop_begin_row(3),
                _frame(),
                _row(4, 8, op=SET_OP),
                _do_loop_end_row(),
                _do_loop_end_row(),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 12)

    def test_back_ref_overlaps_present_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                *_back_ref_row(distance=2, length=4),
            ]
        )
        with self.assertRaises(AssertionError):
            expand_loops(df)

    def test_pattern_replay_zero_overlays_copies_body(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                *_pattern_replay_row(distance=2, length=2, num_overlays=0),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 8)
        self.assertEqual(int(out.iloc[5]["val"]), 8)
        self.assertEqual(int(out.iloc[7]["val"]), 10)

    def test_pattern_replay_with_per_frame_overlay(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                *_pattern_replay_row(distance=2, length=1, num_overlays=1),
                *_pattern_overlay_row(frame_offset=0, target_reg=4, new_val=99),
            ]
        )
        out = expand_loops(df)
        applied = out[(out["reg"] == 4) & (out["val"] == 99)]
        self.assertEqual(len(applied), 1)

    def test_orphan_pattern_overlay_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                *_back_ref_row(distance=1, length=1),
                *_pattern_overlay_row(frame_offset=0, target_reg=4, new_val=99),
            ]
        )
        with self.assertRaises(AssertionError):
            expand_loops(df)

    def test_pattern_replay_target_before_start_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                *_pattern_replay_row(distance=5, length=1, num_overlays=0),
            ]
        )
        with self.assertRaises(AssertionError):
            expand_loops(df)

    def test_pattern_replay_body_wide_freq_delta(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(0, 100, op=SET_OP),
                _frame(),
                *_pattern_replay_row(distance=2, length=1, num_overlays=1),
                *_pattern_overlay_row(
                    frame_offset=-1, target_reg=OVERLAY_BODY_FREQ_DELTA, new_val=5
                ),
            ]
        )
        out = expand_loops(df)
        freq_writes = out[(out["reg"] == 0) & (out["op"] == SET_OP)]
        vals = sorted(int(v) for v in freq_writes["val"].tolist())
        self.assertIn(100, vals)
        self.assertIn(105, vals)

    def test_pattern_replay_overlap_present_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                *_pattern_replay_row(distance=2, length=4),
            ]
        )
        with self.assertRaises(AssertionError):
            expand_loops(df)

    def test_orphan_back_ref_len_is_skipped(self):
        from preframr_tokens.stfconstants import BACK_REF_SUBREG_LEN

        df = pd.DataFrame(
            [
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_LEN,
                    "val": 4,
                    "diff": 32,
                    "op": BACK_REF_OP,
                    "description": 0,
                },
                _frame(),
                _row(4, 8, op=SET_OP),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out.iloc[1]["val"]), 8)

    def test_orphan_back_ref_dist_without_len_is_skipped(self):
        from preframr_tokens.stfconstants import BACK_REF_SUBREG_DIST_HI

        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_DIST_HI,
                    "val": 0,
                    "diff": 32,
                    "op": BACK_REF_OP,
                    "description": 0,
                },
                _row(5, 10, op=SET_OP),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 3)
        self.assertEqual(int(out.iloc[2]["val"]), 10)

    def test_orphan_pattern_replay_continuation_is_skipped(self):
        from preframr_tokens.stfconstants import PATTERN_REPLAY_SUBREG_LEN

        df = pd.DataFrame(
            [
                {
                    "reg": LOOP_OP_REG,
                    "subreg": PATTERN_REPLAY_SUBREG_LEN,
                    "val": 2,
                    "diff": 32,
                    "op": PATTERN_REPLAY_OP,
                    "description": 0,
                },
                _frame(),
                _row(4, 8, op=SET_OP),
            ]
        )
        out = expand_loops(df)
        self.assertEqual(len(out), 2)

    def test_orphan_counter_stamps_attrs_and_logs(self):
        """Each orphan branch increments a labelled counter; the
        per-call tally lands in ``df.attrs["_orphans"]`` and is logged
        at WARNING. Lets incident response distinguish a benign
        tokenizer-slice cut (1-2 orphans, expected) from a real
        corruption (many orphans, or orphans on a full-song parse
        """
        from preframr_tokens.stfconstants import (
            BACK_REF_SUBREG_DIST_HI,
            BACK_REF_SUBREG_LEN,
            PATTERN_REPLAY_SUBREG_LEN,
        )

        df = pd.DataFrame(
            [
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_LEN,
                    "val": 2,
                    "diff": 32,
                    "op": BACK_REF_OP,
                    "description": 0,
                },
                _frame(),
                _row(4, 8, op=SET_OP),
                {
                    "reg": LOOP_OP_REG,
                    "subreg": BACK_REF_SUBREG_DIST_HI,
                    "val": 0,
                    "diff": 32,
                    "op": BACK_REF_OP,
                    "description": 0,
                },
                _row(5, 10, op=SET_OP),
                {
                    "reg": LOOP_OP_REG,
                    "subreg": PATTERN_REPLAY_SUBREG_LEN,
                    "val": 3,
                    "diff": 32,
                    "op": PATTERN_REPLAY_OP,
                    "description": 0,
                },
            ]
        )
        with self.assertLogs("preframr_tokens.macros.loops", level="WARNING") as cm:
            out = expand_loops(df)
        orphans = out.attrs.get("_orphans", {})
        self.assertEqual(orphans.get("br_continuation_without_dist_hi"), 1)
        self.assertEqual(orphans.get("br_dist_hi_no_lo_partner"), 1)
        self.assertEqual(orphans.get("pr_continuation_without_dist_hi"), 1)
        self.assertTrue(
            any("expand_loops orphans" in r for r in cm.output),
            f"expected expand_loops WARNING; got {cm.output}",
        )

    def test_no_orphans_means_no_attrs_stamp(self):
        """Clean inputs leave ``df.attrs["_orphans"]`` unset so callers
        can use ``"_orphans" in attrs`` as a structural-cleanliness
        signal without a False-y dict tripping the check.
        """
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
            ]
        )
        out = expand_loops(df)
        self.assertNotIn("_orphans", out.attrs)


class TestLoopPass(unittest.TestCase):
    def test_lz77_non_adjacent_match(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(6, 100, op=SET_OP),
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
            ]
        )
        result = LoopPass().apply(df)
        backrefs = result[result["op"] == BACK_REF_OP]
        self.assertEqual(len(backrefs), 3)

    def test_do_loop_preferred_for_long_consecutive_runs(self):
        df = pd.DataFrame([_frame(), _row(4, 8, op=SET_OP)] * 4)
        result = LoopPass().apply(df)
        do_begins = result[(result["op"] == DO_LOOP_OP) & (result["subreg"] == 0)]
        self.assertEqual(len(do_begins), 1)
        self.assertEqual(int(do_begins.iloc[0]["val"]), 4)

    def test_no_compression_for_unique_frames(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(6, 100, op=SET_OP),
            ]
        )
        result = LoopPass().apply(df)
        self.assertEqual(int((result["op"] == BACK_REF_OP).sum()), 0)
        self.assertEqual(int((result["op"] == DO_LOOP_OP).sum()), 0)

    def test_disabled_when_flag_off(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(6, 100, op=SET_OP),
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
            ]
        )
        result = LoopPass().apply(
            df, args=FakeArgs(loop_pass=False, fuzzy_loop_pass=False)
        )
        self.assertEqual(int((result["op"] == BACK_REF_OP).sum()), 0)
        self.assertEqual(int((result["op"] == DO_LOOP_OP).sum()), 0)
        self.assertEqual(len(result), len(df))

    def test_round_trip_lz77(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
            ]
        )
        encoded = LoopPass().apply(df.copy())
        decoded = expand_loops(encoded)
        cols = ["reg", "val", "op", "subreg"]
        self.assertEqual(len(df), len(decoded))
        for i in range(len(df)):
            self.assertEqual(
                tuple(int(df.iloc[i][c]) for c in cols),
                tuple(int(decoded.iloc[i][c]) for c in cols),
            )


class TestValidateBackRefs(unittest.TestCase):
    def test_valid_passes(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                _frame(),
                _row(5, 10, op=SET_OP),
                *_back_ref_row(distance=2, length=2),
            ]
        )
        self.assertTrue(validate_back_refs(df, prompt_frame_count=0))

    def test_escapee_in_zero_prompt_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                *_back_ref_row(distance=5, length=1),
            ]
        )
        with self.assertRaises(AssertionError):
            validate_back_refs(df, prompt_frame_count=0)

    def test_escapee_resolved_by_prompt(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(4, 8, op=SET_OP),
                *_back_ref_row(distance=5, length=1),
            ]
        )
        self.assertTrue(validate_back_refs(df, prompt_frame_count=10))


def _pr_row(num_overlays, distance=1, length=1):
    """Quad-row PATTERN_REPLAY head. Caller splats with ``*_pr_row(...)``."""
    from preframr_tokens.stfconstants import (
        PATTERN_REPLAY_SUBREG_DIST_HI,
        PATTERN_REPLAY_SUBREG_DIST_LO,
        PATTERN_REPLAY_SUBREG_LEN,
        PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
    )

    return [
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_DIST_HI,
            "val": (int(distance) >> 8) & 0xFF,
            "diff": 0,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_DIST_LO,
            "val": int(distance) & 0xFF,
            "diff": 0,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_LEN,
            "val": int(length),
            "diff": 0,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_REPLAY_SUBREG_OVERLAY_COUNT,
            "val": int(num_overlays),
            "diff": 0,
            "op": PATTERN_REPLAY_OP,
            "description": 0,
        },
    ]


def _ov_row(frame_offset=0, target_reg=4, val=0):
    """Triple-row PATTERN_OVERLAY for validate_pattern_overlays tests.
    Caller splats with ``*_ov_row(...)``.
    """
    from preframr_tokens.stfconstants import (
        PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
        PATTERN_OVERLAY_SUBREG_TARGET_REG,
        PATTERN_OVERLAY_SUBREG_NEW_VAL,
    )

    return [
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_FRAME_OFFSET,
            "val": int(frame_offset),
            "diff": 0,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_TARGET_REG,
            "val": int(target_reg),
            "diff": 0,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
        {
            "reg": LOOP_OP_REG,
            "subreg": PATTERN_OVERLAY_SUBREG_NEW_VAL,
            "val": int(val),
            "diff": 0,
            "op": PATTERN_OVERLAY_OP,
            "description": 0,
        },
    ]


class TestValidatePatternOverlays(unittest.TestCase):
    def test_no_pr_or_ov_passes(self):
        df = pd.DataFrame([_frame(), _row(4, 8, op=SET_OP)])
        self.assertTrue(validate_pattern_overlays(df))

    def test_complete_block_passes(self):
        df = pd.DataFrame(
            [
                _frame(),
                *_pr_row(num_overlays=2),
                *_ov_row(frame_offset=0),
                *_ov_row(frame_offset=1),
                _frame(),
            ]
        )
        self.assertTrue(validate_pattern_overlays(df))

    def test_orphan_overlay_raises(self):
        df = pd.DataFrame([_frame(), *_ov_row(), _frame()])
        with self.assertRaisesRegex(AssertionError, "orphan PATTERN_OVERLAY"):
            validate_pattern_overlays(df)

    def test_pr_short_raises(self):
        df = pd.DataFrame(
            [
                _frame(),
                *_pr_row(num_overlays=2),
                *_ov_row(frame_offset=0),
                _frame(),
            ]
        )
        with self.assertRaisesRegex(AssertionError, "interrupted"):
            validate_pattern_overlays(df)

    def test_pr_unfinished_at_end_raises(self):
        df = pd.DataFrame([*_pr_row(num_overlays=1)])
        with self.assertRaisesRegex(AssertionError, "unfinished"):
            validate_pattern_overlays(df)

    def test_pr_then_pr_without_overlays_raises(self):
        df = pd.DataFrame(
            [*_pr_row(num_overlays=1), *_pr_row(num_overlays=1), *_ov_row()]
        )
        with self.assertRaisesRegex(AssertionError, "expected"):
            validate_pattern_overlays(df)


class TestIterSelfContainedRowBlocks(unittest.TestCase):
    """Both training and inference funnel through this iterator. Each
    yielded block is self-contained: tokenizing and decoding it does not
    require any frames outside the block.
    """

    def _gate_on(self, voice, ctrl=0x41, ad=0xF0, sr=0x20):
        base = voice * VOICE_REG_SIZE
        return [
            _row(base + 4, ctrl, op=SET_OP),
            _row(base + 5, ad, op=SET_OP),
            _row(base + 6, sr, op=SET_OP),
        ]

    def _gate_off(self, voice, ctrl=0x40):
        base = voice * VOICE_REG_SIZE
        return [_row(base + 4, ctrl, op=SET_OP)]

    def _multi_replay_song(self):
        rows = [_frame()]
        for _ in range(6):
            rows += self._gate_on(0)
            rows += [_frame()]
            rows += self._gate_off(0)
            rows += [_frame()]
        return pd.DataFrame(rows)

    def test_blocks_cover_all_frames(self):
        df = self._multi_replay_song()
        encoded = run_passes(
            df.copy(),
            args=FakeArgs(
                loop_pass=False,
                fuzzy_loop_pass=False,
            ),
        )
        n_frames = int(encoded["reg"].isin([FRAME_REG, -127]).sum())
        total_block_frames = 0
        for block in iter_self_contained_row_blocks(encoded, frames_per_block=4):
            total_block_frames += int(block["reg"].isin([FRAME_REG, -127]).sum())
        self.assertEqual(total_block_frames, n_frames)

    def test_each_block_validates(self):
        df = self._multi_replay_song()
        encoded = run_passes(
            df.copy(),
            args=FakeArgs(
                loop_pass=False,
                fuzzy_loop_pass=False,
            ),
        )
        for block in iter_self_contained_row_blocks(encoded, frames_per_block=3):
            self.assertTrue(validate_back_refs(block))


if __name__ == "__main__":
    unittest.main()
