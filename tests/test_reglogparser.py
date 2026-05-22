import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.reglog_helpers import (
    ad_match,
    adsr_match,
    dump_palettes_attrs,
    freq_match,
    load_palettes_attrs,
    pcm_match,
    sr_match,
)
from preframr_tokens.macros.per_reg_burst import PerRegBurstPass
from preframr_tokens.reglogparser import (
    RegLogParser,
    last_reg_val_frame,
    prepare_df_for_audio,
    remove_voice_reg,
    reset_diffs,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
    FILTER_REG,
    FRAME_REG,
    MAX_REG,
    MIN_DIFF,
    MODEL_PDTYPE,
    VOICE_REG_SIZE,
    VOICES,
    VOICE_REG,
    DIFF_OP,
    FLIP_OP,
    SET_OP,
)


class FakeArgs:
    def __init__(
        self,
        seq_len=128,
        cents=10,
        min_irq=0,
        max_irq=100000,
        min_song_tokens=4,
        exclude_list=None,
    ):
        self.reglog = None
        self.reglogs = ""
        self.seq_len = seq_len
        self.max_files = 1
        self.cents = cents
        self.min_irq = min_irq
        self.max_irq = max_irq
        self.min_song_tokens = min_song_tokens
        self.exclude_list = exclude_list


class TestRegLogParser(unittest.TestCase):
    def test_highbitmask(self):
        loader = RegLogParser(FakeArgs())
        self.assertEqual(loader._highbitmask(7), 128)
        self.assertEqual(loader._highbitmask(4), 240)
        self.assertEqual(loader._highbitmask(1), 254)

    def test_simplfy_ctrl(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 4, "val": 33 + 2**2},
                {"reg": 4, "val": 17 + 2**2},
                {"reg": 4, "val": 33 + 2**1},
                {"reg": 4, "val": 0 + 2**1},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._simplify_ctrl(test_df).astype(dtype=MODEL_PDTYPE)
        expected_df = pd.DataFrame(
            [
                {"reg": 4, "val": 33},
                {"reg": 4, "val": 17 + 2**2},
                {"reg": 4, "val": 33 + 2**1},
                {"reg": 4, "val": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertTrue(expected_df.equals(result_df))

    def test_maskregbits(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 255},
                {"reg": 1, "val": 128},
            ]
        )
        loader._maskregbits(test_df, 1, 1)
        mask_df = pd.DataFrame(
            [
                {"reg": 1, "val": 254},
                {"reg": 1, "val": 128},
            ]
        )
        self.assertTrue(mask_df.equals(test_df))

    def test_squeeze_changes(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 1, "val": 1},
                {"clock": 2, "irq": 2, "reg": 1, "val": 1},
                {"clock": 3, "irq": 3, "reg": 2, "val": 1},
                {"clock": 4, "irq": 4, "reg": 2, "val": 2},
            ]
        )
        squeeze_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 1, "val": 1},
                {"clock": 3, "irq": 3, "reg": 2, "val": 1},
                {"clock": 4, "irq": 4, "reg": 2, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._squeeze_changes(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(squeeze_df.equals(result), result)

    def test_combine_reg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1},
                {"reg": 1, "val": 1},
                {"reg": 2, "val": 1},
                {"reg": 2, "val": 2},
                {"reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        test_df["diff"] = 8
        test_df["clock"] = test_df["diff"].cumsum()
        combine_df = pd.DataFrame(
            [
                {"reg": 1, "val": 1, "diff": 8, "clock": 8},
                {"reg": 1, "val": 257, "diff": 8, "clock": 24},
                {"reg": 1, "val": 514, "diff": 8, "clock": 40},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 16, bits=0).astype(MODEL_PDTYPE)
        self.assertTrue(combine_df.equals(result_df), result_df)
        combine_df = pd.DataFrame(
            [
                {"reg": 1, "val": 0, "diff": 8, "clock": 8},
                {"reg": 1, "val": 256, "diff": 8, "clock": 24},
                {"reg": 1, "val": 514, "diff": 8, "clock": 40},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 16, bits=1).astype(MODEL_PDTYPE)
        self.assertTrue(combine_df.equals(result_df), result_df)
        test_df = pd.DataFrame(
            [
                {"reg": 1, "val": 3},
                {"reg": 2, "val": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        test_df["diff"] = 8
        test_df["clock"] = test_df["diff"].cumsum()
        combine_df = pd.DataFrame(
            [
                {"reg": 1, "val": 11, "diff": 8, "clock": 16},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._combine_reg(test_df, 1, 32, bits=0, lobits=3).astype(
            MODEL_PDTYPE
        )
        self.assertTrue(combine_df.equals(result_df), result_df)

    def test_rotate_voice_augment(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 12, "reg": 11, "val": 2},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        rotate_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1},
                {"clock": 8, "reg": 4, "val": 1},
                {"clock": 12, "reg": 11, "val": 2},
                {"clock": 16, "reg": 23, "val": 1 + 4},
                {"clock": 32, "reg": 7, "val": 2},
                {"clock": 64, "reg": 14, "val": 3},
                {"clock": 0, "reg": 7, "val": 1},
                {"clock": 8, "reg": 11, "val": 1},
                {"clock": 12, "reg": 18, "val": 2},
                {"clock": 16, "reg": 23, "val": 2 + 1},
                {"clock": 32, "reg": 14, "val": 2},
                {"clock": 64, "reg": 0, "val": 3},
                {"clock": 0, "reg": 14, "val": 1},
                {"clock": 8, "reg": 18, "val": 1},
                {"clock": 12, "reg": 4, "val": 2},
                {"clock": 16, "reg": 23, "val": 4 + 2},
                {"clock": 32, "reg": 0, "val": 2},
                {"clock": 64, "reg": 7, "val": 3},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = pd.concat(
            loader._rotate_voice_augment(test_df, VOICES)
        ).reset_index(drop=True)
        self.assertTrue(rotate_df.equals(result_df), result_df)

    def test_add_frame_reg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1, "irq": 0},
                {"clock": 256, "reg": 4, "val": 1, "irq": 0},
                {"clock": 32768, "reg": 11, "val": 2, "irq": 19000},
                {"clock": 32768 + 8, "reg": 23, "val": 1 + 4, "irq": 19000},
                {"clock": 32768 + 16, "reg": 7, "val": 2, "irq": 19000},
                {"clock": 32768 + 32, "reg": 14, "val": 3, "irq": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        frame_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32},
                {"reg": 4, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 11, "val": 2, "diff": 32},
                {"reg": 23, "val": 5, "diff": 32},
                {"reg": 7, "val": 2, "diff": 32},
                {"reg": 14, "val": 3, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        irq, result_df = loader._add_frame_reg(test_df, 512, min_irq_prop=0.5)
        self.assertEqual(irq, 19000)
        self.assertTrue(frame_df.equals(result_df))

    def test_last_reg_val_frame(self):
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": 7, "val": 2, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 3, "diff": 32},
                {"reg": 7, "val": 4, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        last_df = pd.DataFrame(
            [
                {"f": 1, "v": 1, "val": 2, "pval": 0},
                {"f": 2, "v": 1, "val": 4, "pval": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = list(last_reg_val_frame(test_df, [0]))[0]
        self.assertTrue(last_df.equals(result_df))

    def test_add_change_regs_flip_only(self):
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 0, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 65, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        change_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 1, "diff": 32, "op": FLIP_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 0, "diff": 32, "op": FLIP_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 65, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = (
            PerRegBurstPass(opcodes=[DIFF_OP, FLIP_OP])
            .apply(test_df, args=FakeArgs(cents=50))
            .astype(MODEL_PDTYPE)
        )
        self.assertTrue(change_df.equals(result_df))

    def test_norm_pr_order(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 7, "val": 255, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 0},
                {"reg": 14, "val": 3, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        order_df = pd.DataFrame(
            [
                {"reg": 0, "val": 2, "diff": 32, "op": 0},
                {"reg": 7, "val": 255, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 0},
                {"reg": 14, "val": 3, "diff": 32, "op": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._norm_pr_order(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(order_df.equals(result_df))

    def test_add_voice_reg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 1},
                {"reg": 14, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": 7, "val": 2, "diff": 32, "op": 1},
                {"reg": 14, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        voice_df = pd.DataFrame(
            [
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 1},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 57, "diff": 19000, "op": 0},
                {"reg": 0, "val": 1, "diff": 32, "op": 0},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 2, "diff": 32, "op": 1},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": 0},
                {"reg": 0, "val": 3, "diff": 32, "op": 1},
                {"reg": FRAME_REG, "val": 1, "diff": 19000, "op": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._add_voice_reg(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(voice_df.equals(result_df))

    def test_expand_ops(self):
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 0,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
                {
                    "reg": DELAY_REG,
                    "subreg": -1,
                    "val": 3,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 0,
                    "diff": 32,
                    "op": FLIP_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 3, "diff": 32, "description": 0},
                {
                    "reg": FRAME_REG,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 1,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": DELAY_REG,
                    "subreg": -1,
                    "val": 2,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": 9,
                    "subreg": -1,
                    "val": 2,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        expand_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": 7, "val": 1, "diff": 32, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "description": 0},
                {"reg": 7, "val": 2, "diff": 32, "description": 0},
                {"reg": 9, "val": 2, "diff": 32, "description": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = expand_ops(test_df, strict=True).astype(MODEL_PDTYPE)
        self.assertTrue(expand_df.equals(result_df))

    def test_consolidate_frames(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        consolidate_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 3, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(consolidate_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        consolidate_df = pd.DataFrame(
            [
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df).astype(MODEL_PDTYPE)
        self.assertTrue(consolidate_df.equals(result_df))

    def test_consolidate_frames_adjacent_delays(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 19000},
                {"reg": DELAY_REG, "val": 3, "diff": 19000},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df = loader._consolidate_frames(test_df)
        self.assertEqual(len(result_df[result_df["reg"] == DELAY_REG]), 1)

    def test_remove_voice_reg(self):
        test_df = pd.DataFrame(
            [{"reg": 0, "val": 1, "diff": 32, "op": SET_OP}], dtype=MODEL_PDTYPE
        )
        result_df, result_widths = remove_voice_reg(test_df, {})
        self.assertTrue(test_df.equals(result_df))

        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 57, "diff": 19000, "op": SET_OP},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": SET_OP},
                {"reg": 0, "val": 100, "diff": 32, "op": SET_OP},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": SET_OP},
                {"reg": 0, "val": 200, "diff": 32, "op": SET_OP},
                {"reg": VOICE_REG, "val": 0, "diff": 32, "op": SET_OP},
                {"reg": 0, "val": 150, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        reg_widths = {0: 8}
        result_df, result_widths = remove_voice_reg(test_df, reg_widths)
        self.assertFalse((result_df["reg"] == VOICE_REG).any())
        for v in range(VOICES):
            self.assertIn(v * VOICE_REG_SIZE, result_widths)

    def test_reset_diffs(self):
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
                {"reg": DELAY_REG, "val": 2, "diff": 0},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = reset_diffs(test_df, None, 1)
        self.assertIn("delay", result.columns)
        delay_diff = result.loc[result["reg"] == DELAY_REG, "diff"].iloc[0]
        self.assertEqual(delay_diff, 2 * 19000)

        result2 = reset_diffs(test_df, 19000, 1)
        self.assertTrue(result.equals(result2))

    def test_expand_ops_diff_op(self):
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 5,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": -1,
                    "val": 3,
                    "diff": 32,
                    "op": DIFF_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        result = expand_ops(test_df, strict=True)
        reg7_vals = result[result["reg"] == 7]["val"].tolist()
        self.assertEqual(reg7_vals, [5, 8])

    def test_expand_ops_subreg(self):
        test_df = pd.DataFrame(
            [
                {
                    "reg": FRAME_REG,
                    "subreg": -1,
                    "val": 0,
                    "diff": 19000,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": 0,
                    "val": 5,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
                {
                    "reg": 7,
                    "subreg": 1,
                    "val": 3,
                    "diff": 32,
                    "op": SET_OP,
                    "description": 0,
                },
            ],
            dtype=MODEL_PDTYPE,
        )
        result = expand_ops(test_df, strict=True)
        reg7_vals = result[result["reg"] == 7]["val"].tolist()
        self.assertEqual(reg7_vals, [48 + 5])

    def test_prepare_df_for_audio(self):
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP, "subreg": -1},
                {"reg": 7, "val": 1, "diff": 32, "op": SET_OP, "subreg": -1},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP, "subreg": -1},
                {"reg": 7, "val": 2, "diff": 32, "op": SET_OP, "subreg": -1},
            ],
            dtype=MODEL_PDTYPE,
        )
        result_df, _ = prepare_df_for_audio(test_df, {}, irq=19000, sidq=1)
        self.assertIn("delay", result_df.columns)

        result_df2, _ = prepare_df_for_audio(
            test_df, {}, irq=19000, sidq=1, prompt_len=2
        )
        self.assertTrue((result_df2["description"] > 0).any())

    def test_matcher_methods(self):
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1},
                {"reg": 2, "val": 1},
                {"reg": 4, "val": 1},
                {"reg": 5, "val": 1},
                {"reg": 6, "val": 1},
                {"reg": FC_LO_REG, "val": 1},
                {"reg": FILTER_REG, "val": 1},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertTrue(freq_match(test_df).any())
        self.assertTrue(pcm_match(test_df).any())
        self.assertTrue(adsr_match(test_df).any())
        self.assertTrue(ad_match(test_df).any())
        self.assertTrue(sr_match(test_df).any())

    def test_read_df(self):
        loader = RegLogParser(FakeArgs())

        with self.assertRaises(ValueError):
            loader._read_df("/nonexistent/path.parquet")

        test_df = pd.DataFrame(
            [
                {"clock": 1, "irq": 100, "reg": 0, "val": 1, "chipno": 0},
                {"clock": 2, "irq": 200, "reg": 25, "val": 2, "chipno": 0},
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname = f.name
        try:
            test_df.to_parquet(fname)
            result = loader._read_df(fname)
            self.assertEqual(list(result.columns), ["clock", "irq", "reg", "val"])
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(fname)

        test_df2 = pd.DataFrame(
            [
                {"clock": -1, "irq": 100, "reg": 0, "val": 1, "chipno": 0},
                {"clock": 2, "irq": 200, "reg": 1, "val": 2, "chipno": 1},
            ]
        )
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname2 = f.name
        try:
            test_df2.to_parquet(fname2)
            result2 = loader._read_df(fname2)
            self.assertTrue((result2["clock"] < 0).all())
        finally:
            os.unlink(fname2)

    def test_rotate_filter(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame([{"reg": FILTER_REG, "val": 1}], dtype=MODEL_PDTYPE)
        result0 = loader._rotate_filter(test_df.copy(), 0)
        self.assertEqual(result0.loc[result0["reg"] == FILTER_REG, "val"].iloc[0], 1)

        result1 = loader._rotate_filter(test_df.copy(), 1)
        self.assertEqual(result1.loc[result1["reg"] == FILTER_REG, "val"].iloc[0], 2)

    def test_add_frame_reg_no_irqdiff(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1, "irq": 0},
                {"clock": 1, "reg": 4, "val": 1, "irq": 0},
            ],
            dtype=MODEL_PDTYPE,
        )
        irq, _result_df = loader._add_frame_reg(test_df, 512)
        self.assertEqual(irq, 0)

    def test_cap_delay(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": DELAY_REG, "val": 1000},
                {"reg": DELAY_REG, "val": 30},
                {"reg": DELAY_REG, "val": 5},
                {"reg": 7, "val": 1000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._cap_delay(test_df.copy())
        self.assertEqual(result.iloc[0]["val"], 256)
        self.assertEqual(result.iloc[1]["val"], 32)
        self.assertEqual(result.iloc[2]["val"], 5)
        self.assertEqual(result.iloc[3]["val"], 1000)

    def test_split_reg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 512, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._split_reg(test_df, 0)
        self.assertIn(1, result["reg"].values)
        hi_rows = result[result["reg"] == 1]
        self.assertEqual(hi_rows["val"].iloc[0], 2)
        lo_rows = result[result["reg"] == 0]
        self.assertEqual(lo_rows["val"].iloc[0], 0)

    def test_reduce_val_res(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [{"reg": 7, "val": 0b11111111}, {"reg": 8, "val": 0b11111111}],
            dtype=MODEL_PDTYPE,
        )
        result = loader._reduce_val_res(test_df.copy(), reg=7, bits=2)
        self.assertEqual(result.loc[result["reg"] == 7, "val"].iloc[0], 252)
        self.assertEqual(result.loc[result["reg"] == 8, "val"].iloc[0], 255)

    def test_quantize_freq_to_cents(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 0},
                {"reg": 7, "val": 0},
                {"reg": 8, "val": 100},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._quantize_freq_to_cents(test_df.copy())
        self.assertEqual(result.loc[result["reg"] == 8, "val"].iloc[0], 100)
        self.assertIsInstance(
            result.loc[result["reg"] == 0, "val"].iloc[0], (int, np.integer)
        )

    def test_add_voice_reg_empty_regs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_voice_reg(test_df)
        self.assertTrue(test_df.equals(result))

    def test_add_voice_reg_zero_false(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 0, "val": 1, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 2, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 3, "diff": 32, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_voice_reg(test_df, zero_voice_reg=False)
        self.assertIn(VOICE_REG, result["reg"].values)

    def test_simplify_pcm(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 2, "val": 128},
                {"reg": 4, "val": 0b01000001},
                {"reg": 4, "val": 0b00000001},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._simplify_pcm(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("reg", result.columns)
        self.assertGreater(len(result), len(test_df))

    def test_filter_irq(self):
        loader = RegLogParser(FakeArgs(min_irq=1000, max_irq=50000))

        test_df = pd.DataFrame([{"reg": 0, "val": 1}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 100}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 99999}], dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter_irq(test_df, "test"))

        test_df = pd.DataFrame([{"reg": 0, "val": 1, "irq": 19000}], dtype=MODEL_PDTYPE)
        self.assertTrue(loader._filter_irq(test_df, "test"))

    def test_filter(self):
        loader = RegLogParser(FakeArgs(seq_len=2))

        test_df = pd.DataFrame(
            [{"reg": 7, "val": i, "diff": 32} for i in range(10)], dtype=MODEL_PDTYPE
        )
        self.assertFalse(loader._filter(test_df, "test"))

        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 7, "val": 1, "diff": 32},
            ],
            dtype=MODEL_PDTYPE,
        )
        self.assertFalse(loader._filter(test_df, "test"))

        rows = [{"reg": FRAME_REG, "val": 0, "diff": 19000}]
        for i in range(20):
            rows.append({"reg": 24, "val": i % 16, "diff": 32})
        for _ in range(10):
            rows.append({"reg": FRAME_REG, "val": 0, "diff": 19000})
        test_df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        self.assertFalse(loader._filter(test_df, "test"))

        rows = []
        for i in range(16):
            rows.append({"reg": FRAME_REG, "val": 0, "diff": 19000})
            rows.append({"reg": 24, "val": i, "diff": 32})
        test_df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        self.assertTrue(loader._filter(test_df, "test"))

        rows = []
        for _ in range(5):
            rows.append({"reg": FRAME_REG, "val": 0, "diff": 19000})
            rows.append({"reg": 7, "val": 1, "diff": 32})
        test_df = pd.DataFrame(rows, dtype=MODEL_PDTYPE)
        self.assertTrue(loader._filter(test_df, "test"))

    def test_combine_regs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"clock": 8, "irq": 0, "reg": 0, "val": 1},
                {"clock": 16, "irq": 0, "reg": 1, "val": 2},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._combine_regs(test_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("reg", result.columns)

    def test_squeeze_frame_regs(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": 0, "val": 1, "diff": 32},
                {"reg": 0, "val": 2, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._squeeze_frame_regs(test_df)
        reg0_rows = result[result["reg"] == 0]
        self.assertEqual(len(reg0_rows), 1)
        self.assertEqual(reg0_rows["val"].iloc[0], 2)

        test_df2 = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
                {"reg": FC_LO_REG, "val": 10, "diff": 32},
                {"reg": FC_LO_REG, "val": 20, "diff": 32},
                {"reg": FRAME_REG, "val": 0, "diff": 19000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result2 = loader._squeeze_frame_regs(test_df2, regs=[FC_LO_REG])
        fc_rows = result2[result2["reg"] == FC_LO_REG]
        self.assertEqual(len(fc_rows), 1)
        self.assertEqual(fc_rows["val"].iloc[0], 20)

    def test_add_subreg(self):
        loader = RegLogParser(FakeArgs())
        test_df = pd.DataFrame(
            [
                {"reg": 4, "val": 0b11110101},
                {"reg": 7, "val": 0b11110101},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_subreg(test_df)
        self.assertIn("subreg", result.columns)
        reg4_rows = result[result["reg"] == 4]
        self.assertEqual(len(reg4_rows), 2)
        self.assertIn(0, reg4_rows["subreg"].values)
        self.assertIn(1, reg4_rows["subreg"].values)
        reg7_rows = result[result["reg"] == 7]
        self.assertEqual(reg7_rows["subreg"].iloc[0], -1)

    def test_state_df(self):
        loader = RegLogParser(FakeArgs())
        tokens = pd.DataFrame(
            [
                {"n": 0, "reg": 7, "val": 1, "op": SET_OP, "subreg": -1},
                {"n": 1, "reg": FRAME_REG, "val": 0, "op": SET_OP, "subreg": -1},
                {"n": 2, "reg": -MAX_REG - 1, "val": 0, "op": SET_OP, "subreg": -1},
            ],
            dtype=MODEL_PDTYPE,
        )

        class FakeTokenizer:
            pass

        class FakeDataset:
            pass

        tokenizer = FakeTokenizer()
        tokenizer.tokens = tokens
        dataset = FakeDataset()
        dataset.tokenizer = tokenizer

        irq = 19000
        result = loader._state_df([0, 1, 2], dataset, irq)

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[result["reg"] == 7, "diff"].iloc[0], MIN_DIFF)
        self.assertEqual(result.loc[result["reg"] == FRAME_REG, "diff"].iloc[0], irq)
        self.assertEqual(result.loc[result["reg"] == -MAX_REG - 1, "diff"].iloc[0], 0)

    def test_remove_voice_reg_is_inverse_of_add_voice_reg(self):
        loader = RegLogParser(FakeArgs())
        orig_df = pd.DataFrame(
            [
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": 0, "val": 100, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 200, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 150, "diff": 32, "op": SET_OP},
                {"reg": FRAME_REG, "val": 0, "diff": 19000, "op": SET_OP},
                {"reg": 0, "val": 110, "diff": 32, "op": SET_OP},
                {"reg": 7, "val": 210, "diff": 32, "op": SET_OP},
                {"reg": 14, "val": 160, "diff": 32, "op": SET_OP},
            ],
            dtype=MODEL_PDTYPE,
        )

        voice_df = loader._add_voice_reg(orig_df)
        self.assertTrue((voice_df["reg"] == VOICE_REG).any())

        result_df, _ = remove_voice_reg(voice_df, {})

        self.assertFalse((result_df["reg"] == VOICE_REG).any())

        self.assertEqual(len(result_df), len(orig_df))

        orig_regs = orig_df[orig_df["reg"] != FRAME_REG].reset_index(drop=True)
        result_regs = result_df[result_df["reg"] != FRAME_REG].reset_index(drop=True)
        self.assertTrue(orig_regs.equals(result_regs))

        orig_frame = orig_df[orig_df["reg"] == FRAME_REG][["diff", "op"]].reset_index(
            drop=True
        )
        result_frame = result_df[result_df["reg"] == FRAME_REG][
            ["diff", "op"]
        ].reset_index(drop=True)
        self.assertTrue(orig_frame.equals(result_frame))


class TestExcludeList(unittest.TestCase):
    def test_empty_when_no_arg(self):
        loader = RegLogParser(FakeArgs())
        self.assertFalse(loader.exclude_set)
        self.assertFalse(loader._excluded("/foo/bar.dump.parquet"))

    def test_load_csv_and_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "ex.csv")
            with open(csv_path, "w") as f:
                f.write("path,reason\n")
                f.write("Hubbard/Commando.1.dump.parquet,digi_vol_density\n")
                f.write(
                    "/scratch/preframr/train/Galway/Foo.1.dump.parquet,ctrl_burst\n"
                )
            loader = RegLogParser(FakeArgs(exclude_list=csv_path))
            self.assertEqual(len(loader.exclude_set), 2)
            self.assertTrue(
                loader._excluded("/scratch/preframr/train/Galway/Foo.1.dump.parquet")
            )
            self.assertTrue(
                loader._excluded("/anywhere/train/Hubbard/Commando.1.dump.parquet")
            )
            self.assertFalse(
                loader._excluded("/scratch/preframr/train/Galway/Bar.1.dump.parquet")
            )

    def test_missing_file_returns_empty(self):
        loader = RegLogParser(FakeArgs(exclude_list="/no/such/file.csv"))
        self.assertEqual(loader.exclude_set, frozenset())


class TestPalettesSidecar(unittest.TestCase):
    """The sidecar persists engine-fingerprint / cluster only post-cleanup."""

    def test_round_trip_engine_fingerprint(self):
        attrs = {"engine_fingerprint": [float(i) / 10.0 for i in range(132)]}
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = os.path.join(tmpdir, "x.0.parquet")
            dump_palettes_attrs(attrs, pq)
            restored = load_palettes_attrs(pq)
        self.assertEqual(restored["engine_fingerprint"], attrs["engine_fingerprint"])

    def test_round_trip_engine_fp_cluster(self):
        attrs = {"engine_fp_cluster": 4}
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = os.path.join(tmpdir, "x.0.parquet")
            dump_palettes_attrs(attrs, pq)
            restored = load_palettes_attrs(pq)
        self.assertEqual(restored["engine_fp_cluster"], 4)
        self.assertIsInstance(restored["engine_fp_cluster"], int)

    def test_round_trip_engine_fp_cluster_zero(self):
        attrs = {"engine_fp_cluster": 0}
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = os.path.join(tmpdir, "x.0.parquet")
            dump_palettes_attrs(attrs, pq)
            restored = load_palettes_attrs(pq)
        self.assertEqual(restored["engine_fp_cluster"], 0)

    def test_load_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = os.path.join(tmpdir, "x.0.parquet")
            self.assertEqual(load_palettes_attrs(pq), {})

    def test_dump_empty_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pq = os.path.join(tmpdir, "x.0.parquet")
            dump_palettes_attrs({}, pq)
            self.assertFalse(os.path.exists(pq + ".palettes.json"))
            dump_palettes_attrs(None, pq)
            self.assertFalse(os.path.exists(pq + ".palettes.json"))
