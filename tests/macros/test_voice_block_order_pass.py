import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.passes import VoiceBlockOrderPass
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE, FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    MODEL_PDTYPE,
    SET_OP,
    VOICE_REG_SIZE,
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


def _delay(val=2, diff=19000):
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


def _apply_pass(df, **flags):
    flags.setdefault("voice_canonical_block_order", True)
    return VoiceBlockOrderPass().apply(df, args=FakeArgs(**flags))


class TestVoiceBlockOrderPassBasic(unittest.TestCase):
    def test_off_by_default(self):
        df = pd.DataFrame(
            [_frame(), _row(0, 100), _row(7, 200), _row(14, 50)],
            dtype=MODEL_PDTYPE,
        )
        out = VoiceBlockOrderPass().apply(df.copy(), args=FakeArgs())
        pd.testing.assert_frame_equal(out, df)

    def test_single_voice_frame_unchanged(self):
        df = pd.DataFrame(
            [_frame(), _row(0, 100), _row(4, 0x11)],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df)
        self.assertEqual(len(out), len(df))

    def test_row_count_preserved(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1000),
                _row(CTRL_REGS_BY_VOICE[0], 0x11),
                _row(FREQ_REGS_BY_VOICE[1], 5000),
                _row(CTRL_REGS_BY_VOICE[1], 0x11),
                _row(FREQ_REGS_BY_VOICE[2], 3000),
                _row(CTRL_REGS_BY_VOICE[2], 0x11),
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1010),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df)
        self.assertEqual(len(out), len(df))

    def test_audio_invariant_three_voice_pitch_sort(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1000),
                _row(CTRL_REGS_BY_VOICE[0], 0x11),
                _row(FREQ_REGS_BY_VOICE[1], 5000),
                _row(CTRL_REGS_BY_VOICE[1], 0x11),
                _row(FREQ_REGS_BY_VOICE[2], 3000),
                _row(CTRL_REGS_BY_VOICE[2], 0x11),
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1010),
                _row(FREQ_REGS_BY_VOICE[1], 5010),
                _row(FREQ_REGS_BY_VOICE[2], 3010),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df)
        base_writes = _sid_writes(expand_ops(df, strict=False))
        out_writes = _sid_writes(expand_ops(out, strict=False))
        pd.testing.assert_frame_equal(base_writes, out_writes)

    def test_audio_invariant_delay_frame(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 4000),
                _row(CTRL_REGS_BY_VOICE[0], 0x11),
                _row(FREQ_REGS_BY_VOICE[1], 8000),
                _row(CTRL_REGS_BY_VOICE[1], 0x11),
                _delay(val=3),
                _row(CTRL_REGS_BY_VOICE[0], 0x10),
                _row(CTRL_REGS_BY_VOICE[1], 0x10),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df)
        base_writes = _sid_writes(expand_ops(df, strict=False))
        out_writes = _sid_writes(expand_ops(out, strict=False))
        pd.testing.assert_frame_equal(base_writes, out_writes)

    def test_filter_and_modevol_unaffected(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1000),
                _row(CTRL_REGS_BY_VOICE[0], 0x11),
                _row(FREQ_REGS_BY_VOICE[2], 9000),
                _row(CTRL_REGS_BY_VOICE[2], 0x11),
                _row(23, 0x10),
                _row(24, 0x0F),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df)
        base_writes = _sid_writes(expand_ops(df, strict=False))
        out_writes = _sid_writes(expand_ops(out, strict=False))
        pd.testing.assert_frame_equal(base_writes, out_writes)

    def test_higher_pitch_voice_lands_first_post_reorder(self):
        df = pd.DataFrame(
            [
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1000),
                _row(CTRL_REGS_BY_VOICE[0], 0x11),
                _row(FREQ_REGS_BY_VOICE[2], 9000),
                _row(CTRL_REGS_BY_VOICE[2], 0x11),
                _frame(),
                _row(FREQ_REGS_BY_VOICE[0], 1010),
                _row(FREQ_REGS_BY_VOICE[2], 9010),
            ],
            dtype=MODEL_PDTYPE,
        )
        out = _apply_pass(df).reset_index(drop=True)
        frame_idxs = out.index[out["reg"] == FRAME_REG].tolist()
        self.assertEqual(len(frame_idxs), 2)
        f1_start = frame_idxs[1]
        first_voice_row_reg = int(out.iloc[f1_start + 1]["reg"])
        self.assertEqual(first_voice_row_reg // VOICE_REG_SIZE, 2)


if __name__ == "__main__":
    unittest.main()
