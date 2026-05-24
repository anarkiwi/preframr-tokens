"""Tests for VoiceTrackPass + TrackRefDecoder (item 12)."""

import unittest

import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.voice_track_pass import VoiceTrackPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    SET_OP,
    TRACK_REF_OP,
    TRACK_REF_SUBREG_COUNT,
)


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


def _r(reg, val, op=SET_OP, subreg=-1):
    return {
        "reg": reg,
        "val": val,
        "op": op,
        "subreg": subreg,
        "diff": 32,
        "description": 0,
    }


def _frame():
    return _r(FRAME_REG, 0)


def _stream(lead_vals, ratio, detune=0, lead_reg=0, tracker_reg=7):
    """One FRAME per step; lead FREQ then tracker FREQ = round(lead*ratio)+detune."""
    rows = []
    for lv in lead_vals:
        rows.append(_frame())
        rows.append(_r(lead_reg, lv))
        rows.append(_r(tracker_reg, round(lv * ratio) + detune))
    return pd.DataFrame(rows)


def _apply(df, **flags):
    flags.setdefault("voice_track_pass", True)
    return VoiceTrackPass().apply(df.copy(), args=FakeArgs(**flags))


def _decoded_reg(df, reg):
    out = expand_ops(df, strict=False).reset_index(drop=True)
    return out[out["reg"] == reg]["val"].tolist()


class TestVoiceTrackPass(unittest.TestCase):
    def test_disabled_is_noop(self):
        df = _stream(list(range(1000, 1015)), 2.0)
        out = VoiceTrackPass().apply(df.copy(), args=FakeArgs(voice_track_pass=False))
        self.assertTrue(out.equals(df))

    def test_octave_collapses_to_track_ref(self):
        df = _stream(list(range(1000, 1015)), 2.0)
        out = _apply(df)
        track = out[out["op"] == TRACK_REF_OP]
        self.assertEqual(len(track), TRACK_REF_SUBREG_COUNT)
        self.assertEqual(len(out[(out["reg"] == 7) & (out["op"] == SET_OP)]), 0)

    def test_octave_round_trips(self):
        df = _stream(list(range(1000, 1015)), 2.0)
        out = _apply(df)
        self.assertEqual(_decoded_reg(df, 7), _decoded_reg(out, 7))
        self.assertEqual(_decoded_reg(df, 0), _decoded_reg(out, 0))

    def test_fifth_with_detune_round_trips(self):
        df = _stream(list(range(2000, 2020)), 1.5, detune=3)
        out = _apply(df)
        self.assertTrue(bool((out["op"] == TRACK_REF_OP).any()))
        self.assertEqual(_decoded_reg(df, 7), _decoded_reg(out, 7))

    def test_below_min_duration_not_collapsed(self):
        df = _stream(list(range(1000, 1005)), 2.0)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == TRACK_REF_OP).any()))

    def test_non_interval_not_collapsed(self):
        rows = []
        for fr in range(15):
            rows.append(_frame())
            rows.append(_r(0, 1000 + fr))
            rows.append(_r(7, 1234 + fr * 7))
        df = pd.DataFrame(rows)
        out = _apply(df)
        self.assertFalse(bool((out["op"] == TRACK_REF_OP).any()))


if __name__ == "__main__":
    unittest.main()
