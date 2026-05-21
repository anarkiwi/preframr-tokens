"""SetToDiffTransform tests: motion-only conversion, gate-transition anchoring, round-trip."""

import argparse
import unittest

import pandas as pd

from preframr_tokens.macros import (  # noqa: F401  pylint: disable=unused-import
    transforms_audio_bit_exact,
    transforms_bit_exact,
    transforms_set_to_diff,
)
from preframr_tokens.macros.transform import _REGISTRY
from preframr_tokens.macros.transforms_set_to_diff import SetToDiffTransform
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FC_LO_REG,
    FC_PRESET_OP,
    FLIP_OP,
    FLIP2_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    PWM_PRESET_OP,
    PWM_SUSTAIN_OP,
    SET_OP,
    VOICE_REG,
    WAVETABLE_SUSTAIN_OP,
)


def _row(op, reg, val, subreg=-1, diff=0):
    return {
        "op": int(op),
        "reg": int(reg),
        "subreg": int(subreg),
        "val": int(val),
        "diff": int(diff),
    }


def _svt(voice_order):
    out = 0
    for slot, v in enumerate(voice_order):
        out |= (int(v) + 1) << (slot * 2)
    return int(out)


def _df_freq_sweep_no_gate_change():
    """Voice 0 freq_lo sweeps with no CTRL writes — gate stays at initial 0."""
    svt = _svt([0])
    rows = []
    for f, freq in enumerate([0x10, 0x12, 0x14, 0x16, 0x18, 0x1A]):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        rows.append(_row(SET_OP, VOICE_REG, 0))
        rows.append(_row(SET_OP, 0, freq))
    return pd.DataFrame(rows)


def _df_freq_sweep_with_gate_on_at_f3():
    """Voice 0: frames 1-2 gate off (0x40), frame 3 gate on (0x41), then on."""
    svt = _svt([0])
    rows = []
    gate_sequence = [0x40, 0x40, 0x41, 0x41, 0x41, 0x41]
    freq_sequence = [0x10, 0x12, 0x14, 0x16, 0x18, 0x1A]
    for f, (gate, freq) in enumerate(zip(gate_sequence, freq_sequence)):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        rows.append(_row(SET_OP, VOICE_REG, 0))
        rows.append(_row(SET_OP, 0, freq))
        rows.append(_row(SET_OP, 4, gate))
    return pd.DataFrame(rows)


def _df_with_ctrl_and_ad_sweep():
    """Voice 0: sweep ctrl (non-motion) and AD (non-motion) — should never convert."""
    svt = _svt([0])
    rows = []
    ctrls = [0x40, 0x41, 0x42, 0x43]
    ads = [0x10, 0x20, 0x30, 0x40]
    for f, (c, ad) in enumerate(zip(ctrls, ads)):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        rows.append(_row(SET_OP, VOICE_REG, 0))
        rows.append(_row(SET_OP, 4, c))
        rows.append(_row(SET_OP, 5, ad))
    return pd.DataFrame(rows)


def _df_filter_cutoff_sweep():
    svt = _svt([0])
    rows = []
    for f, fc in enumerate([0x10, 0x20, 0x30, 0x40]):
        rows.append(_row(SET_OP, FRAME_REG, svt))
        rows.append(_row(SET_OP, VOICE_REG, 0))
        rows.append(_row(SET_OP, 4, 0x41))
        rows.append(_row(SET_OP, FC_LO_REG, fc))
    return pd.DataFrame(rows)


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("set_to_diff", _REGISTRY)
        self.assertEqual(_REGISTRY["set_to_diff"], SetToDiffTransform)


class TestDefaultOffIsNoOp(unittest.TestCase):
    def test_disabled_passes_through(self):
        t = SetToDiffTransform()
        df = _df_freq_sweep_no_gate_change()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=False))
        pd.testing.assert_frame_equal(out, df)


class TestMotionRegConversion(unittest.TestCase):
    def test_freq_lo_sweep_without_gate_change_converts_after_first(self):
        t = SetToDiffTransform()
        df = _df_freq_sweep_no_gate_change()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        freq_rows = out[(out["reg"] == 0)]
        self.assertEqual(int(freq_rows.iloc[0]["op"]), int(SET_OP))
        for i in range(1, len(freq_rows)):
            self.assertEqual(int(freq_rows.iloc[i]["op"]), int(DIFF_OP))
            self.assertEqual(int(freq_rows.iloc[i]["val"]), 2)

    def test_filter_cutoff_sweep_converts(self):
        t = SetToDiffTransform()
        df = _df_filter_cutoff_sweep()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        fc_rows = out[out["reg"] == int(FC_LO_REG)]
        self.assertEqual(int(fc_rows.iloc[0]["op"]), int(SET_OP))
        for i in range(1, len(fc_rows)):
            self.assertEqual(int(fc_rows.iloc[i]["op"]), int(DIFF_OP))


class TestNonMotionRegsUntouched(unittest.TestCase):
    def test_ctrl_and_ad_writes_stay_set(self):
        t = SetToDiffTransform()
        df = _df_with_ctrl_and_ad_sweep()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        ctrl_rows = out[out["reg"] == 4]
        ad_rows = out[out["reg"] == 5]
        self.assertTrue(all(int(o) == int(SET_OP) for o in ctrl_rows["op"]))
        self.assertTrue(all(int(o) == int(SET_OP) for o in ad_rows["op"]))


class TestGateTransitionAnchor(unittest.TestCase):
    def test_gate_on_frame_and_following_frame_stay_set(self):
        t = SetToDiffTransform()
        df = _df_freq_sweep_with_gate_on_at_f3()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        out = out.reset_index(drop=True)
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        self.assertEqual(int(op_by_frame[1]), int(SET_OP))
        self.assertEqual(int(op_by_frame[2]), int(DIFF_OP))
        self.assertEqual(int(op_by_frame[3]), int(SET_OP))
        self.assertEqual(int(op_by_frame[4]), int(SET_OP))
        self.assertEqual(int(op_by_frame[5]), int(DIFF_OP))
        self.assertEqual(int(op_by_frame[6]), int(DIFF_OP))

    def test_hard_restart_op_acts_as_gate_transition(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x10),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x12),
            _row(HARD_RESTART_OP, 4, (0x40 << 8) | 0x41),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x14),
            _row(SET_OP, 4, 0x41),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x16),
            _row(SET_OP, 4, 0x41),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        out = out.reset_index(drop=True)
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        self.assertEqual(int(op_by_frame[1]), int(SET_OP))
        self.assertEqual(int(op_by_frame[2]), int(SET_OP))
        self.assertEqual(int(op_by_frame[3]), int(SET_OP))
        self.assertEqual(int(op_by_frame[4]), int(DIFF_OP))


class TestAnchorMaterialization(unittest.TestCase):
    def test_diff_at_anchor_frame_becomes_set(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x10),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(DIFF_OP, 0, 5),
            _row(SET_OP, 4, 0x41),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        val_by_frame = dict(zip(freq_rows["f"], freq_rows["val"]))
        self.assertEqual(int(op_by_frame[1]), int(SET_OP))
        self.assertEqual(int(op_by_frame[2]), int(SET_OP))
        self.assertEqual(int(val_by_frame[2]), 0x15)

    def test_flip_at_anchor_becomes_set(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x10),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(FLIP_OP, 0, 3),
            _row(SET_OP, 4, 0x41),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        val_by_frame = dict(zip(freq_rows["f"], freq_rows["val"]))
        self.assertEqual(int(op_by_frame[2]), int(SET_OP))
        self.assertEqual(int(val_by_frame[2]), 0x13)

    def test_flip2_at_anchor_becomes_set(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        a, b = 4, -2
        packed = ((a & 0xFF) << 8) | (b & 0xFF)
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x20),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(FLIP2_OP, 0, packed, subreg=6),
            _row(SET_OP, 4, 0x41),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        val_by_frame = dict(zip(freq_rows["f"], freq_rows["val"]))
        self.assertEqual(int(op_by_frame[2]), int(SET_OP))
        self.assertEqual(int(val_by_frame[2]), 0x20 + a + b)

    def test_diff_at_non_anchor_stays_diff(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x10),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(DIFF_OP, 0, 3),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        f_idx = (out["reg"] == int(FRAME_REG)).astype(int).cumsum()
        freq_rows = out[out["reg"] == 0].copy()
        freq_rows["f"] = f_idx.loc[freq_rows.index].values
        op_by_frame = dict(zip(freq_rows["f"], freq_rows["op"]))
        self.assertEqual(int(op_by_frame[2]), int(DIFF_OP))


def _df_lonely_pwm_sustain():
    """Frame 1: transition (gate off→on). Frame 2: transition+1 (anchor). Frame 3: sustain (non-anchor), voice 0 emits lonely PWM_PRESET."""
    svt = _svt([0])
    rows = [
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(SET_OP, 0, 0x10),
        _row(SET_OP, 4, 0x41),
        _row(SET_OP, 5, 0x90),
        _row(SET_OP, 6, 0xA0),
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(SET_OP, 4, 0x41),
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(PWM_PRESET_OP, 2, 5),
    ]
    return pd.DataFrame(rows)


def _df_wavetable_sustain():
    """3-frame fixture: transition, transition+1 (anchor), then sustain with lonely PWM + global FC_PRESET."""
    svt = _svt([0])
    rows = [
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(SET_OP, 0, 0x10),
        _row(SET_OP, 4, 0x41),
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(SET_OP, 4, 0x41),
        _row(SET_OP, FRAME_REG, svt),
        _row(SET_OP, VOICE_REG, 0),
        _row(PWM_PRESET_OP, 2, 5),
        _row(FC_PRESET_OP, int(FC_LO_REG), 7),
    ]
    return pd.DataFrame(rows)


class TestSustainCollapse(unittest.TestCase):
    def test_lonely_pwm_collapses_to_pwm_sustain(self):
        t = SetToDiffTransform()
        df = _df_lonely_pwm_sustain()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        self.assertEqual(int((out["op"] == int(PWM_SUSTAIN_OP)).sum()), 1)
        self.assertEqual(int((out["op"] == int(PWM_PRESET_OP)).sum()), 0)
        sustain_rows = out[out["op"] == int(PWM_SUSTAIN_OP)]
        self.assertEqual(int(sustain_rows.iloc[0]["reg"]), 2)
        self.assertEqual(int(sustain_rows.iloc[0]["val"]), 5)
        per_frame_voice_markers = []
        cur = 0
        for i in range(len(out)):
            r = int(out.iloc[i]["reg"])
            if r == int(FRAME_REG):
                per_frame_voice_markers.append(cur)
                cur = 0
            elif r == int(VOICE_REG):
                cur += 1
        per_frame_voice_markers.append(cur)
        per_frame_voice_markers = per_frame_voice_markers[1:]
        self.assertEqual(per_frame_voice_markers[-1], 0)

    def test_pwm_plus_fc_collapses_to_wavetable_sustain(self):
        t = SetToDiffTransform()
        df = _df_wavetable_sustain()
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        self.assertEqual(int((out["op"] == int(WAVETABLE_SUSTAIN_OP)).sum()), 1)
        self.assertEqual(int((out["op"] == int(PWM_PRESET_OP)).sum()), 0)
        self.assertEqual(int((out["op"] == int(FC_PRESET_OP)).sum()), 0)
        wt_row = out[out["op"] == int(WAVETABLE_SUSTAIN_OP)].iloc[0]
        packed = int(wt_row["val"])
        self.assertEqual((packed >> 8) & 0xFF, 5)
        self.assertEqual(packed & 0xFF, 7)

    def test_anchor_frame_pwm_is_not_collapsed(self):
        t = SetToDiffTransform()
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 4, 0x41),
            _row(PWM_PRESET_OP, 2, 5),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        self.assertEqual(int((out["op"] == int(PWM_SUSTAIN_OP)).sum()), 0)
        self.assertEqual(int((out["op"] == int(PWM_PRESET_OP)).sum()), 1)

    def test_sustain_collapse_roundtrip(self):
        t = SetToDiffTransform()
        df = _df_lonely_pwm_sustain()
        forward = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True), df.reset_index(drop=True)
        )

    def test_wavetable_collapse_roundtrip(self):
        t = SetToDiffTransform()
        df = _df_wavetable_sustain()
        forward = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True), df.reset_index(drop=True)
        )


class TestRoundTrip(unittest.TestCase):
    def test_freq_sweep_roundtrip(self):
        t = SetToDiffTransform()
        df = _df_freq_sweep_no_gate_change()
        forward = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True), df.reset_index(drop=True)
        )

    def test_gate_anchored_roundtrip(self):
        t = SetToDiffTransform()
        df = _df_freq_sweep_with_gate_on_at_f3()
        forward = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        recovered = t.inverse(forward)
        pd.testing.assert_frame_equal(
            recovered.reset_index(drop=True), df.reset_index(drop=True)
        )


class TestConvertRegsFreqOnly(unittest.TestCase):
    def test_freq_only_convert_regs_skips_pw_and_fc(self):
        from preframr_tokens.stfconstants import PWM_PRESET_OP

        t = SetToDiffTransform(convert_regs=[0, 1])
        svt = _svt([0])
        rows = [
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x10),
            _row(SET_OP, 2, 0x20),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x12),
            _row(SET_OP, 2, 0x22),
            _row(SET_OP, 4, 0x40),
            _row(SET_OP, FRAME_REG, svt),
            _row(SET_OP, VOICE_REG, 0),
            _row(SET_OP, 0, 0x14),
            _row(SET_OP, 2, 0x24),
            _row(SET_OP, 4, 0x40),
        ]
        df = pd.DataFrame(rows)
        out = t.forward(df, args=argparse.Namespace(set_to_diff_pass=True))
        reg0_ops = out[out["reg"] == 0]["op"].tolist()
        reg2_ops = out[out["reg"] == 2]["op"].tolist()
        self.assertEqual(int(reg0_ops[0]), int(SET_OP))
        self.assertTrue(any(int(o) == int(DIFF_OP) for o in reg0_ops[1:]))
        self.assertTrue(all(int(o) == int(SET_OP) for o in reg2_ops))


if __name__ == "__main__":
    unittest.main()
