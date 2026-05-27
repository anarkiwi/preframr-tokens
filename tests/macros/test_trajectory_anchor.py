"""Trajectory-anchor detector + TrajectoryAnchorPass tests: the synthetic cases
a-h from the design doc at P=R=1.0 (+-2 frames) with the arp-vs-fast-melody
autocorrelation discriminator (c/h) as the crucial pair, the ``traj_anchor``
annotation, the FreqTrajectoryPass boundary handoff (every emitted trajectory
begins on an anchor, round-trip byte-exact), and the ``freq_unq`` threading."""

import unittest

import numpy as np
import pandas as pd

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.freq_trajectory_pass import FreqTrajectoryPass
from preframr_tokens.macros.passes_base import _frame_index
from preframr_tokens.macros.trajectory_anchor import (
    AnchorParams,
    TrajectoryAnchorPass,
    detect_anchors,
)
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    FRAME_REG,
    FREQ_TRAJ_OP,
    FT_SUBREG_FLAGS,
    MODEL_PDTYPE,
    SET_OP,
)

P = AnchorParams()
MATCH_W = 2


def _pr(detected, expected, w=MATCH_W):
    """(precision, recall) of ``detected`` vs ``expected`` anchor frames at
    +-``w`` tolerance."""
    if not expected:
        return float("nan"), float("nan")
    recall = sum(any(abs(e - d) <= w for d in detected) for e in expected) / len(
        expected
    )
    precision = (
        sum(any(abs(d - e) <= w for e in expected) for d in detected) / len(detected)
        if detected
        else 0.0
    )
    return precision, recall


class FakeArgs:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)


class TestDetectAnchors(unittest.TestCase):
    """Cases a-h: detect_anchors at P=R=1.0 within +-2 frames."""

    def _assert_pr(self, value, gate_on, kind, expected):
        det = detect_anchors(np.asarray(value, float), list(gate_on), kind, params=P)
        precision, recall = _pr(det, expected)
        self.assertEqual(
            (precision, recall), (1.0, 1.0), f"{kind} det={det} expected={expected}"
        )
        return det

    def test_a_filter_ultraslow_ramp_is_one_onset(self):
        """0->2000 cutoff over 4000 frames with a gate on an unrelated voice (the
        global FILTER is not voice-gated, so the gate is ignored): the whole
        ultra-slow sweep is one ramp, not per-band steps."""
        value = np.linspace(0, 2000, 4000)
        det = detect_anchors(value, [10, 500, 1234], "FILTER", params=P)
        self.assertEqual(len(det), 1, det)

    def test_b_filter_staircase_is_one_ramp(self):
        """30 discrete +60 steps collapse to one ramp, not 30 notes."""
        value = np.repeat(np.arange(30) * 60.0, 8)
        det = detect_anchors(value, [], "FILTER", params=P)
        self.assertEqual(len(det), 1, det)

    def test_c_freq_arp_collapses_to_onset(self):
        """A 3-tone arp under one held gate yields one anchor (the onset)."""
        value = np.tile([60.0, 64.0, 67.0], 40)
        self._assert_pr(value, [0], "FREQ", [0])

    def test_d_freq_vibrato_suppressed(self):
        """A held note plus +-1 semitone vibrato yields one anchor."""
        t = np.arange(240)
        value = 60 + np.round(np.sin(2 * np.pi * t / 6))
        self._assert_pr(value, [0], "FREQ", [0])

    def test_e_freq_stepped_melody_one_per_note(self):
        """Eight distinct held pitches, gate on each, yield eight anchors."""
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        value = np.repeat(pitches, 15).astype(float)
        gate_on = [i * 15 for i in range(8)]
        det = self._assert_pr(value, gate_on, "FREQ", gate_on)
        self.assertEqual(len(det), 8, det)

    def test_f_freq_legato_slide_two_anchors(self):
        """A legato slide under a held gate yields the initial note plus the
        off-gate slide onset."""
        value = np.concatenate(
            [np.full(20, 60.0), np.linspace(60, 72, 40), np.full(20, 72.0)]
        )
        det = detect_anchors(value, [0], "FREQ", params=P)
        self.assertEqual(len(det), 2, det)
        self.assertLessEqual(min(det), 2)
        self.assertGreater(max(det), 20)

    def test_g_freq_repeated_pitch_caught_by_gate(self):
        """A repeated same pitch (no value change) is recovered from the gate."""
        value = np.full(80, 60.0)
        gate_on = [i * 15 for i in range(5)]
        det = self._assert_pr(value, gate_on, "FREQ", gate_on)
        self.assertEqual(len(det), 5, det)

    def test_h_freq_fast_aperiodic_melody_kept(self):
        """A genuinely aperiodic dense line is kept (one anchor per note)."""
        seq = [68, 61, 71, 65, 63, 70, 60, 67, 64, 66, 62, 69]
        value = np.repeat(seq, 4).astype(float)
        expected = [0] + [4 * i for i in range(1, 12)]
        det = self._assert_pr(value, [0], "FREQ", expected)
        self.assertEqual(len(det), 12, det)

    def test_c_vs_h_autocorrelation_discriminator(self):
        """The crucial pair, intrinsic-only: both dense/fast, but the periodic
        arp collapses to one onset while the aperiodic melody is kept."""
        arp = np.tile(np.repeat([60.0, 64.0, 67.0], 4), 10)
        seq = [68, 61, 71, 65, 63, 70, 60, 67, 64, 66, 62, 69]
        melody = np.repeat(seq, 4).astype(float)
        self.assertEqual(len(detect_anchors(arp, [], "FREQ", params=P)), 1)
        self.assertEqual(len(detect_anchors(melody, [], "FREQ", params=P)), 11)

    def test_filter_intrinsic_only_ignores_gate(self):
        """The global FILTER is intrinsic-only: a gate does not add anchors."""
        value = np.repeat(np.arange(30) * 60.0, 8)
        without = detect_anchors(value, [], "FILTER", params=P)
        with_gate = detect_anchors(value, [5, 25, 99], "FILTER", params=P)
        self.assertEqual(without, with_gate)

    def test_all_silent_returns_no_anchors(self):
        self.assertEqual(detect_anchors(np.full(50, np.nan), [], "FREQ", params=P), [])


def _ctrl_reg(voice):
    return voice * 7 + 4


def _frame_row():
    return {"reg": FRAME_REG, "val": 0, "diff": 32, "freq_unq": 0}


def _build_stage_df(rows):
    """Build a df shaped like the FreqTrajectoryPass input: a FRAME marker
    precedes each frame's writes (expand_ops drops writes before the first
    frame). ``rows`` is a list of frames, each a list of ``(reg, val, freq_unq)``
    writes (freq_unq optional)."""
    out = []
    for writes in rows:
        out.append(_frame_row())
        for reg, val, *rest in writes:
            fu = rest[0] if rest else val
            out.append({"reg": reg, "val": val, "diff": 32, "freq_unq": fu})
    out.append(_frame_row())
    return pd.DataFrame(out)


class TestTrajectoryAnchorPass(unittest.TestCase):
    def test_disabled_is_noop(self):
        df = _build_stage_df([[(0, 100)]] * 4)
        out = TrajectoryAnchorPass().apply(
            df.copy(), args=FakeArgs(trajectory_anchor_pass=False)
        )
        self.assertNotIn("traj_anchor", out.columns)

    def test_gate_retrigger_marks_freq_rows(self):
        """A retriggered same pitch has no value change, so its anchors come
        purely from the gate 0->1 transitions."""
        freq = 4000
        rows = []
        for _ in range(5):
            rows.append([(0, 50, freq), (_ctrl_reg(0), 0x11)])
            rows.append([(0, 50, freq)])
            rows.append([(_ctrl_reg(0), 0x10)])
            rows.append([(0, 50, freq)])
        out = TrajectoryAnchorPass().apply(
            _build_stage_df(rows), args=FakeArgs(trajectory_anchor_pass=True)
        )
        self.assertIn("traj_anchor", out.columns)
        anchored = out[(out["reg"] == 0) & out["traj_anchor"]]
        self.assertEqual(len(anchored), 5)

    def test_filter_ramp_single_anchor(self):
        rows = [[(21, v << 8)] for v in range(0, 60, 2)]
        out = TrajectoryAnchorPass().apply(
            _build_stage_df(rows), args=FakeArgs(trajectory_anchor_pass=True)
        )
        anchored = out[(out["reg"] == 21) & out["traj_anchor"]]
        self.assertEqual(len(anchored), 1)


class TestFreqTrajectoryHandoff(unittest.TestCase):
    """FreqTrajectoryPass consumes traj_anchor: trajectories begin on anchors,
    none spans an anchor, and the expand round-trip stays byte-exact. The column
    is set directly here so the handoff is tested independently of the detector."""

    @staticmethod
    def _per_frame(df, reg):
        df = df.copy()
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        if "subreg" not in df.columns:
            df["subreg"] = -1
        dec = expand_ops(df, strict=False).reset_index(drop=True)
        regs = dec["reg"].to_numpy()
        vals = dec["val"].to_numpy()
        cur = 0
        out = []
        for i in range(len(dec)):
            r = int(regs[i])
            if r == FRAME_REG:
                out.append(cur)
            elif r == reg:
                cur = int(vals[i])
        return out

    @staticmethod
    def _run_cluster_df(anchor_at):
        """reg-0 RUN [100,150,200,260,330] (legacy emits one cluster); mark
        ``traj_anchor`` True on the reg-0 SETs at the given occurrence indices."""
        df = _build_stage_df([[(0, v)] for v in (100, 150, 200, 260, 330)])
        df["traj_anchor"] = False
        reg0_rows = df.index[df["reg"] == 0].tolist()
        for k in anchor_at:
            df.loc[reg0_rows[k], "traj_anchor"] = True
        return df

    def _origin_frames(self, out):
        f_idx = _frame_index(out).to_numpy()
        mask = (
            (out["op"] == FREQ_TRAJ_OP) & (out["subreg"] == FT_SUBREG_FLAGS)
        ).to_numpy()
        return sorted(int(f) for f in f_idx[mask])

    def test_anchor_splits_run_origins_on_anchors_lossless(self):
        df = self._run_cluster_df(anchor_at=[0, 2])
        anchor_frames = set(
            int(f) for f in _frame_index(df)[(df["reg"] == 0) & df["traj_anchor"]]
        )
        out = FreqTrajectoryPass().apply(
            df.copy(), args=FakeArgs(freq_trajectory_pass=True)
        )
        origins = self._origin_frames(out)
        self.assertEqual(len(origins), 2, origins)
        self.assertTrue(set(origins).issubset(anchor_frames), (origins, anchor_frames))
        self.assertEqual(self._per_frame(df, 0), self._per_frame(out, 0))

    def test_no_trajectory_spans_an_anchor(self):
        legacy = FreqTrajectoryPass().apply(
            self._run_cluster_df(anchor_at=[]).drop(columns=["traj_anchor"]),
            args=FakeArgs(freq_trajectory_pass=True),
        )
        self.assertEqual(len(self._origin_frames(legacy)), 1)
        split = FreqTrajectoryPass().apply(
            self._run_cluster_df(anchor_at=[0, 2]),
            args=FakeArgs(freq_trajectory_pass=True),
        )
        self.assertGreater(len(self._origin_frames(split)), 1)

    def test_leading_pre_anchor_segment_not_emitted(self):
        """With the only anchor mid-run, the leading SETs stay raw (their origin
        would not be an anchor) yet the round-trip stays lossless."""
        df = self._run_cluster_df(anchor_at=[2])
        out = FreqTrajectoryPass().apply(
            df.copy(), args=FakeArgs(freq_trajectory_pass=True)
        )
        anchor_frames = set(
            int(f) for f in _frame_index(df)[(df["reg"] == 0) & df["traj_anchor"]]
        )
        self.assertTrue(set(self._origin_frames(out)).issubset(anchor_frames))
        self.assertEqual(self._per_frame(df, 0), self._per_frame(out, 0))

    def test_absent_column_matches_legacy_behaviour(self):
        df = self._run_cluster_df(anchor_at=[0, 2]).drop(columns=["traj_anchor"])
        legacy = FreqTrajectoryPass().apply(
            df.copy(), args=FakeArgs(freq_trajectory_pass=True)
        )
        anchored = TrajectoryAnchorPass().apply(
            df.copy(), args=FakeArgs(trajectory_anchor_pass=False)
        )
        out = FreqTrajectoryPass().apply(
            anchored.copy(), args=FakeArgs(freq_trajectory_pass=True)
        )
        self.assertEqual(self._per_frame(legacy, 0), self._per_frame(out, 0))


class TestFreqUnqThreading(unittest.TestCase):
    """The parser preserves freq_unq through the column-restricting stages, and
    only stashes it when the anchor pass is enabled."""

    def _loader(self, enabled):
        return RegLogParser(FakeArgs(trajectory_anchor_pass=enabled, cents=50))

    def test_squeeze_changes_preserves_freq_unq(self):
        loader = self._loader(True)
        df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 0, "val": 5, "freq_unq": 4000},
                {"clock": 2, "irq": 2, "reg": 0, "val": 5, "freq_unq": 4000},
                {"clock": 3, "irq": 3, "reg": 0, "val": 6, "freq_unq": 4500},
            ]
        )
        result = loader._squeeze_changes(df)
        self.assertIn("freq_unq", result.columns)
        self.assertEqual(list(result["freq_unq"]), [4000, 4500])

    def test_add_frame_reg_preserves_freq_unq(self):
        loader = self._loader(True)
        df = pd.DataFrame(
            [
                {"clock": 0, "reg": 0, "val": 1, "irq": 0, "freq_unq": 4000},
                {"clock": 32768, "reg": 7, "val": 2, "irq": 19000, "freq_unq": 5000},
            ],
            dtype=MODEL_PDTYPE,
        )
        result = loader._add_frame_reg(df, 512, min_irq_prop=0.5)[1]
        self.assertIn("freq_unq", result.columns)
        self.assertIn(4000, list(result["freq_unq"]))
        self.assertIn(5000, list(result["freq_unq"]))

    def test_squeeze_changes_without_freq_unq_unchanged(self):
        loader = self._loader(False)
        df = pd.DataFrame(
            [
                {"clock": 1, "irq": 1, "reg": 0, "val": 5},
                {"clock": 2, "irq": 2, "reg": 0, "val": 6},
            ]
        )
        result = loader._squeeze_changes(df)
        self.assertEqual(list(result.columns), ["clock", "irq", "reg", "val"])

    def test_stash_freq_unq_copies_val(self):
        loader = self._loader(True)
        df = pd.DataFrame([{"clock": 1, "irq": 1, "reg": 0, "val": 4000}])
        stashed = loader._stash_freq_unq(df)
        self.assertEqual(list(stashed["freq_unq"]), [4000])

    def test_anchor_enabled_gate(self):
        self.assertTrue(self._loader(True)._anchor_enabled())
        self.assertFalse(self._loader(False)._anchor_enabled())


if __name__ == "__main__":
    unittest.main()
