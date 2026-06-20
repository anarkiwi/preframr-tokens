"""Unit tests for the lossless lane-grammar miner (lane_grammar) and the
libsidplayfp fidelity-gate helpers (lsp_validate). These are pure-function paths
(no fixture needed): the lane parser reconstructs every primitive op-cover
exactly, and the state-sequence / frame-alignment helpers are deterministic."""

import numpy as np

from preframr_tokens.codec import lane_grammar as LG
from preframr_tokens.codec import lsp_validate as LV


def test_parse_lane_reconstructs_every_op_class():
    cases = [
        [5, 5, 5, 1, 2, 3, 4, 4, 4],  # hold + accum + hold
        [0, 1, 2, 3, 4, 5, 6, 7],  # accum
        [7, 7, 7, 7, 7],  # hold
        [1, 2, 1, 2, 1, 2, 1, 2],  # period 2
        [3, 1, 4, 1, 5, 9],  # sets
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # negative accum
    ]
    for seq in cases:
        ops = LG.parse_lane(seq)
        rec = LG.reconstruct(ops)
        assert np.array_equal(rec, np.asarray(seq, dtype=np.int64)), seq


def test_lanes_from_state_partitions_all_lanes():
    s = np.zeros((16, 25), dtype=np.int64)
    s[:, 0] = 100
    s[:, 1] = 1  # freq hi
    s[:, 4] = 0x41  # voice0 ctrl gate+tri
    lanes = LG.lanes_from_state(s)
    assert len(lanes) == 18  # 3 voices x 5 lanes + 3 global lanes
    assert np.array_equal(lanes[(0, "freq")], s[:, 0] + 256 * s[:, 1])


def test_mine_tune_handles_short_input(tmp_path):
    import pandas as pd

    rows = []
    for f in range(8):
        rows.append({"clock": f * int(LV.CPF), "reg": 0, "val": f % 4, "chipno": 0})
        rows.append({"clock": f * int(LV.CPF), "reg": 1, "val": 1, "chipno": 0})
    p = tmp_path / "tiny.parquet"
    pd.DataFrame(rows).to_parquet(p)
    out = LG.mine_tune(str(p), LV.CPF, maxframes=8)
    assert out is None or "frames" in out or "error" in out


def test_lsp_validate_state_and_alignment_helpers():
    cyc = np.array([0, 19656, 39312, 58968, 78624], dtype=np.int64)
    assert LV.first_play_cycle(cyc) == 0
    seq = LV.state_seq(cyc, [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], 0, mask=True)
    assert len(seq) == 5
    raw = LV.state_seq(cyc, [3, 3, 3], [255, 255, 255][:3], 0, mask=False)
    assert raw  # PW-high masking off
    changed = LV.changed_frames(cyc, [0, 1, 0, 1, 0], [1, 2, 3, 4, 5], 0)
    assert changed == {0, 1, 2, 3, 4}
    assert LV.best_lag([(1, 2)] * 30, [(1, 2)] * 30) == 0


def test_cpf_constants():
    assert LV.CPF == 19656.0
    assert LV.NTSC_CPF == 17095.0


def _burst_clocks(period, nframes, writes_per_burst=4):
    """Synthetic write clocks: one tight burst of writes every ``period`` cycles."""
    clocks = []
    for f in range(nframes):
        base = f * period
        clocks += [base + 50 * w for w in range(writes_per_burst)]
    return np.asarray(clocks, dtype=np.int64)


def test_detect_play_period_single_speed_returns_cpf():
    # A clean single-speed cadence (one burst per raster frame) must resolve to
    # exactly CPF -- the detector must never perturb single-speed framing.
    cyc = _burst_clocks(int(LV.CPF), 200)
    assert LV.detect_play_period(cyc) == LV.CPF
    # Mild jitter (real dumps wobble a few cycles) still resolves to CPF.
    rng = np.random.default_rng(0)
    jit = cyc + rng.integers(-30, 30, size=len(cyc))
    assert LV.detect_play_period(np.sort(jit)) == LV.CPF


def test_detect_play_period_multispeed_finds_subframe():
    # 2x multispeed: two bursts per raster frame -> period == CPF/2, classified
    # multispeed (below the single-speed threshold).
    cyc = _burst_clocks(int(LV.CPF // 2), 400)
    period = LV.detect_play_period(cyc)
    assert abs(period - LV.CPF / 2) < 60
    assert period < 0.9 * LV.CPF  # multispeed
    # 3x multispeed.
    cyc3 = _burst_clocks(int(LV.CPF // 3), 600)
    assert abs(LV.detect_play_period(cyc3) - LV.CPF / 3) < 60


def test_detect_play_period_too_few_bursts_is_single_speed():
    # Not enough bursts to infer a cadence -> default to the raster frame.
    assert LV.detect_play_period(np.array([0, 50, 100], dtype=np.int64)) == LV.CPF
    assert LV.detect_play_period(np.empty(0, dtype=np.int64)) == LV.CPF


def test_framing_change_loss_lossless_vs_lossy():
    # 2x multispeed where the two sub-frame play-calls write DIFFERENT values to
    # the same register: single-CPF framing keeps only the 2nd (drops ~half the
    # changes); framing at CPF/2 keeps both (lossless).
    half = int(LV.CPF // 2)
    cyc, reg, val = [], [], []
    for f in range(200):
        cyc += [f * int(LV.CPF), f * int(LV.CPF) + half]
        reg += [0, 0]
        val += [f % 251, (f * 7 + 3) % 251]  # two distinct writes per frame
    cyc = np.asarray(cyc, dtype=np.int64)
    loss_cpf = LV.framing_change_loss(cyc, reg, val, LV.CPF)
    loss_half = LV.framing_change_loss(cyc, reg, val, half)
    assert loss_cpf > 0.4  # single-CPF drops ~half the bus changes
    assert loss_half < 0.005  # sub-frame framing is lossless (modulo 1 edge frame)
