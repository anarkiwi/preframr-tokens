"""Full-pipeline per-frame register fidelity gate: parse a fixture through the
whole pipeline under each macro and assert decoded per-frame register state
matches the no-macro baseline. Single-pass round-trip tests compare value
sequences and miss frame-placement bugs (right values, wrong frames) that break
pitch/waveform; divergence is reported per register-class to localise them."""

import unittest
from types import SimpleNamespace

import numpy as np

from tests.sid_fixtures import FixtureUnavailable, grid_runner_dumps

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    PWM_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
)
from preframr_tokens.reglogparser import RegLogParser, remove_voice_reg
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_TRIPLE_OP,
    FRAME_REG,
    FREQ_RUN_OP,
)

_BASE = dict(
    cents=50,
    exclude_list=None,
    min_irq=int(1.5e4),
    max_irq=int(2.5e4),
    min_song_tokens=0,
    diffq=4,
    loop_lookahead=3,
    coarsen_min_len=16,
    voice_trajectory_window=8,
    pipeline_spec="",
    meta_exclude_digi=False,
    meta_irq_lo=0,
    meta_irq_hi=0,
    meta_require=False,
)
_ALL_FLAGS = (
    "slope_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c3",
    "legato_pass_c4",
    "legato_pass_c7",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "fuzzy_loop_pass",
    "fuzzy_fp_adsr",
    "coarsen_pass",
    "mode_vol_flip_pass",
    "voice_trajectory_pass",
    "voice_trajectory_distributed_pass",
    "set_to_diff_pass",
    "oscillate_env_pass",
    "vibrato_env_pass",
    "freq_nudge_pass",
    "freq_run_pass",
    "release_update_pass",
    "ctrl_triple_pass",
    "lonely_catch_all",
)

_CLASS = {}
for _v in range(3):
    _CLASS[FREQ_REGS_BY_VOICE[_v]] = f"v{_v}.FREQ"
    _CLASS[PWM_REGS_BY_VOICE[_v]] = f"v{_v}.PW"
    _CLASS[CTRL_REGS_BY_VOICE[_v]] = f"v{_v}.CTRL"
    _CLASS[AD_REGS_BY_VOICE[_v]] = f"v{_v}.AD"
    _CLASS[SR_REGS_BY_VOICE[_v]] = f"v{_v}.SR"


def _args(**overrides):
    cfg = dict(_BASE)
    for flag in _ALL_FLAGS:
        cfg[flag] = False
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _decode(args, fixture):
    parser = RegLogParser(args=args)
    xdf = next(parser.parse(fixture, max_perm=1, require_pq=False, reparse=True), None)
    assert xdf is not None and len(xdf), "fixture produced no rows"
    return xdf


def _per_frame_state(xdf):
    """Decoded per-frame state of SID registers 0-24, as ``(n_frames, 25)``."""
    df, _ = remove_voice_reg(xdf.copy(), {})
    dec = expand_ops(df, strict=False).reset_index(drop=True)
    regs = dec["reg"].to_numpy()
    vals = dec["val"].to_numpy()
    n_frames = int((regs == FRAME_REG).sum()) + 1
    state = np.zeros((n_frames, 25), dtype=np.int64)
    cur = np.zeros(25, dtype=np.int64)
    cf = 0
    for i in range(len(dec)):
        reg = int(regs[i])
        if reg == FRAME_REG:
            if cf < n_frames:
                state[cf] = cur
            cf += 1
        elif 0 <= reg <= 24:
            cur[reg] = int(vals[i])
    while cf < n_frames:
        state[cf] = cur
        cf += 1
    return state


def _divergence(ref, st):
    n = min(len(ref), len(st))
    by_class = {}
    diff = ref[:n] != st[:n]
    for reg in range(25):
        c = int(diff[:, reg].sum())
        if c:
            by_class[_CLASS.get(reg, f"reg{reg}")] = (
                by_class.get(_CLASS.get(reg), 0) + c
            )
    return n, by_class


class TestFullPipelineFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Source the Grid Runner dump via the HVSC-regenerating helper (no
        copyrighted SID data is committed); skip when it can't be built."""
        try:
            head, wide = grid_runner_dumps()
        except FixtureUnavailable as err:
            raise unittest.SkipTest(f"Grid Runner dump unavailable: {err}")
        cls.fixture_head = str(head)
        cls.fixture_wide = str(wide)

    def _assert_fired(self, fixture, flags, op, label):
        ops = _decode(_args(**flags), fixture)["op"].to_numpy()
        self.assertGreater(
            int((ops == op).sum()), 0, f"{label} produced no atoms in {fixture}"
        )

    def _assert_lossless(self, fixture, macros):
        raw = _per_frame_state(_decode(_args(), fixture))
        failures = []
        for name, flags in macros.items():
            st = _per_frame_state(_decode(_args(**flags), fixture))
            _, by_class = _divergence(raw, st)
            if by_class or len(st) != len(raw):
                failures.append(
                    f"{name}: frames raw={len(raw)} cfg={len(st)} "
                    f"divergent={by_class}"
                )
        self.assertFalse(
            failures,
            "per-frame register divergence vs raw:\n  " + "\n  ".join(failures),
        )

    def test_ctrl_collapse_lossless(self):
        self._assert_fired(
            self.fixture_head,
            dict(ctrl_bigram_pass=True),
            CTRL_BIGRAM_OP,
            "ctrl_bigram",
        )
        self._assert_fired(
            self.fixture_head,
            dict(ctrl_bigram_pass=True, ctrl_triple_pass=True),
            CTRL_TRIPLE_OP,
            "ctrl_triple",
        )
        self._assert_lossless(
            self.fixture_head,
            {
                "slope_pass": dict(slope_pass=True),
                "ctrl_bigram_pass": dict(ctrl_bigram_pass=True),
                "ctrl_triple_pass": dict(ctrl_bigram_pass=True, ctrl_triple_pass=True),
                "freq_nudge_catch_all": dict(
                    freq_nudge_pass=True, lonely_catch_all=True
                ),
            },
        )

    def test_freq_run_lossless(self):
        self._assert_fired(
            self.fixture_wide, dict(freq_run_pass=True), FREQ_RUN_OP, "freq_run"
        )
        self._assert_lossless(
            self.fixture_wide,
            {
                "freq_run_noslope": dict(freq_run_pass=True),
                "freq_run_slope": dict(slope_pass=True, freq_run_pass=True),
                "ctrl_triple_pass": dict(ctrl_bigram_pass=True, ctrl_triple_pass=True),
            },
        )


if __name__ == "__main__":
    unittest.main()
