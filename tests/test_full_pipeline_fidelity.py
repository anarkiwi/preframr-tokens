"""Full-pipeline per-frame register fidelity gate: parse a fixture through the
whole pipeline under each macro and assert decoded per-frame register state
matches the no-macro baseline. Single-pass round-trip tests compare value
sequences and miss frame-placement bugs (right values, wrong frames) that break
pitch/waveform; divergence is reported per register-class to localise them."""

import os
import unittest

from tests.sid_fixtures import FixtureUnavailable, cache_dir, grid_runner_dumps

from preframr_tokens.audit_primitives import register_state as _per_frame_state
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    PWM_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
)
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_TRIPLE_OP,
    FREQ_TRAJ_OP,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args as _args

_CLASS = {}
for _v in range(3):
    _CLASS[FREQ_REGS_BY_VOICE[_v]] = f"v{_v}.FREQ"
    _CLASS[PWM_REGS_BY_VOICE[_v]] = f"v{_v}.PW"
    _CLASS[CTRL_REGS_BY_VOICE[_v]] = f"v{_v}.CTRL"
    _CLASS[AD_REGS_BY_VOICE[_v]] = f"v{_v}.AD"
    _CLASS[SR_REGS_BY_VOICE[_v]] = f"v{_v}.SR"


def _decode(args, fixture):
    parser = RegLogParser(args=args)
    xdf = next(parser.parse(fixture, max_perm=1, require_pq=False, reparse=True), None)
    assert xdf is not None and len(xdf), "fixture produced no rows"
    return xdf


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
        """Source the Grid Runner dump via the HVSC-regenerating helper (no copyrighted
        SID data is committed). Where a fixture cache is present this env should run the
        real-tune layer, so a failed regeneration FAILS rather than silently skips;
        elsewhere it skips."""
        cache_present = (
            bool(os.environ.get("PREFRAMR_SID_FIXTURE_CACHE"))
            or (cache_dir() / "hvsc").exists()
        )
        try:
            head, wide = grid_runner_dumps()
        except FixtureUnavailable as err:
            if cache_present:
                raise AssertionError(
                    f"fixture cache present but Grid Runner dump unavailable: {err}"
                ) from err
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
                "freq_trajectory": dict(freq_trajectory_pass=True),
                "ctrl_bigram_pass": dict(ctrl_bigram_pass=True),
                "ctrl_triple_pass": dict(ctrl_bigram_pass=True, ctrl_triple_pass=True),
                "freq_nudge_catch_all": dict(
                    freq_nudge_pass=True, lonely_catch_all=True
                ),
            },
        )

    def test_freq_trajectory_lossless(self):
        self._assert_fired(
            self.fixture_wide,
            dict(freq_trajectory_pass=True),
            FREQ_TRAJ_OP,
            "freq_traj",
        )
        self._assert_lossless(
            self.fixture_wide,
            {
                "freq_trajectory": dict(freq_trajectory_pass=True),
                "freq_traj_nudge": dict(
                    freq_trajectory_pass=True,
                    freq_nudge_pass=True,
                    lonely_catch_all=True,
                ),
                "ctrl_triple_pass": dict(ctrl_bigram_pass=True, ctrl_triple_pass=True),
            },
        )


if __name__ == "__main__":
    unittest.main()
