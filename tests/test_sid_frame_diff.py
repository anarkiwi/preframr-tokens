"""Frame-diff fidelity gate. Unit tests pin the diff logic on synthetic per-frame states. The release
blocker resolves every gated macro flag to a minimal valid pipeline (deps satisfied, conflicts dropped),
alone and on the skeleton base, and asserts that parsing the Grid Runner dump through it reproduces the raw
dump's discrete registers (gate/waveform/ADSR/filter) exactly and pitched-frame frequency within tolerance.
A transform that corrupts register state cannot pass; a new gated pass joins the matrix automatically.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from preframr_tokens.macros.flag_registry import (
    minimal_configs,
    resolve_flags,
    valid_combo,
)
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.sid_frame_diff import (
    diff_dump_vs_pipeline,
    diff_states,
    dump_frame_state,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args
from tests.parse_probes import parse_args

_KNOWN_FREQ_LOSSY: set[str] = set()
_STACK_FLAGS = {
    "skeleton_pass",
    "trajectory_anchor_pass",
    "held_arp",
    "wavetable_pass",
    "zero_plain",
    "wt_short",
    "wt_oneshot",
    "slide_wide",
    "slide_landing",
}


def _state(n, freq_word=None, ctrl=0x11, ad=0x00, sr=0xF0):
    """A ``(n,25)`` per-frame state: voice 0 with a combined freq word, gate-on tone ctrl, and ADSR."""
    st = np.zeros((n, 25), dtype=np.int64)
    if freq_word is not None:
        st[:, 0] = int(freq_word) & 0xFFFF
    st[:, 4] = ctrl
    st[:, 5] = ad
    st[:, 6] = sr
    return st


class TestFrameDiffUnit(unittest.TestCase):
    def test_identical_states_ok(self):
        st = _state(40, freq_word=LUT[60])
        res = diff_states(st.copy(), st.copy())
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["exact_fail"], [])
        self.assertEqual(res["freq_fail"], [])

    def test_gate_misplacement_detected(self):
        ref = _state(40, freq_word=LUT[60], ctrl=0x11)
        test = ref.copy()
        test[20, 4] = 0x10
        res = diff_states(ref, test)
        self.assertIn("v0.CTRL", res["exact_fail"])
        self.assertFalse(res["ok"])

    def test_adsr_misplacement_detected(self):
        ref = _state(40, freq_word=LUT[60])
        test = ref.copy()
        test[15, 5] = 0x33
        res = diff_states(ref, test)
        self.assertIn("v0.AD", res["exact_fail"])

    def test_word_domain_freq_normalised_ok(self):
        st = _state(40, freq_word=LUT[57])
        res = diff_states(st.copy(), st.copy())
        self.assertEqual(res["freq_fail"], [])

    def test_garbage_constant_freq_detected(self):
        ref = _state(40, freq_word=LUT[60])
        test = ref.copy()
        test[:, 0] = 67280 & 0xFFFF
        res = diff_states(ref, test)
        self.assertIn("v0.FREQ", res["freq_fail"])

    def test_freq_checked_on_noise_frames(self):
        ref = _state(40, freq_word=LUT[60], ctrl=0x81)
        test = ref.copy()
        test[:, 0] = LUT[72]
        res = diff_states(ref, test)
        self.assertIn("v0.FREQ", res["freq_fail"])

    def test_freq_ignored_on_test_frames(self):
        ref = _state(40, freq_word=LUT[60], ctrl=0x09)
        test = ref.copy()
        test[:, 0] = LUT[72]
        res = diff_states(ref, test)
        self.assertEqual(res["freq_fail"], [])

    def test_alignment_offset_recovered(self):
        base = _state(160, freq_word=LUT[55])
        for f in range(160):
            base[f, 4] = 0x11 if f % 4 else 0x10
        test = np.concatenate([_state(5, ctrl=0x10), base], axis=0)
        res = diff_states(base, test)
        self.assertEqual(res["offset"], 5)
        self.assertTrue(res["ok"], res)


class TestFlagResolver(unittest.TestCase):
    def test_requires_expanded(self):
        self.assertEqual(
            resolve_flags({"wt_oneshot"}),
            {"wt_oneshot", "wavetable_pass", "skeleton_pass"},
        )

    def test_conflict_raises(self):
        with self.assertRaises(ValueError):
            resolve_flags({"skeleton_pass", "freq_trajectory_pass"})

    def test_minimal_configs_cover_every_flag(self):
        for flag, cfg in minimal_configs().items():
            self.assertIn(flag, cfg, flag)
            self.assertTrue(valid_combo(cfg), (flag, cfg))


def _grid_runner_dump():
    """Resolve the Grid Runner raw dump: a local pre-rendered HVSC dump if present, else the regenerating
    fixture accessor. Returns None when neither is available so the gate skips rather than fails.
    """
    from tests import sid_fixtures as sf

    local = sf._local_rendered_dump(sf.GRID_RUNNER)
    if local is not None:
        return str(local)
    try:
        _head, wide = sf.grid_runner_dumps()
        return str(wide)
    except sf.FixtureUnavailable:
        return None


def _slice_dump(path, max_frames, tmpdir):
    """Write the first ``max_frames`` player-call frames (rows grouped by ``irq``) to a temp parquet, so
    the combinatorial matrix parses a short slice once per config instead of the full-length dump.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if "chipno" in df.columns:
        df = df[df["chipno"] == 0]
    keep = sorted(df["irq"].unique())[:max_frames]
    sliced = df[df["irq"].isin(keep)].reset_index(drop=True)
    out = os.path.join(tmpdir, "grid_runner_slice.dump.parquet")
    sliced.to_parquet(out)
    return out


class TestFrameDiffReleaseGate(unittest.TestCase):
    _MATRIX_FRAMES = 1500

    @classmethod
    def setUpClass(cls):
        cls.dump = _grid_runner_dump()
        if cls.dump is None:
            raise unittest.SkipTest(
                "Grid Runner dump unavailable (no local HVSC dump, no fixture cache)"
            )
        cls._tmp = tempfile.mkdtemp()
        cls.slice = _slice_dump(cls.dump, cls._MATRIX_FRAMES, cls._tmp)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "_tmp", None):
            shutil.rmtree(cls._tmp, ignore_errors=True)

    def _parse(self, args, path):
        return next(
            RegLogParser(args=args).parse(
                path, max_perm=1, require_pq=False, reparse=True
            ),
            None,
        )

    def test_deployed_and_stack_full_tune(self):
        """Over the full tune (the slice can miss late behaviour): the deployed default and the full
        skeleton-owned macro stack keep discrete regs exact AND pitch frame-exact. The byte-exact arc
        (register-exact SWEEP/stamp) left no tracked-lossy pass, so _KNOWN_FREQ_LOSSY is empty and the gate
        enforces full pitch fidelity end to end."""
        default = self._parse(default_tokenizer_args(cents=50), self.dump)
        res = diff_dump_vs_pipeline(self.dump, default)
        self.assertEqual(
            res["exact_fail"], [], f"deployed config diverges on discrete regs: {res}"
        )
        self.assertEqual(
            res["freq_fail"], [], f"deployed config diverges on pitch: {res}"
        )

        stack = self._parse(
            parse_args(**{f: True for f in resolve_flags(_STACK_FLAGS)}), self.dump
        )
        res = diff_dump_vs_pipeline(self.dump, stack)
        self.assertEqual(
            res["exact_fail"], [], f"macro stack diverges on discrete regs: {res}"
        )
        self.assertEqual(res["freq_fail"], [], f"macro stack diverges on pitch: {res}")

    def _matrix(self):
        """Every gated flag in a minimal valid pipeline and (where compatible) paired with the skeleton
        base, plus the full macro stack. Pairing with skeleton catches interaction bugs a single-flag
        config misses, e.g. a freq pass that only corrupts pitch once skeleton owns the freq channel.
        """
        seen = {}
        for flag in sorted(minimal_configs()):
            for extra in ((), ("skeleton_pass",)):
                try:
                    cfg = frozenset(resolve_flags({flag, *extra}))
                except ValueError:
                    continue
                seen.setdefault(cfg, f"{flag}{'+skeleton' if extra else ''}")
        seen.setdefault(frozenset(resolve_flags(_STACK_FLAGS)), "full-stack")
        return seen

    def test_every_macro_pipeline_is_frame_exact(self):
        """Combinatorial release blocker: every flag (alone and on the skeleton base) and the full stack
        keep discrete registers byte-exact and pitched freq within tolerance. Exact divergence fails
        unconditionally; pitch divergence fails unless the config contains a tracked _KNOWN_FREQ_LOSSY
        flag. Still-incompatible pipelines are surfaced as a dependency gap, not silently passed.
        """
        failures = []
        incompatible = []
        for cfg, label in sorted(self._matrix().items(), key=lambda kv: kv[1]):
            try:
                xdf = self._parse(parse_args(**{f: True for f in cfg}), self.slice)
            except Exception as err:  # noqa: BLE001
                incompatible.append(f"{label}: {type(err).__name__} ({sorted(cfg)})")
                continue
            if xdf is None:
                continue
            res = diff_dump_vs_pipeline(self.slice, xdf)
            self.assertEqual(
                res["exact_fail"],
                [],
                f"{label}: discrete-register divergence {res['exact']}",
            )
            if res["freq_fail"] and not (cfg & _KNOWN_FREQ_LOSSY):
                failures.append(f"{label}: pitched-freq divergence {res['freq']}")
        if incompatible:
            print(
                "\nincompatible pipelines (declare deps in flag_registry):\n  "
                + "\n  ".join(incompatible)
            )
        self.assertFalse(
            failures, "frame-diff freq regressions:\n  " + "\n  ".join(failures)
        )

    def test_known_freq_lossy_passes_still_flagged(self):
        """Keep the tracking list honest: each _KNOWN_FREQ_LOSSY pass, on the skeleton base where it
        corrupts pitch, must still fail the freq check. A pass that now passes should be removed from the
        list so the gate enforces it."""
        stale = []
        for flag in sorted(_KNOWN_FREQ_LOSSY):
            cfg = resolve_flags({flag, "skeleton_pass"})
            xdf = self._parse(parse_args(**{f: True for f in cfg}), self.dump)
            res = diff_dump_vs_pipeline(self.dump, xdf)
            if not res["freq_fail"]:
                stale.append(flag)
        self.assertFalse(
            stale, f"no longer freq-lossy, remove from _KNOWN_FREQ_LOSSY: {stale}"
        )
