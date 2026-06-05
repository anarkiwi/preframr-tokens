"""Strict register-LEVEL order+timing fidelity gate (no audio): ``register_state`` is
order/timing-blind, so it misses a pass that reorders within-voice writes or emits a
frame-scale ``diff`` (both corrupt audio — see preframr-audio test_register_canonicalization).
This asserts the decoded SID-visible stream keeps the raw dump's CTRL/AD/SR write order +
values per frame/voice, and that every intra-frame write carries the nominal ``_MIN_DIFF``.
"""

import collections
import unittest

import pandas as pd

from tests.sid_fixtures import FixtureUnavailable, grid_runner_dumps

from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.reglogparser import RegLogParser, remove_voice_reg
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, _MIN_DIFF
from preframr_tokens.tokenizer_config import REGISTERED_MACROS, default_tokenizer_args

CTRL = {4, 11, 18}
AD = {5, 12, 19}
SR = {6, 13, 20}
EXACT = CTRL | AD | SR


def _voice(reg):
    return reg // 7 if 0 <= reg <= 20 else 3


def _dump_frames(path):
    """Raw dump -> per-frame {voice: [(reg, val), ...]} for EXACT regs, clock order."""
    df = pd.read_parquet(path)
    df = df[df["chipno"] == 0].sort_values(["irq", "clock"])
    frames = []
    for _irq, g in df.groupby("irq", sort=True):
        cur = collections.defaultdict(list)
        for r, v in zip(g["reg"].to_numpy(), g["val"].to_numpy()):
            r = int(r)
            if r in EXACT:
                cur[_voice(r)].append((r, int(v) & 0xFF))
        frames.append(dict(cur))
    return frames


def _decoded_frames(path, flags):
    """Parse+decode -> per-frame {voice: [(reg, val)]} (EXACT regs) + intra-frame diffs."""
    parser = RegLogParser(args=default_tokenizer_args(**flags))
    xdf = next(parser.parse(path, max_perm=1, require_pq=False, reparse=True))
    df, _ = remove_voice_reg(xdf.copy(), {})
    df = expand_ops(df.copy())
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    diffs = df["diff"].to_numpy() if "diff" in df.columns else [None] * len(df)
    frames, cur, bad_diffs = [], collections.defaultdict(list), 0
    for r, v, d in zip(regs, vals, diffs):
        r = int(r)
        if r in (FRAME_REG, DELAY_REG):
            frames.append(dict(cur))
            cur = collections.defaultdict(list)
            continue
        if d is not None and int(d) != _MIN_DIFF:
            bad_diffs += 1
        if r in EXACT:
            cur[_voice(r)].append((r, int(v) & 0xFF))
    frames.append(dict(cur))
    return frames, bad_diffs


def _best_offset(ref, test, span=24, window=400, skip=8):
    best_k, best_bad = 0, None
    for k in range(-span, span + 1):
        bad = m = 0
        for i in range(skip, skip + window):
            if 0 <= i < len(ref) and 0 <= i + k < len(test):
                m += 1
                if ref[i] != test[i + k]:
                    bad += 1
        if m > 50 and (best_bad is None or bad < best_bad):
            best_bad, best_k = bad, k
    return best_k


def _order_mismatches(ref, test, skip=8):
    k = _best_offset(ref, test)
    bad, examples = 0, []
    for i in range(skip, len(ref)):
        ti = i + k
        if not 0 <= ti < len(test):
            continue
        if ref[i] != test[ti]:
            bad += 1
            if len(examples) < 4:
                examples.append((i, ref[i], test[ti]))
    return bad, examples


class TestRegisterOrderFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _head, wide = grid_runner_dumps()
        except FixtureUnavailable as err:
            raise unittest.SkipTest(f"Grid Runner dump unavailable: {err}")
        cls.dump = str(wide)
        cls.ref = _dump_frames(cls.dump)

    def _parse(self, flags):
        parser = RegLogParser(args=default_tokenizer_args(**flags))
        return next(parser.parse(self.dump, max_perm=1, require_pq=False, reparse=True))

    def test_stamp_byte_exact_register_state_vs_baseline(self):
        """stamp_pass must be byte-exact lossless: per-frame decoded register_state with stamp must
        equal the no-macro baseline (freq included). Guards the PerRegBurst rebasing bug where a freq
        DIFF after a consumed STAMP_REF mis-based against the pre-drum SET (Grid Runner v1 46 vs 86).
        """
        from preframr_tokens.audit_primitives import register_state
        from preframr_tokens.stfconstants import STAMP_REF_OP, STAMP_REL_REF_OP

        sx = self._parse(dict(stamp_pass=True))
        refs = int(
            (sx["op"] == STAMP_REF_OP).sum() + (sx["op"] == STAMP_REL_REF_OP).sum()
        )
        self.assertGreater(
            refs, 0, "stamp_pass did not fire (fallback?) -- nothing validated"
        )
        base, stamp = register_state(self._parse({})), register_state(sx)
        n = min(len(base), len(stamp))
        cells = int((base[:n] != stamp[:n]).sum())
        self.assertEqual(
            cells, 0, f"{cells} register_state cells diverge stamp vs baseline"
        )

    def test_parse_audit_silent_through_post_rotation_loops(self):
        """The byte-exact pipeline must stay lossless END TO END, including the POST-rotation passes
        where LoopPass (in run_post_norm_pre_voice_passes) mints PATTERN_REPLAY/DO_LOOP refs. Parse a
        real loop-heavy tune on the skeleton path with the audit in raise mode: any post-rotation pass
        that breaks per-frame register_state (expand_loops round-trip), the loop-expanded frame budget,
        or loop-aware back-ref integrity (validate_stream) raises here.
        """
        from preframr_tokens.stfconstants import (
            DO_LOOP_OP,
            PATTERN_REPLAY_OP,
            PATTERN_REPLAY_SUBREG_DIST_HI,
        )

        parser = RegLogParser(
            args=default_tokenizer_args(
                skeleton_pass=True,
                loop_pass=True,
                loop_transposed=True,
                parse_audit="raise",
            )
        )
        xdf = next(parser.parse(self.dump, max_perm=1, require_pq=False, reparse=True))
        pr_heads = (xdf["op"] == PATTERN_REPLAY_OP) & (
            xdf["subreg"] == PATTERN_REPLAY_SUBREG_DIST_HI
        )
        loops = int(pr_heads.sum() + (xdf["op"] == DO_LOOP_OP).sum())
        self.assertGreater(
            loops, 0, "tune produced no loops -- post-rotation refs unexercised"
        )

    def test_decoded_ctrl_adsr_order_matches_dump(self):
        """Per frame per voice, the decoded CTRL/AD/SR write sequence (order + value) must
        equal the raw dump's, for the no-macro baseline AND the non-replay macro stack.
        Reg-sorting a voice's writes (the old `_norm_pr_order`) fails this on interleaved
        ADSR/CTRL frames. ``instrument_program`` is excluded -- a per-frame replay re-asserts
        unchanged values (audio-equivalence pinned by register_state in the sibling test).
        """
        configs = {
            "baseline": {},
            "non_replay_macros": {
                f: True for f in REGISTERED_MACROS if f != "instrument_program"
            },
        }
        for name, flags in configs.items():
            test_frames, _ = _decoded_frames(self.dump, flags)
            bad, examples = _order_mismatches(self.ref, test_frames)
            self.assertEqual(
                bad,
                0,
                f"{name}: {bad} frames with CTRL/AD/SR order/value "
                f"divergence vs dump; e.g. {examples}",
            )

    def test_instrument_program_register_state_equivalent(self):
        """``instrument_program`` (now in the production default) replays a per-frame
        ctrl/AD/SR program, re-asserting unchanged values -- so its SID-visible write stream
        differs from the raw dump, but every per-frame register_state cell is unchanged vs
        the same stack without it: identical end-of-frame state, hence identical SID audio.
        The byte-exact invariant the raw-dump order check cannot express for a replay pass.
        """
        from preframr_tokens.audit_primitives import register_state

        non_replay = {f: True for f in REGISTERED_MACROS if f != "instrument_program"}
        full = {f: True for f in REGISTERED_MACROS}
        base = register_state(self._parse(non_replay))
        full_state = register_state(self._parse(full))
        self.assertEqual(len(base), len(full_state), "frame count diverged")
        n = min(len(base), len(full_state))
        cells = int((base[:n] != full_state[:n]).sum())
        self.assertEqual(
            cells,
            0,
            f"{cells} register_state cells diverge full_macros vs non-replay stack",
        )

    def test_intra_frame_writes_carry_nominal_diff(self):
        """Every intra-frame decoded write must carry the nominal `_MIN_DIFF`; a
        frame-scale `diff` (the `diff=irq` class) drives the FRAME budget
        negative and drops samples. Guards that fix across the full stack incl.
        sweep/stamp."""
        flags = dict(
            {f: True for f in REGISTERED_MACROS},
            skeleton_pass=True,
            wavetable_pass=True,
            held_arp=True,
            zero_plain=True,
            wt_short=True,
            wt_oneshot=True,
            slide_wide=True,
            stamp_pass=True,
            sweep_pass=True,
            sweep_loop=True,
            slide_landing=True,
        )
        _frames, bad_diffs = _decoded_frames(self.dump, flags)
        self.assertEqual(
            bad_diffs,
            0,
            f"{bad_diffs} intra-frame writes carry a non-nominal diff "
            f"(want {_MIN_DIFF}); a frame-scale diff breaks the render.",
        )


class TestNormPrOrderPreservesInputOrder(unittest.TestCase):
    def test_interleaved_voice_writes_keep_input_order(self):
        """`_norm_pr_order` groups by voice but must PRESERVE each voice's input write
        order -- never sort by register. Feed a voice-0 frame written SR, CTRL, AD, SR
        (interleaved, as ~17% of single-speed tunes are) and assert the output keeps
        that order, not the reg-ascending CTRL, AD, SR, SR."""
        from preframr_tokens.stfconstants import SET_OP

        rows = [
            {"reg": FRAME_REG, "val": 0, "op": int(SET_OP), "diff": _MIN_DIFF},
            {"reg": 6, "val": 111, "op": int(SET_OP), "diff": _MIN_DIFF},
            {"reg": 4, "val": 0x81, "op": int(SET_OP), "diff": _MIN_DIFF},
            {"reg": 5, "val": 4, "op": int(SET_OP), "diff": _MIN_DIFF},
            {"reg": 6, "val": 104, "op": int(SET_OP), "diff": _MIN_DIFF},
        ]
        df = pd.DataFrame(rows)
        parser = RegLogParser(args=default_tokenizer_args())
        out = parser._norm_pr_order(df)  # pylint: disable=protected-access
        seq = [(int(r), int(v)) for r, v in zip(out["reg"], out["val"]) if int(r) >= 0]
        self.assertEqual(seq, [(6, 111), (4, 0x81), (5, 4), (6, 104)])


if __name__ == "__main__":
    unittest.main()
