"""Real-pipeline structural + balance tests (#10): a synthetic raw dump driven through
the FULL ``RegLogParser.parse`` + block path (NOT a hand-built ``Pass.apply`` df, which
skips ``_combine_regs`` / ``_quantize_freq_to_cents`` and shipped the cent-index no-op
false green). Asserts the per-config op matrix, the skeleton round-trip to the content
floor, and the op55:op54 channel-balance bound. The synthetic core always runs."""

import os
import tempfile
import unittest

import numpy as np

from tests.parse_probes import (
    DumpBuilder,
    block_op_counts,
    cents_to_fn,
    parse_args,
    write_dump,
)
from tests.sid_fixtures import (
    FixtureUnavailable,
    cache_dir,
    ensure_driver_fixture,
)

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_OP,
    ORN_OP,
    SKEL_OP,
)

BALANCE_MAX = 6
_FREQ_REGS = set(int(r) for r in FREQ_REGS_BY_VOICE)
_LUT_GRID = set(LUT)


def _off_lut_grid_freq_cells(dump):
    """Decode a skeleton-parsed dump to per-frame register state and count freq-reg cells
    whose value is neither silent nor an exact LUT semitone -- 0 means the SKEL+ORN
    channel reconstructed every freq onto the content floor. Returns (off_grid, total).
    """
    parser = RegLogParser(args=_skeleton_args())
    parsed = next(parser.parse(dump, max_perm=1, require_pq=False, reparse=True), None)
    assert parsed is not None
    state = register_state(parsed)
    off_grid = total = 0
    for reg in _FREQ_REGS:
        for value in state[:, reg]:
            value = int(value)
            total += 1
            if value != 0 and value not in _LUT_GRID:
                off_grid += 1
    return off_grid, total


def _synthetic_dump(builder=None):
    """A deterministic voice-0 dump exercising every pitch driver mechanism through
    separate lo+hi + per-frame freq writes: held melody, an octave arp, a vibrato, a
    slide, plus ctrl/ADSR/PW -- so the parser's combine + cent-quantize stages run."""
    b = builder or DumpBuilder()
    b.adsr(ad=0x00, sr=0xF0).pw(0x800)
    for note in (60, 62, 64, 65, 67, 64, 60):
        b.note([LUT[note]] * 6)
    b.note([LUT[60 + (0 if f % 2 == 0 else 12)] for f in range(8)])
    b.note([cents_to_fn(67, 20.0 * np.sin(f)) for f in range(8)])
    b.note([LUT[60 + min(7, f)] for f in range(8)])
    b.pw(0x400)
    for note in (62, 65, 69):
        b.note([LUT[note]] * 5)
    return b


def _skeleton_args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


class TestParsePipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Write the synthetic dump once into a temp dir reused across the config
        matrix (each parse is independent / reparse=True)."""
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dump = write_dump(
            _synthetic_dump(), os.path.join(cls._tmp.name, "synthetic.dump.parquet")
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_skeleton_config_owns_freq_channel(self):
        """skeleton on => SKEL(op54)>0 and ORN(op55)>0, and the freq-trajectory
        channel is silent (op45==0). This is the config that would have caught the
        cent-index no-op: a SkeletonPass reading the cent-index ``val`` rather than
        the 16-bit ``freq_unq`` emits no op54/op55 here."""
        counts = block_op_counts(self.dump, _skeleton_args())
        self.assertGreater(counts[SKEL_OP], 0, counts)
        self.assertGreater(counts[ORN_OP], 0, counts)
        self.assertEqual(counts[FREQ_TRAJ_OP], 0, counts)

    def test_default_freq_trajectory_fires(self):
        """The default freq encoder (freq_trajectory) emits op45 on real per-frame
        freq motion (would be 0 if the trajectory pass were a no-op on parsed data)."""
        counts = block_op_counts(
            self.dump,
            parse_args(freq_trajectory_pass=True, trajectory_anchor_pass=True),
        )
        self.assertGreater(counts[FREQ_TRAJ_OP], 0, counts)

    def test_skeleton_roundtrip_to_content_floor(self):
        """The skeleton-encoded stream decodes (via the public expand_ops path) to a
        per-frame freq state that lies EXACTLY on the LUT semitone grid -- the content
        floor the SKEL+ORN channel reconstructs to, with no residual off-grid freq on
        this clean synthetic dump (the ORN replay is verified against the floor)."""
        off_grid, _total = _off_lut_grid_freq_cells(self.dump)
        self.assertEqual(off_grid, 0)

    def test_skeleton_channel_balance(self):
        """Channel-balance bound: op55(ORN):op54(SKEL) <= BALANCE_MAX. Catches the
        channel-drowning that shipped a 13:1 ORN:SKEL ratio (the ornament channel
        swamping the melody skeleton)."""
        counts = block_op_counts(self.dump, _skeleton_args())
        skel = max(counts[SKEL_OP], 1)
        ratio = counts[ORN_OP] / skel
        self.assertLessEqual(
            ratio, BALANCE_MAX, f"ORN:SKEL {counts[ORN_OP]}:{counts[SKEL_OP]} = {ratio}"
        )


class TestRealTuneRoundtrip(unittest.TestCase):
    """Cross-check against real driver output: the skeleton-decoded freq state of a real
    tune (Commando, Hubbard) lies on the LUT semitone grid for all-but-a-tiny-residual of
    cells (the residual is the known RESID-clamp gap, tracked separately). Regenerate-or-
    fail on the fixture cache (never skip); only runs where the cache is present."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_cache_present():
            raise unittest.SkipTest(_NO_CACHE_MSG)
        try:
            cls.dump = str(ensure_driver_fixture("commando"))
        except FixtureUnavailable as err:
            raise AssertionError(
                f"fixture cache present but Commando dump unavailable: {err}"
            ) from err

    def test_commando_roundtrip_on_lut_grid(self):
        off_grid, total = _off_lut_grid_freq_cells(self.dump)
        self.assertGreater(total, 0)
        self.assertLessEqual(off_grid / total, 0.01, (off_grid, total))


_NO_CACHE_MSG = (
    "PREFRAMR_SID_FIXTURE_CACHE unset and no local HVSC tree; real-tune layer runs "
    "only where the fixture cache/headlessvice is available"
)


def _fixture_cache_present():
    """True when this env should run the real-tune layer: an explicit fixture cache
    dir, or the local HVSC dump tree the driver fixtures resolve from."""
    if os.environ.get("PREFRAMR_SID_FIXTURE_CACHE"):
        return True
    return (cache_dir() / "hvsc").exists() or os.path.isdir("/scratch/preframr/hvsc")


if __name__ == "__main__":
    unittest.main()
