"""Real-pipeline structural test (#10): a synthetic raw dump driven through the FULL
``RegLogParser.parse`` + block path (NOT a hand-built ``Pass.apply`` df, which skips
``_combine_regs`` / ``_quantize_freq_to_cents`` and shipped the cent-index no-op false
green). Asserts the default freq encoder fires on real per-frame freq motion."""

import unittest

import numpy as np

from tests.parse_probes import (
    DumpBuilder,
    block_op_counts,
    cents_to_fn,
    parse_args,
    write_dump,
)

from preframr_tokens.macros.freq_lut import LUT
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import FREQ_TRAJ_OP

_FREQ_REGS = set(int(r) for r in FREQ_REGS_BY_VOICE)
_LUT_GRID = set(LUT)


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


class TestParsePipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Write the synthetic dump once into a temp dir reused across the config
        matrix (each parse is independent / reparse=True)."""
        import os
        import tempfile

        cls._tmp = tempfile.TemporaryDirectory()
        cls.dump = write_dump(
            _synthetic_dump(), os.path.join(cls._tmp.name, "synthetic.dump.parquet")
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_default_freq_trajectory_fires(self):
        """The default freq encoder (freq_trajectory) emits op45 on real per-frame
        freq motion (would be 0 if the trajectory pass were a no-op on parsed data)."""
        counts = block_op_counts(
            self.dump,
            parse_args(freq_trajectory_pass=True, trajectory_anchor_pass=True),
        )
        self.assertGreater(counts[FREQ_TRAJ_OP], 0, counts)


if __name__ == "__main__":
    unittest.main()
