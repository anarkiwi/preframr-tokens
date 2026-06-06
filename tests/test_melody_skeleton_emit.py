"""Melody-skeleton layer 2 emit through the real parse: with ``melody_skeleton`` on, a melodic voice's
HOLD/ACCUM freq onsets are re-keyed to MELODY_INTERVAL atoms and the parse stays byte-exact (zero raw
SET on every generator channel under the validate=True arbiter), while a swept/non-melodic voice passes
through as raw generator atoms. The default path (flag off) is unchanged -- covered by the residual-zero
suite."""

import glob
import os
import tempfile
import unittest

from tests.parse_probes import DumpBuilder, write_dump, cents_to_fn
from tests.test_generator_residual_zero import _multi_feature_dump, _raw_gen_sets
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    MELODY_INTERVAL_OP,
    MELODY_INTERVAL_SUBREG_FIRST,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args

_HVSC = "/scratch/preframr/hvsc"


def _parse(path, **over):
    args = default_tokenizer_args(generator_pass=True, instrument_program=True, **over)
    return next(
        RegLogParser(args=args).parse(path, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def _interval_atoms(df):
    return int((df["op"].to_numpy() == MELODY_INTERVAL_OP).sum()) // 9


def _slide_to_grid_dump(path):
    """A voice-0 melodic line: each note a 3-frame micro-slide landing on a grid pitch (an ACCUM onset
    on a stable note grid), so the segmenter onsets per note and the onset re-keys to an interval.
    """
    b = DumpBuilder().adsr().pw(0x800).modevol(0x1F).resfilt(0x00)
    for n in [60, 64, 67, 72, 67, 64, 60, 62, 65, 69, 65, 62]:
        g = cents_to_fn(n, 0)
        b.note([g - 3, g - 2, g - 1, g, g, g, g, g])
    return write_dump(b, path)


class TestMelodySkeletonEmit(unittest.TestCase):
    def test_melodic_onsets_rekeyed_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _slide_to_grid_dump(os.path.join(tmp, "mel.dump.parquet"))
            mel = _parse(path, melody_skeleton=True)
        self.assertIsNotNone(mel)
        self.assertGreater(_interval_atoms(mel), 0, "no melody onsets re-keyed")
        self.assertFalse(
            _raw_gen_sets(mel), "melody_skeleton left a raw generator-channel SET"
        )

    def test_first_atom_is_first_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _slide_to_grid_dump(os.path.join(tmp, "mel.dump.parquet"))
            mel = _parse(path, melody_skeleton=True)
        ops = mel["op"].to_numpy()
        subs = mel["subreg"].to_numpy()
        vals = mel["val"].to_numpy()
        firsts = [
            int(vals[i])
            for i in range(len(mel))
            if int(ops[i]) == MELODY_INTERVAL_OP
            and int(subs[i]) == MELODY_INTERVAL_SUBREG_FIRST
        ]
        self.assertTrue(firsts)
        self.assertEqual(firsts[0], 1, "first melody onset must be FIRST=1 (absolute)")
        self.assertTrue(
            all(f == 0 for f in firsts[1:]), "later onsets must be intervals"
        )

    def test_non_melodic_voice_passes_through_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _multi_feature_dump(os.path.join(tmp, "mf.dump.parquet"))
            mel = _parse(path, melody_skeleton=True)
        self.assertIsNotNone(mel)
        self.assertFalse(
            _raw_gen_sets(mel), "non-melodic passthrough left a raw generator SET"
        )


class TestMelodySkeletonCorpus(unittest.TestCase):
    def test_corpus_sample_byte_exact_and_emits(self):
        paths = sorted(
            glob.glob(os.path.join(_HVSC, "**", "*.dump.parquet"), recursive=True)
        )
        if not paths:
            self.skipTest("HVSC corpus unavailable")
        sample = paths[:: max(1, len(paths) // 50)][:8]
        checked = 0
        total = 0
        for path in sample:
            mel = _parse(path, melody_skeleton=True)
            if mel is None:
                continue
            checked += 1
            self.assertFalse(
                _raw_gen_sets(mel), f"residual raw SET under melody_skeleton: {path}"
            )
            total += _interval_atoms(mel)
        if checked == 0:
            self.skipTest("no non-digi corpus tunes parsed")
        self.assertGreater(total, 0, "no interval atoms across the corpus sample")


if __name__ == "__main__":
    unittest.main()
