"""Real-data fidelity for the motif transform: on parsed SID fixtures, a mined
dictionary's forward then inverse is byte-exact (op/reg/subreg/val/diff), so the
pass is lossless. Skips when the headlessvice fixture image is unavailable."""

import unittest

from tests.sid_fixtures import FixtureUnavailable, grid_runner_dumps
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import default_tokenizer_args
from preframr_tokens.macros.motif_pass import MotifTransform, mine_motifs, _atoms_of

_COLS = ["op", "reg", "subreg", "val", "diff"]


class TestMotifFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.dumps = grid_runner_dumps()
        except FixtureUnavailable as exc:
            raise unittest.SkipTest(str(exc))

    def test_forward_inverse_byte_exact(self):
        for fixture in self.dumps:
            args = default_tokenizer_args()
            parser = RegLogParser(args=args)
            df = next(
                parser.parse(str(fixture), max_perm=1, require_pq=False, reparse=True)
            )
            motif_dict = mine_motifs(
                [_atoms_of(df)], ["fx"], k=256, min_count=3, min_composers=1
            )
            self.assertGreater(len(motif_dict), 0)
            args.motif_pass = True
            args.motif_dict = motif_dict
            transform = MotifTransform()
            encoded = transform.forward(df, args=args)
            self.assertLess(len(encoded), len(df))
            decoded = transform.inverse(encoded, args=args)
            self.assertTrue(
                df[_COLS]
                .reset_index(drop=True)
                .equals(decoded[_COLS].reset_index(drop=True)),
                f"motif round-trip not byte-exact on {fixture}",
            )


if __name__ == "__main__":
    unittest.main()
