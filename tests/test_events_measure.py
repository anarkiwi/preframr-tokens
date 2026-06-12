"""Guards for the collapse/bits measurement instrument: order-0 entropy, greedy corpus BPE train/apply,
and the summary metrics against the raw 2-byte-per-write floor."""

import math
import unittest

from preframr_tokens.events import measure


class TestOrder0Bits(unittest.TestCase):
    def test_empty_corpus_is_zero(self):
        self.assertEqual(measure.order0_bits([]), 0.0)
        self.assertEqual(measure.order0_bits([[]]), 0.0)

    def test_uniform_alphabet_is_log2(self):
        bits = measure.order0_bits([[0, 1, 2, 3]])
        self.assertAlmostEqual(bits, 8.0)
        single = measure.order0_bits([[7, 7, 7, 7]])
        self.assertAlmostEqual(single, 0.0)


class TestBpe(unittest.TestCase):
    def test_no_streams_and_no_repeats_learn_nothing(self):
        self.assertEqual(measure.bpe_train([], 5), [])
        self.assertEqual(measure.bpe_train([[]], 5), [])
        self.assertEqual(measure.bpe_train([[1, 2, 3]], 5), [])

    def test_repeated_pair_is_merged_and_applied(self):
        streams = [[1, 2, 1, 2, 1, 2], [1, 2, 9]]
        merges = measure.bpe_train(streams, 3)
        self.assertTrue(merges)
        first = merges[0]
        self.assertEqual(first[:2], (1, 2))
        applied_one = measure.bpe_apply([1, 2, 1, 2, 9], [first])
        self.assertEqual(applied_one, [first[2], first[2], 9])
        applied_full = measure.bpe_apply([1, 2, 1, 2, 9], merges)
        self.assertLess(len(applied_full), 5)

    def test_measure_reports_collapse_metrics(self):
        streams = [[1, 2, 1, 2, 1, 2, 3], [1, 2, 1, 2, 4]]
        out = measure.measure(streams, n_writes=6, merges=4)
        self.assertEqual(
            set(out),
            {
                "writes",
                "atomic_tokens",
                "bpe_tokens",
                "atomic_bits_per_write",
                "bpe_bits_per_write",
                "raw_bits_per_write",
                "collapse_vs_raw",
                "merges",
            },
        )
        self.assertEqual(out["writes"], 6)
        self.assertEqual(out["raw_bits_per_write"], 16.0)
        self.assertLessEqual(out["bpe_tokens"], out["atomic_tokens"])
        self.assertTrue(math.isfinite(out["collapse_vs_raw"]))


if __name__ == "__main__":
    unittest.main()
