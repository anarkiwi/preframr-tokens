"""Tests for the public ``combine_reg``: the canonical lo+hi byte coalescing the
parser uses and the freq audits reuse. A coordinated lo+hi update inside one
``clock // diffmax`` bucket must read as one settled 16-bit value, never a
half-updated pair."""

import unittest

import pandas as pd

from preframr_tokens.reglogparser import combine_reg


def _df(rows):
    return pd.DataFrame(
        [{"clock": c, "reg": r, "val": v} for c, r, v in rows], dtype="int64"
    )


class TestCombineReg(unittest.TestCase):
    def test_coordinated_lo_hi_in_bucket_is_one_settled_value(self):
        """lo=0x34 then hi=0x12 in one 512-clock bucket -> a single settled 0x1234."""
        out = combine_reg(_df([(0, 0, 0x34), (10, 1, 0x12)]), reg=0)
        v0 = out[out["reg"] == 0]
        self.assertEqual(len(v0), 1)
        self.assertEqual(int(v0["val"].iloc[0]), 0x1234)

    def test_byte_carries_across_buckets(self):
        """hi in bucket 0, lo three buckets later -> hi forward-filled into combine."""
        out = combine_reg(_df([(0, 1, 0x12), (512 * 3, 0, 0x34)]), reg=0).sort_values(
            "clock"
        )
        v0 = out[out["reg"] == 0]
        self.assertEqual(int(v0["val"].iloc[-1]), 0x1234)

    def test_last_write_wins_within_bucket(self):
        """Two lo writes in one bucket settle to the last value."""
        out = combine_reg(_df([(0, 0, 0x01), (5, 0, 0x02)]), reg=0)
        v0 = out[out["reg"] == 0]
        self.assertEqual(int(v0["val"].iloc[0]), 0x02)

    def test_non_target_rows_pass_through(self):
        """A non-target reg (ctrl) is untouched by the freq combine."""
        out = combine_reg(_df([(0, 0, 0x34), (0, 1, 0x12), (0, 4, 0x09)]), reg=0)
        self.assertEqual(int(out[out["reg"] == 4]["val"].iloc[0]), 0x09)

    def test_bits_masks_low_bits(self):
        """bits=4 zeroes the low nibble of the settled value (PW/filter quantise)."""
        out = combine_reg(_df([(0, 0, 0x3F), (0, 1, 0x00)]), reg=0, bits=4)
        self.assertEqual(int(out[out["reg"] == 0]["val"].iloc[0]), 0x30)


if __name__ == "__main__":
    unittest.main()
