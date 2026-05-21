"""Tier-A unit-coverage tests for ``preframr.regdataset``."""

import argparse
import logging
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from preframr_tokens.blocks import materialize_block_array, parser_worker


class TestMaterializeBlockArray(unittest.TestCase):
    def test_empty_raw_df_returns_zero_shape_array(self):
        tokenizer = mock.MagicMock()
        parser = mock.MagicMock()
        with mock.patch(
            "preframr_tokens.blocks.iter_voiced_blocks", return_value=iter([])
        ):
            out = materialize_block_array(
                tokenizer=tokenizer,
                raw_df=pd.DataFrame(),
                seq_len=512,
                parser=parser,
                reg_widths={},
            )
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (0, 513))
        self.assertEqual(out.dtype, np.int32)

    def test_short_block_pads_to_block_size(self):
        seq_len = 8
        block_size = seq_len + 1
        voiced = pd.DataFrame({"n": [1, 2, 3]})
        merged = pd.DataFrame({"n": [1, 2, 3]})

        tokenizer = mock.MagicMock()
        tokenizer.merge_token_df.return_value = merged
        tokenizer.encode.return_value = np.array([10, 20, 30], dtype=np.int32)
        parser = mock.MagicMock()

        with mock.patch(
            "preframr_tokens.blocks.iter_voiced_blocks", return_value=iter([voiced])
        ):
            out = materialize_block_array(
                tokenizer=tokenizer,
                raw_df=pd.DataFrame(),
                seq_len=seq_len,
                parser=parser,
                reg_widths={},
            )
        self.assertEqual(out.shape, (1, block_size))
        np.testing.assert_array_equal(out[0, :3], [10, 20, 30])
        np.testing.assert_array_equal(
            out[0, 3:], np.zeros(block_size - 3, dtype=np.int32)
        )

    def test_oversize_block_truncates_to_block_size(self):
        seq_len = 4
        block_size = seq_len + 1
        voiced = pd.DataFrame({"n": [1] * 20})
        merged = pd.DataFrame({"n": [1] * 20})

        tokenizer = mock.MagicMock()
        tokenizer.merge_token_df.return_value = merged
        tokenizer.encode.return_value = np.arange(20, dtype=np.int32)
        parser = mock.MagicMock()

        with mock.patch(
            "preframr_tokens.blocks.iter_voiced_blocks", return_value=iter([voiced])
        ):
            out = materialize_block_array(
                tokenizer=tokenizer,
                raw_df=pd.DataFrame(),
                seq_len=seq_len,
                parser=parser,
                reg_widths={},
            )
        self.assertEqual(out.shape, (1, block_size))
        np.testing.assert_array_equal(out[0], np.arange(block_size, dtype=np.int32))

    def test_merge_returning_none_raises(self):
        voiced = pd.DataFrame({"n": [1, 2, 3]})
        tokenizer = mock.MagicMock()
        tokenizer.merge_token_df.return_value = None
        parser = mock.MagicMock()
        with mock.patch(
            "preframr_tokens.blocks.iter_voiced_blocks", return_value=iter([voiced])
        ):
            with self.assertRaises(RuntimeError) as ctx:
                materialize_block_array(
                    tokenizer=tokenizer,
                    raw_df=pd.DataFrame(),
                    seq_len=8,
                    parser=parser,
                    reg_widths={},
                )
        self.assertIn("merge_token_df returned no 'n' column", str(ctx.exception))


class TestParserWorker(unittest.TestCase):
    def _stub_args(self, **kw):
        defaults = {
            "seq_len": 8,
            "block_stride": None,
            "require_pq": False,
        }
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_malformed_dump_logs_and_returns_empty(self):
        args = self._stub_args()
        logger = mock.MagicMock(spec=logging.Logger)

        def _raising_parse(*_a, **_kw):
            raise ValueError("malformed dump")
            yield

        with mock.patch("preframr_tokens.blocks.RegLogParser") as MockParser:
            MockParser.return_value.parse = _raising_parse
            dump_file, out = parser_worker(
                args, logger, "/fake/dump.parquet", max_perm=1
            )
        self.assertEqual(dump_file, "/fake/dump.parquet")
        self.assertEqual(out, [])
        logger.warning.assert_called_once()
        warn_args = logger.warning.call_args.args
        self.assertIn("dropping", warn_args[0])

    def test_assertion_error_also_swallowed(self):
        args = self._stub_args()
        logger = mock.MagicMock(spec=logging.Logger)

        def _raising_parse(*_a, **_kw):
            raise AssertionError("bad shape")
            yield

        with mock.patch("preframr_tokens.blocks.RegLogParser") as MockParser:
            MockParser.return_value.parse = _raising_parse
            _, out = parser_worker(args, logger, "/fake/x.parquet", max_perm=1)
        self.assertEqual(out, [])

    def test_happy_path_collects_dfs_and_blocks(self):
        args = self._stub_args()
        logger = mock.MagicMock(spec=logging.Logger)
        fake_df = pd.DataFrame({"reg": [0]})

        def _parse_gen(*_a, **_kw):
            yield fake_df

        with (
            mock.patch("preframr_tokens.blocks.RegLogParser") as MockParser,
            mock.patch(
                "preframr_tokens.blocks.iter_voiced_blocks",
                return_value=iter(["block1"]),
            ),
        ):
            MockParser.return_value.parse = _parse_gen
            dump_file, out = parser_worker(
                args, logger, "/fake/dump.parquet", max_perm=1
            )
        self.assertEqual(dump_file, "/fake/dump.parquet")
        self.assertEqual(len(out), 1)
        df_returned, blocks = out[0]
        self.assertTrue(df_returned.equals(fake_df))
        self.assertEqual(blocks, ["block1"])


if __name__ == "__main__":
    unittest.main()
