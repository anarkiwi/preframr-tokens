"""Guards for the merge-boundary audit tool: the audit frame has the documented columns/summary keys,
and a boundary-isolated unigram reports zero voice/KEYFRAME-crossing merges (the A/B control).
"""

import os
import tempfile
import types
import unittest

import numpy as np
import pandas as pd

from preframr_tokens import bpe_audit
from preframr_tokens.events import dataset
from preframr_tokens.macros import pitch_grid
from preframr_tokens.stfconstants import DUMP_SUFFIX


def _multi_voice_ids(k):
    rng = np.random.default_rng(k)
    writes = []
    for f in range(48):
        for v in range(3):
            base = 7 * v
            nt = 40 + v * 5 + int(rng.integers(0, 12))
            fr = pitch_grid.note_freq_at(nt, 0.0)
            writes += [
                (f, base, fr & 0xFF),
                (f, base + 1, (fr >> 8) & 0xFF),
                (f, base + 4, 0x41 if (f + v) % 6 else 0x40),
                (f, base + 5, 0x08),
                (f, base + 6, 0xA9),
            ]
    writes.sort(key=lambda t: t[0])
    df = pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )
    return dataset.dump_token_ids(df)


def _train(isolate):
    streams = [_multi_voice_ids(k) for k in range(20)]
    with tempfile.TemporaryDirectory() as tmpdir:
        args = types.SimpleNamespace(
            tokenizer="unigram", tkvocab=160, tkmodel=os.path.join(tmpdir, "tk.json")
        )
        tok = dataset.make_tokenizer(args)
        if isolate:
            tok.isolation_ns = dataset.BOUNDARY_ISOLATION_NS
        dfs = [
            (
                os.path.join(tmpdir, f"s{i}{DUMP_SUFFIX}"),
                pd.DataFrame({"n": np.asarray(s, dtype=np.int64)}),
                0,
            )
            for i, s in enumerate(streams)
        ]
        tok.train_tokenizer(dfs)
        return tok


class TestBpeAudit(unittest.TestCase):
    def test_audit_vocab_columns_and_summary_keys(self):
        tok = _train(isolate=False)
        frame = bpe_audit.audit_vocab(tok)
        self.assertEqual(
            list(frame.columns),
            ["piece_id", "n_atoms", "crosses_voice", "n_kinds", "all_digits"],
        )
        self.assertGreater(len(frame), 0)
        summary = bpe_audit.summarize(frame)
        self.assertEqual(
            set(summary),
            {
                "n_pieces",
                "n_multi_atom",
                "n_crossing_voice",
                "frac_crossing_voice",
                "n_multi_kind",
            },
        )

    def test_isolation_on_reports_no_voice_crossing(self):
        tok = _train(isolate=True)
        summary = bpe_audit.summarize(bpe_audit.audit_vocab(tok))
        self.assertEqual(summary["n_crossing_voice"], 0)


if __name__ == "__main__":
    unittest.main()
