"""CI-runnable full-parse coverage: a wholly synthetic raw register dump (no HVSC,
no Docker, no network) driven through the complete ``RegLogParser.parse`` pipeline so
the macro passes / walker / transform / state stack stay exercised in a clean CI
container where the corpus-backed fidelity tests skip."""

import logging
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    DUMP_SUFFIX,
    FRAME_REG,
    MAX_REG,
    VAL_PDTYPE,
)
from preframr_tokens.tokenizer_config import default_tokenizer_args, named_config

PHI = 985248
FRAME_CLOCKS = PHI // 50
MELODY = (0x1C83, 0x1E96, 0x2103, 0x2393, 0x2706, 0x2393, 0x2103, 0x1E96)
BASS = (0x0A41, 0x0B4A, 0x0C81, 0x0B4A)


def _init_writes():
    """Per-voice ADSR + waveform + global volume, all stamped at frame 0."""
    return [
        (24, 0x0F),
        (5, 0x09),
        (6, 0xF0),
        (2, 0x80),
        (3, 0x08),
        (12, 0x09),
        (13, 0xF0),
        (9, 0x80),
        (10, 0x08),
        (19, 0x09),
        (20, 0xF0),
        (16, 0x80),
        (17, 0x08),
    ]


def _synth_rows(frames=140):
    """A two-voice melodic dump: ``irq`` jumps one frame period per frame so
    ``_add_frame_reg`` recovers frame boundaries; voice 0 carries the lead, voice 1
    the bass, with periodic gate-off so note onsets are distinct."""
    rows = []
    clock = 0
    for reg, val in _init_writes():
        rows.append((clock, 0, 0, reg, val))
        clock += 2
    for frame in range(frames):
        clock = (frame + 1) * FRAME_CLOCKS
        irq = (frame + 1) * FRAME_CLOCKS
        lead = MELODY[frame % len(MELODY)]
        for reg, val in ((0, lead & 0xFF), (1, (lead >> 8) & 0xFF), (4, 0x41)):
            rows.append((clock, irq, 0, reg, val))
            clock += 2
        low = BASS[frame % len(BASS)]
        for reg, val in ((7, low & 0xFF), (8, (low >> 8) & 0xFF), (11, 0x41)):
            rows.append((clock, irq, 0, reg, val))
            clock += 2
        if frame % 8 == 0:
            rows.append((clock, irq, 0, 4, 0x40))
            clock += 2
            rows.append((clock, irq, 0, 11, 0x40))
            clock += 2
    return rows


def _write_dump(path, frames=140):
    df = pd.DataFrame(
        _synth_rows(frames), columns=["clock", "irq", "chipno", "reg", "val"]
    )
    df["clock"] = df["clock"].astype("UInt32")
    df["irq"] = df["irq"].astype("UInt32")
    df["chipno"] = df["chipno"].astype("UInt8")
    df["reg"] = df["reg"].astype("UInt8")
    df["val"] = df["val"].astype(VAL_PDTYPE)
    df.to_parquet(path, index=False)


def _parse_first(args, path):
    return next(
        RegLogParser(args=args, logger=logging.getLogger("synth")).parse(
            path, max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


class TestFullParseSynthetic(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._td.name, "synth" + DUMP_SUFFIX)
        _write_dump(self.path)

    def tearDown(self):
        self._td.cleanup()

    def test_baseline_yields_framed_tokens(self):
        df = _parse_first(default_tokenizer_args(min_song_tokens=64), self.path)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 64)
        self.assertIn("reg", df.columns)
        self.assertGreater(int((df["reg"] == FRAME_REG).sum()), 0)
        self.assertLessEqual(int(df["reg"].max()), max(MAX_REG, FRAME_REG))

    def test_full_macros_pipeline(self):
        df = _parse_first(named_config("full_macros", min_song_tokens=64), self.path)
        self.assertIsNotNone(df)
        self.assertGreater(int((df["reg"] == FRAME_REG).sum()), 0)

    def test_each_macro_flag_parses(self):
        for flag in (
            "loop_pass",
            "loop_transposed",
            "voice_lane",
            "coarsen_pass",
            "hard_restart_pass",
            "legato_pass_c2",
            "legato_pass_c4",
            "voice_canonical_block_order",
        ):
            with self.subTest(flag=flag):
                args = default_tokenizer_args(min_song_tokens=64, **{flag: True})
                df = _parse_first(args, self.path)
                self.assertIsNotNone(df, flag)
                self.assertGreater(len(df), 0, flag)

    def test_voice_rotation_augment(self):
        args = default_tokenizer_args(min_song_tokens=64)
        gen = RegLogParser(args=args, logger=logging.getLogger("synth")).parse(
            self.path, max_perm=3, require_pq=False, reparse=True
        )
        perms = list(gen)
        self.assertGreater(len(perms), 1)
        for df in perms:
            self.assertGreater(int((df["reg"] == FRAME_REG).sum()), 0)

    def test_admit_dump_rejects_oversized(self):
        args = default_tokenizer_args(min_song_tokens=64, max_raw_writes=8)
        df = _parse_first(args, self.path)
        self.assertIsNone(df)

    def test_short_dump_filtered(self):
        short = os.path.join(self._td.name, "short" + DUMP_SUFFIX)
        _write_dump(short, frames=2)
        df = _parse_first(default_tokenizer_args(min_song_tokens=256), short)
        self.assertIsNone(df)

    def test_expand_ops_decodes_full_pipeline(self):
        from preframr_tokens.macros.decode import expand_ops

        for cfg in (
            named_config("full_macros", min_song_tokens=64),
            default_tokenizer_args(min_song_tokens=64, loop_pass=True),
        ):
            df = _parse_first(cfg, self.path)
            self.assertIsNotNone(df)
            expanded = expand_ops(df.copy())
            self.assertGreater(len(expanded), len(df))
            self.assertIn("reg", expanded.columns)

    def test_registered_transform_round_trips(self):
        from preframr_tokens.macros.transform import (
            ensure_default_transforms_registered,
            get_transform_class,
        )

        ensure_default_transforms_registered()
        args = default_tokenizer_args(min_song_tokens=64, voice_lane=True)
        df = _parse_first(args, self.path)
        self.assertIsNotNone(df)
        transform = get_transform_class("voice_lane")()
        self.assertTrue(transform.round_trip_check(df, args=args))
        self.assertGreater(len(transform.forward(df, args=args)), len(df))

    def test_tokenizer_round_trip_on_parsed_df(self):
        df = _parse_first(default_tokenizer_args(min_song_tokens=64), self.path)
        self.assertIsNotNone(df)
        targs = SimpleNamespace(
            tkvocab=0, tokenizer="unigram", tkmodel=None, diffq=64, seq_len=512
        )
        tk = RegTokenizer(targs, tokens=None)
        tk.accumulate_tokens(df.copy(), "synth0")
        tokens = tk.make_tokens()
        self.assertGreater(len(tokens), 0)
        tk.tokens = tokens
        merged = tk.merge_tokens(tokens, [df.copy()])
        self.assertIsNotNone(merged)
        self.assertIn("n", merged[0].columns)
        ns = merged[0]["n"].dropna().astype(int).to_numpy()
        decoded = tk.decode(tk.encode(ns))
        self.assertTrue(np.array_equal(ns, decoded))
        self.assertGreater(len(tk.token_metadata()), 0)


if __name__ == "__main__":
    unittest.main()
