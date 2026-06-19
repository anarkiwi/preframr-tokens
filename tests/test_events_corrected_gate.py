"""The corrected-codec fidelity gate: byte-exact to the audio-faithful TARGET =
{settled freq/pw/fc/res/vol grid + the ORDERED ctrl/AD/SR write stream}. This is NOT
byte-exact-to-settled-state (that drops audibly-significant intra-frame env order);
``decode(encode(ow))`` must reproduce both halves exactly. Synthetic cases run in CI;
the real-corpus sweep (>=2000 tunes) runs where the HVSC dumps are present."""

import os

import numpy as np
import pandas as pd
import pytest

from preframr_tokens.events import stream
from preframr_tokens.events.oracle import corrected_writes, env_writes, ordered_writes

_GT = "/scratch/tmp/sidemu/music_groundtruth.parquet"
_PAL_CPF = 19656.0
_NTSC_CPF = 17095.0


def _df(writes):
    writes = sorted(writes, key=lambda t: t[0])
    return pd.DataFrame(
        {
            "clock": np.arange(len(writes), dtype=np.int64),
            "irq": np.array([w[0] for w in writes], dtype=np.int64),
            "chipno": np.zeros(len(writes), dtype=np.int64),
            "reg": np.array([w[1] for w in writes], dtype=np.int64),
            "val": np.array([w[2] for w in writes], dtype=np.int64),
        }
    )


def test_intra_frame_gate_toggle_is_byte_exact():
    """A gate-off then gate-on in ONE frame: settling keeps only the last (0x41), the
    corrected codec keeps BOTH ordered writes."""
    ow = ordered_writes(
        _df(
            [
                (0, 0, 0x21),
                (0, 1, 0x10),
                (0, 4, 0x40),
                (0, 4, 0x41),
                (1, 4, 0x41),
                (2, 4, 0x40),
            ]
        )
    )
    assert (0, 4, 0x40) in env_writes(ow)
    assert (0, 4, 0x41) in env_writes(ow)
    ids = stream.encode(ow)
    assert stream.decode(ids) == corrected_writes(ow)


def test_hard_restart_adsr_order_is_byte_exact():
    """A within-frame AD/SR/ctrl hard-restart sequence round-trips write-for-write in
    its original order."""
    ow = ordered_writes(
        _df(
            [
                (0, 0, 0x10),
                (1, 5, 0x00),
                (1, 6, 0x00),
                (1, 4, 0x08),
                (1, 5, 0x0A),
                (1, 6, 0xF9),
                (1, 4, 0x11),
                (1, 0, 0x80),
            ]
        )
    )
    ids = stream.encode(ow)
    assert stream.decode(ids) == corrected_writes(ow)


@pytest.mark.skipif(
    not os.path.exists(_GT) or not os.environ.get("PREFRAMR_RUN_CORPUS_GATE"),
    reason="set PREFRAMR_RUN_CORPUS_GATE=1 with the HVSC corpus present to run the "
    "heavy real-corpus byte-exact sweep (excluded from the default parallel suite)",
)
def test_corpus_byte_exact_to_corrected_target():
    """Real-corpus sweep: every sampled tune is byte-exact to {settled non-env grid +
    ordered env writes} through the full ``encode``/``decode`` round trip. Must be 100%.
    """
    n = int(os.environ.get("PREFRAMR_CORPUS_SAMPLE", "2000"))
    music = pd.read_parquet(_GT)
    step = max(1, len(music) // n)
    samp = music.iloc[::step][:n]
    checked = 0
    failures = []
    for sid, sub in zip(samp["sid"], samp["subtune"]):
        dump = sid[:-4] + ".%d.dump.parquet" % int(sub)
        if not os.path.exists(dump):
            continue
        df = pd.read_parquet(dump, columns=["clock", "irq", "chipno", "reg", "val"])
        ow = ordered_writes(df)
        if len(ow) == 0:
            continue
        ids = stream.encode(ow, verify=False)
        if stream.decode(ids) != corrected_writes(ow):
            failures.append(dump)
        checked += 1
    assert checked >= min(n, 2000), f"only {checked} tunes checked"
    assert not failures, f"{len(failures)}/{checked} not byte-exact: {failures[:5]}"
