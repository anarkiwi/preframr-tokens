"""Byte-exact ordered-write roundtrip for the v3 stream codec on the 5 driver fixtures and a corpus
sample: ``stream.decode(stream.encode(ow)) == stream.canonical_writes(ow)`` per fixture. This is the
fidelity guard the encode self-verify (verify=True) also enforces on every call."""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from preframr_tokens import dump_meta
from preframr_tokens.events import oracle, stream

_CACHE = os.environ.get(
    "PREFRAMR_SID_FIXTURE_CACHE", "/scratch/preframr/sid_fixture_cache"
)
_CORPUS = os.environ.get("PREFRAMR_HVSC_ROOT", "/scratch/preframr/hvsc/MUSICIANS")

_DRIVERS = {
    "grid_runner": "grid_runner_26s.dump.parquet",
    "commando": "commando_20_20s.dump.parquet",
    "camerock": "camerock_20_20s.dump.parquet",
    "trap": "trap_20_20s.dump.parquet",
    "baggis": "baggis_20_20s.dump.parquet",
}


def _ow_from_writes(writes, n_frames):
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n_frames,
        irq=np.arange(n_frames, dtype=np.int64),
    )


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_driver_canonical_roundtrip(name):
    path = os.path.join(_CACHE, _DRIVERS[name])
    if not os.path.exists(path):
        pytest.skip(f"driver fixture {name} not cached at {path}")
    ow = oracle.ordered_writes(pd.read_parquet(path))
    assert stream.decode(stream.encode(ow, verify=False)) == stream.canonical_writes(
        ow
    ), f"{name} diverged from the ordered write stream"


def test_synthetic_layers_canonical_roundtrip():
    """The stream codec is byte-exact on the tricky synthetic cases without the corpus: freq vibrato
    around a held grid note, a scalar PW ramp, a hard restart, gate-off, and same-frame ctrl activity.
    """
    from preframr_tokens.macros import pitch_grid

    writes = []
    base = pitch_grid.note_freq_at(49, 0.0)
    for f in range(20):
        fr = base if f < 8 else base + int(6 * ((f % 4) - 2))
        writes.append((f, 0, fr & 0xFF))
        writes.append((f, 1, (fr >> 8) & 0xFF))
    for f in range(20):
        writes.append((f, 2, 64 + 3 * f if f else 64))
    for f in range(20):
        writes.append(
            (f, 4, 0x40 if f < 4 else (0x41 if f < 10 else (0x11 if f < 16 else 0x10)))
        )
        writes.append((f, 5, 0xFF if f == 4 else (0x08 if f >= 5 else 0x00)))
        writes.append((f, 6, 0xA9))
    writes.append((5, 4, 0x11))
    writes.append((5, 4, 0x21))
    writes.sort(key=lambda t: t[0])
    ow = _ow_from_writes(writes, 20)
    assert stream.decode(stream.encode(ow, verify=False)) == stream.canonical_writes(ow)


def test_corpus_sample_canonical_roundtrip():
    files = sorted(glob.glob(os.path.join(_CORPUS, "*", "*", "*.dump.parquet")))
    if not files:
        pytest.skip(f"no corpus dumps under {_CORPUS}")
    import random

    random.Random(1).shuffle(files)
    checked = 0
    for path in files[:200]:
        try:
            df = pd.read_parquet(path, columns=["clock", "irq", "chipno", "reg", "val"])
        except Exception:  # pylint: disable=broad-except
            continue
        ow = oracle.ordered_writes(df)
        if len(ow) == 0 or not stream.single_speed(ow):
            continue
        meta = dump_meta.read_meta(path)
        if meta is not None and meta.is_digi:
            continue
        assert stream.decode(
            stream.encode(ow, verify=False)
        ) == stream.canonical_writes(ow), f"diverged: {path}"
        checked += 1
    assert checked >= 50, f"only {checked} corpus tunes checked"
