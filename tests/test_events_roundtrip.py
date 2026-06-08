"""Byte-exact ordered-write roundtrip for the event model (REDESIGN_optionB §9, the §2.8 hard invariant):
the fidelity oracle is the exact ordered ``(frame, reg, val)`` write stream, not the settled grid. The full
pipeline (ordered writes -> encode -> tokenize -> de-tokenize -> decode -> ordered writes) must reproduce
the source byte-for-byte in order on the 5 driver fixtures and a corpus sample. This is the guard every
factored layer (notes, gestures, order descriptor) must keep green.
"""

import glob
import os

import pandas as pd
import pytest

from preframr_tokens.events import decoder, encoder, factored, oracle, tokenize

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


def _roundtrip(df: pd.DataFrame) -> tuple[list, list, int]:
    """Full pipeline; returns ``(decoded_triples, source_triples, n_tokens)``."""
    ow = oracle.ordered_writes(df)
    events = encoder.encode(ow)
    tokens = tokenize.to_tokens(events)
    decoded = decoder.decode(tokenize.from_tokens(tokens))
    return decoded, ow.triples(), len(tokens)


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_driver_byte_exact(name):
    path = os.path.join(_CACHE, _DRIVERS[name])
    if not os.path.exists(path):
        pytest.skip(f"driver fixture {name} not cached at {path}")
    decoded, source, _ = _roundtrip(pd.read_parquet(path))
    assert decoded == source, f"{name} diverged from the ordered write stream"


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_driver_byte_exact_factored(name):
    """The factored v1 codec (gesture lanes + ORDER descriptor) is byte-exact AND no larger than v0."""
    path = os.path.join(_CACHE, _DRIVERS[name])
    if not os.path.exists(path):
        pytest.skip(f"driver fixture {name} not cached at {path}")
    ow = oracle.ordered_writes(pd.read_parquet(path))
    tokens = factored.encode(ow)
    assert factored.decode(tokens) == ow.triples(), f"{name} factored diverged"
    v0_tokens = tokenize.to_tokens(encoder.encode(ow))
    assert len(tokens) <= len(v0_tokens), f"{name} factored expanded vs v0"


def test_corpus_sample_byte_exact():
    files = sorted(glob.glob(os.path.join(_CORPUS, "*", "*", "*.dump.parquet")))
    if not files:
        pytest.skip(f"no corpus dumps under {_CORPUS}")
    import random

    random.Random(1).shuffle(files)
    checked = 0
    for path in files[:200]:
        try:
            df = pd.read_parquet(path, columns=["clock", "irq", "chipno", "reg", "val"])
        except Exception:
            continue
        ow = oracle.ordered_writes(df)
        if len(ow) == 0:
            continue
        decoded, source, _ = _roundtrip(df)
        assert decoded == source, f"v0 diverged: {path}"
        assert (
            factored.decode(factored.encode(ow)) == source
        ), f"factored diverged: {path}"
        checked += 1
    assert checked >= 50, f"only {checked} corpus tunes checked"


def _ow_from_writes(writes, n_frames):
    """Build an OrderedWrites from a list of ``(frame, reg, val)`` in source order."""
    import numpy as np

    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n_frames,
        irq=np.arange(n_frames, dtype=np.int64),
    )


def test_factored_synthetic_layers_byte_exact():
    """The factored codec is byte-exact on the tricky cases: a multi-speed sub-frame repeat (literal
    run), freq vibrato around a held grid note (freq two-layer), a scalar PW ramp (POLY), and a redundant
    write -- without touching the slow corpus."""
    from preframr_tokens.macros import pitch_grid

    writes = []
    base = pitch_grid.note_freq_at(49, 0.0)
    for f in range(20):
        F = base if f < 8 else base + int(6 * ((f % 4) - 2))
        writes.append((f, 0, F & 0xFF))
        writes.append((f, 1, (F >> 8) & 0xFF))
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
    assert factored.decode(factored.encode(ow)) == ow.triples()


def test_vocab_is_complete_and_escape_free():
    """Every token belongs to a bounded field-family range; there is no literal/escape token (§2.2)."""
    assert tokenize.VOCAB_SIZE == tokenize.DELTA_BASE + 32
    ow = oracle.OrderedWrites(
        frame=__import__("numpy").array([0, 0, 1, 5, 5]),
        reg=__import__("numpy").array([4, 4, 0, 1, 1]),
        val=__import__("numpy").array([17, 17, 255, 0, 200]),
        n_frames=6,
        irq=__import__("numpy").array([0, 1, 2, 3, 4, 5]),
    )
    decoded = decoder.decode(
        tokenize.from_tokens(tokenize.to_tokens(encoder.encode(ow)))
    )
    assert decoded == ow.triples()
