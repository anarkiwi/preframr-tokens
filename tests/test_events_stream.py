"""v3 canonical stream codec guards (the corrected fidelity contract): decode reproduces
``canonical_writes`` exactly -- all CTRL/ADSR activity in driver order at sub-frame resolution, settled
freq/PW first per voice group, globals last, gate-offs always derived (no NOTE OFF token), no order
descriptor and no literal mechanism. Scope is single-speed non-digi; the corpus sweep filters accordingly.
"""

import glob
import os
import random

import numpy as np
import pandas as pd
import pytest

from preframr_tokens import dump_meta
from preframr_tokens.events import oracle, stream
from preframr_tokens.macros import pitch_grid

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


def _ow(writes, n):
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def _roundtrip(ow):
    toks = stream.encode(ow)
    got = stream.decode(toks)
    assert got == stream.canonical_writes(ow)
    assert max(toks) < stream.VOCAB_SIZE and min(toks) >= 0
    return toks


def _synthetic_layers():
    """Vibrato around a grid note, a PW ramp, hard restart, gate-off, and same-frame ctrl activity."""
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
    return _ow(writes, 20)


def test_synthetic_layers_canonical_roundtrip():
    _roundtrip(_synthetic_layers())


def test_encode_is_deterministic():
    ow = _synthetic_layers()
    assert stream.encode(ow) == stream.encode(ow)


def test_empty_and_writeless_streams():
    assert stream.decode(stream.encode(_ow([], 0))) == []
    silent = oracle.OrderedWrites(
        frame=np.empty(0, dtype=np.int64),
        reg=np.empty(0, dtype=np.int64),
        val=np.empty(0, dtype=np.int64),
        n_frames=8,
        irq=np.arange(8, dtype=np.int64),
    )
    assert stream.decode(stream.encode(silent)) == []


def test_canonical_drops_subframe_transients_and_rewrites():
    """The canonical contract keeps the settled musical content: sub-frame freq/PW/global transients
    (masked-inaudible, measured -27 dB under coincident content) and same-value rewrites (chip latch
    no-ops) are canonicalized away rather than carried by a literal-style primitive."""
    writes = []
    for f in range(50):
        writes.append((f, 4, 0x41))
        writes.append((f, 0, 0x10 + (f // 10)))
    writes.append((25, 0, 0x99))
    writes.append((25, 0, 0x12))
    writes.sort(key=lambda t: t[0])
    ow = _ow(writes, 50)
    cw = stream.canonical_writes(ow)
    assert [w for w in cw if w[1] == 4] == [(0, 4, 0x41)], "ctrl rewrites are no-ops"
    assert [w for w in cw if w[1] == 0] == [
        (0, 0, 0x10),
        (10, 0, 0x11),
        (20, 0, 0x12),
        (30, 0, 0x13),
        (40, 0, 0x14),
    ], "freq transient (0x99) and unchanged-final writes are dropped"
    _roundtrip(ow)


def test_intra_frame_freq_transient_settles_to_end_of_frame():
    """One frame rewrites voice-0 freq_lo to a DIFFERENT value (0x40 then 0x80): canonical_writes keeps
    exactly one reg-0 write for that frame (the LAST value) and encode(verify=True) self-verifies --
    pinning the post-PRE behavior that intra-frame freq transients settle to end-of-frame state.
    """
    writes = [
        (0, 4, 0x41),
        (0, 0, 0x40),
        (0, 0, 0x80),
    ]
    ow = _ow(writes, 4)
    reg0 = [w for w in stream.canonical_writes(ow) if w[1] == 0]
    assert reg0 == [(0, 0, 0x80)], reg0
    _roundtrip(ow)


def test_chunk_keyframe_carries_note_table_devs():
    """chunk_keyframe snapshots recovered NOTE_TABLE deviations (not just TUNING/TICK), so a chunk's
    conditioning implies the right absolute freqs for deviated notes; the bracketed segment stays
    decode-transparent. An off-grid note plus an on-grid note pins tuning so the off-grid one deviates.
    """
    off = pitch_grid.note_freq_at(49, 0.0) + 7
    on = pitch_grid.note_freq_at(54, 0.0)
    writes = []
    for f in range(40):
        F = off if f < 20 else on
        writes.append((f, 0, F & 0xFF))
        writes.append((f, 1, (F >> 8) & 0xFF))
        writes.append((f, 4, 0x41))
    ow = _ow(sorted(writes, key=lambda t: t[0]), 40)
    tokens = stream.encode(ow)
    assert stream.NOTE_TABLE in tokens, "fixture must recover a note-table deviation"
    kf = stream.chunk_keyframe(tokens, upto=len(tokens))
    assert kf[0] == stream.KEYFRAME and kf[-1] == stream.KEYFRAME
    assert stream.NOTE_TABLE in kf, "keyframe must carry note-table deviations"
    assert stream.decode(stream.strip_keyframes(kf + tokens)) == stream.decode(tokens)


def test_cas_sequence_preserved_in_driver_order():
    """Sub-frame CTRL/ADSR activity (hard restart: two ctrl changes in one frame) survives with the
    onset envelope folded into NOTE_ON and re-emitted on the RECORDED side of the gate edge (AD
    before SR within a side): crossings flip ADSR-bug stall states and driver conventions split
    (preframr-audio ``test_release_write_position``/``test_gate_adsr_reference``). Both envelope
    writes preceded the gate here, so the canonical form keeps them pre-gate."""
    writes = [
        (0, 6, 0xA9),
        (0, 5, 0x18),
        (0, 4, 0x41),
        (4, 4, 0x40),
        (6, 4, 0x80),
        (6, 5, 0xFF),
        (6, 4, 0x81),
    ]
    ow = _ow(writes, 10)
    cw = stream.canonical_writes(ow)
    assert cw == [
        (0, 5, 0x18),
        (0, 6, 0xA9),
        (0, 4, 0x41),
        (4, 4, 0x40),
        (6, 4, 0x80),
        (6, 5, 0xFF),
        (6, 4, 0x81),
    ], "onset envelope keeps the dump's side of the gate edge (AD,SR within the pre side)"
    toks = _roundtrip(ow)
    assert toks.count(stream.FLD_NOTE_ON) == 2
    assert toks.count(stream.FLD_CTRL) == 1
    assert toks.count(stream.FLD_AD) == 0, "onset AD folds into NOTE_ON"
    assert toks.count(stream.FLD_SR) == 0, "onset SR folds into NOTE_ON"


def test_no_note_off_events_ever():
    """Gate 1->0 never appears as an emitted CTRL event: all gate-offs ride NOTE_ON durations."""
    writes = []
    f = 0
    for _n in range(8):
        writes.append((f, 4, 0x41))
        writes.append((f + 11, 4, 0x40))
        f += 12
    ow = _ow(sorted(writes, key=lambda t: t[0]), f + 4)
    toks = _roundtrip(ow)
    assert toks.count(stream.FLD_NOTE_ON) + toks.count(stream.INSTR_REF) == 8
    assert toks.count(stream.FLD_CTRL) == 0
    assert stream.TICK in toks
    offs = [w for w in stream.decode(stream.encode(ow)) if w[2] == 0x40]
    assert len(offs) == 8


def test_gate_off_value_mode_and_drone():
    writes = [
        (0, 4, 0x41),
        (8, 4, 0x10),
        (12, 4, 0x41),
    ]
    ow = _ow(writes, 20)
    toks = _roundtrip(ow)
    assert toks.count(stream.FLD_CTRL) == 0


def test_retrigger_same_frame_keeps_gate_semantics():
    """off-then-on in one frame (retrigger) and on-then-off (zero-duration blip) both reconstruct."""
    writes = [
        (0, 4, 0x41),
        (5, 4, 0x40),
        (5, 4, 0x41),
        (9, 4, 0x40),
        (9, 4, 0x41),
        (9, 4, 0x40),
    ]
    ow = _ow(writes, 12)
    _roundtrip(ow)


def test_canonical_reorders_freq_first_and_globals_last():
    """A frame written as [ctrl, freq, cutoff, pw] canonicalizes to [freq, pw, ctrl] then global."""
    writes = [
        (3, 4, 0x41),
        (3, 0, 0x55),
        (3, 22, 0x30),
        (3, 2, 0x44),
    ]
    ow = _ow(writes, 5)
    assert stream.canonical_writes(ow) == [
        (3, 0, 0x55),
        (3, 2, 0x44),
        (3, 4, 0x41),
        (3, 22, 0x30),
    ]
    _roundtrip(ow)


def test_pitch_is_interval_coded():
    writes = []
    for k, note in enumerate([49, 51, 53, 54, 56, 58, 60, 61]):
        F = pitch_grid.note_freq_at(note, 0.0)
        writes.append((k * 4, 0, F & 0xFF))
        writes.append((k * 4, 1, (F >> 8) & 0xFF))
    ow = _ow(sorted(writes, key=lambda t: t[0]), 32)
    toks = _roundtrip(ow)
    assert toks.count(stream.NI_STEP) == 8
    wide = 0
    i = 0
    while i < len(toks):
        if toks[i] == stream.NI_STEP:
            j = i + 1
            ndig = 1
            while toks[j] & 0x10:
                ndig += 1
                j += 1
            wide += ndig > 1
        i += 1
    assert wide == 1, "intervals after the first must be single-digit"


def test_decoder_rejects_malformed_streams():
    ow = _synthetic_layers()
    toks = stream.encode(ow)
    assert toks[0] & 0x10, "n_frames=20 must be a continued varint digit"
    with pytest.raises((ValueError, IndexError)):
        stream.decode(toks[:1])
    first_voice = next(
        i for i, t in enumerate(toks) if stream._is_voice(t)
    )  # pylint: disable=protected-access
    with pytest.raises((ValueError, IndexError)):
        stream.decode(toks[: first_voice + 1])
    with pytest.raises((ValueError, IndexError, KeyError)):
        bad = list(toks)
        bad.insert(5, stream.SHAPE_POLY)
        stream.decode(bad)


def test_single_speed_scope_helper():
    single = _ow([(f, 4, 0x41 if f % 2 else 0x40) for f in range(20)], 20)
    assert stream.single_speed(single)
    multi = _ow(
        sorted(
            [(f, 0, (3 * f) & 0xFF) for f in range(20)]
            + [(f, 0, (3 * f + 1) & 0xFF) for f in range(20)],
            key=lambda t: t[0],
        ),
        20,
    )
    assert not stream.single_speed(multi)


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_driver_canonical_roundtrip(name):
    path = os.path.join(_CACHE, _DRIVERS[name])
    if not os.path.exists(path):
        pytest.skip(f"driver fixture {name} not cached at {path}")
    ow = oracle.ordered_writes(pd.read_parquet(path))
    toks = stream.encode(ow)
    assert stream.decode(toks) == stream.canonical_writes(ow), f"{name} diverged"


def _corpus_sample():
    """The seeded 200-tune sample (cheap: glob + shuffle, no parquet reads) -- parametrizing one case
    per file lets xdist spread the encode cost across all workers instead of one serial loop.
    """
    files = sorted(glob.glob(os.path.join(_CORPUS, "*", "*", "*.dump.parquet")))
    random.Random(1).shuffle(files)
    return files[:200]


_CORPUS_SAMPLE = _corpus_sample()


def test_corpus_sample_has_enough_candidates():
    if not _CORPUS_SAMPLE:
        pytest.skip(f"no corpus dumps under {_CORPUS}")
    assert len(_CORPUS_SAMPLE) >= 50, "corpus sample too small to be meaningful"


@pytest.mark.parametrize("path", _CORPUS_SAMPLE or [None])
def test_corpus_tune_canonical_roundtrip(path):
    if path is None:
        pytest.skip(f"no corpus dumps under {_CORPUS}")
    try:
        df = pd.read_parquet(path, columns=["clock", "irq", "chipno", "reg", "val"])
    except Exception:  # pylint: disable=broad-except
        pytest.skip("unreadable dump")
    ow = oracle.ordered_writes(df)
    if len(ow) == 0 or not stream.single_speed(ow):
        pytest.skip("out of scope (empty or multispeed)")
    meta = dump_meta.read_meta(path)
    if meta is not None and meta.is_digi:
        pytest.skip("digi")
    assert stream.decode(stream.encode(ow)) == stream.canonical_writes(
        ow
    ), f"diverged: {path}"


def _is_unit_head(tok):
    return (
        stream._is_digit(tok)  # pylint: disable=protected-access
        or stream._is_voice(tok)  # pylint: disable=protected-access
        or tok in stream._EVENT_KINDS  # pylint: disable=protected-access
        or tok == stream.INSTR_DEF
    )


def _assert_unit_starts(atoms):
    starts = stream.unit_starts(atoms)
    assert starts and starts[0] == 0
    assert all(b > a for a, b in zip(starts, starts[1:]))
    assert all(_is_unit_head(atoms[s]) for s in starts)
    bounds = starts[1:] + [len(atoms)]
    assert bounds[-1] == len(atoms)
    assert all(b > a for a, b in zip(starts, bounds))
    return starts


def test_unit_starts_synthetic_segmentation():
    atoms = stream.encode(_synthetic_layers())
    _assert_unit_starts(atoms)
    assert stream.decode(atoms) == stream.canonical_writes(_synthetic_layers())


@pytest.mark.parametrize("name", sorted(_DRIVERS))
def test_unit_starts_driver_segmentation(name):
    path = os.path.join(_CACHE, _DRIVERS[name])
    if not os.path.exists(path):
        pytest.skip(f"driver fixture {name} not cached at {path}")
    ow = oracle.ordered_writes(pd.read_parquet(path))
    atoms = stream.encode(ow)
    _assert_unit_starts(atoms)
    assert stream.decode(atoms) == stream.canonical_writes(ow)
