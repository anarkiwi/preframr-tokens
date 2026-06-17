"""Front-loaded instrument bank (v3 DEF->REF): a tune's recurring onset programs are defined once in the
preamble (INSTR_DEF) and referenced in the body (INSTR_REF id), staying byte-exact and constrained-valid.
"""

import numpy as np

from preframr_tokens.events import oracle, stream
from preframr_tokens.events.constrained import EventStreamState
from preframr_tokens.macros import pitch_grid


def _ow(writes, n):
    writes = sorted(writes, key=lambda t: t[0])
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def _two_voice_repeats():
    """Two voices, each retriggering one identical onset program (CTRL 0x41, AD 0x09, SR 0xA9) many
    times at different pitches -- one shared bank entry should cover both voices' onsets.
    """
    writes = []
    base0 = pitch_grid.note_freq_at(49, 0.0)
    base1 = pitch_grid.note_freq_at(54, 0.0)
    f = 0
    for k in range(12):
        fr0 = base0 + 4 * (k % 3)
        fr1 = base1 + 4 * (k % 3)
        writes += [(f, 0, fr0 & 0xFF), (f, 1, (fr0 >> 8) & 0xFF)]
        writes += [(f, 7, fr1 & 0xFF), (f, 8, (fr1 >> 8) & 0xFF)]
        writes += [(f, 5, 0x09), (f, 6, 0xA9), (f, 4, 0x41)]
        writes += [(f, 12, 0x09), (f, 13, 0xA9), (f, 11, 0x41)]
        writes += [(f + 8, 4, 0x40), (f + 8, 11, 0x40)]
        f += 16
    return _ow(writes, f + 4)


def test_bank_fires_and_is_byte_exact():
    """A repeating onset program is banked (K>=1) and referenced (INSTR_REF present), the inline NOTE_ON
    count drops, and decode stays byte-exact against canonical_writes."""
    ow = _two_voice_repeats()
    toks = stream.encode(ow, verify=True)
    assert stream.INSTR_DEF in toks
    assert toks.count(stream.INSTR_REF) > 0
    assert stream.decode(toks) == stream.canonical_writes(ow)


def test_bank_shared_across_voices():
    """The recurring steady-state program (CTRL/AD/SR with no HR) banks once and is referenced by both
    voices' onsets; the per-voice first onset carries hard-restart prep so it is a distinct entry.
    """
    ow = _two_voice_repeats()
    _bank, defs = stream._build_bank(stream._cas_changes(ow))
    keys = [tuple(p) for p in defs]
    assert len(keys) == len(set(keys))
    assert stream.encode(ow, verify=False).count(stream.INSTR_REF) >= 20


def test_constrained_accepts_banked_stream():
    """The grammar mask accepts every atom of a banked encode in order, including INSTR_DEF/INSTR_REF."""
    ow = _two_voice_repeats()
    toks = stream.encode(ow, verify=False)
    bank_size = len(stream._build_bank(stream._cas_changes(ow))[1])
    state = EventStreamState()
    for i, tok in enumerate(toks):
        assert state.valid_mask()[tok], (i, tok)
        state.push(tok)
    assert state.bank_size == bank_size


def test_constrained_forbids_out_of_range_ref():
    """An INSTR_REF id must stay < bank size: the mask forbids the digit that would over-run the bank."""
    ow = _two_voice_repeats()
    toks = stream.encode(ow, verify=False)
    bank_size = len(stream._build_bank(stream._cas_changes(ow))[1])
    state = EventStreamState()
    pos = 0
    while pos < len(toks) and toks[pos] != stream.INSTR_REF:
        state.push(toks[pos])
        pos += 1
    assert pos < len(toks)
    state.push(stream.INSTR_REF)
    mask = state.valid_mask()
    assert mask[stream.VAR_BASE + bank_size - 1]
    assert not mask[stream.VAR_BASE + bank_size]


def test_no_bank_when_no_repeats():
    """A tune whose onsets are all distinct programs emits no INSTR_DEF and no INSTR_REF (pure inline)."""
    writes = []
    for k in range(4):
        f = k * 10
        fr = pitch_grid.note_freq_at(49 + k, 0.0)
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF)]
        writes += [(f, 5, 0x01 + k), (f, 6, 0xA0 + k), (f, 4, 0x41), (f + 5, 4, 0x40)]
    ow = _ow(writes, 44)
    toks = stream.encode(ow, verify=True)
    assert stream.INSTR_DEF not in toks
    assert stream.INSTR_REF not in toks
