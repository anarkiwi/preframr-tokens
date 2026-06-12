"""Guards for the event-grammar sampling mask: it accepts every token a real encode emits
(conformance), rejects structurally-wrong tokens mid-field (rejection), and a masked random walk
truncated at a group boundary always decodes (fuzz -- the parser must accept the grammar).
"""

import numpy as np

from preframr_tokens.events import oracle, stream, varint
from preframr_tokens.events.constrained import EventStreamState
from preframr_tokens.macros import pitch_grid

_VB = stream.VAR_BASE


def _ow(writes, n):
    writes = sorted(writes, key=lambda t: t[0])
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def _synth():
    writes = []
    base = pitch_grid.note_freq_at(49, 0.0)
    for f in range(20):
        fr = base if f < 8 else base + int(6 * ((f % 4) - 2))
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF)]
    for f in range(20):
        writes.append((f, 2, 64 + 3 * f if f else 64))
    for f in range(20):
        writes += [
            (f, 4, 0x40 if f < 4 else (0x41 if f < 10 else (0x11 if f < 16 else 0x10))),
            (f, 5, 0xFF if f == 4 else (0x08 if f >= 5 else 0x00)),
            (f, 6, 0xA9),
        ]
    writes += [(5, 4, 0x11), (5, 4, 0x21)]
    return _ow(writes, 20)


def _tick():
    writes = []
    f = 0
    for _ in range(8):
        writes += [(f, 4, 0x41), (f + 11, 4, 0x40)]
        f += 12
    return _ow(writes, f + 4)


def _note_table():
    off = pitch_grid.note_freq_at(49, 0.0) + 7
    on = pitch_grid.note_freq_at(54, 0.0)
    writes = []
    for f in range(40):
        fr = off if f < 20 else on
        writes += [(f, 0, fr & 0xFF), (f, 1, (fr >> 8) & 0xFF), (f, 4, 0x41)]
    return _ow(writes, 40)


def _pw_ramp():
    writes = []
    for f in range(40):
        pw = 0x100 + 6 * f
        writes += [(f, 2, pw & 0xFF), (f, 3, (pw >> 8) & 0xFF), (f, 4, 0x41)]
    return _ow(writes, 40)


def _global_and_hr():
    writes = [(0, 4, 0x41), (0, 5, 0x08), (0, 6, 0xA9)]
    writes += [(8, 4, 0x40), (9, 5, 0xFF), (9, 6, 0xF0), (10, 4, 0x41)]
    for f in range(16):
        writes.append((f, 22, 0x30 + f))
    return _ow(writes, 16)


_DUMPS = [_synth(), _tick(), _note_table(), _pw_ramp(), _global_and_hr()]


def test_conformance_accepts_every_encoded_token():
    """For five dumps covering TICK durations, a NOTE_TABLE deviation, a PW ramp, globals and an
    HR-prepped NOTE_ON, the mask accepts each encoded atom in order and push advances without error.
    """
    for ow in _DUMPS:
        toks = stream.encode(ow, verify=False)
        state = EventStreamState()
        for i, tok in enumerate(toks):
            assert state.valid_mask()[tok], (i, tok)
            state.push(tok)


def test_rejection_masks_structurally_wrong_tokens():
    """At every step the mask never offers KEYFRAME and never offers everything; mid-varint it rejects
    VOICE/nibble tokens, and where a nibble is required it rejects digits (≥100 checked positions).
    """
    checked = mid_varint = mid_nibble = 0
    for ow in _DUMPS:
        toks = stream.encode(ow, verify=False)
        state = EventStreamState()
        for tok in toks:
            mask = state.valid_mask()
            assert not mask[stream.KEYFRAME]
            assert mask.sum() < stream.VOCAB_SIZE
            if state.stack and state.stack[-1][0] == "V":
                assert not mask[stream.VOICE_BASE] and not mask[stream.NIB_WAVE]
                mid_varint += 1
            if state.stack and state.stack[-1][0] == "N":
                assert not mask[stream.VAR_BASE]
                mid_nibble += 1
            checked += 1
            state.push(tok)
    assert checked >= 100 and mid_varint > 0 and mid_nibble > 0


def test_fuzz_masked_walk_truncated_at_boundary_decodes():
    """A masked random walk decodes: from a fresh state with a small fixed frame count, sample valid
    tokens (varint length capped so structural repeat-counts stay finite), truncate at the last group
    boundary, and assert stream.decode accepts the prefix (replay may truncate at the frame count).
    """
    for seed in range(8):
        rng = np.random.default_rng(seed)
        state = EventStreamState()
        toks = [_VB + d for d in varint.encode_unsigned(16)]
        for tok in toks:
            state.push(tok)
        last_boundary = 0
        run_cont = 0
        for _ in range(2000):
            choices = np.where(state.valid_mask())[0]
            if run_cont >= 1:
                choices = choices[~((choices >= _VB + 16) & (choices < _VB + 32))]
            tok = int(rng.choice(choices))
            state.push(tok)
            toks.append(tok)
            run_cont = (
                run_cont + 1 if _VB <= tok < _VB + 32 and (tok - _VB) & 0x10 else 0
            )
            if state.at_group_boundary:
                last_boundary = len(toks)
        stream.decode(list(toks[:last_boundary]))
