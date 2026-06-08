"""Acceptance guards for the event model (REDESIGN_optionB §9), the fast (corpus-free) subset: the no-escape
invariant, the expansion guard, the note layer, encode determinism, and tokenizer/grammar completeness. The
byte-exact ordered-stream roundtrip on drivers + corpus lives in ``test_events_roundtrip.py``; the BPE
collapse/bits measurement is ``preframr_tokens.events.measure``.
"""

import numpy as np

from preframr_tokens.events import factored, gestures, oracle, tokenize, varint


def _ow(writes, n):
    return oracle.OrderedWrites(
        frame=np.array([f for f, _, _ in writes], dtype=np.int64),
        reg=np.array([r for _, r, _ in writes], dtype=np.int64),
        val=np.array([v for _, _, v in writes], dtype=np.int64),
        n_frames=n,
        irq=np.arange(n, dtype=np.int64),
    )


def test_no_escape_every_value_decodes_over_one_alphabet():
    """§2.2: every numeric field is the complete escape-free zig-zag/varint over ONE digit alphabet --
    a rare large value is more digits of the same alphabet, never a different path. Exhaustively check
    the signed varint round-trips across the full byte range and large magnitudes."""
    for v in list(range(-300, 301)) + [-70000, 65535, 1 << 20]:
        assert varint.unzigzag(varint.decode_unsigned(varint.encode_signed(v))[0]) == v
    assert factored.VOCAB_SIZE == factored.FLD_SR + 1


def test_gesture_basis_is_lossless_for_all_shapes():
    """Every gesture shape (HOLD/POLY/PERIOD) replays its source series exactly (the lossless cover)."""
    rng = np.random.default_rng(0)
    for s in (
        np.full(30, 7),
        np.arange(30) * 3 + 2,
        np.cumsum(rng.integers(-4, 5, 40)) + 100,
    ):
        s = s.astype(np.int64)
        assert (gestures.replay(gestures.cover(s), len(s)) == s).all()


def test_expansion_guard_factored_not_larger_than_verbatim():
    """§9 expansion guard: the factored encoding never exceeds the v0 verbatim token count on a held
    note + ramp (a case the factoring must win, not lose)."""
    writes = []
    for f in range(40):
        writes.append((f, 21, 100 + 2 * f))
        writes.append((f, 24, 15))
    ow = _ow(sorted(writes, key=lambda t: t[0]), 40)
    factored_n = len(factored.encode(ow))
    from preframr_tokens.events import encoder

    v0_n = len(tokenize.to_tokens(encoder.encode(ow)))
    assert factored_n < v0_n
    assert factored.decode(factored.encode(ow)) == ow.triples()


def test_encode_is_deterministic():
    writes = [(f, 0, f & 0xFF) for f in range(25)] + [(f, 1, 0) for f in range(25)]
    ow = _ow(sorted(writes, key=lambda t: t[0]), 25)
    assert factored.encode(ow) == factored.encode(ow)


def test_note_layer_gate_on_typed_and_byte_exact():
    """§8.3/§6: CTRL/AD/SR ride the note layer, not the byte lanes. A gate-on (CTRL bit0 0->1) is an
    explicit ``FLD_NOTE_ON`` edge; gate-off is just a plain CTRL edge (no note-off token); the sustained
    envelope is the ABSENCE of AD/SR edges between notes. The voice's settled CTRL/AD/SR series round-trip
    exactly through the note section, and a hard-restart (AD rewritten across frames) is reproduced.
    """
    from preframr_tokens.events.schema import ad_reg, ctrl_reg, sr_reg

    n = 30
    settled = np.zeros((n, 25), dtype=np.int64)
    cr, ar, srg = ctrl_reg(0), ad_reg(0), sr_reg(0)
    for f in range(n):
        settled[f, cr] = (
            0x40 if f < 2 else (0x41 if f < 10 else (0x11 if f < 20 else 0x10))
        )
        settled[f, ar] = 0x00 if f < 2 else (0xFF if f == 2 else 0x08)
        settled[f, srg] = 0xA9

    edges = factored._note_edges(settled, 0)
    note_on = [e for e in edges if e[1] == factored.FLD_NOTE_ON]
    assert [f for f, _, _ in note_on] == [
        2
    ], "exactly one gate-on edge, typed NOTE_ON, at frame 2"
    assert any(f == 20 and t == factored.FLD_CTRL for f, t, _ in edges)
    assert sum(1 for _, t, _ in edges if t == factored.FLD_SR) == 1

    out: list[int] = []
    factored._emit_note_voice(out, settled, 0)
    series: dict = {}
    pos = factored._read_note_voice(out, 0, n, series)
    assert pos == len(out)
    assert (series[cr] == settled[:, cr]).all()
    assert (series[ar] == settled[:, ar]).all()
    assert (series[srg] == settled[:, srg]).all()


def test_decoder_rejects_malformed_stream():
    """The token grammar is strict: a truncated varint / missing ORDER_MARK fails loudly (no escape)."""
    import pytest

    with pytest.raises(Exception):
        factored.decode([factored.VAR_BASE + varint.CONT])
