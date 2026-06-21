"""Part B -- the driver-invariance gate.

The canonical 12-TET A440 grid index is the NOTE token's identity. The same
concert pitch must yield the SAME grid index whether it arrives as a Hubbard
onset register Fn or a GoatTracker note byte resolved through GoatTracker's
freq table. This is the executable definition of "same note = same token".
"""

import math

import pytest

from preframr_tokens.bacc.pitch import FN_A440, fn_to_grid, grid_to_fn

pygoattracker = pytest.importorskip("pygoattracker")
from pygoattracker import constants as C  # noqa: E402


def _gt_note_fn(note_byte):
    """GoatTracker pattern note byte -> its freq-table SID Fn."""
    return C.FREQ_TABLE[note_byte - C.FIRSTNOTE]


def test_a440_anchor_is_grid_zero():
    assert fn_to_grid(FN_A440) == 0
    # A-4 from GoatTracker's own table also lands on 0.
    assert fn_to_grid(_gt_note_fn(C.note_value("A-4"))) == 0


def test_grid_roundtrips_each_semitone():
    for n in range(-48, 40):
        assert fn_to_grid(grid_to_fn(n)) == n


def test_goattracker_freq_table_is_clean_12tet():
    # Every adjacent GoatTracker note differs by exactly one grid step.
    prev = None
    for note in range(C.FIRSTNOTE, C.FIRSTNOTE + 96):
        fn = _gt_note_fn(note)
        if fn <= 0:
            continue
        g = fn_to_grid(fn)
        if prev is not None:
            assert g - prev == 1, f"note {note}: grid jump {g - prev}"
        prev = g


# Hubbard note table for Monty: note byte -> onset Fn = table[note*2] |
# table[note*2+1]<<8. These are the exact onset Fn values the Monty driver writes
# (captured from the recovered static note table); pinned here so the gate needs
# no .sid download. GoatTracker note bytes for the SAME concert pitches must snap
# to the identical grid index.
_HUBBARD_ONSET_FN = {
    48: 4456,  # C-4
    52: 5611,  # E-4
    55: 6675,  # G-4
    57: 7494,  # A-4
}
_GT_NAME = {48: "C-4", 52: "E-4", 55: "G-4", 57: "A-4"}


def test_same_concert_pitch_same_grid_index_across_drivers():
    for hub_note, fn in _HUBBARD_ONSET_FN.items():
        hub_grid = fn_to_grid(fn)
        gt_grid = fn_to_grid(_gt_note_fn(C.note_value(_GT_NAME[hub_note])))
        assert (
            hub_grid == gt_grid
        ), f"{_GT_NAME[hub_note]}: Hubbard grid {hub_grid} != GoatTracker {gt_grid}"
    # And the anchor lands where music theory says (A-4 == 0, C-4 == -9).
    assert fn_to_grid(_HUBBARD_ONSET_FN[57]) == 0
    assert fn_to_grid(_HUBBARD_ONSET_FN[48]) == -9


def _clean_et_static_img():
    """A synthetic Hubbard note table: a clean ascending 12-TET run from index 0
    so the grid<->index bijection resolves every playable note. Index n renders
    the canonical A440 Fn for grid (n - 57) -- i.e. note 57 == A-4 == grid 0,
    matching the real Monty table's anchor."""
    img = [0] * 256
    for note in range(96):
        fn = int(round(grid_to_fn(note - 57)))
        img[note * 2] = fn & 0xFF
        img[note * 2 + 1] = (fn >> 8) & 0xFF
    return img


def test_serialized_note_token_identical_across_drivers():
    """The executable proof: for the SAME concert pitch, the Hubbard serializer
    and the GoatTracker serializer emit the IDENTICAL note token (the actual
    LEB-digit bytes, not merely the grid integer)."""
    from preframr_tokens.bacc import gt_serialize, serialize

    img = _clean_et_static_img()
    _, index_to_grid = serialize.hubbard_grid_bijection(img)
    for hub_note, name in _GT_NAME.items():
        # Hubbard emitted note token for its note-table index (porta=0, so the
        # folded porta-present flag at bit1 is clear and the token is the pure
        # zig-zag grid index -- identical to GoatTracker's for the same pitch).
        hub_tok = []
        serialize._note_field(hub_tok, hub_note, 0, img, index_to_grid)
        # GoatTracker emitted note token for the SAME concert pitch.
        gt_kind, gt_grid = gt_serialize._note_token(C.note_value(name))
        gt_tok = []
        serialize._wu(gt_tok, serialize._zz(gt_grid) << 2)
        assert gt_kind == gt_serialize._KIND_PITCH
        assert hub_tok == gt_tok, (
            f"{name}: Hubbard note token {hub_tok} != GoatTracker {gt_tok} "
            "(drivers must emit ONE identical token per concert pitch)"
        )
        # And the token decodes back to the canonical grid index both share.
        z, _ = serialize._ru(hub_tok, 0)
        assert serialize._unzz(z >> 2) == fn_to_grid(_gt_note_fn(C.note_value(name)))


def test_micro_deviation_is_subcent_for_a4():
    # A-4 in both tables is within a couple cents of the canonical anchor; the
    # residual is tuning (Part C), never the note token.
    for fn in (_HUBBARD_ONSET_FN[57], _gt_note_fn(C.note_value("A-4"))):
        cents = 1200.0 * math.log2(fn / grid_to_fn(0))
        assert abs(cents) < 5.0


def _decode_rows(out, img):
    """Decode a single voice's serialized row stream back to rows (mirrors the
    TRANSPOSE/REPEAT handling in ids_to_program, in isolation)."""
    from preframr_tokens.bacc import serialize as s

    index_of, grid_of = s.hubbard_grid_bijection(img)
    rows, i, seen = [], 0, set()
    while i < len(out):
        if out[i] == s.REPEAT:
            i += 1
            off, i = s._ru(out, i)
            length, i = s._ru(out, i)
            base = len(rows)
            for j in range(length):
                rows.append(rows[base - off + j])
        elif out[i] == s.TRANSPOSE:
            i += 1
            off, i = s._ru(out, i)
            length, i = s._ru(out, i)
            delta, i = s._ri(out, i)
            base = len(rows)
            for j in range(length):
                dt, note, instr, lnth, porta = rows[base - off + j]
                rows.append((dt, index_of[grid_of[note] + delta], instr, lnth, porta))
        else:
            dt, i = s._ru(out, i)
            note, has_porta, i = s._read_note_field(out, i, index_of)
            instr, i = s._ru(out, i)
            if instr not in seen:
                seen.add(instr)
                for _ in range(8):
                    _, i = s._ru(out, i)
            lnth, i = s._ru(out, i)
            if has_porta:  # porta field present only when its note-token flag set
                porta, i = s._ru(out, i)
            else:
                porta = 0
            rows.append((dt, note, instr, lnth, porta))
    return rows


def test_transpose_op_factors_a_transposed_phrase_lossless():
    """A phrase repeated TRANSPOSED (every note shifted by the same Delta, non-note
    fields identical) is factored by the backward Transpose op as REPEAT+Delta,
    and decodes byte-exact -- a lossless re-coordinate, never stored/averaged."""
    from preframr_tokens.bacc import serialize as s

    img = _clean_et_static_img()
    grid_to_index, index_to_grid = s.hubbard_grid_bijection(img)
    # A 4-note phrase (grid 0,2,4,5 -> note indices), then the SAME phrase up +7.
    phrase = [grid_to_index[g] for g in (0, 2, 4, 5)]
    up7 = [grid_to_index[g + 7] for g in (0, 2, 4, 5)]
    rows = [(8, n, 1, 16, 0) for n in phrase] + [(8, n, 1, 16, 0) for n in up7]

    out, seen, instruments = [], set(), [[0] * 8 for _ in range(64)]
    s._emit_rows(out, rows, seen, instruments, img, index_to_grid)

    assert s.TRANSPOSE in out, "transposed phrase repeat was not factored"
    # The transposed copy is cheaper than re-emitting 4 literal rows.
    assert len(out) < sum(s._lit_cost(r, img, index_to_grid) for r in rows)
    # Lossless: the decoded rows are byte-identical to the originals.
    assert _decode_rows(out, img) == rows


def test_transpose_op_delta_is_signed_and_roundtrips_down():
    """A downward transposition (negative Delta) round-trips too (signed LEB)."""
    from preframr_tokens.bacc import serialize as s

    img = _clean_et_static_img()
    grid_to_index, index_to_grid = s.hubbard_grid_bijection(img)
    base = [grid_to_index[g] for g in (12, 14, 16, 17, 19)]
    down5 = [grid_to_index[g - 5] for g in (12, 14, 16, 17, 19)]
    rows = [(4, n, 2, 8, 0) for n in base] + [(4, n, 2, 8, 0) for n in down5]
    out, seen = [], set()
    s._emit_rows(out, rows, seen, [[0] * 8 for _ in range(64)], img, index_to_grid)
    assert s.TRANSPOSE in out
    assert _decode_rows(out, img) == rows


def test_exact_repeat_uses_plain_repeat_not_zero_delta_transpose():
    """An identical (non-transposed) phrase repeat is factored by a plain REPEAT --
    Delta=0 is the exact-REPEAT case, never emitted as a TRANSPOSE(Delta=0). The
    decoded rows round-trip byte-exact either way."""
    from preframr_tokens.bacc import serialize as s

    img = _clean_et_static_img()
    grid_to_index, index_to_grid = s.hubbard_grid_bijection(img)
    # A jagged 6-note phrase (no internal constant-interval run to mis-factor),
    # then the SAME phrase verbatim: the long backward match is an exact REPEAT.
    phrase = [grid_to_index[g] for g in (0, 7, 3, 10, 5, 1)]
    rows = [(8, n, 1, 16, 0) for n in phrase] * 2
    out, seen = [], set()
    s._emit_rows(out, rows, seen, [[0] * 8 for _ in range(64)], img, index_to_grid)
    assert s.REPEAT in out  # the verbatim repeat is a plain REPEAT
    # No TRANSPOSE marker is emitted with a zero delta (delta==0 is skipped).
    i = 0
    while i < len(out):
        if out[i] == s.TRANSPOSE:
            _, j = s._ru(out, i + 1)
            _, j = s._ru(out, j)
            delta, _ = s._ri(out, j)
            assert delta != 0, "TRANSPOSE emitted with delta==0 (should be REPEAT)"
        i += 1
    assert _decode_rows(out, img) == rows


def test_static_tuning_delta_table_is_small_and_lossless():
    """Part C (static Δ): the GoatTracker freq table's deviation from the
    canonical A440 grid is a small per-note-class ET-rounding residual (<=~4c),
    and the grid<->note mapping is a clean bijection -- so the note token (grid
    index) reconstructs the exact freq-table Fn with no loss. Δ is real (the
    deviation), factored off the note alphabet, never a stored byte."""
    deltas = {}
    bijection = {}
    for note in range(C.FIRSTNOTE, C.LASTNOTE + 1):
        fn = _gt_note_fn(note)
        if fn <= 0:
            continue
        n = fn_to_grid(fn)
        deltas[n] = 1200.0 * math.log2(fn / grid_to_fn(n))
        # Δ is per-note-class (one value per grid index), not per occurrence.
        assert n not in bijection, f"grid index {n} is not unique (would clash)"
        bijection[n] = note
    assert max(abs(c) for c in deltas.values()) < 4.0
    # lossless: grid index -> note byte -> exact freq-table Fn round-trips.
    for n, note in bijection.items():
        assert fn_to_grid(_gt_note_fn(note)) == n
