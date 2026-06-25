"""The C1-C8 anti-Goodhart structural constraints (FLAT_VOCAB_MIGRATION.md).

These are the REAL gate: each closes one degenerate ("compressed output relabeled
as structure") solution. The alphabet-level invariants (C3 no LZ/offset, C8 no
escape/wide tokens) hold on the VOCAB itself; C7 (BYTE-atom fraction) is checked on
a real flat GoatTracker stream. The render-from-tokens per-lane checks (C1/C2/C5/C6)
attach to the generic flat path as it lands (deferred; see the PR follow-ups).
"""

import pytest

from tools import codec_gate as G


def test_c3_no_lz_offset_tokens_in_vocab():
    """C3: the flat alphabet exposes NO REPEAT/TRANSPOSE/OFFSET back-reference
    token -- repetition is content-addressed (REF / INSTR_REF NAMES)."""
    assert G.c3_no_lz_offset_tokens() is True


def test_c8_no_escape_or_wide_tokens_in_vocab():
    """C8: the flat alphabet has no u16-escape / raw-Fn / DUR_LONG / varint token.
    16-bit fields are a fixed (lo, hi) BYTE pair, not a length-prefixed escape."""
    assert G.c8_no_escape_tokens() is True


def test_c8_field_tail_fat_head_passes():
    """C8 per-field tail: a fat-headed field (few distinct, top-K covers >=95%)
    passes; a flat/long-tailed field fails."""
    fat = [4] * 95 + list(range(5))  # values {0,1,2,3,4}; top-8 covers all 100
    ok, cover, distinct = G.c8_field_tail(fat, top_k=8, cover=0.95)
    assert ok and cover >= 0.95 and distinct == 5
    longtail = list(range(100))  # every value distinct -> a missing generator
    ok2, cover2, _ = G.c8_field_tail(longtail, top_k=8, cover=0.95)
    assert not ok2 and cover2 < 0.95


def test_c7_byte_fraction_on_synthetic_streams():
    """C7: a dump-like stream (mostly BYTE atoms) fails the cap; a structural
    stream (mostly NOTE/structural atoms) passes."""
    from preframr_tokens.bacc import flat_serialize as F

    dump = [F.BYTE_BASE + (i % 256) for i in range(100)]
    ok, frac = G.c7_byte_atom_fraction(dump)
    assert not ok and frac == 1.0
    structural = [F.ROW, F.NOTE_ZERO, F.INSTR_REF_BASE, F.PATTERN_BEGIN] * 25
    ok2, frac2 = G.c7_byte_atom_fraction(structural)
    assert ok2 and frac2 == 0.0


def test_flat_structural_checks_pass_on_real_gt_stream():
    """A real flat GoatTracker stream passes the alphabet invariants and is not a
    relabeled byte dump (C7 byte-fraction under the cap)."""
    pytest.importorskip("pygoattracker")
    from pygoattracker import Instrument, Pattern, Row, Song, build_sng
    from pygoattracker.constants import note_value

    from preframr_tokens.bacc.backends.goattracker import make_program
    from preframr_tokens.bacc.serialize import program_to_ids

    song = Song(name="GATE", author="T", copyright="2026")
    wave_ptr = song.wavetable.add(0x41, 0x00)
    song.wavetable.add(0xFF, 0x00)
    song.instruments.append(
        Instrument(
            attack_decay=0x09,
            sustain_release=0x00,
            wave_ptr=wave_ptr,
            gateoff_timer=2,
            first_wave=0x09,
            name="LEAD",
        )
    )
    # Several note-dense patterns so the score (structural NOTE/ROW atoms)
    # dominates the fixed-size header (the boot[25]/boot1[25] BYTE dump) -- a real
    # tune, not a 16-row toy where the constant boot bytes alone exceed the cap.
    pats = []
    for p in range(8):
        pat = Pattern.empty(32)
        for k in range(32):
            pat.rows[k] = Row(note=note_value("C-4") + ((k + p) % 12), instrument=1)
        pats.append(pat)
    song.patterns = pats
    seed = {
        "subtune": 0,
        "adparam": 0x0900,
        "optimize_pulse": 0,
        "optimize_realtime": 0,
    }
    program = make_program(build_sng(song), seed, 256)
    ids = program_to_ids(program)
    metrics = G.flat_structural_checks(ids)
    assert metrics["c3_no_lz"] and metrics["c8_no_escape"]
    # A note-dense multi-pattern song is structural, not a relabeled byte dump.
    assert metrics["c7_byte_fraction"] < 0.5, metrics["c7_byte_fraction"]
