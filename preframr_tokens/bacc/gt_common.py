"""Driver-neutral pitch/Song helpers shared by the GoatTracker codecs.

These are NOT serialization: they are the canonical-grid note mapping (a note byte
<-> the A440 12-TET grid index, the shared ``NOTE`` atom identity), the abstract
instrument field list, and the orderlist-entry view. They live here, away from any
specific token alphabet, so the flat v2 codec (:mod:`flat_serialize`) can reuse them
without depending on the retired v1 LEB+LZ codec.
"""

from preframr_tokens.bacc.backends.goattracker import make_program_from_song
from preframr_tokens.bacc.pitch import fn_to_grid

# --- canonical note token alphabet ---------------------------------------
# Pitched notes carry the 12-TET grid INTERVAL (signed). Non-pitched markers
# (REST/KEYOFF/KEYON) ride a small reserved namespace so the note field stays a
# single token; the flat codec maps each kind to its own structural token.
_KIND_PITCH = 0
_KIND_REST = 1
_KIND_KEYOFF = 2
_KIND_KEYON = 3
# A playable note byte whose GoatTracker freq-table entry is non-positive (e.g. a
# note below FIRSTNOTE, which the playroutine still sounds as a wrapped/transposed
# tone) has no clean 12-TET grid identity. It rides a literal raw-note escape so
# the row stays byte-exact and lossless instead of feeding log2(<=0) into the grid.
_KIND_RAWNOTE = 4


# --- note tokenization (Part B canonical grid) ----------------------------
def _note_token(note_byte):
    """(kind, grid_interval) for a pattern note byte.

    A pitched note resolves through GoatTracker's freq table to a SID Fn, snaps to
    the canonical A440 12-TET grid (Part B), and carries that grid index as a
    signed interval from the grid origin (n=0=A440) -- driver-invariant, and
    position-independent. REST/KEYOFF/KEYON are non-pitched markers (interval
    unused)."""
    from pygoattracker import constants as c

    if note_byte == c.REST:
        return _KIND_REST, 0
    if note_byte == c.KEYOFF:
        return _KIND_KEYOFF, 0
    if note_byte == c.KEYON:
        return _KIND_KEYON, 0
    idx = note_byte - c.FIRSTNOTE
    fn = c.FREQ_TABLE[idx] if 0 <= idx < len(c.FREQ_TABLE) else 0
    if fn <= 0:
        # No clean freq-table pitch (note below FIRSTNOTE / past the table); keep
        # the raw note byte so the row round-trips byte-exact.
        return _KIND_RAWNOTE, note_byte
    return _KIND_PITCH, fn_to_grid(fn)


def _grid_to_note_byte(grid):
    """Inverse: canonical grid index -> the GoatTracker note byte whose freq-table
    entry snaps to that grid index. The freq table is clean 12-TET so this is a
    bijection on the playable range; built once and cached."""
    return _GRID_NOTE[grid]


def _build_grid_note():
    from pygoattracker import constants as c

    table = {}
    for note in range(c.FIRSTNOTE, c.LASTNOTE + 1):
        fn = c.FREQ_TABLE[note - c.FIRSTNOTE]
        if fn > 0:
            table[fn_to_grid(fn)] = note
    return table


_GRID_NOTE = _build_grid_note()


# --- instruments (generators) ----------------------------------------------
_INSTR_FIELDS = (
    "attack_decay",
    "sustain_release",
    "wave_ptr",
    "pulse_ptr",
    "filter_ptr",
    "vibrato_param",
    "vibrato_delay",
    "gateoff_timer",
    "first_wave",
)


# --- orderlist (abstract entry view) --------------------------------------
# Orderlist entries serialize as (op, value): op 0 PlayPattern(num),
# 1 Repeat(count), 2 Transpose(semitones).
def _orderlist_entries(ol):
    from pygoattracker.model import PlayPattern, Repeat, Transpose

    out = []
    for e in ol.entries:
        if isinstance(e, PlayPattern):
            out.append((0, e.num))
        elif isinstance(e, Repeat):
            out.append((1, e.count))
        elif isinstance(e, Transpose):
            out.append((2, e.semitones))
        else:  # pragma: no cover - model is a closed union
            raise ValueError(f"unknown orderlist entry {e!r}")
    return out


__all__ = [
    "make_program_from_song",
    "_KIND_PITCH",
    "_KIND_REST",
    "_KIND_KEYOFF",
    "_KIND_KEYON",
    "_KIND_RAWNOTE",
    "_note_token",
    "_grid_to_note_byte",
    "_build_grid_note",
    "_GRID_NOTE",
    "_INSTR_FIELDS",
    "_orderlist_entries",
]
