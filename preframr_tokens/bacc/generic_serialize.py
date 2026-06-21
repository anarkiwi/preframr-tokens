"""Generic-driver token (de)serialization for ``driver="generic"`` programs.

A generic :class:`~preframr_tokens.bacc.primitive.BaccProgram` (built by
:mod:`preframr_tokens.bacc.generic.recover`) carries NO score / instruments /
``static_img`` -- it is a per-register fitted program: ``nframes``, a frame-0
25-register ``boot`` seed, the bus-recovered ``note_table`` (None or 128 freqs),
and the serialized generator-lane (``genfits``) + event-lane (``eventfits``)
fits.  The hand-backend codecs in :mod:`preframr_tokens.bacc.serialize` have no
generic case (their default path expects a Hubbard ``static_img`` the generic
program lacks), so this module gives the generic program a LOSSLESS, byte-exact
round-trip through the model-facing token-id stream.

The encoder is a faithful INVERSE of the decoder: it reuses the LEB primitives
from :mod:`serialize` (``_wu``/``_ru``/``_wi``/``_ri``) and encodes the fit
payloads GENERICALLY by JSON-value type (None / bool / int / float / str / list
/ dict) rather than per-archetype, so it stays correct as the archetype set
evolves (e.g. a migration into CITG) -- whatever ``genfits``/``eventfits`` carry
is reproduced verbatim, never approximated or dropped.
"""

import struct

from preframr_tokens.bacc.primitive import BaccProgram
from preframr_tokens.bacc.serialize import _ri, _ru, _wi, _wu

# Generic-value type tags for the JSON-clean fit payloads.  Each tagged value is
# a faithful inverse pair: encode writes the tag then the type's body, decode
# reads the tag and dispatches.  Floats are stored by their exact IEEE-754 bits
# (8 bytes, little-endian) so the round-trip is bit-exact, never approximate.
_T_NONE = 0
_T_FALSE = 1
_T_TRUE = 2
_T_INT = 3
_T_FLOAT = 4
_T_STR = 5
_T_LIST = 6
_T_DICT = 7


def _write_value(out, value):
    """LEB-encode an arbitrary JSON-clean fit value (faithful inverse of
    :func:`_read_value`).  bool is checked before int (``bool`` subclasses
    ``int``) so True/False round-trip as themselves, not as 0/1."""
    if value is None:
        out.append(_T_NONE)
    elif value is True:
        out.append(_T_TRUE)
    elif value is False:
        out.append(_T_FALSE)
    elif isinstance(value, int):
        out.append(_T_INT)
        _wi(out, value)
    elif isinstance(value, float):
        out.append(_T_FLOAT)
        for b in struct.pack("<d", value):
            _wu(out, b)
    elif isinstance(value, str):
        out.append(_T_STR)
        data = value.encode("utf-8")
        _wu(out, len(data))
        for b in data:
            _wu(out, b)
    elif isinstance(value, (list, tuple)):
        out.append(_T_LIST)
        _wu(out, len(value))
        for item in value:
            _write_value(out, item)
    elif isinstance(value, dict):
        out.append(_T_DICT)
        _wu(out, len(value))
        for key, item in value.items():
            _write_value(out, key)  # JSON keys are strings; encoded as values
            _write_value(out, item)
    else:
        raise TypeError(f"generic_serialize: unsupported value type {type(value)!r}")


def _read_value(ids, i):
    """Inverse of :func:`_write_value` -> ``(value, i)``.  Lists decode to
    ``list`` and dicts to ``dict`` (the serialized fit payloads use plain lists
    and string-keyed dicts; the *fits (de)serializers in ``recover`` accept
    those)."""
    tag = ids[i]
    i += 1
    if tag == _T_NONE:
        return None, i
    if tag == _T_TRUE:
        return True, i
    if tag == _T_FALSE:
        return False, i
    if tag == _T_INT:
        return _ri(ids, i)
    if tag == _T_FLOAT:
        raw = bytearray()
        for _ in range(8):
            b, i = _ru(ids, i)
            raw.append(b)
        return struct.unpack("<d", bytes(raw))[0], i
    if tag == _T_STR:
        n, i = _ru(ids, i)
        data = bytearray()
        for _ in range(n):
            b, i = _ru(ids, i)
            data.append(b)
        return data.decode("utf-8"), i
    if tag == _T_LIST:
        n, i = _ru(ids, i)
        items = []
        for _ in range(n):
            item, i = _read_value(ids, i)
            items.append(item)
        return items, i
    if tag == _T_DICT:
        n, i = _ru(ids, i)
        out = {}
        for _ in range(n):
            key, i = _read_value(ids, i)
            item, i = _read_value(ids, i)
            out[key] = item
        return out, i
    raise ValueError(f"generic_serialize: unknown value tag {tag}")


def _genfits_blocks(out, program):
    """Emit nframes / boot / note_table / genfits / eventfits and return the
    cumulative token offset after each block (for :func:`generic_measure`)."""
    bounds = {}
    _wu(out, program.nframes)
    for b in program.boot:
        _wu(out, b)
    bounds["boot"] = len(out)
    note_table = program.tables.get("note_table")
    if note_table is None:
        _wu(out, 0)
    else:
        _wu(out, 1)
        _wu(out, len(note_table))
        for v in note_table:
            _wu(out, v)
    bounds["note_table"] = len(out)
    _write_value(out, program.tables["genfits"])
    bounds["genfits"] = len(out)
    _write_value(out, program.tables["eventfits"])
    bounds["eventfits"] = len(out)
    return bounds


def generic_program_to_ids(program):
    """Serialize a ``driver="generic"`` :class:`BaccProgram` to a flat list of
    token ids (lossless / byte-exact inverse of :func:`generic_ids_to_program`).
    """
    out = []
    _genfits_blocks(out, program)
    return out


def generic_ids_to_program(ids):
    """Inverse of :func:`generic_program_to_ids` -> a ``driver="generic"``
    :class:`BaccProgram`.  ``score``/``instruments`` are empty (a generic program
    is per-register fits, not a score); ``seed`` is omitted (provenance-only, not
    needed to render) so the round-trip is defined by what render consumes:
    ``nframes``, ``boot``, ``note_table``, ``genfits``, ``eventfits``."""
    i = 0
    nframes, i = _ru(ids, i)
    boot = []
    for _ in range(25):
        b, i = _ru(ids, i)
        boot.append(b)
    has_nt, i = _ru(ids, i)
    if has_nt:
        n, i = _ru(ids, i)
        note_table = []
        for _ in range(n):
            v, i = _ru(ids, i)
            note_table.append(v)
    else:
        note_table = None
    genfits, i = _read_value(ids, i)
    eventfits, i = _read_value(ids, i)
    return BaccProgram(
        driver="generic",
        nframes=nframes,
        boot=boot,
        instruments=[],
        score=[],
        seed={},
        tables={
            "note_table": note_table,
            "genfits": genfits,
            "eventfits": eventfits,
        },
    )


def generic_measure(program):
    """Return ``({block: tokens}, nframes)`` for a generic program's token
    stream: per-block breakdown of boot / note_table / genfits / eventfits."""
    out = []
    bounds = _genfits_blocks(out, program)
    nframes_tokens = bounds["boot"] - 25
    brk = {
        "nframes": nframes_tokens,
        "boot": 25,
        "note_table": bounds["note_table"] - bounds["boot"],
        "genfits": bounds["genfits"] - bounds["note_table"],
        "eventfits": bounds["eventfits"] - bounds["genfits"],
        "total": len(out),
    }
    return brk, program.nframes
