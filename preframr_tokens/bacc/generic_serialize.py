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

NO ESCAPE (Stage M2).  The generic program is ALWAYS serialized as the Tracker IR
(:mod:`preframr_tokens.bacc.tracker_serialize`): a shared CITG instrument pool + a
per-lane note-event REPEAT/TRANSPOSE stream.  The former per-register ``genfits``
escape (a 1-token ``_FMT_GENFITS`` tag selecting a raw per-frame fit-payload dump)
has been DELETED: an axis no compact CITG covers is floored to a §3.6 literal-table
CITG (:func:`archetypes.literal_table_citg`) -- byte-exact by construction and the
SAME ``citg`` token vocabulary as every other generator -- so the tracker form is
the universal form and there is no second token class to fall back to.  The
round-trip gate (:func:`_verify_tracker_roundtrip`) stays: it can now only differ by
a LARGER table, never a different format, and a divergence is a BUG to surface (not
silently escape), so it raises rather than falling back (HARD RULE #0).
"""

import numpy as np

from preframr_tokens.bacc.tracker_serialize import (
    tracker_ids_to_program,
    tracker_measure,
    tracker_program_to_ids,
)


def _verify_tracker_roundtrip(program, body):
    """Render-equality gate (HARD RULE #0): the tracker form must reconstruct a
    program that renders byte-identically to ``program``.  The lift is lossless by
    construction and the §3.6 floor makes every axis expressible, so this can only
    fail on a genuine BUG -- in which case we RAISE (surface it) rather than silently
    escaping to a different format (the escape no longer exists)."""
    from preframr_tokens.bacc.generic.recover import render_generic

    rebuilt = tracker_ids_to_program(body)
    if not np.array_equal(render_generic(rebuilt), render_generic(program)):
        raise ValueError(
            "generic_serialize: tracker IR round-trip is not byte-exact "
            "(no escape remains -- this is a bug, not a fallback)"
        )


def generic_program_to_ids(program):
    """Serialize a ``driver="generic"`` :class:`BaccProgram` to a flat list of
    token ids (lossless / byte-exact inverse of :func:`generic_ids_to_program`).

    Always emits the Tracker IR form (a shared CITG instrument pool + per-lane
    note-event REPEAT/TRANSPOSE LZ); verified byte-exact before return."""
    body = tracker_program_to_ids(program)
    _verify_tracker_roundtrip(program, body)
    return body


def generic_ids_to_program(ids):
    """Inverse of :func:`generic_program_to_ids` -> a ``driver="generic"``
    :class:`BaccProgram`.  ``score``/``instruments`` are empty (a generic program
    is per-register fits, not a score); ``seed`` is omitted (provenance-only) so the
    round-trip is defined by what render consumes: ``nframes``, ``boot``,
    ``note_table``, ``genfits``, ``eventfits``."""
    return tracker_ids_to_program(ids)


def generic_measure(program):
    """Return ``({block: tokens}, nframes)`` for a generic program's token stream:
    the Tracker IR breakdown (header / instr_def / score).  The block sizes sum to the
    full serialized length, so no second (expensive) lift is needed for the total."""
    brk, nframes = tracker_measure(program)
    brk["fmt"] = "tracker"
    return brk, nframes
