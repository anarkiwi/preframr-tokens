"""Lift the per-register generic fits into a TRACKER-shaped program, lossless.

The per-register generic recovery (:mod:`recover`) reaches whole-tune
residual-zero, but its *fits* (``genfits``/``eventfits``) are the per-register
EXECUTION: thousands of per-note-on piecewise segments that re-emit the SAME
instrument program at every note (Grid_Runner: 33,913 generator segments but only
~241 distinct PITCH-INVARIANT fit signatures -- a ~143x redundancy), plus per-frame
``carry`` arrays.  Serialized verbatim this is ~2.9M tokens vs the hand backend's
~2.8k.

This module is the generic DECOMPILER: it LIFTS the byte-exact fits into a
tracker-like program -- a small shared INSTRUMENT pool (the deduped pitch-invariant
fit programs, recovered once) plus a per-lane NOTE-EVENT stream (each event
references an instrument + carries the note's pitch + the minimal per-note seed) --
so the EXISTING shared score machinery (REPEAT/TRANSPOSE backward-LZ,
inline-define-on-first-use) compresses it to hand-backend scale.

The lift is a LOSSLESS RE-EXPRESSION of the already-residual-zero fits, NOT a
re-fit (HARD RULE #0): every instrument is a genuine reused program object, every
per-note residue is a few ints (never a per-frame array), and the carry is
RECOMPUTED from the freq instrument at render, never stored.  :func:`unlift`
reconstructs the original ``genfits``/``eventfits`` byte-for-byte, so the recovered
tracker program renders identically (residual-zero) to the per-register program.

The pitch axis is factored out exactly as the STEP/TRACKER reframe did (the
"pitch-invariant instruments" that collapsed Monty's 818 freq-bodies -> 45): a
generator-lane value table (an arp / table-walk over absolute freqs) is recoded as
NOTE-TABLE-INDEX offsets from the note's base index, so the same arp played at many
pitches is ONE instrument; the per-note base index rides the event.  A table whose
entries are not all on the recovered note-table grid is left absolute (an aliased /
swept body) -- lossless, just not pitch-shared.
"""

import json
from dataclasses import dataclass

import numpy as np

from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import fitter as F
from preframr_tokens.bacc.generic.busstate import NREG

# ---------------------------------------------------------------------------
# The Tracker IR (Stage M0): the named, no-escape intermediate representation
# the generic decompiler compiles into and the serializer consumes.
#
# The IR is the (pool, lanes, note_table) triple :func:`lift` already produces,
# promoted here to a named datatype so both interfaces (backend->IR and IR->tokens)
# are functions over ONE structure.  The IR has three primitive families:
#   * P1  the CITG (the SOLE generator pool-entry shape -- see CITG_VALUE_KEYS);
#   * P2  the NOTE EVENT (a (dt, dur, instr_ref, base, seed) lane record);
#   * P3  the INDEXED STRUCTURAL WALK (a lane = LZ'd event stream; orderlist /
#         pattern reuse the SAME REPEAT/TRANSPOSE backward-LZ machinery).
# No behaviour change: :func:`lift` returns the same triple; :class:`TrackerIR`
# is a thin named view, and :func:`from_program` / :meth:`as_triple` round-trip it.
# ---------------------------------------------------------------------------

# The canonical CITG value schema (the design doc's §1.1 nine-field tuple) for the
# CORE ``read`` / ``accum`` modes -- the SOLE generator pool-entry shape.  ``mode`` +
# ``table`` + ``clock`` are the always-present core; the rest are optional with the
# documented defaults so a minimal generator (e.g. a ``hold``) carries only what it
# needs.  The parametric TABLE-SHAPE / composition modes (§1.1: a closed-form shape
# tag -- ``vibrato`` / ``reflect`` / ``pingfold`` / ``vibreflect`` -- that EXPANDS to
# the array, and the §2 ``vibskydive`` / ``arp_decay`` / ``glide`` / ``additive_pw``
# composites) carry ``mode`` plus their own shape params INSTEAD of an explicit
# ``table``; that is a within-CITG choice, not a separate primitive.  The serializer's
# generic-by-value encoder emits whatever keys a CITG params dict holds, so this
# constant is the executable core schema the canonicalization test pins.
CITG_VALUE_KEYS = (
    "mode",  # READ (ptr selects a value) | ACCUM (ptr selects a signed step)
    "table",  # the period-P loop body (values for READ, signed steps for ACCUM)
    "clock",  # the advance schedule (a {"kind": ...} dict; see _citg_gates)
    "seed",  # the READ hold value0 / the ACCUM acc0 (default 0)
    "lead",  # frames of stall before the clock arms (default 0)
    "phase",  # the pointer start index (default 0)
    "loop",  # the index the pointer wraps to (default 0; a LOOP is mandatory)
    "width",  # the accumulator mask (ACCUM only; default 0xFFFF)
    "wrap",  # the accumulator wrap rule (modulo[lo,hi); default None = modulo-width)
)
# The CITG modes :func:`archetypes.render_citg` dispatches.  ``read`` / ``accum``
# are the two core P1 modes; the rest are the parametric TABLE-SHAPE expanders
# (§1.1: a closed-form table body kept as a tag instead of an explicit array) and
# the §2 composition rules -- all WITHIN-CITG choices, not separate primitives.
CITG_MODES = (
    "read",
    "accum",
    "vibrato",
    "vibrato_exact",
    "reflect",
    "pingfold",
    "vibreflect",
    "vibskydive",
    "arp_decay",
    "glide",
    "additive_pw",
)


@dataclass
class TrackerIR:
    """The named Tracker IR: a shared generator/instrument ``pool`` (CITG programs,
    P1), the per-lane note-event ``lanes`` (P2 records walked as P3), and the shared
    ``note_table`` every event ``base`` indexes.  This is exactly the triple
    :func:`lift` returns; :class:`TrackerIR` names it so the migration can refer to
    one IR datatype rather than an anonymous tuple."""

    pool: list
    lanes: dict
    note_table: object  # list[int] | None

    @classmethod
    def from_program(cls, program):
        """Lift a generic ``program`` into the named IR (== :func:`lift`)."""
        pool, lanes, note_table = lift(program)
        return cls(pool, lanes, note_table)

    def as_triple(self):
        """The ``(pool, lanes, note_table)`` triple the serializer / :func:`unlift`
        consume (the IR is still that triple under the hood)."""
        return self.pool, self.lanes, self.note_table


# The §3.6 no-escape literal-table CITG builder lives in :mod:`archetypes` (it is a
# pure CITG params constructor the fitter floor emits); re-exported here so the IR
# module is the one place that names the canonical schema + the floor construction.
literal_table_citg = A.literal_table_citg

# Per-note residue: the accumulator seed / phase / held value -- a few ints per
# note, NOT per-frame data.  Stripped from the instrument struct (so the struct is
# the pitch-and-phase-invariant program) and carried on the event.
SEED_KEYS = (
    "seed",
    "v0",
    "base",
    "ctr0",
    "value",
    "p0",
    "acc0",
    "phase",
    "sfh0",
    "dir0",
    "d0",
    "value0",
    "lead",
)
# Explicit value tables that hold ABSOLUTE freqs on a generator FREQ lane; recoded
# as note-table-index offsets so a phrase's arp/table is pitch-invariant.
_PITCH_LIST = ("table", "freqs")
_REL = "rel"  # sentinel marking a pitch-factored table inside an instrument struct


def _note_index_lookup(note_table):
    """An exact freq -> note-table-index map (only exact grid hits factor out the
    pitch; a swept / off-grid value stays absolute, lossless)."""
    if note_table is None:
        return None
    arr = np.asarray(note_table, dtype=np.int64)
    index = {}
    for j, freq in enumerate(arr):
        index.setdefault(int(freq), j)  # first index wins (lowest octave)
    return index


def _split_fit(name, params, is_freq, idx_of):
    """Split a single ``(name, params)`` fit into ``(struct, seed, base)``:

    * ``struct`` -- the pitch-and-seed-invariant instrument program (the dedup key).
    * ``seed`` -- the per-note residue (the SEED_KEYS present in ``params``).
    * ``base`` -- the note-table base index a pitch-factored table is relative to,
      or ``-1`` when the fit carries no pitch-factored table.
    """
    struct = {k: v for k, v in params.items() if k not in SEED_KEYS}
    seed = {k: params[k] for k in params if k in SEED_KEYS}
    base = -1
    if is_freq and idx_of is not None:
        for key in _PITCH_LIST:
            seq = struct.get(key)
            if isinstance(seq, list) and seq and not _is_rel(seq):
                idxs = [idx_of.get(int(v)) for v in seq]
                if all(j is not None for j in idxs):
                    base = idxs[0]
                    struct[key] = [_REL] + [j - base for j in idxs]
    return struct, seed, base


def _is_rel(seq):
    return bool(seq) and seq[0] == _REL


def _rebuild_fit(name, struct, seed, base, note_table):
    """Inverse of :func:`_split_fit`: re-expand an instrument struct at a note's
    ``base`` index + ``seed`` into the original ``(name, params)`` fit."""
    params = dict(struct)
    params.update(seed)
    if base >= 0 and note_table is not None:
        arr = np.asarray(note_table, dtype=np.int64)
        for key in _PITCH_LIST:
            seq = params.get(key)
            if isinstance(seq, list) and _is_rel(seq):
                params[key] = [int(arr[base + off]) for off in seq[1:]]
    return (name, params)


# ---------------------------------------------------------------------------
# An EVENT is a per-lane note-on record: (dt, dur, instr_ref, base, seed).
#   dt    -- frames since the previous event on this lane (the note-on delta).
#   dur   -- segment length (frames the instrument body covers).
#   ref   -- index into the shared instrument pool (-1 = an UN-FIT / surfaced gap).
#   base  -- note-table base index for a pitch-factored body (-1 = absolute), or a
#            list of per-piece bases for a piecewise (composite) instrument.
#   seed  -- the per-note residue (a dict for a simple body, a list of per-piece
#            dicts for a piecewise body); the held note value lives here.
# The instrument pool entry is ("S", struct) for a simple body or
# ("P", [[name, struct, piece_len], ...]) for a piecewise (composite) body.
# ---------------------------------------------------------------------------
def lift(program):
    """Lift a generic ``program``'s ``genfits``/``eventfits`` into
    ``(pool, lanes, note_table)``: a shared instrument ``pool`` (list of pool
    entries) and ``lanes`` (an ordered dict ``lane_id -> [event, ...]``).

    Pure structural re-expression -- :func:`unlift` inverts it byte-for-byte.
    """
    note_table = program.tables.get("note_table")
    idx_of = _note_index_lookup(note_table)
    pool = []
    pool_index = {}

    def ref_of(entry, disc=None):
        # ``disc`` is the seed-schema discriminator: two fits with an identical struct but a
        # different seed-key SET (an optional residue key like ``phase`` present on one note,
        # absent on another) MUST get distinct refs, else the per-ref seed schema collides
        # (the serializer stores ONE schema per ref).  See tracker_ir._pool_builder.
        key = (json.dumps(entry, sort_keys=True), disc)
        ref = pool_index.get(key)
        if ref is None:
            ref = len(pool)
            pool_index[key] = ref
            pool.append(entry)
        return ref

    def _schema(seed):
        return tuple(seed.keys()) if seed else ()

    def event_of(start, prev, dur, fit, is_freq):
        if fit is None:
            return (start - prev, dur, -1, -1, None)
        name, params = fit
        if name == "piecewise":
            structs, seeds, bases = [], [], []
            for pname, pparams, plen in params["pieces"]:
                struct, seed, base = _split_fit(pname, pparams, is_freq, idx_of)
                structs.append([pname, struct, plen])
                seeds.append(seed)
                bases.append(base)
            disc = ("P", tuple(_schema(s) for s in seeds))
            return (start - prev, dur, ref_of(["P", structs], disc), bases, seeds)
        struct, seed, base = _split_fit(name, params, is_freq, idx_of)
        return (
            start - prev,
            dur,
            ref_of(["S", [name, struct]], ("S", _schema(seed))),
            base,
            seed,
        )

    lanes = {}
    genfits = program.tables["genfits"]
    for key, payload in genfits.items():
        is_freq = key.endswith("freq")
        events, prev = [], 0
        for start, stop, fit in payload["segments"]:
            events.append(event_of(start, prev, stop - start, fit, is_freq))
            prev = start
        lanes[("g", key)] = events
    eventfits = program.tables["eventfits"]
    for reg, segs in eventfits.items():
        events, prev = [], 0
        for start, stop, name, params in segs:
            events.append(event_of(start, prev, stop - start, (name, params), False))
            prev = start
        lanes[("e", int(reg))] = events
    return pool, lanes, note_table


def unlift(pool, lanes, note_table):
    """Inverse of :func:`lift`: reconstruct ``(genfits, eventfits)`` in the shape
    :func:`recover.render_generic` consumes, byte-for-byte identical to the input.
    The PW ``carry`` is recomputed from the rebuilt freq fits (never stored)."""

    def rebuild_event(event):
        dt, dur, ref, base, seed = event
        prev = unlift.prev  # set by caller loop below
        start = prev + dt
        if ref < 0:
            return start, (start, start + dur, None)
        tag, body = pool[ref]
        if tag == "P":
            pieces = []
            for (name, struct, plen), pbase, pseed in zip(body, base, seed):
                fit = _rebuild_fit(name, struct, pseed, pbase, note_table)
                pieces.append((fit[0], fit[1], plen))
            return start, (start, start + dur, ("piecewise", {"pieces": pieces}))
        name, struct = body
        fit = _rebuild_fit(name, struct, seed, base, note_table)
        return start, (start, start + dur, fit)

    genfits = {}
    eventfits = {}
    for lane_id, events in lanes.items():
        kind, key = lane_id
        segs = []
        unlift.prev = 0
        for event in events:
            start, seg = rebuild_event(event)
            unlift.prev = start
            segs.append(seg)
        if kind == "g":
            genfits[key] = segs
        else:
            eventfits[key] = [(s, t, f[0], f[1]) for s, t, f in segs]
    return genfits, eventfits


def lifted_render(program):
    """Render a generic ``program`` THROUGH the lift (lift -> unlift -> render):
    proves the lift is lossless (== :func:`recover.render_generic` byte-for-byte)
    and is the render path for a tracker-serialized generic program."""
    pool, lanes, note_table = lift(program)
    genfits, eventfits = unlift(pool, lanes, note_table)
    return render_from_fits(genfits, eventfits, note_table, program.nframes)


def render_from_fits(genfits, eventfits, note_table, nframes):
    """Render reconstructed ``genfits``/``eventfits`` to ``(nframes, 25)``.  The PW
    ``carry`` is recomputed from the freq fits (HARD RULE #0: never stored)."""
    nt_arr = np.asarray(note_table, dtype=np.int64) if note_table else None
    rendered = np.zeros((nframes, NREG), dtype=np.int64)
    for voice in range(3):
        fres = genfits[f"{voice}:freq"]
        pres = genfits[f"{voice}:pw"]
        carry = A.freq_carry_sequence(fres, nframes)
        flane, _ = F.render_generator_lane(fres, nframes, nt_arr, None)
        plane, _ = F.render_generator_lane(pres, nframes, nt_arr, carry)
        rendered[:, F.FREQ_LO[voice]] = flane & 0xFF
        rendered[:, F.FREQ_HI[voice]] = (flane >> 8) & 0xFF
        rendered[:, F.PW_LO[voice]] = plane & 0xFF
        rendered[:, F.PW_HI[voice]] = (plane >> 8) & 0x0F
    for reg, segs in eventfits.items():
        rendered[:, reg] = A.render_event_lane(segs, nframes)
    return rendered
