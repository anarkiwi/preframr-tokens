"""The canonical Tracker IR every SID-music driver converges on (Stage 2).

Stage 1 (:mod:`preframr_tokens.bacc.generic.cover`) gives a byte-exact,
token-MINIMIZING cover of each of the 16 SID register lanes (3 voices x
{freq, pw, ctrl, ad, sr} + 4 globals).  Serialized one lane at a time the
per-segment STREAM dominates -- 16 lanes each carrying their own
``(dt, dur, ref, base, seed)`` events, and a voice's freq/pw/ctrl/ad/sr that move
together as ONE note still cost five independent rows.  This module closes that gap
by compiling the per-lane covers into ONE canonical structure -- the **Tracker IR**
-- that BUNDLES a voice's lanes into instrument-referencing rows, factors pitch out
of the freq generators, and exposes the row streams to the shared REPEAT/TRANSPOSE
backward-LZ.

The IR is a LOSSLESS re-expression of the byte-exact lane covers (HARD RULE #0):

* :func:`lift` covers every lane (:func:`cover.cover_lane`), groups each voice's
  lanes into a per-voice :class:`VoiceTrack` (the freq lane is the note spine; a
  sibling lane whose cover shares the spine's segment boundaries is BUNDLED into the
  row, the rest stay free generator streams), pitch-factors a note-table-relative
  freq table, and dedups every instrument struct into a shared pool.
* :func:`unlift` reconstructs the EXACT ``genfits``/``eventfits`` the per-register
  renderer (:func:`recover.render_generic` / :func:`tracker.render_from_fits`)
  consumes, byte-for-byte identical to the cover -- so the recovered tracker program
  renders identically (residual-zero) to the per-register program.

:mod:`preframr_tokens.bacc.tracker_serialize` serializes the IR to the model-facing
token stream and runs the render-equality round-trip self-check.
"""

from dataclasses import dataclass, field

import numpy as np

from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic.tracker import (
    _REL,
    _note_index_lookup,
    _rebuild_fit,
    _split_fit,
    render_from_fits,
)

# The five SID lanes that make up one voice's "instrument row" + the fixed order
# they are bundled in (freq is always the note spine, first).  ``CLASSES`` excludes
# freq (the spine); a subset of it is BUNDLED into a voice's rows, the rest stay free.
SPINE = "freq"
CLASSES = ("pw", "ctrl", "ad", "sr")
# The four global registers (filter cutoff lo/hi, resonance/routing, volume): one
# free event lane each, never bundled into a voice.
GLOBALS = (21, 22, 23, 24)

# The register index of each (voice, class) lane -- the chip layout the cover's
# :func:`cover._lane_specs` derives, repeated here so the IR can address a lane by
# its (voice, class) name and reconstruct the exact ``genfits``/``eventfits`` shape.
_LANE_REG = {
    "pw": (2, 9, 16),
    "ctrl": (4, 11, 18),
    "ad": (5, 12, 19),
    "sr": (6, 13, 20),
}
# A generator lane (freq / pw) reconstructs into ``genfits`` (rendered with the
# pitch table + carry); the event lanes (ctrl / ad / sr / globals) into
# ``eventfits``.  ``freq`` and ``pw`` are the only generator classes.
_GEN_CLASSES = (SPINE, "pw")


@dataclass
class VoiceTrack:
    """One voice's bundled instrument rows + its un-bundled (free) sibling lanes.

    The freq lane is the note spine; ``bundled`` lists the sibling classes (a subset
    of :data:`CLASSES`) whose cover shares the spine's segment boundaries exactly, so
    they fold into the row at one shared ``(dt, dur)`` instead of separate streams.
    ``rows`` is the bundled stream: each row is ``(dur, refs, bases, seeds)`` with one
    ``(ref, base, seed)`` per lane in ``[freq] + bundled`` order (the row start is the
    running sum of durs -- not stored).  ``free`` holds the sibling lanes kept as their
    own event streams (a dense/misaligned lane the spine would only fragment -- e.g.
    A Mind Is Born voice-1 pw)."""

    bundled: list  # class names folded into the spine, in CLASSES order
    rows: list  # [(dur, [ref...], [base...], [seed...]), ...]  spine-aligned
    free: dict = field(default_factory=dict)  # class -> [event, ...]  (own stream)


@dataclass
class TrackerIR:
    """The canonical Tracker IR: a shared instrument/generator ``pool``, three
    per-voice :class:`VoiceTrack` row streams, the four global event lanes, the
    shared ``note_table`` every pitch-factored freq ``base`` indexes, and the frame
    metadata (``nframes`` + frame-0 ``boot``).  :func:`lift` builds it from a
    rendered ``(nframes, 25)`` state; :func:`unlift` inverts it to renderable fits."""

    note_table: object  # list[int] | None
    pool: list  # deduped instrument structs: ["S",[name,struct]] | ["P",[...]]
    voices: list  # [VoiceTrack, VoiceTrack, VoiceTrack]
    globals: dict  # reg(21..24) -> [event, ...]
    nframes: int
    boot: list  # 25-register frame-0 seed


# ---------------------------------------------------------------------------
# An EVENT is a per-lane segment record (dur, ref, base, seed): dur the segment
# length in frames, ref into the shared pool (-1 = un-fit gap), base the pitch-factored
# note-table index (-1 = absolute / no pitch table; a list for piecewise), seed the
# per-note residue (a dict, or a list of per-piece dicts for piecewise).  A segment's
# START is NOT stored -- the cover tiles a lane contiguously from frame 0, so start is
# the running sum of prior durs.  A bundled voice row carries one (ref, base, seed) per
# lane at the row's shared dur; a free lane carries a plain event stream.
# ---------------------------------------------------------------------------
def _pool_builder():
    """A deduping pool: returns ``(ref_of, pool)`` where ``ref_of(entry, disc)`` interns a
    pool entry (a JSON-clean list) and returns its index (HARD RULE #0: a struct is
    stored ONCE; every reuse is a cheap reference).

    ``disc`` is a SEED-SCHEMA discriminator folded into the dedup key: two fits with an
    identical struct but a DIFFERENT seed-key SET (e.g. one note's ``citg`` carries an
    optional ``phase`` residue and another's does not -- both default-strip to the same
    struct) MUST NOT share a ref, because the serializer stores ONE seed schema per ref and
    decodes each row's bare seed values positionally against it.  Without the discriminator
    they collide and :func:`tracker_serialize._collect_schemas` raises "inconsistent seed
    schema".  Folding the schema into the key gives them distinct refs (pointing at
    identical struct entries -- a negligible, correct duplication), so every ref has ONE
    consistent schema."""
    import json

    pool, index = [], {}

    def ref_of(entry, disc=None):
        key = (json.dumps(entry, sort_keys=True), disc)
        ref = index.get(key)
        if ref is None:
            ref = len(pool)
            index[key] = ref
            pool.append(entry)
        return ref

    return ref_of, pool


def _seed_schema(seed):
    """The ordered seed-key tuple of one seed dict (``()`` for None/empty) -- the per-ref
    seed-schema discriminator (mirrors :func:`tracker_serialize._schema_of_seed`)."""
    return tuple(seed.keys()) if seed else ()


def _fit_to_pool(fit, is_freq, idx_of, ref_of):
    """Split a cover ``(name, params)`` fit (or None) into ``(ref, base, seed)``,
    interning the seed-and-pitch-invariant struct into the pool.  A ``piecewise``
    composite interns as a ``["P", ...]`` entry with per-piece bases/seeds (lists).

    The pool ref is discriminated by the fit's seed SCHEMA (the present SEED_KEYS), so two
    notes sharing a struct but differing in which optional residue keys they carry get
    distinct refs -- each with the single consistent schema the serializer requires."""
    if fit is None:
        return -1, -1, None
    name, params = fit
    if name == "piecewise":
        structs, seeds, bases = [], [], []
        for pname, pparams, plen in params["pieces"]:
            struct, seed, base = _split_fit(pname, pparams, is_freq, idx_of)
            structs.append([pname, struct, plen])
            seeds.append(seed)
            bases.append(base)
        disc = ("P", tuple(_seed_schema(s) for s in seeds))
        return ref_of(["P", structs], disc), bases, seeds
    struct, seed, base = _split_fit(name, params, is_freq, idx_of)
    return ref_of(["S", [name, struct]], ("S", _seed_schema(seed))), base, seed


def _freq_onset_values(cover):
    """The distinct POSITIVE freq values a freq cover visits as a held note or a
    pitched-table entry -- the candidate note-table (A440-grid) pitches.  A swept /
    accumulator body is left out (its table holds steps/wrapped values, not pitches);
    only a hold's value and a READ-mode table's values (the discrete notes a melody /
    arp visits) are pitches."""
    vals = set()
    for _s, _t, fit in cover:
        if fit is None:
            continue
        name, prm = fit
        if name == "hold":
            v = int(prm.get("value", 0))
            if v > 0:
                vals.add(v)
        elif name == "citg" and isinstance(prm, dict) and prm.get("mode") == "read":
            table = prm.get("table")
            if isinstance(table, list) and not (table and table[0] == _REL):
                for v in table:
                    if int(v) > 0:
                        vals.add(int(v))
    return vals


def _synth_note_table(freq_cover, min_notes=2):
    """Synthesize a note table from the distinct freq-onset values across all voices
    (sorted ascending so a constant note-index shift is a constant pitch shift -- what
    a tracker orderlist TRANSPOSE expresses).  Returns ``None`` when too few notes to
    be worth pitch-factoring (a constant / non-melodic tune keeps absolute holds)."""
    vals = set()
    for cov in freq_cover.values():
        vals |= _freq_onset_values(cov)
    if len(vals) < min_notes:
        return None
    return sorted(vals)


def _pitch_factor_freq(cover, idx_of):
    """Recode a freq cover's held notes as seedless note-table READs so the existing
    pitch-factoring (:func:`tracker._split_fit`) collapses every pitch to ONE struct.

    A ``("hold", {"value": V})`` with ``V`` on the note-table grid becomes the CITG
    ``("citg", {mode: read, table: [V], clock: every, loop: 0})`` -- byte-identical
    (a one-element table walked every frame IS a hold) but pitch-factorable: split
    rewrites ``table -> ["rel", 0]`` (pitch-invariant) with the note index riding the
    row as ``base``, and carries NO seed (the value is fully determined by the note
    table).  A rest (V<=0) or off-grid value stays an absolute hold."""
    out = []
    for s, t, fit in cover:
        if fit is not None and fit[0] == "hold":
            v = int(fit[1].get("value", 0))
            if v > 0 and idx_of.get(v) is not None:
                fit = (
                    "citg",
                    {
                        "mode": "read",
                        "table": [v],
                        "clock": {"kind": "every"},
                        "loop": 0,
                    },
                )
        out.append((s, t, fit))
    return out


def _events_of(cover, is_freq, idx_of, ref_of):
    """Turn one lane's cover ``[(start, stop, fit), ...]`` into an event stream
    ``[(dur, ref, base, seed), ...]``.  The cover tiles the lane contiguously from
    frame 0, so a segment's start is the running sum of the prior ``dur``\\ s -- it is
    NOT stored (the dominant per-event saving; the serializer reconstructs it)."""
    events = []
    for start, stop, fit in cover:
        ref, base, seed = _fit_to_pool(fit, is_freq, idx_of, ref_of)
        events.append((stop - start, ref, base, seed))
    return events


def _starts(cover):
    """The set of segment start frames of a cover (its note-onset boundaries)."""
    return {start for start, _stop, _fit in cover}


def _forced_key(v, cls):
    """The covers-dict key for a voice's sibling lane RE-SLICED at the freq spine."""
    return f"{v}:{cls}:forced"


def cover_all_lanes(state, note_table=None):
    """Cover all 16 register lanes (the slow Stage-1 step) ONCE.  Returns
    ``{lane_id: [(start, stop, fit), ...]}`` keyed ``"v:cls"`` (3 voices x freq/pw/
    ctrl/ad/sr) + ``"gN"`` (the four globals).  The freq lanes are covered first so
    each pw lane's cover can read the sibling freq carry (mirrors render_from_fits).

    For each voice it ALSO computes, under :func:`_forced_key`, a SPINE-ALIGNED cover
    of every sibling -- the sibling RE-SLICED with the freq-note onsets forced as
    mandatory breakpoints, so every sibling segment is contained inside ONE freq note.
    :func:`build_ir` folds these per-note segments into the voice's bundled rows (a note
    with more than one sibling segment becomes a piecewise row entry) and keeps the fold
    only where it is cheaper than the free stream.  A sibling that forcing fragments
    (many mid-note changes) then simply loses the cost test and stays free."""
    from preframr_tokens.bacc.generic import (
        cover as C,
    )  # local: avoids cover<->serialize cycle

    state = np.asarray(state, dtype=np.int64)
    nframes = int(state.shape[0])
    idx_of = _note_index_lookup(note_table)
    specs = {
        lid: (vals, width, is_freq, is_pw)
        for lid, vals, width, is_freq, is_pw in C._lane_specs(state)
    }
    covers = {}
    for v in range(3):
        vals = specs[f"{v}:{SPINE}"][0]
        covers[f"{v}:{SPINE}"] = C.cover_lane(
            vals, 0xFFFF, note_table, None, idx_of, True
        )
    for v in range(3):
        res = [(s, t, f) for (s, t, f) in covers[f"{v}:{SPINE}"]]
        carry = A.freq_carry_sequence(res, nframes)
        fstarts = _starts(covers[f"{v}:{SPINE}"])
        for cls in CLASSES:
            vals, width, _is_freq, is_pw = specs[f"{v}:{cls}"]
            cseq = carry if is_pw else None
            covers[f"{v}:{cls}"] = C.cover_lane(
                vals, width, note_table, cseq, idx_of, False
            )
            # The SPINE-ALIGNED re-slice: the sibling covered with the freq onsets forced
            # as breakpoints, so every segment falls inside one freq note (build_ir folds
            # it into the rows; the cost test there drops it when forcing fragmented it).
            covers[_forced_key(v, cls)] = C.cover_lane(
                vals, width, note_table, cseq, idx_of, False, force_breakpoints=fstarts
            )
    for reg in GLOBALS:
        covers[f"g{reg}"] = C.cover_lane(
            state[:, reg], 0xFF, note_table, None, idx_of, False
        )
    return covers


def _span_fits(covers, v, cls, fcov):
    """The SPINE-ALIGNED (1:1 with the freq notes) cover of voice ``v``'s sibling
    ``cls`` as ONE fit per freq note (the row entries it would fold into).

    The sibling's spine-aligned cover (``_forced_key``, every segment inside one note)
    is grouped by freq span: a note covered by a SINGLE sibling segment contributes that
    fit; a note covered by SEVERAL contributes a ``("piecewise", ...)`` of them (each
    piece carrying its frame length).  The freq spans tile the lane, so this yields
    exactly ``len(fcov)`` fits -- one per row.  ``cover_lane`` only ever emits ``citg``/
    ``hold`` pieces (no ``off``-phased ``tablewalk``), and a piecewise pw piece is handed
    the correctly-offset carry slice by the renderer, so folding is byte-exact (the
    serializer's render-equality self-check is the guard)."""
    forced = covers.get(_forced_key(v, cls))
    if forced is None:
        return None
    # Bucket the forced segments into freq spans (both tile the lane from 0 in order).
    spans = [(s, t) for s, t, _f in fcov]
    fits = []
    si = 0
    for s, t in spans:
        pieces = []
        while si < len(forced) and forced[si][0] < t:
            ps, pt, pfit = forced[si]
            if pfit is None:
                pieces = None  # an un-fit gap can't be a piecewise piece -> bail span
                break
            pieces.append((pfit[0], pfit[1], pt - ps))
            si += 1
        if not pieces:
            return None
        if len(pieces) == 1:
            name, prm, _plen = pieces[0]
            fits.append((name, prm))
        else:
            fits.append(("piecewise", {"pieces": pieces}))
    return fits


def _assemble_ir(
    freq_cover, span_fits, chosen, covers, note_table, idx_of, nframes, boot
):
    """Build a :class:`TrackerIR` for a fixed per-voice bundled-class CHOICE.

    ``chosen[v]`` is the list of sibling classes (in :data:`CLASSES` order) folded into
    voice ``v``'s rows; ``span_fits[(v, cls)]`` is that sibling's one-fit-per-note list
    (spine-aligned, possibly piecewise) for a folded sibling.  The pool is built fresh
    here (shared across voices/globals) so a struct introduced by a fold dedups with the
    rest -- the measurement in :func:`build_ir` serializes this exact IR."""
    ref_of, pool = _pool_builder()
    voices = []
    for v in range(3):
        fcov = freq_cover[v]
        bundled = [cls for cls in CLASSES if cls in chosen[v]]
        lane_fits = [[f for _s, _t, f in fcov]] + [
            span_fits[(v, cls)] for cls in bundled
        ]
        lane_isfreq = [True] + [False] * len(bundled)
        rows = []
        for i, (start, stop, _f) in enumerate(fcov):
            refs, bases, seeds = [], [], []
            for fits, isf in zip(lane_fits, lane_isfreq):
                ref, base, seed = _fit_to_pool(fits[i], isf, idx_of, ref_of)
                refs.append(ref)
                bases.append(base)
                seeds.append(seed)
            rows.append((stop - start, refs, bases, seeds))
        free = {
            cls: _events_of(covers[f"{v}:{cls}"], False, idx_of, ref_of)
            for cls in CLASSES
            if cls not in bundled
        }
        voices.append(VoiceTrack(bundled=bundled, rows=rows, free=free))
    globals_ = {
        reg: _events_of(covers[f"g{reg}"], False, idx_of, ref_of) for reg in GLOBALS
    }
    return TrackerIR(
        note_table=note_table,
        pool=pool,
        voices=voices,
        globals=globals_,
        nframes=nframes,
        boot=boot,
    )


def build_ir(covers, note_table, nframes, boot, synth_pitch=True):
    """Assemble the Tracker IR from per-lane ``covers`` (the fast step).

    Each voice's freq lane is the note spine.  A sibling lane (pw/ctrl/ad/sr)
    RE-SLICED at the freq-note onsets into one fit per note (:func:`_span_fits`, a
    multi-segment note becoming a piecewise) is a BUNDLE CANDIDATE; it is folded into
    the spine rows -- riding the shared ``(dt, dur)`` and its struct deduped in the
    shared pool -- only when that is CHEAPER than carrying it as its own free event
    stream (decided per (voice, sibling) by the serialized token count, the prompt's
    "whichever is fewer tokens").  A sibling forcing only fragments (so its row stream
    costs more than its free stream) stays free.

    When ``synth_pitch`` and the tune has no note table, one is synthesized from the
    freq-onset values and every held note recoded as a seedless note-table READ
    (pitch-factored, TRANSPOSE-able); every instrument struct dedups into the shared
    pool.  Pure structural re-expression -- :func:`unlift` inverts it byte-for-byte."""
    freq_cover = {v: covers[f"{v}:{SPINE}"] for v in range(3)}
    # PITCH FACTORING.  Synthesize a note table from the EXACT freq-onset values the
    # cover holds (the A440 12-TET grid the freq lane actually visits -- see
    # :mod:`pitch`) and recode every held note as a seedless note-table READ: a held
    # note then rides as a 1-token note-table ``base`` (not a 3-token absolute Fn) and
    # a phrase replayed at a different pitch TRANSPOSE-dedups (a constant base shift) --
    # the cross-driver alphabet.  A rest (V<=0) / off-grid value stays an absolute hold.
    if synth_pitch and note_table is None:
        note_table = _synth_note_table(freq_cover)
    idx_of = _note_index_lookup(note_table)
    if idx_of is not None:
        freq_cover = {
            v: _pitch_factor_freq(cov, idx_of) for v, cov in freq_cover.items()
        }

    # The spine-aligned one-fit-per-note list for every bundle-able (voice, sibling).
    span_fits = {}
    candidates = {v: [] for v in range(3)}
    for v in range(3):
        for cls in CLASSES:
            fits = _span_fits(covers, v, cls, freq_cover[v])
            if fits is not None:
                span_fits[(v, cls)] = fits
                candidates[v].append(cls)

    # Decide the bundled SET per voice by SERIALIZED token count.  Bundling is
    # SUPER-ADDITIVE: folding ONE sibling into the rows costs the per-note padding up
    # front, but the payoff is the whole note (freq + every co-moving sibling) becoming
    # ONE row that the row-stream REPEAT/TRANSPOSE collapses against a repeated/transposed
    # phrase -- a win that only appears once the co-moving siblings are bundled TOGETHER.
    # So per voice we evaluate every SUBSET of its bundle candidates (<=4, so <=16) and
    # keep the cheapest, holding the other voices fixed; coordinate-descent over the three
    # voices until none changes (the shared pool couples them).  A subset that only
    # fragments simply never beats all-free, so that sibling stays free.
    from preframr_tokens.bacc import tracker_serialize as TS

    def measure(ch):
        ir = _assemble_ir(
            freq_cover, span_fits, ch, covers, note_table, idx_of, nframes, boot
        )
        return len(TS._ir_to_ids(ir))

    chosen = {v: [] for v in range(3)}
    improved = True
    while improved:
        improved = False
        for v in range(3):
            cand = candidates[v]
            best_sub, best_tok = chosen[v], None
            for mask in range(1 << len(cand)):
                sub = [cand[k] for k in range(len(cand)) if mask & (1 << k)]
                trial = dict(chosen)
                trial[v] = sub
                tok = measure(trial)
                if best_tok is None or tok < best_tok:
                    best_tok, best_sub = tok, sub
            if set(best_sub) != set(chosen[v]):
                chosen[v] = best_sub
                improved = True

    return _assemble_ir(
        freq_cover, span_fits, chosen, covers, note_table, idx_of, nframes, boot
    )


def lift(state, note_table=None, nframes=None, boot=None, synth_pitch=True):
    """Lift a byte-exact per-frame ``(nframes, 25)`` ``state`` into the Tracker IR
    (covers every lane, bundles per voice, pitch-factors, dedups the pool).  Pure
    structural re-expression -- :func:`unlift` inverts it byte-for-byte."""
    state = np.asarray(state, dtype=np.int64)
    if nframes is None:
        nframes = int(state.shape[0])
    if boot is None:
        boot = [int(v) for v in state[0]]
    covers = cover_all_lanes(state, note_table)
    return build_ir(covers, note_table, nframes, boot, synth_pitch=synth_pitch)


# ---------------------------------------------------------------------------
# unlift: reconstruct the exact genfits/eventfits the per-register renderer consumes.
# ---------------------------------------------------------------------------
def _rebuild_seg(pool, note_table, start, dur, ref, base, seed):
    """Rebuild one cover segment ``(start, start+dur, fit)`` from a pool reference +
    the per-note ``base``/``seed`` (inverse of :func:`_fit_to_pool`)."""
    if ref < 0:
        return (start, start + dur, None)
    tag, body = pool[ref]
    if tag == "P":
        pieces = []
        for (name, struct, plen), pbase, pseed in zip(body, base, seed):
            fit = _rebuild_fit(name, struct, pseed, pbase, note_table)
            pieces.append((fit[0], fit[1], plen))
        return (start, start + dur, ("piecewise", {"pieces": pieces}))
    name, struct = body
    fit = _rebuild_fit(name, struct, seed, base, note_table)
    return (start, start + dur, fit)


def _events_to_segs(pool, note_table, events):
    """Rebuild a free lane's cover ``[(start, stop, fit), ...]`` from its events
    (start = the running sum of prior durs; the cover tiles the lane from frame 0)."""
    segs, start = [], 0
    for dur, ref, base, seed in events:
        segs.append(_rebuild_seg(pool, note_table, start, dur, ref, base, seed))
        start += dur
    return segs


def unlift(ir):
    """Inverse of :func:`lift`: reconstruct ``(genfits, eventfits)`` in the shape
    :func:`tracker.render_from_fits` consumes, byte-for-byte identical to the lane
    covers :func:`lift` started from.  The PW ``carry`` is recomputed from the freq
    fits at render (never stored)."""
    pool, note_table = ir.pool, ir.note_table
    genfits = {}
    eventfits = {}
    for v, track in enumerate(ir.voices):
        # Rebuild the spine + bundled lanes from the bundled rows (start = sum of durs).
        lane_classes = [SPINE] + list(track.bundled)
        lane_segs = {cls: [] for cls in lane_classes}
        start = 0
        for dur, refs, bases, seeds in track.rows:
            for cls, ref, base, seed in zip(lane_classes, refs, bases, seeds):
                lane_segs[cls].append(
                    _rebuild_seg(pool, note_table, start, dur, ref, base, seed)
                )
            start += dur
        # The free siblings: their own event streams.
        for cls, events in track.free.items():
            lane_segs[cls] = _events_to_segs(pool, note_table, events)
        # Place each lane into genfits (freq/pw) or eventfits (ctrl/ad/sr).
        genfits[f"{v}:{SPINE}"] = lane_segs[SPINE]
        for cls in CLASSES:
            segs = lane_segs.get(cls)
            if segs is None:  # a lane with no events (shouldn't happen) -> empty
                segs = []
            if cls in _GEN_CLASSES:
                genfits[f"{v}:{cls}"] = segs
            else:
                reg = _LANE_REG[cls][v]
                eventfits[reg] = [(s, t, f[0], f[1]) for s, t, f in segs]
    for reg in GLOBALS:
        segs = _events_to_segs(pool, note_table, ir.globals[reg])
        eventfits[reg] = [(s, t, f[0], f[1]) for s, t, f in segs]
    return genfits, eventfits


def render(ir):
    """Render the IR back to an ``(nframes, 25)`` state via the per-register renderer
    (proves the IR is lossless: equals the state :func:`lift` started from)."""
    genfits, eventfits = unlift(ir)
    return render_from_fits(genfits, eventfits, ir.note_table, ir.nframes)
