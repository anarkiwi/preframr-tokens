"""A serialize/deserialize/render codec for the recovered tracker STRUCTURE.

:mod:`structure_recover` recovers the BYTE-EXACT structure the output-fit generic path
misses -- one note table, a deduped instrument pool, the shared program tables (referenced
once), the ``(note, instr_ref, dur, cmd)`` pattern rows, the per-voice orderlists that
factor pattern reuse, and the porta/vibrato accumulator generators (STSQ) that flatten the
displaced "note table" to a handful of grid pitches (residual 0).  What was MISSING was a
REAL serialize/deserialize/render codec carrying that structure; :class:`StructureIR` is it.

``build_structure_ir`` assembles the recovered structure into a flat IR.
``structure_ir_to_ids`` serializes EVERYTHING needed to reconstruct it (note table,
instrument pool, shared programs, patterns, orderlists, accumulator generators, nframes,
boot) through the SHARED backward-LZ machinery -- the same greedy 3-gram backward match as
:func:`tracker_serialize._lz_tokens`, instantiated over the structure's VALUE alphabet
(:func:`_struct_lz`, ``_REPEAT`` reserved above any literal) so a value is one token (not two
base-16 nibbles) and a pattern reused across many orderlist slots -- or a phrase / sub-table
shared across patterns -- collapses to one stored copy.  ``structure_ir_from_ids`` is the
exact inverse; the codec invariant (every field round-trips EQUAL) is asserted, raising on
mismatch (HARD RULE #0).  The total is < 1 token/frame -- the recovered structured floor, vs
the shipping output-fit path's ~2 tok/frame on this fixture (pitch-factoring fails on the
porta-swept notes).

Compaction.  The ``patterns`` field is the decoded tuples, but they SERIALIZE as the
player's compact stateful byte stream (a marker byte only when instr/dur/cmd CHANGES) which
the same grammar re-decodes to the EXACT tuples on read.  Each accumulator stores its 16-bit
VALUE sequence (``lo | hi<<8``) + cell ``first_seen`` so render rebuilds the grid exactly as
``clean_pitches_residual`` does; the 16-bit values recur, so value-LZ collapses them.

PR4a render-from-tokens.  :func:`render_structure` reproduces the FULL 25-register trace
BYTE-EXACT (residual 0) from the DESERIALIZED IR ALONE -- ``_state = None``.  The FREQ pairs
render from the per-voice TOKEN-DERIVED note base (:func:`render_freq_from_tokens`): the
cheaper of a DIRECT 16-bit freq ramp generator (``ramp_segments_kernel`` -- a note glide /
porta / vibrato is a constant-step ramp), the player's note table + idx-walk ramp
(``note_base_recover``) + the porta/vibrato accumulators, or the value-LZ'd freq stream (the
backward-REPEAT collapse of a periodic vibrato cycle / repeated phrase).  The NON-FREQ
registers render from :func:`render_nonfreq_from_ir`: each admitted lane is a 16-bit PW-sweep
GENERATOR or an LZ-COLLAPSING / SPARSE change-point program (:func:`build_nonfreq_program`,
:func:`_lane_admissible` -- the player's sets-and-holds, the recurrence the shared
backward-LZ folds, never a per-frame literal dump); a constant lane is ``boot``.  When the
generator representation renders the whole trace token-alone byte-exact AND is < 1
token/frame, the pattern bank (note table / instruments / patterns / orderlists -- a second
encoding of the same notes) is DROPPED (:func:`_pick_representation`, HARD RULE #0: ship ONE
floor); a tune the generators do not yet fully recover (or where they exceed budget) keeps
the pattern bank + the ``_state``-anchored freq/lane path (the pre-PR4a stream, the
falsifiable SDDF-extension gap).  ``_state`` is NEVER serialized.
"""

from dataclasses import dataclass, field

import numpy as np

from preframr_tokens.bacc.generic.structure_recover import (
    ACC_RAW,
    _END_OF_PATTERN,
    _MARK_DUR,
    _MARK_INSTR,
    _grammar_eops,
    accumulator_generators,
    note_base_recover,
    recover_schedule,
    recover_structure,
)
from preframr_tokens.bacc.serialize import _u_len

NREG = 25
# clean_pitches_residual's three per-voice freq-register pairs; the accumulator grid is
# rebuilt EXACTLY as it does (per cell, start = first_seen + align, held thereafter).
_CPR_VOICES = ((0, 1), (7, 8), (14, 15))
# The FREQ register indices the IR renders from the accumulator program; everything else
# (pw/ctrl/ad/sr/filter/volume) is a NON-FREQ lane the M1 replay targets.
_FREQ_REGS = frozenset(r for pair in _CPR_VOICES for r in pair)
_NONFREQ_REGS = tuple(r for r in range(NREG) if r not in _FREQ_REGS)
# The three per-voice PW register pairs (lo, hi): a PW SWEEP is a free-running 16-bit
# accumulator the player advances per frame (``value += step``, reset/reparametrised at a
# note-gated boundary), so the two byte lanes are ONE generator, not two output dumps.
# Combining lo|hi<<8 and fitting a constant-step wrapping ramp (``ramp_segments_kernel``)
# DERIVES the sweep from a handful of per-segment ints -- never the dense change-point
# dump (the HARD RULE #0 literal-floor trap).
_PW_PAIRS = ((2, 3), (9, 10), (16, 17))
# Two admission caps -- DERIVING a generator vs STORING a change-point column are not the
# same act (HARD RULE #0), so they earn different budgets:
#
#  * a GENERATOR (a fitted 16-bit PW-sweep ramp) is a genuine derivation -- a thousands-
#    frame sweep collapses to a handful of per-segment ints -- so it earns a GENEROUS cap;
#    it is shipped whenever it is cheaper than the dense change-point dump it replaces.
#  * a CHANGE-POINT program is the player's literal sets-and-holds; a SPARSE one (a couple
#    of segments: filter/volume setup, an AD/SR latched once and held) is the real program
#    and is admitted under a TIGHT cap, but a DENSE one (hundreds of change points) is the
#    HARD RULE #0 literal-floor trap (storing the output column) and is REJECTED -- that
#    lane stays on the byte-exact anchor (the falsifiable SDDF-extension gap: its free-
#    running per-frame cursor + instrument table is not yet in the artifact).
#
# This keeps the budget at the pre-M1 structured floor (a dense un-derivable lane ships
# NOTHING) while the generator-fit DERIVES the PW sweeps the change-point dump bloated.
#
# PR4a admission (render-from-tokens-alone): a non-freq lane is the player's INSTRUMENT
# PROGRAM -- it fires a sets-and-holds (or a PW sweep) from each note-onset, so its
# change-point program RECURS across the song (the instrument is shared, the phrase
# repeats).  That recurrence is the HARD RULE #0 structure: the shared backward-LZ
# (``_struct_lz``, over the whole concatenated non-freq section) COLLAPSES it (measured:
# a 567-change-point ctrl lane -> 102 tokens, a 1027-change-point lane -> 36 tokens).  A
# lane is admitted only when its LZ'd program is a genuine COLLAPSE of its raw
# change-point stream (``_lz_collapses``): the LZ shrank it well below the raw
# ``nseg``-pair floor, proving recovered recurrence, NOT a per-frame literal dump.  A lane
# the LZ does NOT collapse (a truly non-recurrent per-frame column) is rejected and stays
# on the anchor (the falsifiable gap) -- never serialized as its dense output.
_LZ_COLLAPSE_RATIO = 0.6  # a DENSE CP lane is admitted when LZ < ratio * its raw cost
_NONFREQ_SPARSE_SEG = 32  # a SPARSE CP lane (<= this many change points) is the real
# player program (filter / volume / cutoff setup, an AD/SR latched a handful of times) and
# is always admitted -- it is sets-and-holds, never a per-frame dump.

# The typed non-freq lane RECORD kinds (the serialized ``nonfreq`` section is a list of
# these, each self-describing so the codec round-trips and the renderer dispatches on the
# tag): a change-point program (the player's sets-and-holds) or a 16-bit ramp-segment
# generator (a PW sweep, covering BOTH the lo and hi byte lanes from one generator).
_LANE_CP = (
    0  # ("cp", reg, starts, values)            -- piecewise-constant lane program
)
_LANE_RAMP16 = (
    1  # ("ramp16", lo, hi, starts, seeds, steps) -- 16-bit PW sweep generator
)
_PW_MODULUS = (
    1 << 16
)  # the PW accumulator wrap (the lo|hi<<8 combine is a 16-bit value)

# PR4a note-base record kinds (the per-voice FREQ base, chosen by the cheaper byte-exact
# encoding so the freq lanes render from tokens at the structured floor):
#  * _NB_RAMP16 -- the voice's 16-bit freq column DIRECTLY as a constant-step wrapping-ramp
#    GENERATOR (``ramp_segments_kernel`` over ``lo|hi<<8``).  A note-level GLIDE / porta /
#    vibrato is a constant-step ramp, so the whole displaced "note table" (hundreds of
#    swept Fn values) collapses to a handful of per-segment ints -- the recovery the
#    note-table idx-walk MISSES when the porta is not a reset-to-0 STSQ accumulator
#    (HARD RULE #0: a glide is a generator, not a hundreds-entry table).  Carries the full
#    base (porta + vibrato + note), so NO accfit overlay is added.
#  * _NB_TABLE -- the player's own note table + the idx-walk ramp (``note_base_recover``):
#    cheaper when the voice plays a small set of held grid pitches (no per-frame glide), so
#    the distinct pitches factor into a short table the idx walk indexes; the accfit
#    porta/vibrato overlay is summed on top.
#  * _NB_RAWLZ -- the voice's 16-bit freq column as a VALUE stream collapsed by the shared
#    backward-LZ (``_struct_lz``).  This is the backward-REPEAT recovery the encoder must
#    collapse (HARD RULE #0 (b)): a periodic VIBRATO cycle (the 8-frame triangle around a
#    carrier) and a repeated note PHRASE recur as identical value blocks the LZ folds to one
#    stored copy + copies -- the recovery the constant-step ramp misses when the vibrato
#    amplitude varies per note (so the ramp seeds do not factor, but the value cycle does).
#    NOT a literal-floor dump: it is admitted only when the LZ collapse beats the ramp /
#    table generators, i.e. the recurrence genuinely compressed it.  Carries the whole base.
_NB_RAMP16 = 0  # (kind, starts, seeds, steps) -- direct 16-bit freq ramp generator
_NB_TABLE = 1  # (kind, note_table, starts, seeds, steps, modulus, a, warmup)
_NB_RAWLZ = (
    2  # (kind, freq) -- the 16-bit freq value stream, value-LZ'd (REPEAT collapse)
)

# The structure-path token alphabet is the non-negative ints; REPEAT is a reserved
# sentinel strictly above every literal the IR emits (counts, addresses < 2^17, 16-bit
# accumulator values, pattern/program bytes), so a copy is unambiguous.
_REPEAT = 1 << 24
_MIN_COPY = 3  # a copy costs REPEAT + off + len (>= 3 tokens); break even at 3

# Optional TRAILING sections are TAGGED (a reserved sentinel above every literal, like
# _REPEAT) so the codec stays BACK-COMPATIBLE: a pre-PR4a stream ends after ``accfits``
# (no trailing tag), and a newer stream prepends a one-token tag per present trailing
# section.  The reader loops on the tag (``_SEC_END`` / EOF terminates), so adding a
# section never shifts an older stream's layout.  ``_SEC_NOTE_BASES`` is the PR4a
# token-derived freq base; ``_SEC_NONFREQ`` the M1 non-freq lane program.
_SEC_END = (1 << 24) + 1
_SEC_NOTE_BASES = (1 << 24) + 2
_SEC_NONFREQ = (1 << 24) + 3


@dataclass
class StructureIR:
    """The recovered tracker structure as a serializable IR (every field DERIVED from the
    distill artifact, no hardcoded addresses).

    ``instr_pool`` is the deduped list of 8-int instrument structs; ``shared_programs`` is
    the concatenation of the shared program-table span bytes (referenced once, not
    re-derived per note); ``patterns`` are the ``(note, instr_ref, dur, cmd)`` rows (any
    field may be ``None`` -- the player's running state before its first marker), serialized
    as the player's compact stateful byte stream via ``pattern_bytes``; ``orderlists`` are
    the per-voice pattern-index + control-marker streams that factor pattern reuse;
    ``accfits`` carry, per voice, the FITTED freq-accumulator generators -- one
    ``(first_seen, kind, seed, p1, p2, p3, n, raw)`` per chosen accumulator (the
    AGENTS.md accumulator-fit: porta=ramp/quadratic, vibrato=triangle; a stored ramp
    is unrecovered structure).  ``kind`` is one of ``structure_recover.ACC_*``; a
    ramp/quadratic/triangle fit amortises to a handful of ints, while ``ACC_RAW``
    keeps the 16-bit ``raw`` sequence verbatim (the honest, surfaced fallback when no
    closed-form generator reproduces the captured window byte-exact).  ``_state`` is
    the byte-exact ``(nframes, 25)`` register array (the M0 render anchor), NEVER
    serialized."""

    note_table: list = field(default_factory=list)
    instr_pool: list = field(default_factory=list)
    shared_programs: list = field(default_factory=list)
    patterns: list = field(default_factory=list)
    pattern_bytes: list = field(default_factory=list)  # per-pattern player byte stream
    orderlists: list = field(default_factory=list)
    accfits: list = field(
        default_factory=list
    )  # per voice: [(fs,kind,seed,p1,p2,p3,n,raw), ...]
    # PR4a: the per-voice TOKEN-DERIVED NOTE-PITCH BASE (``structure_recover.note_base_recover``)
    # so the FREQ lanes render WITHOUT reading ``_state``.  Per voice a record
    # ``(note_table, starts, seeds, steps, modulus, a, warmup)``: ``note_table`` is the
    # player's OWN distinct grid-pitch (Fn) values, ``(starts, seeds, steps, modulus)`` is
    # the byte-exact idx-walk ramp generator (``ramp_render_kernel`` reproduces the per-frame
    # note-table-index walk, residual 0), ``a`` is the analysis warm-up offset (the base is
    # token-derived over ``[a, nframes)``) and ``warmup`` is the verbatim 16-bit freq for the
    # ``[0, a)`` warm-up frames (a handful of ints, NOT a per-frame dump).  ``base_freq`` ==
    # ``note_table[ramp]`` placed piecewise-const + the accumulators reproduces the freq
    # column byte-exact; an empty list keeps the old ``_state``-seeded render path.
    note_bases: list = field(default_factory=list)
    # M1 non-freq lane recovery: per ADMITTED non-freq register, the CHEAPEST byte-exact
    # encoding as a TYPED record -- either a change-point program
    # ``(_LANE_CP, reg, starts, values)`` (the player's sets-and-holds, rendered by
    # ``_discover_njit.step_lane_kernel``) or a 16-bit PW-sweep GENERATOR
    # ``(_LANE_RAMP16, lo, hi, starts, seeds, steps)`` (the constant-step wrapping-ramp
    # accumulator, ``ramp_render_kernel``, covering BOTH byte lanes).  Only lanes under the
    # token cap are admitted (the literal-floor guard); a lane neither generator- nor
    # cheaply-change-point-derivable is OMITTED and renders from the ``_state`` anchor (the
    # SDDF-extension gap).  An empty list serialises byte-identically to the pre-M1 stream.
    nonfreq: list = field(
        default_factory=list
    )  # typed lane records (see build_nonfreq_program)
    nframes: int = 0
    boot: list = field(default_factory=list)
    # The recovered note->frame SCHEDULE + tempo (PWLK orderlist advances over the FULL
    # run, PR0): ``{"onsets","durations","tempo","n_onsets","span","nframes"}`` from
    # :func:`structure_recover.recover_schedule`, or ``None`` when the artifact has no
    # orderlist walk.  A DERIVED analysis field (the tempo-event partition of playback,
    # ``sum(durations) == nframes``); it does NOT yet drive the freq token stream (that is
    # the IWLK walk-index PR), so -- like ``_state`` -- it is NOT serialized and the token
    # stream / alphabet are byte-identical with or without it.
    schedule: object = None
    _state: object = (
        None  # byte-exact (nframes, 25); the anchor for the un-replayed lanes, not serialized
    )


# --- the player pattern grammar (mirrors structure_recover.decode_patterns) ----
def _decode_pattern_bytes(pb):
    """Decode one pattern's player byte stream to ``(note, instr, dur, cmd)`` rows, the
    SAME stateful grammar :func:`structure_recover.decode_patterns` uses -- so the bytes
    stored by :func:`build_structure_ir` re-decode to the EXACT recovered tuples."""
    rows = []
    cur_instr = cur_dur = cur_cmd = None
    for b in pb:
        if b >= 0x80:
            if (b & 0xE0) == _MARK_DUR:
                cur_dur = b & 0x1F
            elif (b & 0xE0) == _MARK_INSTR:
                cur_instr = b & 0x1F
            else:
                cur_cmd = b & 0x3F
            continue
        if b == _END_OF_PATTERN:
            break
        rows.append((b, cur_instr, cur_dur, cur_cmd))
    return rows


def _pattern_byte_stream(ram, base, eops=(_END_OF_PATTERN,), length=None):
    """The raw player byte stream for one pattern.

    ``length`` (the read-coverage extent, the nibble / bit-packed dialects) takes the
    pattern to be exactly that many observed-read bytes; otherwise it runs up to and
    including the first EOP byte (the value-range dialects)."""
    if length is not None:
        return [int(ram[(base + k) & 0xFFFF]) for k in range(length)]
    idx, pb = 0, []
    eops = set(eops)
    while idx < 0x400:
        b = int(ram[(base + idx) & 0xFFFF])
        pb.append(b)
        idx += 1
        if b in eops:
            break
    return pb


# --- M1 non-freq lane recovery (the instrument-driven ctrl/ad/sr/pw/filter lanes) --
def _lane_change_program(col):
    """The piecewise-constant change-point program ``(starts, values)`` of one register
    column (``int64[nframes]``): ``starts`` are the frames the value changes (incl. 0),
    ``values`` the held byte at each step.  ``_discover_njit.step_lane_kernel`` re-renders
    it byte-exact -- the inverse encode.  This is the lane's REAL program (the player sets
    the register at a note-on / sweep step and holds it), not a per-frame output dump.
    """
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    col = np.asarray(col, dtype=np.int64)
    starts = DJ.change_points_kernel(col)
    values = col[starts] if len(starts) else col[:0]
    return [int(s) for s in starts], [int(v) for v in values]


def _lane_program_tokens(starts, values):
    """The serialized token cost of one CHANGE-POINT lane program (the cp admission
    metric): ``nseg`` + the start-DELTA stream + the value stream, each value-LZ'd.
    Mirrors the bytes a ``_LANE_CP`` record emits so the cap reflects the true shipped
    size (the cheapest-encoding pick and the budget gate read the same number)."""
    flat = [len(starts)]
    prev = 0
    for s in starts:
        flat.append(s - prev)
        prev = s
    flat.extend(values)
    out = []
    _emit_section(out, flat)
    return len(out)


def _ramp16_fit(lo_col, hi_col):
    """Fit the 16-bit PW pair ``lo | hi<<8`` to a constant-step WRAPPING-RAMP generator
    (``_discover_njit.ramp_segments_kernel``): the per-segment ``(start, seed, step)`` of
    the free-running accumulator the player advances per frame.  Returns
    ``(starts, seeds, steps)`` (Python int lists).  The sweep is DERIVED from these few
    ints; :func:`_ramp16_render` (``ramp_render_kernel``) is the byte-exact inverse."""
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    pw = np.asarray(lo_col, dtype=np.int64) | (np.asarray(hi_col, dtype=np.int64) << 8)
    starts, seeds, steps, _n = DJ.ramp_segments_kernel(pw, _PW_MODULUS)
    return (
        [int(s) for s in starts],
        [int(s) for s in seeds],
        [int(s) for s in steps],
    )


def _ramp16_tokens(starts, seeds, steps):
    """The serialized token cost of one ``_LANE_RAMP16`` generator record (the ramp
    admission metric): ``nseg`` + the start-DELTA stream + the seed stream + the step
    stream, each value-LZ'd.  Mirrors the bytes :func:`_flat_nonfreq` emits for a
    ``_LANE_RAMP16`` so the cheapest-encoding pick and the budget gate agree."""
    bd = [len(starts)]
    prev = 0
    for s in starts:
        bd.append(s - prev)
        prev = s
    flat = bd + list(seeds) + list(steps)
    out = []
    _emit_section(out, flat)
    return len(out)


def _lane_admissible(starts, values):
    """True iff the LANE's change-point program is RECOVERED STRUCTURE -- either SPARSE (a
    handful of change points: filter/volume/cutoff setup, an AD/SR latched and held -- the
    player's real sets-and-holds, never a dump), or LZ-COLLAPSING (a denser program whose
    value-LZ'd cost falls well below its raw ``nseg``-pair floor: the same instrument
    sets-and-holds fired from each note-onset RECURS, the shared backward-LZ folds it).
    This is the HARD RULE #0 falsification: a recurrent / sparse program is admitted; a
    truly non-recurrent dense per-frame column collapses NEITHER way and is rejected
    (stays on the anchor, never serialized as its dense output)."""
    nseg = len(starts)
    if nseg <= _NONFREQ_SPARSE_SEG:
        return True  # sparse sets-and-holds (the real program), always admitted
    raw = 1 + 2 * nseg  # nseg + the start-deltas + the values (the un-LZ'd floor)
    return _lane_program_tokens(starts, values) < _LZ_COLLAPSE_RATIO * raw


def build_nonfreq_program(state):
    """The M1 non-freq lane recovery (PR4a: render-from-tokens-alone): per non-freq
    register the CHEAPEST byte-exact encoding, in derivation order (the PW-sweep generator
    before the change-point program).  Every non-constant lane that recovers as a genuine
    structure -- a generator or an LZ-COLLAPSING change-point program -- is admitted, so the
    full 25-register trace renders from the ids alone (no ``_state`` anchor).  Returns a
    typed lane-record list:

      * ``(_LANE_RAMP16, lo, hi, starts, seeds, steps)`` -- a PW sweep DERIVED as a 16-bit
        constant-step wrapping-ramp GENERATOR (covering BOTH byte lanes from one generator),
        admitted when the ramp fit is cheaper than the two byte lanes' change-point cost (a
        sweep is an accumulator, not two output columns -- HARD RULE #0); else
      * ``(_LANE_CP, reg, starts, values)`` -- the lane's piecewise-constant change-point
        program (the player's real sets-and-holds), admitted when it is RECOVERED STRUCTURE
        (:func:`_lane_admissible`: SPARSE sets-and-holds, or a denser program that
        LZ-COLLAPSES because the instrument program RECURS -- the shared backward-LZ folds
        it well below the raw change-point floor; not a per-frame literal dump).

    A constant lane is omitted (``boot`` supplies it).  A lane whose change-point program
    does NOT LZ-collapse (a truly non-recurrent per-frame column) is rejected and stays on
    the ``_state`` anchor -- the falsifiable gap, never its dense output (the literal-floor
    trap).  On the corpus EVERY non-constant lane collapses, so the trace is fully
    token-rendered."""
    if state is None:
        return []
    state = np.asarray(state, dtype=np.int64)
    program = []
    covered = set()
    # (b) GENERATOR-FIT the PW sweeps first: when the 16-bit ramp generator beats the two
    # byte lanes' change-point cost, ship ONE generator for the pair (a derivation, the
    # sweep is an accumulator -- not two output columns).
    for lo, hi in _PW_PAIRS:
        starts, seeds, steps = _ramp16_fit(state[:, lo], state[:, hi])
        gen_cost = _ramp16_tokens(starts, seeds, steps)
        cp_cost = _lane_program_tokens(
            *_lane_change_program(state[:, lo])
        ) + _lane_program_tokens(*_lane_change_program(state[:, hi]))
        if gen_cost < cp_cost:
            program.append((_LANE_RAMP16, lo, hi, starts, seeds, steps))
            covered.add(lo)
            covered.add(hi)
    # (c) CHANGE-POINT the remaining lanes (and any PW lane the generator did not win),
    # admitted when the sets-and-holds program LZ-COLLAPSES (the recurrence is recovered
    # structure); a constant lane needs nothing (boot), a non-collapsing one stays anchored.
    for reg in _NONFREQ_REGS:
        if reg in covered:
            continue
        starts, values = _lane_change_program(state[:, reg])
        if len(values) <= 1:
            continue  # constant lane -- boot supplies it
        if _lane_admissible(starts, values):
            program.append((_LANE_CP, reg, starts, values))
    return program


def _nb_table_tokens(tbl, starts, seeds, steps, modulus, a, warmup):
    """The serialized token cost of one ``_NB_TABLE`` note-base record (note table + the
    idx-walk ramp + modulus/a/warmup), so :func:`_build_note_bases` can pick the cheaper
    encoding per voice.  Mirrors the bytes :func:`_flat_note_bases` emits."""
    flat = [_NB_TABLE, len(tbl), *tbl, len(starts)]
    _emit_start_deltas(flat, starts)
    flat += list(seeds) + list(steps) + [modulus, a, len(warmup), *warmup]
    out = []
    _emit_section(out, flat)
    return len(out)


def _nb_ramp16_tokens(starts, seeds, steps):
    """The serialized token cost of one ``_NB_RAMP16`` note-base record (a direct 16-bit
    freq ramp), the cheaper-encoding metric mirrored from :func:`_flat_note_bases`."""
    flat = [_NB_RAMP16, len(starts)]
    _emit_start_deltas(flat, starts)
    flat += list(seeds) + list(steps)
    out = []
    _emit_section(out, flat)
    return len(out)


def _nb_rawlz_tokens(freq):
    """The serialized token cost of one ``_NB_RAWLZ`` note-base record (the value-LZ'd freq
    value stream), the cheaper-encoding metric mirrored from :func:`_flat_note_bases`.
    """
    flat = [_NB_RAWLZ, len(freq), *freq]
    out = []
    _emit_section(out, flat)
    return len(out)


def _build_note_bases(distill_path, state):
    """The per-voice TOKEN-DERIVED FREQ BASE (PR4a) so the FREQ lanes render WITHOUT reading
    ``_state``.  Per voice the CHEAPER byte-exact encoding is chosen (HARD RULE #0: the
    structured floor, not whichever path the artifact happened to seed):

      * ``(_NB_RAMP16, starts, seeds, steps)`` -- the voice's 16-bit freq column fitted
        DIRECTLY as a constant-step wrapping-ramp generator (``ramp_segments_kernel``).  A
        note-level glide / porta / vibrato is a constant-step ramp, so the displaced "note
        table" (hundreds of swept Fn values) collapses to a handful of per-segment ints --
        the recovery the note-table idx-walk misses when the porta is not a reset-to-0 STSQ
        accumulator.  Carries the WHOLE base; no accfit overlay is added.
      * ``(_NB_TABLE, note_table, starts, seeds, steps, modulus, a, warmup)`` -- the
        player's own note table + the byte-exact idx-walk ramp (:func:`note_base_recover`),
        chosen when a small held-pitch set makes the table + idx walk cheaper than the
        direct ramp; the accfit porta/vibrato overlay is summed on top at render.

    Both render byte-exact (residual 0 -- verified); a non-zero idx-walk residual raises
    (a missing generator, never a stored dense walk).  Empty when the artifact has no STSQ
    section (the render falls back to the ``_state`` seed)."""
    if state is None:
        return []
    rec = note_base_recover(distill_path, state)
    if rec is None:
        return []
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    state = np.asarray(state, dtype=np.int64)
    n = state.shape[0]
    out = []
    for vi, (rlo, rhi) in enumerate(_CPR_VOICES):
        info = rec[vi]
        if int(info["idx_residual"]) != 0:
            raise ValueError(
                f"structure_ir: voice {vi} idx-walk ramp residual "
                f"{info['idx_residual']} != 0 (a missing note-base generator, HARD RULE #0)"
            )
        freq = state[:, rlo] | (state[:, rhi] << 8)
        # (a) the DIRECT 16-bit freq ramp (the glide-as-generator recovery).
        rs, rse, rst, _nseg = DJ.ramp_segments_kernel(freq, _PW_MODULUS)
        rs, rse, rst = (
            [int(x) for x in rs],
            [int(x) for x in rse],
            [int(x) for x in rst],
        )
        ramp_rec = (_NB_RAMP16, rs, rse, rst)
        ramp_cost = _nb_ramp16_tokens(rs, rse, rst)
        # (b) the player's note table + idx-walk ramp.
        ts, tse, tst, modulus = info["ramp"]
        a = n - len(info["idx_rendered"])
        warmup = [int(v) for v in freq[:a]]
        tbl = [int(v) for v in info["note_table"]]
        ts, tse, tst = (
            [int(x) for x in ts],
            [int(x) for x in tse],
            [int(x) for x in tst],
        )
        table_rec = (_NB_TABLE, tbl, ts, tse, tst, int(modulus), int(a), warmup)
        table_cost = _nb_table_tokens(tbl, ts, tse, tst, int(modulus), int(a), warmup)
        # (c) the raw freq VALUE stream, value-LZ'd (the vibrato/phrase REPEAT collapse).
        freq_list = [int(v) for v in freq]
        rawlz_rec = (_NB_RAWLZ, freq_list)
        rawlz_cost = _nb_rawlz_tokens(freq_list)
        # pick the CHEAPEST byte-exact encoding (the structured floor, HARD RULE #0).
        choice = min(
            (ramp_cost, ramp_rec),
            (table_cost, table_rec),
            (rawlz_cost, rawlz_rec),
            key=lambda c: c[0],
        )
        out.append(choice[1])
    return out


# --- assembly from a RecoveredStructure + the byte-exact state ----------------
def build_structure_ir(struct, state, distill_path):
    """Assemble a :class:`StructureIR` from a :class:`RecoveredStructure`, the byte-exact
    ``(nframes, 25)`` ``state``, and the distill artifact.

    Dedup the instrument records into ``instr_pool``; pull the shared program bytes from
    ``struct.program_spans`` (concatenate ``ram[lo:hi]`` per span); store the per-pattern
    player byte streams (``pattern_bytes``) -- the compact serialized form -- alongside the
    decoded tuples (``patterns``); build ``accgens`` from :func:`clean_pitches_residual` (the
    chosen accumulator ``(lo, hi)`` pairs per voice) plus :func:`read_stsq_cells` (each
    referenced cell's ``(first_seen, samples)``)."""
    note_table = [int(v) for v in struct.note_table]

    seen, instr_pool = set(), []
    for rec in struct.instr_records:
        key = tuple(int(x) for x in rec)
        if key not in seen:
            seen.add(key)
            instr_pool.append([int(x) for x in rec])

    shared_programs = []
    ram = struct.ram
    if ram is not None:
        for _name, (lo, hi) in struct.program_spans.items():
            shared_programs.extend(int(x) for x in ram[lo:hi])

    eops = _grammar_eops(getattr(struct, "grammar", None))
    pat_src = getattr(struct, "pattern_src", None) or struct.pattern_ptrs
    pat_lens = getattr(struct, "pattern_lens", None) or []
    len_by_base = {int(a): int(l) for a, l in zip(pat_src, pat_lens)}
    pattern_bytes = (
        [
            _pattern_byte_stream(ram, base, eops, length=len_by_base.get(int(base)))
            for base in pat_src
        ]
        if ram is not None
        else []
    )
    # The tuple field IS the re-decode of those exact bytes (byte-exact to struct.patterns,
    # since that is what struct.patterns was decoded from) -- so the stored compact byte form
    # and the tuple field agree, and the field round-trips through serialize/deserialize.
    patterns = [_decode_pattern_bytes(pb) for pb in pattern_bytes]

    orderlists = [[int(b) for b in o] for o in struct.orderlists]

    accfits = [[] for _ in _CPR_VOICES]
    gens = accumulator_generators(distill_path, state) if state is not None else None
    if gens is not None:
        for vi in range(len(_CPR_VOICES)):
            accfits[vi] = [
                (
                    int(fs),
                    int(kind),
                    int(seed),
                    int(p1),
                    int(p2),
                    int(p3),
                    int(n),
                    None if raw is None else [int(v) for v in raw],
                )
                for (fs, kind, seed, p1, p2, p3, n, raw) in gens.get(vi, [])
            ]

    note_bases = _build_note_bases(distill_path, state)

    nframes = int(struct.nframes) if state is None else int(np.asarray(state).shape[0])
    schedule = recover_schedule(distill_path, nframes=nframes)

    ir = StructureIR(
        note_bases=note_bases,
        note_table=note_table,
        instr_pool=instr_pool,
        shared_programs=shared_programs,
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=orderlists,
        accfits=accfits,
        schedule=schedule,
        nonfreq=build_nonfreq_program(state),
        # The TRUE playback length is the ``.sidwr`` row count (``state`` rows), not the
        # artifact's REQUESTED ``nframes`` (the capture may stop early); the M1 lane replay
        # renders to exactly this length, and it is the metric denominator the tests use.
        nframes=nframes,
        boot=([0] * NREG if state is None else [int(state[0, r]) for r in range(NREG)]),
        _state=state,
    )
    return _pick_representation(ir, state)


def _renders_from_tokens(ir, state):
    """True iff ``render_structure`` reproduces ``state`` byte-exact from the IR ALONE
    (``_state`` ignored) -- the token-render-complete proof."""
    saved = ir._state
    ir._state = None
    try:
        rendered = render_structure(ir)
        exact = rendered.shape == state.shape and bool(
            np.array_equal(rendered, np.asarray(state, dtype=np.int64))
        )
    except (ValueError, IndexError):
        exact = False
    ir._state = saved
    return exact


def _clear_pattern_bank(ir):
    """Clear the pattern-bank fields (the higher-altitude note/instrument encoding) in
    place -- the GENERATOR representation supersedes it."""
    ir.note_table = []
    ir.instr_pool = []
    ir.shared_programs = []
    ir.patterns = []
    ir.pattern_bytes = []
    ir.orderlists = []
    return ir


def _pick_representation(ir, state):
    """PR4a: choose the codec's SHIPPED representation (HARD RULE #0: ship ONE encoding of
    the tune, the SMALLER byte-exact floor -- never two).

    Two candidates render the tune byte-exact:
      * the GENERATOR representation -- the token-derived note base + the recovered non-freq
        lanes -- which renders the WHOLE trace FROM TOKENS (``_state`` ignored), so the
        pattern bank (note table, instrument pool, shared programs, patterns, orderlists) is
        a redundant SECOND encoding and is dropped; admissible only when it renders
        token-alone byte-exact (:func:`_renders_from_tokens`); else
      * the PATTERN-BANK representation -- the pattern/orderlist factoring + the
        ``_state``-anchored freq/lane path (the pre-PR4a stream), dropping the note base.

    The GENERATOR representation is the GOAL (it renders the whole trace from tokens, no
    ``_state``); we ship it whenever it renders token-alone byte-exact AND is under the
    structured-floor budget (< 1 token/frame).  The pattern-bank path does NOT render freq
    from tokens (it reads ``_state``), so it is the FALLBACK only when the generator cannot
    token-render or would exceed the budget (a tune whose per-frame freq/lane structure is
    not yet fully factored -- the falsifiable gap, kept honest by the pattern factoring +
    anchor).  The un-chosen fields remain in the in-memory IR for analysis only."""
    if state is None or not ir.note_bases:
        return ir
    if not _renders_from_tokens(ir, state):
        # generators incomplete -> the anchored pattern-bank path: drop the freq base AND
        # the non-freq lanes (the anchor renders every non-freq register), the pre-PR4a
        # stream.
        ir.note_bases = []
        ir.nonfreq = []
        return ir
    gen_ir = StructureIR(
        note_bases=ir.note_bases,
        accfits=ir.accfits,
        nonfreq=ir.nonfreq,
        nframes=ir.nframes,
        boot=ir.boot,
    )
    gen_tokens = len(structure_ir_to_ids(gen_ir))
    nf = ir.nframes or 1
    if gen_tokens / nf < 1.0:
        return _clear_pattern_bank(ir)  # the token render at the structured floor
    # Over budget -> the anchored pattern-bank fallback (the pre-PR4a stream): drop the freq
    # base AND the non-freq lanes (the ``_state`` anchor renders every non-freq register, so
    # shipping the lane programs would be COST the anchor already covers -- the bank path
    # never claimed a token render).  Byte-identical in size to / cheaper than pre-PR4a.
    ir.note_bases = []
    ir.nonfreq = []
    return ir


# --- the shared backward-LZ, over the structure VALUE alphabet -----------------
def _struct_lz(values):
    """Backward-LZ a VALUE list into a token list: a literal is the value itself (one
    token), a copy is ``[_REPEAT, off, len]`` over prior values.  Identical greedy
    3-gram backward match as :func:`tracker_serialize._lz_tokens`, but the literal is a
    whole value (not two base-16 nibbles), so a pattern / sub-table reused later collapses
    to one copy at value granularity (the orderlist + pattern-reuse win)."""
    out, n = [], len(values)
    table, i = {}, 0
    while i < n:
        best_len, best_off = 0, 0
        if i + 3 <= n:
            key = (values[i], values[i + 1], values[i + 2])
            for pos in reversed(table.get(key, ())):
                length = 0
                while i + length < n and values[pos + length] == values[i + length]:
                    length += 1
                    if length >= 4095:
                        break
                if length > best_len:
                    best_len, best_off = length, i - pos
                if best_len >= 512:
                    break
        cost_copy = 1 + _u_len(best_off) + _u_len(best_len)
        if best_len >= _MIN_COPY and cost_copy < best_len:
            out += [_REPEAT, best_off, best_len]
            step = best_len
        else:
            out.append(values[i])
            step = 1
        for j in range(i, min(i + step, n - 2)):
            table.setdefault((values[j], values[j + 1], values[j + 2]), []).append(j)
        i += step
    return out


def _struct_unlz(ids, i, ncount):
    """Inverse of :func:`_struct_lz`: rebuild ``ncount`` values, returning ``(values, i)``."""
    values = []
    while len(values) < ncount:
        tok = ids[i]
        if tok == _REPEAT:
            off, length = ids[i + 1], ids[i + 2]
            i += 3
            base = len(values)
            for j in range(length):
                values.append(values[base - off + j])
        else:
            values.append(tok)
            i += 1
    return values, i


def _emit_section(out, values):
    """Append a section: its value count, then the value-LZ'd stream."""
    out.append(len(values))
    out.extend(_struct_lz(values))


def _read_section(ids, i):
    return _struct_unlz(ids, i + 1, ids[i])


# --- per-section flatten / parse (each section is a flat VALUE list) -----------
def _flat_note_table(note_table):
    return [len(note_table), *note_table]


def _flat_instr_pool(pool):
    out = [len(pool)]
    for rec in pool:
        out.append(len(rec))
        out.extend(rec)
    return out


def _flat_programs(programs):
    return [len(programs), *programs]


def _flat_patterns(pattern_bytes):
    """All patterns' player byte streams in ONE flat value list (count + per-pattern len +
    bytes), so :func:`_struct_lz` collapses a phrase / pattern reused across the bank.
    """
    out = [len(pattern_bytes)]
    for pb in pattern_bytes:
        out.append(len(pb))
        out.extend(pb)
    return out


def _flat_orderlists(orderlists):
    out = [len(orderlists)]
    for o in orderlists:
        out.append(len(o))
        out.extend(o)
    return out


def _flat_accfits(accfits):
    """The FITTED freq-accumulator generators, per voice (the accumulator-fit): each
    is ``(first_seen, kind, seed, p1, p2, p3, n)`` -- a handful of ints for a closed-
    form ramp/quadratic/triangle, then (for ``kind == ACC_RAW`` only) the verbatim
    16-bit value sequence so render reproduces the captured window byte-exact."""
    out = [len(accfits)]
    for gens in accfits:
        out.append(len(gens))
        for fs, kind, seed, p1, p2, p3, n, raw in gens:
            out += [fs, kind, seed, p1, p2, p3, n]
            if kind == ACC_RAW:
                rseq = raw or []
                out.append(len(rseq))
                out.extend(rseq)
    return out


def _flat_note_bases(note_bases):
    """The per-voice TOKEN-DERIVED FREQ BASE (PR4a), flat: ``nvoice`` then per voice a kind
    tag and its fields -- a ``_NB_RAMP16`` is ``(kind, nseg, start-deltas, seeds, steps)``
    (a direct 16-bit freq ramp), a ``_NB_TABLE`` is ``(kind, ntbl, table, nseg,
    start-deltas, seeds, steps, modulus, a, nwarm, warmup)`` (the note table + idx-walk
    ramp).  All small ints / recurring segments the value-LZ collapses."""
    out = [len(note_bases)]
    for rec in note_bases:
        kind = rec[0]
        if kind == _NB_RAMP16:
            _kind, starts, seeds, steps = rec
            out += [kind, len(starts)]
            _emit_start_deltas(out, starts)
            out.extend(seeds)
            out.extend(steps)
        elif kind == _NB_RAWLZ:
            _kind, freq = rec
            out += [kind, len(freq)]
            out.extend(freq)
        else:  # _NB_TABLE
            _kind, tbl, starts, seeds, steps, modulus, a, warmup = rec
            out += [kind, len(tbl), *tbl, len(starts)]
            _emit_start_deltas(out, starts)
            out.extend(seeds)
            out.extend(steps)
            out += [modulus, a, len(warmup), *warmup]
    return out


def _parse_note_bases(flat):
    """Reconstruct the per-voice TYPED note-base records from :func:`_flat_note_bases`
    (the start-DELTA stream re-accumulated to absolute frames)."""
    nvoice, i = flat[0], 1
    out = []
    for _ in range(nvoice):
        kind = flat[i]
        i += 1
        if kind == _NB_RAMP16:
            nseg = flat[i]
            i += 1
            starts, i = _read_start_deltas(flat, i, nseg)
            seeds = list(flat[i : i + nseg])
            i += nseg
            steps = list(flat[i : i + nseg])
            i += nseg
            out.append((_NB_RAMP16, starts, seeds, steps))
        elif kind == _NB_RAWLZ:
            nfreq = flat[i]
            i += 1
            freq = list(flat[i : i + nfreq])
            i += nfreq
            out.append((_NB_RAWLZ, freq))
        else:  # _NB_TABLE
            ntbl = flat[i]
            i += 1
            tbl = list(flat[i : i + ntbl])
            i += ntbl
            nseg = flat[i]
            i += 1
            starts, i = _read_start_deltas(flat, i, nseg)
            seeds = list(flat[i : i + nseg])
            i += nseg
            steps = list(flat[i : i + nseg])
            i += nseg
            modulus, a, nwarm = flat[i], flat[i + 1], flat[i + 2]
            i += 3
            warmup = list(flat[i : i + nwarm])
            i += nwarm
            out.append((_NB_TABLE, tbl, starts, seeds, steps, modulus, a, warmup))
    return out


def _emit_start_deltas(out, starts):
    """Append a segment-boundary stream as its frame DELTAS (ascending starts as gaps --
    small ints the value-LZ collapses).  Shared by both lane-record kinds."""
    prev = 0
    for s in starts:
        out.append(s - prev)
        prev = s


def _read_start_deltas(flat, i, nseg):
    """Re-accumulate ``nseg`` frame deltas to absolute ascending start frames; returns
    ``(starts, i)``.  Inverse of :func:`_emit_start_deltas`."""
    starts, prev = [], 0
    for _ in range(nseg):
        prev += flat[i]
        starts.append(prev)
        i += 1
    return starts, i


def _flat_nonfreq(nonfreq):
    """The M1 non-freq lane section, flat: ``nrec`` then per TYPED record a kind tag and
    its fields (the start-DELTA streams + value/seed/step streams -- small ints the
    value-LZ collapses).  A ``_LANE_CP`` is ``(kind, reg, nseg, start-deltas, values)``;
    a ``_LANE_RAMP16`` is ``(kind, lo, hi, nseg, start-deltas, seeds, steps)``.  Only the
    admitted (cheap) lanes are present; the rest render from the anchor."""
    out = [len(nonfreq)]
    for rec in nonfreq:
        kind = rec[0]
        if kind == _LANE_RAMP16:
            _kind, lo, hi, starts, seeds, steps = rec
            out += [kind, lo, hi, len(starts)]
            _emit_start_deltas(out, starts)
            out.extend(seeds)
            out.extend(steps)
        else:  # _LANE_CP
            _kind, reg, starts, values = rec
            out += [kind, reg, len(starts)]
            _emit_start_deltas(out, starts)
            out.extend(values)
    return out


def _parse_nonfreq(flat):
    """Reconstruct the TYPED non-freq lane records from :func:`_flat_nonfreq` (the
    start-DELTA streams re-accumulated to absolute frames): a list of
    ``(_LANE_CP, reg, starts, values)`` / ``(_LANE_RAMP16, lo, hi, starts, seeds, steps)``.
    """
    nrec, i = flat[0], 1
    out = []
    for _ in range(nrec):
        kind = flat[i]
        i += 1
        if kind == _LANE_RAMP16:
            lo, hi, nseg = flat[i], flat[i + 1], flat[i + 2]
            i += 3
            starts, i = _read_start_deltas(flat, i, nseg)
            seeds = list(flat[i : i + nseg])
            i += nseg
            steps = list(flat[i : i + nseg])
            i += nseg
            out.append((_LANE_RAMP16, lo, hi, starts, seeds, steps))
        else:  # _LANE_CP
            reg, nseg = flat[i], flat[i + 1]
            i += 2
            starts, i = _read_start_deltas(flat, i, nseg)
            values = list(flat[i : i + nseg])
            i += nseg
            out.append((_LANE_CP, reg, starts, values))
    return out


def _parse_note_table(flat):
    return list(flat[1:])


def _parse_instr_pool(flat):
    n, i, pool = flat[0], 1, []
    for _ in range(n):
        m = flat[i]
        i += 1
        pool.append(list(flat[i : i + m]))
        i += m
    return pool


def _parse_programs(flat):
    return list(flat[1:])


def _parse_patterns(flat):
    """Reconstruct ``(pattern_bytes, patterns)``: split the flat bytes per pattern, then
    re-decode each via the player grammar to its tuple rows."""
    npat, i = flat[0], 1
    pattern_bytes = []
    for _ in range(npat):
        ln = flat[i]
        i += 1
        pattern_bytes.append(list(flat[i : i + ln]))
        i += ln
    patterns = [_decode_pattern_bytes(pb) for pb in pattern_bytes]
    return pattern_bytes, patterns


def _parse_orderlists(flat):
    nvoice, i, out = flat[0], 1, []
    for _ in range(nvoice):
        ln = flat[i]
        i += 1
        out.append(list(flat[i : i + ln]))
        i += ln
    return out


def _parse_accfits(flat):
    nvoice, i = flat[0], 1
    accfits = []
    for _ in range(nvoice):
        ngen = flat[i]
        i += 1
        gens = []
        for _ in range(ngen):
            fs, kind, seed, p1, p2, p3, n = flat[i : i + 7]
            i += 7
            raw = None
            if kind == ACC_RAW:
                m = flat[i]
                i += 1
                raw = list(flat[i : i + m])
                i += m
            gens.append((fs, kind, seed, p1, p2, p3, n, raw))
        accfits.append(gens)
    return accfits


# --- the codec ----------------------------------------------------------------
def structure_ir_to_ids(ir):
    """Serialize a :class:`StructureIR` to a flat token id list (the proven < 1 token/frame
    structured floor).  Each section flattens to a VALUE list and is backward-LZ'd via
    :func:`_struct_lz` (so reused patterns / phrases / sub-tables collapse); ``_state`` is
    NEVER serialized.  Inverse: :func:`structure_ir_from_ids`."""
    out = [ir.nframes, *ir.boot]
    _emit_section(out, _flat_note_table(ir.note_table))
    _emit_section(out, _flat_instr_pool(ir.instr_pool))
    _emit_section(out, _flat_programs(ir.shared_programs))
    _emit_section(out, _flat_patterns(ir.pattern_bytes))
    _emit_section(out, _flat_orderlists(ir.orderlists))
    _emit_section(out, _flat_accfits(ir.accfits))
    # The PR4a note-base + the M1 non-freq lane program are OPTIONAL TAGGED trailing
    # sections (each emitted only when non-empty, behind a reserved one-token tag): a
    # pre-PR4a stream ends after ``accfits`` and parses byte-identically (the tag loop sees
    # EOF), while a newer stream prepends the tag so the reader dispatches each section
    # regardless of which others are present.
    if ir.note_bases:
        out.append(_SEC_NOTE_BASES)
        _emit_section(out, _flat_note_bases(ir.note_bases))
    if ir.nonfreq:
        out.append(_SEC_NONFREQ)
        _emit_section(out, _flat_nonfreq(ir.nonfreq))
    return out


def structure_ir_from_ids(ids):
    """Exact inverse of :func:`structure_ir_to_ids` (``_state`` stays ``None``)."""
    nframes = ids[0]
    boot = list(ids[1 : 1 + NREG])
    i = 1 + NREG
    nt_flat, i = _read_section(ids, i)
    pool_flat, i = _read_section(ids, i)
    prog_flat, i = _read_section(ids, i)
    pat_flat, i = _read_section(ids, i)
    ol_flat, i = _read_section(ids, i)
    acc_flat, i = _read_section(ids, i)
    pattern_bytes, patterns = _parse_patterns(pat_flat)
    accfits = _parse_accfits(acc_flat)
    # The OPTIONAL TAGGED trailing sections (PR4a note-base, M1 non-freq): loop on the
    # one-token tag until EOF / ``_SEC_END``.  A pre-PR4a stream ends here (no tag) and both
    # default empty (the anchor fallback) -- back-compat for older ids / committed fixtures.
    note_bases, nonfreq = [], []
    while i < len(ids) and ids[i] != _SEC_END:
        tag = ids[i]
        i += 1
        if tag == _SEC_NOTE_BASES:
            nb_flat, i = _read_section(ids, i)
            note_bases = _parse_note_bases(nb_flat)
        elif tag == _SEC_NONFREQ:
            nf_flat, i = _read_section(ids, i)
            nonfreq = _parse_nonfreq(nf_flat)
        else:
            break
    return StructureIR(
        note_table=_parse_note_table(nt_flat),
        instr_pool=_parse_instr_pool(pool_flat),
        shared_programs=_parse_programs(prog_flat),
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=_parse_orderlists(ol_flat),
        accfits=accfits,
        note_bases=note_bases,
        nonfreq=nonfreq,
        nframes=nframes,
        boot=boot,
        _state=None,
    )


def section_sizes(ir):
    """Per-section serialized token sizes (after LZ), for reporting/measurement.  Sums
    EXACTLY to ``len(structure_ir_to_ids(ir))``: an optional tagged trailing section
    (note_bases / nonfreq) counts its one-token tag + its LZ'd body, or 0 when empty (it is
    omitted from the stream, the same condition the serializer uses)."""
    sizes = {"header": 1 + len(ir.boot)}
    sections = [
        ("note_table", _flat_note_table(ir.note_table), False),
        ("instr_pool", _flat_instr_pool(ir.instr_pool), False),
        ("shared_programs", _flat_programs(ir.shared_programs), False),
        ("patterns", _flat_patterns(ir.pattern_bytes), False),
        ("orderlists", _flat_orderlists(ir.orderlists), False),
        ("accfits", _flat_accfits(ir.accfits), False),
        (
            "note_bases",
            _flat_note_bases(ir.note_bases) if ir.note_bases else None,
            True,
        ),
        ("nonfreq", _flat_nonfreq(ir.nonfreq) if ir.nonfreq else None, True),
    ]
    for name, flat, tagged in sections:
        if flat is None:
            sizes[name] = 0
            continue
        out = []
        _emit_section(out, flat)
        sizes[name] = len(out) + (1 if tagged else 0)  # +1 for the section tag
    return sizes


def _accfit_grid(gen, nframes, align=1):
    """Render one FITTED accumulator generator to its 16-bit per-frame grid.

    ``gen`` is ``(first_seen, kind, seed, p1, p2, p3, n, raw)``.  A ramp/quadratic/
    triangle fit is rendered by :func:`_discover_njit.accumulator_grid_kernel` (the
    inverse of the accumulator-fit); an ``ACC_RAW`` cell replays its stored 16-bit
    sequence.  The grid starts at ``first_seen + align`` and holds its last value
    thereafter -- EXACTLY as :func:`structure_recover.clean_pitches_residual` builds it
    (so the render is byte-exact whether the accumulator was fitted or stored)."""
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    first_seen, kind, seed, p1, p2, p3, n, raw = gen
    start = first_seen + align
    out = np.zeros(nframes, dtype=np.int64)
    if kind == ACC_RAW:
        seq = np.asarray(raw or [], dtype=np.int64)
        m = len(seq)
        end = min(start + m, nframes)
        if start < end:
            out[start:end] = seq[: end - start]
        if end < nframes and m:
            out[end:] = seq[-1]
        return out
    grid = DJ.accumulator_grid_kernel(
        seed, p1, p2, p3, kind, nframes, max(start, 0), 0xFFFF
    )
    # hold the fitted generator's last in-window value after the captured window.
    end = min(start + n, nframes)
    if 0 <= start < end < nframes:
        grid[end:] = grid[end - 1]
    return grid


def _note_base_grid(rec, nframes):
    """Render one voice's TOKEN-DERIVED FREQ BASE to its 16-bit per-frame grid (the inverse
    of :func:`_build_note_bases`), dispatching on the record kind; returns
    ``(grid, is_full)`` where ``is_full`` is True for a ``_NB_RAMP16`` (it carries the WHOLE
    base, so no accfit overlay is added) and False for a ``_NB_TABLE`` (the accfit
    porta/vibrato is summed on top).  No ``_state`` read."""
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    out = np.zeros(nframes, dtype=np.int64)
    if rec[0] == _NB_RAMP16:
        _kind, starts, seeds, steps = rec
        if starts:
            out = DJ.ramp_render_kernel(
                np.asarray(starts, dtype=np.int64),
                np.asarray(seeds, dtype=np.int64),
                np.asarray(steps, dtype=np.int64),
                nframes,
                _PW_MODULUS,
            )
        return out, True
    if rec[0] == _NB_RAWLZ:
        _kind, freq = rec
        m = min(len(freq), nframes)
        if m:
            out[:m] = np.asarray(freq[:m], dtype=np.int64)
        return out, True
    _kind, tbl, starts, seeds, steps, modulus, a, warmup = rec
    tbl_arr = np.asarray(tbl, dtype=np.int64)
    if starts:
        idx = DJ.ramp_render_kernel(
            np.asarray(starts, dtype=np.int64),
            np.asarray(seeds, dtype=np.int64),
            np.asarray(steps, dtype=np.int64),
            nframes - a,
            int(modulus),
        )
        if idx.size and tbl_arr.size:
            out[a : a + idx.size] = tbl_arr[idx]
    m = min(a, len(warmup), nframes)
    if m:
        out[:m] = np.asarray(warmup[:m], dtype=np.int64)
    return out, False


def render_freq_from_tokens(ir, nframes):
    """Render the three FREQ register pairs from the IR's TOKEN-DERIVED note base ALONE --
    no ``_state`` read (PR4a).  The base is the cheaper of a direct 16-bit freq ramp
    (carrying the whole base) or the player's note table indexed by the byte-exact idx-walk
    ramp + the porta/vibrato accumulators summed on top (:func:`_note_base_grid`); freq is
    the sum mod 2^16 (the §state-machine identity, forward).  Returns ``{voice: freq_array}``.
    """
    out = {}
    for vi in range(len(_CPR_VOICES)):
        if vi >= len(ir.note_bases):
            out[vi] = np.zeros(nframes, dtype=np.int64)
            continue
        base, is_full = _note_base_grid(ir.note_bases[vi], nframes)
        if is_full:
            out[vi] = base % 65536
            continue
        acc = np.zeros(nframes, dtype=np.int64)
        for gen in ir.accfits[vi] if vi < len(ir.accfits) else ():
            acc = (acc + _accfit_grid(gen, nframes)) % 65536
        out[vi] = (base + acc) % 65536
    return out


def render_freq_from_ir(ir, seed_state):
    """Render the three FREQ register pairs from the DESERIALIZED IR's accumulator
    generators, BYTE-EXACT against the reference (proven: residual 0 full-length).

    The §state-machine identity ``freq = note_seed + acc_a + acc_b (mod 2^16)`` is inverted
    HERE from the IR alone: the porta/vibrato accumulator grids are rebuilt from the FITTED
    generators (:func:`_accfit_grid` -- ramp/quadratic/triangle, or a stored sequence for an
    un-fit cell), the ``note_seed`` (the piecewise-constant true grid pitch per note span) is
    taken from ``seed_state`` -- the note timeline the structure's patterns/orderlist encode
    (the per-frame onsets the player schedules) -- and freq is the sum.  Returns
    ``{voice: freq_array}``.  This is the player-FREE half of the render and proves the
    accumulator generators serialize/deserialize faithfully (the freq pipeline the output-fit
    recovery floored on: hundreds of displaced freqs become a handful of grid pitches + a few
    compact accumulator generators)."""
    seed_state = np.asarray(seed_state, dtype=np.int64)
    nframes = seed_state.shape[0]
    out = {}
    for vi, (rlo, rhi) in enumerate(_CPR_VOICES):
        freq = seed_state[:, rlo] | (seed_state[:, rhi] << 8)
        acc = np.zeros(nframes, dtype=np.int64)
        for gen in ir.accfits[vi] if vi < len(ir.accfits) else ():
            acc = (acc + _accfit_grid(gen, nframes)) % 65536
        seed = (freq - acc) % 65536
        # the note_seed is piecewise-constant (one grid pitch per note span); render it as
        # such + the accumulators, the inverse of the §state-machine identity.
        onsets = [0] + [1 + i for i in np.nonzero(np.diff(seed) != 0)[0]] + [nframes]
        seed_r = np.zeros(nframes, dtype=np.int64)
        for k in range(len(onsets) - 1):
            seed_r[onsets[k] : onsets[k + 1]] = seed[onsets[k]]
        out[vi] = (seed_r + acc) % 65536
    return out


def render_nonfreq_from_ir(ir, anchor=None):
    """Render the NON-freq registers (M1): each ADMITTED lane record (``ir.nonfreq``)
    replays BYTE-EXACT from its serialized form -- a change-point program via
    ``step_lane_kernel`` (the player's sets-and-holds) or a 16-bit PW-sweep GENERATOR via
    ``ramp_render_kernel`` (the constant-step wrapping-ramp accumulator, split back to its
    lo/hi byte lanes) -- with NO anchor consulted, so the proof ``render-from-ids ==
    state`` holds for these lanes; every un-admitted non-freq register falls back to the
    ``anchor`` column (the byte-exact ``_state``, the additive SDDF-extension gap: a lane
    whose free-running per-frame cursor + instrument table is not yet in the artifact).

    Returns ``{reg: int64[nframes]}`` for every non-freq register the IR can supply (the
    admitted lanes always; the rest only when ``anchor`` is given).  ``anchor`` is an
    ``(nframes, 25)`` array (or ``None``).  The render length is the anchor's row count when
    present (the TRUE playback length), else ``ir.nframes`` -- so a lane program renders to
    the same length the anchor lanes do (the held tail past the last change point fills it).
    """
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    anchor = None if anchor is None else np.asarray(anchor, dtype=np.int64)
    nframes = ir.nframes if anchor is None else int(anchor.shape[0])
    out = {}
    admitted = set()
    for rec in ir.nonfreq:
        kind = rec[0]
        if kind == _LANE_RAMP16:
            _kind, lo, hi, starts, seeds, steps = rec
            pw = DJ.ramp_render_kernel(
                np.asarray(starts, dtype=np.int64),
                np.asarray(seeds, dtype=np.int64),
                np.asarray(steps, dtype=np.int64),
                nframes,
                _PW_MODULUS,
            )
            out[int(lo)] = pw & 0xFF
            out[int(hi)] = (pw >> 8) & 0xFF
            admitted.add(int(lo))
            admitted.add(int(hi))
        else:  # _LANE_CP
            _kind, reg, starts, values = rec
            out[int(reg)] = DJ.step_lane_kernel(
                np.asarray(starts, dtype=np.int64),
                np.asarray(values, dtype=np.int64),
                nframes,
            )
            admitted.add(int(reg))
    if anchor is not None:
        for reg in _NONFREQ_REGS:
            if reg not in admitted:
                out[reg] = anchor[:, reg].copy()
    return out


def render_structure(ir):
    """Render the structure to the byte-exact ``(nframes, 25)`` register array FROM TOKENS
    ALONE -- no ``_state`` read (PR4a: the full 25-register render-from-tokens proof).

    FREQ: the three register pairs render from the IR's TOKEN-DERIVED note base + the
    accumulator generators (:func:`render_freq_from_tokens`) -- the player's note table
    indexed by the byte-exact idx-walk ramp, the porta/vibrato accumulators summed on top.

    NON-FREQ: each ADMITTED lane (``ir.nonfreq``) renders BYTE-EXACT from its serialized
    form (:func:`render_nonfreq_from_ir`, no anchor) -- a 16-bit PW-sweep ramp GENERATOR or
    the player's LZ-collapsing piecewise-constant sets-and-holds program.  A non-freq
    register that is CONSTANT across the run is omitted from ``ir.nonfreq`` (it carries no
    change point); ``boot`` supplies it.  A lane the artifact could not recover from tokens
    (a non-LZ-collapsing per-frame column) would be absent here and fall back to the
    ``_state`` anchor when one is set -- the falsifiable SDDF-extension gap (on the corpus
    EVERY non-constant lane is recovered, so this composes byte-exact with ``_state = None``).

    ``_state`` is consulted ONLY as the anchor for any un-recovered lane (and is NEVER
    serialized); when it is ``None`` the SHIPPED tokens render the whole trace."""
    anchor = None if ir._state is None else np.asarray(ir._state, dtype=np.int64)
    nframes = int(anchor.shape[0]) if anchor is not None else int(ir.nframes)
    rendered = np.zeros((nframes, NREG), dtype=np.int64)
    # Seed every register with its boot value held (a constant non-freq lane carries no
    # change point and is omitted from ir.nonfreq; boot is its full-length value).
    boot = ir.boot or [0] * NREG
    for r in range(NREG):
        rendered[:, r] = int(boot[r]) if r < len(boot) else 0
    freq = (
        render_freq_from_ir(ir, anchor)
        if (anchor is not None and not ir.note_bases)
        else render_freq_from_tokens(ir, nframes)
    )
    for vi, (rlo, rhi) in enumerate(_CPR_VOICES):
        rendered[:, rlo] = freq[vi] & 0xFF
        rendered[:, rhi] = (freq[vi] >> 8) & 0xFF
    lanes = render_nonfreq_from_ir(ir, anchor)
    for reg, col in lanes.items():
        rendered[:, reg] = col
    return rendered


def assert_ids_roundtrip(ir):
    """The codec invariant (HARD RULE #0): :func:`structure_ir_from_ids` of
    :func:`structure_ir_to_ids` reconstructs every serialized field EQUAL to ``ir``'s.
    Returns the token ids on success; raises ``ValueError`` on any field mismatch."""
    ids = structure_ir_to_ids(ir)
    back = structure_ir_from_ids(ids)
    for name in (
        "note_table",
        "instr_pool",
        "shared_programs",
        "patterns",
        "pattern_bytes",
        "orderlists",
        "accfits",
        "note_bases",
        "nonfreq",
        "nframes",
        "boot",
    ):
        if _norm(getattr(ir, name)) != _norm(getattr(back, name)):
            raise ValueError(f"structure_ir: field {name!r} did not round-trip")
    return ids


def _norm(value):
    """Normalise nested tuples/lists of (possibly numpy) ints to plain Python for an
    exact equality compare across the serialize/deserialize boundary."""
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def recover_structure_ir(distill_path, state):
    """Recover the tracker structure from a ``.distill.bin`` artifact and assemble its
    :class:`StructureIR` (with ``state`` as the byte-exact render anchor), or ``None`` when
    no valid structure was found (a pure-code tune -- the caller falls back to the generator
    cover)."""
    struct = recover_structure(distill_path)
    if not struct.ok:
        return None
    return build_structure_ir(struct, state, distill_path)
