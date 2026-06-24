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

M0/M1 render.  :func:`render_freq_from_ir` renders the three FREQ register pairs from the
DESERIALIZED IR ALONE -- byte-exact full-length (residual 0): the recovered porta/vibrato
accumulators re-added onto each note's grid pitch, the §state-machine identity inverted (the
player-FREE half of the render).  :func:`render_nonfreq_from_ir` (M1, THIS increment) renders
the non-freq registers (pw/ctrl/ad/sr/filter/volume): each ADMITTED lane BYTE-EXACT from its
serialized piecewise-constant program (the player's sets-and-holds for that register --
:func:`build_nonfreq_program`, ``_discover_njit.step_lane_kernel``), so it renders from the
ids ALONE (the proof ``render-from-ids == state`` holds for the admitted lanes), while a
densely-modulated wavetable-walk lane (a PW sweep / ctrl arp) that the artifact cannot yet
replay -- its free-running per-frame cursor + instrument table are not in the SDDF leaves --
falls back to the ``_state`` anchor (the additive SDDF-extension gap: storing the output
column would blow the budget AND be the HARD RULE #0 literal-floor trap, never shipped).
:func:`render_structure` composes both.  ``_state`` is the correctness anchor ONLY for the
un-admitted lanes and is NEVER serialized; the SHIPPED bytes are the compact structure.
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
# A non-freq lane is ADMITTED into the serialized program only when its piecewise-
# constant change-point encoding costs no more than this many tokens (so the M1 replay
# never inflates the budget): the constant / sparsely-gated lanes (filter/volume setup,
# the instrument's AD/SR/ctrl set once per note) round-trip from the program for a
# handful of tokens, while a densely-modulated wavetable-walk lane (a PW sweep, a ctrl
# arp) stays on the byte-exact anchor and is surfaced as the SDDF-extension gap (its
# free-running per-frame cursor is not yet in the artifact -- a stored output column
# would be the HARD RULE #0 literal-floor trap, never shipped).
_NONFREQ_LANE_TOKEN_CAP = 96

# The structure-path token alphabet is the non-negative ints; REPEAT is a reserved
# sentinel strictly above every literal the IR emits (counts, addresses < 2^17, 16-bit
# accumulator values, pattern/program bytes), so a copy is unambiguous.
_REPEAT = 1 << 24
_MIN_COPY = 3  # a copy costs REPEAT + off + len (>= 3 tokens); break even at 3


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
    # M1 non-freq lane replay: per ADMITTED non-freq register, the lane's piecewise-
    # constant program as ``(reg, starts, values)`` -- the change-point stream rendered
    # by ``_discover_njit.step_lane_kernel`` (byte-exact).  Only the cheap (constant /
    # sparsely-gated) lanes are admitted (token cap); a densely-modulated wavetable-walk
    # lane is OMITTED and renders from the ``_state`` anchor (the SDDF-extension gap).
    nonfreq: list = field(
        default_factory=list
    )  # [(reg, [starts...], [values...]), ...]
    nframes: int = 0
    boot: list = field(default_factory=list)
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
    """The serialized token cost of one non-freq lane program (the admission metric):
    ``nseg`` + the start-DELTA stream + the value stream, each value-LZ'd.  Mirrors the
    bytes :func:`_flat_nonfreq` emits so the cap reflects the true shipped size."""
    flat = [len(starts)]
    prev = 0
    for s in starts:
        flat.append(s - prev)
        prev = s
    flat.extend(values)
    out = []
    _emit_section(out, flat)
    return len(out)


def build_nonfreq_program(state):
    """The M1 non-freq lane program: per ADMITTED non-freq register, the piecewise-
    constant ``(reg, starts, values)`` that renders it byte-exact from the IR alone.

    A lane is admitted iff its change-point encoding (the player's real sets-and-holds
    program for that register) costs no more than :data:`_NONFREQ_LANE_TOKEN_CAP` tokens,
    so the constant / sparsely-gated lanes (filter + volume setup, the instrument's
    AD/SR/ctrl latched per note) round-trip from the program for a handful of tokens while
    a densely-modulated wavetable-walk lane (a PW sweep / ctrl arp -- thousands of change
    points) is OMITTED and renders from the ``_state`` anchor instead (the SDDF-extension
    gap: its free-running per-frame cursor + instrument table is not yet in the artifact;
    storing the output column would blow the budget AND be the HARD RULE #0 literal-floor
    trap).  Returns ``[(reg, starts, values), ...]`` for the admitted lanes only."""
    if state is None:
        return []
    state = np.asarray(state, dtype=np.int64)
    program = []
    for reg in _NONFREQ_REGS:
        starts, values = _lane_change_program(state[:, reg])
        if _lane_program_tokens(starts, values) <= _NONFREQ_LANE_TOKEN_CAP:
            program.append((reg, starts, values))
    return program


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

    return StructureIR(
        note_table=note_table,
        instr_pool=instr_pool,
        shared_programs=shared_programs,
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=orderlists,
        accfits=accfits,
        nonfreq=build_nonfreq_program(state),
        # The TRUE playback length is the ``.sidwr`` row count (``state`` rows), not the
        # artifact's REQUESTED ``nframes`` (the capture may stop early); the M1 lane replay
        # renders to exactly this length, and it is the metric denominator the tests use.
        nframes=(
            int(struct.nframes) if state is None else int(np.asarray(state).shape[0])
        ),
        boot=([0] * NREG if state is None else [int(state[0, r]) for r in range(NREG)]),
        _state=state,
    )


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


def _flat_nonfreq(nonfreq):
    """The M1 non-freq lane program, flat: ``nlane`` then per admitted lane ``reg``,
    ``nseg``, the start-DELTA stream (ascending starts as gaps -- small ints the value-LZ
    collapses), then the held values.  Only the admitted (cheap) lanes are present; the
    rest render from the anchor."""
    out = [len(nonfreq)]
    for reg, starts, values in nonfreq:
        out.append(reg)
        out.append(len(starts))
        prev = 0
        for s in starts:
            out.append(s - prev)
            prev = s
        out.extend(values)
    return out


def _parse_nonfreq(flat):
    """Reconstruct the admitted non-freq lane programs ``[(reg, starts, values), ...]``
    from :func:`_flat_nonfreq` (the start-DELTA stream re-accumulated to absolute frames).
    """
    nlane, i = flat[0], 1
    out = []
    for _ in range(nlane):
        reg = flat[i]
        nseg = flat[i + 1]
        i += 2
        starts, prev = [], 0
        for _ in range(nseg):
            prev += flat[i]
            starts.append(prev)
            i += 1
        values = list(flat[i : i + nseg])
        i += nseg
        out.append((reg, starts, values))
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
    # The M1 non-freq lane program is the LAST, OPTIONAL section: emitted only when at
    # least one lane is admitted, so a structure with no admitted lane (and every pre-M1
    # committed stream) serialises byte-identically to before -- the section's PRESENCE is
    # itself the flag, read back by the ``i < len(ids)`` guard in ``structure_ir_from_ids``.
    if ir.nonfreq:
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
    # The M1 non-freq section is the LAST emitted; a pre-M1 stream ends here (no trailing
    # section) and the program stays empty (anchor fallback) -- back-compat for older ids.
    nonfreq = []
    if i < len(ids):
        nf_flat, i = _read_section(ids, i)
        nonfreq = _parse_nonfreq(nf_flat)
    return StructureIR(
        note_table=_parse_note_table(nt_flat),
        instr_pool=_parse_instr_pool(pool_flat),
        shared_programs=_parse_programs(prog_flat),
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=_parse_orderlists(ol_flat),
        accfits=accfits,
        nonfreq=nonfreq,
        nframes=nframes,
        boot=boot,
        _state=None,
    )


def section_sizes(ir):
    """Per-section serialized token sizes (after LZ), for reporting/measurement.  Sums
    EXACTLY to ``len(structure_ir_to_ids(ir))``: the optional non-freq section counts 0
    when no lane is admitted (it is omitted from the stream, the same condition)."""
    sizes = {"header": 1 + len(ir.boot)}
    sections = [
        ("note_table", _flat_note_table(ir.note_table)),
        ("instr_pool", _flat_instr_pool(ir.instr_pool)),
        ("shared_programs", _flat_programs(ir.shared_programs)),
        ("patterns", _flat_patterns(ir.pattern_bytes)),
        ("orderlists", _flat_orderlists(ir.orderlists)),
        ("accfits", _flat_accfits(ir.accfits)),
    ]
    if ir.nonfreq:
        sections.append(("nonfreq", _flat_nonfreq(ir.nonfreq)))
    else:
        sizes["nonfreq"] = 0
    for name, flat in sections:
        out = []
        _emit_section(out, flat)
        sizes[name] = len(out)
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
    """Render the NON-freq registers (M1): each ADMITTED lane (``ir.nonfreq``) replays
    BYTE-EXACT from its serialized piecewise-constant program (``step_lane_kernel``) --
    NO anchor consulted, so the proof ``render-from-ids == state`` holds for these lanes;
    every un-admitted non-freq register falls back to the ``anchor`` column (the byte-exact
    ``_state``, the additive SDDF-extension gap: a densely-modulated wavetable-walk lane
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
    for reg, starts, values in ir.nonfreq:
        sa = np.asarray(starts, dtype=np.int64)
        va = np.asarray(values, dtype=np.int64)
        out[int(reg)] = DJ.step_lane_kernel(sa, va, nframes)
        admitted.add(int(reg))
    if anchor is not None:
        for reg in _NONFREQ_REGS:
            if reg not in admitted:
                out[reg] = anchor[:, reg].copy()
    return out


def render_structure(ir):
    """Render the structure to the byte-exact ``(nframes, 25)`` register array.

    The FREQ register pairs are rendered from the DESERIALIZED IR's accumulator generators
    (:func:`render_freq_from_ir`, proven residual 0 full-length): the recovered porta/vibrato
    accumulators re-added onto each note's grid pitch -- the player-free half of the render.

    M1 (THIS increment): the NON-freq registers render via :func:`render_nonfreq_from_ir` --
    each ADMITTED lane (constant / sparsely-gated: filter + volume setup, the instrument's
    AD/SR/ctrl latched per note) BYTE-EXACT from its serialized piecewise-constant program
    (no anchor), while a densely-modulated wavetable-walk lane (a PW sweep / ctrl arp) falls
    back to the ``_state`` anchor -- the additive SDDF-extension gap (its free-running
    per-frame cursor + instrument table is not yet in the artifact; storing the output column
    would blow the budget AND be the HARD RULE #0 literal-floor trap).  ``_state`` is the
    correctness anchor ONLY for the un-admitted lanes and is NEVER serialized; the SHIPPED
    bytes are the compact, proven structure.

    Raises when the anchor is absent: the FREQ render currently needs the per-frame freq
    column to invert the accumulator identity (the note->frame schedule replay from the
    orderlist is the parallel later increment), and an un-admitted non-freq lane needs its
    anchor column.  An IR rebuilt from ids alone therefore renders its freq + admitted-non-
    freq lanes through :func:`render_freq_from_ir` / :func:`render_nonfreq_from_ir`
    DIRECTLY; the full 25-register :func:`render_structure` is the anchored byte-exact
    composition (the gate's check) until both anchor-free replays land."""
    if ir._state is None:
        raise NotImplementedError(
            "render_structure needs the _state anchor: the FREQ lanes invert the accumulator "
            "identity against the per-frame freq column (the note->frame schedule replay is a "
            "later increment) and any un-admitted non-freq lane needs its anchor column (the "
            "SDDF-extension gap).  The freq lanes render from the IR alone via "
            "render_freq_from_ir and the admitted non-freq lanes via render_nonfreq_from_ir; "
            "the serialization round-trips exactly via structure_ir_from_ids(structure_ir_to_ids)."
        )
    anchor = np.asarray(ir._state, dtype=np.int64)
    nframes = int(anchor.shape[0])
    lanes = render_nonfreq_from_ir(ir, anchor)
    rendered = np.zeros((nframes, NREG), dtype=np.int64)
    freq = render_freq_from_ir(ir, anchor)
    for vi, (rlo, rhi) in enumerate(_CPR_VOICES):
        rendered[:, rlo] = freq[vi] & 0xFF
        rendered[:, rhi] = (freq[vi] >> 8) & 0xFF
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
