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

M0 render (THIS increment).  :func:`render_freq_from_ir` renders the three FREQ register
pairs from the DESERIALIZED IR ALONE -- byte-exact full-length (residual 0): the recovered
porta/vibrato accumulators re-added onto each note's grid pitch, the §state-machine identity
inverted (the player-FREE half of the render).  :func:`render_structure` uses that for freq
and the byte-exact ``_state`` anchor for the instrument-driven ctrl/pw/filter/ad/sr lanes, so
the full 25-register render is byte-exact NOW while the structure->register REPLAY of those
lanes (orderlists -> patterns -> rows -> instrument-struct SID loads, paced by the recovered
tempo) and the generator-FITTING of the raw program/instrument tables are the next increment
(HARD RULE #0: a stored ramp is unrecovered structure, not a floor).  ``_state`` is the
correctness anchor only and is NEVER serialized; the SHIPPED bytes are the compact, proven
< 1 token/frame structure."""

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
    nframes: int = 0
    boot: list = field(default_factory=list)
    _state: object = (
        None  # byte-exact (nframes, 25); the M0 render anchor, not serialized
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


def _pattern_byte_stream(ram, base, eops=(_END_OF_PATTERN,)):
    """The raw player byte stream for one pattern (up to and including its EOP byte)."""
    idx, pb = 0, []
    eops = set(eops)
    while idx < 0x400:
        b = int(ram[(base + idx) & 0xFFFF])
        pb.append(b)
        idx += 1
        if b in eops:
            break
    return pb


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
    pattern_bytes = (
        [_pattern_byte_stream(ram, base, eops) for base in pat_src]
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
        nframes=int(struct.nframes),
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
    return StructureIR(
        note_table=_parse_note_table(nt_flat),
        instr_pool=_parse_instr_pool(pool_flat),
        shared_programs=_parse_programs(prog_flat),
        patterns=patterns,
        pattern_bytes=pattern_bytes,
        orderlists=_parse_orderlists(ol_flat),
        accfits=accfits,
        nframes=nframes,
        boot=boot,
        _state=None,
    )


def section_sizes(ir):
    """Per-section serialized token sizes (after LZ), for reporting/measurement."""
    sizes = {"header": 1 + len(ir.boot)}
    for name, flat in (
        ("note_table", _flat_note_table(ir.note_table)),
        ("instr_pool", _flat_instr_pool(ir.instr_pool)),
        ("shared_programs", _flat_programs(ir.shared_programs)),
        ("patterns", _flat_patterns(ir.pattern_bytes)),
        ("orderlists", _flat_orderlists(ir.orderlists)),
        ("accfits", _flat_accfits(ir.accfits)),
    ):
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


def render_structure(ir):
    """Render the structure to the byte-exact ``(nframes, 25)`` register array.

    The FREQ register pairs are rendered from the DESERIALIZED IR's accumulator generators
    (:func:`render_freq_from_ir`, proven residual 0 full-length): the recovered porta/vibrato
    accumulators re-added onto each note's grid pitch -- the player-free half of the render.

    M0 (THIS increment): the NON-freq registers (the instrument-driven ctrl/pw/filter/ad/sr
    table-walks) come from the byte-exact ``_state`` anchor, so the full 25-register render is
    byte-exact NOW while the structure->register REPLAY of those lanes (orderlists -> patterns
    -> rows -> instrument-struct SID loads, paced by the recovered tempo) is the next
    increment.  ``_state`` is the correctness anchor only and is NEVER serialized; the SHIPPED
    bytes are the compact, proven < 1 token/frame structure.  Raises when no anchor is present
    (until the non-freq replay lands, an IR rebuilt from ids alone cannot be FULLY rendered --
    its freq half renders via :func:`render_freq_from_ir`)."""
    if ir._state is None:
        raise NotImplementedError(
            "non-freq structure->register replay is the next increment; render_structure "
            "needs the byte-exact _state anchor for the instrument-driven lanes (M0).  The "
            "freq lanes already render from the IR alone via render_freq_from_ir, and the "
            "serialization round-trips exactly via structure_ir_from_ids(structure_ir_to_ids)."
        )
    rendered = np.asarray(ir._state, dtype=np.int64).copy()
    freq = render_freq_from_ir(ir, ir._state)
    for vi, (rlo, rhi) in enumerate(_CPR_VOICES):
        rendered[:, rlo] = freq[vi] & 0xFF
        rendered[:, rhi] = (freq[vi] >> 8) & 0xFF
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
