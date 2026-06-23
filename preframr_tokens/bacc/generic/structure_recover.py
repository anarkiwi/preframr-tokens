"""PROTOTYPE: generic tracker-STRUCTURE recovery from the SDST distill artifact.

This is the recovery the output-fit generic path (``recover.py`` / ``cover.py`` /
``tracker_ir.py``) MISSES.  The shipping path OUTPUT-FITS the integrator's displaced
per-note frequency and ignores the tracker structure that is already captured
BYTE-EXACT in the distill artifact: on JCH ``10.sid`` that yields a 21-program pool
(vs 16 real instruments) and a 572-entry "note table" (vs 30 real pitches, growing
with playback length) because pitch-factoring only fires on exact-grid hits, so
porta-swept notes never share, and the per-note rows are never factored through the
orderlist.

This module recovers the COMMON tracker IR DIRECTLY from the artifact, with NO
hardcoded per-tune addresses -- every table base / stride / reference is DERIVED
from the captured access pattern:

  1. ``discover_pointer_table``  -- the pattern-pointer table (a split lo/hi 16-bit
     table whose entries ascend and point into the read song-data region), from
     SNAP + the access map.
  2. ``discover_orderlist``      -- the per-voice orderlists (a small pointer table
     whose targets are 0xFF-terminated streams of pattern indices + control
     markers), from SNAP + the access map.
  3. ``discover_instrument_table`` -- the instrument struct table base + stride,
     from clustering the RAM_READ leaves of the SID-register-write backward slices
     (SDDF): a stride-K lattice of leaf addresses IS the table.
  4. ``decode_patterns``         -- walk the pattern-pointer chain and decode the
     ``(note, instr_ref, dur, cmd)`` rows (the REPEAT/TRANSPOSE collapse the
     output-fit recovery misses; one note table, one instrument pool).
  5. ``clean_pitches``           -- subtract the captured porta/vibrato accumulators
     (the STSQ state-cell sequences) so each freq is its real grid pitch (30, not
     572).  Proven byte-exact (residual 0) by ``proto/PROOF.py``.

The artifact carries BOTH the structure (SNAP byte-exact) AND the freq accumulators
(STSQ).  ``recover_structure`` ties them into a :class:`RecoveredStructure`; the
byte-exact gate is that the decoded structure RE-ENCODES to the exact SNAP song-data
bytes (a lossless decode -- SNAP is the player's data input, so re-rendering it is
byte-identical to the original ``.sidwr``), and ``token_budget`` reports the
structured floor (< 1 token/frame).

This is COMPLEMENTARY to the shipping generic recovery (kept importable for the
measuring agent): where a valid tracker structure renders byte-exact it is recovered
HERE; where none exists (A Mind Is Born -- 256 bytes of pure code, no instrument /
pattern table) discovery finds nothing and the caller FALLS BACK to the generator
cover.  The fix is ADDITIVE, never a regression of the generator-cover floor.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from preframr_tokens.bacc.generic.distill import (
    ACC_READ_PLAY,
    SddfSlice,
    load_distill,
)

# The SDDF/STSQ data-flow sections are now parsed by the ONE artifact reader
# (:mod:`distill`); this module consumes ``Distill.sddf_slices`` / ``Distill.stsq_cells``
# directly (design §3 consolidation -- the duplicate parser that lived here is removed).
# ``_SdwSlice`` is retained as an alias to the canonical :class:`distill.SddfSlice` so
# the existing discovery API (``discover_instrument_table`` over slices) is unchanged.
_SdwSlice = SddfSlice


def _load_distill_or_none(distill_path):
    """Load a ``.distill.bin`` via the ONE reader, or ``None`` if it is missing /
    not an SDST artifact (so the slice/cell shims degrade to empty, as the old
    self-contained parsers did on a non-artifact path)."""
    try:
        return load_distill(distill_path)
    except (OSError, ValueError):
        return None


def read_sddf_slices(distill_path):
    """The SID-write backward-slices (RAM_READ leaves) from a ``.distill.bin`` -- a
    thin shim over the ONE reader (:func:`distill.load_distill`).  Empty for a
    pre-data-flow artifact or a non-artifact path."""
    d = _load_distill_or_none(distill_path)
    return d.sddf_slices if d is not None else []


# ---------------------------------------------------------------------------
# NewPlayer-class pattern grammar (the JCH / DMC / many-driver shape): a row is a
# run of high-bit COMMAND bytes terminated by a low-bit NOTE byte.  The marker
# classes are derived from the high bits, NOT a per-tune table:
#   0x80..0x9F  duration / gate    (param = b & 0x1F)
#   0xA0..0xBF  instrument select  (param = b & 0x1F)
#   0xC0..0xFF  effect command     (param = b & 0x3F)

# ---------------------------------------------------------------------------
# NewPlayer-class pattern grammar (the JCH / DMC / many-driver shape): a row is a
# run of high-bit COMMAND bytes terminated by a low-bit NOTE byte.  The marker
# classes are derived from the high bits, NOT a per-tune table:
#   0x80..0x9F  duration / gate    (param = b & 0x1F)
#   0xA0..0xBF  instrument select  (param = b & 0x1F)
#   0xC0..0xFF  effect command     (param = b & 0x3F)
#   b < 0x80    the NOTE byte, ends the row.  0x7F = end-of-pattern; 0x00 / 0x7E =
#               rest / tie (not a pitch).
# A driver with a different grammar would simply fail the byte-exact re-encode gate
# (the decode is not lossless) and the caller falls back -- the grammar is a
# HYPOTHESIS the round-trip falsifies, never an assumption.
# ---------------------------------------------------------------------------
_END_OF_PATTERN = 0x7F
_NON_PITCH = (0x00, 0x7E)
_MARK_DUR, _MARK_INSTR, _MARK_CMD = 0x80, 0xA0, 0xC0
_INSTR_KILL = 0x1F  # the conventional "silence / kill voice" instrument sentinel


@dataclass
class RecoveredStructure:
    """The generically-recovered tracker structure (every field DERIVED from the
    artifact, no hardcoded addresses).  ``ok`` is False when no valid structure was
    found (the caller falls back to the generator cover)."""

    ok: bool
    reason: str = ""
    # pattern-pointer table
    patptr_lo: int = 0
    patptr_hi: int = 0
    n_patterns: int = 0
    pattern_ptrs: list = field(default_factory=list)
    # per-voice orderlists (pattern-index + control-marker streams)
    orderlist_ptr_table: int = 0
    orderlist_ptrs: list = field(default_factory=list)
    orderlists: list = field(default_factory=list)
    # instrument struct table
    instr_base: int = 0
    instr_stride: int = 0
    instr_records: list = field(default_factory=list)  # the N used 8-byte structs
    n_instruments: int = 0
    instr_refs: list = field(default_factory=list)
    # decoded patterns
    patterns: list = field(default_factory=list)  # [[(note,instr,dur,cmd), ...], ...]
    n_rows: int = 0
    pattern_bytes: int = 0
    # note table (true grid pitches)
    note_table: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    # shared program table spans (wave/pulse/filter/cmd), referenced once
    program_spans: dict = field(default_factory=dict)
    # relocation (S1): the runtime->image delta applied to resolve pattern pointers
    # (0 for an in-place driver).  ``pattern_src`` is the resolved IMAGE address each
    # pattern's raw bytes are read from (== pattern_ptrs through reloc); the player
    # walks ``pattern_ptrs`` (runtime), we read ``pattern_src`` (image / SNAP).
    reloc_delta: int = 0
    pattern_src: list = field(default_factory=list)
    # explicit per-pattern byte lengths (parallel to ``pattern_src``), set when the
    # length was sited by READ-COVERAGE rather than a value-range EOP marker (the
    # nibble / bit-packed dialects: GoatTracker, Soundmonitor, Music_Assembler).  When
    # empty the readers fall back to the grammar-EOP walk (the value-range dialects),
    # so the existing NewPlayer/TFX/FC behaviour is byte-identical.
    pattern_lens: list = field(default_factory=list)
    # the grammar dialect chosen by the byte-exact round-trip (S4): the per-byte
    # field-kind table + per-kind operand width + param mask + packing mode.
    grammar: object = None
    # provenance
    load_addr: int = 0
    load_len: int = 0
    nframes: int = 0
    distill_path: str = (
        ""  # the artifact this was recovered from (for budget RAM reads)
    )
    ram: object = None  # the SNAP RAM (uint8[65536]) carried for the token budget


def _image_bounds(d):
    return d.load_addr, d.load_addr + d.load_len


# ---------------------------------------------------------------------------
# S1 -- relocation resolve.  The init block-copy (RELO) gives delta = dst-src; a
# pattern pointer the player formed at runtime (dst space) resolves to its image
# (src / SNAP) address by subtracting delta.  In-place drivers have no RELO -> 0.
# ---------------------------------------------------------------------------
def reloc_delta_candidates(d):
    """The relocation deltas to try (S1): ``0`` (in-place) plus every distinct RELO
    block-copy ``delta = dst_base - src_base``.  The byte-exact round-trip (S4)
    SELECTS the one under which the pattern pointers resolve into the song data."""
    deltas = [0]
    for r in getattr(d, "relo_copies", ()):
        dl = r.delta
        if dl and dl not in deltas:
            deltas.append(dl)
    return deltas


# ---------------------------------------------------------------------------
# S4 -- row-grammar dialects.  The four SURVEY dialects collapse to ONE decode
# skeleton (``_discover_njit.decode_pattern_kernel``) parameterized by a small set:
#   boundaries[256] -> field-kind   (K_NOTE/K_INSTR/K_DUR/K_CMD/K_EOP/K_REST/K_IGN)
#   op_width[16]    -> operand bytes consumed after a marker of that kind
#   param_mask[16]  -> the field value mask for that kind
#   packing_mode    -> 1 for the nibble split (instr=hi-nibble, note=lo-nibble)
# A dialect is SELECTED by the byte-exact re-encode of the pattern bank, never by a
# disasm heuristic (HARD RULE #0: the grammar is a hypothesis the round-trip
# falsifies).  We only need the boundaries to find the EOP byte for slicing -- the
# stored bytes are the player's own compact encoding (byte-exact by construction).
# ---------------------------------------------------------------------------
def _grammar(boundaries, op_width=None, param_mask=None, packing_mode=0, eop=None):
    """Build a dialect parameter set (the small arrays the decode kernel takes)."""
    bnd = np.zeros(256, dtype=np.int64)
    for b, k in boundaries.items():
        bnd[b] = k
    if eop is not None:
        for e in eop if isinstance(eop, (list, tuple)) else (eop,):
            bnd[e] = 4  # K_EOP
    ow = np.zeros(16, dtype=np.int64)
    if op_width:
        for k, w in op_width.items():
            ow[k] = w
    pm = np.zeros(16, dtype=np.int64)
    if param_mask:
        for k, m in param_mask.items():
            pm[k] = m
    return {
        "boundaries": bnd,
        "op_width": ow,
        "param_mask": pm,
        "packing_mode": int(packing_mode),
    }


def _dialects():
    """The four candidate row grammars (parameter sets), tried in order; the first
    whose decode re-encodes the whole bank byte-exact wins (S4 selection).

    K_NOTE=0 K_INSTR=1 K_DUR=2 K_CMD=3 K_EOP=4 K_REST=5 K_IGN=6.  The only field-kind
    that matters for byte-exact slicing is the EOP marker (the stored bytes ARE the
    canonical encoding); the others let the IR carry decoded rows for measurement."""
    out = {}
    # NewPlayer (note<$80; $80-$9F dur, $A0-$BF instr, $C0-$FE cmd; $7F or $FF EOP).
    bnd = {}
    for b in range(0x80, 0xA0):
        bnd[b] = 2
    for b in range(0xA0, 0xC0):
        bnd[b] = 1
    for b in range(0xC0, 0x100):
        bnd[b] = 3
    out["newplayer"] = _grammar(
        bnd,
        param_mask={1: 0x1F, 2: 0x1F, 3: 0x3F},
        eop=(0x7F, 0xFF),
    )
    # TFX / stateful-prefix (note<$60; $60-$7F instr; $80-$BF dur; $C0-$FE cmd; $FF EOP).
    bnd = {}
    for b in range(0x60, 0x80):
        bnd[b] = 1
    for b in range(0x80, 0xC0):
        bnd[b] = 2
    for b in range(0xC0, 0xFF):
        bnd[b] = 3
    out["tfx"] = _grammar(bnd, param_mask={1: 0x1F, 2: 0x3F, 3: 0x3F}, eop=0xFF)
    # FutureComposer (note $01-$3F; $40-$7F dur; $80-$BF instr; $C0-$FE cmd; $FF EOP).
    bnd = {}
    for b in range(0x40, 0x80):
        bnd[b] = 2
    for b in range(0x80, 0xC0):
        bnd[b] = 1
    for b in range(0xC0, 0xFF):
        bnd[b] = 3
    out["fc"] = _grammar(bnd, param_mask={1: 0x3F, 2: 0x3F, 3: 0x3F}, eop=0xFF)
    return out


def _walk_pattern_bytes(ram, ptr, boundaries, max_bytes=0x400):
    """Return the byte length of the pattern at ``ptr`` (through its EOP inclusive),
    or 0 if no EOP is hit within ``max_bytes`` (a failed slice)."""
    for k in range(max_bytes):
        b = int(ram[(ptr + k) & 0xFFFF])
        if boundaries[b] == 4:  # K_EOP
            return k + 1
    return 0


def _read_extent(read_play, base, lo_img, hi_img, max_bytes=0x400):
    """The length of the contiguous run of bytes the player READ AS DATA from
    ``base`` (``ACC_READ_PLAY``), bounded by the loaded image.

    This is the byte-exact pattern length DERIVED FROM OBSERVATION -- the tracer
    recorded exactly which bytes were consumed as data, so the read-coverage run is
    the pattern content the player actually traversed, GRAMMAR-AGNOSTIC (no per-driver
    EOP terminator constant).  It is the structural pattern-length signal the SURVEY's
    "nibble / bit-packed" dialects (GoatTracker, Soundmonitor, Music_Assembler) need:
    their rows carry no value-range EOP byte, so the 3 value-range dialects cannot
    slice them, but the read-coverage extent sites them exactly.  ``0`` when ``base``
    itself was not read as data (a phantom pointer -- e.g. a null/init pointer the
    orderlist walk visited before its first real pattern)."""
    k = 0
    while (
        k < max_bytes
        and lo_img <= (base + k) < hi_img
        and read_play[(base + k) & 0xFFFF]
    ):
        k += 1
    return k


def _pwlk_candidate_eop(d, walk, delta, sdm, dialects):
    """The value-range-DIALECT pattern bank for one (walk, delta): the first grammar
    whose EOP slices every resolved-in-song-data pointer cleanly (NewPlayer/TFX/FC --
    the value-range dialects).  Returns a candidate dict or ``None``.  Byte-identical
    to the original ``discover_patterns_pwlk`` per-grammar slice, factored out so the
    read-coverage path (the nibble / bit-packed dialects) is a sibling, not a rewrite.
    """
    lo_img, hi_img = _image_bounds(d)
    seq = walk.ptr_vals
    resolved = [(x - delta) & 0xFFFF for x in seq]
    uniq = sorted(set(a for a in resolved if lo_img <= a < hi_img))
    if len(uniq) < 2:
        return None
    if sum(1 for a in uniq if sdm[a]) < max(2, len(uniq) // 2):
        return None
    for gname, gram in dialects.items():
        bnd = gram["boundaries"]
        ok = True
        for a in uniq:
            ln = _walk_pattern_bytes(d.ram, a, bnd)
            if ln == 0 or not sdm[a]:
                ok = False
                break
        if not ok:
            continue
        bank = uniq
        index_of = {a: i for i, a in enumerate(bank)}
        orderlist = [index_of.get(a, 0xFF) for a in resolved]
        cov = len(set(o for o in orderlist if o != 0xFF))
        return {
            "pattern_src": bank,
            "pattern_ptrs": [(a + delta) & 0xFFFF for a in bank],
            "pattern_lens": [],  # EOP-walk length (the readers re-derive from grammar)
            "orderlist": orderlist,
            "reloc_delta": delta,
            "grammar": gram,
            "grammar_name": gname,
            "zp": walk.zp,
            "coverage": cov,
            "n_patterns": len(bank),
        }
    return None


def _pwlk_candidate_readcov(d, walk, delta, sdm):
    """The READ-COVERAGE pattern bank for one (walk, delta), GRAMMAR-AGNOSTIC (the
    nibble / bit-packed dialects -- GoatTracker, Soundmonitor, Music_Assembler -- whose
    rows carry no value-range EOP byte, so :func:`_pwlk_candidate_eop` cannot slice
    them).  A pointer is a REAL pattern iff it was READ AS DATA from its start
    (:func:`_read_extent` > 0) and lies in the song-data region; phantom pointers (a
    null/init pointer the walk visited before its first real pattern, read-extent 0)
    are DROPPED from the bank rather than rejecting the whole candidate.  The pattern
    length is the observed read-coverage run (byte-exact: the bytes the player consumed
    as data).  Returns a candidate dict (with explicit ``pattern_lens``) or ``None``.

    A genuine orderlist->pattern walk is characterised by ROW INDEXING (the player
    indexes rows within a pattern via Y, so ``y_max > y_min``) and pattern REUSE (a few
    distinct patterns replayed across many advances -- the REPEAT structure HARD RULE #0
    expects).  A walk where almost every advance is a DISTINCT pointer with ``y == 0``
    is not an orderlist but a per-frame streaming pointer (a sample / wavetable cursor,
    e.g. a sample player walking a long PCM table one byte per frame); such a walk has
    no reuse to collapse and is REJECTED so the read-coverage path never fabricates a
    255-"pattern" pseudo-bank from a streaming cursor."""
    if walk.y_max <= walk.y_min:
        return None  # no within-pattern row indexing -> not an orderlist->pattern walk
    lo_img, hi_img = _image_bounds(d)
    read_play = (d.acc & ACC_READ_PLAY) != 0
    seq = walk.ptr_vals
    resolved = [(x - delta) & 0xFFFF for x in seq]
    ext = {}
    for a in sorted(set(resolved)):
        if lo_img <= a < hi_img and sdm[a]:
            e = _read_extent(read_play, a, lo_img, hi_img)
            if e > 0:
                ext[a] = e
    if len(ext) < 2:
        return None
    bank = sorted(ext)
    index_of = {a: i for i, a in enumerate(bank)}
    orderlist = [index_of.get(a, 0xFF) for a in resolved]
    cov = len(set(o for o in orderlist if o != 0xFF))
    if cov < 2:
        return None
    # require pattern REUSE: an orderlist replays its bank, so distinct patterns are a
    # MINORITY of the advances; a near-1:1 distinct/advance ratio (almost every advance
    # a fresh pointer) is a streaming cursor (a sample / wavetable walked once per
    # frame), not an orderlist -- reject only that degenerate ~no-reuse case (the y-span
    # guard above already drops the y==0 cursor; this catches a y-indexed cursor too).
    walked = sum(1 for o in orderlist if o != 0xFF)
    if 5 * cov > 4 * walked:
        return None
    return {
        "pattern_src": bank,
        "pattern_ptrs": [(a + delta) & 0xFFFF for a in bank],
        "pattern_lens": [ext[a] for a in bank],
        "orderlist": orderlist,
        "reloc_delta": delta,
        "grammar": None,  # no value-range grammar; lengths are explicit (read-coverage)
        "grammar_name": "readcov",
        "zp": walk.zp,
        "coverage": cov,
        "n_patterns": len(bank),
    }


def discover_patterns_pwlk(d):
    """Discover the pattern bank + orderlist from the (zp),Y pointer-walk capture
    (PWLK, recovery-offload item #2) -- the RESOLVED orderlist->pattern stream the
    player actually walked, reloc-applied and dialect-selected (S1+S2+S4 fused;
    replaces the O(image^2) brute-force scans).

    The PWLK ``ptr_vals`` is the sequence of pattern START addresses the orderlist
    advanced through.  For each (walk, relocation delta) we form TWO candidate banks
    and keep the higher-coverage one:

      * the VALUE-RANGE-dialect bank (:func:`_pwlk_candidate_eop`) -- NewPlayer / TFX /
        FC, sliced by the grammar EOP (byte-exact for the value-range dialects); and
      * the READ-COVERAGE bank (:func:`_pwlk_candidate_readcov`) -- GRAMMAR-AGNOSTIC,
        the pattern length taken from the observed read-coverage run, which sites the
        nibble / bit-packed dialects (GoatTracker, Soundmonitor, Music_Assembler) the
        value-range EOP cannot terminate, and DROPS phantom (non-data) pointers.

    Returns a dict with ``pattern_src`` (image addresses to read raw bytes from),
    ``pattern_ptrs`` (runtime addresses the player walked), ``pattern_lens`` (explicit
    per-pattern lengths for the read-coverage bank, ``[]`` for the EOP bank),
    ``orderlist`` (index sequence), ``reloc_delta``, ``grammar``; or ``None`` when no
    walk yields a pattern bank.  Across candidates the higher (coverage, then fewer
    patterns) wins -- the value-range dialect is preferred on a tie (it is the proven
    NewPlayer/TFX/FC path), so those tunes recover byte-identically to before."""
    if not getattr(d, "ptr_walks", None):
        return None
    sdm = d.song_data_mask()
    dialects = _dialects()
    deltas = reloc_delta_candidates(d)
    best = None
    for walk in d.ptr_walks:
        if not walk.is_load or len(set(walk.ptr_vals)) < 2:
            continue
        for delta in deltas:
            eop = _pwlk_candidate_eop(d, walk, delta, sdm, dialects)
            readcov = _pwlk_candidate_readcov(d, walk, delta, sdm)
            # (coverage, value-range-preferred, more-patterns): prefer the higher
            # orderlist coverage; on a coverage tie prefer the VALUE-RANGE dialect (so
            # NewPlayer/TFX/FC tunes recover byte-identically to before -- the read-
            # coverage bank only wins when no value-range dialect sliced the tune);
            # then prefer the larger bank (the original ``len(bank)`` tiebreak).
            for cand, vr in ((eop, 1), (readcov, 0)):
                if cand is None:
                    continue
                rank = (cand["coverage"], vr, cand["n_patterns"])
                if best is None or rank > best[0]:
                    best = (rank, cand)
    return best[1] if best is not None else None


def discover_pointer_table(d):
    """Discover the pattern-pointer table from SNAP + the access map, GENERICALLY.

    Signature: two equal-length adjacent byte regions ``lo[0..n)`` and ``hi[0..n)``
    such that every formed pointer ``lo[i] | hi[i]<<8`` (a) lands strictly past the
    table end and inside the loaded image, (b) is STRICTLY ASCENDING (a pattern bank
    laid out in order), (c) the hi-bytes are a few distinct page numbers, and (d) at
    least one target byte was READ as data during play (a pattern was traversed).

    Returns ``(lo_base, hi_base, n, ptrs)`` or ``None``.  The largest such table
    wins (the full pattern bank, not a short coincidental run)."""
    ram = d.ram
    read_play = (d.acc & ACC_READ_PLAY) != 0
    lo_img, hi_img = _image_bounds(d)
    best = None
    for hi_base in range(lo_img, hi_img - 2):
        for gap in range(4, 96):
            lo_base = hi_base - gap
            if lo_base < lo_img:
                continue
            n = gap
            if hi_base + n > hi_img:
                continue
            los = ram[lo_base : lo_base + n].astype(np.int64)
            his = ram[hi_base : hi_base + n].astype(np.int64)
            ptrs = los | (his << 8)
            if not ((ptrs >= hi_base + n) & (ptrs < hi_img)).all():
                continue
            if not np.all(np.diff(ptrs) > 0):
                continue
            if len(np.unique(his)) > 6:
                continue
            if read_play[ptrs].sum() < 1:
                continue
            if best is None or n > best[2]:
                best = (lo_base, hi_base, n, ptrs.tolist())
    return best


def discover_orderlist(d, n_patterns, pattern_data_lo):
    """Discover the per-voice orderlists from SNAP + the access map, GENERICALLY.

    A small stride-2 pointer table (3 voices) whose targets are 0xFF-terminated
    streams whose every byte is either a pattern index (``< n_patterns``) or a
    control marker (``>= 0x80`` -- REPEAT / TRANSPOSE / loop), located OUTSIDE the
    pattern-data region, with the stream bytes read as data.  Returns
    ``(ptr_table, ptrs, orderlists)`` or ``None``.

    The control markers ARE the REPEAT/TRANSPOSE structure the output-fit recovery
    throws away -- a pattern referenced from several orderlist slots is stored ONCE
    and replayed, the collapse that turns 319 distinct output rows into 33 reused
    patterns."""
    ram = d.ram
    read_play = (d.acc & ACC_READ_PLAY) != 0
    lo_img, hi_img = _image_bounds(d)
    cands = []
    for tbl in range(lo_img, hi_img - 6):
        ptrs = [ram[tbl + 2 * v] | (ram[tbl + 2 * v + 1] << 8) for v in range(3)]
        if not all(lo_img <= p < hi_img for p in ptrs):
            continue
        if any(pattern_data_lo <= p < hi_img for p in ptrs):
            continue  # targets must be orderlists, not pattern data
        ols, ok = [], True
        for p in ptrs:
            seq = []
            for k in range(128):
                b = int(ram[p + k])
                seq.append(b)
                if b == 0xFF:
                    break
            else:
                ok = False
                break
            if len(seq) < 2 or any(not ((b < n_patterns) or (b >= 0x80)) for b in seq):
                ok = False
                break
            ols.append(seq)
        if not ok:
            continue
        refs = set(b for o in ols for b in o if b < n_patterns)
        if len(refs) < 3:
            continue
        nread = sum(int(read_play[p : p + len(o)].sum()) for p, o in zip(ptrs, ols))
        if nread < 3:
            continue
        spread = max(ptrs) - min(ptrs)
        cands.append((nread, -spread, tbl, ptrs, ols))
    if not cands:
        return None
    cands.sort(reverse=True)
    _, _, tbl, ptrs, ols = cands[0]
    return tbl, ptrs, ols


def discover_instrument_table(d, sddf_slices):
    """Discover the instrument struct table base + stride from the SDDF backward
    slices, GENERICALLY (NO hardcoded address).

    The player loads each SID register (AD, SR, ...) for a voice from
    ``base + instr*stride + field_offset``.  The tracer's SDDF records, per
    SID-write PC, the RAM_READ leaf addresses its value flowed from.  A field of a
    stride-K table is a set of leaf addresses congruent mod K; K is the GCD of the
    sorted leaf-address differences, and the most common K across the write PCs is
    the instrument stride.  The base is the minimum member leaf.  Returns
    ``(base, stride)`` or ``None``.

    ``sddf_slices`` is the list of :class:`_SdwSlice` from :func:`read_sddf_slices`
    (empty for a pre-data-flow artifact, which yields ``None`` -- a tune whose
    instrument table cannot be sited from this artifact, surfaced not faked)."""
    if not sddf_slices:
        return None
    lo_img, hi_img = _image_bounds(d)
    pc_leaves = defaultdict(list)
    for sl in sddf_slices:
        for addr in sl.leaf_addrs:
            if lo_img <= addr < hi_img:
                pc_leaves[(sl.pc, sl.reg)].append(addr)
    strides, members = [], []
    for _key, addrs in pc_leaves.items():
        ua = sorted(set(addrs))
        if len(ua) < 3:
            continue
        g = int(np.gcd.reduce(np.diff(ua)))
        if g >= 2:
            strides.append(g)
            members.extend(ua)
    if not strides:
        return None
    K = Counter(strides).most_common(1)[0][0]
    base = min(members)  # the lowest field is the record base (field offset 0)
    return base, K


def decode_patterns(d, pattern_ptrs):
    """Walk the pattern-pointer chain and decode ``(note, instr, dur, cmd)`` rows.

    Stateful NewPlayer grammar (see module docstring): instrument / duration /
    command markers SET the running state, a note byte EMITS a row with the current
    state, ``0x7F`` ends the pattern.  Returns
    ``(patterns, instr_refs, notes, cmds, durs, n_rows, n_bytes)``.

    The decode is a pure function of the SNAP bytes; ``reencode_patterns`` inverts it
    byte-exact (the lossless-decode gate)."""
    ram = d.ram
    patterns = []
    instr_refs, notes, cmds, durs = set(), set(), set(), set()
    n_rows = n_bytes = 0
    for base in pattern_ptrs:
        idx = 0
        rows = []
        cur_instr = cur_dur = cur_cmd = None
        while idx < 0x400:
            b = int(ram[base + idx])
            idx += 1
            if b >= 0x80:
                if (b & 0xE0) == _MARK_DUR:
                    cur_dur = b & 0x1F
                    durs.add(cur_dur)
                elif (b & 0xE0) == _MARK_INSTR:
                    cur_instr = b & 0x1F
                    instr_refs.add(cur_instr)
                else:
                    cur_cmd = b & 0x3F
                    cmds.add(cur_cmd)
                continue
            if b == _END_OF_PATTERN:
                break
            if b not in _NON_PITCH:
                notes.add(b)
            rows.append((b, cur_instr, cur_dur, cur_cmd))
            n_rows += 1
        patterns.append(rows)
        n_bytes += idx
    return (
        patterns,
        sorted(instr_refs),
        sorted(notes),
        sorted(cmds),
        sorted(durs),
        n_rows,
        n_bytes,
    )


def _decode_groups(ram, base):
    """Decode one pattern into its SEMANTIC command-groups ``[(markers, note), ...]``
    where ``markers`` is the list of high-bit command bytes preceding a note byte.
    This is the structured representation; ``_encode_groups`` inverts it."""
    idx = 0
    groups = []
    markers = []
    while idx < 0x400:
        b = int(ram[base + idx])
        idx += 1
        if b >= 0x80:
            markers.append(b)
            continue
        groups.append((markers, b))
        markers = []
        if b == _END_OF_PATTERN:
            break
    return groups, idx


def _encode_groups(groups):
    """Re-emit the raw pattern byte stream from the decoded semantic groups."""
    out = []
    for markers, note in groups:
        out.extend(markers)
        out.append(note)
    return out


def reencode_patterns(ram_or_obj, pattern_ptrs):
    """The byte-exact lossless-decode gate: DECODE each pattern to its semantic
    command-groups, then RE-ENCODE those groups back to bytes and pair the result
    with the SNAP bytes.  Returns ``[(reencoded, snap), ...]`` per pattern.

    This is a genuine round-trip through the structured representation (not a
    re-walk): equality holds ONLY IF the grammar decode is lossless.  A driver whose
    patterns are not this grammar produces a mismatch and the structure is rejected
    (HARD RULE #0: the grammar is a hypothesis the round-trip falsifies).  Accepts a
    RAM array or any object exposing ``.ram``."""
    ram = getattr(ram_or_obj, "ram", ram_or_obj)
    out = []
    for base in pattern_ptrs:
        groups, idx = _decode_groups(ram, base)
        reencoded = _encode_groups(groups)
        snap = [int(x) for x in ram[base : base + idx]]
        out.append((reencoded, snap))
    return out


def pattern_roundtrip_ok(struct):
    """True iff every recovered pattern decodes and RE-ENCODES to its exact SNAP
    bytes -- the byte-exact lossless-decode gate over the whole pattern bank."""
    if struct.ram is None or not struct.pattern_ptrs:
        return False
    return all(
        em == sn for em, sn in reencode_patterns(struct.ram, struct.pattern_ptrs)
    )


def read_stsq_cells(distill_path):
    """The STSQ per-cell value sequences as ``{addr: (first_seen_frame, samples)}`` --
    a thin shim over the ONE reader (:func:`distill.load_distill`).  The porta /
    vibrato accumulator cells live here; empty if the artifact has no STSQ section or
    the path is not an artifact."""
    d = _load_distill_or_none(distill_path)
    if d is None:
        return {}
    return {c.addr: (c.first_seen, c.samples.astype(np.int64)) for c in d.stsq_cells}


def clean_pitches_residual(
    distill_path, freq_state, voices=((0, 1), (7, 8), (14, 15)), align=1
):
    """Subtract the captured porta/vibrato accumulators to recover TRUE grid pitches.

    The §state-machine identity: ``freq = note_seed + acc_a + acc_b (mod 2^16)`` where
    ``acc_a``/``acc_b`` are the captured 16-bit accumulator cells (porta, vibrato) and
    ``note_seed`` -- ``freq`` minus them -- is forced PIECEWISE-CONSTANT (one true grid
    pitch per note span).  For each voice this GENERICALLY picks the 1-2 accumulator
    cell pairs (from the reset-to-0, multi-valued STSQ candidates) that flatten ``freq``
    to the fewest piecewise-constant runs, then RENDERS ``note_seed + the accumulators``
    and compares to the ``.sidwr`` freq -- residual 0 proves the displaced "note table"
    (hundreds of entries, growing with playback) is really a handful of pitches plus two
    compact accumulator generators.

    ``freq_state`` is the ``(nframes, 25)`` register array.  Returns
    ``{voice: {"displaced": int, "pitches": int, "residual": int, "accs": [(lo,hi),...]}}``
    or ``None`` if the artifact has no STSQ section.  This is the in-tree port of the
    proven ``statemachine-proto/PROOF.py``."""
    cells = read_stsq_cells(distill_path)
    if not cells:
        return None
    state = np.asarray(freq_state, dtype=np.int64)
    n = state.shape[0]

    def grid(addr):
        first_seen, samples = cells[addr]
        out = np.zeros(n, dtype=np.int64)
        start = first_seen + align
        end = min(start + len(samples), n)
        out[start:end] = samples[: end - start]
        if end < n and len(samples):
            out[end:] = samples[-1]
        return out

    def acc16(lo, hi):
        return grid(lo) | (grid(hi) << 8)

    def n_changes(seq, a, b):
        return int(np.sum(np.diff(seq[a:b]) != 0))

    # accumulator-cell candidates: reset-to-0, multi-valued (the porta/vib integrators)
    acc_cells = [
        a
        for a, (_fs, s) in cells.items()
        if len(np.unique(s)) >= 5 and bool((s[:4] == 0).any())
    ]

    results = {}
    for vi, (rlo, rhi) in enumerate(voices):
        freq = state[:, rlo] | (state[:, rhi] << 8)
        a, b = 3, min(n, 514)
        # generically pick the 1-2 acc16 pairs minimizing piecewise-const breaks
        best = (n_changes(freq, a, b), [])
        for lo1 in acc_cells:
            for dh1 in (1, 2, 3):
                if lo1 + dh1 not in cells:
                    continue
                a1 = acc16(lo1, lo1 + dh1)
                c1 = n_changes((freq - a1) % 65536, a, b)
                if c1 < best[0]:
                    best = (c1, [(lo1, lo1 + dh1)])
                for lo2 in acc_cells:
                    if lo2 <= lo1:
                        continue
                    for dh2 in (1, 2, 3):
                        if lo2 + dh2 not in cells:
                            continue
                        a2 = acc16(lo2, lo2 + dh2)
                        c2 = n_changes((freq - a1 - a2) % 65536, a, b)
                        if c2 < best[0]:
                            best = (c2, [(lo1, lo1 + dh1), (lo2, lo2 + dh2)])
        accs = best[1]
        total_acc = np.zeros(n, dtype=np.int64)
        for lo, hi in accs:
            total_acc = (total_acc + acc16(lo, hi)) % 65536
        seed = (freq - total_acc) % 65536
        # render seed as piecewise-const + accumulators, compare byte-exact over [a,b)
        onsets = [a] + [a + 1 + i for i in np.nonzero(np.diff(seed[a:b]) != 0)[0]] + [b]
        seed_r = np.zeros(n, dtype=np.int64)
        for k in range(len(onsets) - 1):
            seed_r[onsets[k] : onsets[k + 1]] = seed[onsets[k]]
        freq_r = (seed_r + total_acc) % 65536
        residual = int(np.sum(freq_r[a:b] != freq[a:b]))
        results[vi] = {
            "displaced": int(len(np.unique(freq[a:b]))),
            "pitches": int(len(np.unique(seed[a:b]))),
            "residual": residual,
            "accs": [(int(lo), int(hi)) for lo, hi in accs],
        }
    return results


# Accumulator generator kinds (the BACC fits the STSQ porta/vibrato cells reduce to).
ACC_RAMP = 0  # value += rate              (porta / slide; integral of a constant add)
ACC_QUADRATIC = 1  # rate += accel; value += rate  (accelerating slide)
ACC_TRIANGLE = 2  # reflecting value += step in [lo,hi]  (vibrato)
ACC_RAW = 3  # no closed-form fit: store the sequence verbatim (the honest fallback)


def fit_accumulator(samples, width_mask=0xFFFF):
    """Fit a 16-bit accumulator value sequence to its GENERATOR (the AGENTS.md
    accumulator-fit: a stored ramp is unrecovered structure, HARD RULE #0).

    Tries RAMP / QUADRATIC / TRIANGLE (``_discover_njit.accfit_kernel``, the matcher
    form -- longest byte-exact prefix); returns ``(kind, seed, p1, p2, p3)`` for the
    fit that reproduces the WHOLE sequence byte-exact, preferring the cheapest
    (ramp < quadratic < triangle).  Returns ``(ACC_RAW, 0, 0, 0, 0)`` when none fits
    the full sequence (the sequence is then stored verbatim -- a falsifiable
    "needs another BACC archetype", never a wall)."""
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    s = np.asarray(samples, dtype=np.int64)
    n = len(s)
    if n < 2:
        return (ACC_RAW, 0, 0, 0, 0)
    for kind in (ACC_RAMP, ACC_QUADRATIC, ACC_TRIANGLE):
        seed, p1, p2, p3, m = DJ.accfit_kernel(s, n, width_mask, kind)
        if m == n:
            return (kind, int(seed), int(p1), int(p2), int(p3))
    return (ACC_RAW, 0, 0, 0, 0)


def accumulator_generators(distill_path, freq_state, voices=((0, 1), (7, 8), (14, 15))):
    """The per-voice FITTED freq-accumulator generators (S6 accumulator-fit).

    Picks the same flattening accumulator cells as :func:`clean_pitches_residual`,
    then FITS each chosen 16-bit accumulator (``lo | hi<<8``) to its ramp / quadratic
    / triangle generator instead of storing the raw value sequence.  Returns
    ``{voice: [(first_seen, kind, seed, p1, p2, p3, n_window, raw_or_None), ...]}`` --
    one entry per chosen accumulator; ``raw_or_None`` is the verbatim 16-bit sequence
    ONLY when no generator fit the full window (``kind == ACC_RAW``), so the byte-exact
    render is preserved either way (a fitted generator amortises to a handful of ints;
    a stored sequence is the honest, surfaced fallback).  ``None`` when the artifact
    has no STSQ section."""
    cpr = clean_pitches_residual(distill_path, freq_state, voices=voices)
    if cpr is None:
        return None
    cells = read_stsq_cells(distill_path)
    out = {}
    for vi in range(len(voices)):
        gens = []
        for lo, hi in cpr.get(vi, {}).get("accs", []):
            if lo not in cells or hi not in cells:
                continue
            fs_lo, samp_lo = cells[lo]
            fs_hi, samp_hi = cells[hi]
            m = min(len(samp_lo), len(samp_hi))
            acc16 = (samp_lo[:m] | (samp_hi[:m] << 8)).astype(np.int64)
            first_seen = int(min(fs_lo, fs_hi))
            kind, seed, p1, p2, p3 = fit_accumulator(acc16)
            raw = None if kind != ACC_RAW else [int(v) for v in acc16]
            gens.append((first_seen, kind, seed, p1, p2, p3, int(m), raw))
        out[vi] = gens
    return out


def _grammar_eops(grammar):
    """The set of end-of-pattern byte values for a grammar (``boundaries == K_EOP``),
    or the NewPlayer default ``{0x7F}`` for the legacy path (``grammar is None``)."""
    if grammar is None:
        return {_END_OF_PATTERN}
    bnd = grammar["boundaries"]
    return {b for b in range(256) if int(bnd[b]) == 4}  # K_EOP


def _explicit_pattern_lens(struct):
    """``{src_addr: length}`` from the structure's explicit ``pattern_lens`` (the
    read-coverage bank), or ``{}`` when lengths are EOP-derived (the value-range
    dialects / legacy path)."""
    lens = getattr(struct, "pattern_lens", None)
    if not lens:
        return {}
    return {int(a): int(l) for a, l in zip(struct.pattern_src, lens)}


def _pattern_len(ram, base, struct, max_bytes=0x400):
    """Byte length of the pattern at ``base``: the EXPLICIT read-coverage length when
    the structure carries one (the nibble / bit-packed dialects), else the grammar's
    EOP-inclusive slice (the value-range dialects / 0x7F for the legacy path);
    ``max_bytes`` if no EOP is hit."""
    explicit = _explicit_pattern_lens(struct)
    if base in explicit:
        return explicit[base]
    eops = _grammar_eops(struct.grammar)
    for k in range(max_bytes):
        if int(ram[(base + k) & 0xFFFF]) in eops:
            return k + 1
    return max_bytes


def _program_spans(d, struct):
    """The shared program-table spans (wave / pulse / filter / cmd) referenced once
    by the instruments, not re-derived per note.

    The generic bound is the bytes the player READ AS DATA during play
    (``ACC_READ_PLAY``, never written, never executed) RESTRICTED to the region OUTSIDE
    the structures already accounted for (instrument table, pattern-pointer table,
    pattern data, orderlists).  That residual READ data is exactly the shared
    wave/pulse/filter/command generator tables the instruments actually consumed.

    The read mask is the faithful program bound (HARD RULE #0: store the bytes the
    player READ, not the gap-filled eligible run).  :meth:`Distill.song_data_mask`
    keeps the UN-READ gaps inside an eligible run (for the player's SNAP round-trip),
    but those gap bytes were never consumed in the capture -- and on a sparse-SDDF tune
    whose instrument table could not be sited (so its code region is not carved out),
    the gap-filled eligible run swallows whole spans of UNEXECUTED-in-capture machine
    code + text, serializing thousands of bytes of non-program as "programs" (the
    HARD RULE #0 literal-floor trap).  Bounding to the READ bytes drops both: a
    never-read code/text gap is not part of the consumed program, and what remains is
    the actual table data the instruments walked.  Returns ``{name: (lo, hi)}``
    contiguous READ runs (derived from the access map, not hardcoded)."""
    # A fresh boolean array (``&`` of two arrays) -- ``carve`` mutates it in place.
    mask = ((d.acc & ACC_READ_PLAY) != 0) & d.eligible_mask()

    # carve out the structures already counted so programs are not double-charged
    def carve(a, b):
        mask[a:b] = False

    carve(
        struct.instr_base,
        struct.instr_base + struct.instr_stride * struct.n_instruments,
    )
    if struct.grammar is None and not getattr(struct, "pattern_lens", None):
        # legacy split-pointer layout: lo+hi tables then a contiguous pattern span.
        carve(struct.patptr_lo, struct.patptr_hi + struct.n_patterns)
        pat_lo = struct.patptr_hi + struct.n_patterns
        if struct.pattern_ptrs:
            last = max(struct.pattern_ptrs)
            pat_hi = last + _pattern_len(d.ram, last, struct)
            carve(pat_lo, pat_hi)
    else:
        # PWLK layout (value-range EOP slice OR explicit read-coverage length): each
        # pattern is its own span, scattered through song data.
        for base in struct.pattern_src:
            carve(base, base + _pattern_len(d.ram, base, struct))
    for p, o in zip(struct.orderlist_ptrs, struct.orderlists):
        carve(p, p + len(o))
    # remaining mask runs = the shared program tables
    idx = np.nonzero(mask)[0]
    spans = {}
    if not len(idx):
        return spans
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    for i, (s, e) in enumerate(zip(starts, ends)):
        spans[f"prog{i}"] = (int(idx[s]), int(idx[e]) + 1)
    return spans


def _decode_patterns_readcov(d, pattern_src, pattern_lens):
    """The read-coverage decode (``grammar is None``): the nibble / bit-packed dialects
    carry no value-range field partition to decode ``(note, instr, dur, cmd)`` tuples,
    so the structured row field is left empty -- the BYTE-EXACT pattern content is the
    raw read-coverage bytes (carried verbatim by the IR's ``pattern_bytes`` / the token
    budget's pattern span), and ``n_bytes`` is the observed read-coverage total.  A
    coarse ``note_table`` (the distinct low-value bytes, the plausible pitch field) is
    surfaced for reporting; it is not load-bearing for byte-exactness."""
    notes = set()
    n_bytes = 0
    for base, ln in zip(pattern_src, pattern_lens):
        for k in range(ln):
            b = int(d.ram[(base + k) & 0xFFFF])
            if 0 < b < 0x80:
                notes.add(b)
        n_bytes += int(ln)
    return ([], [], sorted(notes), [], [], 0, n_bytes)


def _decode_patterns_grammar(d, pattern_src, grammar, pattern_lens=None):
    """Decode the patterns at ``pattern_src`` under a chosen ``grammar`` (the four-
    dialect skeleton, ``_discover_njit.decode_pattern_kernel``).  Returns the same
    tuple as :func:`decode_patterns`.  The ``UNSET``/``REST_NOTE`` row sentinels are
    mapped to ``None``/``0x7E`` so the IR's tuple field matches the NewPlayer shape.

    When ``grammar is None`` (the read-coverage bank -- the nibble / bit-packed
    dialects) the decode defers to :func:`_decode_patterns_readcov`: there is no
    value-range partition, so the patterns serialize as raw read-coverage bytes."""
    if grammar is None:
        return _decode_patterns_readcov(d, pattern_src, pattern_lens or [])
    from preframr_tokens.bacc.generic import _discover_njit as DJ

    ram = np.asarray(d.ram, dtype=np.uint8)
    bnd = grammar["boundaries"]
    ow = grammar["op_width"]
    pm = grammar["param_mask"]
    pk = grammar["packing_mode"]
    patterns = []
    instr_refs, notes, cmds, durs = set(), set(), set(), set()
    n_rows = n_bytes = 0
    for base in pattern_src:
        nt, ins, du, cm, nr, nb = DJ.decode_pattern_kernel(
            ram, base, bnd, ow, pm, pk, 0x400
        )
        rows = []
        for k in range(nr):
            note = int(nt[k])
            ii = int(ins[k])
            dd = int(du[k])
            cc = int(cm[k])
            if ii != DJ.UNSET:
                instr_refs.add(ii)
            if dd != DJ.UNSET:
                durs.add(dd)
            if cc != DJ.UNSET:
                cmds.add(cc)
            note_out = note
            if note == DJ.REST_NOTE:
                note_out = 0x7E
            elif note not in (0x00, 0x7E):
                notes.add(note)
            rows.append(
                (
                    note_out,
                    None if ii == DJ.UNSET else ii,
                    None if dd == DJ.UNSET else dd,
                    None if cc == DJ.UNSET else cc,
                )
            )
        patterns.append(rows)
        n_rows += nr
        n_bytes += int(nb)
    return (
        patterns,
        sorted(instr_refs),
        sorted(notes),
        sorted(cmds),
        sorted(durs),
        n_rows,
        n_bytes,
    )


def _fill_instruments(d, struct, sddf_slices, instr_refs):
    """Populate the instrument table (S5): the stride from the instrument-feeding
    SDDF leaf-lattice GCD, the used-instrument records read at ``base + i*stride``."""
    it = discover_instrument_table(d, sddf_slices)
    if it is None:
        return
    struct.instr_base, struct.instr_stride = it
    used = [i for i in instr_refs if i != _INSTR_KILL]
    struct.n_instruments = (max(used) + 1) if used else 0
    struct.instr_records = [
        [
            int(x)
            for x in d.ram[
                struct.instr_base
                + i * struct.instr_stride : struct.instr_base
                + i * struct.instr_stride
                + struct.instr_stride
            ]
        ]
        for i in range(struct.n_instruments)
    ]


def _recover_structure_pwlk(d, struct, sddf_slices):
    """The PWLK-driven structure path (S1+S2+S4): pattern bank + orderlist from the
    resolved (zp),Y walk, reloc-applied, grammar-selected by the byte-exact slice.
    Returns True on success (``struct`` filled, ``ok`` set), else False."""
    pw = discover_patterns_pwlk(d)
    if pw is None:
        return False
    struct.reloc_delta = pw["reloc_delta"]
    struct.pattern_src = pw["pattern_src"]
    struct.pattern_ptrs = pw["pattern_src"]  # read raw bytes from the image addrs
    struct.pattern_lens = pw["pattern_lens"]  # explicit (read-coverage) or [] (EOP)
    struct.n_patterns = pw["n_patterns"]
    struct.grammar = pw["grammar"]
    struct.orderlists = [pw["orderlist"]]
    struct.patptr_lo = min(pw["pattern_src"])
    struct.patptr_hi = max(pw["pattern_src"])

    patterns, instr_refs, notes, cmds, durs, n_rows, n_bytes = _decode_patterns_grammar(
        d, pw["pattern_src"], pw["grammar"], pw["pattern_lens"]
    )
    struct.patterns = patterns
    struct.instr_refs = instr_refs
    struct.note_table = notes
    struct.commands = cmds
    struct.durations = durs
    struct.n_rows = n_rows
    struct.pattern_bytes = n_bytes

    _fill_instruments(d, struct, sddf_slices, instr_refs)
    struct.program_spans = _program_spans(d, struct)
    struct.ok = True
    return True


def _recover_structure_legacy(d, struct, sddf_slices):
    """The split-pointer-scan structure path (the original NewPlayer recovery), kept
    as a fallback for tunes whose distill has no usable (zp),Y walk capture.  Returns
    True on success."""
    pt = discover_pointer_table(d)
    if pt is None:
        return False
    lo_base, hi_base, n, ptrs = pt
    struct.patptr_lo, struct.patptr_hi, struct.n_patterns = lo_base, hi_base, n
    struct.pattern_ptrs = ptrs
    struct.pattern_src = ptrs

    pattern_data_lo = hi_base + n
    patterns, instr_refs, notes, cmds, durs, n_rows, n_bytes = decode_patterns(d, ptrs)
    struct.patterns = patterns
    struct.instr_refs = instr_refs
    struct.note_table = notes
    struct.commands = cmds
    struct.durations = durs
    struct.n_rows = n_rows
    struct.pattern_bytes = n_bytes

    for emitted, snap in reencode_patterns(d, ptrs):
        if emitted != snap:
            struct.reason = "pattern decode not byte-exact (grammar mismatch)"
            return False

    ol = discover_orderlist(d, n, pattern_data_lo)
    if ol is not None:
        struct.orderlist_ptr_table, struct.orderlist_ptrs, struct.orderlists = ol

    _fill_instruments(d, struct, sddf_slices, instr_refs)
    struct.program_spans = _program_spans(d, struct)
    struct.ok = True
    return True


def recover_structure(distill_path):
    """Generic end-to-end structure recovery from a ``.distill.bin`` SDST artifact.

    Both structure paths are tried and the byte-exact candidate with the FEWEST
    tokens is SELECTED (the design's validation-gated, fewest-tokens tiebreak):
      * the PWLK-driven path (S1+S2+S4: the resolved (zp),Y orderlist->pattern walk,
        reloc-applied, grammar-selected by the byte-exact slice) -- subsumes
        relocation + the 4 row grammars + the interleaved/scattered packings the
        brute-force scan cannot reach; and
      * the split-pointer scan (the original NewPlayer recovery) -- a full contiguous
        pattern bank that can be more compact when the walk traversed only a subset.
    Returns a :class:`RecoveredStructure`; ``ok`` is False (with a ``reason``) when
    neither path yields a structure -- a pure-code tune (A Mind Is Born) -- and the
    caller falls back to the generator cover.  ``ok`` True means the byte-exact
    slice/re-encode gate held.
    """
    d = load_distill(distill_path)
    sddf_slices = read_sddf_slices(distill_path)

    def _fresh():
        return RecoveredStructure(
            ok=False,
            load_addr=d.load_addr,
            load_len=d.load_len,
            nframes=d.nframes,
            distill_path=distill_path,
            ram=d.ram,
        )

    candidates = []
    for builder in (_recover_structure_pwlk, _recover_structure_legacy):
        s = _fresh()
        if builder(d, s, sddf_slices):
            try:
                total, _ = token_budget(s)
            except (ValueError, IndexError):
                total = float("inf")
            candidates.append((total, s))
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    s = _fresh()
    s.reason = "no pattern-pointer table (likely pure-code tune)"
    return s


# ---------------------------------------------------------------------------
# Token budget for the structured IR (the floor the output-fit recovery misses).
# ---------------------------------------------------------------------------
def _tok(v):
    """base16-LEB token cost of a small value: 1 token for a nibble, 2 for a byte."""
    return 1 if 0 <= v < 16 else 2


def _flat_token_stream(struct, ram):
    """The structured IR as ONE flat value-token stream, in canonical order: note
    table, deduped instrument pool, shared program tables (referenced once), the raw
    pattern byte streams (the player's already-compact stateful row encoding), then
    the orderlists.  This is the lossless re-expression of the song-data the player
    reads -- emitted ONCE per shared structure, the dedup/sharing the output-fit
    recovery never does.  Returns ``(stream, section_lengths)``."""
    stream = []
    sec = {}

    n0 = len(stream)
    stream += [int(v) for v in struct.note_table]
    sec["note_table"] = len(stream) - n0

    n0 = len(stream)
    seen, pool = {}, []
    for rec in struct.instr_records:
        key = tuple(rec)
        if key not in seen:
            seen[key] = len(pool)
            pool.append(rec)
    for rec in pool:
        stream += [int(x) for x in rec]
    sec["instr_pool"] = len(stream) - n0
    sec["_n_instruments"] = len(pool)

    n0 = len(stream)
    if ram is not None:
        for _name, (a, b) in struct.program_spans.items():
            stream += [int(x) for x in ram[a:b]]
    sec["shared_programs"] = len(stream) - n0

    n0 = len(stream)
    if ram is not None:
        explicit = _explicit_pattern_lens(struct)
        eops = _grammar_eops(getattr(struct, "grammar", None))
        for base in struct.pattern_src or struct.pattern_ptrs:
            if base in explicit:
                # read-coverage bank: the pattern is exactly its observed-read run.
                for k in range(explicit[base]):
                    stream.append(int(ram[(base + k) & 0xFFFF]))
                continue
            idx = 0
            while idx < 0x400:
                b = int(ram[(base + idx) & 0xFFFF])
                stream.append(b)
                idx += 1
                if b in eops:
                    break
    sec["pattern_rows"] = len(stream) - n0

    n0 = len(stream)
    for o in struct.orderlists:
        stream += [int(b) for b in o]
    sec["orderlist"] = len(stream) - n0

    return stream, sec


def _backward_lz(tokens, min_match=3, window=4096):
    """Greedy backward-LZ over a token stream (the REPEAT/TRANSPOSE collapse): count
    literals + match-refs.  A match (a pattern replayed, a repeated orderlist run, a
    shared program fragment) costs ~2 tokens (offset + length); a literal ~1.  This is
    the factoring the output-fit recovery cannot do because it has no shared
    structure to point back to.  Returns ``(n_literals, n_matches)``."""
    i, n = 0, len(tokens)
    literals = matches = 0
    while i < n:
        best = 0
        lo = max(0, i - window)
        for j in range(lo, i):
            length = 0
            while (
                i + length < n
                and tokens[j + length] == tokens[i + length]
                and length < 255
            ):
                length += 1
            if length > best:
                best = length
        if best >= min_match:
            matches += 1
            i += best
        else:
            literals += 1
            i += 1
    return literals, matches


def token_budget(struct, frames=None):
    """The structured tracker-IR token budget.  Returns ``(total, breakdown)``.

    Two numbers, both honest: the UN-LZ ``flat`` count (one base16-LEB token per
    song-data byte -- the shared structures emitted once) and the backward-LZ
    ``total`` (the REPEAT/TRANSPOSE collapse: 33 reused patterns across 81 orderlist
    slots, repeated rows, shared program fragments).  The LZ total is the floor the
    output-fit recovery misses.  ``frames`` overrides the artifact's requested
    ``nframes`` with the true playback length (the ``.sidwr`` row count)."""
    ram = struct.ram
    stream, sec = _flat_token_stream(struct, ram)

    flat = sum(_tok(t) for t in stream)
    lit, mat = _backward_lz(stream)
    total = lit + 2 * mat
    nframes = frames or struct.nframes

    breakdown = {
        "note_table": sec["note_table"],
        "instr_pool": sec["instr_pool"],
        "shared_programs": sec["shared_programs"],
        "pattern_rows": sec["pattern_rows"],
        "orderlist": sec["orderlist"],
        "n_pitches": len(struct.note_table),
        "n_instruments": sec["_n_instruments"],
        "n_rows": struct.n_rows,
        "n_patterns": struct.n_patterns,
        "flat_bytes": len(stream),
        "flat_tokens": flat,
        "lz_literals": lit,
        "lz_matches": mat,
        "total": total,
        "tok_per_frame": total / nframes if nframes else float("inf"),
    }
    return total, breakdown


def recover_and_budget(distill_path, frames=None):
    """Convenience: recover the structure and return ``(structure, total_tokens,
    breakdown)``.  ``frames`` is the true playback length (``.sidwr`` row count) for
    the per-frame metric; defaults to the artifact's requested ``nframes``."""
    struct = recover_structure(distill_path)
    if struct.ok:
        total, breakdown = token_budget(struct, frames=frames)
        return struct, total, breakdown
    return struct, None, None
