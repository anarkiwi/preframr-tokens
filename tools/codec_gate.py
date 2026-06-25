"""Mechanical 'done' gate for the GENERIC codec path.

'Done' is NOT an agent's judgment. It is, objectively:
  residual == 0 (byte-exact)  AND  tok/frame < threshold  AND  ~0 literal-floor.
Run this; a FAIL on tok/frame means the remaining tokens are UNRECOVERED STRUCTURE
(phrases / instruments / generators), never an 'entropy wall'
(see /scratch/anarkiwi/preframr/preframr-xpt/AGENTS.md HARD RULE #0).

    SIDTRACE_BIN=.../sidtrace PYTHONPATH=. python tools/codec_gate.py <tune.sid> [subtune] [nframes]
"""

import sys

import numpy as np

# --- C1-C8 anti-Goodhart structural constraints (flat alphabet) -------------
# FLAT_VOCAB_MIGRATION.md Sec "Anti-Goodhart structural constraints (C1-C7)" + C8.
# These are the REAL gate: each closes one degenerate solution (a dumped lane, an
# LZ offset, a raw-value escape, a stored generator). tok/frame is a REPORTED
# metric, never a pass/fail. The checks below are the ones that are well-defined on
# the flat id stream / alphabet TODAY (C3, C7, C8-no-escape); the render-from-tokens
# per-lane checks (C1/C2/C5/C6) attach to the generic flat path as it lands.


class CheckFailure(AssertionError):
    """A C-check rejected the stream; the message names the offending atom/lane."""


def c3_no_lz_offset_tokens():
    """C3 -- no LZ: the flat alphabet contains NO offset / REPEAT / TRANSPOSE token.
    Repetition is content-addressed (REF / INSTR_REF NAMES), never a back-offset.
    This is a structural invariant on the VOCAB, not a per-stream test."""
    from preframr_tokens.bacc import flat_serialize as F

    # EXACT names: the v1 LZ markers were `REPEAT` / `TRANSPOSE`; an offset/copy/
    # backref token would be `OFFSET` / `COPY` / `BACKREF`. (ORDER_REPEAT is an
    # orderlist loop COUNT and ORDER_TRANSPOSE a semitone shift -- content, not a
    # back-offset -- so they are NOT forbidden; REF is a content-addressed NAME.)
    forbidden = {"REPEAT", "TRANSPOSE", "OFFSET", "COPY", "BACKREF", "LZ"}
    for name in dir(F):
        if name in forbidden and isinstance(getattr(F, name), int):
            raise CheckFailure(f"C3: LZ/offset token {name} present in flat VOCAB")
    return True


def c3_no_lz_in_measured_stream(ids):
    """C3 (shipped-stream invariant) -- the ENTIRE gate-measured shipped stream carries NO
    general-compression / back-reference token, in ANY section regardless of the path that
    produced it.

    The campaign mandate is NO LZ (or any equivalent general repetition-compression of the
    output symbol stream) in the MEASURED path -- the tok/frame the gate certifies must come
    from RECOVERED STRUCTURE, never from compressing the serialized stream.  The codec's
    backward-LZ ``_struct_lz`` emits the reserved sentinel ``_REPEAT`` (``_REPEAT, off,
    len``) for every copy; an equivalent learned-dictionary / grammar scheme (Re-Pair, LZ,
    any back-reference) likewise introduces reserved high sentinels above the codec's literal
    range.  The ban is on the MECHANISM (general compression of the stream), not a single
    token id: a bounded AUTHORED-vocabulary reference (an INSTR_REF, an orderlist
    ``ORDER_REPEAT`` loop-count, a content-addressed pattern REF) is a literal in the codec's
    own value range and is allowed; an unbounded compression sentinel is not.

    Earlier this ban applied ONLY to the tagged ``note_bases`` / ``nonfreq`` sections, while
    the pattern-bank sections were permitted to use ``_struct_lz``.  But the gate measures the
    SHIPPED representation -- and when a tune ships the pattern-bank path its sub-1 tok/frame
    rode entirely on ``_struct_lz`` there (the loophole: the ban sat on a path that did not
    ship).  This now scans the WHOLE shipped id stream and FAILS on ANY occurrence of a
    reserved compression sentinel (``_REPEAT`` or any token at/above ``_REPEAT`` that is not a
    section frame marker), so the no-LZ ban applies to whatever actually ships.

    A FAIL here means the certified tok/frame RODE ON compression -- the structure was not
    recovered (HARD RULE #0).  Returns True when the shipped stream is compression-free.  Do
    NOT relax this by re-permitting a section; the fix is to RECOVER the structure so the
    shipped stream needs no ``_struct_lz``."""
    from preframr_tokens.bacc.generic import structure_ir as SI

    if not ids:
        return True
    # Section frame markers are reserved high sentinels but are STRUCTURE, not compression;
    # everything else at/above _REPEAT is a back-reference / learned-dictionary symbol.
    frame_markers = {SI._SEC_END, SI._SEC_NOTE_BASES, SI._SEC_NONFREQ}
    for tok in ids:
        if tok == SI._REPEAT:
            raise CheckFailure(
                "C3: shipped stream contains _REPEAT (the _struct_lz back-offset) -- the "
                "certified tok/frame rides on LZ in SOME shipped section (pattern-bank "
                "included), the structure was not recovered (HARD RULE #0). Recover the "
                "structure (instrument-program execution / content-addressed authored REFs) "
                "so the shipped stream needs no _struct_lz; do NOT re-permit a section."
            )
        if tok >= SI._REPEAT and tok not in frame_markers:
            raise CheckFailure(
                f"C3: shipped stream contains reserved compression sentinel {tok} -- a "
                "general learned-dictionary / grammar / back-reference over the symbol "
                "stream (Re-Pair / LZ), banned regardless of token id (the MECHANISM is "
                "what is forbidden, HARD RULE #0)."
            )
    return True


def c3_no_raw_value_stream(ir, nframes):
    """C3 (generic render path) -- no raw per-frame VALUE stream: inspect the recovered
    ``note_bases`` / ``nonfreq`` REPRESENTATION TAGS and FAIL on any record that stores a
    raw per-frame freq/register value stream of length ~nframes (a relabeled literal-floor
    dump), rather than a GENERATOR / TABLE / content-addressed REF / sparse change-point
    program.  HARD RULE #0: the recovered structure is the player's program, never its
    output column.

    A note base must be a known kind (RAMP16 ramp generator / TABLE+idx-walk / ARP per-onset
    ref pool); a non-freq lane a known kind (RAMP16 / sparse CP / per-onset SEG ref pool).
    A per-frame dump is exactly ``nframes`` long (the banned ``_NB_RAWLZ`` / value-LZ /
    register-log reproduction); the authored-level streams (per-onset bases / refs / a small
    distinct-shape pool, the sparse change-points) are all STRICTLY shorter than the frame
    count by construction (fewer notes than frames), so the threshold is ``nframes``."""
    from preframr_tokens.bacc.generic import structure_ir as SI

    limit = max(8, int(nframes))

    def _dump_field(rec):
        # any list/tuple field as long as the frame count is a raw per-frame value stream
        # (the authored-level base/ref/pool streams are all shorter -- one per note, not
        # per frame).  Nested pools (a list of shape tuples) are judged by their OUTER
        # length (the distinct-shape count, bounded small), not the per-frame extent.
        for fld in rec:
            if isinstance(fld, (list, tuple)) and len(fld) >= limit:
                return len(fld)
        return 0

    nb_kinds = {SI._NB_RAMP16, SI._NB_TABLE, SI._NB_ARP}
    for vi, rec in enumerate(ir.note_bases):
        if not rec or rec[0] not in nb_kinds:
            raise CheckFailure(
                f"C3: note_base voice {vi} kind {rec[0] if rec else None!r} is not a "
                "generator/table/ref representation (a raw per-frame freq stream)"
            )
        big = _dump_field(rec)
        if big:
            raise CheckFailure(
                f"C3: note_base voice {vi} stores a raw per-frame value stream "
                f"(field len {big} >= {limit}) -- the banned literal-floor dump"
            )
    lane_kinds = {SI._LANE_CP, SI._LANE_RAMP16, SI._LANE_SEG}
    for rec in ir.nonfreq:
        if not rec or rec[0] not in lane_kinds:
            raise CheckFailure(
                f"C3: non-freq lane kind {rec[0] if rec else None!r} is not a "
                "generator/sparse-CP/ref representation (a raw per-frame register stream)"
            )
        # a SPARSE _LANE_CP is the player's real sets-and-holds (bounded by the cap); a CP
        # whose change-point stream spans ~nframes is a per-frame dump.
        if rec[0] == SI._LANE_CP and len(rec[2]) >= limit:
            raise CheckFailure(
                f"C3: non-freq lane {rec[1]} CP has {len(rec[2])} change points "
                f">= {limit} -- a per-frame dump, not a sparse sets-and-holds"
            )
        big = _dump_field(rec)
        if big:
            raise CheckFailure(
                f"C3: non-freq lane {rec[1]} stores a raw per-frame value stream "
                f"(field len {big} >= {limit}) -- the banned literal-floor dump"
            )
    return True


def c8_no_escape_tokens():
    """C8 -- no wide values, no escapes: the flat alphabet has no u16-escape /
    raw-Fn / DUR_LONG / NOTE_RAWFN / nframes-wide token. A 16-bit field is a fixed
    (lo, hi) BYTE pair (positional, not a varint); there is no length-prefixed
    escape. Structural invariant on the VOCAB."""
    from preframr_tokens.bacc import flat_serialize as F

    # EXACT names: an escape/wide field would be ESCAPE / NOTE_RAWFN / DUR_LONG /
    # a varint/LEB digit token. (NOTE_RAW is a raw-note KIND marker -- a single
    # note byte, bounded 0..255 -- not a wide value, so it is allowed.)
    forbidden = {"ESCAPE", "NOTE_RAWFN", "DUR_LONG", "WIDE", "VARINT", "LEB"}
    for name in dir(F):
        if name in forbidden and isinstance(getattr(F, name), int):
            raise CheckFailure(f"C8: escape/wide token {name} present in flat VOCAB")
    return True


def c7_byte_atom_fraction(ids, cap=0.5):
    """C7 -- BYTE-atom fraction cap (cheap canary). A genuine stream is mostly
    NOTE/INSTR_REF/CMD/GEN_*/structural atoms; a relabeled dump is mostly raw BYTE.
    Returns (ok, fraction)."""
    from preframr_tokens.bacc import flat_serialize as F

    if not ids:
        return True, 0.0
    n_byte = sum(1 for t in ids if F.BYTE_BASE <= t < F.BYTE_BASE + F.BYTE_SPAN)
    frac = n_byte / len(ids)
    return frac < cap, frac


def c8_field_tail(values, top_k=8, cover=0.95):
    """C8 per-field tail check: a numeric field's top-K atoms must cover >= ``cover``
    of its occurrences (a fat head / short tail). A wide or long-tailed field is a
    MISSING generator, never an escape. Returns (ok, coverage, distinct)."""
    if not values:
        return True, 1.0, 0
    from collections import Counter

    counts = Counter(values)
    total = len(values)
    top = sum(c for _, c in counts.most_common(top_k))
    return top / total >= cover, top / total, len(counts)


def flat_structural_checks(ids):
    """Run the alphabet-level C-checks (C3, C7, C8-no-escape) on a flat id stream.
    Returns a metrics dict; raises CheckFailure on a hard structural violation
    (C3/C8 escape). C7 is reported (a soft canary on small in-process streams)."""
    c3_no_lz_offset_tokens()
    c8_no_escape_tokens()
    c7_ok, byte_frac = c7_byte_atom_fraction(ids)
    return {
        "c3_no_lz": True,
        "c8_no_escape": True,
        "c7_byte_fraction": round(byte_frac, 3),
        "c7_ok": c7_ok,
    }


def gate_state(state, max_tok_per_frame=1.0):
    """Gate a per-frame (nframes, 25) state: (ok, metrics).  Byte-exact via the
    lift/unlift round-trip; tok/frame from the production min(pitch, plain) build."""
    from preframr_tokens.bacc.generic.tracker import render_from_fits
    from preframr_tokens.bacc.tracker_ir import lift, unlift
    from preframr_tokens.bacc.tracker_serialize import _ir_to_ids

    nf = len(state)
    boot = [int(v) for v in state[0]]
    tokens = min(
        len(_ir_to_ids(lift(state, None, nf, boot, synth_pitch=sp)))
        for sp in (True, False)
    )
    ir = lift(state, None, nf, boot)
    gen, ev = unlift(ir)
    resid = int(np.sum(render_from_fits(gen, ev, ir.note_table, nf) != state))
    tpf = tokens / nf if nf else float("inf")
    ok = resid == 0 and tpf < max_tok_per_frame
    return ok, {
        "resid": resid,
        "tok_per_frame": round(tpf, 3),
        "nframes": nf,
        "tokens": tokens,
    }


def gate_sid(sid, subtune=1, nframes=2500, max_tok_per_frame=1.0):
    """Gate a ``.sid`` through the PRODUCTION recovery (:func:`recover_tune`): the
    S0-S7 structure path (byte-exact + value-LZ'd) with the generator-cover fallback,
    whichever is the smaller byte-exact cover.  Byte-exactness is checked by re-
    rendering the chosen path against the ``.sidwr`` state."""
    import numpy as _np

    from preframr_tokens.bacc.generic.recover import recover_tune
    from preframr_tokens.bacc.generic.structure_ir import (
        render_structure,
        structure_ir_from_ids,
    )

    ids, kind, state = recover_tune(sid, subtune, nframes)
    nf = len(state)
    if nf < 2:
        return False, {
            "resid": -1,
            "tok_per_frame": float("inf"),
            "nframes": 0,
            "tokens": 0,
        }
    tpf = len(ids) / nf
    if kind == "structure":
        ir = structure_ir_from_ids(ids)
        # The C3 structural checks are HARD gate failures (HARD RULE #0): a violation means
        # the certified tok/frame is not backed by recovered structure.  A CheckFailure is
        # surfaced as a clean FAIL result (not an uncaught exception) so the gate is a usable
        # red/green signal -- the message names the offending mechanism.
        try:
            # C3 (generic render path): the recovered representation must be a
            # generator/table/ref/sparse-CP, never a raw per-frame value stream.
            c3_no_raw_value_stream(ir, nf)
            # C3 (shipped-stream): the WHOLE shipped stream must be LZ-free -- a tok/frame
            # that rides on _struct_lz in ANY shipped section (pattern-bank included) is
            # rejected.
            c3_no_lz_in_measured_stream(ids)
        except CheckFailure as exc:
            return False, {
                "resid": -1,
                "tok_per_frame": round(tpf, 3),
                "nframes": nf,
                "tokens": len(ids),
                "kind": kind,
                "c3": str(exc),
            }
        ir._state = state  # pylint: disable=protected-access
        resid = int(_np.sum(render_structure(ir) != state))
    else:
        resid = gate_state(state, max_tok_per_frame)[1]["resid"]
    ok = resid == 0 and tpf < max_tok_per_frame
    return ok, {
        "resid": resid,
        "tok_per_frame": round(tpf, 3),
        "nframes": nf,
        "tokens": len(ids),
        "kind": kind,
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    sub = int(argv[2]) if len(argv) > 2 else 1
    nf = int(argv[3]) if len(argv) > 3 else 2500
    # C3 / C8 are alphabet invariants (no stream needed); fail hard if violated.
    try:
        c3_no_lz_offset_tokens()
        c8_no_escape_tokens()
        print("C3 (no LZ/offset tokens): PASS   C8 (no escape/wide tokens): PASS")
    except CheckFailure as exc:
        print("STRUCTURAL CHECK FAILED:", exc)
        return 1
    ok, m = gate_sid(argv[1], sub, nf)
    print(("PASS" if ok else "FAIL"), m)
    if m["resid"] != 0:
        print(
            "  byte-exact FAILED -- a missing PARAMETER of a known generator. STOP and diagnose."
        )
    if m["tok_per_frame"] >= 1.0:
        print(
            "  >= 1 tok/frame -- the remaining tokens are UNRECOVERED STRUCTURE (phrases / instruments"
            " / generators), NOT an entropy wall. Run the HARD RULE #0 falsification protocol."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
