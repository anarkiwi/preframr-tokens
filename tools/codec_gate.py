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
    if kind == "structure":
        ir = structure_ir_from_ids(ids)
        ir._state = state  # pylint: disable=protected-access
        resid = int(_np.sum(render_structure(ir) != state))
    else:
        resid = gate_state(state, max_tok_per_frame)[1]["resid"]
    tpf = len(ids) / nf
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
