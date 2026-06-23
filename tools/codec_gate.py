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
