"""Mechanical 'done' gate for the GENERIC codec path.

'Done' is NOT an agent's judgment. It is, objectively:
  residual == 0 (byte-exact)  AND  tok/frame < threshold  AND  ~0 literal-floor.
Run this; a FAIL on tok/frame means the remaining tokens are UNRECOVERED STRUCTURE
(phrases / instruments / generators), never an 'entropy wall'
(see /scratch/anarkiwi/preframr/preframr-xpt/AGENTS.md HARD RULE #0).

    SIDTRACE_BIN=.../sidtrace PYTHONPATH=. python tools/codec_gate.py <tune.sid> [subtune] [nframes]
"""

import os
import sys
import tempfile

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
        len(_ir_to_ids(lift(state, None, nf, boot, synth_pitch=sp))) for sp in (True, False)
    )
    ir = lift(state, None, nf, boot)
    gen, ev = unlift(ir)
    resid = int(np.sum(render_from_fits(gen, ev, ir.note_table, nf) != state))
    tpf = tokens / nf if nf else float("inf")
    ok = resid == 0 and tpf < max_tok_per_frame
    return ok, {"resid": resid, "tok_per_frame": round(tpf, 3), "nframes": nf, "tokens": tokens}


def gate_sid(sid, subtune=1, nframes=2500, max_tok_per_frame=1.0):
    from preframr_tokens.bacc.generic.sidtrace import run_sidtrace, sidwr_state

    pre = os.path.join(tempfile.mkdtemp(), "t")
    sw, _ = run_sidtrace(sid, pre, subtune, nframes)
    state, _ = sidwr_state(sw)
    if state is None or len(state) < 2:
        return False, {"resid": -1, "tok_per_frame": float("inf"), "nframes": 0, "tokens": 0}
    return gate_state(state, max_tok_per_frame)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    sub = int(argv[2]) if len(argv) > 2 else 1
    nf = int(argv[3]) if len(argv) > 3 else 2500
    ok, m = gate_sid(argv[1], sub, nf)
    print(("PASS" if ok else "FAIL"), m)
    if m["resid"] != 0:
        print("  byte-exact FAILED -- a missing PARAMETER of a known generator. STOP and diagnose.")
    if m["tok_per_frame"] >= 1.0:
        print(
            "  >= 1 tok/frame -- the remaining tokens are UNRECOVERED STRUCTURE (phrases / instruments"
            " / generators), NOT an entropy wall. Run the HARD RULE #0 falsification protocol."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
