"""PERMANENT GATE — Monty-on-the-Run encodes at < 1 token / frame, losslessly.

Recovers a BACC program from the Monty_on_the_Run (sub 1) .sid by running its
playroutine, verifies the program regenerates the dump RESIDUAL-ZERO (byte-exact),
and FAILS unless the BACC token log is < 1 token / frame. This is the executable
form of the AGENTS.md headline goal. The lossless check is part of the gate: a
lossy codec trivially hits any budget, so the budget only means something when
residual = 0.

This test MUST pass, MUST run in CI, and may NEVER be removed, skipped, xfailed,
or bypassed. Both fixtures are auto-acquired (the .sid is downloaded if absent and
the dump rendered from it in the anarkiwi/headlessvice container) — there is no
skip path.
"""

import os

from preframr_tokens import CPF, measure, recover_program, verify_residual
from tests._dump_fixture import acquire

_MONTY_REL = "MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid"
_MONTY_URL = os.environ.get(
    "MONTY_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
)


def test_monty_under_one_token_per_frame():
    sid, dump = acquire(_MONTY_REL, _MONTY_URL, subtune=1)
    assert verify_residual(
        sid, dump, CPF
    ), "BACC codec is NOT residual-zero (lossless) on Monty"
    program = recover_program(sid, dump, CPF)
    brk, frames = measure(program)
    tokens = brk["total"]
    assert tokens / frames < 1.0, (
        f"Monty: {tokens} tokens / {frames} frames = {tokens / frames:.3f} tok/frame "
        f">= 1.0 (must be < 1 token/frame, residual-zero)"
    )


if __name__ == "__main__":
    test_monty_under_one_token_per_frame()
    print("PASS")
