"""PERMANENT GATE — Monty-on-the-Run encodes at < 1 token / frame, losslessly.

Recovers a BACC program from the Monty_on_the_Run (sub 1) .sid by running its
playroutine, verifies the program regenerates the dump RESIDUAL-ZERO (byte-exact),
and FAILS unless the BACC token log is < 1 token / frame. This is the executable
form of the AGENTS.md headline goal. The lossless check is part of the gate: a
lossy codec trivially hits any budget, so the budget only means something when
residual = 0.

This test MUST pass, MUST run in CI, and may NEVER be removed, skipped, xfailed,
or bypassed. Both fixtures are auto-acquired (the .sid is downloaded if absent and
the dump rendered from it) — there is no skip path.
"""

import os
import subprocess
import urllib.request

from preframr_tokens import CPF, measure, recover_program, verify_residual

_LOCAL_SID = (
    "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid"
)
_LOCAL_DUMP = "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.1.dump.parquet"
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_fixtures")
_CACHE_SID = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.sid")
_CACHE_DUMP = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.1.dump.parquet")
_SID_URL = os.environ.get(
    "MONTY_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
)


def _resolve_sid():
    for p in (_LOCAL_SID, _CACHE_SID):
        if os.path.exists(p):
            return p
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    urllib.request.urlretrieve(_SID_URL, _CACHE_SID)
    return _CACHE_SID


def _render_dump(sid_path, out_path):
    rc = subprocess.call(
        ["sidtrace", "--sid", sid_path, "--subtune", "1", "--out", out_path]
    )
    assert rc == 0 and os.path.exists(out_path), (
        "sidtrace render failed; the Monty dump fixture could not be built. "
        "Build/install sidtrace (build_sidtrace.sh) — this gate has no skip path."
    )


def _resolve_dump(sid_path):
    for p in (_LOCAL_DUMP, _CACHE_DUMP):
        if os.path.exists(p):
            return p
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    _render_dump(sid_path, _CACHE_DUMP)
    return _CACHE_DUMP


def test_monty_under_one_token_per_frame():
    sid = _resolve_sid()
    dump = _resolve_dump(sid)
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
