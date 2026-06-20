"""PERMANENT GATE — Monty-on-the-Run encodes at < 1 token / frame, losslessly.

Parses the FULL Rob Hubbard Monty_on_the_Run (sub 1) dump and FAILS unless the
codec encodes it RESIDUAL-ZERO (byte-exact) at < 1 token / frame. This is the
executable form of the AGENTS.md headline goal (< 1 token/frame). The lossless
check is part of the gate: a lossy codec trivially hits any token budget, so the
budget only means something when residual = 0.

This test MUST pass, MUST run in CI, and may NEVER be removed, skipped, xfailed,
or bypassed. The fixture is auto-acquired (rendered from the .sid if the dump is
absent) — there is no skip path.
"""

import os
import subprocess
import urllib.request

from preframr_tokens import CPF, measure, per_frame_state, verify_residual

_LOCAL_DUMP = "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.1.dump.parquet"
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_fixtures")
_CACHE_DUMP = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.1.dump.parquet")
_CACHE_SID = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.sid")
_SID_URL = os.environ.get(
    "MONTY_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
)


def _render_dump(sid_path, out_path):
    """Render a .sid to a register dump via the offline sidtrace tooling. Loud
    failure (never a skip) if the renderer is unavailable."""
    rc = subprocess.call(
        ["sidtrace", "--sid", sid_path, "--subtune", "1", "--out", out_path]
    )
    assert rc == 0 and os.path.exists(out_path), (
        "sidtrace render failed; the Monty fixture could not be built. Build/install "
        "sidtrace (build_sidtrace.sh) — this gate has no skip path."
    )


def _resolve_dump():
    """Local dump -> cached dump -> download .sid + render. Never skips."""
    for p in (_LOCAL_DUMP, _CACHE_DUMP):
        if os.path.exists(p):
            return p
    os.makedirs(_FIXTURE_DIR, exist_ok=True)
    if not os.path.exists(_CACHE_SID):
        urllib.request.urlretrieve(_SID_URL, _CACHE_SID)
    _render_dump(_CACHE_SID, _CACHE_DUMP)
    return _CACHE_DUMP


def test_monty_under_one_token_per_frame():
    dump = _resolve_dump()
    s = per_frame_state(dump, CPF, 1_000_000)  # FULL tune, no frame cap
    assert s is not None and len(s) >= 2, "Monty dump did not parse to frames"
    assert verify_residual(s), "codec is NOT residual-zero (lossless) on Monty"
    brk, frames = measure(s)
    tokens = brk["total"]
    assert tokens / frames < 1.0, (
        f"Monty: {tokens} tokens / {frames} frames = {tokens / frames:.3f} tok/frame "
        f">= 1.0 (must be < 1 token/frame, residual-zero)"
    )


if __name__ == "__main__":
    test_monty_under_one_token_per_frame()
    print("PASS")
