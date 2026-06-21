"""Phase 2a parity proof: per_frame_state(sidtrace) == per_frame_state(headlessvice).

Proves the new preframr-sidtrace dump backend reproduces the VICE/headlessvice
register dump byte-exact on regs 0-24, per the pinned dump contract, for every
gate fixture. This is the step that begins retiring headlessvice.

Both dumps are framed by the production ``per_frame_state``. The two emulators'
init/boot prologs differ -- VICE logs an init write at the first cycles, while
libsidplayfp's first writes can appear dozens of frames in -- so the two
per-frame grids sit a constant integer number of frames apart. ``frame_parity``
recovers that boot lag (the SAME alignment the corpus fidelity census uses) and
we require **100.0%** byte-exact over the overlap. The lag is an init-timing
offset, never an emulation discrepancy (HARD RULE #0).

This test is SKIPPED unless the sidtrace binary is available (``SIDTRACE_BIN`` /
``PREFRAMR_SIDTRACE_ROOT`` / a sibling ``preframr-sidtrace/build/sidtrace``) AND
the headlessvice VICE dump for each fixture can be acquired, so default CI -- which
has neither -- is unaffected. Build sidtrace and run locally to exercise it:

    PREFRAMR_SIDTRACE_ROOT=/path/to/preframr-sidtrace \\
        pytest tests/test_sidtrace_parity.py -v
"""

import os
import tempfile

import pytest

from preframr_tokens import per_frame_state
from preframr_tokens.codec import sidtrace_dump as S
from tests import _dump_fixture as F

MAXFRAMES = 600
_HVSC_BASE = "https://hvsc.brona.dk/HVSC/C64Music"

# Real C64 ROMs for RSID/KERNAL-calling tunes, if present (PSID tunes need none;
# every gate fixture is byte-exact with or without them on the fixtures tested).
_ROM_DIR = os.environ.get("C64_ROM_DIR", "/scratch/anarkiwi/cbm/asid-vice/data/C64")
_ROMS = tuple(
    os.path.join(_ROM_DIR, n)
    for n in (
        "kernal-901227-03.bin",
        "basic-901226-01.bin",
        "chargen-901225-01.bin",
    )
)
_ROM_ARGS = _ROMS if all(os.path.exists(r) for r in _ROMS) else ()


def _sidtrace_available():
    return S.find_binary() is not None


pytestmark = pytest.mark.skipif(
    not _sidtrace_available(),
    reason="sidtrace binary unavailable (set SIDTRACE_BIN/PREFRAMR_SIDTRACE_ROOT)",
)


@pytest.mark.parametrize("hvsc_rel,subtune", F.GATE_FIXTURES)
def test_sidtrace_matches_headlessvice(hvsc_rel, subtune):
    """sidtrace dump == headlessvice dump byte-exact (regs 0-24) for one fixture."""
    # Headlessvice/VICE dump via the existing (unchanged) backend.
    try:
        sid, vice_dump = F.acquire(hvsc_rel, f"{_HVSC_BASE}/{hvsc_rel}", subtune)
    except Exception as exc:  # noqa: BLE001 -- no HVSC / no docker -> skip, not fail
        pytest.skip(f"headlessvice dump unavailable: {exc}")

    vstate = per_frame_state(vice_dump, maxframes=MAXFRAMES)
    assert vstate is not None, "empty headlessvice per_frame_state"

    # sidtrace dump via the new backend.
    with tempfile.TemporaryDirectory() as work:
        st_dump = os.path.join(work, "st.dump.parquet")
        S.render_dump(sid, subtune, MAXFRAMES, st_dump, roms=_ROM_ARGS)
        sstate = per_frame_state(st_dump, maxframes=MAXFRAMES)
    assert sstate is not None, "empty sidtrace per_frame_state"

    ok, n, lag = S.frame_parity(sstate, vstate)
    assert n > 20, f"too little frame overlap (n={n}, lag={lag})"
    assert ok == n, (
        f"{os.path.basename(hvsc_rel)}: {ok}/{n} frames byte-exact at boot lag "
        f"{lag}; first divergence is a real (frame,register) mismatch, not boot "
        f"alignment -- trace it (HARD RULE #0)."
    )
