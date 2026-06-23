"""Single-file ``.sid`` -> generic BACC recovery (no ``.dump.parquet`` input).

``recover_from_sid`` collapses the two-file (``.sid`` + pre-rendered dump) path to
ONE input: a single deterministic ``preframr-sidtrace`` run over the ``.sid``
generates BOTH the per-frame register dump (``.sidwr.bin``) and the bus trace
(``.bus.bin``) in-process, the GENERIC recovery fits the program from the bus, and
the render is verified residual-zero against the SAME-run dump.

The whole-tune residual-zero proof needs the ``preframr-sidtrace`` binary -- a
real external tool -- so the proof tests are env-gated (skip-if-binary-absent via
``SIDTRACE_BIN``).  This keeps the DEFAULT CI gate self-contained: it runs the
committed-dump oracle in a container with NO sidtrace binary and never reaches the
binary-dependent tests.  The ``.sidwr.bin`` parser + helper plumbing are covered
self-contained below (synthetic packed records, no binary).
"""

import numpy as np
import pytest

from preframr_tokens.bacc.generic import recover_from_sid
from preframr_tokens.bacc.generic.busstate import NREG, per_frame_state_from_bus
from preframr_tokens.bacc.generic.bustrace import BUS_DT
from preframr_tokens.bacc.generic.sidtrace import (
    SIDWR_DT,
    _frame_starts,
    sid_to_dump_and_bustrace,
    sidtrace_bin,
    sidwr_state,
)
from preframr_tokens.bacc.primitive import BaccProgram

_CPF = 19656  # PAL raster frame; the synthetic traces use a steady cadence
_FIXTURES = "tests/test_fixtures"
_SID_ONLY_TUNES = ["Grid_Runner", "Need_More_NOPs"]


# --------------------------------------------------------------------------- #
# Self-contained: the .sidwr.bin parser + helpers (no binary, synthetic records)
# --------------------------------------------------------------------------- #
def _synth_sidwr(nframes=160):
    """A small packed ``.sidwr.bin`` (the SID-write substream of the same run a
    ``.bus.bin`` carries), re-blitting all 25 registers at a steady cadence.

    Returns ``(records, bus_records)`` -- the ``SIDWR_DT`` writes and the matching
    ``BUS_DT`` bus trace -- so the parser's output can be checked against
    ``per_frame_state_from_bus`` (they must be byte-identical: same run, same
    blit-group framing)."""
    arp = [0x0800, 0x0900, 0x0A00, 0x0900]
    sid_recs, bus_recs = [], []
    cyc = 1000
    reg = [0] * 25
    for frame in range(nframes):
        v0f = arp[frame % 4]
        reg[0], reg[1] = v0f & 0xFF, (v0f >> 8) & 0xFF
        pw0 = 0x0400 + 4 * (frame // 2)
        reg[2], reg[3] = pw0 & 0xFF, (pw0 >> 8) & 0x0F
        reg[4] = 0x41 if frame >= 2 else 0x40
        reg[24] = 0x0F
        for index in range(25):
            addr = 0xD400 + index
            sid_recs.append((cyc, addr, index, reg[index]))
            bus_recs.append((cyc, addr, reg[index], 1))
            cyc += 2
        cyc += _CPF - 2 * 25
    return (
        np.array(sid_recs, dtype=SIDWR_DT),
        np.array(bus_recs, dtype=BUS_DT),
    )


def test_sidwr_state_matches_bus_state(tmp_path):
    # The in-process dump (.sidwr.bin) and the bus-state come from the SAME run
    # and are framed identically -> byte-identical by construction (no shift).
    sid_recs, bus_recs = _synth_sidwr()
    path = tmp_path / "trace.sidwr.bin"
    sid_recs.tofile(str(path))
    dump_state, t0 = sidwr_state(str(path))
    bus_state, _, _ = per_frame_state_from_bus(bus_recs, t0=t0)
    assert dump_state.shape == bus_state.shape
    assert np.array_equal(dump_state, bus_state)


def test_sidwr_state_masks_pw_high(tmp_path):
    # PW-high bits 4-7 are don't-care (12-bit pulse width) -> masked to 4 bits.
    recs = np.array(
        [(1000 + 2 * i, 0xD400 + r, r, 0xFF) for i, r in enumerate((3, 10, 17))],
        dtype=SIDWR_DT,
    )
    path = tmp_path / "pw.sidwr.bin"
    recs.tofile(str(path))
    state, _ = sidwr_state(str(path))
    for pw_reg in (3, 10, 17):
        assert np.all(state[:, pw_reg] == 0x0F)


def test_sidwr_state_empty_trace(tmp_path):
    path = tmp_path / "empty.sidwr.bin"
    np.array([(1000, 0x0000, 0x20, 0x10)], dtype=SIDWR_DT).tofile(str(path))  # reg>=25
    state, t0 = sidwr_state(str(path))
    assert state is None and t0 is None


def test_frame_starts_groups_by_gap():
    cyc = np.array([0, 2, 4, 10000, 10002, 20000], dtype=np.int64)
    assert _frame_starts(cyc).tolist() == [0, 3, 5]
    assert _frame_starts(np.empty(0, dtype=np.int64)).tolist() == []


def test_sidtrace_bin_env(monkeypatch, tmp_path):
    # absent path -> None (the sid-only path is then skipped; CI stays render-free)
    monkeypatch.setenv("SIDTRACE_BIN", str(tmp_path / "nope"))
    assert sidtrace_bin() is None
    # present path -> returned verbatim
    fake = tmp_path / "sidtrace"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SIDTRACE_BIN", str(fake))
    assert sidtrace_bin() == str(fake)


def test_run_sidtrace_missing_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SIDTRACE_BIN", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="preframr-sidtrace binary not found"):
        sid_to_dump_and_bustrace("x.sid")


# --------------------------------------------------------------------------- #
# Env-gated: whole-tune residual-zero from the .sid ALONE (needs the binary).
# Skipped when no sidtrace binary is present, so the default CI gate -- which
# runs the committed-dump oracle with NO binary -- stays self-contained.
# --------------------------------------------------------------------------- #
_HAVE_BIN = sidtrace_bin() is not None


@pytest.mark.skipif(
    not _HAVE_BIN,
    reason="set SIDTRACE_BIN to the built preframr-sidtrace 'sidtrace' binary",
)
@pytest.mark.parametrize("tune", _SID_ONLY_TUNES)
def test_recover_from_sid_residual_zero(tune):
    """SINGLE FILE: recover a generic BACC program from JUST the ``.sid`` (NO
    ``.dump.parquet`` argument) and verify the render is residual-zero against the
    in-process-generated dump across all 25 registers."""
    program, resid, dump_state = recover_from_sid(f"{_FIXTURES}/{tune}.sid")
    assert isinstance(program, BaccProgram)
    assert program.driver == "generic"
    assert program.nframes > 100
    assert dump_state.shape[1] == NREG
    total = sum(resid.values())
    assert total == 0, f"{tune}: non-zero residual cells {resid}"
