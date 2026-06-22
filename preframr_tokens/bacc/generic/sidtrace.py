"""Single-file ``.sid`` -> (per-frame register dump, bus trace) via preframr-sidtrace.

One run of the deterministic ``preframr-sidtrace`` binary (PR #5) over a ``.sid``
emits BOTH artifacts the generic recovery needs, internally self-consistent
(same emulator, same run):

  * ``<prefix>.sidwr.bin`` -- every SID register write as packed records
    ``int64 cyc, uint16 addr, uint8 reg, uint8 val`` -- the per-frame register
    DUMP (the same role the corpus ``.dump.parquet`` plays), and
  * ``<prefix>.bus.bin`` -- the full CPU bus trace (:data:`bustrace.BUS_DT`) --
    the provenance substrate the GENERIC recovery reads.

:func:`sid_to_dump_and_bustrace` runs the binary once (subprocess, exactly as the
hand backends shell out to ``headlessvice``) and returns the dump as an
``(nframes, 25)`` register array -- the SAME schema and SAME blit-group framing
:func:`busstate.per_frame_state_from_bus` produces, so the in-process dump and
the bus-state are byte-identical by construction (no pre-rendered dump file, no
fragile frame-shift).  The binary location is taken from ``SIDTRACE_BIN`` (the
built path as default); a clear error is raised when it is absent so the default,
self-contained CI -- which never invokes the binary -- is unaffected (HARD RULE
#0: the dump is the real emulator output, never fabricated).
"""

import os
import subprocess
import tempfile

import numpy as np

from preframr_tokens.bacc.generic.busstate import NREG, PW_HI
from preframr_tokens.codec.lsp_validate import (
    BURST_GAP,
    detect_play_period,
    first_play_cycle,
)

# Packed ``.sidwr.bin`` record: int64 cyc, uint16 addr, uint8 reg, uint8 val.
SIDWR_DT = np.dtype([("cyc", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")])

# The built binary in the sibling preframr-sidtrace checkout (REUSE; never rebuilt).
_DEFAULT_SIDTRACE_BIN = "/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace"


def sidtrace_bin():
    """The ``preframr-sidtrace`` binary path (``SIDTRACE_BIN`` env or the built
    default).  Returns ``None`` when no binary is present -- the caller skips the
    sid-only path so the default render-free CI gate stays self-contained."""
    cand = os.environ.get("SIDTRACE_BIN", _DEFAULT_SIDTRACE_BIN)
    return cand if cand and os.path.exists(cand) else None


def sidwr_state(sidwr_path, t0=None):
    """Parse a ``.sidwr.bin`` into the per-frame ``(nframes, 25)`` register array.

    Vectorised load of the packed records, then the SAME blit-group framing as
    :func:`busstate.per_frame_state_from_bus`: a SID-write gap above
    :data:`BURST_GAP` cycles starts a new play-call, the last value written to
    each register in a play-call is that frame's value, the PW-high registers are
    masked to their 4 SID-significant bits (a pure chip semantic), and frame 0 is
    the play-call at the tune's first steady play cycle (``t0``).

    Returns ``(state, t0)``; ``state`` is ``None`` for a trace with no SID writes.
    """
    recs = np.fromfile(sidwr_path, dtype=SIDWR_DT)
    recs = recs[recs["reg"] < NREG]
    if len(recs) == 0:
        return None, None
    cyc = recs["cyc"].astype(np.int64)
    reg = recs["reg"].astype(int)
    val = recs["val"].astype(int).copy()
    for pw_reg in PW_HI:
        val[reg == pw_reg] &= 0x0F
    cpf = detect_play_period(cyc)
    starts = _frame_starts(cyc)
    ends = np.concatenate((starts[1:], [len(cyc)]))
    gstart_cyc = cyc[starts]
    if t0 is None:
        t0 = first_play_cycle(cyc, cpf)
    boot_off = int(np.searchsorted(gstart_cyc, t0 - cpf / 2))
    nframes = len(starts) - boot_off
    seq = np.zeros((nframes, NREG), dtype=np.int64)
    cur = [0] * NREG
    for group in range(len(starts)):
        for k in range(starts[group], ends[group]):
            cur[reg[k]] = val[k]
        if group >= boot_off:
            seq[group - boot_off] = cur
    return seq, t0


def _frame_starts(cyc, gap=BURST_GAP):
    """Indices of the first write of each play-call burst (blit-group boundary)."""
    if len(cyc) == 0:
        return np.empty(0, dtype=np.int64)
    big = np.nonzero(np.diff(cyc) > gap)[0]
    return np.concatenate(([0], big + 1))


def run_sidtrace(sid_path, out_prefix, subtune=1, nframes=200, sidtrace_path=None):
    """Run ``preframr-sidtrace`` once over ``sid_path`` (subtune is 1-based),
    emitting ``<out_prefix>.sidwr.bin`` (the small timestamped SID-write stream,
    the render gate) and ``<out_prefix>.distill.bin`` (the compact SDST artifact
    the SMC-correct recovery consumes -- a few KB, NOT the retired multi-GB raw
    bus trace).

    Returns ``(sidwr_path, distill_path)``.  Raises :class:`FileNotFoundError`
    when no binary is available (env-gated; default CI never reaches here)."""
    binary = sidtrace_path or sidtrace_bin()
    if binary is None:
        raise FileNotFoundError(
            "preframr-sidtrace binary not found; set SIDTRACE_BIN to the built "
            f"'sidtrace' (looked for {_DEFAULT_SIDTRACE_BIN})"
        )
    subprocess.run(
        [binary, str(sid_path), str(int(subtune)), str(int(nframes)), str(out_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return f"{out_prefix}.sidwr.bin", f"{out_prefix}.distill.bin"


def sidwr_to_bus(sidwr_path):
    """Synthesise a SID-write-only :data:`bustrace.BUS_DT` record array from a
    ``.sidwr.bin``.

    The full cycle-by-cycle bus trace is no longer emitted (it was GBs/tune); the
    legacy per-register generic recovery (:func:`recover_generic`) needs only the
    SID-write substream to reconstruct the per-frame register state, which the
    small ``.sidwr.bin`` carries verbatim.  Reads are absent, so the optional
    note-table-by-read enhancer simply does not fire -- the per-register fit path
    still covers the output.  The SMC-correct song-data recovery uses the
    ``.distill.bin`` artifact, not this synthesised stream.
    """
    from preframr_tokens.bacc.generic.bustrace import BUS_DT  # local: avoid cycle

    recs = np.fromfile(sidwr_path, dtype=SIDWR_DT)
    bus = np.zeros(len(recs), dtype=BUS_DT)
    bus["cyc"] = recs["cyc"]
    bus["addr"] = recs["addr"]
    bus["val"] = recs["val"]
    bus["rw"] = 1  # SID writes
    return bus


def sid_to_dump_and_bustrace(
    sid_path, subtune=1, nframes=200, sidtrace_path=None, out_prefix=None
):
    """From a ``.sid`` ALONE, produce the per-frame register dump, a SID-write
    bus-state array, and the compact distill artifact via ONE
    ``preframr-sidtrace`` run -- no pre-rendered ``.dump.parquet`` input and no
    multi-GB raw trace.

    Returns ``(dump_state, bus, t0, distill_path)`` where ``dump_state`` is the
    ``(nframes, 25)`` register array (the dump), ``bus`` is the SID-write-only
    :data:`bustrace.BUS_DT` array (synthesised from ``.sidwr.bin`` for the legacy
    per-register recovery), ``t0`` is the dump's frame-0 anchor, and
    ``distill_path`` is the SDST artifact for the SMC-correct identity recovery.

    When ``out_prefix`` is given the artifacts persist there; otherwise they land
    in a temporary directory the caller owns.
    """
    if out_prefix is None:
        out_prefix = os.path.join(
            tempfile.mkdtemp(prefix="preframr_sidtrace_"), "trace"
        )
    sidwr_path, distill_path = run_sidtrace(
        sid_path, out_prefix, subtune, nframes, sidtrace_path
    )
    dump_state, t0 = sidwr_state(sidwr_path)
    bus = sidwr_to_bus(sidwr_path)
    return dump_state, bus, t0, distill_path
