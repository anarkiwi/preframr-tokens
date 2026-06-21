"""Per-frame 25-register state from the trusted VICE bus trace -- blit-aware.

The bus trace's SID-write substream (addr ``$D400..$D418``, rw=1) comes from the
SAME vsid run as the register dump.  Many drivers (GoatTracker, and Hubbard's
shadow file) re-blit the WHOLE 25-register shadow file to the SID every
play-call, so the bus carries ~25 writes per frame where the coalesced dump
records only the changed registers.  Per play-call (= per frame) the LAST value
written to each register is that register's frame value -- identical to the
dump's per-frame state once we

  (a) frame on the steady IRQ/play cadence (blit-group boundaries), and
  (b) mask the PW-high registers (3, 10, 17) to 4 bits -- a pure SID-chip
      semantic (pulse width is 12-bit; bits 4-7 of PW-high are don't-care), the
      SAME mask ``codec.lsp_validate`` applies; NOT driver-specific, and
  (c) align frame 0 to the dump's first steady play-call (boot-prolog offset).

The boot offset is derived from the bus's OWN cadence: the blit group whose
start cycle == ``first_play_cycle`` (the steady-cadence anchor the dump uses) is
dump-frame 0.  No driver layout knowledge anywhere.
"""

import numpy as np

from preframr_tokens.codec.lsp_validate import detect_play_period, first_play_cycle

NREG = 25
PW_HI = (3, 10, 17)
_BLIT_GAP = 2000  # cycles; an inter-write gap above this starts a new play-call


def sid_writes(records):
    """SID-write substream of a bus trace: ``(cyc, reg, val)`` with reg in
    0..24 and the PW-high registers masked to their 4 SID-significant bits."""
    sel = (
        (records["addr"] >= 0xD400) & (records["addr"] <= 0xD418) & (records["rw"] == 1)
    )
    writes = records[sel]
    reg = (writes["addr"] - 0xD400).astype(int)
    val = writes["val"].astype(int).copy()
    for pw_reg in PW_HI:
        val[reg == pw_reg] &= 0x0F
    return writes["cyc"].astype(np.int64), reg, val


def _frame_starts(cyc, gap=_BLIT_GAP):
    """Blit-group / play-call boundaries: an inter-write gap above ``gap``
    cycles starts a new play-call burst.  Returns the index of the first write
    of each burst."""
    if len(cyc) == 0:
        return np.empty(0, dtype=np.int64)
    big = np.nonzero(np.diff(cyc) > gap)[0]
    return np.concatenate(([0], big + 1))


def _running_state(bin_idx, reg, val, nbins):
    """Per-bin running 25-register file from writes labelled with their bin index.

    Scatters each write into its ``(bin, reg)`` cell keeping the LAST write per
    cell (the bin's register value), then forward-fills the running register file
    down the bin axis where a bin wrote nothing.  Vectorised: O(nwrites +
    nbins*NREG), so a multispeed trace (tens of millions of writes) reconstructs
    in milliseconds rather than the minutes a Python per-write loop costs.  Shared
    by the gap-grouped and cadence-binned framings so both produce byte-identical
    running state from the same labelling.
    """
    grid = np.zeros((nbins, NREG), dtype=np.int64)
    seen = np.zeros((nbins, NREG), dtype=bool)
    valid = (reg < NREG) & (bin_idx >= 0) & (bin_idx < nbins)
    b_v, r_v, v_v = bin_idx[valid], reg[valid], val[valid]
    grid[b_v, r_v] = v_v  # later writes overwrite earlier -> last write per cell
    seen[b_v, r_v] = True
    idx = np.where(seen, np.arange(nbins)[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    return np.take_along_axis(grid, idx, axis=0)


def _static_hold_state(reg, val, t0, cpf, trace_end):
    """Whole-tune CONSTANT-HOLD state for a tune that pokes the SID once at boot
    and then never writes it again -- the register file rings on, unchanged, for
    the rest of the run.

    Some tunes (BASIC stubs, init-only / silent subtunes, single-chord intros) emit
    a single boot burst of SID writes and then leave the chip alone; the bus carries
    that one burst near ``t0`` and millions of quiet cycles after it.  Gap grouping
    sees one blit group (one frame) and the cadence fallback sees no multi-frame
    SID-write span, so both degenerate to a single frame and the recovery rejects
    the tune as unparseable -- even though the chip state is perfectly well defined:
    it is the boot burst's final per-register value, HELD for the whole duration.

    That held state is the canonical closed-form generator -- every register is a
    constant ``hold`` -- so it is recovered, not faked: the trace's OWN extent
    (``trace_end``, the last bus access cycle) relative to frame-0 anchor ``t0`` at
    the play cadence ``cpf`` gives the frame count over which the value persists, and
    the boot file (last-write-wins per register) is broadcast across every frame.
    Pure cadence arithmetic; no driver layout knowledge.
    """
    nframes = int(round((trace_end - t0) / cpf)) + 1 if cpf else 0
    if nframes < 2:
        return np.zeros((0, NREG), dtype=np.int64)
    # The held file is last-write-wins per register over the whole (boot-only)
    # write stream -- the value left ringing on the chip after the final write.
    boot = np.zeros((1, NREG), dtype=np.int64)
    valid = reg < NREG
    boot[0, reg[valid]] = val[valid]  # stable order -> later write wins per reg
    return np.repeat(boot, nframes, axis=0)


def _cadence_state(cyc, reg, val, t0, cpf):
    """Per-frame state binned on the IRQ/play CADENCE rather than on write gaps.

    Frame ``f``'s state is the running register file after every write whose cycle
    rounds to frame ``f`` (``round((cyc - t0) / cpf)``), exactly the binning the
    register-dump side uses (:func:`codec.lsp_validate.state_seq`).  This is the
    fallback for a player that writes the SID CONTINUOUSLY across the whole frame
    (no quiet inter-play gap), where the gap-based blit grouping collapses every
    play-call into one group and the gap framing degenerates to a single frame.
    Pure cadence arithmetic -- no driver layout knowledge.
    """
    bin_idx = np.round((cyc - t0) / cpf).astype(np.int64)
    nbins = int(bin_idx.max()) + 1 if len(bin_idx) else 0
    if nbins <= 0:
        return np.zeros((0, NREG), dtype=np.int64)
    return _running_state(bin_idx, reg, val, nbins)


def per_frame_state_from_bus(records, cpf=None, t0=None):
    """Reconstruct the per-frame ``(nframes, 25)`` state aligned to the dump's
    frame grid.

    Returns ``(state, t0_frame_cycle, cpf)``.  Frame ``f``'s state is the
    running register file after applying every write in blit-group
    ``boot_off + f``.

    The boot-prolog offset is the blit group whose start cycle == the tune's
    first steady play-call.  When the dump's ``t0`` is supplied we anchor to it
    (the dump defines the frame grid we must reproduce byte-exact); otherwise we
    fall back to the bus's own ``first_play_cycle``.
    """
    cyc, reg, val = sid_writes(records)
    if len(cyc) == 0:
        return None, None, None
    if cpf is None:
        cpf = detect_play_period(cyc)
    starts = _frame_starts(cyc)
    gstart_cyc = cyc[starts]
    if t0 is None:
        t0 = first_play_cycle(cyc, cpf)
    boot_off = int(np.searchsorted(gstart_cyc, t0 - cpf / 2))
    ngroups = len(starts)
    nframes = ngroups - boot_off
    # A player that writes the SID continuously across the whole frame leaves no
    # quiet inter-play gap, so the gap-based grouping collapses every play-call
    # into ONE blit group and the gap framing degenerates to a single frame even
    # though the trace spans hundreds of cadence frames.  Detect that degeneracy
    # -- gap grouping yields essentially no frames while the trace clearly spans
    # many -- and fall back to cadence binning, which is robust to dense writes.
    # The proven gap framing (byte-exact on the fixtures) is unchanged for any
    # tune that does expose per-play gaps; only the degenerate single-group case
    # switches to the cadence path.
    span_frames = (int(cyc[-1]) - t0) / cpf if cpf else 0
    if nframes < 2 and span_frames >= 2:
        # ``t0`` is frame 0, so ``round((cyc - t0)/cpf)`` puts the first play-call
        # at bin 0 and any boot-prolog write (cycle < t0) at a negative bin that
        # the running-state binner drops -- no separate boot offset needed.
        return _cadence_state(cyc, reg, val, t0, cpf), t0, cpf
    # Static-hold tune: a single boot burst of SID writes (no per-frame cadence, so
    # gap AND cadence framing both degenerate to one frame) followed by a quiet
    # chip that rings on the boot value for the rest of the run.  The SID-write span
    # is ~0 frames but the OVERALL bus trace (``records["cyc"][-1]``) spans many
    # cadence frames, so the held state is well defined: recover it as a whole-tune
    # constant rather than rejecting the tune as unparseable.
    trace_end = int(records["cyc"][-1])
    if nframes < 2 and cpf and (trace_end - t0) / cpf >= 2:
        return _static_hold_state(reg, val, t0, cpf, trace_end), t0, cpf
    if nframes <= 0:
        return np.zeros((0, NREG), dtype=np.int64), t0, cpf
    # Vectorised per-frame state: scatter each write into its (group, reg) cell
    # keeping the LAST write per cell, then forward-fill the running register file
    # across groups (see :func:`_running_state`).  Equivalent byte-for-byte to the
    # running-file loop but O(nwrites + ngroups*NREG), so multispeed traces (tens
    # of millions of writes) reconstruct in milliseconds.
    group = np.repeat(np.arange(ngroups), np.diff(np.append(starts, len(cyc))))
    full = _running_state(group, reg, val, ngroups)
    seq = full[boot_off:].copy()
    return seq, t0, cpf


def dump_first_play_cycle(dump_path):
    """The dump's ``first_play_cycle`` -- the generic frame-0 anchor the bus
    state must align to.  Reads only the dump's chip-0 write clocks."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    frame = pd.read_parquet(dump_path, columns=["clock", "chipno"])
    cyc = frame[frame.chipno == 0]["clock"].to_numpy(np.int64)
    return first_play_cycle(cyc, detect_play_period(cyc))
