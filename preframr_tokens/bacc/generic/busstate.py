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
    if nframes <= 0:
        return np.zeros((0, NREG), dtype=np.int64), t0, cpf
    # Vectorised per-frame state: scatter each write into its (group, reg) cell
    # keeping the LAST write per cell (the frame's register value), then forward
    # -fill the running register file across groups.  Equivalent byte-for-byte to
    # the running-file loop but O(nwrites + ngroups*NREG) in numpy, so multispeed
    # traces (tens of millions of writes) reconstruct in milliseconds, not the
    # minutes a Python per-write loop costs.
    group = np.repeat(np.arange(ngroups), np.diff(np.append(starts, len(cyc))))
    valid = reg < NREG
    grid = np.zeros((ngroups, NREG), dtype=np.int64)
    seen = np.zeros((ngroups, NREG), dtype=bool)
    g_v, r_v, v_v = group[valid], reg[valid], val[valid]
    grid[g_v, r_v] = v_v  # later writes overwrite earlier -> last write per cell
    seen[g_v, r_v] = True
    # forward-fill held values down the group axis where a group wrote nothing.
    idx = np.where(seen, np.arange(ngroups)[:, None], 0)
    idx = np.maximum.accumulate(idx, axis=0)
    full = np.take_along_axis(grid, idx, axis=0)
    seq = full[boot_off:].copy()
    return seq, t0, cpf


def dump_first_play_cycle(dump_path):
    """The dump's ``first_play_cycle`` -- the generic frame-0 anchor the bus
    state must align to.  Reads only the dump's chip-0 write clocks."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    frame = pd.read_parquet(dump_path, columns=["clock", "chipno"])
    cyc = frame[frame.chipno == 0]["clock"].to_numpy(np.int64)
    return first_play_cycle(cyc, detect_play_period(cyc))
