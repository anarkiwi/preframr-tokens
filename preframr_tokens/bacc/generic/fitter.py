"""Generic full 25-register whole-tune fitter from the per-frame bus state.

Given the byte-exact per-frame state (:func:`busstate.per_frame_state_from_bus`)
this fits a GENERIC per-register program with NO raw-byte storage of the output
and NO per-driver constants:

  - generator lanes (freq lo/hi, pw lo/hi): the BACC archetype library
    (:mod:`archetypes`), sliced at gate-rise note-ons, plus a pre-roll segment
    covering ``[0, first note-on)`` (the player sets an initial note/sweep before
    the first gate-rise) and the generic maskaccum / ratewalk / tablewalk /
    tablewalk_lead periodic generators and the advance-clocked wavetable_ptr
    (a per-voice groove-paced pointer walk over a value table).
  - non-generator lanes (ctrl / AD / SR / filter cutoff+res / volume): a compact
    piecewise program whose switch points are the note/pattern boundaries the bus
    exposes -- this is also the orderlist / song-structure reconstruction.

The recovered program renders back to ``(nframes, 25)`` and is required ==
state byte-exact (residual=0) per register, so any gap is precisely attributed.
"""

from collections import defaultdict

import numpy as np

from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic.busstate import NREG

FREQ_LO = (0, 7, 14)
FREQ_HI = (1, 8, 15)
PW_LO = (2, 9, 16)
PW_HI = (3, 10, 17)
CTRL = (4, 11, 18)
AD = (5, 12, 19)
SR = (6, 13, 20)
EVENT_REGS = set(CTRL + AD + SR) | {21, 22, 23, 24}  # + filter cutoff/res + volume


def _noteon_points(noteons, voice):
    """Note-on frames for a voice with the pre-roll boundary at frame 0."""
    ons = noteons[voice]
    if not ons or ons[0] != 0:
        ons = [0] + list(ons)
    return ons


def fit_generator_lanes(state, note_table):
    """Fit the freq + pw (16-bit) lanes via the BACC archetype library.  Returns
    ``(fits, noteons)`` where fits maps ``(voice, 'freq'|'pw')`` to
    ``(segments, carry)``.

    The generator lanes are sliced at :func:`archetypes.note_boundaries` -- every
    bus-visible note-on retrigger (gate rise, control-byte change, or ADSR change),
    not just a gate rise -- so a legato / hard-restart phrase that keeps the gate
    high across notes is segmented per note instead of collapsing into one
    over-long, unfittable segment.  The FREQ lane additionally slices at
    :func:`archetypes.pw_sweep_resets` (a per-note pulse-sweep re-seed) so a
    pure-legato melody with NO control/ADSR retrigger is still segmented per note;
    the PW lane is NOT sliced there, so an irreducible reflecting-triangle PW would
    stay one segment and surface rather than fragment into raw-byte pieces."""
    noteons = A.note_boundaries(state)
    nframes = len(state)
    out = {}
    for voice in range(3):
        ons = _noteon_points(noteons, voice)
        freq_ons = sorted(set(ons) | set(A.pw_sweep_resets(state, voice)))
        flane = A.lane_freq(state, voice)
        fres = A.fit_lane(flane, freq_ons, nframes, note_table, None, 0xFFFF)
        carry = A.freq_carry_sequence(fres, nframes)
        plane = A.lane_pw(state, voice)
        pres = A.fit_lane(plane, ons, nframes, note_table, carry, 0xFFFF)
        out[(voice, "freq")] = (fres, None)
        out[(voice, "pw")] = (pres, carry)
    return out, noteons


def render_generator_lane(res, nframes, note_table, carry):
    """Rebuild a 16-bit generator lane from its fitted ``(start, stop, fit)``
    segments.  Returns ``(lane, bad)`` where ``bad`` counts frames left uncovered
    by an un-fit (None) segment."""
    lane = np.zeros(nframes, dtype=np.int64)
    bad = 0
    for start, stop, fit in res:
        if fit is None:
            bad += stop - start
            continue
        cseg = carry[start:stop] if carry is not None else None
        lane[start:stop] = A.render_fit(fit, stop - start, note_table, cseg)
    return lane, bad


def fit_full_tune(state, note_table):
    """Fit every register of the per-frame state.  Returns
    ``(rendered, resid, nseg, archtally, genfits, eventfits)`` where ``resid``
    maps register -> residual frame count and ``rendered`` is the re-rendered
    ``(nframes, 25)`` state."""
    nframes = len(state)
    genfits, _ = fit_generator_lanes(state, note_table)
    rendered = np.zeros_like(state)
    resid = {}
    nseg = defaultdict(int)
    archtally = defaultdict(int)

    for voice in range(3):
        flo, fhi = FREQ_LO[voice], FREQ_HI[voice]
        plo, phi = PW_LO[voice], PW_HI[voice]
        fres, _ = genfits[(voice, "freq")]
        pres, carry = genfits[(voice, "pw")]
        flane, _ = render_generator_lane(fres, nframes, note_table, None)
        plane, _ = render_generator_lane(pres, nframes, note_table, carry)
        rendered[:, flo] = flane & 0xFF
        rendered[:, fhi] = (flane >> 8) & 0xFF
        rendered[:, plo] = plane & 0xFF
        rendered[:, phi] = (plane >> 8) & 0x0F
        nseg["freq"] += len(fres)
        nseg["pw"] += len(pres)
        for name, count in A.archetype_tally(fres).items():
            archtally[name] += count

    eventfits = {}
    for reg in sorted(EVENT_REGS):
        segs = A.fit_event_lane(state[:, reg])
        rendered[:, reg] = A.render_event_lane(segs, nframes)
        eventfits[reg] = segs
        nseg[f"r{reg}"] = len(segs)
        for name, count in A.archetype_tally(segs).items():
            archtally[name] += count

    for reg in range(NREG):
        resid[reg] = int(np.sum(rendered[:, reg] != state[:, reg]))
    return rendered, resid, dict(nseg), dict(archtally), genfits, eventfits


def discover_note_table_from_bus(records, min_hits=8):
    """Recover the freq note table by VALUE PROVENANCE from the bus: a freq-lo
    SID write ($D400/$D407/$D40E) whose value was just read from a contiguous,
    2-byte-strided RAM region whose pairs equal written (lo, hi) freqs.  Returns
    a list of 128 16-bit freqs or None.  No hardcoded address."""
    addr, val, rw = records["addr"], records["val"], records["rw"]
    score = defaultdict(int)
    sidlo = {0xD400, 0xD407, 0xD40E}
    widx = np.nonzero(np.isin(addr, list(sidlo)) & (rw == 1))[0]
    for write in widx[:6000]:
        value = int(val[write])
        for k in range(write - 1, max(0, write - 24), -1):
            if rw[k] == 0 and 0x0100 <= addr[k] < 0xD000 and int(val[k]) == value:
                score[int(addr[k])] += 1
                break
    if not score:
        return None
    if any(count >= min_hits for count in score.values()):
        base = min(k for k, c in score.items() if c >= min_hits)
    else:
        base = min(score, key=lambda k: -score[k])
    # Snapshot the 256-byte note-table window [base, base+256) as last-write-wins
    # per address.  Vectorised so a multispeed trace (tens of millions of RAM
    # accesses) builds the table in milliseconds rather than a Python per-access
    # loop; equivalent to the running dict (a later write to an address overwrites
    # an earlier one) restricted to the window we actually read.
    window = np.zeros(256, dtype=np.int64)
    sel = (addr >= base) & (addr < base + 256)
    if np.any(sel):
        off = addr[sel].astype(np.int64) - base
        window[off] = val[sel].astype(np.int64)  # last write wins (stable order)
    table = []
    for note in range(128):
        lo = int(window[2 * note]) if 2 * note < 256 else 0
        hi = int(window[2 * note + 1]) if 2 * note + 1 < 256 else 0
        table.append(lo | (hi << 8))
    return table
