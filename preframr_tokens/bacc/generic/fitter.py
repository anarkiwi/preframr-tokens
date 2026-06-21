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

# Hardware register widths drive the accumulator WRAP boundary the generic
# matchers fold a swept lane into (a hardware fact, not a per-tune constant): the
# SID frequency register is 16-bit but the pulse-width register is 12-bit
# (pw-hi only exposes its low nibble), so a free-running / swept PW accumulator
# wraps modulo 4096, not 65536.  Fitting the PW lane with the wrong 16-bit wrap
# turns each 4096-wrap into a spurious rate change, so a table-driven PW
# accumulator (maskaccum / dwellratewalk) cannot span a wrap and a single wrapping
# pulse-sweep fragments into one short piece per wrap cycle (past the cover's piece
# cap), leaving the whole lane un-fit; the 12-bit wrap folds it into one
# accumulator that wraps exactly as the chip does at 0xFFF.
FREQ_WIDTH = 0xFFFF
PW_WIDTH = 0xFFF


def _noteon_points(noteons, voice):
    """Note-on frames for a voice with the pre-roll boundary at frame 0."""
    ons = noteons[voice]
    if not ons or ons[0] != 0:
        ons = [0] + list(ons)
    return ons


def _unfit_frames(res):
    """Frames left uncovered by an un-fit (None) segment in a fitted lane -- the
    measure of how much of a lane the cover failed to recover."""
    return sum(stop - start for start, stop, fit in res if fit is None)


def fit_generator_lanes(state, note_table):
    """Fit the freq + pw (16-bit) lanes via the BACC archetype library.  Returns
    ``(fits, noteons)`` where fits maps ``(voice, 'freq'|'pw')`` to
    ``(segments, carry)``.

    The generator lanes are sliced at :func:`archetypes.note_boundaries` -- every
    bus-visible note-on retrigger (gate rise, control-byte change, or ADSR change),
    not just a gate rise -- so a legato / hard-restart phrase that keeps the gate
    high across notes is segmented per note instead of collapsing into one
    over-long, unfittable segment.  The FREQ lane additionally slices at
    :func:`archetypes.pw_sweep_resets` (a per-note pulse-sweep re-seed) and at
    :func:`archetypes.freq_note_onsets` (a freq jump far larger than the intra-note
    modulation step) so a pure-legato melody with NO control/ADSR retrigger -- one
    advancing its notes ONLY in the freq lane (per-note glide-vibrato or
    arp/vibrato cells) -- is still segmented per note instead of collapsing into one
    over-long, unfittable segment.

    The PW lane is fitted FIRST without any note-on re-slicing, so a smoothly-paced
    reflecting-triangle PW (FamiCommodore-style, covered whole by ``wavetable_ptr``)
    is never fragmented.  Only when that whole-segment cover leaves an un-fit (None)
    span -- a legato melody whose pulse-width re-seeds the sweep at each new note,
    collapsing the whole note into one over-long, unfittable PW block -- do we RETRY
    the PW lane sliced at the same bus-visible note-ons the freq lane uses
    (:func:`archetypes.pw_sweep_resets` -- PW's own re-seed drops -- and
    :func:`archetypes.freq_note_onsets` -- the FREQ-lane note jump, a chip-wide note
    event, NOT the PW lane's own reflection drops), and adopt the retry ONLY when it
    strictly reduces the un-fit span.  A genuinely irreducible PW (no clean per-note
    re-seed structure) reduces nothing on the retry and stays one surfaced segment
    rather than being faked into raw-byte pieces (HARD RULE #0)."""
    noteons = A.note_boundaries(state)
    nframes = len(state)
    # The song's global per-frame tick grid (note-ons across ALL voices).  Used
    # ONLY as a last-resort EXTRA slice set when a voice's OWN boundaries leave a
    # generator lane un-fit -- a voice whose freq/pw registers keep churning on the
    # song tick while its own gate is held low (a muted-voice intro / pre-roll /
    # arpeggio table under a silenced channel) exposes none of its own retriggers,
    # so the whole churn collapses into one over-long unfittable block; the other
    # voices' note-ons mark exactly those ticks.  Kept only when it strictly
    # recovers more of the lane, so an irreducible lane is still surfaced.
    grid = A.all_voice_boundaries(state)
    out = {}
    for voice in range(3):
        ons = _noteon_points(noteons, voice)
        resets = A.pw_sweep_resets(state, voice)
        freq_onsets = A.freq_note_onsets(state, voice)
        freq_ons = sorted(set(ons) | set(resets) | set(freq_onsets))
        flane = A.lane_freq(state, voice)
        fres = A.fit_lane(flane, freq_ons, nframes, note_table, None, FREQ_WIDTH)
        if _unfit_frames(fres):
            # A purely-legato melody can advance with NO control/ADSR/PW/gate
            # retrigger by zeroing the oscillator between held notes (a freq-zero
            # rest separator).  Then almost every nonzero step is a note-sized jump
            # to/from 0, so freq_note_onsets's 4*median threshold catches nothing and
            # the whole phrase is one over-long un-fit block.  Re-slice at the
            # bus-visible zero-crossing boundaries (note<->rest) and keep it only if
            # it strictly recovers more of the lane, so a generator that merely passes
            # through 0 (a sweep crossing zero) reduces nothing and is not fragmented.
            rest_ons = sorted(set(freq_ons) | set(A.freq_rest_boundaries(state, voice)))
            fres_rest = A.fit_lane(
                flane, rest_ons, nframes, note_table, None, FREQ_WIDTH
            )
            if _unfit_frames(fres_rest) < _unfit_frames(fres):
                fres = fres_rest
                freq_ons = rest_ons
        if _unfit_frames(fres):
            # The per-voice cover left a gap; re-slice the freq lane at the global
            # tick grid (the muted-voice churn case) and keep it only if it strictly
            # recovers more of the lane.
            fres_grid = A.fit_lane(
                flane,
                sorted(set(freq_ons) | set(grid)),
                nframes,
                note_table,
                None,
                FREQ_WIDTH,
            )
            if _unfit_frames(fres_grid) < _unfit_frames(fres):
                fres = fres_grid
        if _unfit_frames(fres):
            # A per-frame software arp / two-note chord (gate held high) writes a
            # different note slot every frame, so note_boundaries finds no retrigger
            # and the whole phrase collapses into one over-long un-fit block (every
            # frame is a note-sized jump, defeating freq_note_onsets too).  Retry the
            # un-fit span(s) the grid re-slice still left over as an N-phase
            # INTERLEAVE -- deinterleave into the round-robin note slots, each a normal
            # per-note vibrato/glide melody -- and adopt it ONLY where it actually
            # closes the gap, so a genuinely irreducible freq lane (which the
            # interleave matcher rejects) still surfaces rather than being faked into
            # raw bytes (HARD RULE #0).
            retried = []
            for start, stop, fit in fres:
                if fit is None:
                    inter = A.fit_interleaved_lane(
                        flane, start, stop, note_table, FREQ_WIDTH
                    )
                    if inter is not None:
                        fit = inter
                retried.append((start, stop, fit))
            fres = retried
        carry = A.freq_carry_sequence(fres, nframes)
        plane = A.lane_pw(state, voice)
        # The pulse-width register is 12-bit; fitting it with its true width lets a
        # table-driven PW accumulator (dwellratewalk) wrap exactly as the chip does
        # at the 0xFFF boundary rather than at 0xFFFF.
        pres = A.fit_lane(plane, ons, nframes, note_table, carry, PW_WIDTH)
        if _unfit_frames(pres):
            # The whole-note PW cover left a gap; a per-note PW re-seed (the sweep
            # snapping back at each new legato note) makes the held note one
            # over-long unfittable block.  Re-slice at the note-ons the freq lane
            # exposes (PW's own re-seed drops AND the chip-wide freq note jump) and
            # keep the result only if it actually recovers more of the lane, so an
            # irreducible reflecting-triangle PW (which reduces nothing) is never
            # fragmented.
            pres_sliced = A.fit_lane(
                plane, freq_ons, nframes, note_table, carry, PW_WIDTH
            )
            if _unfit_frames(pres_sliced) < _unfit_frames(pres):
                pres = pres_sliced
        if _unfit_frames(pres):
            # Still un-fit: the same muted-voice churn can drive the PW lane (a
            # pulse sweep stepping under a silenced channel).  Re-slice the PW lane
            # at the global tick grid as a last resort, kept only on strict gain.
            pres_grid = A.fit_lane(
                plane,
                sorted(set(ons) | set(grid)),
                nframes,
                note_table,
                carry,
                PW_WIDTH,
            )
            if _unfit_frames(pres_grid) < _unfit_frames(pres):
                pres = pres_grid
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
