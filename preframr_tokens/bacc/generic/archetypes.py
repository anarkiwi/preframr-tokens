"""The generic bounded-accumulator (BACC) archetype library.

Each archetype is a parameterised renderer plus a longest-byte-exact-prefix
matcher.  Given a note-on segment (the frames between two gate-rises) the fitter
greedily covers it with archetype runs whose rendered output reproduces the
observed lane EXACTLY (residual-zero) -- a generator program, never stored data.

Every archetype reads only SID-chip / arithmetic semantics; there is NO
per-driver constant here.  This is the driver-agnostic form of the recovery the
hand backends do by disassembly: hold / accum / dwellaccum / wrapaccum / arp /
glide / vibrato / pingpong / decay (+ the composites) cover the proven library,
and six generic periodic / wavetable generators close the generator-lane gaps:
:func:`render_maskaccum` (a fixed-period-paced accumulator), :func:`render_ratewalk`
(a period-P signed-rate wavetable accumulator -- the fractional-rate /
wider-internal-width sweep), :func:`render_dwellratewalk` (a period-P signed-step
table HELD a fixed dwell per entry and accumulated at the lane's true width -- the
table-driven reflecting / ramping PWM whose effective period ``dwell*P`` exceeds the
ratewalk cap, e.g. a HardTrack-style pulse wavetable), :func:`render_tablewalk`
(a period-P value table beyond the arp cap), :func:`render_tablewalk_lead` (a lead
hold then a period-P value table -- a delayed long-period modulation), and
:func:`render_wavetable_ptr` (a pointer over a period-P value table stepped by an
EXTERNAL per-voice advance clock -- a wavetable-paced walk whose drifting,
non-periodic dwell is the separable groove tick, not stored output).  ``pingpong``
also reflects at EITHER the visible extreme or one past it, so a fine triangle
vibrato that turns on its apex is one piece, not a stub.

``wavetable_ptr`` is admitted on the NON-GENERATOR lanes too (ctrl / AD / SR /
filter / volume), not just freq/pw: the dwell-paced PAGE-WALK player (e.g. Master
Composer) drives the WHOLE register file as ``page[ptr]`` with ``ptr`` advanced by
a non-uniform per-step dwell -- a SINGLE chip-wide step clock shared across every
lane -- so each non-generator lane is literally ``tablewalk(page_reg, ptr)`` over
that shared pointer.  Recovering it there collapses what the cheap library would
otherwise fragment into one short arp/hold piece per dwell run (its non-uniform
groove defeats the fixed-period ``arp`` / ``dwellaccum``) into one period-P table
plus the separable advance clock -- a genuine reused generator, never per-step
stored output (HARD RULE #0).

The ~19 clean generators above are all special cases of ONE op, the **Clocked
Indexed-Table Generator** (:func:`render_citg` + :func:`_prefix_citg`): a period-P
loop table read through a pointer advanced by a recovered advance-clock (every-frame
/ periodic-dwell / periodic-mask / external groove tick), the pointer either
SELECTING a value (READ) or selecting a signed step ADDED to a width-wrapped
accumulator (ACCUM), with a lead-stall, seed, phase and loop point.  CITG is tried
FIRST in the cover search (:func:`_longest_archetype_aug`) with the full zoo as the
FALLBACK; because every matcher here is byte-exact-or-None, trying CITG first can
never regress correctness, and :data:`CITG_FALLBACK_COUNTS` measures exactly which
archetypes CITG already subsumes vs. which still fall back (the trivial ``hold``
baseline, sub-3-frame arps, and the parametric/composite shapes of §2d --
``vibrato`` / ``pingpong`` / ``decay`` / ``wrapaccum`` / ``glide`` /
``vibskydive`` / ``arp_decay`` / ``additive_pw`` -- which stay in the zoo).
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np

_WINDOW = 384  # max frames a single archetype-run search inspects.
_MINRUN = 3  # minimum frames for a structured archetype run to beat hold.
# Max archetype pieces in one note-on cover before we give up (None).  Bounds the
# search and rejects a cover so fragmented it would be raw-byte storage in
# disguise (more pieces than a genuine generator program should need).
_MAXPIECES = 64

# CITG fallback instrumentation.  The unified Clocked Indexed-Table Generator is
# tried FIRST in the cover search; when it covers a run at least as long as the
# zoo would, CITG WINS and the zoo never runs (a measured step toward retiring the
# zoo).  When CITG declines or covers strictly less, the full archetype zoo is the
# fallback and we record which archetype it fell back to -- so a campaign can SEE
# exactly which cases CITG does not yet subsume.  Counting is cheap and always on
# (a module-level Counter); a per-decision stderr line is gated behind CITG_TRACE
# so normal runs are never spammed.  ``CITG_DISABLE`` turns the whole CITG-first
# path off (the zoo runs unchanged) -- an escape hatch, never needed for
# correctness since the matcher is byte-exact-or-None.
CITG_FALLBACK_COUNTS = Counter()
_CITG_TRACE = bool(os.environ.get("CITG_TRACE"))
_CITG_DISABLE = bool(os.environ.get("CITG_DISABLE"))


def reset_citg_counts():
    """Clear the CITG win/fallback tally (used by the campaign harness / tests to
    measure the CITG-won vs zoo-fallback proportion of a single fit)."""
    CITG_FALLBACK_COUNTS.clear()


def _citg_record(outcome, detail=""):
    """Tally one cover decision (``outcome`` is ``"citg_won"`` or
    ``"fallback:<zoo archetype>"``) and, under CITG_TRACE, emit a stderr line."""
    CITG_FALLBACK_COUNTS[outcome] += 1
    if _CITG_TRACE:
        print(f"CITG {outcome} {detail}".rstrip(), file=sys.stderr)


# ---------------------------------------------------------------------------
# Generic lane extraction + note-on detection (SID chip semantics).
# ---------------------------------------------------------------------------
def gate_noteons(state):
    """Per-voice note-on frames: the gate bit (ctrl bit0) rising 0->1."""
    res = {}
    for voice in range(3):
        gate = state[:, 7 * voice + 4] & 1
        rise = (gate[1:] == 1) & (gate[:-1] == 0)
        res[voice] = (np.nonzero(rise)[0] + 1).tolist()
    return res


def note_boundaries(state):
    """Per-voice note-on frames, generalising :func:`gate_noteons` to every
    bus-visible note-on RETRIGGER, not just a 0->1 gate edge.

    Some players advance the melody on an internal tempo counter while the gate
    bit stays HIGH across consecutive notes (legato / hard-restart style); the new
    note is then signalled on the bus not by a gate rise but by a fresh write to
    the voice's CONTROL byte (a waveform change or a 1-frame gate hard-restart) or
    to its ADSR registers (a new attack/decay/sustain/release at note-on).  A
    generator lane that is only sliced at gate rises therefore collapses a whole
    multi-note phrase into ONE over-long segment that no single archetype run can
    cover, leaving the lane unfit.

    This returns, per voice, the union of (a) the gate-rise frames, (b) the frames
    where the voice's control byte changes, and (c) the frames where the voice's
    AD or SR byte changes.  Every signal is a pure SID note-on semantic read from
    the reconstructed register state -- no per-driver constant.  It is PER VOICE
    (never unioned across voices): a genuinely single sustained note exposes none
    of these retriggers and stays one segment, so an irreducible lane is still
    surfaced, never sliced into raw-byte pieces.
    """
    res = {}
    for voice in range(3):
        base = 7 * voice
        ctrl = state[:, base + 4].astype(np.int64)
        ad = state[:, base + 5].astype(np.int64)
        sr = state[:, base + 6].astype(np.int64)
        gate = ctrl & 1
        rise = (gate[1:] == 1) & (gate[:-1] == 0)
        ctrl_change = np.diff(ctrl) != 0
        adsr_change = (np.diff(ad) != 0) | (np.diff(sr) != 0)
        ons = np.nonzero(rise | ctrl_change | adsr_change)[0] + 1
        res[voice] = ons.tolist()
    return res


def pw_sweep_resets(state, voice):
    """Frames where the voice's pulse-width SWEEP is re-seeded: a downward PW step
    far larger than the sweep's own typical per-frame step.

    A pure-legato instrument can advance the melody with NO control / ADSR / gate
    retrigger at all (:func:`note_boundaries` finds nothing), yet still RESET its
    pulse-width sweep at each new note -- a sharp PW drop back to the sweep start.
    That drop is a bus-visible note-on for the otherwise silent freq lane, so it is
    a useful EXTRA slice point for the freq lane.  The threshold is derived from the
    lane's own median nonzero step (no per-driver constant): a genuinely smooth PWM
    sweep never trips it, so a single sustained note is not over-sliced.

    NB: used ONLY to slice the FREQ lane.  It is deliberately NOT applied to the PW
    lane itself -- a reflecting-triangle PW has its own large reflection drops, and
    slicing the PW lane at them would fragment a genuine generator (or an
    irreducible lane) into raw-byte pieces and FAKE a residual-zero; the PW lane is
    instead covered as a whole (e.g. by ``pingpong`` / ``wavetable_ptr``) or, if
    irreducible, left as one segment so the gap surfaces (HARD RULE #0).
    """
    pw = lane_pw(state, voice)
    diff = np.diff(pw)
    nonzero = np.abs(diff[diff != 0])
    if nonzero.size == 0:
        return []
    threshold = max(8, 4 * int(np.median(nonzero)))
    return (np.nonzero(diff < -threshold)[0] + 1).tolist()


def freq_note_onsets(state, voice):
    """Frames where the voice's FREQUENCY lane jumps to a new note, the analogue of
    :func:`pw_sweep_resets` for a player that advances the melody PURELY in the freq
    lane -- gate held high, with no control / ADSR / PW retrigger at all, so
    :func:`note_boundaries` and :func:`pw_sweep_resets` both find nothing.

    Such a tune (e.g. RoMuzak's per-note glide-vibrato, Digitalizer's per-note
    arp/vibrato cell, or an Electrosound-style fixed-period vibrato riding on a centre
    that JUMPS by a note-table interval each note) signals each new note only as a
    freq step far larger than the intra-note modulation step (the vibrato wiggle /
    glide ramp).  Such a phrase is otherwise ONE over-long segment no single archetype
    can cover (a fixed-period vibrato around a *changing* centre is neither a single
    ``vibrato`` nor a periodic ``tablewalk``); that big step is a bus-visible note-on
    for the freq lane, so it is a useful EXTRA slice point and each per-note vibrato /
    glide / arp cell then becomes one short, fittable fixed-centre segment (a per-note
    ``vibrato`` / ``tablewalk`` / ``glide`` / ``hold``).

    The threshold is derived from the lane's OWN median nonzero step (no per-driver
    constant), so a genuinely smooth single sweep -- whose steps are all near the
    median -- and a fine vibrato around one centre never trip it and are left as one
    segment to surface if irreducible.  Both up and down jumps are returned (a melody
    moves in either direction).

    NB: used ONLY to slice the FREQ lane (like :func:`pw_sweep_resets`).  It is NOT
    applied to the PW lane, where a reflecting-triangle PWM's own large reflection
    drops would be misread as note-ons and fragment a genuine generator into raw
    bytes, faking a residual-zero (HARD RULE #0)."""
    fr = lane_freq(state, voice)
    diff = np.diff(fr.astype(np.int64))
    nonzero = np.abs(diff[diff != 0])
    if nonzero.size == 0:
        return []
    threshold = max(8, 4 * int(np.median(nonzero)))
    return (np.nonzero(np.abs(diff) > threshold)[0] + 1).tolist()


def freq_rest_boundaries(state, voice):
    """Frames where the voice's FREQUENCY lane enters or leaves a hard rest -- a
    write of frequency 0 -- the analogue of :func:`freq_note_onsets` for a player
    that separates legato notes by zeroing the oscillator.

    Some bespoke game drivers (e.g. Parker Bros.: Gyruss, Party_Quiz) advance a
    PURELY-legato melody -- gate held high forever, no control / ADSR / PW / gate
    retrigger, so :func:`note_boundaries`, :func:`pw_sweep_resets` and
    :func:`freq_note_onsets` all find nothing -- by writing frequency 0 for one (or
    a few) frames BETWEEN held notes (a freq-zero note separator / rest).  Because
    almost every nonzero freq step is then itself a note-sized jump to or from 0,
    the median nonzero step is huge and :func:`freq_note_onsets`'s ``4*median``
    threshold catches no jump at all, so the whole phrase collapses into one
    over-long, unfittable segment.

    A frequency of exactly 0 under a held gate is a chip-visible event -- the player
    explicitly silences the oscillator -- so the frames where the lane crosses the
    zero boundary (note -> rest and rest -> note) are bus-visible note-ons for the
    freq lane.  Slicing there turns each held note into one closed-form ``hold``
    (the note from the table) and each rest into one ``hold(0)``: the recovered
    melody is the note/rest list (the score), every value a generator, no per-frame
    output stored (HARD RULE #0).  This is purely a bus-derived, last-resort EXTRA
    slice set: it is adopted only when it strictly recovers more of the lane (see
    :func:`fitter.fit_generator_lanes`), so a genuine generator that merely passes
    through 0 (a sweep crossing zero, a vibrato around a low centre) reduces nothing
    on the retry and is left as one segment, never fragmented."""
    fr = lane_freq(state, voice).astype(np.int64)
    is_rest = fr == 0
    if not np.any(is_rest):
        return []
    return (np.nonzero(is_rest[:-1] != is_rest[1:])[0] + 1).tolist()


def all_voice_boundaries(state):
    """The union, across ALL three voices, of every bus-visible note-on
    (:func:`note_boundaries`) and pulse-sweep re-seed (:func:`pw_sweep_resets`) --
    the song's global per-frame tick grid.

    Some players (AMP / TFX / Cyberlogic-style) re-write a voice's freq / pulse
    registers EVERY frame the song advances, including while that voice's own gate
    is held LOW (a muted-voice intro, an arpeggio table that keeps stepping under a
    silenced channel, or a long pre-roll before the channel's first gate rise).
    The per-voice :func:`note_boundaries` then finds NO slice points inside that
    span -- the voice exposes none of its own retriggers -- so the whole churning
    span collapses into ONE over-long segment no single archetype run can cover.

    But the churn is NOT structureless: it advances on the same song tick as the
    OTHER voices, whose note-ons mark exactly those ticks on the bus.  Slicing the
    silent voice's lane at the global tick grid (this union) exposes the per-tick
    generator the player runs.  This is purely a bus-derived, last-resort EXTRA set
    of slice points: it is applied only as a fallback when the per-voice cover left
    an un-fit span and is kept only when it strictly recovers more of the lane
    (see :func:`fitter.fit_generator_lanes`), so a genuinely irreducible lane
    reduces nothing on the retry and is still surfaced, never sliced into raw-byte
    pieces (HARD RULE #0)."""
    grid = set()
    for voice in range(3):
        grid.update(note_boundaries(state)[voice])
        grid.update(pw_sweep_resets(state, voice))
    return sorted(grid)


def lane_freq(state, voice):
    """The voice's 16-bit frequency lane (freq-lo | freq-hi << 8)."""
    base = 7 * voice
    return state[:, base + 0].astype(np.int64) + 256 * state[:, base + 1].astype(
        np.int64
    )


def lane_pw(state, voice):
    """The voice's 12-bit pulse-width lane (pw-lo | (pw-hi & 0xF) << 8)."""
    base = 7 * voice
    return state[:, base + 2].astype(np.int64) + 256 * (
        state[:, base + 3].astype(np.int64) & 0xF
    )


# ---------------------------------------------------------------------------
# Archetype renderers.
# ---------------------------------------------------------------------------
def tri_phase(ctr):
    """Vibrato triangle phase: osc = ctr & 7; if osc >= 4 osc ^= 7 -> 0..3 up,
    3..0 down."""
    osc = ctr & 7
    if osc >= 4:
        osc ^= 7
    return osc


def _tri_phase_seq(ctr0, seg_len):
    """The triangle phase 0..3 for the ``seg_len`` consecutive counters starting at
    ``ctr0`` -- the vectorised :func:`tri_phase`.  Folding the phase into one numpy
    array lets the vibrato renderers index a precomputed per-phase value table
    instead of looping :func:`tri_phase` per frame (the recovery's dominant cost on
    a long vibrato lane was the per-frame Python phase loop)."""
    osc = (ctr0 + np.arange(seg_len, dtype=np.int64)) & 7
    return np.where(osc >= 4, osc ^ 7, osc)


def render_vibrato(seg_len, base, amp_step, ctr0):
    """value = base + tri_phase(ctr) * amp_step, 16-bit wrap.

    The triangle phase only ever takes the four values 0..3, so the four possible
    outputs are computed once and indexed by the per-frame phase sequence -- a
    vectorised form byte-identical to the per-frame ``(base + tri_phase(ctr)*amp)``
    loop but O(seg_len) numpy rather than a Python loop."""
    phase = _tri_phase_seq(ctr0, seg_len)
    table = (base + np.arange(4, dtype=np.int64) * amp_step) & 0xFFFF
    return table[phase]


def render_accum(seg_len, v0, rate, width_mask):
    """Linear accumulator (portamento / sweep): value += rate each frame."""
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val += rate
    return out


def render_wrapaccum(seg_len, v0, rate, lo_b, hi_b):
    """Modulo-wrap accumulator (free-running sawtooth PWM): value += rate every
    frame, wrapping by (hi_b - lo_b) when it crosses a bound."""
    out = np.empty(seg_len, dtype=np.int64)
    span = hi_b - lo_b
    val = v0
    for i in range(seg_len):
        out[i] = val
        val += rate
        if rate > 0 and val >= hi_b:
            val -= span
        elif rate < 0 and val < lo_b:
            val += span
    return out


def render_arp(seg_len, notes_freqs, period, ctr0, dwell=1):
    """Table-walk arp: cycle period-P over a small freq list, each held
    ``dwell`` frames."""
    out = np.empty(seg_len, dtype=np.int64)
    period = len(notes_freqs)
    for i in range(seg_len):
        step = (ctr0 + i) // dwell
        out[i] = notes_freqs[step % period]
    return out


def render_glide(seg_len, n0, step, dwell, lead, note_table, ctr0=0):
    """Note-table-walk glide: after a ``lead``-frame hold at index n0, walk the
    note table by ``step`` entries every ``dwell`` frames."""
    _ = ctr0
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        k = 0 if i < lead else (i - lead) // dwell
        idx = (n0 + step * k) & 0xFF
        out[i] = note_table[idx] if 0 <= idx < len(note_table) else 0
    return out


def _vibrato_exact_phase_tables(base, amp):
    """The four (freq, carry) outcomes of the byte-wise vibrato add for triangle
    phases osc=0..3 -- a single byte-wise add of ``amp`` repeated osc times from
    ``base``.  ``freq[osc] = (base + osc*amp) & 0xFFFF``; ``carry[osc]`` is the
    no-CLC carry-out the LAST hi-byte add leaves (1 for osc=0, since no add occurs
    and the player's CLC-less path inherits the prior set carry).  Precomputing the
    four outcomes lets :func:`render_vibrato_exact` index them by the phase sequence
    rather than re-run the inner add loop per frame -- byte-identical, O(1) here."""
    base &= 0xFFFF
    dlo, dhi = amp & 0xFF, (amp >> 8) & 0xFF
    lo, hi = base & 0xFF, (base >> 8) & 0xFF
    freq = [lo | (hi << 8)]
    carry = [1]  # osc==0: no add, carry-out is the player's pre-set carry (1)
    carry_bit = 0
    for _ in range(3):
        tmp = lo + dlo
        lo = tmp & 0xFF
        tmp = hi + dhi + (tmp >> 8)
        hi = tmp & 0xFF
        carry_bit = tmp >> 8
        freq.append(lo | (hi << 8))
        carry.append(carry_bit)
    return np.array(freq, dtype=np.int64), np.array(carry, dtype=np.int64)


def render_vibrato_exact(seg_len, base, amp, ctr0):
    """Exact byte-wise vibrato: triangle phase 0..3, freq computed by repeating
    a byte-wise 16-bit add of ``amp`` osc times.  Returns (freq_seq, carry_seq)
    where carry_seq[i] is the no-CLC carry-out the add leaves on frame i (the
    freq->pw coupling).

    The triangle phase only ever takes the four values 0..3, so the four (freq,
    carry) outcomes are precomputed (:func:`_vibrato_exact_phase_tables`) and
    indexed by the per-frame phase sequence -- byte-identical to the nested
    per-frame add loop but O(seg_len) numpy, which removes the dominant per-frame
    Python cost the vibrato/vibskydive search incurred on long lanes."""
    phase = _tri_phase_seq(ctr0, seg_len)
    freq_t, carry_t = _vibrato_exact_phase_tables(base, amp)
    return freq_t[phase], carry_t[phase]


def render_vibrato_table(seg_len, base, amp, phase_table, ctr0=0):
    """Generic byte-wise vibrato with a searched period-P LFO phase table.
    Returns (freq_seq, carry_seq) -- the generic form of render_vibrato_exact."""
    period = len(phase_table)
    out = np.empty(seg_len, dtype=np.int64)
    carry = np.zeros(seg_len, dtype=np.int64)
    base &= 0xFFFF
    dlo, dhi = amp & 0xFF, (amp >> 8) & 0xFF
    for i in range(seg_len):
        osc = int(phase_table[(ctr0 + i) % period])
        carry_bit = 1 if osc == 0 else 0
        lo, hi = base & 0xFF, (base >> 8) & 0xFF
        for _ in range(osc):
            tmp = lo + dlo
            lo = tmp & 0xFF
            tmp = hi + dhi + (tmp >> 8)
            hi = tmp & 0xFF
            carry_bit = tmp >> 8
        out[i] = lo | (hi << 8)
        carry[i] = carry_bit
    return out, carry


def _hi_overlay(base_lane, sfh0, par, ctr0):
    """Overlay a descending hi-byte counter (drums/skydive) on a base lane."""
    base_lane = np.asarray(base_lane, dtype=np.int64)
    length = len(base_lane)
    out = np.empty(length, dtype=np.int64)
    sfh = sfh0 & 0xFF
    for i in range(length):
        lo = int(base_lane[i]) & 0xFF
        hi = (int(base_lane[i]) >> 8) & 0xFF
        if ((ctr0 + i) & 1) == par and sfh != 0:
            old = sfh
            sfh = (sfh - 1) & 0xFF
            hi = old
        out[i] = lo | (hi << 8)
    return out


def render_vibskydive(seg_len, base, amp, ctr0, sfh0, par):
    """Composite vibrato + skydive: vibrato on the full value, a descending
    hi-byte counter overlaid on parity ``par``."""
    vib, _ = render_vibrato_exact(seg_len, base, amp, ctr0)
    return _hi_overlay(vib, sfh0, par, ctr0)


def render_arp_decay(seg_len, freqs, period, dwell, sfh0, par, ctr0=0):
    """Arp (full-value table-walk) with a drums/skydive hi-byte countdown
    overlaid on parity ``par``."""
    base = render_arp(seg_len, freqs, period, ctr0, dwell)
    return _hi_overlay(base, sfh0, par, ctr0)


def render_additive_pw(seg_len, p0, pulsevalue, carry_seq, width_mask=0xFFF):
    """Carry-coupled additive simple-pw: pwlo += pulsevalue + carry every frame,
    where carry is the freq generator's per-frame carry-out."""
    out = np.empty(seg_len, dtype=np.int64)
    hi = p0 & ~0xFF
    lo = p0 & 0xFF
    for i in range(seg_len):
        out[i] = (hi | lo) & width_mask
        carry = int(carry_seq[i]) if i < len(carry_seq) else 0
        lo = (lo + pulsevalue + carry) & 0xFF
    return out


def render_pingpong(seg_len, v0, rate, lo_b, hi_b, dwell, d0, dir0):
    """Reflect accumulator (pulse ping-pong / triangle PWM): value steps +/-rate
    every (dwell+1) frames, reflecting at lo_b/hi_b."""
    out = np.empty(seg_len, dtype=np.int64)
    val, dwell_left, direction = v0, d0, dir0
    for i in range(seg_len):
        out[i] = val
        dwell_left -= 1
        if dwell_left < 0:
            dwell_left = dwell
            if direction:
                nxt = val - rate
                if nxt < lo_b:
                    direction = 0
                    nxt = val + rate
                val = nxt
            else:
                nxt = val + rate
                if nxt > hi_b:
                    direction = 1
                    nxt = val - rate
                val = nxt
    return out


def render_pingfold(seg_len, step, frac, lo, hi, acc0, dir0):
    """Mirror-reflecting fixed-point triangle accumulator.

    An INTERNAL accumulator runs at ``frac`` extra fractional bits and steps by a
    constant ``step`` per frame (so the observable per-frame increment is the
    fractional ``step / 2**frac`` -- e.g. the alternating +2/+3 of a step-5,
    1-fractional-bit ramp); the emitted lane value is ``acc >> frac``.  When the
    accumulator crosses a bound it MIRROR-folds -- ``acc = 2*bound - acc`` -- and
    reverses direction, so the overshoot past the extreme is reflected back exactly
    as it would by an ``abs``-style bounce rather than clamped at the extreme.

    This is the freq/PW vibrato/LFO a player runs as a triangle over an 8-bit
    register: a constant internal increment bouncing between an upper and lower
    bound (MusicShop's voice freq-lo triangle, a slow per-frame LFO sweep over a
    fixed window).  It generalises :func:`render_pingpong` -- whose turn-back
    convention clamps at the visible extreme and so cannot reproduce the gradual
    apex drift a fractional internal increment leaves -- to a true mirror-fold at
    the bound with internal sub-register precision, recovering the WHOLE
    long-period triangle as one closed-form rule (step, frac, bounds) rather than
    storing the per-frame ramp.  ``lo``/``hi`` are the internal (shifted) fold
    bounds and ``acc0``/``dir0`` the internal seed, all bus-recovered."""
    out = np.empty(seg_len, dtype=np.int64)
    acc, direction = acc0, dir0
    for i in range(seg_len):
        out[i] = acc >> frac
        acc += step * direction
        if acc > hi:
            acc = 2 * hi - acc
            direction = -direction
        elif acc < lo:
            acc = 2 * lo - acc
            direction = -direction
    return out


def render_decay(seg_len, v0, rate, every, ctr0):
    """Drum / skydive: value decrements by ``rate`` every ``every`` frames,
    emitting the pre-decrement value."""
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val
        if (ctr0 + i + 1) % every == 0:
            val = (val - rate) & 0xFFFF
    return out


def render_dwell_accum(seg_len, v0, rate, dwell, lead, ctr0):
    """value += rate every ``dwell`` frames after a ``lead``-frame hold."""
    _ = ctr0
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    counter = 0
    for i in range(seg_len):
        out[i] = val & 0xFFFF
        if i >= lead:
            counter += 1
            if counter % dwell == 0:
                val = (val + rate) & 0xFFFF
    return out


def render_maskaccum(seg_len, v0, rate, mask, width_mask=0xFFFF):
    """Periodic-dwell accumulator: value += rate on frames where the period-P
    boolean ``mask`` is set (0 = hold).  A wavetable-paced sweep that steps the
    accumulator on a fixed-period pattern rather than every frame."""
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    period = len(mask)
    for i in range(seg_len):
        out[i] = val & width_mask
        if mask[i % period]:
            val = (val + rate) & width_mask
    return out


def render_tablewalk(seg_len, table, ctr0=0):
    """Periodic table walk: out[i] = table[(ctr0 + i) % P].  Closed-form
    periodic generator for LFO modulations of any period P (the arp primitive
    without the P<=6 cap)."""
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        out[i] = table[(ctr0 + i) % period]
    return out


def render_ratewalk(seg_len, v0, rate_table, ctr0=0, width_mask=0xFFFF):
    """Wavetable-rate accumulator: value += rate_table[(ctr0 + i) % P] each frame,
    width-masked.  This is the wider-internal-width / fractional-rate sweep where
    the player accumulates an RMW variable whose per-frame step is sequenced by a
    short period-P rate wavetable, viewed through the (possibly narrower) register.
    It generalises :func:`render_maskaccum` (one rate gated by a 0/1 mask) to a
    full period-P signed-rate table, so a sub-resolution sweep whose effective rate
    drifts on a fixed pattern is one rule, not stored data."""
    period = len(rate_table)
    out = np.empty(seg_len, dtype=np.int64)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[(ctr0 + i) % period]) & width_mask
    return out


def render_dwellratewalk(seg_len, v0, rate_table, dwell, ctr0=0, width_mask=0xFFFF):
    """Dwelled wavetable-rate accumulator: a pointer over a period-P signed-rate
    ``rate_table`` whose CURRENT entry is held ``dwell`` frames before the pointer
    advances, accumulating ``value += rate_table[(ctr//dwell) % P]`` each frame,
    width-masked.

    This is the table-driven pulse / sweep accumulator a tracker player runs from a
    pulse wavetable whose columns are (signed step, direction, dwell): each table
    entry contributes a fixed signed step for ``dwell`` ticks, then the pointer
    steps to the next entry, looping at the table end.  It generalises
    :func:`render_ratewalk` (one rate per frame) by folding the dwell column out of
    the rate table, so a reflecting / ramping PWM whose step magnitude and direction
    are sequenced by a short wavetable (held several frames each) is ONE rule -- the
    P-entry step table plus the scalar dwell -- not a ``dwell*P``-entry rate table.
    The ``width_mask`` is the lane's own width (0xFFF for the 12-bit pulse-width
    register), so a step crossing the register boundary wraps exactly as the chip's
    accumulator does, never as stored data."""
    out = np.empty(seg_len, dtype=np.int64)
    period = len(rate_table)
    val = v0
    for i in range(seg_len):
        out[i] = val & width_mask
        val = (val + rate_table[((ctr0 + i) // dwell) % period]) & width_mask
    return out


def render_tablewalk_lead(seg_len, lead, value0, table, ctr0=0):
    """A ``lead``-frame constant hold at ``value0`` followed by a period-P value
    table walk -- a DELAYED periodic modulation (a long sustain, then an LFO
    offset table).  Folding the lead hold into one rule lets the cover reach the
    long-period table in a single piece, so a short coincidental arp at the note
    start cannot shadow the genuine (longer-period) modulation that follows."""
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    for i in range(seg_len):
        out[i] = value0 if i < lead else table[(ctr0 + i - lead) % period]
    return out


def render_wavetable_ptr(seg_len, table, phase, advance):
    """Advance-clocked wavetable-pointer walk: a pointer over a period-P value
    ``table`` that advances +1 (mod P) on each frame where the EXTERNAL advance
    clock ``advance`` is set, and HOLDS the prior table entry otherwise.

    This is the wavetable-pointer engine common to tracker players: a per-tune
    value table read through a pointer the player steps on a wavetable TICK, where
    the tick stream is a separate per-voice groove/tempo divider -- so the pointer
    advances APERIODICALLY across frames (some frames step, some hold).  It
    generalises :func:`render_maskaccum` (a single rate gated by a periodic mask)
    to a full value-table walk gated by the voice's advance clock: ``maskaccum``
    answers "how much to add and when", ``wavetable_ptr`` answers "which table
    entry, and when to step".  ``advance`` is the bus-recovered tick stream (the
    frames at which the lane steps); it is SHARED across the voice's animated lanes
    (the same groove paces them all), so it is a separable driver clock, never the
    lane's stored output -- the closed-form value content is the period-P table."""
    period = len(table)
    out = np.empty(seg_len, dtype=np.int64)
    ptr = phase % period
    for i in range(seg_len):
        if i > 0 and advance[(i - 1) % len(advance)]:
            ptr = (ptr + 1) % period
        out[i] = table[ptr]
    return out


# ---------------------------------------------------------------------------
# The unified Clocked Indexed-Table Generator (CITG).
#
# Every generator archetype above is a special case of ONE op: a period-P loop
# table TABLE read through a pointer that advances on a recovered ADVANCE-CLOCK,
# the pointer either SELECTING a value (MODE="read") or selecting a signed step
# ADDED to a width-wrapped accumulator (MODE="accum").  The clock is one of:
#   * "every"   -- the pointer advances every frame (tablewalk / accum / ratewalk);
#   * "dwell"   -- the pointer advances every ``dwell`` frames (arp / dwellaccum /
#                  dwellratewalk), after an optional ``lead``-frame stall;
#   * "mask"    -- the pointer advances on a period-Q boolean mask (maskaccum: the
#                  mask gates the accumulate of a single-entry step table);
#   * "advance" -- the pointer advances on an EXTERNAL per-voice advance vector --
#                  the separable groove tick that paces a wavetable_ptr walk.
# ``lead`` holds the seed for ``lead`` frames before the clock arms; ``phase`` is
# the pointer's start index (the ``ctr0`` of the periodic forms); ``loop`` is the
# index the pointer wraps to (default 0 -- HARD RULE #0 requires a loop, never a
# one-shot store).  ``width`` masks the accumulator (16-bit freq / 12-bit pw) and
# is the only WRAP rule the accumulator forms here need (modulo); the reflecting /
# triangle shapes stay in the zoo as ``pingpong`` / ``pingfold`` fallbacks.
#
# This renderer is a STRICT superset of the per-archetype renderers: every preset
# in :func:`citg_preset` renders byte-identically to its zoo renderer (proven by a
# parametrized parity test), so trying CITG first can never regress correctness --
# the matcher is byte-exact-or-None and the zoo runs as fallback when it declines.
# ---------------------------------------------------------------------------
def _citg_gates(seg_len, clock, lead):
    """The (accumulate-gate, pointer-step) clock streams for a CITG run.

    Returns two boolean ``int64`` arrays of length ``seg_len``: ``acc_gate[i]`` is
    whether the accumulator adds the current table entry on frame ``i`` (ACCUM
    mode), and ``ptr_step[i]`` is whether the pointer advances to the next table
    entry AFTER frame ``i``.  Splitting the two is what lets ONE op subsume both the
    "hold the accumulator on non-clock frames" forms (maskaccum / dwellaccum -- a
    single-entry step table gated by the clock, pointer never moving) and the
    "accumulate every frame but step the entry on a sub-clock" form (dwellratewalk
    -- the pointer advances on the dwell while the accumulate fires every frame).
    Both gates are inert during the ``lead``-frame stall (the clock arms after it),
    so a lead-hold is a genuine stall, never stored output.

    Clock kinds:
      * ``every``     -- add and step every armed frame (accum / ratewalk / the
                         every-frame tablewalk read);
      * ``dwell``     -- add and step every ``dwell`` armed frames (dwellaccum, and
                         the arp read whose pointer advances every dwell);
      * ``dwell_ptr`` -- add every armed frame, step the pointer every ``dwell``
                         armed frames (dwellratewalk);
      * ``mask``      -- add and step on a period-Q boolean mask (maskaccum);
      * ``advance``   -- add and step on an EXTERNAL advance vector (the
                         groove-paced wavetable_ptr walk)."""
    acc_gate = np.zeros(seg_len, dtype=np.int64)
    ptr_step = np.zeros(seg_len, dtype=np.int64)
    kind = clock["kind"]
    advance = clock.get("advance")
    alen = len(advance) if advance else 1
    mask = clock.get("mask")
    dwell = clock.get("dwell", 1)
    fired = 0  # armed frames seen so far (after the lead stall)
    for i in range(seg_len):
        if i < lead:
            continue
        if kind == "every":
            add = step = True
        elif kind == "dwell":
            add = step = (fired + 1) % dwell == 0
        elif kind == "dwell_ptr":
            add = True
            step = (fired + 1) % dwell == 0
        elif kind == "mask":
            add = step = bool(mask[fired % len(mask)])
        else:  # advance
            add = step = bool(advance[fired % alen])
        acc_gate[i] = add
        ptr_step[i] = step
        fired += 1
    return acc_gate, ptr_step


def render_citg(params, seg_len):
    """Render the unified Clocked Indexed-Table Generator to a lane of length
    ``seg_len``.  ``params`` carries ``mode`` ("read" | "accum"), ``table`` (values
    for read, signed steps for accum), ``clock`` (the advance schedule -- see
    :func:`_citg_gates`), ``seed`` (the read value0 / accum acc0), ``lead``,
    ``phase`` (the pointer's start index), ``loop`` (the index the pointer wraps to,
    default 0), and ``width`` (the accumulator mask, ACCUM only).

    In READ mode the lane is the constant ``seed`` during the ``lead`` stall and
    then ``table[ptr]`` with the pointer stepped by the clock -- a lead-hold then a
    table walk.  In ACCUM mode the lane is the running accumulator: the value
    emitted on frame ``i`` is the accumulator BEFORE this frame's add, so the seed
    is emitted first and every step is width-wrapped exactly as the chip's RMW
    accumulator wraps (modulo the lane width) -- a closed-form program, never
    stored output (HARD RULE #0)."""
    mode = params["mode"]
    table = params["table"]
    period = len(table)
    lead = params.get("lead", 0)
    phase = params.get("phase", 0)
    loop = params.get("loop", 0)
    seed = params.get("seed", 0)
    acc_gate, ptr_step = _citg_gates(seg_len, params["clock"], lead)
    out = np.empty(seg_len, dtype=np.int64)
    ptr = phase % period if period else 0
    if mode == "read":
        for i in range(seg_len):
            out[i] = seed if i < lead else table[ptr]
            if ptr_step[i] and period:
                ptr += 1
                if ptr >= period:
                    ptr = loop
        return out
    width = params.get("width", 0xFFFF)
    val = seed & width
    for i in range(seg_len):
        out[i] = val
        if acc_gate[i]:
            val = (val + int(table[ptr])) & width
        if ptr_step[i] and period:
            ptr += 1
            if ptr >= period:
                ptr = loop
    return out


def citg_preset(name, prm):
    """Map a zoo archetype ``(name, params)`` to the equivalent CITG ``params`` so
    :func:`render_citg` reproduces that archetype's output byte-for-byte, or None
    for an archetype that is NOT a single clean CITG (the ``vibrato`` /
    ``pingpong`` parametric shapes and the ``vibskydive`` / ``arp_decay`` /
    ``additive_pw`` composites -- §2d of the design doc -- which stay in the zoo).

    This is the executable form of the doc's §2c parameterization table; it is used
    only by the render-parity test (each preset == its zoo renderer over parameter
    sweeps), proving CITG's renderer is a strict superset of the zoo's."""
    if name in ("hold", "empty"):
        value = prm.get("value", 0)
        return {
            "mode": "read",
            "table": [value],
            "clock": {"kind": "every"},
            "seed": value,
        }
    if name == "tablewalk":
        table = prm["table"]
        return {"mode": "read", "table": table, "clock": {"kind": "every"}}
    if name == "tablewalk_lead":
        return {
            "mode": "read",
            "table": prm["table"],
            "clock": {"kind": "every"},
            "lead": prm["lead"],
            "seed": prm["value0"],
        }
    if name == "arp":
        return {
            "mode": "read",
            "table": prm["freqs"],
            "clock": {"kind": "dwell", "dwell": prm.get("dwell", 1)},
        }
    if name == "wavetable_ptr":
        return {
            "mode": "read",
            "table": prm["table"],
            "clock": {"kind": "advance", "advance": prm["advance"]},
            "phase": prm.get("phase", 0),
        }
    if name == "accum":
        return {
            "mode": "accum",
            "table": [prm["rate"]],
            "clock": {"kind": "every"},
            "seed": prm["v0"],
            "width": 0xFFFF,
        }
    if name == "maskaccum":
        return {
            "mode": "accum",
            "table": [prm["rate"]],
            "clock": {"kind": "mask", "mask": prm["mask"]},
            "seed": prm["v0"],
            "width": prm.get("width", 0xFFFF),
        }
    if name == "dwellaccum":
        return {
            "mode": "accum",
            "table": [prm["rate"]],
            "clock": {"kind": "dwell", "dwell": prm["dwell"]},
            "lead": prm["lead"],
            "seed": prm["v0"],
            "width": 0xFFFF,
        }
    if name == "ratewalk":
        return {
            "mode": "accum",
            "table": prm["rate_table"],
            "clock": {"kind": "every"},
            "seed": prm["v0"],
            "width": prm.get("width", 0xFFFF),
        }
    if name == "dwellratewalk":
        return {
            "mode": "accum",
            "table": prm["rate_table"],
            "clock": {"kind": "dwell_ptr", "dwell": prm["dwell"]},
            "seed": prm["v0"],
            "width": prm.get("width", 0xFFFF),
        }
    return None


# ---------------------------------------------------------------------------
# Prefix matchers (each returns the LONGEST byte-exact prefix it covers).
# ---------------------------------------------------------------------------
def _match_prefix(rend, seg):
    length = min(len(rend), len(seg))
    eq = rend[:length] == seg[:length]
    if eq.all():
        return length
    return int(np.argmin(eq))


def _detect_period(arr, maxp=8):
    arr = np.asarray(arr)
    length = len(arr)
    for period in range(1, min(maxp, length) + 1):
        if length < 2 * period:
            continue
        if all(arr[i] == arr[i % period] for i in range(length)):
            return period
    return None


def _prefix_wrapaccum(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 6:
        return None
    diffs = np.diff(seg)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 3:
        return None
    vals, cnts = np.unique(nonzero, return_counts=True)
    rate = int(vals[np.argmax(cnts)])
    if rate == 0:
        return None
    lo_b = int(seg.min())
    seen_max = int(seg.max())
    # The wrap boundary is DATA-DETERMINED, not brute-forced.  For rate>0 a wrap
    # frame shows up as ``out[i+1] = out[i] + rate - span`` (span = hi_b - lo_b), so
    # every step against the rate's sign pins exactly one candidate boundary
    # ``hi_b = lo_b + rate - step`` (and mirror for rate<0).  Iterating only those
    # candidates (plus the no-wrap boundary just past ``seen_max``) tests every hi_b
    # that could EVER produce a matching wrap -- byte-exact-identical to the old
    # scan -- in O(N) instead of O(|rate|), which had blown up to millions of renders
    # on wide multispeed freq sweeps (rate ~ thousands).
    steps = np.diff(seg)
    if rate > 0:
        # wrap: out[i+1] = out[i] + rate - span  ->  span = rate - step (step < 0)
        wrap_steps = steps[steps < 0]
        cand = {lo_b + rate - int(s) for s in wrap_steps}
    else:
        # wrap: out[i+1] = out[i] + rate + span  ->  span = step - rate (step > 0)
        wrap_steps = steps[steps > 0]
        cand = {lo_b + int(s) - rate for s in wrap_steps}
    cand.add(seen_max + 1)  # the no-wrap boundary (pure accum within the window)
    best = None
    for hi_b in sorted(c for c in cand if c > seen_max):
        rend = render_wrapaccum(length, int(seg[0]), rate, lo_b, hi_b)
        match = _match_prefix(rend, seg)
        if match >= 6 and (best is None or match > best[0]):
            best = (
                match,
                "wrapaccum",
                {"v0": int(seg[0]), "rate": rate, "lo": lo_b, "hi": hi_b},
            )
            if match == length:
                break
    return best


def _amp_divisors(pos):
    gcd = int(pos[0])
    for val in pos[1:]:
        gcd = np.gcd(gcd, int(val))
    cands = set()
    if 0 < gcd < 0x4000:
        cands.add(gcd)
    for phase in (1, 2, 3):
        if gcd % phase == 0 and 0 < gcd // phase < 0x4000:
            cands.add(gcd // phase)
    return sorted(cands, reverse=True)


def _prefix_vibrato(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < _MINRUN:
        return []
    bases = {int(seg[0]), int(seg.min())}
    cand_amp = set()
    for vib_base in bases:
        devs = seg - vib_base
        for ph0 in range(8):
            for i in range(min(length, 24)):
                ph = tri_phase(ph0 + i)
                if ph > 0 and devs[i] > 0 and devs[i] % ph == 0:
                    cand_amp.add(int(devs[i] // ph))
    cand_amp = sorted(a for a in cand_amp if 0 < a < 0x4000)
    best = None
    for vib_base in bases:
        for amp in cand_amp:
            for ph0 in range(8):
                rend = render_vibrato(length, vib_base, amp, ph0)
                match = _match_prefix(rend, seg)
                if match >= _MINRUN and (best is None or match > best[0]):
                    best = (
                        match,
                        "vibrato",
                        {"base": vib_base, "amp_step": amp, "ctr0": ph0},
                    )
                rex, _ = render_vibrato_exact(length, vib_base, amp, ph0)
                mex = _match_prefix(rex, seg)
                if mex >= _MINRUN and (best is None or mex > best[0]):
                    best = (
                        mex,
                        "vibrato_exact",
                        {"base": vib_base, "amp": amp, "ctr0": ph0},
                    )
    return [best] if best is not None else []


def _has_hi_countdown(seg):
    seg = np.asarray(seg, dtype=np.int64)
    hi = (seg >> 8) & 0xFF
    for par in (0, 1):
        idx = hi[par::2]
        if len(idx) >= 4:
            diff = np.diff(idx[:8].astype(int))
            if np.sum(diff == -1) >= 3:
                return True
    return False


def _prefix_vibskydive(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 8:
        return None
    lo = (seg & 0xFF).astype(np.int64)
    hi = ((seg >> 8) & 0xFF).astype(np.int64)
    base = int(seg[0])
    best = None
    lo0 = base & 0xFF
    cand_lo = set()
    for ph0 in range(8):
        for i in range(min(length, 16)):
            if tri_phase(ph0 + i) == 1:
                cand_lo.add((int(lo[i]) - lo0) & 0xFF)
                break
    for amp_lo in cand_lo:
        for amp_hi in range(0, 0x40):
            amp = amp_lo | (amp_hi << 8)
            if amp == 0:
                continue
            for ph0 in range(8):
                vib, _ = render_vibrato_exact(length, base, amp, ph0)
                if not np.array_equal((vib[:8] & 0xFF), lo[:8]):
                    continue
                for par in (0, 1):
                    first = next(
                        (i for i in range(length) if ((ph0 + i) & 1) == par), None
                    )
                    if first is None:
                        continue
                    sfh0 = int(hi[first])
                    rend = render_vibskydive(length, base, amp, ph0, sfh0, par)
                    match = _match_prefix(rend, seg)
                    if match >= 8 and (best is None or match > best[0]):
                        best = (
                            match,
                            "vibskydive",
                            {
                                "base": base,
                                "amp": amp,
                                "ctr0": ph0,
                                "sfh0": sfh0,
                                "par": par,
                            },
                        )
                        if match == length:
                            return best
    return best


def _prefix_arp_decay(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 12:
        return None
    best = None
    for par in (0, 1):
        overlay = [i for i in range(length) if (i & 1) == par]
        if len(overlay) < 4:
            continue
        ov_hi = [(int(seg[i]) >> 8) & 0xFF for i in overlay]
        if not all(ov_hi[k + 1] <= ov_hi[k] for k in range(min(8, len(ov_hi) - 1))):
            continue
        non_overlay = [i for i in range(length) if (i & 1) != par]
        nov = seg[non_overlay]
        for period in range(1, 7):
            full_p = 2 * period
            if len(nov) < 2 * period or length < full_p + 2:
                continue
            cyc_nov = [int(nov[k]) for k in range(period)]
            if not np.array_equal(
                nov[: 2 * period], np.array(cyc_nov * 2, dtype=np.int64)
            ):
                continue
            cyc = [int(seg[j]) for j in range(full_p)]
            first_ov = next(
                (
                    i
                    for i in range(length)
                    if (i & 1) == par and ((int(seg[i]) >> 8) & 0xFF) != 0
                ),
                None,
            )
            if first_ov is None:
                continue
            prior_ov = sum(1 for i in range(first_ov) if (i & 1) == par)
            sfh0 = (int((seg[first_ov] >> 8) & 0xFF) + prior_ov) & 0xFF
            rend = render_arp_decay(length, cyc, full_p, 1, sfh0, par, 0)
            match = _match_prefix(rend, seg)
            if match >= 12 and (best is None or match > best[0]):
                best = (
                    match,
                    "arp_decay",
                    {
                        "freqs": cyc,
                        "period": full_p,
                        "dwell": 1,
                        "sfh0": sfh0,
                        "par": par,
                    },
                )
                if match == length:
                    return best
    return best


def _prefix_arp(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 2:
        return None
    best = None
    for dwell in (1, 2, 3, 4):
        for period in range(2, 7):
            need = period * dwell
            if length < need:
                continue
            cyc = [int(seg[(k * dwell)]) for k in range(period)]
            if len(set(cyc)) < 2:
                continue
            rend = render_arp(length, cyc, period, 0, dwell)
            match = _match_prefix(rend, seg)
            if match >= need and (best is None or match > best[0]):
                best = (
                    match,
                    "arp",
                    {"period": period, "freqs": cyc, "dwell": dwell},
                )
    return best


def _prefix_glide(seg, note_table):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < _MINRUN or note_table is None:
        return None
    nt = np.asarray(note_table, dtype=np.int64)
    hit = np.nonzero(nt == int(seg[0]))[0]
    if len(hit) == 0:
        return None
    best = None
    for n0 in hit.tolist():
        lead = 0
        while lead + 1 < length and seg[lead + 1] == seg[0]:
            lead += 1
        for step in (1, -1, 2, -2):
            for dwell in (1, 2, 3, 4, 6, 8):
                rend = render_glide(length, n0, step, dwell, lead, nt)
                match = _match_prefix(rend, seg)
                if match >= lead + dwell + _MINRUN and (
                    best is None or match > best[0]
                ):
                    best = (
                        match,
                        "glide",
                        {"n0": n0, "step": step, "dwell": dwell, "lead": lead},
                    )
    return best


def _prefix_decay(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 2:
        return None
    base = int(seg[0])
    diffs0 = np.diff(seg)
    nonzero = diffs0[diffs0 != 0]
    if len(nonzero) == 0 or not np.all(nonzero < 0):
        return None
    rate = int(-nonzero[0]) & 0xFFFF
    if not 0 < rate < 0x4000:
        return None
    lead = 0
    while lead + 1 < length and seg[lead + 1] == seg[0]:
        lead += 1
    best = None
    for every in (1, 2, 3, 4):
        for phase in range(every):
            body = render_decay(length - lead, base, rate, every, phase)
            cand = np.concatenate([np.full(lead, base, dtype=np.int64), body])
            match = _match_prefix(cand, seg)
            if match >= max(_MINRUN, lead + 2) and (best is None or match > best[0]):
                best = (
                    match,
                    "decay",
                    {
                        "v0": base,
                        "rate": rate,
                        "every": every,
                        "ctr0": phase,
                        "lead": lead,
                    },
                )
    return best


def _prefix_pingpong(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < _MINRUN:
        return None
    base = int(seg[0])
    diffs = np.diff(seg)
    nonzero = np.abs(diffs[diffs != 0])
    if len(nonzero) == 0:
        return None
    lo_b, hi_b = int(seg.min()), int(seg.max())
    # Two reflection conventions are observed on the bus, and a triangle PWM player
    # may use either: it can reflect ONE PAST the visible extreme (the step lands on
    # ``min-1``/``max+1`` internally and the next step turns back, so the extreme
    # itself is never emitted as a turning point) OR reflect AT the visible extreme
    # (the player clamps to ``min``/``max`` and turns there, emitting the extreme as
    # the apex).  ``render_pingpong`` reflects when the next step crosses the bound,
    # so the first convention needs bound ``min-1``/``max+1`` and the second needs
    # bound ``min``/``max``.  Both are stored as ``lo``/``hi`` adjusted so that
    # :func:`render_fit`'s ``lo-1``/``hi+1`` reproduces the chosen reflect bound, and
    # both are tried; the exact-extreme convention is the one a fine ``+/-1`` triangle
    # vibrato uses, which the past-extreme-only form left as a 7-frame stub.
    bound_pairs = ((lo_b - 1, hi_b + 1), (lo_b, hi_b))
    best = None
    for rate in sorted(set(nonzero.tolist()))[:5]:
        for refl_lo, refl_hi in bound_pairs:
            for dwell in range(0, 12):
                for d0 in range(dwell + 1):
                    for dir0 in (0, 1):
                        rend = render_pingpong(
                            length, base, int(rate), refl_lo, refl_hi, dwell, d0, dir0
                        )
                        match = _match_prefix(rend, seg)
                        if match >= _MINRUN and (best is None or match > best[0]):
                            best = (
                                match,
                                "pingpong",
                                {
                                    "v0": base,
                                    "rate": int(rate),
                                    "lo": refl_lo + 1,
                                    "hi": refl_hi - 1,
                                    "dwell": dwell,
                                    "d0": d0,
                                    "dir0": dir0,
                                },
                            )
                            if match == length:
                                return best
    return best


def _prefix_pingfold(seg, minrun=24):
    """Longest byte-exact mirror-fold fixed-point triangle prefix.

    Recovers ``(step, frac, lo, hi, acc0, dir0)`` for :func:`render_pingfold` from
    the segment's own shape: the internal increment ``step / 2**frac`` is the mean
    magnitude of the lane's nonzero per-frame deltas, so ``step`` is read at each
    candidate ``frac`` as ``round(mean_step * 2**frac)`` (a tiny candidate set, not
    a blind scan), and the fold bounds are the visible extremes shifted into the
    internal precision (``vis_max << frac`` and one above it -- the two reflection
    conventions :func:`_prefix_pingpong` also tries).  A candidate is accepted only
    when it replays a substantial prefix (>= ``minrun``), so a short coincidental
    ramp is left to the cheaper accum / pingpong rules and only a genuine
    long-period triangle -- which the clamping :func:`_prefix_pingpong` cannot
    reproduce once the fractional apex drifts -- is promoted to this one closed-form
    rule rather than fragmenting into raw-byte pieces (HARD RULE #0)."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diffs = np.diff(seg)
    nonzero = np.abs(diffs[diffs != 0]).astype(np.float64)
    if nonzero.size == 0:
        return None
    mean_step = float(np.mean(nonzero))
    vis_lo, vis_hi = int(seg.min()), int(seg.max())
    base = int(seg[0])
    best = None
    for frac in range(0, 4):
        scale = 1 << frac
        step = int(round(mean_step * scale))
        if step <= 0:
            continue
        # Two fold-bound conventions, as in _prefix_pingpong: the player either folds
        # one past the visible extreme or exactly at it.  lo stays 0 (the register
        # floor) plus the at-extreme variant; hi is the visible max raised into the
        # internal precision.
        for hi in (vis_hi << frac, (vis_hi + 1) << frac):
            for lo in (vis_lo << frac, 0):
                if hi <= lo:
                    continue
                # acc0 sub-resolution seed: the visible value occupies the high bits;
                # try each fractional remainder so a ramp starting mid-step matches.
                for sub in range(scale):
                    acc0 = (base << frac) + sub
                    if acc0 > hi or acc0 < lo:
                        continue
                    for dir0 in (1, -1):
                        rend = render_pingfold(length, step, frac, lo, hi, acc0, dir0)
                        match = _match_prefix(rend, seg)
                        if match >= minrun and (best is None or match > best[0]):
                            best = (
                                match,
                                "pingfold",
                                {
                                    "step": step,
                                    "frac": frac,
                                    "lo": lo,
                                    "hi": hi,
                                    "acc0": acc0,
                                    "dir0": dir0,
                                },
                            )
                            if match == length:
                                return best
    return best


def _prefix_additive_pw(seg, carry_seg):
    _ = carry_seg
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 12:
        return None
    p0 = int(seg[0])
    if np.any((seg & ~0xFF) != (p0 & ~0xFF)):
        return None
    lo = (seg & 0xFF).astype(np.int64)
    diff = np.diff(lo) % 256
    vals, cnts = np.unique(diff, return_counts=True)
    pulsevalue = int(vals[np.argmax(cnts)])
    if pulsevalue == 0:
        return None
    cstep = np.zeros(length, dtype=np.int64)
    ok = True
    for i in range(length - 1):
        dval = (int(lo[i + 1]) - int(lo[i])) % 256
        if dval == pulsevalue:
            cstep[i] = 0
        elif dval == (pulsevalue + 1) & 0xFF:
            cstep[i] = 1
        else:
            ok = False
            break
    if not ok or not np.any(cstep):
        return None
    period = _detect_period(cstep[: length - 1], maxp=8)
    if period is None:
        return None
    table = cstep[:period].tolist()
    cseq = np.array([table[i % period] for i in range(length)], dtype=np.int64)
    rend = render_additive_pw(length, p0, pulsevalue, cseq)
    match = _match_prefix(rend, seg)
    if match >= 12:
        return (
            match,
            "additive_pw",
            {
                "p0": p0,
                "pulsevalue": pulsevalue,
                "carry_table": table,
                "carry_phase": 0,
            },
        )
    return None


def _longest_dwell_accum(seg):
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 4:
        return None
    diffs = np.diff(seg)
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 2 or len(set(nonzero.tolist())) != 1:
        return None
    rate = int(nonzero[0])
    lead = 0
    while lead + 1 < length and seg[lead + 1] == seg[0]:
        lead += 1
    change_idx = [i + 1 for i in range(length - 1) if seg[i + 1] != seg[i]]
    gaps = [change_idx[k + 1] - change_idx[k] for k in range(len(change_idx) - 1)]
    if not gaps or len(set(gaps)) != 1:
        return None
    dwell = gaps[0]
    if dwell < 1 or dwell > 8:
        return None
    full = render_dwell_accum(length, int(seg[0]), rate, dwell, lead, 0)
    match = length
    for i in range(length):
        if full[i] != seg[i]:
            match = i
            break
    if match < 4:
        return None
    return (
        match,
        "dwellaccum",
        {"v0": int(seg[0]), "rate": rate, "dwell": dwell, "lead": lead},
    )


def _prefix_maskaccum(seg, width_mask=0xFFFF):
    """Longest byte-exact periodic-dwell accumulator prefix.  Recovers the
    single nonzero rate (must be unique) and the period-P advance mask."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 8:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    nonzero = dsign[dsign != 0]
    if len(nonzero) < 3:
        return None
    vals, _ = np.unique(nonzero, return_counts=True)
    if len(vals) != 1:
        return None
    rate = int(vals[0])
    advance = (dsign != 0).astype(int)
    period = _detect_period(advance, maxp=12)
    if period is None:
        return None
    mask = advance[:period].tolist()
    if not any(mask):
        return None
    rend = render_maskaccum(length, int(seg[0]), rate, mask, width_mask)
    match = _match_prefix(rend, seg)
    if match >= 8:
        return (
            match,
            "maskaccum",
            {"v0": int(seg[0]), "rate": rate, "mask": mask, "width": width_mask},
        )
    return None


def _prefix_maskaccum_stall(seg, width_mask=0xFFFF, maxp=24, minrun=12, mincycles=3):
    """Longest byte-exact single-rate periodic-stall accumulator prefix.  Recovers
    the dominant nonzero rate and the period-P advance mask whose stepped
    accumulator replays the LONGEST prefix.

    The driver-agnostic model of a swept register (pulse-width / freq) whose
    accumulator steps by a fixed amount on the player's continuous-effect frames
    but HOLDS on the periodic tick-0 frames -- the fixed-period 'stall' a
    tempo-paced player imposes on its continuous effects.  Unlike
    :func:`_prefix_maskaccum` (which needs ONE global rate and a short period), this
    is a true longest-prefix matcher that stops where the rate changes (the next
    table step), so the greedy cover chains one piece per table step.  It requires
    >=``mincycles`` full mask cycles, and the caller only lets it win when it covers
    SUBSTANTIALLY more than the proven library's run (see
    :func:`_longest_archetype_aug`), so a coincidental long period cannot shadow a
    genuine accumulator/arp -- an over-eager 'win on any length' form fragments the
    rest of the cover and nets more residual than it closes."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    nonzero = dsign[dsign != 0]
    if len(nonzero) < 3:
        return None
    vals, cnts = np.unique(nonzero, return_counts=True)
    rate = int(vals[np.argmax(cnts)])
    # advance frames are those stepping by exactly the dominant rate; a frame that
    # steps by a different rate (the next table step) is NOT an advance and ends
    # the longest prefix this matcher can cover with one (rate, mask) pair.
    advance = (dsign == rate).astype(int)
    best = None
    for period in range(1, min(maxp, len(advance)) + 1):
        if len(advance) < mincycles * period:
            continue
        mask = advance[:period].tolist()
        steps = sum(mask)
        # require at least one advance AND (for period>1) at least one genuine
        # stall: an all-advance mask is just a plain accum the proven library
        # already covers and must not be re-described here.
        if steps < 1 or (period > 1 and steps == period):
            continue
        rend = render_maskaccum(length, int(seg[0]), rate, mask, width_mask)
        match = _match_prefix(rend, seg)
        if match >= max(minrun, mincycles * period) and (
            best is None or match > best[0]
        ):
            best = (
                match,
                "maskaccum",
                {"v0": int(seg[0]), "rate": rate, "mask": mask, "width": width_mask},
            )
        if best and best[0] == length:
            break
    return best


def _prefix_tablewalk(seg, maxp=48):
    """Longest byte-exact periodic table-walk prefix: smallest period P (2..maxp)
    whose value table replays the segment, requiring >=2 distinct values and
    >=2 full cycles."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 6:
        return None
    best = None
    for period in range(2, min(maxp, length // 2) + 1):
        if np.array_equal(seg[:period], seg[period : 2 * period]):
            table = seg[:period].tolist()
            if len(set(table)) < 2:
                continue
            rend = render_tablewalk(length, table, 0)
            match = _match_prefix(rend, seg)
            if match >= 2 * period and (best is None or match > best[0]):
                best = (match, "tablewalk", {"table": table})
            if best and best[0] == length:
                break
    return best


def _prefix_ratewalk(seg, width_mask=0xFFFF, maxp=48, minrun=8):
    """Longest byte-exact wavetable-rate accumulator prefix.  Recovers the
    period-P signed-rate table from the segment's own deltas: find the smallest
    period whose rate table replays the segment, requiring at least one nonzero
    rate (so a constant hold is left to :func:`render_fit`'s cheaper ``hold``).
    The generalisation of :func:`_prefix_maskaccum` to a per-step rate table that
    closes the fractional-rate / wider-internal-width sweep.

    The period cap admits the longer SID-Wizard PW/filter sweep wavetables (a
    ramp-up / apex-dwell / ramp-down reflecting triangle is a period-~45 signed-rate
    table), but a candidate is accepted only when the matched run covers at least
    TWO full periods -- so the rate table is a genuinely reused loop, never a single
    pass over a long table that would amount to storing the per-step deltas raw."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    best = None
    for period in range(1, min(maxp, max(1, len(dsign))) + 1):
        if len(dsign) < period:
            continue
        table = dsign[:period].tolist()
        if not any(table):
            continue
        rend = render_ratewalk(length, int(seg[0]), table, 0, width_mask)
        match = _match_prefix(rend, seg)
        if match >= max(minrun, 2 * period) and (best is None or match > best[0]):
            best = (
                match,
                "ratewalk",
                {"v0": int(seg[0]), "rate_table": table, "width": width_mask},
            )
        if best and best[0] == length:
            break
    return best


def _prefix_dwellratewalk(seg, width_mask=0xFFFF, maxp=24, maxdwell=16, minrun=24):
    """Longest byte-exact dwelled wavetable-rate accumulator prefix.

    Recovers a (signed-step ``rate_table``, scalar ``dwell``) pair whose accumulator
    replays the longest prefix: the per-frame signed delta is run-length encoded, the
    common run length is taken as ``dwell``, and the per-run step values (capped at
    period ``maxp``) form the table.  This is the table-driven pulse / sweep
    accumulator (the HardTrack-style pulse wavetable: step+dir held ``dwell`` frames
    per entry, looping) whose effective period (``dwell*P``) is far beyond the plain
    :func:`_prefix_ratewalk` period cap.  Folding the dwell column out keeps the
    stored form a SMALL step table plus one scalar, never a ``dwell*P``-entry rate
    table and never per-frame output.  Requires >=2 distinct steps and a genuine
    dwell (>1) so a plain per-frame ratewalk / accum is left to its cheaper rule, and
    a substantial ``minrun`` so a short coincidental ramp is not promoted."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    diff = np.diff(seg) % (width_mask + 1)
    dsign = np.where(diff > width_mask // 2, diff - (width_mask + 1), diff)
    nonzero = dsign[dsign != 0]
    if len(nonzero) < 3:
        return None
    # The dwell is the dominant run length of the signed-delta stream; a table-driven
    # accumulator holds each step a fixed number of frames, so the first few runs are
    # all that length.  Recover it from the leading runs rather than trusting a single
    # value, so a one-frame turning glitch does not mis-set the dwell.
    runs = []
    i = 0
    while i < len(dsign) and len(runs) < maxp + 2:
        j = i
        while j < len(dsign) and dsign[j] == dsign[i]:
            j += 1
        runs.append((int(dsign[i]), j - i))
        i = j
    cand_dwells = sorted(
        {rl for _, rl in runs if 1 < rl <= maxdwell},
        reverse=True,
    )
    best = None
    for dwell in cand_dwells:
        # The step table is one entry per dwell-segment; read it off the segment at
        # the dwell stride from the per-frame deltas.
        nsteps = min(maxp, (len(dsign)) // dwell)
        if nsteps < 2:
            continue
        table = [int(dsign[k * dwell]) for k in range(nsteps)]
        # smallest period whose step table replays the longest prefix
        for period in range(2, nsteps + 1):
            tbl = table[:period]
            if len(set(tbl)) < 2:
                continue
            rend = render_dwellratewalk(length, int(seg[0]), tbl, dwell, 0, width_mask)
            match = _match_prefix(rend, seg)
            if match >= max(minrun, 2 * dwell * period) and (
                best is None or match > best[0]
            ):
                best = (
                    match,
                    "dwellratewalk",
                    {
                        "v0": int(seg[0]),
                        "rate_table": tbl,
                        "dwell": dwell,
                        "width": width_mask,
                    },
                )
            if best and best[0] == length:
                return best
    return best


def _prefix_tablewalk_lead(seg, maxp=24, minrun=8):
    """Longest byte-exact lead-hold-then-table-walk prefix: try absorbing 0..lead
    of the constant prefix into a ``lead`` hold, then the smallest period-P value
    table (>=2 distinct values, >=2 full cycles) that replays the remainder.  This
    admits a long-period delayed modulation the plain table walk misses because a
    short coincidental arp at the note start otherwise shadows it in the cover."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    lead0 = 1
    while lead0 < length and seg[lead0] == seg[0]:
        lead0 += 1
    best = None
    for lead in range(0, min(lead0, length - 1) + 1):
        body = seg[lead:]
        if len(body) < minrun:
            continue
        for period in range(2, min(maxp, len(body) // 2) + 1):
            if not np.array_equal(body[:period], body[period : 2 * period]):
                continue
            table = body[:period].tolist()
            if len(set(table)) < 2:
                continue
            rend = render_tablewalk_lead(length, lead, int(seg[0]), table)
            match = _match_prefix(rend, seg)
            if match >= lead + 2 * period and (best is None or match > best[0]):
                best = (
                    match,
                    "tablewalk_lead",
                    {"lead": lead, "value0": int(seg[0]), "table": table},
                )
        if best and best[0] == length:
            break
    return best


def _walked_table(seg):
    """The pointer-walk decomposition of a value lane: the distinct-value sequence
    visited (one entry per value change) and the per-frame advance clock
    (``seg[i] != seg[i-1]``).  Returns ``(walked, advance)`` where walking
    ``walked`` one step per set ``advance`` bit replays ``seg`` exactly -- this is
    tautological, so the recovery's honesty lives entirely in whether ``walked``
    folds into a SMALL looping table (a genuine generator) vs. arbitrary data."""
    seg = np.asarray(seg, dtype=np.int64)
    advance = (np.diff(seg) != 0).astype(int)
    walked = [int(seg[0])]
    for i in range(1, len(seg)):
        if seg[i] != seg[i - 1]:
            walked.append(int(seg[i]))
    return walked, advance


def _fold_loop(walked, maxp, mincycles):
    """Fold a walked-value sequence into the smallest period-P loop ``table``
    (P in 4..maxp, >=3 distinct values) the sequence steps through cyclically from
    its first entry, requiring at least ``mincycles`` full table cycles of coverage
    (so a coincidental short repeat cannot pass as a reused generator).  The
    pointer phase is 0 by construction (``table[0] == walked[0]``); any leading
    note-onset transient that is NOT on this loop is peeled by the greedy cover as
    a cheaper ``hold`` before this rule fires.  Returns ``table`` or None."""
    length = len(walked)
    for period in range(4, min(maxp, length) + 1):
        if length < period * mincycles:
            continue
        table = walked[:period]
        if len(set(table)) < 3:
            continue
        if all(walked[k] == table[k % period] for k in range(length)):
            return table
    return None


def _fold_loop_prefix(walked, maxp, mincycles):
    """Fold the longest PREFIX of a walked-value sequence into the smallest
    period-P loop ``table`` (P in 4..maxp, >=3 distinct values) it steps through
    cyclically from its first entry, requiring at least ``mincycles`` full cycles.

    This generalises :func:`_fold_loop` (which admits a lane only when ALL of the
    walked values lie on one loop) to the longest-prefix contract every other
    matcher in this module follows: a song whose voice runs one looping arp/note
    wavetable for several pattern rows and then SWITCHES to another (a multi-pattern
    melody is one note-on segment when the gate never retriggers) folds its FIRST
    pattern here, and the greedy cover lays the next pattern down as the next
    wavetable_ptr piece.  Each piece is still a genuine reused generator (the
    period-P table) paced by the shared advance clock -- never the whole melody
    stored byte-for-byte.  Returns ``(table, walked_used)`` for the longest such
    prefix, or None.  Among periods that fold a prefix, the one whose prefix covers
    the most walked entries wins (ties broken by the smaller period)."""
    length = len(walked)
    best = None
    for period in range(4, min(maxp, length) + 1):
        if length < period * mincycles:
            continue
        table = walked[:period]
        if len(set(table)) < 3:
            continue
        used = period
        while used < length and walked[used] == table[used % period]:
            used += 1
        if used < period * mincycles:
            continue
        if best is None or used > best[1]:
            best = (table, used)
    return best


def _prefix_wavetable_ptr(seg, maxp=32, minrun=12, mincycles=2):
    """Longest byte-exact advance-clocked wavetable-pointer prefix.

    The lane is decomposed into a pointer walk (:func:`_walked_table`): a sequence
    of distinct values stepped by the per-frame advance clock.  The value content
    is admitted ONLY when it folds into a small looping table (:func:`_fold_loop`:
    period 4..maxp, >=3 distinct values, >=``mincycles`` full cycles), so the
    closed-form part is a genuine reused generator -- the period-P table -- and the
    only per-frame stream is the advance clock, the separable per-voice groove tick
    the player runs to pace the walk.  Requires the advance clock to contain HOLDS
    (some frames do not step); a step-every-frame walk is a plain ``tablewalk`` and
    is left to that cheaper rule.  This closes the wavetable-paced reflecting
    triangle whose drifting (non-periodic) dwell defeats ``tablewalk`` /
    ``ratewalk`` -- those would store one stride per step (raw data); here the
    table is the generator and the dwell is the shared advance clock."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < minrun:
        return None
    # Fast pre-filter (the wavetable_ptr signature is a SMALL value table looped):
    # a lane whose distinct-value count exceeds the period cap cannot fold into a
    # period<=maxp loop, so skip the O(P*N) fold for sweep/arp lanes outright.
    if len(np.unique(seg)) > maxp:
        return None
    walked, advance = _walked_table(seg)
    if not advance.any() or advance.all():
        return None  # a constant hold, or a step-every-frame plain tablewalk
    # Fold the LONGEST PREFIX of the walk onto one loop (not necessarily the whole
    # segment): a multi-pattern melody held under one un-retriggered note is a chain
    # of looping arp/note-wavetable sections, so cover its first section here and let
    # the greedy cover lay the next section down as the next wavetable_ptr piece --
    # each piece a genuine period-P generator, never the whole melody stored raw.
    fold = _fold_loop_prefix(walked, maxp, mincycles)
    if fold is None:
        return None
    table, _ = fold
    rend = render_wavetable_ptr(length, table, 0, advance)
    match = _match_prefix(rend, seg)
    if match >= minrun:
        # Trim the advance clock to the matched run: a prefix piece only consumes
        # advance[:match], so the remaining bits belong to the next piece's clock.
        return (
            match,
            "wavetable_ptr",
            {"table": table, "phase": 0, "advance": advance[:match].tolist()},
        )
    return None


def _prefix_citg(seg, note_table=None, width_mask=0xFFFF, ctr0=0):
    """The ONE unified matcher: recover a byte-exact longest-prefix CITG cover of a
    lane segment, or None.

    Rather than a parallel reimplementation of every per-archetype recovery, this
    enumerates the small candidate set the design doc's §3b procedure describes --
    one per (MODE, CLOCK-class, table-shape) the unified op admits -- by REUSING the
    proven per-archetype synthesis (the period folders, the advance-clock /
    maskaccum machinery, the dwell recovery) and translating each clean winner to
    its CITG parameterization (:func:`citg_preset`).  Each candidate is re-rendered
    through :func:`render_citg` and accepted only on the LONGEST byte-exact prefix
    (``_match_prefix``), so the matcher is byte-exact-or-None exactly like the zoo:
    trying it first can never regress correctness, and where it declines the zoo
    fallback runs.

    The recovered structures are all genuine reused generators gated by the same
    HARD RULE #0 minima the zoo matchers enforce (>=2 cycles, >=3 distinct values
    for a folded table, a substantial run for a long period): the closed-form
    content is the period-P table, the only per-frame stream is the separable
    advance clock.  The ``vibrato`` / ``pingpong`` parametric shapes and the
    ``vibskydive`` / ``arp_decay`` / ``additive_pw`` composites are NOT single
    CITGs (§2d) and are deliberately left to the zoo fallback -- CITG declines them
    here so they are never faked into a value table."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length < 1:
        return None
    best = None  # (match_len, citg_params)

    def _consider(zoo_name, zoo_prm):
        nonlocal best
        params = citg_preset(zoo_name, zoo_prm)
        if params is None:
            return
        if width_mask is not None and params["mode"] == "accum":
            params["width"] = width_mask
        rend = render_citg(params, length)
        match = _match_prefix(rend, seg)
        if match >= _MINRUN and (best is None or match > best[0]):
            best = (match, params)

    # CLOCK = every-frame --------------------------------------------------
    # ACCUM: a constant-rate accumulator (the cheap ``accum``) and the period-P
    # signed-rate wavetable accumulator (``ratewalk``) -- recovered from the
    # segment's own deltas.
    if length >= _MINRUN:
        delta = int(seg[1]) - int(seg[0])
        if delta != 0:
            j = 0
            while j + 1 < length and int(seg[j + 1]) - int(seg[j]) == delta:
                j += 1
            if j + 1 >= _MINRUN:
                _consider("accum", {"v0": int(seg[0]), "rate": delta})
    rate_walk = _prefix_ratewalk(seg, width_mask)
    if rate_walk is not None:
        _consider(rate_walk[1], rate_walk[2])
    # READ: an undelayed period-P value table (``tablewalk``) and a lead-hold then
    # a period-P table (``tablewalk_lead``).
    table_walk = _prefix_tablewalk(seg)
    if table_walk is not None:
        _consider(table_walk[1], table_walk[2])
    lead_walk = _prefix_tablewalk_lead(seg)
    if lead_walk is not None:
        _consider(lead_walk[1], lead_walk[2])

    # CLOCK = periodic dwell ----------------------------------------------
    # READ: the small-period dwelled value cycle (``arp``).  ACCUM: a single-rate
    # accumulator stepped every ``dwell`` frames after a lead (``dwellaccum``) and
    # the dwelled signed-step wavetable accumulator (``dwellratewalk``).
    arp = _prefix_arp(seg)
    if arp is not None:
        _consider(arp[1], arp[2])
    dwell_accum = _longest_dwell_accum(seg)
    if dwell_accum is not None:
        _consider(dwell_accum[1], dwell_accum[2])
    dwell_walk = _prefix_dwellratewalk(seg, width_mask)
    if dwell_walk is not None:
        _consider(dwell_walk[1], dwell_walk[2])

    # CLOCK = periodic mask -----------------------------------------------
    # ACCUM: a single-rate accumulator gated by a recovered period-Q advance mask,
    # both the short-period (``maskaccum``) and longest-prefix stall forms.
    mask_acc = _prefix_maskaccum(seg, width_mask)
    if mask_acc is not None:
        _consider(mask_acc[1], mask_acc[2])
    stall = _prefix_maskaccum_stall(seg, width_mask)
    if stall is not None:
        _consider(stall[1], stall[2])

    # CLOCK = external advance vector -------------------------------------
    # READ: a pointer over a looping value table paced by the per-voice groove tick
    # (``wavetable_ptr``) -- the general case every other table clock specialises.
    wptr = _prefix_wavetable_ptr(seg)
    if wptr is not None:
        _consider(wptr[1], wptr[2])

    if best is None:
        return None
    return (best[0], "citg", best[1])


# ---------------------------------------------------------------------------
# Greedy cover.
# ---------------------------------------------------------------------------
def _longest_archetype(seg, ctr0, note_table=None, carry_seg=None):
    """Longest archetype run starting at seg[0], byte-exact.  Greedy: prefer the
    longest run; on ties prefer a structured generator over a bare hold (and an
    accumulator over an arp, which can memorise a short accumulator cycle)."""
    seg = np.asarray(seg[:_WINDOW], dtype=np.int64)
    length = len(seg)
    hold = 1
    while hold < length and seg[hold] == seg[0]:
        hold += 1
    cands = [(hold, "hold", {"value": int(seg[0])})]

    if length >= _MINRUN:
        delta = int(seg[1]) - int(seg[0])
        if delta != 0:
            j = 0
            while j + 1 < length and int(seg[j + 1]) - int(seg[j]) == delta:
                j += 1
            if j + 1 >= _MINRUN:
                cands.append(
                    (j + 1, "accum", {"v0": int(seg[0]), "rate": delta, "width": 16})
                )

    dwell_accum = _longest_dwell_accum(seg)
    if dwell_accum is not None:
        cands.append(dwell_accum)

    additive = _prefix_additive_pw(seg, carry_seg)
    if additive is not None:
        cands.append(additive)

    arp = _prefix_arp(seg)
    if arp is not None:
        cands.append(arp)

    if note_table is not None:
        glide = _prefix_glide(seg, note_table)
        if glide is not None:
            cands.append(glide)

    decay = _prefix_decay(seg)
    if decay is not None:
        cands.append(decay)

    wrap = _prefix_wrapaccum(seg)
    if wrap is not None:
        cands.append(wrap)

    if max(c[0] for c in cands) < length:
        for match in _prefix_vibrato(seg):
            cands.append(match)

    if max(c[0] for c in cands) < length:
        pingpong = _prefix_pingpong(seg)
        if pingpong is not None:
            cands.append(pingpong)

    if max(c[0] for c in cands) < length and _has_hi_countdown(seg):
        vibskydive = _prefix_vibskydive(seg)
        if vibskydive is not None:
            cands.append(vibskydive)
        arp_decay = _prefix_arp_decay(seg)
        if arp_decay is not None:
            cands.append(arp_decay)

    def _rank(cand):
        name = cand[1]
        if name == "hold":
            prio = 0
        elif name == "arp":
            prio = 1
        else:
            prio = 2
        return (cand[0], prio)

    cands.sort(key=_rank, reverse=True)
    run_len, name, prm = cands[0]
    return (name, prm, run_len)


def _longest_archetype_aug(seg, ctr0, note_table, carry_seg, width_mask):
    """The cover search's per-run matcher, with the unified CITG tried FIRST and the
    full archetype zoo (:func:`_longest_archetype_zoo`) as the FALLBACK.

    CITG (:func:`_prefix_citg`) is a byte-exact-or-None matcher just like every zoo
    matcher, so trying it first can NEVER regress correctness: at worst it declines
    or covers less and the zoo runs unchanged.  CITG WINS when it covers a run at
    least as long as the zoo's (ties to CITG, the canonical unified form), and the
    zoo wins -- and we record which archetype -- only when it covers strictly more.
    This guarantees coverage parity by construction (the chosen run is always
    ``max(citg, zoo)``), so a whole-tune that was residual-zero stays residual-zero,
    while the per-fit win/fallback tally (:data:`CITG_FALLBACK_COUNTS`) measures
    exactly how much of the zoo CITG already subsumes.

    ``CITG_DISABLE`` short-circuits to the zoo alone (an escape hatch; never needed
    for correctness).  When CITG and the zoo cover the same length, CITG is chosen,
    but a zoo win that the doc flags as a genuine non-CITG case (the ``vibrato`` /
    ``pingpong`` shapes and the ``vibskydive`` / ``arp_decay`` / ``additive_pw``
    composites, §2d) shows up in the fallback tally as the cases still to close."""
    if _CITG_DISABLE:
        return _longest_archetype_zoo(seg, ctr0, note_table, carry_seg, width_mask)
    citg = _prefix_citg(seg, note_table, width_mask, ctr0)
    zoo = _longest_archetype_zoo(seg, ctr0, note_table, carry_seg, width_mask)
    citg_len = citg[0] if citg is not None else -1
    zoo_len = zoo[2] if zoo is not None else -1
    if citg is not None and citg_len >= zoo_len:
        _citg_record("citg_won", citg[2].get("mode", ""))
        return (citg[1], citg[2], citg[0])
    _citg_record(f"fallback:{zoo[0]}" if zoo is not None else "fallback:none")
    return zoo


def _longest_archetype_zoo(seg, ctr0, note_table, carry_seg, width_mask):
    """:func:`_longest_archetype` plus the generic periodic / wavetable generators.

    ``maskaccum`` (a fixed-period-paced accumulator) and ``ratewalk`` (a period-P
    signed-rate wavetable accumulator) are allowed to win on length or to break a
    hold tie -- a structured sweep beats a bare hold.  ``ratewalk`` closes the
    fractional-rate / wider-internal-width sweep (an RMW accumulator stepped by a
    short rate wavetable).  ``tablewalk_lead`` (a lead hold then a period-P value
    table) is allowed to win on length so a DELAYED long-period modulation is
    covered in one piece rather than shadowed by a short coincidental arp prefix.
    ``wavetable_ptr`` (an advance-clocked pointer walk over a looping value table)
    is likewise allowed to win on length: the wavetable-paced reflecting triangle
    whose drifting dwell defeats the periodic generators is a single long piece
    here, where a coincidental pingpong/arp prefix would otherwise shadow it.
    ``maskaccum_stall`` (a single-rate accumulator that HOLDS on a recovered
    periodic tick-0 stall mask -- the tempo-paced player's continuous-effect skip)
    wins only when it covers SUBSTANTIALLY more than the proven library's run, so a
    short coincidental arp/accum prefix cannot fragment a genuine sustained sweep
    while a merely-as-long coincidental period cannot shadow a real generator.
    ``tablewalk`` (an undelayed period-P value table beyond the arp cap) stays a
    LAST RESORT -- it fires only where the proven library returns None, so a
    coincidental short period never shadows a genuine accumulator/arp."""
    base = _longest_archetype(seg, ctr0, note_table, carry_seg)
    for matcher in (_prefix_maskaccum, _prefix_ratewalk):
        cand = matcher(seg, width_mask)
        if cand is not None and (
            base is None
            or cand[0] > base[2]
            or (cand[0] == base[2] and base[0] == "hold")
        ):
            base = (cand[1], cand[2], cand[0])
    # A ratewalk that fully covers the window may be the unrolled dwell*P form of a
    # more compact dwelled rate-wavetable; prefer the dwelled form (a short step table
    # plus one scalar) when it ties.  Checked HERE, before the full-cover early-return
    # below, because that return assumes every later matcher only wins by covering
    # strictly more -- and the dwellratewalk tie-break wins on equal length.  Confined
    # to the full-cover ratewalk case so the cheap many-short-pieces path is unaffected.
    if base is not None and base[0] == "ratewalk" and base[2] >= len(seg):
        dwell_walk = _prefix_dwellratewalk(seg, width_mask)
        if dwell_walk is not None and dwell_walk[0] >= base[2]:
            base = (dwell_walk[1], dwell_walk[2], dwell_walk[0])
    # Every matcher below can only REPLACE ``base`` by covering strictly MORE frames
    # (or fires only when ``base is None``); none can win once the cheap library plus
    # maskaccum/ratewalk already reach the end of this window.  Returning here when
    # the window is fully covered skips their per-piece renders -- behaviour-identical
    # (a full cover is already maximal) but it removes the bulk of the recovery's cost
    # on a long lane the cheap rules tile in many short, fully-covered pieces.
    if base is not None and base[2] >= len(seg):
        return base
    lead_walk = _prefix_tablewalk_lead(seg)
    if lead_walk is not None and (base is None or lead_walk[0] > base[2]):
        base = (lead_walk[1], lead_walk[2], lead_walk[0])
    # A dwelled rate-wavetable accumulator (a pulse/sweep wavetable: signed step held
    # ``dwell`` frames per entry, looping) runs far longer than any periodic generator
    # the proven library reaches, since its effective period (dwell*P) exceeds the
    # ratewalk cap; let it win on length so the table-driven reflecting/ramping PWM is
    # one piece rather than a chain of short accum/pingpong stubs.
    dwell_walk = _prefix_dwellratewalk(seg, width_mask)
    if dwell_walk is not None and (base is None or dwell_walk[0] > base[2]):
        base = (dwell_walk[1], dwell_walk[2], dwell_walk[0])
    wptr = _prefix_wavetable_ptr(seg)
    if wptr is not None and (base is None or wptr[0] > base[2]):
        base = (wptr[1], wptr[2], wptr[0])
    # A mirror-fold fixed-point triangle (a slow per-frame LFO/vibrato bouncing
    # between two bounds with a fractional internal increment) runs far longer than
    # the short accum/pingpong stubs the cheap library grabs at each ramp/apex, since
    # its fractional apex drift defeats the clamping pingpong; let it win on length so
    # the whole triangle collapses to one closed-form (step, frac, bounds) rule
    # rather than a chain of short pieces that would amount to storing the ramp.
    pingfold = _prefix_pingfold(seg)
    if pingfold is not None and (base is None or pingfold[0] > base[2]):
        base = (pingfold[1], pingfold[2], pingfold[0])
    # A periodic-stall accumulator may legitimately run far longer than a short
    # coincidental arp/accum prefix the proven library grabs first; let it win
    # only when it covers SUBSTANTIALLY more (>= twice the base run, and an
    # absolute floor) so a genuine sustained sweep is not fragmented, while a
    # merely-as-long coincidental period cannot shadow a real generator.
    stall = _prefix_maskaccum_stall(seg, width_mask)
    if stall is not None and (
        base is None or (stall[0] > 2 * base[2] and stall[0] >= 36)
    ):
        base = (stall[1], stall[2], stall[0])
    # A period-P value table that replays the WHOLE remaining segment is a genuine
    # reused generator -- a per-note freq arp whose 3-note table is itself looped
    # to form a longer super-period (e.g. Digitalizer's period-48 = 3 notes x 16),
    # or any undelayed wavetable beyond the arp cap.  The cheap local matchers grab
    # only a short prefix of such a loop (one ratewalk/arp cycle) and would
    # fragment the rest into ~one piece per cycle past the cover cap, leaving the
    # lane un-fit.  As with ``tablewalk_lead`` / ``dwellratewalk`` / ``stall``, let
    # ``tablewalk`` win when it covers SUBSTANTIALLY more than the base run (>= twice
    # it, and an absolute floor) so a single looping table collapses to one piece,
    # while a merely-as-long coincidental period can never shadow a real
    # accumulator/arp (which keeps the LAST-RESORT base-is-None path below too).
    tablewalk = _prefix_tablewalk(seg)
    if tablewalk is not None and (
        base is None or (tablewalk[0] > 2 * base[2] and tablewalk[0] >= 36)
    ):
        # base is None keeps tablewalk a LAST RESORT for a lane the proven library
        # cannot start at all; the substantial-length guard lets a genuine looping
        # table win over a short coincidental local prefix.
        return (tablewalk[1], tablewalk[2], tablewalk[0])
    return base


def fit_segment(seg, ctr0, note_table=None, carry_seg=None, width_mask=0xFFFF):
    """Greedily cover one note-on segment with archetype runs, byte-exact.
    Returns ``(name, params)`` for a single-piece cover, ``("piecewise", ...)``
    for a multi-piece cover, or None if some offset is un-fit."""
    seg = np.asarray(seg, dtype=np.int64)
    length = len(seg)
    if length == 0:
        return ("empty", {})
    pieces = []
    i = 0
    while i < length:
        cseg = carry_seg[i:] if carry_seg is not None else None
        run = _longest_archetype_aug(
            seg[i:], (ctr0 + i) & 0xFF, note_table, cseg, width_mask
        )
        if run is None:
            return None
        name, prm, plen = run
        pieces.append((name, prm, plen))
        i += plen
        if len(pieces) > _MAXPIECES:
            return None
    if len(pieces) == 1:
        return (pieces[0][0], pieces[0][1])
    return ("piecewise", {"pieces": pieces})


def fit_lane(lane, noteons, nframes, note_table=None, carry=None, width_mask=0xFFFF):
    """Fit a generator lane as a per-note-on sliced cover.  Returns a list of
    ``(start, stop, fit)`` segments."""
    pts = sorted(noteons) + [nframes]
    out = []
    for i in range(len(pts) - 1):
        start, stop = pts[i], pts[i + 1]
        if stop <= start:
            continue
        cseg = carry[start:stop] if carry is not None else None
        fit = fit_segment(lane[start:stop], start & 0xFF, note_table, cseg, width_mask)
        out.append((start, stop, fit))
    return out


def _interleave_period(lane, max_period=4):
    """The period N of an INTERLEAVED freq lane -- a per-frame software arp/chord
    that round-robins N independent note slots, writing a different slot's note
    each frame (gate held high, no control/ADSR retrigger), so the COMBINED lane
    swings by a note-sized step every single frame while each phase ``lane[ph::N]``
    moves only by its own intra-note vibrato/glide step.

    A genuine single sweep/vibrato has small per-frame steps; an N-slot interleave
    has a large step almost every frame BUT, deinterleaved by the right N, each
    phase is smooth again.  So N is the smallest divisor for which the per-phase
    median step collapses far below the combined-lane median step: a pure
    chip/arithmetic signal (the lane's own step statistics), no per-driver constant.
    Returns the period, or 0 if the lane is not interleaved (one phase already as
    smooth as the whole)."""
    diff = np.abs(np.diff(lane.astype(np.int64)))
    nz = diff[diff != 0]
    if nz.size < 4:
        return 0
    whole_step = int(np.median(nz))
    if whole_step == 0:
        return 0
    for period in range(2, max_period + 1):
        steps = []
        for ph in range(period):
            sub = lane[ph::period].astype(np.int64)
            if len(sub) < 4:
                steps = []
                break
            d = np.abs(np.diff(sub))
            d = d[d != 0]
            steps.append(int(np.median(d)) if d.size else 0)
        if not steps:
            continue
        # Each phase must move by a small fraction of the combined per-frame swing
        # -- i.e. the big every-frame step really is the round-robin, not the music.
        if max(steps) * 4 <= whole_step:
            return period
    return 0


def fit_interleaved_lane(lane, start, stop, note_table, width_mask=0xFFFF):
    """Fit ``lane[start:stop)`` as an N-phase interleaved generator: deinterleave
    into ``lane[start+ph::N]`` phase streams, note-slice and fit each phase with the
    ordinary archetype library (each slot is a normal per-note vibrato/glide/hold
    melody), and return an ``("interleave", {"period": N, "phases": [...]})`` fit
    that renders by re-interleaving the phase covers.

    This closes the per-frame software-arp / two-note-chord freq lane (e.g. David
    Whittaker's octave/chord arps) where the gate stays high so
    :func:`note_boundaries` finds no retrigger and EVERY combined-lane frame is a
    note-sized jump, defeating both the whole-segment cover and
    :func:`freq_note_onsets` (which would then slice at every frame).  The recovered
    form is N closed-form per-phase melodies plus the single interleave period --
    every value a generator, no stored per-frame output (HARD RULE #0).  Returns
    None if the span is not interleaved or any phase is itself irreducible (so a
    genuinely un-fit lane still surfaces rather than being faked)."""
    span = lane[start:stop].astype(np.int64)
    period = _interleave_period(span)
    if period == 0:
        return None
    phases = []
    for ph in range(period):
        sub = span[ph::period]
        if len(sub) == 0:
            phases.append(("empty", {}))
            continue
        # Note-slice the phase by its OWN large freq jumps (the same note-onset rule
        # the freq lane uses), so each slot's per-note vibrato/glide cell is one
        # fittable fixed-centre segment.
        d = np.abs(np.diff(sub))
        nz = d[d != 0]
        thr = max(8, 4 * int(np.median(nz))) if nz.size else 8
        ons = sorted({0, *(np.nonzero(d > thr)[0] + 1).tolist()})
        pts = ons + [len(sub)]
        pieces = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if b <= a:
                continue
            fit = fit_segment(sub[a:b], 0, note_table, None, width_mask)
            if fit is None:
                return None
            pieces.append((fit[0], fit[1], b - a))
        if len(pieces) == 1:
            phases.append((pieces[0][0], pieces[0][1]))
        else:
            phases.append(("piecewise", {"pieces": pieces}))
    return ("interleave", {"period": period, "phases": phases})


def render_interleaved(seg_len, period, phases, note_table):
    """Re-interleave the ``period`` per-phase covers back into one lane of length
    ``seg_len``: phase ``ph`` fills frames ``ph, ph+period, ph+2*period, ...``."""
    out = np.zeros(seg_len, dtype=np.int64)
    for ph in range(period):
        plen = len(range(ph, seg_len, period))
        if plen == 0:
            continue
        out[ph::period] = render_fit(phases[ph], plen, note_table, None)
    return out


def fit_event_lane(col):
    """Cover an 8-bit non-generator register (ctrl/AD/SR/filter/volume) with the
    cheap structured archetypes between change points, byte-exact.  Returns a
    list of ``(start, stop, name, params)``."""
    nframes = len(col)
    col = np.asarray(col, dtype=np.int64)
    segs = []
    i = 0
    while i < nframes:
        run = _longest_event_archetype(col[i:])
        if run is None:
            segs.append((i, i + 1, "hold", {"value": int(col[i])}))
            i += 1
            continue
        plen, name, prm = run
        segs.append((i, i + plen, name, prm))
        i += plen
    return segs


def _longest_event_archetype(seg):
    """Longest byte-exact prefix of a non-generator lane: the cheap structured
    archetypes (hold / accum / dwellaccum / arp), which are far faster than the
    full vibrato/pingpong search and still residual-exact for these never-carry-
    coupled registers, PLUS the advance-clocked ``wavetable_ptr`` (a pointer over
    a small looping value table stepped by an EXTERNAL, possibly non-uniform
    advance clock).

    The ``wavetable_ptr`` matcher is what closes the dwell-paced PAGE-WALK player
    (e.g. Master Composer), where the WHOLE register file is ``page[ptr]`` with
    ``ptr`` advanced by a non-uniform per-step dwell table -- a SINGLE chip-wide
    step clock shared across every lane.  Each non-generator lane is then literally
    ``tablewalk(page_reg, ptr)``: a small per-step value table read through that
    shared pointer.  Without it the groove (a non-uniform dwell with no fixed
    period) defeats ``arp`` / ``dwellaccum`` and the cover fragments into one
    short ``arp`` / ``hold`` piece per dwell run -- effectively storing the lane's
    output piecemeal (a HARD RULE #0 risk).  With it the lane collapses to one
    period-P table plus the separable advance clock (the groove tick, shared
    across all lanes), a genuine reused generator with no per-step data storage.
    It is allowed to win on length so a non-uniform groove that a coincidental
    short ``arp`` prefix would otherwise shadow is covered in a single piece."""
    seg = np.asarray(seg[:_WINDOW], dtype=np.int64)
    length = len(seg)
    hold = 1
    while hold < length and seg[hold] == seg[0]:
        hold += 1
    cands = [(hold, "hold", {"value": int(seg[0])})]
    if length >= _MINRUN:
        delta = int(seg[1]) - int(seg[0])
        if delta != 0:
            j = 0
            while j + 1 < length and int(seg[j + 1]) - int(seg[j]) == delta:
                j += 1
            if j + 1 >= _MINRUN:
                cands.append(
                    (j + 1, "accum", {"v0": int(seg[0]), "rate": delta, "width": 16})
                )
    dwell_accum = _longest_dwell_accum(seg)
    if dwell_accum is not None:
        cands.append(dwell_accum)
    arp = _prefix_arp(seg)
    if arp is not None:
        cands.append(arp)
    wptr = _prefix_wavetable_ptr(seg)
    if wptr is not None:
        cands.append(wptr)

    def _rank(cand):
        name = cand[1]
        prio = 0 if name == "hold" else (1 if name == "arp" else 2)
        return (cand[0], prio)

    cands.sort(key=_rank, reverse=True)
    return cands[0]


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def render_fit(fit, seg_len, note_table=None, carry=None, off=0):
    """Render a fitted ``(name, params)`` back to a lane of length seg_len."""
    name, prm = fit
    if name == "piecewise":
        out = []
        offset = 0
        for pname, pprm, plen in prm["pieces"]:
            out.append(render_fit((pname, pprm), plen, note_table, carry, off + offset))
            offset += plen
        return np.concatenate(out)
    plen = seg_len
    if name in ("hold", "empty"):
        return np.full(plen, prm.get("value", 0), dtype=np.int64)
    if name == "vibrato":
        return render_vibrato(plen, prm["base"], prm["amp_step"], prm["ctr0"])
    if name == "vibrato_exact":
        rend, _ = render_vibrato_exact(plen, prm["base"], prm["amp"], prm["ctr0"])
        return rend
    if name == "vibskydive":
        return render_vibskydive(
            plen, prm["base"], prm["amp"], prm["ctr0"], prm["sfh0"], prm["par"]
        )
    if name == "arp_decay":
        return render_arp_decay(
            plen, prm["freqs"], prm["period"], prm["dwell"], prm["sfh0"], prm["par"], 0
        )
    if name == "arp":
        return render_arp(plen, prm["freqs"], prm["period"], 0, prm.get("dwell", 1))
    if name == "glide":
        return render_glide(
            plen, prm["n0"], prm["step"], prm["dwell"], prm["lead"], note_table
        )
    if name == "accum":
        return render_accum(plen, prm["v0"], prm["rate"], 0xFFFF)
    if name == "wrapaccum":
        return render_wrapaccum(plen, prm["v0"], prm["rate"], prm["lo"], prm["hi"])
    if name == "dwellaccum":
        return render_dwell_accum(
            plen, prm["v0"], prm["rate"], prm["dwell"], prm["lead"], 0
        )
    if name == "decay":
        lead = prm.get("lead", 0)
        body = render_decay(
            plen - lead, prm["v0"], prm["rate"], prm["every"], prm["ctr0"]
        )
        return np.concatenate([np.full(lead, prm["v0"], dtype=np.int64), body])
    if name == "pingpong":
        return render_pingpong(
            plen,
            prm["v0"],
            prm["rate"],
            prm["lo"] - 1,
            prm["hi"] + 1,
            prm["dwell"],
            prm.get("d0", prm["dwell"]),
            prm["dir0"],
        )
    if name == "pingfold":
        return render_pingfold(
            plen,
            prm["step"],
            prm["frac"],
            prm["lo"],
            prm["hi"],
            prm["acc0"],
            prm["dir0"],
        )
    if name == "maskaccum":
        return render_maskaccum(
            plen, prm["v0"], prm["rate"], prm["mask"], prm.get("width", 0xFFFF)
        )
    if name == "tablewalk":
        return render_tablewalk(plen, prm["table"], off)
    if name == "ratewalk":
        return render_ratewalk(
            plen, prm["v0"], prm["rate_table"], 0, prm.get("width", 0xFFFF)
        )
    if name == "tablewalk_lead":
        return render_tablewalk_lead(plen, prm["lead"], prm["value0"], prm["table"])
    if name == "dwellratewalk":
        # phase 0: the matcher anchors v0 at the piece start and the dwell phase
        # resets there, so a piecewise chain renders each piece from its own start.
        return render_dwellratewalk(
            plen,
            prm["v0"],
            prm["rate_table"],
            prm["dwell"],
            0,
            prm.get("width", 0xFFFF),
        )
    if name == "wavetable_ptr":
        return render_wavetable_ptr(plen, prm["table"], prm["phase"], prm["advance"])
    if name == "citg":
        return render_citg(prm, plen)
    if name == "interleave":
        return render_interleaved(plen, prm["period"], prm["phases"], note_table)
    if name == "additive_pw":
        table = prm.get("carry_table")
        if table:
            period = len(table)
            cseq = np.array([table[i % period] for i in range(plen)], dtype=np.int64)
        elif carry is not None:
            cseq = carry[off : off + plen]
        else:
            cseq = np.zeros(plen, dtype=np.int64)
        return render_additive_pw(plen, prm["p0"], prm["pulsevalue"], cseq)
    return np.zeros(plen, dtype=np.int64)


def render_event_lane(segs, nframes):
    """Render a non-generator lane's ``(start, stop, name, params)`` cover."""
    out = np.zeros(nframes, dtype=np.int64)
    for start, stop, name, prm in segs:
        out[start:stop] = render_fit((name, prm), stop - start, None, None)
    return out


def freq_carry_sequence(res, nframes):
    """Reconstruct the per-frame no-CLC carry-out the FREQ generator leaves (the
    carry the additive simple-pw inherits); only vibrato_exact pieces carry."""
    carry = np.zeros(nframes, dtype=np.int64)
    for start, stop, fit in res:
        if fit is None:
            continue
        pieces = (
            fit[1]["pieces"]
            if fit[0] == "piecewise"
            else [(fit[0], fit[1], stop - start)]
        )
        off = start
        for name, prm, plen in pieces:
            if name == "vibrato_exact":
                _, carry_seq = render_vibrato_exact(
                    plen, prm["base"], prm["amp"], prm["ctr0"]
                )
                carry[off : off + plen] = carry_seq[:plen]
            off += plen
    return carry


def archetype_tally(segs):
    """Count archetype occurrences across a list of ``(_, _, fit)`` or
    ``(_, _, name, params)`` segments."""
    tally = defaultdict(int)
    for seg in segs:
        if len(seg) == 4:
            tally[seg[2]] += 1
            continue
        fit = seg[2]
        if fit is None:
            tally["unfit"] += 1
        elif fit[0] == "piecewise":
            for name, _, _ in fit[1]["pieces"]:
                tally[name] += 1
        else:
            tally[fit[0]] += 1
    return dict(tally)
