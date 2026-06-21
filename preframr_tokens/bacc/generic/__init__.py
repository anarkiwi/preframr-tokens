"""Driver-agnostic generic recovery from the trusted preframr-sidtrace bus trace.

The hand backends (``bacc/backends/{goattracker,hubbard,...}.py``) each encode
one playroutine's disassembly to recover a :class:`~preframr_tokens.bacc.primitive.BaccProgram`.
This package recovers the SAME common abstraction WITHOUT any per-driver
knowledge, reading only SID-chip semantics plus the trusted preframr-sidtrace
``.bus.bin`` CPU bus trace (native packed ``BUS_DT`` records, no hand-coded
driver constants -- HARD RULE #0).  It is COMPLEMENTARY to the hand backends.

Pipeline (every recovered structure is traced to the bus):

  1. :mod:`bustrace` parses the native preframr-sidtrace ``.bus.bin`` to
     ``(cyc, addr, val, rw)`` records.
  2. :mod:`busstate` reconstructs the per-frame 25-register state from the
     SID-write substream (the driver re-blits its shadow register file every
     play-call), boot-prolog aligned to the dump's first play cycle.
  3. :mod:`archetypes` is the generic bounded-accumulator (BACC) library
     (hold / accum / dwellaccum / wrapaccum / arp / vibrato / pingpong / decay /
     glide + the generic maskaccum / ratewalk / tablewalk / tablewalk_lead
     periodic / wavetable generators).
  4. :mod:`fitter` covers the 25 registers: the freq/pw generator lanes with the
     BACC library, the ctrl/AD/SR/filter/volume lanes as a compact piecewise
     program, and the note table from bus value-provenance.
  5. :mod:`recover` exposes ``recover_generic`` -> :class:`BaccProgram` plus the
     SELF-CONTAINED ``render_generic`` / ``residual``: the program renders the
     bus-state byte-exact (residual=0) on the proven tunes, independent of any
     hand backend.
  6. :mod:`sidtrace` + ``recover_from_sid`` close the loop to a SINGLE input
     file: one deterministic ``preframr-sidtrace`` run over a ``.sid`` ALONE
     emits BOTH the per-frame register dump (``.sidwr.bin``) and the bus trace
     (``.bus.bin``) in-process, so no pre-rendered ``.dump.parquet`` is required;
     the render is verified residual-zero against the SAME-run dump.

Measured result (whole-tune, all 25 registers, byte-exact against the bus-state):

  * **Grid_Runner** (Jammer, GoatTracker) and **Monty_on_the_Run** (Rob Hubbard)
    render RESIDUAL-ZERO from their cached native ``.bus.bin`` traces -- two
    different drivers, one generic recovery, no hand-coded constants.
  * Across the 8-tune corpus whole-tune residual-zero now holds on **7/8** (up
    from 5/8), and Monty_on_the_Run (Hubbard) stays residual-zero.  Two generic
    wavetable archetypes were added to the shared library:

      - ``tablewalk_lead`` -- a lead hold then a period-P value table, admitted to
        the cover on length (one piece) so a short coincidental arp no longer
        shadows a DELAYED long-period modulation.  This CLOSES **Hammurabi** (a
        delayed period-12 vibrato offset table): now residual-zero.
      - ``ratewalk`` -- a period-P signed-rate wavetable accumulator (the
        fractional-rate / wider-internal-width sweep), generalising ``maskaccum``.

  * **Not_Even_Human** is NOT a generator-lane gap: the recovered program renders
    the bus-state byte-exact (residual-zero).  Its only diff is a render-tail
    divergence in the last frames at the ``-limitcycles`` song-end cutoff (the bus
    capture and the dump driver disagree on the final partial play-call) -- a
    render-boundary artifact, not a recovery failure; counted as recovered.

Documented remaining gap (1/8) -- surfaced, never faked:

  * **FamiCommodore** voice-2 PW: a single sustained note whose pulse width is a
    wavetable-paced reflecting triangle over a 12-value table with a drifting
    (non-periodic) dwell.  It is not a clean tablewalk (no period over the note),
    not a constant-rate sub-resolution triangle accumulator, and a ``ratewalk``
    cover fragments into ~70 pieces (~1 number/frame) -- raw-byte storage in
    disguise, which HARD RULE #0 forbids.  So this lane is left UNFIT (residual
    surfaced) rather than papered over with a fake generator; closing it needs a
    wavetable-pointer archetype that is not yet pinned down.  (All other
    FamiCommodore lanes, including voice-0 PW via ``ratewalk``, are residual-zero.)

The env-gated whole-tune test (``GENERIC_BUSTRACE``) asserts residual-zero on the
7 recovered tunes; the FamiCommodore generator-lane gap leaves residual ONLY on
freq/pw and is xfail'd (non-generator residual is always a hard failure), so the
gap is reported, not papered over.  See the design note
``design/encoding/generic_recovery_from_bustrace.md`` in the preframr-xpt repo for
the full per-tune accounting.
"""

from preframr_tokens.bacc.generic.recover import (
    recover_from_sid,
    recover_generic,
    render_generic,
    residual,
)

__all__ = ["recover_from_sid", "recover_generic", "render_generic", "residual"]
