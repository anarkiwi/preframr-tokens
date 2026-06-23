"""Driver-agnostic generic recovery from the trusted preframr-sidtrace bus trace.

The hand backend (``bacc/backends/goattracker.py``, kept for reference) encodes
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
     periodic / wavetable generators + the advance-clocked wavetable_ptr).
  4. :mod:`fitter` covers the 25 registers: the freq/pw generator lanes with the
     BACC library, the ctrl/AD/SR/filter/volume lanes as a compact piecewise
     program, and the note table from bus value-provenance.  The generator lanes
     are sliced at :func:`archetypes.note_boundaries` -- every bus-visible note-on
     retrigger (gate rise, control-byte change, or ADSR change), not just a gate
     rise -- so a LEGATO / HARD-RESTART player that advances the melody on an
     internal tempo counter while holding the gate bit high (e.g. Music_Assembler)
     is segmented per note instead of collapsing a whole phrase into one
     over-long, unfittable segment.  The FREQ lane additionally slices at
     :func:`archetypes.pw_sweep_resets` (a per-note pulse-sweep re-seed) so a
     PURE-LEGATO melody with no control/ADSR retrigger at all is still segmented
     per note; the PW lane is deliberately not sliced there, so an irreducible
     reflecting-triangle PW (e.g. one not covered by ``wavetable_ptr``) would stay
     one segment and surface rather than be fragmented into raw-byte pieces.  Every
     signal is per voice, so a genuinely single sustained note (no retrigger) is
     left as one segment and an irreducible lane is still surfaced, never sliced
     into raw-byte pieces (HARD RULE #0).
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
  * Across the 8-tune corpus whole-tune residual-zero now holds on **8/8** (up
    from 5/8), and Monty_on_the_Run (Hubbard) stays residual-zero.  Three generic
    wavetable archetypes were added to the shared library:

      - ``tablewalk_lead`` -- a lead hold then a period-P value table, admitted to
        the cover on length (one piece) so a short coincidental arp no longer
        shadows a DELAYED long-period modulation.  This CLOSES **Hammurabi** (a
        delayed period-12 vibrato offset table): now residual-zero.
      - ``ratewalk`` -- a period-P signed-rate wavetable accumulator (the
        fractional-rate / wider-internal-width sweep), generalising ``maskaccum``.
      - ``wavetable_ptr`` -- an advance-clocked wavetable-pointer walk: a pointer
        over a period-P value table that steps on an EXTERNAL per-voice advance
        clock (the groove/tempo tick) and holds otherwise.  This CLOSES
        **FamiCommodore** (a voice-2 PW reflecting triangle over a 12-level table
        paced by a drifting, non-periodic groove dwell): now residual-zero.  It
        generalises ``maskaccum`` (a single rate gated by a periodic mask) to a
        full value-table walk gated by the voice's bus-recovered advance clock --
        the value content is the compact table (a genuine reused generator), and
        the only per-frame stream is the separable groove tick, SHARED across the
        voice's PW lanes, never the lane's stored output (HARD RULE #0 clean: no
        per-step data storage, the ~900-frame sweep collapses to one 22-value
        table + one advance clock).

  * **Not_Even_Human** is NOT a generator-lane gap: the recovered program renders
    the bus-state byte-exact (residual-zero).  Its only diff is a render-tail
    divergence in the last frames at the ``-limitcycles`` song-end cutoff (the bus
    capture and the dump driver disagree on the final partial play-call) -- a
    render-boundary artifact, not a recovery failure; counted as recovered.

The env-gated whole-tune test (``GENERIC_BUSTRACE``) asserts whole-tune
residual-zero (all 25 registers byte-exact) on every recovered tune, including
FamiCommodore.  Any genuinely irreducible lane would surface as residual rather
than be papered over with a fake generator (a cover that fragments into ~one
number per frame is a HARD RULE #0 violation and is refused, leaving the gap
visible).  See the design note
``design/encoding/generic_recovery_from_bustrace.md`` in the preframr-xpt repo for
the full per-tune accounting.
"""

from preframr_tokens.bacc.generic.recover import (
    recover_from_sid,
    recover_generic,
    render_generic,
    residual,
)
from preframr_tokens.bacc.generic.structure_ir import (
    StructureIR,
    recover_structure_ir,
    render_structure,
    structure_ir_from_ids,
    structure_ir_to_ids,
)

__all__ = [
    "recover_from_sid",
    "recover_generic",
    "render_generic",
    "residual",
    # the structure-recovery path (a structured tune's tracker source -- a deduped
    # instrument pool + factored patterns/orderlist + the porta/vibrato accumulator
    # generators -- serialized < 1 token/frame where the output-fit cover floors >= 1).
    "StructureIR",
    "recover_structure_ir",
    "render_structure",
    "structure_ir_to_ids",
    "structure_ir_from_ids",
]
