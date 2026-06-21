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
     glide + the generic maskaccum / tablewalk periodic generators).
  4. :mod:`fitter` covers the 25 registers: the freq/pw generator lanes with the
     BACC library, the ctrl/AD/SR/filter/volume lanes as a compact piecewise
     program, and the note table from bus value-provenance.
  5. :mod:`recover` exposes ``recover_generic`` -> :class:`BaccProgram` plus the
     SELF-CONTAINED ``render_generic`` / ``residual``: the program renders the
     bus-state byte-exact (residual=0) on the proven tunes, independent of any
     hand backend.

Measured result (whole-tune, all 25 registers, byte-exact against the bus-state):

  * **Grid_Runner** (Jammer, GoatTracker) and **Monty_on_the_Run** (Rob Hubbard)
    render RESIDUAL-ZERO from their cached native ``.bus.bin`` traces -- two
    different drivers, one generic recovery, no hand-coded constants.
  * Across the 8-tune corpus the generator lanes reach residual-zero on **5/8**;
    the remaining **3/8** leave residual ONLY on the freq/pw generator lanes
    (the ctrl/AD/SR/filter/volume piecewise program is byte-exact everywhere).

Documented generator-lane gaps (3/8) -- surfaced, never faked:

  * a driver-specific vibrato LFO whose phase table is outside the searched
    triangle / period-P shapes;
  * a freq->pw carry coupling whose additive-PW carry rule is not the recovered
    period-P LFO carry;
  * a free-running PW accumulator whose wrap bound the wrapaccum/maskaccum search
    does not pin down.

The env-gated whole-tune test (``GENERIC_BUSTRACE``) xfails on a tune with a
generator-lane gap so the gap is reported, not papered over.  See the design
note ``design/encoding/generic_recovery_from_sidtrace.md`` in the preframr-xpt
repo for the full per-tune accounting.
"""

from preframr_tokens.bacc.generic.recover import (
    recover_generic,
    render_generic,
    residual,
)

__all__ = ["recover_generic", "render_generic", "residual"]
