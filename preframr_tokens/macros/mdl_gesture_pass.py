"""MdlGesturePass: the single MDL optimal-parse pass that replaces InstrumentProgramPass + GeneratorPass
(MDL_PARSER_IMPLEMENTATION.md §4), parsing each settled value channel into the driver's own HOLD /
POLY(N) forward-difference / PERIOD primitives and emitting them as the unified ``gesture`` codebook
family; introduced here as a no-op scaffold (Step 1) and grown the scalar-channel parse (Step 2) and the
joint 2-D freq parse (Step 3), so the production pipeline stays byte-exact until it owns the channels.
"""

from __future__ import annotations

__all__ = ["MdlGesturePass"]

from preframr_tokens.macros.passes_base import MacroPass


class MdlGesturePass(MacroPass):
    """Replace the per-gesture greedy passes with one MDL optimal parse over HOLD/POLY/PERIOD gestures
    interned in the corpus-global ``gesture`` codebook; a pass-through scaffold today, the encoder body
    lands in build-order Steps 2-3."""

    GATE_FLAGS: frozenset = frozenset()

    def apply(self, df, args=None):
        return df
