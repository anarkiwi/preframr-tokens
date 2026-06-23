"""Permanent anti-"irreducible-wall" gate (HARD RULE #0).

A Mind Is Born (Linus Akesson / lft) is 256 bytes of arbitrary 6502 machine code
with NO note table, patterns, or instruments -- the adversarial "this looks
irreducible" case. It is not irreducible: its register output is a handful of
generators (period-32 accumulators, a dwell-128 filter ramp, a repeating drum
lane). The GENERIC driver MUST encode it byte-exact at < 1 token/frame.

If this test fails you have NOT hit a wall -- you have stopped recovering the
generators (output-fitting, a literal-table floor, or "store the trace"). Recover
the program; never declare a SID trace irreducible. The token budget is the gate,
co-equal with residual-0: byte-exact-but-dense is a failure.

The fixture is a committed per-frame register state (so this runs in CI with no
``preframr-sidtrace`` binary); it is the real trace of the tune, reproduced from
the .sid with the binary when regenerated.
"""

import os

import numpy as np

from preframr_tokens.bacc.generic.tracker import render_from_fits
from preframr_tokens.bacc.tracker_ir import lift, unlift
from preframr_tokens.bacc.tracker_serialize import _ir_to_ids

_STATE = os.path.join(
    os.path.dirname(__file__), "test_fixtures", "A_Mind_Is_Born.generic_state.npz"
)


def test_a_mind_is_born_generic_under_one_token_per_frame():
    """A Mind Is Born encodes byte-exact at < 1 token/frame via the generic driver."""
    state = np.load(_STATE)["state"].astype(np.int64)
    nframes = len(state)
    boot = [int(v) for v in state[0]]

    ir = lift(state, None, nframes, boot)

    # Residual-0: the recovered generators reproduce the trace byte-for-byte.
    genfits, eventfits = unlift(ir)
    rendered = render_from_fits(genfits, eventfits, ir.note_table, nframes)
    assert np.array_equal(rendered, state), "generic render is not byte-exact"

    # Token budget: < 1 token/frame. 256 bytes of machine code cannot be a wall.
    tok_per_frame = len(_ir_to_ids(ir)) / nframes
    assert tok_per_frame < 1.0, (
        f"A_Mind_Is_Born generic encoding regressed to {tok_per_frame:.3f} "
        f"token/frame (>= 1.0). This is NOT an irreducible wall -- 256 bytes of "
        f"machine code generate this output. Recover the generators; do not store "
        f"the trace. (known-good ~0.41)"
    )
