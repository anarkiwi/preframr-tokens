"""Permanent CPU-time gate: the cover-path encode stays < 5s per song.

The generic output-fit cover path (``lift -> unlift -> render -> serialize``)
encoded A Mind Is Born in ~32s before the njit + content-cache optimization; it
must not regress past the 5s/song budget. CPU time (``process_time``) is used so
the gate is robust to parallel test load. Numba's kernels compile once
(``cache=True``), so we WARM the JIT with an untimed encode and then time the
steady-state per-song cost -- the gate measures the per-song work, not the
one-time compile.

This complements the token gate (``test_amib_generic_budget``): a recovery that
is byte-exact and under-budget but slow is still a failure of the < 5s target.
"""

import os
import time

import numpy as np
import pytest

from preframr_tokens.bacc.generic.tracker import render_from_fits
from preframr_tokens.bacc.tracker_ir import lift, unlift
from preframr_tokens.bacc.tracker_serialize import _ir_to_ids

_FIX = os.path.join(os.path.dirname(__file__), "test_fixtures")
_MAX_CPU_S = 5.0

# (id, fixture) tuples that ship via the COVER path (no structure recovery); each
# must encode under the per-song CPU budget. Parametrized so the list extends as
# more cover-path tunes are pinned.
_COVER_TUNES = [
    ("A_Mind_Is_Born", "A_Mind_Is_Born.generic_state.npz"),
]


def _encode(state):
    nframes = len(state)
    boot = [int(v) for v in state[0]]
    ir = lift(state, None, nframes, boot)
    genfits, eventfits = unlift(ir)
    rendered = render_from_fits(genfits, eventfits, ir.note_table, nframes)
    return rendered, _ir_to_ids(ir)


@pytest.mark.parametrize("tune_id,fixture", _COVER_TUNES)
def test_cover_path_encode_under_5s_cpu(tune_id, fixture):
    """The cover-path encode runs in < 5s CPU per song (njit warmed)."""
    state = np.load(os.path.join(_FIX, fixture))["state"].astype(np.int64)

    # Warm the numba JIT (one-time compile, cached) and check byte-exactness once.
    rendered, _ = _encode(state)
    assert np.array_equal(rendered, state), f"{tune_id}: cover render is not byte-exact"

    # Steady-state CPU time (min of 3) must be under the per-song budget.
    times = []
    for _ in range(3):
        t = time.process_time()
        _encode(state)
        times.append(time.process_time() - t)
    cpu = min(times)

    assert cpu < _MAX_CPU_S, (
        f"{tune_id} cover-path encode regressed to {cpu:.2f}s CPU "
        f"(>= {_MAX_CPU_S}s). The njit + content-cache cover optimization "
        f"(known-good ~4.2s) has regressed -- a perf gate, never a wall."
    )
