"""MdlGesturePass byte-exact pins (MDL_PARSER_IMPLEMENTATION.md §9): the MDL optimal parse of a value
channel into HOLD/POLY/PERIOD gestures is a lossless cover, so encoding a settled value series to the
gesture codebook family and decoding it back through the production expand_ops path reproduces every
per-frame register byte-for-byte, and the value SETs it owns are consumed into gestures.
"""

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.codebook import gesture_value_series
from preframr_tokens.macros.mdl_gesture_pass import (
    MdlGesturePass,
    _emit_specs,
    _SCALAR_REGS,
    gesture_shape_uses,
)
from preframr_tokens.stfconstants import (
    FRAME_REG,
    GESTURE_REF_OP,
    NOTE_INTERVAL_OP,
    SET_OP,
)


def _reconstruct(series, wrap=False):
    """Re-derive the value series from its MDL gesture runs via the SAME shape split the pass emits and
    the SAME replay the codec decodes -- the encoder/decoder consistency the byte-exact cover rests on.
    """
    out = []
    for _key, kind, degree, cell, anchor, d1, d2, _start, length in _emit_specs(
        np.asarray(series, dtype=np.int64), wrap
    ):
        shape = {"kind": kind, "degree": degree, "cell": list(cell)}
        out.extend(gesture_value_series(shape, anchor, (d1, d2), length))
    return out


def test_core_roundtrip_is_byte_exact_over_random_series():
    """Every shape kind (HOLD constant, POLY ramp/curve, PERIOD cell, plus noise literals) round-trips
    through the gesture split + replay exactly, across a spread of value ranges and lengths.
    """
    rng = np.random.default_rng(7)
    cases = [
        np.full(40, 2048),
        np.array([100 + 5 * f for f in range(50)]),
        np.array([300 + f * f // 3 for f in range(40)]),
        np.array(
            [200 + (7 if f % 3 == 0 else -3 if f % 3 == 1 else 0) for f in range(60)]
        ),
        rng.integers(0, 4096, 80),
        np.array([0]),
        np.array([5, 5, 5, 9, 9, 1, 2, 3, 4, 4, 4, 4]),
    ]
    for s in cases:
        s = s.astype(np.int64)
        assert _reconstruct(s) == [int(x) for x in s], s[:8]


def _setdf(channels, n):
    """A per-frame FRAME + SET-on-change stream for ``channels`` (reg -> per-frame value list)."""
    rows, cur = [], {}
    for f in range(n):
        rows.append(_set_row(FRAME_REG, 0))
        for reg, vals in channels.items():
            v = int(vals[f])
            if cur.get(reg) != v:
                rows.append(_set_row(reg, v))
                cur[reg] = v
    return pd.DataFrame(rows)


def _set_row(reg, val):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": 32,
        "op": int(SET_OP),
        "subreg": -1,
        "irq": 19656,
        "description": 0,
    }


def test_pass_is_byte_exact_and_consumes_value_sets():
    """The pass over a multi-channel stream stays byte-exact (register_state identical) and converts the
    multi-frame value runs into gesture REFs, draining the per-frame value SETs it owns.
    """
    n = 64
    rng = np.random.default_rng(1)
    chans = {
        23: [60] * n,
        2: [128 + 4 * f for f in range(n)],
        9: [400 + f * f // 5 for f in range(n)],
        24: [10 + (5 if f % 2 else 0) for f in range(n)],
        4: list(rng.integers(0, 256, n)),
    }
    df = _setdf(chans, n)
    before = register_state(df.copy())
    out = MdlGesturePass().encode(df.copy())
    after = register_state(out.copy())
    assert before.shape == after.shape
    assert not (before != after).any()
    assert int((out["op"] == GESTURE_REF_OP).sum()) > 0


def test_pass_is_idempotent():
    """A second application is a no-op: once the channels carry gesture ops the pass bails (it never
    re-parses its own output), so the stream is unchanged."""
    n = 32
    chans = {2: [100 + 3 * f for f in range(n)], 23: [42] * n}
    df = _setdf(chans, n)
    once = MdlGesturePass().encode(df.copy())
    twice = MdlGesturePass().encode(once.copy())
    assert once.reset_index(drop=True).equals(twice.reset_index(drop=True))


def test_freq_two_layer_is_byte_exact():
    """The freq channel splits into a note-index layer (NOTE_INTERVAL) + a freq-delta gesture layer and
    decodes back byte-exact: a melody of held grid notes, an arp, and a vibrato around one held note all
    reconstruct via note_freq(note) + delta, with the vibrato kept in the delta layer (note held).
    """
    tuning = 0.0
    notes = [49, 53, 56, 49, 51, 53]
    seq = []
    for k, m in enumerate(notes):
        base = pitch_grid.note_freq_at(m, tuning)
        hold = 8
        if k == 0:
            seq.extend(
                int(base + round(40 * np.sin(2 * np.pi * j / 6))) for j in range(20)
            )
        else:
            seq.extend([base] * hold)
    for m in (49, 52, 55):
        seq.extend([pitch_grid.note_freq_at(m, tuning)] * 2)
    freq = [int(max(0, min(65535, v))) for v in seq]
    n = len(freq)
    df = _setdf({0: freq}, n)
    before = register_state(df.copy())
    out = MdlGesturePass().encode(df.copy())
    after = register_state(out.copy())
    assert before.shape == after.shape
    assert not (before != after).any()
    assert int((out["op"] == NOTE_INTERVAL_OP).sum()) > 0
    assert int((out["op"] == GESTURE_REF_OP).sum()) > 0
    assert int(((out["reg"] == 0) & (out["op"] == SET_OP)).sum()) == 0


def test_gesture_shape_uses_interns_recurring_shapes():
    """gesture_shape_uses (the dictionary builder's input) keys each run by its reusable shape and
    counts reuse: a scalar channel that ramps with the same step twice yields one POLY shape used
    twice, and the freq layer's note-table notes add freq-delta shapes -- the corpus dictionary scope.
    """
    n = 48
    state = np.zeros((n, 25), dtype=np.int64)
    state[:, 23] = [100 + 5 * (f % 12) for f in range(n)]
    state[:, 0] = [pitch_grid.note_freq_at(49 + (f // 12), 0.0) for f in range(n)]
    uses = gesture_shape_uses(state)
    assert uses
    assert max(uses.values()) >= 2
    assert all(isinstance(k, tuple) and k[0] in ("H", "D", "P") for k in uses)


def test_corpus_global_scope_shrinks_the_alphabet():
    """The dictionary-scope recommendation (MDL_PARSER_IMPLEMENTATION.md §2): a shape recurring across
    tunes is ONE corpus-global id but a fresh per-tune id each time, so the corpus-global alphabet is
    strictly smaller than the sum of per-tune alphabets -- the learnability win (frequent shared shapes
    vs under-trained singletons)."""
    tunes = []
    for k in range(3):
        st = np.zeros((40, 25), dtype=np.int64)
        st[:, 23] = [200 + 7 * (f % 10) for f in range(40)]
        st[:, 24] = [10 + k + 2 * f for f in range(40)]
        tunes.append(gesture_shape_uses(st))
    per_tune_sum = sum(len(u) for u in tunes)
    corpus_global = len(set().union(*tunes))
    assert corpus_global < per_tune_sum
    shared = set(tunes[0]) & set(tunes[1]) & set(tunes[2])
    assert shared


def test_frame_markers_preserved():
    """The pass rewrites only value rows: the FRAME markers (the decode frame budget every lossless
    transform must conserve) survive verbatim in count, and the freq reg is owned by the freq two-layer
    parse -- not the scalar channel set."""
    n = 24
    chans = {0: [9000 + 10 * f for f in range(n)], 2: [256] * n}
    df = _setdf(chans, n)
    out = MdlGesturePass().encode(df.copy())
    assert int((out["reg"] == FRAME_REG).sum()) == n
    assert 0 not in set(_SCALAR_REGS)
