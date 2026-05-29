"""SkeletonPass (op54) unit + round-trip tests on synthetic dfs (no HVSC): held
single-value semitone-clean freq notes collapse to one SKEL atom each (abs first, signed
interval after); the encode+decode per-frame register state is byte-exact via the public
expand_ops oracle; skeleton_pass OFF is a no-op."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT, SkeletonPass
from preframr_tokens.stfconstants import (
    FRAME_REG,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
)

_NOTES = [60, 62, 64, 60, 67, 65]
_FRAMES_PER_NOTE = 3
_IRQ = 19656


def _args(**over):
    """Args namespace with skeleton on and the mutually-excluded freq passes off."""
    cfg = dict(skeleton_pass=True, freq_trajectory_pass=False, freq_onset_pass=False)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _row(reg, val, op=SET_OP, subreg=-1, diff=_IRQ, anchor=False):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(_IRQ),
        "description": 0,
        "traj_anchor": bool(anchor),
    }


def _note_stream():
    """Reg-0 held LUT-exact notes: a gated/voiced voice, one FRAME per frame, each note
    re-stated as a held freq SET anchored on its onset frame so notes are byte-exact."""
    rows = [
        _row(5, 0x00),
        _row(6, 0xF0),
        _row(24, 0x0F),
        _row(4, 0x41),
    ]
    for note in _NOTES:
        for f in range(_FRAMES_PER_NOTE):
            rows.append(_row(FRAME_REG, 0, diff=_IRQ))
            rows.append(_row(0, LUT[note], anchor=(f == 0)))
    rows.append(_row(FRAME_REG, 0, diff=_IRQ))
    return pd.DataFrame(rows)


def test_skeleton_emits_one_atom_per_note():
    """One SKEL atom per note, first absolute then signed intervals, low-cardinality."""
    out = SkeletonPass().apply(_note_stream(), args=_args())
    skel = out[out["op"] == SKEL_OP]
    assert len(skel) == len(_NOTES)
    subregs = skel["subreg"].to_numpy()
    assert int(subregs[0]) == SKEL_SUBREG_ABS
    assert all(int(s) == SKEL_SUBREG_INTERVAL for s in subregs[1:])
    intervals = skel["val"].to_numpy()[1:]
    assert len(set(int(v) for v in intervals)) <= len(_NOTES)
    assert not (out["op"] == SET_OP)[out["reg"] == 0].any()


def test_skeleton_round_trip_byte_exact():
    """Encode+decode per-frame register state matches the raw stream byte-exactly."""
    raw = _note_stream()
    enc = SkeletonPass().apply(raw.copy(), args=_args())
    raw_state = register_state(raw.drop(columns=["traj_anchor"]))
    enc_state = register_state(enc)
    assert raw_state.shape == enc_state.shape
    assert np.array_equal(raw_state, enc_state)


def test_skeleton_off_is_noop():
    """skeleton_pass OFF leaves the df unchanged."""
    raw = _note_stream()
    out = SkeletonPass().apply(raw.copy(), args=_args(skeleton_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
