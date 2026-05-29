"""SkeletonPass + ornament channel (op54 SKEL / op55 ORN) unit + round-trip tests on
synthetic dfs (no HVSC). Each note segments (semitone-run + MIN_HOLD UNION gate-on) to one
SKEL atom plus one ORN descriptor that collapses its intra-note arps/vibrato/slide; the
encode+decode per-frame register state is byte-exact via the public expand_ops oracle;
skeleton_pass OFF is a no-op. PLAIN dominates a held-note stream."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import LUT, SkeletonPass, midi_to_fn
from preframr_tokens.stfconstants import (
    FRAME_REG,
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_SLIDE,
    ORN_TYPE_VIB,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
)

_IRQ = 19656
_GATE_REG = 4
_FREQ_REG = 0


def _args(**over):
    """Args namespace with skeleton on and the mutually-excluded freq passes off."""
    cfg = dict(skeleton_pass=True, freq_trajectory_pass=False, freq_onset_pass=False)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _row(reg, val, op=SET_OP, subreg=-1, diff=_IRQ):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(_IRQ),
        "description": 0,
    }


def _fn_cents(note, cents):
    """16-bit freq for a note offset by ``cents`` (the vibrato sub-semitone channel)."""
    return max(
        0,
        min(
            0xFFFF,
            int(
                round(
                    440.0
                    * 2 ** ((note + cents / 100.0 - 69) / 12.0)
                    * 16777216.0
                    / 985248
                )
            ),
        ),
    )


class _StreamBuilder:
    """Build a squeezed per-voice freq+gate stream of NOTES (each a (base, per-frame-freqs)
    sequence with a gate-on rising edge on its first frame), the way the parser feeds the
    pass (consecutive identical freqs squeezed out)."""

    def __init__(self):
        self.rows = [_row(_GATE_REG, 0x40)]
        self._last_fn = None

    def _frame(self):
        self.rows.append(_row(FRAME_REG, 0))

    def _freq(self, fn):
        if fn != self._last_fn:
            self.rows.append(_row(_FREQ_REG, fn))
            self._last_fn = fn

    def _gate(self):
        self.rows.append(_row(_GATE_REG, 0x40))
        self.rows.append(_row(_GATE_REG, 0x41))

    def note(self, per_frame_fns):
        for i, fn in enumerate(per_frame_fns):
            self._frame()
            if i == 0:
                self._gate()
            self._freq(fn)
        return self

    def df(self):
        self._frame()
        return pd.DataFrame(self.rows)


def _held(note, n=4):
    return [LUT[note]] * n


def _arp(base, cycle, reps=2):
    return [LUT[base + o] for o in cycle] * reps


def _octave(base, n=4):
    return [LUT[base + (0 if f % 2 == 0 else 12)] for f in range(n)]


def _slide(base, up, n=4):
    return [LUT[base + min(up, f)] for f in range(n)]


def _orn_types(enc):
    return Counter(
        enc[(enc["op"] == ORN_OP) & (enc["subreg"] == ORN_SUBREG_TYPE)]["val"].tolist()
    )


def _roundtrip_exact(raw):
    enc = SkeletonPass().apply(raw.copy(), args=_args())
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(rs, es)
    return enc


def test_skeleton_emits_one_skel_and_one_orn_per_note():
    """One SKEL + one ORN type atom per note; SKEL is abs first then signed intervals."""
    raw = (
        _StreamBuilder()
        .note(_held(60))
        .note(_held(62))
        .note(_held(64))
        .note(_held(60))
        .df()
    )
    enc = SkeletonPass().apply(raw, args=_args())
    skel = enc[enc["op"] == SKEL_OP]
    assert len(skel) == 4
    orn_type = enc[(enc["op"] == ORN_OP) & (enc["subreg"] == ORN_SUBREG_TYPE)]
    assert len(orn_type) == 4
    subregs = skel["subreg"].to_numpy()
    assert int(subregs[0]) == SKEL_SUBREG_ABS
    assert all(int(s) == SKEL_SUBREG_INTERVAL for s in subregs[1:])
    assert not (enc["op"] == SET_OP)[enc["reg"] == _FREQ_REG].any()


def test_plain_dominates_held_stream():
    """A held-note stream is all-PLAIN ornament (the dominant, cheap case)."""
    raw = (
        _StreamBuilder()
        .note(_held(60))
        .note(_held(64))
        .note(_held(67))
        .note(_held(72))
        .df()
    )
    enc = SkeletonPass().apply(raw, args=_args())
    types = _orn_types(enc)
    assert types[ORN_TYPE_PLAIN] == 4
    assert sum(types.values()) == 4


def test_arp_classified_and_byte_exact():
    """An intra-note offset cycle classifies as ARP and round-trips byte-exactly."""
    raw = _StreamBuilder().note(_held(48)).note(_arp(60, [0, 3, 7], reps=3)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_ARP] == 1
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0


def test_octave_classified_and_byte_exact():
    """A note/note+12 alternation classifies as OCTAVE and round-trips byte-exactly."""
    raw = _StreamBuilder().note(_held(48)).note(_octave(60, 6)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_OCTAVE] == 1


def test_slide_classified_and_byte_exact():
    """A monotone intra-note ramp classifies as SLIDE and round-trips byte-exactly."""
    raw = _StreamBuilder().note(_held(48)).note(_slide(55, 4, 6)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_SLIDE] == 1


def test_vibrato_classified_and_byte_exact():
    """A sub-semitone cents wobble on one semitone classifies as VIB and round-trips
    byte-exactly (raw per-frame freq escape preserves the wobble)."""
    wobble = [_fn_cents(60, c) for c in (0, 25, -25, 20, -20, 25)]
    raw = _StreamBuilder().note(_held(48)).note(wobble).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_VIB] == 1


def test_mixed_stream_byte_exact_and_no_raw_freq():
    """A mixed PLAIN/ARP/OCTAVE/SLIDE stream collapses every freq write to SKEL+ORN
    (zero residual raw op0 freq SET) and round-trips byte-exactly."""
    raw = (
        _StreamBuilder()
        .note(_held(60))
        .note(_arp(64, [0, 3, 7], reps=2))
        .note(_octave(67, 4))
        .note(_slide(55, 3, 4))
        .note(_held(62))
        .df()
    )
    enc = _roundtrip_exact(raw)
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0
    types = _orn_types(enc)
    assert types[ORN_TYPE_PLAIN] == 2
    assert types[ORN_TYPE_ARP] == 1
    assert types[ORN_TYPE_OCTAVE] == 1
    assert types[ORN_TYPE_SLIDE] == 1


def test_resid_escape_byte_exact():
    """An intra-note motion that fits no primitive falls back to a raw per-frame escape
    and still round-trips byte-exactly."""
    chaotic = [midi_to_fn(60 + o) for o in (0, 1, -7, 9, 2, -11, 5, -3)]
    raw = _StreamBuilder().note(_held(48)).note(chaotic).df()
    _roundtrip_exact(raw)


def test_skeleton_off_is_noop():
    """skeleton_pass OFF leaves the df unchanged."""
    raw = _StreamBuilder().note(_held(60)).note(_held(62)).df()
    out = SkeletonPass().apply(raw.copy(), args=_args(skeleton_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
