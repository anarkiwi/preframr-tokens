"""SkeletonPass + ornament channel (op54 SKEL / op55 ORN) unit + round-trip tests on
synthetic dfs (no HVSC). Each note -> one SKEL atom plus one driver-native constant-size ORN
descriptor (PLAIN / OCTAVE+ARP period+length / SLIDE target+rate / VIB depth+rate / RESID
raw-offset escape); encode+decode per-frame freq matches the content-tier semitone floor;
skeleton_pass OFF is a no-op. PLAIN dominates a held-note stream."""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.macros.skeleton_pass import (
    LUT,
    SkeletonPass,
    fn_to_note_resid,
    midi_to_fn,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    FRAME_REG,
    ORN_OP,
    ORN_SUBREG_P1,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_VIB,
    SET_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    SKEL_SUBREG_INTERVAL,
)

_FREQ_REGS = set(int(r) for r in FREQ_REGS_BY_VOICE)


def _snap_freq_floor(state):
    """Snap each freq-reg column to its nearest LUT semitone (the content-tier floor the ORN
    channel reconstructs to); other registers pass through unchanged."""
    snapped = state.copy()
    for reg in _FREQ_REGS:
        col = snapped[:, reg]
        for i, fn in enumerate(col):
            res = fn_to_note_resid(int(fn))
            if res is not None:
                col[i] = LUT[max(0, min(127, res[0]))]
    return snapped


_IRQ = 19656
_GATE_REG = 4
_FREQ_REG = 0


def _args(**over):
    """Args namespace with skeleton on and the mutually-excluded freq passes off."""
    cfg = dict(skeleton_pass=True, freq_trajectory_pass=False)
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
    """Encode then decode and assert the per-frame register state matches the content-tier
    semitone floor (freq regs snapped to LUT; all other regs byte-exact)."""
    enc = SkeletonPass().apply(raw.copy(), args=_args())
    rs = register_state(raw)
    es = register_state(enc)
    assert rs.shape == es.shape, (rs.shape, es.shape)
    assert np.array_equal(_snap_freq_floor(rs), es)
    return enc


def _orn_runs(enc):
    """Each note's ORN descriptor as (type, atom_count), a run from a TYPE atom to the next."""
    runs = []
    for _, r in enc[enc["op"] == ORN_OP].iterrows():
        if int(r["subreg"]) == ORN_SUBREG_TYPE:
            runs.append([int(r["val"]), 1])
        elif runs:
            runs[-1][1] += 1
    return runs


def _orn_atoms_for_type(enc, orn_type):
    """Atom count of the (single) note whose ORN descriptor has type ``orn_type``."""
    sized = [n for t, n in _orn_runs(enc) if t == orn_type]
    assert len(sized) == 1, sized
    return sized[0]


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


def test_arp_classified_parametric_and_floor_exact():
    """An intra-note offset cycle classifies as ARP, round-trips at the floor, and encodes as
    a constant-size descriptor (TYPE + the 3-offset period + length), not one atom per frame.
    """
    raw = _StreamBuilder().note(_held(48)).note(_arp(60, [0, 3, 7], reps=4)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_ARP] == 1
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0
    assert _orn_atoms_for_type(enc, ORN_TYPE_ARP) == 5


def test_octave_classified_parametric_and_floor_exact():
    """A note/note+12 alternation classifies as OCTAVE and encodes as a constant-size
    descriptor (TYPE + 2-offset period + length)."""
    raw = _StreamBuilder().note(_held(48)).note(_octave(60, 8)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_OCTAVE] == 1
    assert _orn_atoms_for_type(enc, ORN_TYPE_OCTAVE) == 4


def test_slide_classified_parametric_and_floor_exact():
    """A monotone intra-note ramp classifies as SLIDE and encodes as TYPE + target + rate +
    length (4 atoms), independent of the ramp length."""
    raw = _StreamBuilder().note(_held(48)).note(_slide(55, 4, 8)).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_SLIDE] == 1
    assert _orn_atoms_for_type(enc, ORN_TYPE_SLIDE) == 4


def test_vibrato_classified_parametric_and_floor_exact():
    """A sub-semitone cents wobble on one semitone classifies as VIB and encodes as TYPE +
    depth + rate + length (4 atoms); at the content-tier floor it reconstructs to the held
    semitone (the wobble is below the floor), not one raw freq per frame."""
    wobble = [_fn_cents(60, c) for c in (0, 25, -25, 20, -20, 25)]
    raw = _StreamBuilder().note(_held(48)).note(wobble).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_VIB] == 1
    assert _orn_atoms_for_type(enc, ORN_TYPE_VIB) == 4


def test_arp_constant_size_independent_of_duration():
    """The ARP descriptor size depends only on the period, not the note duration: a long
    repeat encodes in the same atom count as a short one."""
    short = _StreamBuilder().note(_held(48)).note(_arp(60, [0, 4, 7], reps=3)).df()
    long = _StreamBuilder().note(_held(48)).note(_arp(60, [0, 4, 7], reps=12)).df()
    es = _orn_atoms_for_type(SkeletonPass().apply(short, args=_args()), ORN_TYPE_ARP)
    el = _orn_atoms_for_type(SkeletonPass().apply(long, args=_args()), ORN_TYPE_ARP)
    assert es == el


def test_mixed_stream_floor_exact_and_no_raw_freq():
    """A mixed PLAIN/ARP/OCTAVE/SLIDE stream collapses every freq write to SKEL+ORN
    (zero residual raw op0 freq SET) and round-trips at the floor."""
    raw = (
        _StreamBuilder()
        .note(_held(60))
        .note(_arp(64, [0, 3, 7], reps=3))
        .note(_octave(67, 6))
        .note(_slide(55, 3, 5))
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


def test_resid_escape_offset_floor_exact():
    """An intra-note motion that fits no primitive falls back to a raw signed-offset-per-frame
    escape (semitone floor) and round-trips at the floor."""
    chaotic = [midi_to_fn(60 + o) for o in (0, 1, -7, 9, 2, -11, 5, -3)]
    raw = _StreamBuilder().note(_held(48)).note(chaotic).df()
    enc = _roundtrip_exact(raw)
    assert _orn_types(enc)[ORN_TYPE_RESID] == 1
    resid = enc[(enc["op"] == ORN_OP) & (enc["subreg"] == ORN_SUBREG_P1)]
    assert len(resid) > 0


class _HeldGateBuilder:
    """Build a single-gate (legato) phrase: gate on once, then per-frame freqs with NO further
    gate retrigger -- a held-gate phrase whose internal note changes have no attack (Hubbard
    note-flag bit6). Used to exercise held-gate re-segmentation."""

    def __init__(self):
        self.rows = [_row(_GATE_REG, 0x40), _row(_GATE_REG, 0x41)]
        self._last_fn = None

    def frames(self, per_frame_fns):
        for fn in per_frame_fns:
            self.rows.append(_row(FRAME_REG, 0))
            if fn != self._last_fn:
                self.rows.append(_row(_FREQ_REG, fn))
                self._last_fn = fn
        return self

    def df(self):
        self.rows.append(_row(FRAME_REG, 0))
        return pd.DataFrame(self.rows)


def _slide_between(a, b, steps):
    """Per-frame freqs sliding from semitone a to b over ``steps`` connecting frames."""
    return [LUT[a + (1 if b > a else -1) * s] for s in range(1, steps + 1)]


def _skel_notes(enc):
    """Decode the absolute semitone of each SKEL atom (abs first, signed intervals after)."""
    notes = []
    cur = 0
    for _, r in enc[enc["op"] == SKEL_OP].iterrows():
        if int(r["subreg"]) == SKEL_SUBREG_ABS:
            cur = int(r["val"])
        else:
            v = int(r["val"])
            cur += v if v < 128 else v - 256
        notes.append(cur)
    return notes


def test_held_gate_phrase_splits_into_notes():
    """A held-gate phrase A ... B ... C (one gate-on; three stable >= MIN_HOLD semitone plateaus,
    no re-gate) de-merges into its constituent notes -- A, B and C each appear as a SKEL note --
    instead of collapsing to ONE note with a giant RESID, and the phrase round-trips at the
    semitone floor (the connecting motion becomes ornament, not raw freq)."""
    a, b, c = 60, 64, 67
    per_frame = (
        _held(a)
        + _slide_between(a, b, 2)
        + _held(b)
        + _slide_between(b, c, 2)
        + _held(c)
    )
    raw = _HeldGateBuilder().frames(per_frame).df()
    enc = _roundtrip_exact(raw)
    notes = _skel_notes(enc)
    assert len(notes) >= 3, notes
    assert {a, b, c} <= set(notes), notes
    assert int(((enc["op"] == SET_OP) & (enc["reg"] == _FREQ_REG)).sum()) == 0
    assert not _orn_types(enc)[
        ORN_TYPE_RESID
    ], "held-gate phrase must not leave a RESID"


def test_fast_arp_under_held_gate_stays_ornament():
    """A held plateau followed by a fast (< MIN_HOLD) arp under one held gate de-merges into the
    plateau note + ONE arp note -- the arp steps are NOT each turned into a note (the < MIN_HOLD
    fast-step ornament guard is kept) -- and the arp becomes a parametric ARP ornament.
    """
    base = 60
    per_frame = _held(base) + _arp(base, [0, 4, 7], reps=6)
    raw = _HeldGateBuilder().frames(per_frame).df()
    enc = _roundtrip_exact(raw)
    skel = enc[enc["op"] == SKEL_OP]
    assert len(skel) <= 3, (len(skel), "arp steps must NOT each become a note")
    assert _orn_types(enc)[ORN_TYPE_ARP] >= 1
    assert not _orn_types(enc)[ORN_TYPE_RESID]


def test_skeleton_off_is_noop():
    """skeleton_pass OFF leaves the df unchanged."""
    raw = _StreamBuilder().note(_held(60)).note(_held(62)).df()
    out = SkeletonPass().apply(raw.copy(), args=_args(skeleton_pass=False))
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), raw.reset_index(drop=True)
    )
