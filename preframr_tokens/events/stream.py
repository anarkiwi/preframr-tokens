"""v3 canonical event codec: the oracle is :func:`canonical_writes` -- CTRL/AD/SR change
activity as ordered typed events at sub-frame resolution (gate-on = NOTE_ON with duration;
gate-off ALWAYS derived, no NOTE OFF token), with freq/PW/globals emitted from SETTLED
end-of-frame state (intra-frame transients and same-value rewrites are canonicalized away --
licensed by raw-vs-canonical renders at the reSID noise floor). Scope: single-speed non-digi.
"""

from __future__ import annotations

import collections

import numpy as np

from preframr_tokens.macros import pitch_grid
from . import varint
from .gestures import Shape, cover
from .oracle import NUM_REGS, OrderedWrites, ordered_writes, settled_grid
from .schema import GLOBAL, ad_reg, ctrl_reg, freq_regs, pw_regs, sr_reg

_VSPAN = 32
VAR_BASE = 0
REG_BASE = VAR_BASE + _VSPAN
VOICE_BASE = REG_BASE + NUM_REGS
TUNING = VOICE_BASE + 4
NOTE_TABLE = TUNING + 1
TICK = NOTE_TABLE + 1
NI_STEP = TICK + 1
NI_RAMP = NI_STEP + 1
FD_STEP = NI_RAMP + 1
FD_RAMP = FD_STEP + 1
PW_STEP = FD_RAMP + 1
PW_RAMP = PW_STEP + 1
FLD_NOTE_ON = PW_RAMP + 1
FLD_CTRL = FLD_NOTE_ON + 1
FLD_AD = FLD_CTRL + 1
FLD_SR = FLD_AD + 1
G_STEP = FLD_SR + 1
G_RAMP = G_STEP + 1
SHAPE_POLY = G_RAMP + 1
SHAPE_PERIOD = SHAPE_POLY + 1
NIB_WAVE = SHAPE_PERIOD + 1
NIB_ART = NIB_WAVE + 16
NIB_ENV = NIB_ART + 16
KEYFRAME = NIB_ENV + 16
VOCAB_SIZE = KEYFRAME + 1
EVENT_FORMAT_VERSION = 2

_HEADER_KINDS = (TUNING, NOTE_TABLE, TICK)
_EVENT_KINDS = frozenset(range(NI_STEP, G_RAMP + 1))
_DEFAULT_TUNING_Q = pitch_grid.tuning_to_q(0.0)

_GO_NONE, _GO_DERIVE, _GO_VALUE = 0, 1, 2

GLOBAL_REGS = (21, 22, 23, 24)

_RANK_NI, _RANK_FD, _RANK_PW, _RANK_CAS = 0, 1, 2, 3
_FLAG_OAD, _FLAG_OSR, _FLAG_HRAD, _FLAG_HRSR = 1, 2, 4, 8
_FLAG_OAD_PRE, _FLAG_OSR_PRE = 16, 32


def _is_digit(tok: int) -> bool:
    return VAR_BASE <= tok < VAR_BASE + _VSPAN


def is_content_atom(tok: int) -> bool:
    """Whether atom ``tok`` is musical *content* vs structural scaffolding (loss tiers): varint value
    digits and the typed value nibbles (waveform/articulation, AD/SR envelope, PW duty-class) carry
    the payload the model must predict -- intervals, durations, timbre bits -- so they are content;
    every other atom (reg ids, voice tags, kind/shape markers, KEYFRAME) is structural.
    """
    return _is_digit(tok) or NIB_WAVE <= tok < KEYFRAME


def _emit_ctrl_val(out: list[int], val: int) -> None:
    out.append(NIB_WAVE + ((val >> 4) & 0xF))
    out.append(NIB_ART + (val & 0xF))


def _emit_env_val(out: list[int], val: int) -> None:
    out.append(NIB_ENV + ((val >> 4) & 0xF))
    out.append(NIB_ENV + (val & 0xF))


def _read_nib(tokens, pos: int, base: int) -> tuple[int, int]:
    tok = tokens[pos]
    if not base <= tok < base + 16:
        raise ValueError(f"expected nibble token at {pos}")
    return tok - base, pos + 1


def _read_ctrl_val(tokens, pos: int) -> tuple[int, int]:
    hi, pos = _read_nib(tokens, pos, NIB_WAVE)
    lo, pos = _read_nib(tokens, pos, NIB_ART)
    return (hi << 4) | lo, pos


def _read_env_val(tokens, pos: int) -> tuple[int, int]:
    hi, pos = _read_nib(tokens, pos, NIB_ENV)
    lo, pos = _read_nib(tokens, pos, NIB_ENV)
    return (hi << 4) | lo, pos


def _is_voice(tok: int) -> bool:
    return VOICE_BASE <= tok < VOICE_BASE + 4


def _emit_u(out: list[int], value: int) -> None:
    for d in varint.encode_unsigned(int(value)):
        out.append(VAR_BASE + d)


def _emit_s(out: list[int], value: int) -> None:
    for d in varint.encode_signed(int(value)):
        out.append(VAR_BASE + d)


def _read_var(tokens, pos: int) -> tuple[int, int]:
    shifted = []
    while True:
        if pos >= len(tokens) or not _is_digit(tokens[pos]):
            raise ValueError(f"expected varint digit at {pos}")
        shifted.append(tokens[pos] - VAR_BASE)
        pos += 1
        if not (shifted[-1] & varint.CONT):
            break
    return varint.decode_unsigned(shifted, 0)[0], pos


def _read_u(tokens, pos):
    return _read_var(tokens, pos)


def _read_s(tokens, pos):
    u, pos = _read_var(tokens, pos)
    return varint.unzigzag(u), pos


def _ulen(v: int) -> int:
    return len(varint.encode_unsigned(int(v)))


def _slen(v: int) -> int:
    return len(varint.encode_signed(int(v)))


def _iround(x: float) -> int:
    """Deterministic round-half-up (x >= 0)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def single_speed(ow: OrderedWrites) -> bool:
    """Multi-speed test: a tune is multi-speed when the median per-frame max same-reg repeat
    count is >= 2 (the player runs N times per IRQ frame)."""
    reps = []
    for writes in ow.by_frame():
        if not writes:
            continue
        c = collections.Counter(r for r, _ in writes)
        reps.append(max(c.values()))
    return not reps or float(np.median(reps)) < 2


class _SeriesCost:
    """Exact emitted-token cost of a candidate segment under this serialization. ``interval`` channels
    (note index) encode levels relative to the previous frame's value; a HOLD equal to the held value is
    suppressed entirely (cost 0). ``head`` is the per-event overhead ([VOICE][KIND], +1 for the global
    lane's reg token)."""

    def __init__(
        self, signed: bool, interval: bool = False, head: int = 2, pw: bool = False
    ):
        self.signed = signed
        self.interval = interval
        self.head = head
        self.pw = pw

    def _lvl(self, s, i) -> int:
        v = int(s[i])
        if self.pw:
            return 1 + _ulen(v & 0xFF)
        if self.interval:
            prev = int(s[i - 1]) if i > 0 else 0
            return _slen(v - prev)
        return _slen(v) if self.signed else _ulen(v)

    def hold(self, s, i, j) -> int:
        prev = int(s[i - 1]) if i > 0 else 0
        if int(s[i]) == prev:
            return 0
        return self.head + self._lvl(s, i)

    def poly(self, s, i, j, N, dt) -> int:
        return (
            self.head
            + 1
            + _ulen(j - i)
            + _ulen(N)
            + self._lvl(s, i)
            + sum(_slen(int(d)) for d in dt[1:])
        )

    def period(self, s, i, j, cell) -> int:
        return (
            self.head
            + 1
            + _ulen(j - i)
            + _ulen(len(cell))
            + self._lvl(s, i)
            + sum(_slen(int(c)) for c in cell)
        )


def _emit_level(out, value, cur, signed, interval, pw):
    if pw:
        out.append(NIB_ENV + ((value >> 8) & 0xF))
        _emit_u(out, value & 0xFF)
    elif interval:
        _emit_s(out, value - cur)
    elif signed:
        _emit_s(out, value)
    else:
        _emit_u(out, value)


def _ramp_tokens(kind, g, cur, signed, interval, reg_tok=None, pw=False):
    out = [kind]
    if reg_tok is not None:
        out.append(reg_tok)
    out.append(SHAPE_POLY if g.shape == Shape.POLY else SHAPE_PERIOD)
    _emit_u(out, g.length)
    _emit_u(out, len(g.params) - 1)
    _emit_level(out, int(g.params[0]), cur, signed, interval, pw)
    for d in g.params[1:]:
        _emit_s(out, d)
    return out


def _step_tokens(kind, value, cur, signed, interval, reg_tok=None, pw=False):
    out = [kind]
    if reg_tok is not None:
        out.append(reg_tok)
    _emit_level(out, value, cur, signed, interval, pw)
    return out


def _series_events(
    series,
    voice,
    rank,
    step_kind,
    ramp_kind,
    signed,
    interval=False,
    reg=None,
    pw=False,
):
    """A settled series -> [(frame, sort_key, body_tokens)] value events (kind-led bodies; the VOICE
    token is the frame group lead). ``sort_key`` = (voice, rank, sub). ``pw`` levels split as a typed
    duty-class nibble + fine byte (12-bit values; the coarse nibble is the timbre-relevant part).
    """
    head = 2 if reg is None else 3
    cm = _SeriesCost(signed=signed, interval=interval, head=head, pw=pw)
    reg_tok = None if reg is None else REG_BASE + reg
    evs = []
    cur = 0
    for g in cover(series, cost_model=cm):
        if g.shape == Shape.HOLD:
            v = int(g.params[0])
            if v != cur:
                evs.append(
                    (
                        g.start,
                        (voice, rank, reg or 0),
                        _step_tokens(step_kind, v, cur, signed, interval, reg_tok, pw),
                    )
                )
            cur = v
        else:
            evs.append(
                (
                    g.start,
                    (voice, rank, reg or 0),
                    _ramp_tokens(ramp_kind, g, cur, signed, interval, reg_tok, pw),
                )
            )
            cur = int(series[g.start + g.length - 1])
    return evs


def _freq16(settled, v: int) -> np.ndarray:
    lo, hi = freq_regs(v)
    return (settled[:, hi].astype(np.int64) << 8) | settled[:, lo].astype(np.int64)


def _note_base(ni: np.ndarray, tuning: float, devs: dict) -> np.ndarray:
    base = pitch_grid.note_freq(ni, tuning).copy()
    for note, dev in devs.items():
        base[ni == note] = pitch_grid.note_freq_at(note, tuning) + dev
    return base


_TUNING_WRAP_DEADBAND = 0.43


def _canonical_tuning(t: float) -> float:
    """Snap a recovered tuning inside the half-semitone wrap deadband to exactly -0.5, so content near
    the wrap (e.g. measured camerock q in {13..16, 254}) gets ONE canonical note-index grid instead of
    flipping spelling by a semitone between encodes; the residual lane absorbs the offset exactly.
    """
    return -0.5 if abs(t) >= _TUNING_WRAP_DEADBAND else t


def _freq_layer(settled, v: int):
    """Voice freq -> (header token-lists, ni series, delta series); shared reconstruction contract:
    ``freq = note_freq(ni, tuning) + dev[ni] + delta`` (nonzero deviations only).
    """
    F = _freq16(settled, v)
    q = pitch_grid.tuning_to_q(_canonical_tuning(pitch_grid.voice_tuning(F)))
    tuning = pitch_grid.q_to_tuning(q)
    table = pitch_grid.recover_table(F, tuning)
    devs = {}
    for note in sorted(table):
        dev = table[note] - pitch_grid.note_freq_at(note, tuning)
        if dev:
            devs[note] = dev
    ni = pitch_grid.note_index(F, tuning)
    delta = F - _note_base(ni, tuning, devs)
    headers = []
    voice_tok = VOICE_BASE + v
    if q != _DEFAULT_TUNING_Q:
        h = [voice_tok, TUNING]
        _emit_u(h, q)
        headers.append(h)
    if devs:
        h = [voice_tok, NOTE_TABLE]
        _emit_u(h, len(devs))
        prev = 0
        for note in sorted(devs):
            _emit_s(h, note - prev)
            _emit_s(h, devs[note])
            prev = note
        headers.append(h)
    return headers, ni, delta


def _cas_changes(ow: OrderedWrites) -> dict[int, list[tuple[int, int, int]]]:
    """Per-voice ordered CTRL/AD/SR *change* sequences ``[(frame, reg, val)]`` in driver order
    (sub-frame resolution). Same-value rewrites are chip no-ops (latch semantics, verified) and are
    canonicalized away."""
    seqs: dict[int, list[tuple[int, int, int]]] = {0: [], 1: [], 2: []}
    cur = [0] * NUM_REGS
    for f, reg, val in zip(ow.frame.tolist(), ow.reg.tolist(), ow.val.tolist()):
        if reg <= 20 and reg % 7 >= 4:
            if cur[reg] != val:
                seqs[reg // 7].append((f, reg, val))
            cur[reg] = val
    return seqs


def _typed_cas(seq):
    """Type a voice's cas change sequence (gate 0->1 CTRL changes become ``FLD_NOTE_ON``) and pair
    each note to its gate-off: ``durinfo[i] = (mode, dur, off_val, off_idx)`` plus the off indices --
    every gate 1->0 pairs structurally, since gate only reaches 1 via a NOTE_ON."""
    raw = []
    gate = 0
    for f, reg, val in seq:
        off = reg % 7
        if off == 4:
            ng = val & 1
            tok = FLD_NOTE_ON if (ng and not gate) else FLD_CTRL
            raw.append((f, tok, val))
            gate = ng
        elif off == 5:
            raw.append((f, FLD_AD, val))
        else:
            raw.append((f, FLD_SR, val))
    durinfo: dict = {}
    remove: set = set()
    for i, (f, tok, val) in enumerate(raw):
        if tok != FLD_NOTE_ON:
            continue
        prev_c = val
        goff = None
        for j in range(i + 1, len(raw)):
            _fj, tj, vj = raw[j]
            if tj == FLD_NOTE_ON:
                break
            if tj == FLD_CTRL:
                if not (vj & 1):
                    goff = j
                    break
                prev_c = vj
        if goff is None:
            durinfo[i] = (_GO_NONE, 0, 0, None)
        else:
            fj, _tj, vj = raw[goff]
            mode = _GO_DERIVE if vj == (prev_c & ~1) else _GO_VALUE
            durinfo[i] = (mode, fj - f, vj, goff)
            remove.add(goff)
    return raw, durinfo, remove


def _recover_tick(durs) -> tuple[int, int]:
    """Per-voice ``(tick, offset)`` for durations: the unit in [2,32] whose exact residual grid
    (offset constrained to {-1,0,+1}) covers >=90% of gate-on durations, ranked by mass then smallest
    ``|offset|`` then largest tick; >=4 durations required else raw tick=1. The exactness + offset
    constraints keep recovery stable (±1/unconstrained criteria measured degenerate); the off-grid
    tail stays exactly encodable through ``r``."""
    if len(durs) < 4:
        return 1, 0
    best = None
    for t in range(2, 33):
        res = [d - _iround(d / t) * t for d in durs]
        cnt = collections.Counter(res)
        off = max((0, -1, 1), key=lambda r, c=cnt: (c.get(r, 0), -abs(r)))
        mass = cnt.get(off, 0)
        if mass >= 0.9 * len(durs):
            cand = (mass, -abs(off), t, off)
            if best is None or cand > best:
                best = cand
    if best is None:
        return 1, 0
    return best[2], best[3]


def _fold_envelope(raw, durinfo, remove):
    """Claim each NOTE_ON's envelope lifecycle from the change sequence: the onset-frame AD/SR (the
    instrument) and the gate-OFF hard-restart prep AD/SR at onset-k (timing chip-inert there, value
    essential for the attack -- measured). Returns ``(folds, claimed)``: ``folds[i] = (flags, oad, osr,
    hr_off, hr_ad, hr_sr)`` per NOTE_ON index; claimed indices stop being standalone events.
    """
    gates = []
    gate = 0
    for _f, tok, val in raw:
        if tok in (FLD_NOTE_ON, FLD_CTRL):
            gate = val & 1
        gates.append(gate)
    by_frame: dict = collections.defaultdict(dict)
    for i, (f, tok, _val) in enumerate(raw):
        if tok in (FLD_AD, FLD_SR) and i not in remove:
            by_frame[f][tok] = i
    folds: dict = {}
    claimed: set = set()
    for i, (f, tok, val) in enumerate(raw):
        if tok != FLD_NOTE_ON:
            continue
        flags = 0
        oad = osr = hr_ad = hr_sr = 0
        hr_off = 0
        slot = by_frame.get(f, {})
        j = slot.get(FLD_AD)
        if j is not None and j not in claimed:
            flags |= _FLAG_OAD | (_FLAG_OAD_PRE if j < i else 0)
            oad = raw[j][2]
            claimed.add(j)
        j = slot.get(FLD_SR)
        if j is not None and j not in claimed:
            flags |= _FLAG_OSR | (_FLAG_OSR_PRE if j < i else 0)
            osr = raw[j][2]
            claimed.add(j)
        for k in range(1, 9):
            slot = by_frame.get(f - k, {})
            cand = [
                (tok2, j2)
                for tok2, j2 in slot.items()
                if j2 not in claimed and gates[j2] == 0
            ]
            if not cand:
                continue
            hr_off = k
            for tok2, j2 in cand:
                if tok2 == FLD_AD:
                    flags |= _FLAG_HRAD
                    hr_ad = raw[j2][2]
                else:
                    flags |= _FLAG_HRSR
                    hr_sr = raw[j2][2]
                claimed.add(j2)
            break
        folds[i] = (flags, oad, osr, hr_off, hr_ad, hr_sr)
    return folds, claimed


def _note_layer(seq, v: int):
    """One voice's cas change sequence -> (header token-lists, [(frame, sort_key, tokens)] events).
    Gate-offs are removed and derived from NOTE_ON durations -- always (no NOTE OFF, no fallback);
    each off's DERIVE/VALUE mode is decided at its canonical slot (:func:`_assign_gate_off_modes`).
    """
    raw, durinfo, remove = _typed_cas(seq)
    modes = _assign_gate_off_modes(raw, durinfo, remove)
    folds, claimed = _fold_envelope(raw, durinfo, remove)
    durs = [d for (m, d, _c, _g) in durinfo.values() if m != _GO_NONE]
    tick, offset = _recover_tick(durs)
    voice_tok = VOICE_BASE + v
    headers = []
    if tick > 1:
        h = [voice_tok, TICK]
        _emit_u(h, tick)
        _emit_s(h, offset)
        headers.append(h)
    evs = []
    sub = 0
    for i, (f, tok, val) in enumerate(raw):
        if i in remove or i in claimed:
            continue
        body = [tok]
        if tok in (FLD_NOTE_ON, FLD_CTRL):
            _emit_ctrl_val(body, val)
        else:
            _emit_env_val(body, val)
        if tok == FLD_NOTE_ON:
            flags, oad, osr, hr_off, hr_ad, hr_sr = folds[i]
            _emit_u(body, flags)
            if flags & _FLAG_OAD:
                _emit_env_val(body, oad)
            if flags & _FLAG_OSR:
                _emit_env_val(body, osr)
            if flags & (_FLAG_HRAD | _FLAG_HRSR):
                _emit_u(body, hr_off)
                if flags & _FLAG_HRAD:
                    _emit_env_val(body, hr_ad)
                if flags & _FLAG_HRSR:
                    _emit_env_val(body, hr_sr)
            d = durinfo[i][1]
            mode, c_off = modes.get(i, (_GO_NONE, 0))
            _emit_u(body, mode)
            if mode != _GO_NONE:
                if tick > 1:
                    q = _iround(d / tick)
                    _emit_u(body, q)
                    _emit_s(body, d - q * tick - offset)
                else:
                    _emit_u(body, d)
                if mode == _GO_VALUE:
                    _emit_ctrl_val(body, c_off)
        evs.append((f, (v, _RANK_CAS, sub), body))
        sub += 1
    return headers, evs


def _slot_gate_offs(entries, offs):
    """Insert derived gate-offs (``("OFF", payload)``) into one voice+frame's ordered cas ``entries``:
    an inherited off (onset in an earlier frame) goes before the frame's first NOTE_ON's pre-side
    OAD/OSR group (retrigger keeps gate semantics) else at the end; the i-th same-frame off goes after
    the i-th NOTE_ON's post-side group -- glued onset envelope never crosses a gate edge it did not
    cross in the dump (preframr-audio liveness matrix / gate reference)."""
    if not offs:
        return list(entries)
    inherited = [o for o in offs if o[0]]
    same = [o for o in offs if not o[0]]
    out = []
    si = 0
    inh = list(inherited)
    entries = list(entries)
    k = 0
    while k < len(entries):
        e = entries[k]
        if e[0] == FLD_NOTE_ON and inh:
            at = len(out)
            while at > 0 and out[at - 1][0] in ("OAD", "OSR"):
                at -= 1
            out.insert(at, ("OFF", inh.pop(0)[1]))
        out.append(e)
        if e[0] == FLD_NOTE_ON:
            while k + 1 < len(entries) and entries[k + 1][0] in ("OAD", "OSR"):
                k += 1
                out.append(entries[k])
            if si < len(same):
                out.append(("OFF", same[si][1]))
                si += 1
        k += 1
    for o in inh:
        out.append(("OFF", o[1]))
    return out


def _insert_hr(merged, hr):
    """Place a frame's HR prep writes on the gate=0 side of the gate edge: after the frame's last
    derived gate-off when one is present (the dump wrote them with the gate already off; an AD/SR
    compare change crossing the off edge flips ADSR-bug stall states -- preframr-audio
    ``test_gate_adsr_reference``), else leading the frame (the gate is 0 throughout)."""
    if not hr:
        return merged
    idx = 0
    for j, e in enumerate(merged):
        if e[0] == "OFF":
            idx = j + 1
    return merged[:idx] + list(hr) + merged[idx:]


def _voice_assembly(raw, durinfo, remove, folds=None, claimed=None):
    """The canonical per-frame cas assembly of one voice: ``{frame: [("OFF", note_idx) | ("OAD"/"OSR"/
    "HRAD"/"HRSR", val) | (tok, val)]}`` -- kept entries in driver order, folded onset envelope on its
    RECORDED side of the gate edge (``_FLAG_O*_PRE``, AD before SR within a side: crossings flip
    ADSR-bug stalls, driver conventions split), HR prep on the gate=0 side of the prep frame's off
    (:func:`_insert_hr`), offs per :func:`_slot_gate_offs`; shared by all three."""
    claimed = claimed or set()
    kept_by_f: dict = collections.defaultdict(list)
    hr_by_f: dict = collections.defaultdict(list)
    offs_by_f: dict = collections.defaultdict(list)
    for i, (f, tok, val) in enumerate(raw):
        if i in remove or i in claimed:
            continue
        if folds and tok == FLD_NOTE_ON and i in folds:
            flags, oad, osr, hr_off, hr_ad, hr_sr = folds[i]
            if flags & _FLAG_OAD and flags & _FLAG_OAD_PRE:
                kept_by_f[f].append(("OAD", oad))
            if flags & _FLAG_OSR and flags & _FLAG_OSR_PRE:
                kept_by_f[f].append(("OSR", osr))
            kept_by_f[f].append((tok, val))
            if flags & _FLAG_OAD and not flags & _FLAG_OAD_PRE:
                kept_by_f[f].append(("OAD", oad))
            if flags & _FLAG_OSR and not flags & _FLAG_OSR_PRE:
                kept_by_f[f].append(("OSR", osr))
            if flags & _FLAG_HRAD:
                hr_by_f[f - hr_off].append(("HRAD", hr_ad))
            if flags & _FLAG_HRSR:
                hr_by_f[f - hr_off].append(("HRSR", hr_sr))
        else:
            kept_by_f[f].append((tok, val))
    for i, (mode, _d, _c, goff) in durinfo.items():
        if mode == _GO_NONE:
            continue
        onset_f = raw[i][0]
        off_f = raw[goff][0]
        offs_by_f[off_f].append((off_f > onset_f, i, onset_f))
    out = {}
    for f in sorted(set(kept_by_f) | set(hr_by_f) | set(offs_by_f)):
        offs = sorted(offs_by_f.get(f, ()), key=lambda o: (not o[0], o[2]))
        merged = _slot_gate_offs(kept_by_f.get(f, []), [(o[0], o[1]) for o in offs])
        out[f] = _insert_hr(merged, hr_by_f.get(f, []))
    return out


def _assign_gate_off_modes(raw, durinfo, remove):
    """Walk the voice's canonical assembly, deciding each gate-off's mode at its canonical slot:
    DERIVE when the dump's off byte equals (ctrl state at the slot) & ~1, else VALUE carrying the byte.
    Returns ``{note_idx: (mode, off_val)}``."""
    modes = {}
    state = 0
    asm = _voice_assembly(raw, durinfo, remove)
    for f in sorted(asm):
        for e in asm[f]:
            if e[0] == "OFF":
                i = e[1]
                off_val = raw[durinfo[i][3]][2]
                mode = _GO_DERIVE if off_val == (state & ~1) else _GO_VALUE
                modes[i] = (mode, off_val)
                state = off_val
            elif e[0] in (FLD_NOTE_ON, FLD_CTRL):
                state = e[1]
    return modes


def canonical_writes(ow: OrderedWrites) -> list[tuple[int, int, int]]:
    """The fidelity target: the dump's audibly-faithful canonical form. Per frame: voices
    ascending, each as [settled-changed freq lo,hi][settled-changed pw lo,hi][cas write sequence
    in driver order, gate-offs derived, onset envelope on its recorded gate-edge side, HR prep on
    the gate=0 side of the off]; then settled-changed globals reg-ascending. CAS order is
    preserved exactly; freq/PW/global intra-frame transients settle to end-of-frame."""
    n = ow.n_frames
    if n == 0:
        return []
    settled = settled_grid(ow)
    seqs = _cas_changes(ow)
    asm = {}
    raws = {}
    durinfos = {}
    for v in range(3):
        raw, durinfo, remove = _typed_cas(seqs[v])
        folds, claimed = _fold_envelope(raw, durinfo, remove)
        asm[v] = _voice_assembly(raw, durinfo, remove, folds, claimed)
        raws[v] = raw
        durinfos[v] = durinfo
    out: list[tuple[int, int, int]] = []
    prev = np.zeros(NUM_REGS, dtype=np.int64)
    for f in range(n):
        row = settled[f]
        for v in range(3):
            lo, hi = freq_regs(v)
            plo, phi = pw_regs(v)
            for r in (lo, hi, plo, phi):
                if row[r] != prev[r]:
                    out.append((f, int(r), int(row[r])))
            cr = ctrl_reg(v)
            for e in asm[v].get(f, ()):
                if e[0] == "OFF":
                    out.append((f, cr, int(raws[v][durinfos[v][e[1]][3]][2])))
                else:
                    tok, val = e
                    if tok in (FLD_NOTE_ON, FLD_CTRL):
                        reg = cr
                    elif tok in (FLD_AD, "OAD", "HRAD"):
                        reg = ad_reg(v)
                    else:
                        reg = sr_reg(v)
                    out.append((f, reg, int(val)))
        for r in GLOBAL_REGS:
            if row[r] != prev[r]:
                out.append((f, r, int(row[r])))
        prev = row
    return out


def encode(ow: OrderedWrites, verify: bool = True) -> list[int]:
    """Ordered write stream -> v3 canonical token stream. Verifies
    ``decode(out) == canonical_writes(ow)`` by default (fail loudly)."""
    n = ow.n_frames
    out: list[int] = []
    _emit_u(out, n)
    if n == 0:
        return out

    settled = settled_grid(ow)
    written = set(int(r) for r in ow.reg.tolist())
    seqs = _cas_changes(ow)

    headers: list[list[int]] = []
    events: list[tuple[int, tuple, list[int]]] = []

    for v in range(3):
        lo, hi = freq_regs(v)
        if (lo in written or hi in written) and (
            settled[:, lo].any() or settled[:, hi].any()
        ):
            hs, ni, delta = _freq_layer(settled, v)
            headers += hs
            events += _series_events(
                ni, v, _RANK_NI, NI_STEP, NI_RAMP, signed=True, interval=True
            )
            events += _series_events(delta, v, _RANK_FD, FD_STEP, FD_RAMP, signed=True)
    for v in range(3):
        if seqs[v]:
            hs, evs = _note_layer(seqs[v], v)
            headers += hs
            events += evs
    for v in range(3):
        lo, hi = pw_regs(v)
        if (lo in written or hi in written) and (
            settled[:, lo].any() or settled[:, hi].any()
        ):
            combined = (settled[:, hi].astype(np.int64) << 8) | settled[:, lo].astype(
                np.int64
            )
            events += _series_events(
                combined, v, _RANK_PW, PW_STEP, PW_RAMP, signed=False, pw=True
            )
    for reg in GLOBAL_REGS:
        if reg in written and settled[:, reg].any():
            events += _series_events(
                settled[:, reg], GLOBAL, reg, G_STEP, G_RAMP, signed=False, reg=reg
            )
    for h in headers:
        out.extend(h)

    by_f: dict[int, list] = collections.defaultdict(list)
    for f, key, toks in events:
        by_f[f].append((key, toks))
    prev_f = 0
    for f in sorted(by_f):
        _emit_u(out, f - prev_f)
        prev_f = f
        cur_v = None
        for key, toks in sorted(by_f[f], key=lambda kt: kt[0]):
            if key[0] != cur_v:
                out.append(VOICE_BASE + key[0])
                cur_v = key[0]
            out.extend(toks)

    if verify:
        got = decode(out)
        want = canonical_writes(ow)
        if got != want:
            k = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                min(len(got), len(want)),
            )
            raise AssertionError(
                f"v3 roundtrip diverged at canonical write {k}: "
                f"got {got[k] if k < len(got) else None} "
                f"want {want[k] if k < len(want) else None}"
            )
    return out


class _Chan:
    """One value channel: current value + an active POLY/PERIOD ramp, replayed on demand. Queries and
    events arrive in nondecreasing frame order."""

    __slots__ = ("f", "v", "_shape", "_end", "_st")

    def __init__(self):
        self.f = 0
        self.v = 0
        self._shape = None
        self._end = 0
        self._st = None

    def at(self, f: int) -> int:
        while self.f < f:
            self.f += 1
            if self._shape is None:
                continue
            if self.f >= self._end:
                self._shape = None
                continue
            if self._shape == SHAPE_POLY:
                st = self._st
                for k in range(len(st) - 1):
                    st[k] += st[k + 1]
                self.v = st[0]
            else:
                cur, cell, k = self._st
                cur += cell[k % len(cell)]
                self._st = [cur, cell, k + 1]
                self.v = cur
        return self.v

    def set(self, f: int, v: int) -> None:
        self.at(f)
        self._shape = None
        self.v = int(v)

    def ramp(self, f: int, shape: int, length: int, params) -> None:
        self.at(f)
        self.v = int(params[0])
        self._shape = shape
        self._end = f + length
        if shape == SHAPE_POLY:
            self._st = [int(x) for x in params]
        else:
            self._st = [int(params[0]), [int(x) for x in params[1:]], 0]


class _Decoder:
    def __init__(self, tokens):
        self.t = list(tokens)
        self.pos = 0
        self.ni = [_Chan() for _ in range(3)]
        self.fd = [_Chan() for _ in range(3)]
        self.pw = [_Chan() for _ in range(3)]
        self.g = {reg: _Chan() for reg in GLOBAL_REGS}
        self.tuning = [pitch_grid.q_to_tuning(_DEFAULT_TUNING_Q)] * 3
        self.q = [_DEFAULT_TUNING_Q] * 3
        self.devs = [dict(), dict(), dict()]
        self._base = [dict(), dict(), dict()]
        self.tick = [(1, 0)] * 3
        self.freq_active = [False] * 3
        self.pw_active = [False] * 3
        self.cas_active = [False] * 3
        self.g_active = {reg: False for reg in GLOBAL_REGS}
        self.chan_ops = collections.defaultdict(list)
        self.cas = collections.defaultdict(list)
        self.offs = collections.defaultdict(list)
        self.hr = collections.defaultdict(list)
        self.unit_starts: list[int] = []

    def _u(self):
        v, self.pos = _read_u(self.t, self.pos)
        return v

    def _s(self):
        v, self.pos = _read_s(self.t, self.pos)
        return v

    def _parse_ramp(self, chan, f, signed, interval, pw=False):
        shape = self.t[self.pos]
        if shape not in (SHAPE_POLY, SHAPE_PERIOD):
            raise ValueError(f"expected shape token at {self.pos}")
        self.pos += 1
        length = self._u()
        deg = self._u()
        if interval:
            rel = self._s()
            rest = [self._s() for _ in range(deg)]
            self.chan_ops[f].append((chan, "ramp_rel", (shape, length, rel, rest)))
        else:
            if pw:
                hi, self.pos = _read_nib(self.t, self.pos, NIB_ENV)
                p0 = (hi << 8) | self._u()
            else:
                p0 = self._s() if signed else self._u()
            rest = [self._s() for _ in range(deg)]
            self.chan_ops[f].append((chan, "ramp", (shape, length, [p0, *rest])))

    def _parse_event(self, f: int, voice: int) -> None:
        kind = self.t[self.pos]
        self.pos += 1
        if kind == NI_STEP:
            self.freq_active[voice] = True
            self.chan_ops[f].append((self.ni[voice], "set_rel", self._s()))
        elif kind == NI_RAMP:
            self.freq_active[voice] = True
            self._parse_ramp(self.ni[voice], f, True, True)
        elif kind == FD_STEP:
            self.freq_active[voice] = True
            self.chan_ops[f].append((self.fd[voice], "set", self._s()))
        elif kind == FD_RAMP:
            self.freq_active[voice] = True
            self._parse_ramp(self.fd[voice], f, True, False)
        elif kind == PW_STEP:
            self.pw_active[voice] = True
            hi, self.pos = _read_nib(self.t, self.pos, NIB_ENV)
            self.chan_ops[f].append((self.pw[voice], "set", (hi << 8) | self._u()))
        elif kind == PW_RAMP:
            self.pw_active[voice] = True
            self._parse_ramp(self.pw[voice], f, False, False, pw=True)
        elif kind == G_STEP:
            reg = self.t[self.pos] - REG_BASE
            self.pos += 1
            self.g_active[reg] = True
            self.chan_ops[f].append((self.g[reg], "set", self._u()))
        elif kind == G_RAMP:
            reg = self.t[self.pos] - REG_BASE
            self.pos += 1
            self.g_active[reg] = True
            self._parse_ramp(self.g[reg], f, False, False)
        elif kind in (FLD_NOTE_ON, FLD_CTRL, FLD_AD, FLD_SR):
            if kind in (FLD_NOTE_ON, FLD_CTRL):
                val, self.pos = _read_ctrl_val(self.t, self.pos)
            else:
                val, self.pos = _read_env_val(self.t, self.pos)
            self.cas_active[voice] = True
            post = []
            if kind == FLD_NOTE_ON:
                flags = self._u()
                if flags & _FLAG_OAD:
                    oad, self.pos = _read_env_val(self.t, self.pos)
                    if flags & _FLAG_OAD_PRE:
                        self.cas[(voice, f)].append(("OAD", oad))
                    else:
                        post.append(("OAD", oad))
                if flags & _FLAG_OSR:
                    osr, self.pos = _read_env_val(self.t, self.pos)
                    if flags & _FLAG_OSR_PRE:
                        self.cas[(voice, f)].append(("OSR", osr))
                    else:
                        post.append(("OSR", osr))
                if flags & (_FLAG_HRAD | _FLAG_HRSR):
                    hr_off = self._u()
                    if flags & _FLAG_HRAD:
                        hr_ad, self.pos = _read_env_val(self.t, self.pos)
                        self.hr[(voice, f - hr_off)].append(("HRAD", hr_ad))
                    if flags & _FLAG_HRSR:
                        hr_sr, self.pos = _read_env_val(self.t, self.pos)
                        self.hr[(voice, f - hr_off)].append(("HRSR", hr_sr))
            self.cas[(voice, f)].append((kind, val))
            if kind == FLD_NOTE_ON:
                self.cas[(voice, f)].extend(post)
                mode = self._u()
                if mode != _GO_NONE:
                    tick, offset = self.tick[voice]
                    if tick > 1:
                        q = self._u()
                        r = self._s()
                        d = q * tick + r + offset
                    else:
                        d = self._u()
                    go_val = None
                    if mode == _GO_VALUE:
                        go_val, self.pos = _read_ctrl_val(self.t, self.pos)
                    self.offs[(voice, f + d)].append((d > 0, mode, go_val, f))
        else:
            raise ValueError(f"unknown event kind {kind} at {self.pos - 1}")

    def _parse_header(self) -> None:
        voice = self.t[self.pos] - VOICE_BASE
        self.pos += 1
        kind = self.t[self.pos]
        self.pos += 1
        if kind == TUNING:
            self.q[voice] = self._u()
            self.tuning[voice] = pitch_grid.q_to_tuning(self.q[voice])
        elif kind == NOTE_TABLE:
            count = self._u()
            prev = 0
            for _ in range(count):
                prev += self._s()
                self.devs[voice][prev] = self._s()
        elif kind == TICK:
            self.tick[voice] = (self._u(), self._s())
        else:
            raise ValueError(f"unknown header kind {kind} at {self.pos - 1}")

    def _freq_base(self, v: int, note: int) -> int:
        b = self._base[v].get(note)
        if b is None:
            b = pitch_grid.note_freq_at(note, self.tuning[v]) + self.devs[v].get(
                note, 0
            )
            self._base[v][note] = b
        return b

    def parse(self, tolerant: bool = False) -> tuple[int, int]:
        """Parse headers + frame groups into the per-frame op/event tables; returns ``(n_frames,
        last_group_frame)``. ``tolerant`` swallows a trailing truncated group (for state snapshots at
        arbitrary chunk boundaries) instead of raising."""
        t = self.t
        try:
            self.unit_starts.append(self.pos)
            n = self._u()
            while (
                self.pos + 1 < len(t)
                and _is_voice(t[self.pos])
                and t[self.pos + 1] in _HEADER_KINDS
            ):
                self.unit_starts.append(self.pos)
                self._parse_header()
        except (ValueError, IndexError):
            if not tolerant:
                raise
            return 0, 0
        if n == 0:
            if self.pos != len(t) and not tolerant:
                raise ValueError("trailing tokens after empty stream")
            return 0, 0
        cur_f = 0
        last_f = 0
        m = len(t)
        try:
            while self.pos < m:
                self.unit_starts.append(self.pos)
                dt = self._u()
                cur_f += dt
                while self.pos < m and _is_voice(t[self.pos]):
                    self.unit_starts.append(self.pos)
                    voice = t[self.pos] - VOICE_BASE
                    self.pos += 1
                    while self.pos < m and t[self.pos] in _EVENT_KINDS:
                        self.unit_starts.append(self.pos)
                        self._parse_event(cur_f, voice)
                if self.pos < m and not _is_digit(t[self.pos]):
                    raise ValueError(f"expected DT, VOICE or event at {self.pos}")
                last_f = cur_f
        except (ValueError, IndexError):
            if not tolerant:
                raise
        return n, last_f

    def replay(self, n: int) -> list[tuple[int, int, int]]:
        """Replay parsed state over frames ``[0, n)``, deriving the canonical write stream; leaves the
        channel/cas/global state at frame ``n - 1`` (read by :meth:`state`)."""
        out: list[tuple[int, int, int]] = []
        prev_byte = np.zeros(NUM_REGS, dtype=np.int64)
        ctrl_state = self._ctrl_state = [0, 0, 0]
        self._ad_state = [0, 0, 0]
        self._sr_state = [0, 0, 0]
        for f in range(n):
            for chan, op, args in self.chan_ops.get(f, ()):
                if op == "set":
                    chan.set(f, args)
                elif op == "set_rel":
                    chan.set(f, chan.at(f) + args)
                elif op == "ramp":
                    shape, length, params = args
                    chan.ramp(f, shape, length, params)
                else:
                    shape, length, rel, rest = args
                    chan.ramp(f, shape, length, [chan.at(f) + rel, *rest])
            for v in range(3):
                if self.freq_active[v]:
                    lo, hi = freq_regs(v)
                    freq = self._freq_base(v, self.ni[v].at(f)) + self.fd[v].at(f)
                    for r, b in ((lo, int(freq) & 0xFF), (hi, (int(freq) >> 8) & 0xFF)):
                        if b != prev_byte[r]:
                            out.append((f, r, b))
                            prev_byte[r] = b
                if self.pw_active[v]:
                    plo, phi = pw_regs(v)
                    pwv = int(self.pw[v].at(f))
                    for r, b in ((plo, pwv & 0xFF), (phi, (pwv >> 8) & 0xFF)):
                        if b != prev_byte[r]:
                            out.append((f, r, b))
                            prev_byte[r] = b
                offs = self.offs.get((v, f), ())
                offs = sorted(offs, key=lambda o: (not o[0], o[3]))
                merged = _slot_gate_offs(
                    self.cas.get((v, f), []),
                    [(o[0], (o[1], o[2])) for o in offs],
                )
                merged = _insert_hr(merged, self.hr.get((v, f), []))
                cr = ctrl_reg(v)
                for e in merged:
                    if e[0] == "OFF":
                        mode, go_val = e[1]
                        val = (
                            (ctrl_state[v] & ~1) if mode == _GO_DERIVE else int(go_val)
                        )
                        ctrl_state[v] = val
                        out.append((f, cr, val))
                    else:
                        tok, val = e
                        if tok in (FLD_NOTE_ON, FLD_CTRL):
                            ctrl_state[v] = val
                            out.append((f, cr, val))
                        elif tok in (FLD_AD, "OAD", "HRAD"):
                            self._ad_state[v] = val
                            out.append((f, ad_reg(v), val))
                        else:
                            self._sr_state[v] = val
                            out.append((f, sr_reg(v), val))
            for r in GLOBAL_REGS:
                if not self.g_active[r]:
                    continue
                b = int(self.g[r].at(f)) & 0xFF
                if b != prev_byte[r]:
                    out.append((f, r, b))
                    prev_byte[r] = b
        return out

    def run(self, extend: bool = False) -> list[tuple[int, int, int]]:
        n, last = self.parse()
        return self.replay(max(n, last + 1) if extend else n)


def decode(tokens: list[int], extend: bool = False) -> list[tuple[int, int, int]]:
    """v3 token stream -> the canonical ordered ``(frame, reg, value)`` writes. Strict grammar parser:
    malformed streams raise. Default truncates at the declared frame count; ``extend=True`` replays
    through the last parsed group (for model-generated continuations past the header count).
    """
    return _Decoder(tokens).run(extend=extend)


def chunk_keyframe(tokens: list[int], upto: int) -> list[int]:
    """Conditioning prefix for a training chunk starting at atom position ``upto`` of a whole-tune
    stream: a ``[KEYFRAME ... KEYFRAME]``-bracketed segment carrying the decoder state at the last
    complete frame group before ``upto`` (per-voice TUNING/TICK headers + current note index, freq
    residual, PW, CTRL/AD/SR and global values, in the ordinary event grammar). Conditioning only:
    :func:`strip_keyframes` removes segments before decode, so the encoding stays redundancy-free.
    """
    if upto <= 0:
        return []
    d = _Decoder(tokens[:upto])
    n, last_f = d.parse(tolerant=True)
    if n == 0:
        return []
    d.replay(min(last_f + 1, n))
    out = [KEYFRAME]
    f = last_f
    for v in range(3):
        voice_tok = VOICE_BASE + v
        if d.freq_active[v]:
            out += [voice_tok, TUNING]
            _emit_u(out, d.q[v])
        if d.freq_active[v] and d.devs[v]:
            out += [voice_tok, NOTE_TABLE]
            _emit_u(out, len(d.devs[v]))
            prev = 0
            for note in sorted(d.devs[v]):
                _emit_s(out, note - prev)
                _emit_s(out, d.devs[v][note])
                prev = note
        if d.tick[v] != (1, 0):
            out += [voice_tok, TICK]
            _emit_u(out, d.tick[v][0])
            _emit_s(out, d.tick[v][1])
    for v in range(3):
        voice_tok = VOICE_BASE + v
        body: list[int] = []
        if d.freq_active[v]:
            body.append(NI_STEP)
            _emit_s(body, int(d.ni[v].at(f)))
            body.append(FD_STEP)
            _emit_s(body, int(d.fd[v].at(f)))
        if d.pw_active[v]:
            pwv = int(d.pw[v].at(f))
            body.append(PW_STEP)
            body.append(NIB_ENV + ((pwv >> 8) & 0xF))
            _emit_u(body, pwv & 0xFF)
        if d.cas_active[v]:
            body.append(FLD_CTRL)
            _emit_ctrl_val(body, d._ctrl_state[v])  # pylint: disable=protected-access
            body.append(FLD_AD)
            _emit_env_val(body, d._ad_state[v])  # pylint: disable=protected-access
            body.append(FLD_SR)
            _emit_env_val(body, d._sr_state[v])  # pylint: disable=protected-access
        if body:
            out.append(voice_tok)
            out.extend(body)
    gbody: list[int] = []
    for reg in GLOBAL_REGS:
        if d.g_active[reg]:
            gbody += [G_STEP, REG_BASE + reg]
            _emit_u(gbody, int(d.g[reg].at(f)) & 0xFF)
    if gbody:
        out.append(VOICE_BASE + GLOBAL)
        out.extend(gbody)
    out.append(KEYFRAME)
    return out


def strip_keyframes(tokens: list[int]) -> list[int]:
    """Remove ``[KEYFRAME ... KEYFRAME]`` conditioning segments (chunk prefixes) from an atom stream,
    leaving the pure canonical encoding for :func:`decode`."""
    out: list[int] = []
    inside = False
    for t in tokens:
        if t == KEYFRAME:
            inside = not inside
            continue
        if not inside:
            out.append(t)
    return out


def roundtrip_ok(df) -> bool:
    """Encode then decode a raw dump df and compare to :func:`canonical_writes`."""
    ow = ordered_writes(df)
    return decode(encode(ow, verify=False)) == canonical_writes(ow)


def unit_starts(tokens) -> list[int]:
    """Grammar-unit start indices of an atom stream (frame-count varint, per-voice headers, DT runs,
    voice markers, events) -- the parser itself is the segmenter, so payload digits and DT digits are
    distinguished exactly. Raises on invalid or KEYFRAME-bearing streams (segment whole-tune ``encode``
    output only; ``strip_keyframes`` first if needed)."""
    d = _Decoder(tokens)
    d.parse()
    return d.unit_starts


__all__ = [
    "EVENT_FORMAT_VERSION",
    "KEYFRAME",
    "VOCAB_SIZE",
    "canonical_writes",
    "chunk_keyframe",
    "decode",
    "encode",
    "is_content_atom",
    "roundtrip_ok",
    "single_speed",
    "strip_keyframes",
    "unit_starts",
]
