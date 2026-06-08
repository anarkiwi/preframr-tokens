"""Factored v1 codec (REDESIGN_optionB §7.0, §8): per-register gesture lanes + a per-frame ORDER descriptor
+ a freq two-layer + a note/attack layer, each a complete escape-free encoding. Value lanes cover each
register's series losslessly with HOLD/POLY/PERIOD gestures; the ORDER descriptor is the ordered reg-id list
per frame (byte-exact intra-frame order, §13); multi-speed sub-frame repeats emit a literal write-run, not
an escape. Decode is asserted equal to the source write stream (§7) over one self-delimiting alphabet (§7.1).
"""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros import pitch_grid
from . import varint
from .gestures import Gesture, cover, replay
from .oracle import OrderedWrites, ordered_writes, settled_grid
from .schema import (
    AD,
    CTRL,
    FREQ_HI,
    FREQ_LO,
    NUM_REGS,
    SR,
    Shape,
    VOICE_OF,
    ad_reg,
    ctrl_reg,
    freq_regs,
    sr_reg,
)

CAS_REGS = CTRL | AD | SR

VOICE_FREQ = {r: VOICE_OF[r] for r in (FREQ_LO | FREQ_HI)}

_VSPAN = 32
VAR_BASE = 0
REG_BASE = VAR_BASE + _VSPAN
SHAPE_HOLD = REG_BASE + NUM_REGS
SHAPE_POLY = SHAPE_HOLD + 1
SHAPE_PERIOD = SHAPE_POLY + 1
ORDER_MARK = SHAPE_PERIOD + 1
LIT_MARK = ORDER_MARK + 1
FREQ_MARK = LIT_MARK + 1
NOTE_MARK = FREQ_MARK + 1
FLD_NOTE_ON = NOTE_MARK + 1
FLD_CTRL = FLD_NOTE_ON + 1
FLD_AD = FLD_CTRL + 1
FLD_SR = FLD_AD + 1
VOCAB_SIZE = FLD_SR + 1

_CTRL_FLDS = (FLD_NOTE_ON, FLD_CTRL)

_SHAPE_TOK = {
    Shape.HOLD: SHAPE_HOLD,
    Shape.POLY: SHAPE_POLY,
    Shape.PERIOD: SHAPE_PERIOD,
}
_TOK_SHAPE = {v: k for k, v in _SHAPE_TOK.items()}


def _emit_u(out: list[int], value: int) -> None:
    for d in varint.encode_unsigned(value):
        out.append(VAR_BASE + d)


def _emit_s(out: list[int], value: int) -> None:
    for d in varint.encode_signed(value):
        out.append(VAR_BASE + d)


def _read_var(tokens: list[int], pos: int) -> tuple[int, int]:
    shifted = []
    while True:
        if pos >= len(tokens) or not (VAR_BASE <= tokens[pos] < VAR_BASE + _VSPAN):
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


def _emit_level(out: list[int], v: int, signed: bool) -> None:
    (_emit_s if signed else _emit_u)(out, v)


def _read_level(tokens, pos, signed: bool):
    return (_read_s if signed else _read_u)(tokens, pos)


def _emit_gesture(out: list[int], g: Gesture, signed: bool = False) -> None:
    out.append(_SHAPE_TOK[g.shape])
    _emit_u(out, g.length)
    if g.shape == Shape.HOLD:
        _emit_level(out, g.params[0], signed)
    elif g.shape == Shape.POLY:
        _emit_u(out, len(g.params) - 1)
        _emit_level(out, g.params[0], signed)
        for d in g.params[1:]:
            _emit_s(out, d)
    else:
        _emit_u(out, len(g.params) - 1)
        _emit_level(out, g.params[0], signed)
        for c in g.params[1:]:
            _emit_s(out, c)


def _read_gesture(
    tokens: list[int], pos: int, start: int, signed: bool = False
) -> tuple[Gesture, int]:
    shape = _TOK_SHAPE[tokens[pos]]
    pos += 1
    length, pos = _read_u(tokens, pos)
    if shape == Shape.HOLD:
        v, pos = _read_level(tokens, pos, signed)
        params = (v,)
    elif shape == Shape.POLY:
        deg, pos = _read_u(tokens, pos)
        v0, pos = _read_level(tokens, pos, signed)
        rest = []
        for _ in range(deg):
            d, pos = _read_s(tokens, pos)
            rest.append(d)
        params = (v0, *rest)
    else:
        period, pos = _read_u(tokens, pos)
        anchor, pos = _read_level(tokens, pos, signed)
        cell = []
        for _ in range(period):
            c, pos = _read_s(tokens, pos)
            cell.append(c)
        params = (anchor, *cell)
    return Gesture(shape, start, length, tuple(params)), pos


def _emit_cover(out: list[int], series, signed: bool, wrap: bool = False) -> None:
    """Emit a full contiguous gesture cover of ``series`` (length sum == len(series))."""
    for g in cover(series, wrap=wrap):
        _emit_gesture(out, g, signed=signed)


def _read_cover(tokens: list[int], pos: int, n: int, signed: bool, wrap: bool = False):
    """Read a contiguous gesture cover tiling ``[0, n)``; returns ``(series_array, next_pos)``."""
    gestures: list[Gesture] = []
    acc = 0
    while acc < n:
        g, pos = _read_gesture(tokens, pos, acc, signed=signed)
        gestures.append(g)
        acc += g.length
    if acc != n:
        raise ValueError(f"gesture tiling {acc} != n {n}")
    return replay(gestures, n, wrap=wrap), pos


def _freq16(settled, v: int) -> np.ndarray:
    """The per-frame 16-bit freq for voice ``v`` from its settled lo/hi bytes."""
    lo, hi = freq_regs(v)
    return (settled[:, hi].astype(np.int64) << 8) | settled[:, lo].astype(np.int64)


def _note_base(ni: np.ndarray, tuning: float, table: dict) -> np.ndarray:
    """Per-frame freq base: the recovered note-table entry where present, else the equal-tempered grid
    (§8.2). With the table, a static note's residual is 0 -- the delta lane collapses to HOLD(0).
    """
    base = pitch_grid.note_freq(ni, tuning).copy()
    for note, freq in table.items():
        base[ni == note] = freq
    return base


def _emit_freq_voice(out: list[int], settled, v: int) -> None:
    """Emit a voice's freq as a tuning header + note-table deviations + note-index cover + delta cover
    (§8.2-8.4). ``freq = base(note_index) + delta`` is exact by construction (``delta`` is the residual),
    so the lane is byte-exact for any note-index choice; nearest-grid index + the recovered table is the
    greedy bootstrap of the §8.4 joint DP. Static notes -> delta HOLD(0); a transpose is the same
    note-index intervals at a different anchor (key-invariant, §8 transpose)."""
    F = _freq16(settled, v)
    q = pitch_grid.tuning_to_q(pitch_grid.voice_tuning(F))
    tuning = pitch_grid.q_to_tuning(q)
    table = pitch_grid.recover_table(F, tuning)
    ni = pitch_grid.note_index(F, tuning)
    delta = F - _note_base(ni, tuning, table)
    out.append(FREQ_MARK)
    _emit_u(out, v)
    _emit_u(out, q)
    _emit_u(out, len(table))
    for note in sorted(table):
        _emit_s(out, note)
        _emit_s(out, table[note] - pitch_grid.note_freq_at(note, tuning))
    _emit_cover(out, ni, signed=True)
    _emit_cover(out, delta, signed=True)


def _read_freq_voice(tokens: list[int], pos: int, n: int, series: dict) -> int:
    """Read one freq-voice section, populating ``series[lo]`` / ``series[hi]`` from the recon."""
    pos += 1
    v, pos = _read_u(tokens, pos)
    q, pos = _read_u(tokens, pos)
    tuning = pitch_grid.q_to_tuning(q)
    n_tab, pos = _read_u(tokens, pos)
    table = {}
    for _ in range(n_tab):
        note, pos = _read_s(tokens, pos)
        dev, pos = _read_s(tokens, pos)
        table[note] = pitch_grid.note_freq_at(note, tuning) + dev
    ni, pos = _read_cover(tokens, pos, n, signed=True)
    delta, pos = _read_cover(tokens, pos, n, signed=True)
    F = _note_base(ni, tuning, table) + delta
    lo, hi = freq_regs(v)
    series[lo] = F & 0xFF
    series[hi] = (F >> 8) & 0xFF
    return pos


def _note_edges(settled, v: int) -> list[tuple[int, int, int]]:
    """Ordered ``(frame, field_tok, value)`` edges covering the settled CTRL/AD/SR change-points of voice
    ``v``. CTRL edges that turn the gate on (bit0 0->1) are typed ``FLD_NOTE_ON`` (§6).
    """
    cr, ar, srg = ctrl_reg(v), ad_reg(v), sr_reg(v)
    C = settled[:, cr]
    A = settled[:, ar]
    S = settled[:, srg]
    edges: list[tuple[int, int, int]] = []
    pc = pa = ps = 0
    gate = 0
    n = C.shape[0]
    for f in range(n):
        c = int(C[f])
        if c != pc:
            ng = c & 1
            tok = FLD_NOTE_ON if (ng and not gate) else FLD_CTRL
            edges.append((f, tok, c))
            gate = ng
            pc = c
        a = int(A[f])
        if a != pa:
            edges.append((f, FLD_AD, a))
            pa = a
        s = int(S[f])
        if s != ps:
            edges.append((f, FLD_SR, s))
            ps = s
    return edges


def _emit_note_voice(out: list[int], settled, v: int) -> None:
    edges = _note_edges(settled, v)
    out.append(NOTE_MARK)
    _emit_u(out, v)
    _emit_u(out, len(edges))
    prev_f = 0
    for f, tok, val in edges:
        _emit_u(out, f - prev_f)
        out.append(tok)
        _emit_u(out, val)
        prev_f = f


def _read_note_voice(tokens: list[int], pos: int, n: int, series: dict) -> int:
    """Read one voice's note section, populating ``series[ctrl/ad/sr]`` with the forward-filled series."""
    pos += 1
    v, pos = _read_u(tokens, pos)
    n_edges, pos = _read_u(tokens, pos)
    cr, ar, srg = ctrl_reg(v), ad_reg(v), sr_reg(v)
    C = np.zeros(n, dtype=np.int64)
    A = np.zeros(n, dtype=np.int64)
    S = np.zeros(n, dtype=np.int64)
    cur_f = 0
    edges: list[tuple[int, int, int]] = []
    for _ in range(n_edges):
        dt, pos = _read_u(tokens, pos)
        cur_f += dt
        tok = tokens[pos]
        pos += 1
        val, pos = _read_u(tokens, pos)
        edges.append((cur_f, tok, val))
    c = a = s = 0
    ei = 0
    for f in range(n):
        while ei < len(edges) and edges[ei][0] == f:
            _, tok, val = edges[ei]
            if tok in _CTRL_FLDS:
                c = val
            elif tok == FLD_AD:
                a = val
            elif tok == FLD_SR:
                s = val
            else:
                raise ValueError(f"unexpected note-edge field {tok}")
            ei += 1
        C[f] = c
        A[f] = a
        S[f] = s
    series[cr] = C
    series[ar] = A
    series[srg] = S
    return pos


def encode(ow: OrderedWrites) -> list[int]:
    """Ordered write stream -> factored v1 token id list (byte-exact; see module docstring)."""
    n = ow.n_frames
    out: list[int] = []
    _emit_u(out, n)
    if n == 0:
        out.append(ORDER_MARK)
        return out

    by_frame = ow.by_frame()
    repeat_frames = set()
    for f, writes in enumerate(by_frame):
        seen = set()
        for reg, _ in writes:
            if reg in seen:
                repeat_frames.add(f)
                break
            seen.add(reg)

    settled = settled_grid(ow)
    lane_regs = set()
    for f, writes in enumerate(by_frame):
        if f in repeat_frames or not writes:
            continue
        for reg, _ in writes:
            lane_regs.add(reg)

    freq_voices = sorted({VOICE_FREQ[r] for r in lane_regs if r in VOICE_FREQ})
    for v in freq_voices:
        _emit_freq_voice(out, settled, v)
        lane_regs.discard(freq_regs(v)[0])
        lane_regs.discard(freq_regs(v)[1])

    note_voices = sorted({VOICE_OF[r] for r in lane_regs if r in CAS_REGS})
    for v in note_voices:
        _emit_note_voice(out, settled, v)
        lane_regs.discard(ctrl_reg(v))
        lane_regs.discard(ad_reg(v))
        lane_regs.discard(sr_reg(v))

    for reg in sorted(lane_regs):
        out.append(REG_BASE + reg)
        for g in cover(settled[:, reg], wrap=False):
            _emit_gesture(out, g)

    out.append(ORDER_MARK)
    prev_f = 0
    for f, writes in enumerate(by_frame):
        if not writes:
            continue
        _emit_u(out, f - prev_f)
        prev_f = f
        if f in repeat_frames:
            out.append(LIT_MARK)
            _emit_u(out, len(writes))
            for reg, val in writes:
                out.append(REG_BASE + reg)
                _emit_u(out, val)
        else:
            for reg, _ in writes:
                out.append(REG_BASE + reg)
    return out


def decode(tokens: list[int]) -> list[tuple[int, int, int]]:
    """Factored v1 token id list -> ordered ``(frame, reg, value)`` writes (the §7 fidelity target)."""
    pos = 0
    n, pos = _read_u(tokens, pos)
    series = {}
    while pos < len(tokens) and tokens[pos] == FREQ_MARK:
        pos = _read_freq_voice(tokens, pos, n, series)
    while pos < len(tokens) and tokens[pos] == NOTE_MARK:
        pos = _read_note_voice(tokens, pos, n, series)
    while pos < len(tokens) and REG_BASE <= tokens[pos] < REG_BASE + NUM_REGS:
        reg = tokens[pos] - REG_BASE
        pos += 1
        gestures: list[Gesture] = []
        acc = 0
        while acc < n:
            g, pos = _read_gesture(tokens, pos, acc)
            gestures.append(g)
            acc += g.length
        if acc != n:
            raise ValueError(f"reg {reg} gesture tiling {acc} != n_frames {n}")
        series[reg] = replay(gestures, n, wrap=False)

    if pos >= len(tokens) or tokens[pos] != ORDER_MARK:
        raise ValueError(f"expected ORDER_MARK at {pos}")
    pos += 1

    out: list[tuple[int, int, int]] = []
    cur_f = 0
    m = len(tokens)
    while pos < m:
        dt, pos = _read_u(tokens, pos)
        cur_f += dt
        if pos < m and tokens[pos] == LIT_MARK:
            pos += 1
            count, pos = _read_u(tokens, pos)
            for _ in range(count):
                reg = tokens[pos] - REG_BASE
                pos += 1
                val, pos = _read_u(tokens, pos)
                out.append((cur_f, reg, val))
        else:
            while pos < m and REG_BASE <= tokens[pos] < REG_BASE + NUM_REGS:
                reg = tokens[pos] - REG_BASE
                pos += 1
                out.append((cur_f, reg, int(series[reg][cur_f])))
    return out


def roundtrip_ok(df) -> bool:
    """Convenience: encode then decode a raw dump df and compare to the oracle byte-for-byte."""
    ow = ordered_writes(df)
    return decode(encode(ow)) == ow.triples()
