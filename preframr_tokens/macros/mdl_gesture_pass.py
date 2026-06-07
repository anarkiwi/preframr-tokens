"""MdlGesturePass: the single MDL optimal-parse pass that replaces InstrumentProgramPass + GeneratorPass
(MDL_PARSER_IMPLEMENTATION.md §4), parsing each settled value channel into the driver's own HOLD /
POLY(N) forward-difference / PERIOD primitives and emitting them as the unified ``gesture`` codebook
family; it owns the non-freq scalar channels here (Step 2) and grows the joint 2-D freq parse (Step 3),
arbitrated with ``validate=True`` so every claim is byte-exact or dropped to the literal stream.
"""

from __future__ import annotations

__all__ = ["MdlGesturePass"]

import numpy as np

from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.generator_fit import zig
from preframr_tokens.macros.mdl_core import difftable, mdl_parse
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    GEN_FREQ_REGS,
    GEN_TUNING_OP,
    GEN_TUNING_SUBREG_FREQ_HI,
    GEN_TUNING_SUBREG_FREQ_LO,
    GEN_TUNING_SUBREG_NOTE,
    GEN_TUNING_SUBREG_REF,
    GEN_TUNING_SUBREG_VOICE,
    NOTE_INTERVAL_OP,
    NOTE_INTERVAL_SUBREG_FIRST,
    NOTE_INTERVAL_SUBREG_INTERVAL_HI,
    NOTE_INTERVAL_SUBREG_INTERVAL_LO,
    NOTE_INTERVAL_SUBREG_VOICE,
    GESTURE_DEF_OP,
    GESTURE_END_OP,
    GESTURE_KIND_HOLD,
    GESTURE_KIND_PERIOD,
    GESTURE_KIND_POLY,
    GESTURE_REF_OP,
    GESTURE_REF_SUBREG_ANCHOR_HI,
    GESTURE_REF_SUBREG_ANCHOR_LO,
    GESTURE_REF_SUBREG_D1_HI,
    GESTURE_REF_SUBREG_D1_LO,
    GESTURE_REF_SUBREG_D2_HI,
    GESTURE_REF_SUBREG_D2_LO,
    GESTURE_REF_SUBREG_ID,
    GESTURE_REF_SUBREG_LEN_HI,
    GESTURE_REF_SUBREG_LEN_LO,
    GESTURE_STEP_OP,
    GESTURE_SUBREG_CELL_HI,
    GESTURE_SUBREG_CELL_LO,
    GESTURE_SUBREG_DEGREE,
    GESTURE_SUBREG_KIND,
    SET_OP,
)

_GESTURE_PRIORITY = -8
_MAX_LEN = 0xFFFF

_SCALAR_REGS = (2, 9, 16, 21, 23, 24, 4, 5, 6, 11, 12, 13, 18, 19, 20)


def _row(reg, op, subreg, val, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=irq, irq=irq)


def _emit_specs(series, wrap):
    """Yield ``(shape_key, kind, degree, cell, anchor, d1, d2, start, length)`` per MDL gesture run of
    ``series``; ``shape_key`` interns the reusable shape (HOLD kind, POLY (degree, N-th diff), or PERIOD
    cell), the rest are the per-instance anchor + lower-order initial diffs + length the REF carries.
    """
    s = np.asarray(series, dtype=np.int64)
    for kind, i, j, param in mdl_parse(s, wrap):
        length = j - i
        anchor = int(s[i])
        if kind == "H":
            yield ("H",), GESTURE_KIND_HOLD, 0, (), anchor, 0, 0, i, length
        elif kind == "D":
            degree, ndiff = int(param[0]), int(param[1])
            dt = difftable(s, i, degree, wrap)
            d1 = dt[1] if degree >= 2 else 0
            d2 = dt[2] if degree >= 3 else 0
            cell = (ndiff,)
            yield (
                "D",
                degree,
                ndiff,
            ), GESTURE_KIND_POLY, degree, cell, anchor, d1, d2, i, length
        else:
            cell = tuple(int(c) for c in param)
            yield ("P", cell), GESTURE_KIND_PERIOD, len(
                cell
            ), cell, anchor, 0, 0, i, length


def _def_rows(cb_id, kind, degree, cell, irq):
    """The DEF/STEP/END rows serialising one reusable shape into the dictionary (reg 0, voice-relative):
    KIND + DEGREE one step each, then a signed-16-bit CELL_LO/CELL_HI pair per shape value.
    """
    rows = [
        _row(0, GESTURE_DEF_OP, -1, cb_id, irq),
        _row(0, GESTURE_STEP_OP, GESTURE_SUBREG_KIND, kind, irq),
        _row(0, GESTURE_STEP_OP, GESTURE_SUBREG_DEGREE, degree, irq),
    ]
    for c in cell:
        rows.append(_row(0, GESTURE_STEP_OP, GESTURE_SUBREG_CELL_LO, c & 0xFF, irq))
        rows.append(
            _row(0, GESTURE_STEP_OP, GESTURE_SUBREG_CELL_HI, (c >> 8) & 0xFF, irq)
        )
    rows.append(_row(0, GESTURE_END_OP, -1, cb_id, irq))
    return rows


_NOTE_SUSTAIN = 3


def _held_notes(note_target, f0):
    """The note-index layer: re-anchor to a new note only when ``note_target`` settles there for
    ``_NOTE_SUSTAIN`` frames, so transient vibrato wobbles across a semitone boundary stay in the
    freq-delta layer (not minted as note events) while sustained arp/melody steps become note events --
    the validated greedy re-anchor (MDL_TRANSITION_DESIGN.md), the joint-DP's tractable approximation.
    """
    nt = np.asarray(note_target, dtype=np.int64)
    n = len(nt)
    out = np.empty(n, dtype=np.int64)
    cur = int(nt[f0]) if f0 < n else 0
    for t in range(f0, n):
        v = int(nt[t])
        if v != cur and all(
            int(nt[u]) == v for u in range(t, min(n, t + _NOTE_SUSTAIN))
        ):
            cur = v
        out[t] = cur
    out[:f0] = cur if f0 < n else 0
    return out


def _tuning_rows(voice, tuning_q, table, irq):
    """Head GEN_TUNING rows carrying the per-voice tuning byte + the recovered note->freq table, so the
    decoder rebuilds the exact base the freq-delta gestures ride (note in signed-byte range).
    """
    rows = [
        _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_VOICE, voice, irq),
        _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_REF, tuning_q, irq),
    ]
    for note, freq in sorted(table.items()):
        f = int(freq) & 0xFFFF
        rows.append(
            _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_NOTE, int(note) & 0xFF, irq)
        )
        rows.append(_row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_FREQ_LO, f & 0xFF, irq))
        rows.append(
            _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_FREQ_HI, (f >> 8) & 0xFF, irq)
        )
    return rows


def _note_interval_rows(voice, first, interval, irq):
    """A NOTE_INTERVAL atom: the zig-zag interval to this note (FIRST marks an absolute note index)."""
    z = zig(int(interval)) & 0xFFFF
    return [
        _row(0, NOTE_INTERVAL_OP, NOTE_INTERVAL_SUBREG_VOICE, voice, irq),
        _row(0, NOTE_INTERVAL_OP, NOTE_INTERVAL_SUBREG_FIRST, 1 if first else 0, irq),
        _row(
            0, NOTE_INTERVAL_OP, NOTE_INTERVAL_SUBREG_INTERVAL_HI, (z >> 8) & 0xFF, irq
        ),
        _row(0, NOTE_INTERVAL_OP, NOTE_INTERVAL_SUBREG_INTERVAL_LO, z & 0xFF, irq),
    ]


def _ref_rows(reg, cb_id, anchor, d1, d2, length, irq):
    """The fixed-layout REF rows replaying shape ``cb_id`` on ``reg``: ID, ANCHOR, D1, D2 (signed 16-bit
    pairs), then LEN_HI/LEN_LO last so LEN_LO triggers the replay (matching the gesture codec).
    """
    a, d1u, d2u, ln = anchor & 0xFFFF, d1 & 0xFFFF, d2 & 0xFFFF, length & 0xFFFF
    return [
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_ID, cb_id, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_ANCHOR_LO, a & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_ANCHOR_HI, (a >> 8) & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_D1_LO, d1u & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_D1_HI, (d1u >> 8) & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_D2_LO, d2u & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_D2_HI, (d2u >> 8) & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_LEN_HI, (ln >> 8) & 0xFF, irq),
        _row(reg, GESTURE_REF_OP, GESTURE_REF_SUBREG_LEN_LO, ln & 0xFF, irq),
    ]


class MdlGesturePass(MacroPass):
    """Replace the per-gesture greedy passes with one MDL optimal parse over HOLD/POLY/PERIOD gestures
    interned in the corpus-global ``gesture`` codebook: scalar channels (PW, cutoff, res, vol, ctrl/AD/SR)
    parse directly, the freq channels split into a note-index layer (NOTE_INTERVAL) + a freq-delta gesture
    layer; ``encode`` is byte-exact standalone and ``apply`` routes to it at the Step-5 production swap.
    """

    GATE_FLAGS: frozenset = frozenset()

    def apply(self, df, args=None):
        """No-op in the pipeline until the build-order Step-5 swap routes it to ``encode`` and retires the
        generator/instrument passes; kept in FREQ_BLOCK_PASSES so the reference-producer contract holds
        while the encoder is grown and validated standalone (test_mdl_gesture)."""
        return df

    def encode(self, df):
        """Parse the scalar value channels into gesture DEF/REF tokens, arbitrated ``validate=True`` so
        every claim is byte-exact or dropped to the literal stream; the build-order Step-5 entry point
        that ``apply`` will delegate to once it owns the channels."""
        from preframr_tokens.audit_primitives import register_state

        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        if df["op"].isin((GESTURE_DEF_OP, GESTURE_REF_OP)).any():
            return df
        irq = _first_irq(df)
        state = register_state(df)
        writes = self._collect_writes(df, _SCALAR_REGS)
        frame_rows = self._frame_marker_rows(df)
        bank: dict = {}
        def_rows: list = []
        head_rows: list = []
        claims: list = []
        for reg in _SCALAR_REGS:
            if reg >= state.shape[1] or not writes[reg]:
                continue
            claims.extend(
                self._channel_claims(
                    reg, state[:, reg], writes[reg], frame_rows, bank, def_rows, irq
                )
            )
        freq_writes = self._collect_writes(df, GEN_FREQ_REGS)
        for reg in GEN_FREQ_REGS:
            if reg >= state.shape[1] or not freq_writes[reg]:
                continue
            claims.extend(
                self._freq_claims(
                    reg,
                    state[:, reg],
                    freq_writes[reg],
                    frame_rows,
                    bank,
                    def_rows,
                    head_rows,
                    irq,
                )
            )
        if head_rows:
            claims.append(
                Claim(
                    writes=(),
                    tokens=head_rows,
                    priority=_GESTURE_PRIORITY,
                    label="gesture_tuning",
                )
            )
        if def_rows:
            claims.append(
                Claim(
                    writes=(),
                    tokens=def_rows,
                    priority=_GESTURE_PRIORITY,
                    label="gesture_defs",
                )
            )
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @classmethod
    def _channel_claims(cls, reg, series, writes, frame_rows, bank, def_rows, irq):
        """One Claim per MDL gesture run over the channel's active region (first write -> last frame):
        each REF drops the raw SETs in its span and splices at the run's anchored frame, interning its
        shape DEF once at the stream head so no REF can precede its DEF."""
        f0 = int(writes[0][0])
        anchor_pos: dict = {}
        drop_at: dict = {}
        for frame, ri, _v in writes:
            anchor_pos.setdefault(int(frame), int(ri))
            drop_at.setdefault(int(frame), []).append(int(ri))
        active = np.asarray(series, dtype=np.int64)[f0:]
        claims = []
        for shape_key, kind, degree, cell, anchor, d1, d2, start, length in _emit_specs(
            active, False
        ):
            if kind == GESTURE_KIND_HOLD and length == 1:
                continue
            for off in range(0, length, _MAX_LEN):
                if off != 0 and kind != GESTURE_KIND_HOLD:
                    break
                seg_len = min(_MAX_LEN, length - off)
                frame = f0 + start + off
                pos = anchor_pos.get(frame)
                if pos is None:
                    pos = frame_rows.get(frame)
                if pos is None:
                    continue
                cb_id = cls._intern(shape_key, kind, degree, cell, def_rows, bank, irq)
                seg_anchor = anchor if off == 0 else int(np.asarray(series)[frame])
                seg_d1, seg_d2 = (d1, d2) if off == 0 else (0, 0)
                rows = _ref_rows(reg, cb_id, seg_anchor, seg_d1, seg_d2, seg_len, irq)
                for r in rows:
                    r["__pos"] = pos
                drop = tuple(
                    ri
                    for fr in range(frame, frame + seg_len)
                    for ri in drop_at.get(fr, ())
                )
                claims.append(
                    Claim(
                        writes=drop,
                        tokens=rows,
                        priority=_GESTURE_PRIORITY,
                        label="gesture",
                    )
                )
        return claims

    @classmethod
    def _freq_claims(
        cls, reg, series, writes, frame_rows, bank, def_rows, head_rows, irq
    ):
        """The two-layer freq parse (MDL_PARSER_IMPLEMENTATION.md §1.1): recover the per-voice tuning +
        note table, split each frame into a note index + a raw-Hz freq-delta, then emit a NOTE_INTERVAL
        per note span and HOLD/POLY/PERIOD gesture REFs over the delta. Decode is ``note_freq(note) +
        delta``, byte-exact; arbitrate validate=True drops any misaligned span to the literal stream.
        """
        f = np.asarray(series, dtype=np.int64)
        n = len(f)
        voice = reg // 7
        f0 = int(writes[0][0])
        tuning_q = pitch_grid.tuning_to_q(pitch_grid.voice_tuning(f))
        tuning = pitch_grid.q_to_tuning(tuning_q)
        table = {
            m: v
            for m, v in pitch_grid.recover_table(f, tuning).items()
            if -128 <= m <= 127
        }
        note = _held_notes(pitch_grid.note_index(f, tuning), f0)
        base = np.array(
            [table.get(int(m), pitch_grid.note_freq_at(int(m), tuning)) for m in note],
            dtype=np.int64,
        )
        delta = f - base
        head_rows.extend(_tuning_rows(voice, tuning_q, table, irq))
        anchor_pos: dict = {}
        drop_at: dict = {}
        for frame, ri, _v in writes:
            anchor_pos.setdefault(int(frame), int(ri))
            drop_at.setdefault(int(frame), []).append(int(ri))
        claims = []
        prev_note = None
        t = f0
        while t < n:
            m = int(note[t])
            b = t
            while b < n and int(note[b]) == m:
                b += 1
            pos = anchor_pos.get(t, frame_rows.get(t))
            if pos is None:
                t = b
                continue
            first = prev_note is None
            interval = m if first else m - prev_note
            prev_note = m
            span_tokens = []
            for nr in _note_interval_rows(voice, first, interval, irq):
                nr["__pos"] = pos
                span_tokens.append(nr)
            for skey, kind, degree, cell, danchor, d1, d2, rstart, rlen in _emit_specs(
                delta[t:b], False
            ):
                if rlen > _MAX_LEN:
                    continue
                cb_id = cls._intern(skey, kind, degree, cell, def_rows, bank, irq)
                rframe = t + rstart
                rpos = anchor_pos.get(rframe, frame_rows.get(rframe, pos))
                for rr in _ref_rows(reg, cb_id, danchor, d1, d2, rlen, irq):
                    rr["__pos"] = rpos
                    span_tokens.append(rr)
            drop = tuple(ri for fr in range(t, b) for ri in drop_at.get(fr, ()))
            claims.append(
                Claim(
                    writes=drop,
                    tokens=span_tokens,
                    priority=_GESTURE_PRIORITY,
                    label="gesture_freq",
                )
            )
            t = b
        return claims

    @staticmethod
    def _intern(shape_key, kind, degree, cell, def_rows, bank, irq):
        """Return the dictionary id for ``shape_key``, appending its DEF rows to the stream-head
        ``def_rows`` (``__pos`` -1) on first sight so each shape is defined once before any REF.
        """
        if shape_key in bank:
            return bank[shape_key]
        cb_id = len(bank)
        bank[shape_key] = cb_id
        rows = _def_rows(cb_id, kind, degree, cell, irq)
        for r in rows:
            r["__pos"] = -1
        def_rows.extend(rows)
        return cb_id

    @staticmethod
    def _collect_writes(df, target_regs):
        """Per target reg, ordered ``(real_frame, row_idx, val)`` for plain SETs (subreg -1); real_frame
        counts a FRAME_REG as one frame and a DELAY_REG as its ``val`` frames."""
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        out = {int(r): [] for r in target_regs}
        rf = 0
        for i in range(len(df)):
            reg = int(regs[i])
            if reg == FRAME_REG:
                rf += 1
            elif reg == DELAY_REG:
                rf += int(vals[i])
            elif int(ops[i]) != SET_OP or int(subregs[i]) != -1:
                continue
            elif reg in out:
                out[reg].append((rf, int(i), int(vals[i])))
        return out

    @staticmethod
    def _frame_marker_rows(df):
        """``real_frame -> the marker row a spliced gesture decodes inside for that frame``, so a run
        starting on a write-less held frame splices at that frame's FRAME/DELAY marker row.
        """
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        out = {0: 0}
        rf = 0
        for i in range(len(df)):
            reg = int(regs[i])
            if reg == FRAME_REG:
                rf += 1
                out[rf] = int(i)
            elif reg == DELAY_REG:
                rf += int(vals[i])
                out[rf] = int(i)
        return out
