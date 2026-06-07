"""Single source of truth for the inline-codebook families (INSTRUMENT, GENERATOR).
Each shares one machine: DEF opens a pending entry, STEP(s) accumulate its payload, a COMMIT
(an END op or a STEP at a terminal subreg) makes the id live in an ``id -> entry`` table, and
REF(s) replay a live id. This leaf (stfconstants only) declares that as ``CodebookFamily``
records and derives the ``op_contracts.CODEBOOK_SPECS`` table the registry test pins equal.
"""

from __future__ import annotations

from dataclasses import dataclass

from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.generator_fit import recon
from preframr_tokens.stfconstants import (
    GEN_FREQ_REGS,
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
    GEN_TABLE_DEF_OP,
    GEN_TABLE_END_OP,
    GEN_TABLE_MODE_NOTE,
    GEN_TABLE_MODE_NOTE_UNIV,
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_BASE_NOTE,
    GEN_TABLE_REF_SUBREG_ID,
    GEN_TABLE_REF_SUBREG_LEN_HI,
    GEN_TABLE_REF_SUBREG_LEN_LO,
    GEN_TABLE_REF_SUBREG_RESID_HI,
    GEN_TABLE_REF_SUBREG_RESID_LO,
    GEN_TABLE_STEP_OP,
    GEN_TABLE_SUBREG_ABS_HI,
    GEN_TABLE_SUBREG_ABS_LO,
    GEN_TABLE_SUBREG_BASE_NOTE,
    GEN_TABLE_SUBREG_MODE,
    GEN_TABLE_SUBREG_OFFSET,
    GEN_TABLE_SUBREG_PERIOD,
    GEN_TABLE_SUBREG_RESID_HI,
    GEN_TABLE_SUBREG_RESID_LO,
    INSTR_DEF_OP,
    INSTR_END_OP,
    INSTR_OFF_CTRL,
    INSTR_REF_OP,
    INSTR_STEP_OP,
    INSTR_SUBREG_FRAME,
)

__all__ = [
    "CODEBOOK_TABLE_NAMES",
    "RefSpec",
    "CodebookFamily",
    "CODEBOOK_FAMILIES",
    "family_by_name",
    "family_for_op",
    "codebook_spec_tuples",
    "CodebookDecoder",
    "codebook_decoders",
    "DEAD_REF_POLICY",
    "_Codebook",
]

DEAD_REF_POLICY = "drop"

CODEBOOK_TABLE_NAMES: tuple[str, ...] = (
    "instrument",
    "generator",
    "gesture",
)


@dataclass(frozen=True)
class RefSpec:
    """One REF op of a family. ``id_subreg`` is the subreg carrying the codebook id (a
    multi-row ref like GEN_TABLE_REF) or ``None`` when the op's own row carries
    it (a single-row ref like INSTR_REF). ``table_less`` marks a ref that
    carries its payload inline and looks up no id; such ops are not
    liveness-tracked and stay out of the spec table."""

    op: int
    id_subreg: int | None = None
    table_less: bool = False


@dataclass(frozen=True)
class CodebookFamily:
    """Complete declaration of one inline-codebook family. ``def_op``/``commit_op``/
    ``commit_subreg``/``refs`` derive the per-op ``CodebookSpec`` consumed by validation and the
    legality mask; ``step_ops`` and ``def_emits`` additionally drive the unified decoder.
    ``commit_subreg`` is ``None`` when ``commit_op`` is a dedicated END op, else the terminal
    STEP subreg that triggers the commit."""

    name: str
    def_op: int
    commit_op: int
    refs: tuple[RefSpec, ...]
    commit_subreg: int | None = None
    step_ops: tuple[int, ...] = ()
    def_emits: bool = False

    @property
    def table_index(self) -> int:
        return CODEBOOK_TABLE_NAMES.index(self.name)

    @property
    def ops(self) -> frozenset[int]:
        """Every op code this family owns: def, steps, commit, and refs (incl. table-less)."""
        out = {self.def_op, self.commit_op, *self.step_ops}
        out.update(r.op for r in self.refs)
        return frozenset(out)

    def spec_tuples(self) -> dict[int, tuple[str, str, int | None]]:
        """``op -> (table, kind, subreg)`` for the liveness-tracked ops, matching the legacy
        ``op_contracts.CODEBOOK_SPECS`` shape. Non-commit STEP ops and table-less refs are
        omitted (they look up no id), exactly as in the legacy literal."""
        out: dict[int, tuple[str, str, int | None]] = {
            self.def_op: (self.name, "def", None),
            self.commit_op: (self.name, "commit", self.commit_subreg),
        }
        for r in self.refs:
            if not r.table_less:
                out[r.op] = (self.name, "ref", r.id_subreg)
        return out


CODEBOOK_FAMILIES: dict[str, CodebookFamily] = {
    "instrument": CodebookFamily(
        name="instrument",
        def_op=INSTR_DEF_OP,
        step_ops=(INSTR_STEP_OP,),
        commit_op=INSTR_END_OP,
        commit_subreg=None,
        refs=(RefSpec(INSTR_REF_OP),),
        def_emits=False,
    ),
    "generator": CodebookFamily(
        name="generator",
        def_op=GEN_TABLE_DEF_OP,
        step_ops=(GEN_TABLE_STEP_OP,),
        commit_op=GEN_TABLE_END_OP,
        commit_subreg=None,
        refs=(RefSpec(GEN_TABLE_REF_OP, id_subreg=GEN_TABLE_REF_SUBREG_ID),),
        def_emits=False,
    ),
    "gesture": CodebookFamily(
        name="gesture",
        def_op=GESTURE_DEF_OP,
        step_ops=(GESTURE_STEP_OP,),
        commit_op=GESTURE_END_OP,
        commit_subreg=None,
        refs=(RefSpec(GESTURE_REF_OP, id_subreg=GESTURE_REF_SUBREG_ID),),
        def_emits=False,
    ),
}


def family_by_name(name: str) -> CodebookFamily:
    return CODEBOOK_FAMILIES[name]


_OP_TO_FAMILY: dict[int, CodebookFamily] = {
    op: fam for fam in CODEBOOK_FAMILIES.values() for op in fam.ops
}


def family_for_op(op: int) -> CodebookFamily | None:
    return _OP_TO_FAMILY.get(int(op))


def codebook_spec_tuples() -> dict[int, tuple[str, str, int | None]]:
    """Union of every family's ``spec_tuples`` -- the registry-derived ``op -> (table, kind,
    subreg)`` equivalent of ``op_contracts.CODEBOOK_SPECS``, asserted equal by the registry test.
    """
    out: dict[int, tuple[str, str, int | None]] = {}
    for fam in CODEBOOK_FAMILIES.values():
        out.update(fam.spec_tuples())
    return out


class _Codebook:
    """One inline-codebook family's live decode state: the ``id -> entry`` table plus the two pending
    buffers the unified machine reassembles into (``pending`` for an in-flight DEF, ``pending_ref`` for
    a multi-row / table-less REF). Collapses the per-family ``pending_*`` fields onto ``DecodeState``.
    """

    __slots__ = ("table", "pending", "pending_ref")

    def __init__(self):
        self.table = {}
        self.pending = None
        self.pending_ref = None


class _Codec:
    """Per-family payload codec lifted verbatim from the legacy per-op decoder. ``CodebookDecoder``
    owns the def/step/commit/ref lifecycle and the id table; the codec only serialises/replays the
    family's payload onto ``state`` (via the family's ``_Codebook`` ``cb``) so behaviour is
    byte-identical to the decoder it replaces."""

    def def_open(self, row, state, cb):
        raise NotImplementedError

    def step(self, state, row, cb):
        return None

    def commit(self, state, row, cb):
        return None

    def ref(self, state, row, cb):
        return None


class _InstrumentCodec(_Codec):
    """Note-onset timbre program: DEF/STEP/END buffer a per-frame ``(ctrl, AD, SR)`` walk into the id
    table, REF replays it on the target voice -- each frame's fields queue onto the voice's ctrl/AD/SR
    regs (voice base ``reg - INSTR_OFF_CTRL`` plus the field's voice-relative subreg), one drained per
    frame tick. A voice-relative per-frame write-series codebook."""

    def def_open(self, row, state, cb):
        cb.pending = {"id": int(row.val), "frames": [[]]}

    def step(self, state, row, cb):
        prog = cb.pending
        if prog is None:
            return None
        if int(row.subreg) == INSTR_SUBREG_FRAME:
            prog["frames"].append([])
        else:
            prog["frames"][-1].append((int(row.subreg), int(row.val)))
        return None

    def commit(self, state, row, cb):
        prog = cb.pending
        if prog is not None:
            cb.table[int(prog["id"])] = prog["frames"]
            cb.pending = None
        return None

    @staticmethod
    def _offsets_in_order(frames):
        """Voice-relative subregs in first-write order across the buffered frames."""
        offsets, seen = [], set()
        for fr in frames:
            for sub, _val in fr:
                if sub not in seen:
                    seen.add(sub)
                    offsets.append(sub)
        return offsets

    def ref(self, state, row, cb):
        frames = cb.table.get(int(row.val))
        if not frames:
            return None
        base = int(row.reg) - INSTR_OFF_CTRL
        offsets = self._offsets_in_order(frames)
        pre = state.maybe_flush_for(int(row.reg), -1)
        cur = {}
        for fr in frames:
            for sub, val in fr:
                cur[sub] = val
            for sub in offsets:
                if sub in cur:
                    state.pending_set_writes[base + sub].append(int(cur[sub]))
        return pre or None


class _GeneratorCodec(_Codec):
    """Generator-MDL TABLE codebook: DEF/STEP/END buffer one periodic cycle (absolute byte cycle for
    scalar channels, or note-relative offset+residual cycle for freq) into the id table; the multi-row
    REF carries the per-instance base_note + LEN and queues ``cycle[k % P]`` for LEN frames onto its
    reg. The note-relative decode is ``recon(base_note + offset[k], ref) + resid[k]`` -- exact, with
    ``ref`` the per-tune semitone-LUT offset a ``GEN_TUNING`` atom stored on the state.
    """

    def def_open(self, row, state, cb):
        cb.pending = {
            "id": int(row.val),
            "period": 0,
            "mode": 0,
            "base_note": 0,
            "abs": [],
            "off": [],
            "resid": [],
            "_lo": None,
        }

    def step(self, state, row, cb):
        gen = cb.pending
        if gen is None:
            return None
        sub = int(row.subreg)
        val = int(row.val)
        if sub == GEN_TABLE_SUBREG_PERIOD:
            gen["period"] = val
        elif sub == GEN_TABLE_SUBREG_MODE:
            gen["mode"] = val
        elif sub == GEN_TABLE_SUBREG_BASE_NOTE:
            gen["base_note"] = val
        elif sub == GEN_TABLE_SUBREG_ABS_LO:
            gen["_lo"] = val & 0xFF
        elif sub == GEN_TABLE_SUBREG_ABS_HI:
            gen["abs"].append(((val & 0xFF) << 8) | (gen["_lo"] or 0))
            gen["_lo"] = None
        elif sub == GEN_TABLE_SUBREG_OFFSET:
            v = val & 0xFF
            gen["off"].append(v - 256 if v >= 128 else v)
        elif sub == GEN_TABLE_SUBREG_RESID_LO:
            gen["_lo"] = val & 0xFF
        elif sub == GEN_TABLE_SUBREG_RESID_HI:
            raw = ((val & 0xFF) << 8) | (gen["_lo"] or 0)
            gen["resid"].append(raw - 0x10000 if raw >= 0x8000 else raw)
            gen["_lo"] = None
        return None

    def commit(self, state, row, cb):
        gen = cb.pending
        if gen is not None:
            cb.table[int(gen["id"])] = gen
            cb.pending = None
        return None

    def ref(self, state, row, cb):
        sub = int(row.subreg)
        val = int(row.val)
        if sub == GEN_TABLE_REF_SUBREG_ID:
            cb.pending_ref = {
                "id": val,
                "reg": int(row.reg),
                "base_note": 0,
                "len": 0,
                "resid": [],
                "_rlo": None,
            }
            return None
        pend = cb.pending_ref
        if pend is None:
            return None
        if sub == GEN_TABLE_REF_SUBREG_BASE_NOTE:
            pend["base_note"] = val
            return None
        if sub == GEN_TABLE_REF_SUBREG_RESID_LO:
            pend["_rlo"] = val & 0xFF
            return None
        if sub == GEN_TABLE_REF_SUBREG_RESID_HI:
            raw = ((val & 0xFF) << 8) | (pend["_rlo"] or 0)
            pend["resid"].append(raw - 0x10000 if raw >= 0x8000 else raw)
            pend["_rlo"] = None
            return None
        if sub == GEN_TABLE_REF_SUBREG_LEN_HI:
            pend["len"] |= (val & 0xFF) << 8
            return None
        if sub != GEN_TABLE_REF_SUBREG_LEN_LO:
            return None
        pend["len"] |= val & 0xFF
        cb.pending_ref = None
        return self._replay(pend, state, cb)

    @staticmethod
    def _replay(pend, state, cb):
        gen = cb.table.get(int(pend["id"]))
        if gen is None:
            return None
        period = int(gen["period"])
        if period <= 0:
            return None
        reg = int(pend["reg"])
        ref = float(getattr(state, "gen_ref", 0.0))
        base = int(pend["base_note"])
        if int(gen["mode"]) == GEN_TABLE_MODE_NOTE_UNIV and not 0 <= base <= 255:
            return None
        resid = pend["resid"] if pend.get("resid") else gen["resid"]
        pre = state.maybe_flush_for(reg, -1)
        queue = state.pending_set_writes[reg]
        mode = int(gen["mode"])
        length = int(pend["len"])
        if mode == GEN_TABLE_MODE_NOTE_UNIV:
            voice = reg // 7
            tuning = getattr(state, "gen_ref_by_voice", {}).get(voice, 0.0)
            tbl = getattr(state, "gen_table_by_voice", {}).get(voice, {})
            cyc = [
                (
                    (
                        tbl[base + gen["off"][m]]
                        if (base + gen["off"][m]) in tbl
                        else pitch_grid.note_freq_at(base + gen["off"][m], tuning)
                    )
                    + resid[m]
                )
                & 0xFFFF
                for m in range(period)
            ]
        elif mode == GEN_TABLE_MODE_NOTE:
            cyc = [
                (recon(base + gen["off"][m], ref) + resid[m]) & 0xFFFF
                for m in range(period)
            ]
        else:
            cyc = [int(x) for x in gen["abs"]]
        for k in range(length):
            queue.append(cyc[k % period])
        return pre or None


def _s16(lo, hi):
    """Signed 16-bit two's-complement from a (lo, hi) byte pair -- the consistent field encoding the
    gesture REF/STEP value pairs use for anchors, initial diffs, and cell deltas."""
    raw = ((int(hi) & 0xFF) << 8) | (int(lo) & 0xFF)
    return raw - 0x10000 if raw >= 0x8000 else raw


def gesture_value_series(shape, anchor, diffs, length):
    """Replay one gesture SHAPE into its ``length``-frame value series (no note layer, no channel wrap).
    The inverse of the encoder's gesture split: HOLD repeats the anchor; POLY forward-differences the
    initial difference table ``[anchor, *diffs, N-th-diff]``; PERIOD walks the looped delta cell from the
    anchor. Pure integer arithmetic -- byte-exact by construction (see mdl_codec.decode).
    """
    kind = int(shape["kind"])
    out = []
    if kind == GESTURE_KIND_HOLD:
        out = [int(anchor)] * int(length)
    elif kind == GESTURE_KIND_POLY:
        degree = int(shape["degree"])
        dt = (
            [int(anchor)]
            + [int(d) for d in diffs[: degree - 1]]
            + [int(shape["cell"][0])]
        )
        for _ in range(int(length)):
            out.append(dt[0])
            for k in range(degree):
                dt[k] += dt[k + 1]
    elif kind == GESTURE_KIND_PERIOD:
        period = int(shape["degree"])
        cell = [int(c) for c in shape["cell"]]
        cur = int(anchor)
        for k in range(int(length)):
            if k:
                cur = cur + cell[(k - 1) % period]
            out.append(cur)
    return out


def _gesture_note_base(state, voice):
    """The freq a gesture REF on a freq voice rides: the voice's current note (set by NOTE_INTERVAL)
    mapped through the recovered note->freq table, the per-voice tuning grid, or the global ref grid --
    the SAME base the encoder subtracts to form the freq-delta, so ``base + delta`` is byte-exact.
    """
    cur_note = getattr(state, "gesture_cur_note", {}).get(voice)
    if cur_note is None:
        return 0
    tbl = state.gen_table_by_voice.get(voice)
    if tbl is not None and cur_note in tbl:
        return int(tbl[cur_note])
    if voice in state.gen_ref_by_voice:
        return int(pitch_grid.note_freq_at(cur_note, state.gen_ref_by_voice[voice]))
    return int(recon(cur_note, state.gen_ref)) & 0xFFFF


class _GestureCodec(_Codec):
    """MDL gesture codebook (subsumes GENERATOR + INSTRUMENT): DEF/STEP/END buffer one reusable SHAPE
    (HOLD / POLY(N) forward-difference / PERIOD delta-cell) into the id table; the fixed-layout REF
    replays it with a per-instance anchor + lower-order initial diffs + length, queuing one value per
    frame onto its reg. A freq REG additionally rides the note layer: ``freq = note_table[note_index] +
    delta`` with a 16-bit value wrap (set by NOTE_INTERVAL; absent => the series is written directly).
    """

    def def_open(self, row, state, cb):
        cb.pending = {
            "id": int(row.val),
            "kind": GESTURE_KIND_HOLD,
            "degree": 0,
            "cell": [],
            "_lo": None,
        }

    def step(self, state, row, cb):
        g = cb.pending
        if g is None:
            return None
        sub = int(row.subreg)
        val = int(row.val)
        if sub == GESTURE_SUBREG_KIND:
            g["kind"] = val
        elif sub == GESTURE_SUBREG_DEGREE:
            g["degree"] = val
        elif sub == GESTURE_SUBREG_CELL_LO:
            g["_lo"] = val & 0xFF
        elif sub == GESTURE_SUBREG_CELL_HI:
            g["cell"].append(_s16(g["_lo"] or 0, val))
            g["_lo"] = None
        return None

    def commit(self, state, row, cb):
        g = cb.pending
        if g is not None:
            cb.table[int(g["id"])] = g
            cb.pending = None
        return None

    def ref(self, state, row, cb):
        sub = int(row.subreg)
        val = int(row.val)
        if sub == GESTURE_REF_SUBREG_ID:
            cb.pending_ref = {
                "id": val,
                "reg": int(row.reg),
                "anchor_lo": 0,
                "d1_lo": 0,
                "d2_lo": 0,
                "anchor": 0,
                "d1": 0,
                "d2": 0,
                "len": 0,
            }
            return None
        pend = cb.pending_ref
        if pend is None:
            return None
        if sub == GESTURE_REF_SUBREG_ANCHOR_LO:
            pend["anchor_lo"] = val & 0xFF
        elif sub == GESTURE_REF_SUBREG_ANCHOR_HI:
            pend["anchor"] = _s16(pend["anchor_lo"], val)
        elif sub == GESTURE_REF_SUBREG_D1_LO:
            pend["d1_lo"] = val & 0xFF
        elif sub == GESTURE_REF_SUBREG_D1_HI:
            pend["d1"] = _s16(pend["d1_lo"], val)
        elif sub == GESTURE_REF_SUBREG_D2_LO:
            pend["d2_lo"] = val & 0xFF
        elif sub == GESTURE_REF_SUBREG_D2_HI:
            pend["d2"] = _s16(pend["d2_lo"], val)
        elif sub == GESTURE_REF_SUBREG_LEN_HI:
            pend["len"] |= (val & 0xFF) << 8
        elif sub == GESTURE_REF_SUBREG_LEN_LO:
            pend["len"] |= val & 0xFF
            cb.pending_ref = None
            return self._replay(pend, state, cb)
        return None

    @staticmethod
    def _replay(pend, state, cb):
        shape = cb.table.get(int(pend["id"]))
        if shape is None:
            return None
        reg = int(pend["reg"])
        length = int(pend["len"])
        series = gesture_value_series(
            shape, pend["anchor"], (pend["d1"], pend["d2"]), length
        )
        pre = state.maybe_flush_for(reg, -1)
        queue = state.pending_set_writes[reg]
        if reg in GEN_FREQ_REGS:
            base = _gesture_note_base(state, reg // 7)
            for delta in series:
                queue.append((base + int(delta)) & 0xFFFF)
        else:
            for v in series:
                queue.append(int(v))
        return pre or None


_CODECS: dict[str, _Codec] = {
    "instrument": _InstrumentCodec(),
    "generator": _GeneratorCodec(),
    "gesture": _GestureCodec(),
}


class CodebookDecoder:
    """One decoder, instantiated per family, registered for every op the family owns. Routes a row to
    its lifecycle phase from the registry (def opens a pending entry, step/commit make the id live,
    ref replays it) and delegates the payload codec against the family's ``_Codebook``, so all
    families share one machine."""

    op_code = -1

    def __init__(self, family: CodebookFamily, codec: _Codec):
        self.family = family
        self.codec = codec
        self.table_index = family.table_index
        self._ref_ops = frozenset(r.op for r in family.refs)
        self._step_ops = frozenset(family.step_ops)

    def expand(self, row, state):
        op = int(row.op)
        fam = self.family
        cb = state.codebooks[self.table_index]
        if op == fam.def_op:
            self.codec.def_open(row, state, cb)
            return None
        if op in self._ref_ops:
            return self.codec.ref(state, row, cb)
        if op in self._step_ops:
            return self.codec.step(state, row, cb)
        if op == fam.commit_op:
            return self.codec.commit(state, row, cb)
        return None


def codebook_decoders() -> dict[int, CodebookDecoder]:
    """``op -> CodebookDecoder`` for every codebook op, ready to merge into ``decoders.DECODERS``. One
    decoder instance per family is shared across that family's ops, matching the legacy registration.
    """
    out: dict[int, CodebookDecoder] = {}
    for fam in CODEBOOK_FAMILIES.values():
        dec = CodebookDecoder(fam, _CODECS[fam.name])
        for op in fam.ops:
            out[op] = dec
    return out
