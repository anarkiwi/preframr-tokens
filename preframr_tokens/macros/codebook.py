"""Single source of truth for the inline-codebook families (STAMP, PATCH, WAVETABLE, CTRL_WT).
Each shares one machine: DEF opens a pending entry, STEP(s) accumulate its payload, a COMMIT
(an END op or a STEP at a terminal subreg) makes the id live in an ``id -> entry`` table, and
REF(s) replay a live id. This leaf (stfconstants only) declares that as ``CodebookFamily``
records and derives the ``op_contracts.CODEBOOK_SPECS`` table the registry test pins equal.
"""

from __future__ import annotations

from dataclasses import dataclass

from preframr_tokens.macros.wavetable import unroll as wt_unroll
from preframr_tokens.stfconstants import (
    CTRL_WT_DEF_OP,
    CTRL_WT_SET_OP,
    CTRL_WT_STEP_OP,
    CTRL_WT_SUBREG_VAL,
    PATCH_AD_OFFSET,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_SR_OFFSET,
    PATCH_STEP_OP,
    PATCH_SUBREG_AD,
    PATCH_SUBREG_SR,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    STAMP_REL_SUBREG_BASE_HI,
    STAMP_REL_SUBREG_BASE_LO,
    STAMP_REL_SUBREG_ID,
    STAMP_STEP_OP,
    STAMP_SUBREG_FRAME,
    WAVETABLE_DEF_OP,
    WAVETABLE_END_OP,
    WAVETABLE_ONESHOT_OP,
    WAVETABLE_REF_OP,
    WAVETABLE_STEP_OP,
    WT_ONESHOT_SUBREG_HOLD,
    WT_ONESHOT_SUBREG_LEN_HI,
    WT_ONESHOT_SUBREG_LEN_LO,
    WT_ONESHOT_SUBREG_OFFSET,
    WT_REF_SUBREG_ID,
    WT_REF_SUBREG_LEAD,
    WT_REF_SUBREG_LEADOFF,
    WT_REF_SUBREG_LEN_HI,
    WT_REF_SUBREG_LEN_LO,
    WT_STEP_SUBREG_HOLD,
    WT_STEP_SUBREG_LOOP,
    WT_STEP_SUBREG_OFFSET,
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

CODEBOOK_TABLE_NAMES: tuple[str, ...] = ("stamp", "patch", "wavetable", "ctrl_wt")


@dataclass(frozen=True)
class RefSpec:
    """One REF op of a family. ``id_subreg`` is the subreg carrying the codebook id (a
    multi-row ref like WAVETABLE_REF/STAMP_REL_REF) or ``None`` when the op's own row carries
    it (a single-row ref like PATCH_SET/CTRL_WT_SET/STAMP_REF). ``table_less`` marks a ref that
    carries its payload inline and looks up no id (WAVETABLE_ONESHOT); such ops are not
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
    STEP subreg that triggers the commit (PATCH/CTRL_WT)."""

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
    "stamp": CodebookFamily(
        name="stamp",
        def_op=STAMP_DEF_OP,
        step_ops=(STAMP_STEP_OP,),
        commit_op=STAMP_END_OP,
        commit_subreg=None,
        refs=(
            RefSpec(STAMP_REF_OP),
            RefSpec(STAMP_REL_REF_OP, id_subreg=STAMP_REL_SUBREG_ID),
        ),
        def_emits=False,
    ),
    "patch": CodebookFamily(
        name="patch",
        def_op=PATCH_DEF_OP,
        step_ops=(PATCH_STEP_OP,),
        commit_op=PATCH_STEP_OP,
        commit_subreg=PATCH_SUBREG_SR,
        refs=(RefSpec(PATCH_SET_OP),),
        def_emits=True,
    ),
    "wavetable": CodebookFamily(
        name="wavetable",
        def_op=WAVETABLE_DEF_OP,
        step_ops=(WAVETABLE_STEP_OP,),
        commit_op=WAVETABLE_END_OP,
        commit_subreg=None,
        refs=(
            RefSpec(WAVETABLE_REF_OP, id_subreg=WT_REF_SUBREG_ID),
            RefSpec(WAVETABLE_ONESHOT_OP, table_less=True),
        ),
        def_emits=False,
    ),
    "ctrl_wt": CodebookFamily(
        name="ctrl_wt",
        def_op=CTRL_WT_DEF_OP,
        step_ops=(CTRL_WT_STEP_OP,),
        commit_op=CTRL_WT_STEP_OP,
        commit_subreg=CTRL_WT_SUBREG_VAL,
        refs=(RefSpec(CTRL_WT_SET_OP),),
        def_emits=True,
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


def _skel_lut():
    """The skeleton freq LUT, imported lazily to break the codebook->skeleton_pass->passes_base->
    state->codebook import cycle (this leaf must load before state finishes initialising).
    """
    # pylint: disable=import-outside-toplevel
    from preframr_tokens.macros.skeleton_pass import LUT

    return LUT


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


class _StampCodec(_Codec):
    """Voice-relative percussion stamp: DEF/STEP/END buffer a write-series into the id table, REF
    replays it on the target voice and REL_REF additionally adds the per-hit base delta at offset 0.
    """

    def def_open(self, row, state, cb):
        cb.pending = {"id": int(row.val), "frames": [[]]}

    def step(self, state, row, cb):
        stamp = cb.pending
        if stamp is None:
            return None
        if int(row.subreg) == STAMP_SUBREG_FRAME:
            stamp["frames"].append([])
        else:
            stamp["frames"][-1].append((int(row.subreg), int(row.val)))
        return None

    def commit(self, state, row, cb):
        stamp = cb.pending
        if stamp is not None:
            cb.table[int(stamp["id"])] = stamp["frames"]
            cb.pending = None
        return None

    def ref(self, state, row, cb):
        if int(row.op) == STAMP_REL_REF_OP:
            return self._rel_ref(row, state, cb)
        return self._ref(row, state, cb)

    @staticmethod
    def _offsets_in_order(frames):
        """Voice-relative offsets in first-write order across the buffered frames -- preserves the
        drum's intra-frame freq<->ctrl order (the per-frame drain emits regs in insertion order).
        """
        offsets, seen = [], set()
        for fr in frames:
            for off, _val in fr:
                if off not in seen:
                    seen.add(off)
                    offsets.append(off)
        return offsets

    @classmethod
    def _ref(cls, row, state, cb):
        frames = cb.table.get(int(row.val))
        if not frames:
            return None
        base = int(row.reg)
        offsets = cls._offsets_in_order(frames)
        pre = state.maybe_flush_for(base, -1)
        cur = {}
        for fr in frames:
            for off, val in fr:
                cur[off] = val
            for off in offsets:
                if off in cur:
                    state.pending_set_writes[base + off].append(int(cur[off]))
        return pre or None

    def _rel_ref(self, row, state, cb):
        subreg = int(row.subreg)
        if subreg == STAMP_REL_SUBREG_ID:
            cb.pending_ref = {
                "id": int(row.val),
                "reg": int(row.reg),
                "base": 0,
            }
            return None
        pend = cb.pending_ref
        if pend is None:
            return None
        if subreg == STAMP_REL_SUBREG_BASE_HI:
            pend["base"] |= (int(row.val) & 0xFF) << 8
            return None
        if subreg != STAMP_REL_SUBREG_BASE_LO:
            return None
        pend["base"] |= int(row.val) & 0xFF
        cb.pending_ref = None
        return self._replay_rel(pend, state, cb)

    @classmethod
    def _replay_rel(cls, pend, state, cb):
        frames = cb.table.get(int(pend["id"]))
        if not frames:
            return None
        base = int(pend["base"])
        voice = int(pend["reg"])
        offsets = cls._offsets_in_order(frames)
        pre = state.maybe_flush_for(voice, -1)
        cur = {}
        for fr in frames:
            for off, val in fr:
                cur[off] = val
            for off in offsets:
                if off not in cur:
                    continue
                val = int(cur[off])
                if off == 0:
                    signed = val if val < 0x8000 else val - 0x10000
                    val = (base + signed) & 0xFFFF
                state.pending_set_writes[voice + off].append(val)
        return pre or None


class _PatchCodec(_Codec):
    """Melodic-instrument patch: DEF + two STEP rows buffer an (AD,SR) envelope into the id table and
    emit it on the def's voice, SET re-emits a defined patch's AD/SR on the ref's voice.
    """

    def def_open(self, row, state, cb):
        cb.pending = {
            "id": int(row.val),
            "freq_reg": int(row.reg),
            "ad": None,
            "sr": None,
        }

    def step(self, state, row, cb):
        pend = cb.pending
        if pend is None:
            return None
        if int(row.subreg) == PATCH_SUBREG_AD:
            pend["ad"] = int(row.val)
        elif int(row.subreg) == PATCH_SUBREG_SR:
            pend["sr"] = int(row.val)
        if pend["ad"] is None or pend["sr"] is None:
            return None
        cb.pending = None
        cb.table[int(pend["id"])] = (int(pend["ad"]), int(pend["sr"]))
        return self._emit(int(pend["freq_reg"]), pend["ad"], pend["sr"], row, state)

    def ref(self, state, row, cb):
        patch = cb.table.get(int(row.val))
        if patch is None:
            return None
        return self._emit(int(row.reg), patch[0], patch[1], row, state)

    @staticmethod
    def _emit(freq_reg, ad, sr, row, state):
        writes = []
        for reg, val in (
            (freq_reg + PATCH_AD_OFFSET, int(ad)),
            (freq_reg + PATCH_SR_OFFSET, int(sr)),
        ):
            writes.extend(state.maybe_flush_for(reg, -1))
            state.last_val[reg] = val
            state.last_diff[reg] = row.diff
            writes.append((reg, val, row.diff, row.description))
        return writes


class _CtrlWtCodec(_Codec):
    """Inline ctrl-state codebook: DEF + STEP buffer a single ctrl byte into the id table and emit it
    on the def's voice, SET re-emits the defined byte on the ref's voice."""

    def def_open(self, row, state, cb):
        cb.pending = {"id": int(row.val), "reg": int(row.reg)}

    def step(self, state, row, cb):
        pend = cb.pending
        if pend is None or int(row.subreg) != CTRL_WT_SUBREG_VAL:
            return None
        cb.pending = None
        val = int(row.val)
        cb.table[int(pend["id"])] = val
        return self._emit(int(pend["reg"]), val, row, state)

    def ref(self, state, row, cb):
        val = cb.table.get(int(row.val))
        if val is None:
            return None
        return self._emit(int(row.reg), int(val), row, state)

    @staticmethod
    def _emit(reg, val, row, state):
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(val), row.diff, row.description)]


class _WavetableCodec(_Codec):
    """Note-relative wavetable codebook: DEF/STEP/END buffer an RLE offset program into the id table,
    REF unrolls it to LEN frames after the per-hit LEAD, ONESHOT carries an inline program with no id;
    both queue base + LUT[note+off] per frame onto last_skel_note via pending_set_writes.
    """

    def def_open(self, row, state, cb):
        cb.pending = {"id": int(row.val), "steps": [], "loop": 0}

    def step(self, state, row, cb):
        wt = cb.pending
        if wt is None:
            return None
        subreg = int(row.subreg)
        if subreg == WT_STEP_SUBREG_OFFSET:
            v = int(row.val) & 0xFF
            wt["steps"].append([v if v < 128 else v - 256, 1])
        elif subreg == WT_STEP_SUBREG_HOLD:
            if wt["steps"]:
                wt["steps"][-1][1] = int(row.val) & 0xFFFF
        elif subreg == WT_STEP_SUBREG_LOOP:
            wt["loop"] = int(row.val) & 0xFFFF
        return None

    def commit(self, state, row, cb):
        wt = cb.pending
        if wt is not None:
            cb.table[int(wt["id"])] = (wt["steps"], int(wt["loop"]))
            cb.pending = None
        return None

    def ref(self, state, row, cb):
        if int(row.op) == WAVETABLE_ONESHOT_OP:
            return self._oneshot(row, state, cb)
        return self._wt_ref(row, state, cb)

    def _oneshot(self, row, state, cb):
        subreg = int(row.subreg)
        if subreg == WT_ONESHOT_SUBREG_LEN_HI:
            cb.pending_ref = {
                "reg": int(row.reg),
                "len": (int(row.val) & 0xFF) << 8,
                "steps": [],
            }
            return None
        pend = cb.pending_ref
        if pend is None:
            return None
        if subreg == WT_ONESHOT_SUBREG_LEN_LO:
            pend["len"] |= int(row.val) & 0xFF
            return None
        if subreg == WT_ONESHOT_SUBREG_OFFSET:
            v = int(row.val) & 0xFF
            pend["steps"].append([v if v < 128 else v - 256, 1])
            return None
        if subreg == WT_ONESHOT_SUBREG_HOLD:
            if pend["steps"]:
                pend["steps"][-1][1] = int(row.val) & 0xFFFF
            return None
        return self._replay_oneshot(pend, state, cb)

    @staticmethod
    def _replay_oneshot(pend, state, cb):
        cb.pending_ref = None
        steps = pend["steps"]
        reg = int(pend["reg"])
        note = int(state.last_skel_note.get(reg, 0))
        offsets = wt_unroll(steps, len(steps), int(pend["len"]), [])
        lut = _skel_lut()
        queue = state.pending_set_writes[reg]
        queue.append(int(lut[max(0, min(127, note))]))
        for off in offsets:
            queue.append(int(lut[max(0, min(127, note + int(off)))]))
        return None

    def _wt_ref(self, row, state, cb):
        subreg = int(row.subreg)
        if subreg == WT_REF_SUBREG_ID:
            cb.pending_ref = {
                "id": int(row.val),
                "reg": int(row.reg),
                "len": 0,
                "lead": [],
                "lead_n": 0,
            }
            return None
        pend = cb.pending_ref
        if pend is None:
            return None
        if subreg == WT_REF_SUBREG_LEN_HI:
            pend["len"] |= (int(row.val) & 0xFF) << 8
            return None
        if subreg == WT_REF_SUBREG_LEN_LO:
            pend["len"] |= int(row.val) & 0xFF
            return None
        if subreg == WT_REF_SUBREG_LEAD:
            pend["lead_n"] = int(row.val) & 0xFFFF
            if pend["lead_n"] == 0:
                return self._replay(pend, state, cb)
            return None
        if subreg == WT_REF_SUBREG_LEADOFF:
            v = int(row.val) & 0xFF
            pend["lead"].append(v if v < 128 else v - 256)
            if len(pend["lead"]) >= pend["lead_n"]:
                return self._replay(pend, state, cb)
            return None
        return None

    @staticmethod
    def _replay(pend, state, cb):
        cb.pending_ref = None
        program = cb.table.get(int(pend["id"]))
        if program is None:
            return None
        steps, loop = program
        reg = int(pend["reg"])
        note = int(state.last_skel_note.get(reg, 0))
        offsets = wt_unroll(steps, loop, int(pend["len"]), pend["lead"])
        lut = _skel_lut()
        queue = state.pending_set_writes[reg]
        queue.append(int(lut[max(0, min(127, note))]))
        for off in offsets:
            queue.append(int(lut[max(0, min(127, note + int(off)))]))
        return None


_CODECS: dict[str, _Codec] = {
    "stamp": _StampCodec(),
    "patch": _PatchCodec(),
    "wavetable": _WavetableCodec(),
    "ctrl_wt": _CtrlWtCodec(),
}


class CodebookDecoder:
    """One decoder, instantiated per family, registered for every op the family owns. Routes a row to
    its lifecycle phase from the registry (def opens a pending entry, step/commit make the id live,
    ref replays it) and delegates the payload codec against the family's ``_Codebook``, so all four
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
