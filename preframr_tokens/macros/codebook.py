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
        # Precompute the period's output values ONCE; the frame loop then just indexes the
        # cycle. The notes repeat every `period` frames, so computing note_freq_at/recon per
        # replayed frame (and re-running that under the arbiter's validate=True re-decode) is an
        # O(len x cost) blowup -- this hoists it to O(period). Byte-exact (identical values).
        if mode == GEN_TABLE_MODE_NOTE_UNIV:
            voice = reg // 7
            tuning = getattr(state, "gen_ref_by_voice", {}).get(voice, 0.0)
            cyc = [
                (pitch_grid.note_freq_at(base + gen["off"][m], tuning) + resid[m]) & 0xFFFF
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


_CODECS: dict[str, _Codec] = {
    "instrument": _InstrumentCodec(),
    "generator": _GeneratorCodec(),
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
