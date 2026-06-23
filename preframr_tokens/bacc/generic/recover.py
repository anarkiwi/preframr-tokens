"""Driver-agnostic generic recovery -> :class:`BaccProgram`, byte-exact.

``recover_generic(sid, dump, bustrace)`` reconstructs the per-frame 25-register
state from the trusted preframr-sidtrace ``.bus.bin`` (blit-aware, boot-prolog
aligned to the dump's first play cycle), then fits a GENERIC per-register program
with NO per-driver constants (HARD RULE #0): the freq/pw generator lanes via the
BACC archetype library, the ctrl/AD/SR/filter/volume lanes as a compact piecewise
program, and the freq note table from bus value-provenance.

The recovered program is carried in a :class:`BaccProgram` (``driver="generic"``)
and rendered back by the SELF-CONTAINED :func:`render_generic` -- no hand backend
is consulted.  :func:`residual` reports per-register residual frame counts against
the bus-state; on the proven 8/8 corpus the program renders whole-tune
residual-zero (the Hammurabi generator-lane gap is closed by the generic
tablewalk_lead archetype, ratewalk adds the fractional-rate / wider-internal-width
sweep, and the advance-clocked wavetable_ptr closes FamiCommodore's voice-2 PW
groove-paced reflecting triangle).  Any genuinely irreducible lane would be
surfaced as residual rather than faked with a per-step-storage cover.
"""

import numpy as np

from preframr_tokens.bacc.generic import archetypes as A
from preframr_tokens.bacc.generic import fitter as F
from preframr_tokens.bacc.generic.bustrace import load_bus
from preframr_tokens.bacc.generic.busstate import (
    NREG,
    dump_first_play_cycle,
    per_frame_state_from_bus,
)
from preframr_tokens.bacc.primitive import BaccProgram

DRIVER = "generic"


def recover_generic(sid, dump, bustrace, t0=None):
    """Recover a generic :class:`BaccProgram` from a tune's trusted bus trace.

    ``sid`` is retained for provenance only (the recovery reads the bus trace,
    not the playroutine).  ``dump`` supplies the frame-0 anchor (the dump defines
    the grid the bus-state must reproduce); when ``dump`` is None the bus's own
    first-play cycle is used.  ``bustrace`` is a native preframr-sidtrace
    ``.bus.bin`` path or a pre-loaded ``BUS_DT`` record array.  ``t0`` overrides
    the anchor with a known frame-0 cycle -- the sid-only path passes the
    ``.sidwr.bin`` dump's first-play cycle so the bus-state is framed onto the
    SAME grid as the in-process dump (``recover_from_sid``).

    The returned program's ``tables`` carry the per-register fitted programs and
    the bus-recovered note table; ``boot`` is the frame-0 register seed.
    """
    records = bustrace if isinstance(bustrace, np.ndarray) else load_bus(bustrace)
    if t0 is None and dump is not None:
        t0 = dump_first_play_cycle(dump)
    state, _, cpf = per_frame_state_from_bus(records, t0=t0)
    if state is None or len(state) < 2:
        raise ValueError(f"bus trace did not parse to frames: {bustrace}")
    note_table = F.discover_note_table_from_bus(records)
    nt_arr = np.array(note_table, dtype=np.int64) if note_table else None
    _, _, _, archtally, genfits, eventfits = F.fit_full_tune(state, nt_arr)
    program = BaccProgram(
        driver=DRIVER,
        nframes=int(len(state)),
        boot=list(int(v) for v in state[0]),
        instruments=[],
        score=[],
        seed={"sid": str(sid), "cpf": int(cpf) if cpf is not None else None},
        tables={
            "note_table": note_table,
            "genfits": _serialize_genfits(genfits),
            "eventfits": _serialize_eventfits(eventfits),
            "archetypes": archtally,
        },
    )
    return program


def recover_from_sid(
    sid_path, subtune=1, nframes=200, sidtrace_path=None, out_prefix=None
):
    """Recover a generic :class:`BaccProgram` from a ``.sid`` ALONE.

    The two-file (``.sid`` + ``.dump.parquet``) input collapses to a single file:
    one deterministic ``preframr-sidtrace`` run generates BOTH the per-frame
    register dump and the bus trace in-process (no pre-rendered dump), the GENERIC
    recovery fits the program from the bus trace, and the render is verified
    residual-zero against the SAME-run dump (the two are internally
    self-consistent -- boot-prolog/tail divergences vs VICE are irrelevant).

    Returns ``(program, resid, dump_state)`` where ``program`` is the recovered
    ``driver="generic"`` :class:`BaccProgram`, ``resid`` maps register -> residual
    frame count against the in-process dump (``sum == 0`` is whole-tune,
    all-25-register byte-exact), and ``dump_state`` is the generated dump.

    Raises :class:`FileNotFoundError` when no ``preframr-sidtrace`` binary is
    available (``SIDTRACE_BIN``); the default render-free CI never invokes it.
    """
    from preframr_tokens.bacc.generic.sidtrace import (  # local: optional binary dep
        sid_to_dump_and_bustrace,
    )

    dump_state, bus, t0, _distill_path = sid_to_dump_and_bustrace(
        sid_path, subtune, nframes, sidtrace_path, out_prefix
    )
    if dump_state is None or len(dump_state) < 2:
        raise ValueError(f"sidtrace produced no frames for {sid_path}")
    program = recover_generic(sid_path, None, bus, t0=t0)
    rendered = render_generic(program)
    nf = min(len(rendered), len(dump_state))
    resid = {
        reg: int(np.sum(rendered[:nf, reg] != dump_state[:nf, reg]))
        for reg in range(NREG)
    }
    return program, resid, dump_state[:nf]


def structure_ids_from_sid(
    sid_path, subtune=1, nframes=200, sidtrace_path=None, out_prefix=None
):
    """Recover and serialize a structured tune's tracker source from a ``.sid`` ALONE.

    The FORK of the generic recovery (the structure path).  One deterministic
    ``preframr-sidtrace`` run emits the per-frame register state (``.sidwr.bin``) AND the
    SMC-correct distill artifact (``.distill.bin``); :func:`structure_ir.recover_structure_ir`
    recovers the tracker STRUCTURE -- a deduped instrument pool, the factored
    patterns/orderlist, and the porta/vibrato accumulator generators -- DIRECTLY from the
    artifact (no hardcoded addresses).  When a valid structure is found (``ok``) the compact
    structure serialization is returned; otherwise the tune is unstructured (a pure-code tune
    like A Mind Is Born, or a driver whose pattern grammar the round-trip falsifies) and the
    caller FALLS BACK to the output-fit generator cover (:func:`generic_program_to_ids`) --
    the additive-fix invariant.

    Returns ``(ids, structure, state)`` where ``ids`` is the structure token stream (or
    ``None`` when no structure was found), ``structure`` is the :class:`StructureIR` (or
    ``None``), and ``state`` is the byte-exact ``(nframes, 25)`` register array.  Raises
    :class:`FileNotFoundError` when no ``preframr-sidtrace`` binary is available."""
    import os
    import tempfile

    from preframr_tokens.bacc.generic.sidtrace import (  # local: optional binary dep
        run_sidtrace,
        sidwr_state,
    )
    from preframr_tokens.bacc.generic.structure_ir import (
        recover_structure_ir,
        structure_ir_to_ids,
    )

    if out_prefix is None:
        out_prefix = os.path.join(
            tempfile.mkdtemp(prefix="preframr_sidtrace_"), "trace"
        )
    sidwr_path, distill_path = run_sidtrace(
        sid_path, out_prefix, subtune, nframes, sidtrace_path
    )
    state, _ = sidwr_state(sidwr_path)
    if state is None or len(state) < 2:
        raise ValueError(f"sidtrace produced no frames for {sid_path}")
    structure = recover_structure_ir(distill_path, state)
    ids = None if structure is None else structure_ir_to_ids(structure)
    return ids, structure, state


def _cover_ids(state):
    """The generator-cover token stream (the table-less fallback path): lift the
    byte-exact state to the Tracker IR and serialize it.  Reached only when the
    structure path returns no structure or is not the smaller byte-exact cover."""
    from preframr_tokens.bacc.tracker_ir import lift
    from preframr_tokens.bacc.tracker_serialize import _ir_to_ids

    nframes = len(state)
    boot = [int(v) for v in state[0]]
    return _ir_to_ids(lift(state, None, nframes, boot))


def recover_tune(
    sid_path, subtune=1, nframes=2500, sidtrace_path=None, out_prefix=None
):
    """THE single public generic-recovery entry: ``.sid`` -> ``(ids, kind, state)``.

    Runs the S0-S7 structure recovery (:func:`structure_ir.recover_structure_ir`,
    byte-exact + value-LZ'd) and the generator cover (the table-less fallback), and
    returns the SMALLER byte-exact token stream (the design's fewest-tokens tiebreak):
    a structured tracker tune serializes from its recovered source (note table +
    deduped instruments + factored patterns/orderlist + fitted freq accumulators); a
    pure-code / table-less tune (A Mind Is Born) falls back to the generator cover.
    ``kind`` is ``"structure"`` or ``"cover"``.  This lifts the structure-vs-cover fork
    -- previously living only in tests/tools -- into ONE documented library entry.

    Returns ``(ids, kind, state)`` with ``state`` the byte-exact ``(nframes, 25)``
    register array.  Raises :class:`FileNotFoundError` when no ``preframr-sidtrace``
    binary is available."""
    ids, structure, state = structure_ids_from_sid(
        sid_path, subtune, nframes, sidtrace_path, out_prefix
    )
    if structure is not None and ids is not None and len(ids) < len(state):
        return ids, "structure", state
    return _cover_ids(state), "cover", state


def render_generic(program):
    """Render a generic :class:`BaccProgram` back to an ``(nframes, 25)`` register
    array, SELF-CONTAINED (no hand backend).  Re-runs each fitted archetype
    program per register, exactly as the bus-state was covered.
    """
    nframes = program.nframes
    note_table = program.tables.get("note_table")
    nt_arr = np.array(note_table, dtype=np.int64) if note_table else None
    genfits = _deserialize_genfits(program.tables["genfits"])
    eventfits = _deserialize_eventfits(program.tables["eventfits"])
    rendered = np.zeros((nframes, NREG), dtype=np.int64)

    for voice in range(3):
        flo, fhi = F.FREQ_LO[voice], F.FREQ_HI[voice]
        plo, phi = F.PW_LO[voice], F.PW_HI[voice]
        fres, _ = genfits[(voice, "freq")]
        pres, carry = genfits[(voice, "pw")]
        flane, _ = F.render_generator_lane(fres, nframes, nt_arr, None)
        plane, _ = F.render_generator_lane(pres, nframes, nt_arr, carry)
        rendered[:, flo] = flane & 0xFF
        rendered[:, fhi] = (flane >> 8) & 0xFF
        rendered[:, plo] = plane & 0xFF
        rendered[:, phi] = (plane >> 8) & 0x0F

    for reg, segs in eventfits.items():
        rendered[:, reg] = A.render_event_lane(segs, nframes)
    return rendered


def residual(program, bustrace, dump=None, t0=None):
    """Per-register residual frame counts of :func:`render_generic` against the
    bus-state ``program`` was recovered from.  Returns ``(resid, rendered,
    state)`` where ``resid`` maps register -> residual frame count (0 = byte-exact
    for that register).  ``sum(resid.values()) == 0`` is whole-tune residual-zero.

    ``t0`` (the sid-only ``.sidwr.bin`` dump anchor) overrides the parquet
    ``dump`` anchor so the bus-state is framed onto the dump's grid.
    """
    records = bustrace if isinstance(bustrace, np.ndarray) else load_bus(bustrace)
    if t0 is None and dump is not None:
        t0 = dump_first_play_cycle(dump)
    state, _, _ = per_frame_state_from_bus(records, t0=t0)
    rendered = render_generic(program)
    nframes = min(len(rendered), len(state))
    resid = {
        reg: int(np.sum(rendered[:nframes, reg] != state[:nframes, reg]))
        for reg in range(NREG)
    }
    return resid, rendered, state


# ---------------------------------------------------------------------------
# (De)serialisation: the fitter's per-segment tuples carry numpy scalars; coerce
# to plain Python so a BaccProgram is JSON-clean and round-trips through render.
# ---------------------------------------------------------------------------
def _py(value):
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_genfits(genfits):
    out = {}
    for (voice, cls), (segments, carry) in genfits.items():
        out[f"{voice}:{cls}"] = {
            "segments": [
                [int(start), int(stop), _serialize_fit(fit)]
                for start, stop, fit in segments
            ],
            "carry": None if carry is None else [int(c) for c in carry],
        }
    return out


def _serialize_fit(fit):
    if fit is None:
        return None
    return [fit[0], _py(fit[1])]


def _serialize_eventfits(eventfits):
    return {
        str(reg): [
            [int(start), int(stop), name, _py(prm)] for start, stop, name, prm in segs
        ]
        for reg, segs in eventfits.items()
    }


def _deserialize_genfits(blob):
    out = {}
    for key, payload in blob.items():
        voice_str, cls = key.split(":")
        segments = [
            (start, stop, _deserialize_fit(fit))
            for start, stop, fit in payload["segments"]
        ]
        carry = payload["carry"]
        carry_arr = None if carry is None else np.array(carry, dtype=np.int64)
        out[(int(voice_str), cls)] = (segments, carry_arr)
    return out


def _deserialize_fit(fit):
    if fit is None:
        return None
    return (fit[0], fit[1])


def _deserialize_eventfits(blob):
    return {
        int(reg): [(start, stop, name, prm) for start, stop, name, prm in segs]
        for reg, segs in blob.items()
    }
