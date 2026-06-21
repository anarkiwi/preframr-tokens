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
the bus-state; on the proven tunes the generator lanes render residual-zero and
the documented generator-lane gaps (3/8 corpus tunes) are surfaced, never faked.
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


def recover_generic(sid, dump, bustrace):
    """Recover a generic :class:`BaccProgram` from a tune's trusted bus trace.

    ``sid`` is retained for provenance only (the recovery reads the bus trace,
    not the playroutine).  ``dump`` supplies the frame-0 anchor (the dump defines
    the grid the bus-state must reproduce); when ``dump`` is None the bus's own
    first-play cycle is used.  ``bustrace`` is a native preframr-sidtrace
    ``.bus.bin`` path or a pre-loaded ``BUS_DT`` record array.

    The returned program's ``tables`` carry the per-register fitted programs and
    the bus-recovered note table; ``boot`` is the frame-0 register seed.
    """
    records = bustrace if isinstance(bustrace, np.ndarray) else load_bus(bustrace)
    t0 = dump_first_play_cycle(dump) if dump is not None else None
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


def residual(program, bustrace, dump=None):
    """Per-register residual frame counts of :func:`render_generic` against the
    bus-state ``program`` was recovered from.  Returns ``(resid, rendered,
    state)`` where ``resid`` maps register -> residual frame count (0 = byte-exact
    for that register).  ``sum(resid.values()) == 0`` is whole-tune residual-zero.
    """
    records = bustrace if isinstance(bustrace, np.ndarray) else load_bus(bustrace)
    t0 = dump_first_play_cycle(dump) if dump is not None else None
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
