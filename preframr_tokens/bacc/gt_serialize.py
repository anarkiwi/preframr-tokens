"""Serialize a GoatTracker BaccProgram to/from the token-id stream.

The GoatTracker program is the song itself (canonical .SNG bytes) -- the composer's
tracker module, the sparse per-tune content, not a per-frame trace. It serializes
through the same base-16 LEB digit alphabet as the Hubbard codec (no new tokens):
a small header (nframes, subtune, the packed player's render params, length) then
the song bytes. The render params (adparam + the pulse/realtime skip optimization
flags) are baked into the packed player, not the .SNG, so they ride the header.
"""

from preframr_tokens.bacc.backends.goattracker import make_program
from preframr_tokens.bacc.serialize import _ru, _wu


def _seed(program):
    return {
        "subtune": int(program.seed["subtune"]),
        "adparam": int(program.seed["adparam"]),
        "optimize_pulse": int(bool(program.seed["optimize_pulse"])),
        "optimize_realtime": int(bool(program.seed["optimize_realtime"])),
    }


NREG = 25


def gt_program_to_ids(program):
    out = []
    seed = _seed(program)
    _wu(out, program.nframes)
    _wu(out, seed["subtune"])
    _wu(out, seed["adparam"])
    _wu(out, seed["optimize_pulse"])
    _wu(out, seed["optimize_realtime"])
    # boot frames (the dump's startup seed) anchor render to the dump frame grid.
    boot = list(program.boot) or [0] * NREG
    boot1 = program.tables.get("boot1") or [0] * NREG
    for byte in boot:
        _wu(out, byte)
    for byte in boot1:
        _wu(out, byte)
    sng = program.tables["sng"]
    _wu(out, len(sng))
    for byte in sng:
        _wu(out, byte)
    return out


def gt_ids_to_program(ids):
    i = 0
    nframes, i = _ru(ids, i)
    subtune, i = _ru(ids, i)
    adparam, i = _ru(ids, i)
    optimize_pulse, i = _ru(ids, i)
    optimize_realtime, i = _ru(ids, i)
    boot = []
    for _ in range(NREG):
        byte, i = _ru(ids, i)
        boot.append(byte)
    boot1 = []
    for _ in range(NREG):
        byte, i = _ru(ids, i)
        boot1.append(byte)
    length, i = _ru(ids, i)
    sng = []
    for _ in range(length):
        byte, i = _ru(ids, i)
        sng.append(byte)
    seed = {
        "subtune": subtune,
        "adparam": adparam,
        "optimize_pulse": optimize_pulse,
        "optimize_realtime": optimize_realtime,
    }
    program = make_program(sng, seed, nframes)
    program.boot = boot
    program.tables["boot1"] = boot1
    return program


def gt_measure(program):
    ids = gt_program_to_ids(program)
    sng_tokens = sum(len(_u(b)) for b in program.tables["sng"])
    return {"sng": sng_tokens, "total": len(ids)}, program.nframes


def _u(n):
    out = []
    _wu(out, n)
    return out
