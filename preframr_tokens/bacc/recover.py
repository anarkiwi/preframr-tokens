"""Two-file (.sid + .dump) recovery + render + residual verification.

recover_program runs the .sid's playroutine (white-box, py65) to recover a
BaccProgram and seeds frame 0 from the dump; render_program regenerates the
per-frame registers; verify_residual checks the render is byte-exact vs the dump.
"""

import numpy as np

from preframr_tokens.bacc.backends import select_backend
from preframr_tokens.bacc.sidemu import load_psid
from preframr_tokens.codec.lane_grammar import per_frame_state
from preframr_tokens.codec.lsp_validate import CPF


def recover_program(sid_path, dump_path, cpf=CPF, subtune=0, maxframes=10**9):
    """Recover a BaccProgram from the .sid, framed to the dump's grid.

    Most drivers are framed at the PAL/NTSC raster (the default ``cpf``); a
    backend may declare a native IRQ period (``native_cpf``) when the dump must
    be binned at the tune's own IRQ rate instead -- e.g. lft's CIA-timer tunes
    are framed at ~16422 cycles, not the 19656-cycle raster frame.
    """
    psid = load_psid(sid_path)
    backend = select_backend(psid)
    cpf = getattr(backend, "native_cpf", None) or cpf
    state = per_frame_state(dump_path, cpf, maxframes)
    if state is None or len(state) < 2:
        raise ValueError(f"dump did not parse to frames: {dump_path}")
    program = backend.recover(psid, len(state), subtune)
    program.boot = list(state[0])
    program.tables["boot1"] = list(
        state[1]
    )  # 2nd boot frame (used by multi-boot drivers)
    return program


def render_program(program):
    """Render a recovered BaccProgram to an (nframes, 25) register array."""
    psid_backend = _backend_for(program.driver)
    return psid_backend.render(program)


def verify_residual(sid_path, dump_path, cpf=CPF, subtune=0):
    """True iff render(recover(sid)) == per_frame_state(dump) byte-exact.

    Residual equality is taken modulo the backend's declared don't-care mask
    (``mask_state``, identity by default) -- the same documented logging masks
    the dump validator applies (PW-high unused bits; the lft filter-external
    bit). Drivers without a mask compare raw, exactly as before.
    """
    psid = load_psid(sid_path)
    backend = select_backend(psid)
    cpf = getattr(backend, "native_cpf", None) or cpf
    state = per_frame_state(dump_path, cpf, 10**9)
    program = recover_program(sid_path, dump_path, cpf, subtune)
    mask = getattr(backend, "mask_state", None) or (lambda s: s)
    return bool(np.array_equal(mask(render_program(program)), mask(state)))


def _backend_for(driver):
    from preframr_tokens.bacc.backends.goattracker import GoatTrackerBackend
    from preframr_tokens.bacc.backends.hubbard import (
        Hubbard5TTBackend,
        HubbardMontyBackend,
    )
    from preframr_tokens.bacc.backends.lft import LftBackend

    for backend in (
        HubbardMontyBackend(),
        Hubbard5TTBackend(),
        LftBackend(),
        GoatTrackerBackend(),
    ):
        if backend.name == driver:
            return backend
    raise ValueError(f"no backend named {driver}")
