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
    """Recover a BaccProgram from the .sid, framed to the dump's grid."""
    state = per_frame_state(dump_path, cpf, maxframes)
    if state is None or len(state) < 2:
        raise ValueError(f"dump did not parse to frames: {dump_path}")
    psid = load_psid(sid_path)
    backend = select_backend(psid)
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
    """True iff render(recover(sid)) == per_frame_state(dump) byte-exact."""
    state = per_frame_state(dump_path, cpf, 10**9)
    program = recover_program(sid_path, dump_path, cpf, subtune)
    return bool(np.array_equal(render_program(program), state))


def _backend_for(driver):
    from preframr_tokens.bacc.backends.goattracker import GoatTrackerBackend
    from preframr_tokens.bacc.backends.hubbard import (
        Hubbard5TTBackend,
        HubbardMontyBackend,
    )

    for backend in (HubbardMontyBackend(), Hubbard5TTBackend(), GoatTrackerBackend()):
        if backend.name == driver:
            return backend
    raise ValueError(f"no backend named {driver}")
