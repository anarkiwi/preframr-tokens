"""Public API for preframr-tokens: the two-file BACC SID codec.

preframr-tokens recovers a per-voice bounded-accumulator (BACC) program from a
.sid by running its playroutine white-box (py65) and tapping driver RAM, verifies
the program regenerates the .dump byte-exact, and emits an inline BACC token log:

  recover_program(sid, dump, cpf, subtune) -- (.sid, .dump) -> BaccProgram
  render_program(program)                  -- BaccProgram -> (nframes, 25) state
  verify_residual(sid, dump, cpf, subtune) -- True iff render == dump byte-exact
  program_to_ids(program) / ids_to_program -- model-facing token id stream
  measure(program)                         -- ({block: tokens}, nframes)
  VOCAB / PAD_ID                           -- token alphabet size + padding id

The dump reader (per_frame_state / CPF / cpf_from_meta) is re-exported for framing.
The reference budget gate (tests/test_goattracker.py::test_grid_runner_context_budget):
on the full Grid_Runner dump, verify_residual is True and
measure(program)['total'] / frames < 1.0.
"""

from preframr_tokens.bacc import (
    BaccProgram,
    NoteOn,
    PAD_ID,
    VOCAB,
    ids_to_program,
    measure,
    program_to_ids,
    recover_program,
    render_program,
    verify_residual,
)
from preframr_tokens.codec import CPF, NTSC_CPF, cpf_from_meta, per_frame_state

__all__ = [
    "recover_program",
    "render_program",
    "verify_residual",
    "program_to_ids",
    "ids_to_program",
    "measure",
    "BaccProgram",
    "NoteOn",
    "VOCAB",
    "PAD_ID",
    "per_frame_state",
    "CPF",
    "NTSC_CPF",
    "cpf_from_meta",
]
