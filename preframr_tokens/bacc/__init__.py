"""BACC codec: two-file (.sid + .dump) white-box SID decompiler.

Recover a per-voice bounded-accumulator (BACC) program from the .sid by running
the playroutine in py65 and tapping driver RAM, verify it regenerates the .dump
byte-exact, and emit an inline BACC token/program log for the model.
"""

from preframr_tokens.bacc.primitive import BaccProgram, NoteOn
from preframr_tokens.bacc.recover import (
    recover_program,
    render_program,
    verify_residual,
)
from preframr_tokens.bacc.serialize import (
    PAD_ID,
    VOCAB,
    ids_to_program,
    measure,
    program_to_ids,
)

__all__ = [
    "BaccProgram",
    "NoteOn",
    "recover_program",
    "render_program",
    "verify_residual",
    "program_to_ids",
    "ids_to_program",
    "measure",
    "VOCAB",
    "PAD_ID",
]
