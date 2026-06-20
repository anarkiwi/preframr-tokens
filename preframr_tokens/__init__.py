"""Public API for preframr-tokens: the step/tracker SID codec.

preframr-tokens IS the white-box decompiler codec — dump (.parquet) -> inline
op-program tokens, residual-zero (decode == dump byte-exact) at < 1 token/frame
on Monty-on-the-Run. The public surface lives in the ``preframr_tokens.codec``
subpackage and is re-exported here:

  per_frame_state(dump, cpf, maxframes) -- dump (.parquet) -> per-frame 25-reg state
  CPF / NTSC_CPF / cpf_from_meta        -- the frame clock for the state grid
  measure(state)                        -- token breakdown + frame count
  verify_residual(state)                -- True iff decode == dump byte-exact

Every other ``preframr_tokens.codec.*`` submodule path is internal and may move
between releases.
"""

from preframr_tokens.codec import (
    CPF,
    NTSC_CPF,
    cpf_from_meta,
    measure,
    per_frame_state,
    verify_residual,
)

__all__ = [
    "per_frame_state",
    "CPF",
    "NTSC_CPF",
    "cpf_from_meta",
    "measure",
    "verify_residual",
]
