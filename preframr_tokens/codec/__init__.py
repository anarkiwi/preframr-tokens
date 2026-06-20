"""Step/tracker SID codec: dump (.parquet) -> inline op-program tokens, residual-zero.

This subpackage is the canonical preframr-tokens tokenizer. It is the white-box
decompiler codec validated on Monty-on-the-Run at < 1 token/frame, byte-exact
(residual-zero). The public entry points:

  per_frame_state(dump, cpf, maxframes) -- dump (.parquet) -> per-frame 25-reg state
  CPF                                    -- PAL cycles/frame (frame clock for state)
  measure(state)                         -- token breakdown + frame count
  verify_residual(state)                 -- True iff decode == dump byte-exact

The invariant (executable form in tests/test_monty_context_budget.py): on the full
Monty dump, verify_residual is True and measure(state)['total'] / frames < 1.0.
"""

from preframr_tokens.codec.lane_grammar import per_frame_state
from preframr_tokens.codec.lsp_validate import CPF, NTSC_CPF, cpf_from_meta
from preframr_tokens.codec.unified2_codec import measure, verify_residual

__all__ = [
    "per_frame_state",
    "CPF",
    "NTSC_CPF",
    "cpf_from_meta",
    "measure",
    "verify_residual",
]
