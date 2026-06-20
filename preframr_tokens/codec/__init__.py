"""Dump-reading helpers for the BACC codec.

per_frame_state parses a register .dump.parquet into the per-frame 25-register
state array the BACC renderer verifies against; CPF / cpf_from_meta frame it.
"""

from preframr_tokens.codec.lane_grammar import per_frame_state
from preframr_tokens.codec.lsp_validate import (
    CPF,
    NTSC_CPF,
    cpf_from_meta,
    detect_play_period,
)

__all__ = [
    "per_frame_state",
    "CPF",
    "NTSC_CPF",
    "cpf_from_meta",
    "detect_play_period",
]
