"""Torch-free parse-side probe helpers shared by the deterministic encoding
suite and the xpt audit probes."""

from types import SimpleNamespace

from preframr_tokens.tokenizer_config import PARSER_DEFAULTS


def parse_args(**over):
    """A full parser args namespace (every macro flag present + off) with the freq
    family flags forced off so callers opt in to exactly one encoding config."""
    cfg = dict(PARSER_DEFAULTS)
    cfg.update(
        freq_trajectory_pass=False,
        loop_pass=False,
        loop_transposed=False,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)
