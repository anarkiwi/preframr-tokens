"""Per-vocab-id audio-frame-time weighting. Used to scale per-token CE loss by how much audio time a token represents, so a single ``DELAY`` worth 32 frames doesn't get the same gradient weight as one ``FRAME`` worth 1."""

from __future__ import annotations

import numpy as np

from preframr_tokens.stfconstants import (
    BACK_REF_OP,
    BACK_REF_SUBREG_LEN,
    DELAY_REG,
    DO_LOOP_OP,
    FRAME_REG,
    SLOPE_OPS,
    SLOPE_SUBREG_RUNTIME,
)

_SLOPE_OP_SET = frozenset(SLOPE_OPS)


def vocab_frame_weights(rt, tokens, n_vocab: int) -> np.ndarray:
    """Return per-vocab-id frame-time weight as float32 numpy array. Default 1.0 for tokens with no explicit weight. Weight sources: BACK_REF_LEN (val), DO_LOOP (val), SLOPE_RUNTIME (val), DELAY (val), FRAME (+1.0)."""
    weights = np.ones(n_vocab, dtype=np.float32)
    if tokens is None or len(tokens) == 0:
        return weights
    n_base = len(tokens)
    for vid in range(n_vocab):
        if rt.tkmodel:
            base_ids = rt.decode([vid])
        else:
            base_ids = [vid]
        w = 0.0
        for bid in base_ids:
            bid = int(bid)
            if bid >= n_base:
                continue
            row = tokens.iloc[bid]
            reg = int(row.reg)
            op = int(row.op)
            val = int(row.val)
            subreg = int(row.subreg)
            if op == BACK_REF_OP and subreg == BACK_REF_SUBREG_LEN:
                w += val
            elif op == DO_LOOP_OP and subreg == 0:
                w += val
            elif op in _SLOPE_OP_SET and subreg == SLOPE_SUBREG_RUNTIME:
                w += val
            elif reg == DELAY_REG:
                w += val
            elif reg == FRAME_REG:
                w += 1.0
        if w > 0.0:
            weights[vid] = w
    return weights
