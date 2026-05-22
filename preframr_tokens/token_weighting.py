"""Per-vocab-id audio-frame-time weighting. Thin wrapper over ``VocabSignature``; consumers that also need tier ids should build a ``VocabSignature`` directly to avoid two passes."""

from __future__ import annotations

import numpy as np

from preframr_tokens.vocab_signature import VocabSignature


def vocab_frame_weights(rt, tokens, n_vocab: int) -> np.ndarray:
    """Per-vocab-id frame-time weight (float32). Default 1.0 for tokens with no explicit weight. Weight sources: BACK_REF_LEN (val), DO_LOOP (val), SLOPE_RUNTIME (val), DELAY (val), FRAME (+1.0)."""
    return VocabSignature(rt, tokens, n_vocab).frame_weights
