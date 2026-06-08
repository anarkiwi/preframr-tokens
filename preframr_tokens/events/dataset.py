"""Events-native tokenizer + dataset build (REDESIGN_optionB §7.1, step 2-3). The event token stream is a
flat list of small atom ids, so the existing ``RegTokenizer`` unicode-serialize + BPE + encode/decode layer
is reused verbatim with a synthetic 68-atom alphabet; only the (op,reg,subreg,val) alphabet-building and
``merge_token_df`` are bypassed (event ids ARE the "n" stream). Atom ids are offset by +1 into "n" space so
id 0 is reserved for PAD, matching the model's zero-padded fixed-size blocks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    OP_PDTYPE,
    PAD_REG,
    SET_OP,
    SUBREG_PDTYPE,
    TOKEN_PDTYPE,
    VAL_PDTYPE,
)

from . import factored
from .oracle import ordered_writes
from .pipeline import VOCAB_SIZE, iter_windows

PAD_ID = 0


def events_alphabet() -> pd.DataFrame:
    """The fixed pre-BPE alphabet: a PAD row at n=0 then one row per event atom at n = atom_id + 1. The
    op/reg/subreg/val fields are synthetic (only ``n`` drives unicode-serialize); no corpus scan needed.
    """
    rows = [{"op": 0, "reg": PAD_REG, "subreg": -1, "val": 0, "count": 0}]
    for a in range(VOCAB_SIZE):
        rows.append({"op": SET_OP, "reg": 0, "subreg": -1, "val": a, "count": 1})
    tokens = pd.DataFrame(rows)
    tokens["n"] = tokens.index
    tokens = tokens.astype(
        {
            "val": VAL_PDTYPE,
            "n": TOKEN_PDTYPE,
            "subreg": SUBREG_PDTYPE,
            "op": OP_PDTYPE,
        }
    )
    return tokens


def make_tokenizer(args, logger=logging) -> RegTokenizer:
    """A ``RegTokenizer`` over the fixed event alphabet (splitters resync to 0: there is no FRAME_REG)."""
    tk = RegTokenizer(args, events_alphabet(), logger)
    tk._resync_splitters_from_tokens()  # pylint: disable=protected-access
    return tk


def block_to_ids(ow_window) -> list[int]:
    """One frame window -> its pre-BPE token-id block in n-space (atom_id + 1; 0 is PAD)."""
    return [a + 1 for a in factored.encode(ow_window)]


def ids_to_writes(n_ids) -> list[tuple[int, int, int]]:
    """Inverse of :func:`block_to_ids` (dropping PAD): n-space ids -> ordered ``(frame, reg, val)``."""
    return factored.decode([int(n) - 1 for n in n_ids if int(n) > 0])


def dump_block_ids(
    df, frames_per_block: int, stride: int | None = None
) -> list[list[int]]:
    """Raw dump -> list of self-contained pre-BPE token-id blocks (n-space), the §7.1 tokenization."""
    ow = ordered_writes(df)
    return [block_to_ids(w) for w in iter_windows(ow, frames_per_block, stride)]


def encode_block_array(
    tokenizer: RegTokenizer, df, block_size: int, frames_per_block: int, stride=None
) -> np.ndarray:
    """Materialise a tune into the model's ``(n_blocks, block_size)`` int32 array: per window, BPE-encode
    the event token block (n-space) via the finalised tokenizer, zero-pad / truncate to ``block_size``.
    """
    rows = []
    for ids in dump_block_ids(df, frames_per_block, stride):
        seq = tokenizer.encode(np.asarray(ids, dtype=np.int32)).astype(np.int32)
        row = np.zeros(block_size, dtype=np.int32)
        row[: min(len(seq), block_size)] = seq[:block_size]
        rows.append(row)
    if not rows:
        return np.zeros((0, block_size), dtype=np.int32)
    return np.stack(rows)


__all__ = [
    "PAD_ID",
    "block_to_ids",
    "dump_block_ids",
    "encode_block_array",
    "events_alphabet",
    "ids_to_writes",
    "make_tokenizer",
]
