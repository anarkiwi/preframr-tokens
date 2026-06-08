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
    """Raw dump -> list of self-contained pre-BPE token-id blocks (n-space), one per frame window."""
    ow = ordered_writes(df)
    return [block_to_ids(w) for w in iter_windows(ow, frames_per_block, stride)]


def dump_token_ids(df) -> list[int]:
    """A whole tune's pre-BPE event token stream in n-space (the unit BPE trains over and the model
    decodes from). Byte-exact: ``ids_to_writes(dump_token_ids(df))`` reproduces the ordered writes.
    """
    return [a + 1 for a in factored.encode(ordered_writes(df))]


def encode_block_array(
    tokenizer: RegTokenizer, df, block_size: int, stride: int | None = None
) -> np.ndarray:
    """Materialise a tune into the model's ``(n_blocks, block_size)`` int32 array: BPE-encode the whole-
    tune event token stream, then chunk it into fixed ``block_size`` windows (stride ``block_size`` by
    default), zero-padding the final partial chunk. Decodability is whole-stream (a chunk boundary may
    fall mid-gesture); ``tokenizer.decode`` then :func:`ids_to_writes` over the full stream is byte-exact.
    """
    seq = tokenizer.encode(np.asarray(dump_token_ids(df), dtype=np.int32)).astype(
        np.int32
    )
    if stride is None:
        stride = block_size
    n = len(seq)
    if n == 0:
        return np.zeros((0, block_size), dtype=np.int32)
    rows = []
    for start in range(0, n, stride):
        chunk = seq[start : start + block_size]
        row = np.zeros(block_size, dtype=np.int32)
        row[: len(chunk)] = chunk
        rows.append(row)
        if start + block_size >= n:
            break
    return np.stack(rows)


__all__ = [
    "PAD_ID",
    "block_to_ids",
    "dump_block_ids",
    "dump_token_ids",
    "encode_block_array",
    "events_alphabet",
    "ids_to_writes",
    "make_tokenizer",
]
