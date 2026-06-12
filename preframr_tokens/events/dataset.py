"""Events-native tokenizer + dataset build. The event token stream is a
flat list of small atom ids, so the existing ``RegTokenizer`` unicode-serialize + BPE + encode/decode layer
is reused verbatim with the fixed ``stream.VOCAB_SIZE``-atom alphabet; only the (op,reg,subreg,val)
alphabet-building and ``merge_token_df`` are bypassed (event ids ARE the "n" stream). Atom ids are offset
by +1 into "n" space so id 0 is reserved for PAD, matching the model's zero-padded fixed-size blocks.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import zstandard as zstd

from preframr_tokens.regtokenizer import RegTokenizer
from preframr_tokens.stfconstants import (
    DUMP_SUFFIX,
    OP_PDTYPE,
    PAD_REG,
    SET_OP,
    SUBREG_FLUSH_OP,
    SUBREG_PDTYPE,
    TOKEN_PDTYPE,
    VAL_PDTYPE,
)

from . import stream
from .oracle import ordered_writes
from .pipeline import VOCAB_SIZE, iter_windows

PAD_ID = 0

BOUNDARY_ISOLATION_NS = tuple(
    [stream.VOICE_BASE + 1 + v for v in range(4)] + [stream.KEYFRAME + 1]
)

ATOM_CACHE_VERSION = 1
_ATOM_DTYPE = np.int32


def _atom_cache_path(df_file: str) -> str:
    """In-place, codec-version-keyed atom-stream cache path next to the real dump
    (symlinks resolved). Lets a tkvocab sweep reuse the tkvocab-independent pre-BPE
    encode instead of re-running ``stream.encode`` + its self-verify. Bump
    ``ATOM_CACHE_VERSION`` whenever the event codec (``ordered_writes`` /
    ``stream.encode``) changes so stale caches are skipped, never mis-read."""
    real = os.path.realpath(df_file)
    return real.replace(DUMP_SUFFIX, f".{ATOM_CACHE_VERSION}.atoms.zst")


def _read_atom_cache(path: str) -> list[int] | None:
    """Return the cached n-space atom ids, or ``None`` when absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with zstd.open(path, "rb") as fh:
            buf = fh.read()
        return np.frombuffer(buf, dtype=_ATOM_DTYPE).tolist()
    except (OSError, ValueError):
        return None


def _write_atom_cache(path: str, ids: list[int]) -> None:
    """Best-effort atomic write of the atom stream; silently skips a read-only tree."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with zstd.open(tmp, "wb") as fh:
            fh.write(np.asarray(ids, dtype=_ATOM_DTYPE).tobytes())
        os.replace(tmp, path)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def events_alphabet() -> pd.DataFrame:
    """The fixed pre-BPE alphabet: a PAD row at n=0 then one row per event atom at n = atom_id + 1; only
    ``n`` drives unicode-serialize, while ``op`` carries the loss tier: ``stream.is_content_atom`` atoms
    are ``SET_OP`` (content) and structural scaffolding is ``SUBREG_FLUSH_OP`` (a registry-structural op
    borrowed purely for its tier). ``reg`` stays 0 -- FRAME_REG would register frame splitters and change
    BPE -- and both ops are frame-weight-neutral, so ``frame_weights`` stay 1.0."""
    rows = [{"op": 0, "reg": PAD_REG, "subreg": -1, "val": 0, "count": 0}]
    for a in range(VOCAB_SIZE):
        op = SET_OP if stream.is_content_atom(a) else SUBREG_FLUSH_OP
        rows.append({"op": op, "reg": 0, "subreg": -1, "val": a, "count": 1})
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
    return [a + 1 for a in stream.encode(ow_window)]


def ids_to_writes(n_ids, extend=False) -> list[tuple[int, int, int]]:
    """Inverse of :func:`block_to_ids` (dropping PAD and any KEYFRAME conditioning segments): n-space
    ids -> ordered ``(frame, reg, val)``. ``extend=True`` replays continuations past the declared
    frame count (generation)."""
    return stream.decode(
        stream.strip_keyframes([int(n) - 1 for n in n_ids if int(n) > 0]),
        extend=extend,
    )


def dump_block_ids(
    df, frames_per_block: int, stride: int | None = None
) -> list[list[int]]:
    """Raw dump -> list of self-contained pre-BPE token-id blocks (n-space), one per frame window."""
    ow = ordered_writes(df)
    return [block_to_ids(w) for w in iter_windows(ow, frames_per_block, stride)]


def dump_token_ids(df, df_file: str | None = None) -> list[int]:
    """A whole tune's pre-BPE event token stream in n-space (the unit BPE trains over; byte-exact:
    ``ids_to_writes(dump_token_ids(df))`` reproduces the ordered writes). When ``df_file`` is given, a
    codec-version-keyed ``.atoms.zst`` sidecar next to the dump is reused if present (skipping
    ``stream.encode`` + its self-verify) and populated otherwise (best-effort; read-only trees
    recompute) -- the atom stream is tkvocab-independent, so pre-populating it skips a sweep's encode.
    """
    path = _atom_cache_path(df_file) if df_file is not None else None
    if path is not None:
        cached = _read_atom_cache(path)
        if cached is not None:
            return cached
    ids = [a + 1 for a in stream.encode(ordered_writes(df))]
    if path is not None:
        _write_atom_cache(path, ids)
    return ids


def encode_block_array(
    tokenizer: RegTokenizer,
    df,
    block_size: int,
    stride: int | None = None,
    df_file: str | None = None,
) -> np.ndarray:
    """Materialise a tune into the model's ``(n_blocks, block_size)`` int32 array: BPE-encode the whole-
    tune event stream and chunk it to ``block_size``; with the default stride each chunk is led by a
    BPE-encoded KEYFRAME conditioning segment (:func:`stream.chunk_keyframe`: tick/tuning + per-voice
    state at the boundary) so every training chunk can interpret its durations/intervals. An explicit
    ``stride`` keeps plain prefix-free chunking; ``ids_to_writes`` strips segments before decode.
    """
    atoms_n = dump_token_ids(df, df_file)
    seq = tokenizer.encode(np.asarray(atoms_n, dtype=np.int32)).astype(np.int32)
    n = len(seq)
    if n == 0:
        return np.zeros((0, block_size), dtype=np.int32)
    rows = []
    if stride is not None:
        for start in range(0, n, stride):
            chunk = seq[start : start + block_size]
            row = np.zeros(block_size, dtype=np.int32)
            row[: len(chunk)] = chunk
            rows.append(row)
            if start + block_size >= n:
                break
        return np.stack(rows)
    atoms = [int(a) - 1 for a in atoms_n]
    alen_cache: dict[int, int] = {}

    def _alen(i: int) -> int:
        if i not in alen_cache:
            alen_cache[i] = len(tokenizer.decode(np.asarray([i], dtype=np.uint32)))
        return alen_cache[i]

    start = 0
    apos = 0
    while start < n:
        prefix = np.zeros(0, dtype=np.int32)
        if apos:
            kf = stream.chunk_keyframe(atoms, apos)
            if kf:
                kf_ids = tokenizer.encode(
                    np.asarray([a + 1 for a in kf], dtype=np.int32)
                ).astype(np.int32)
                if len(kf_ids) <= block_size // 4:
                    prefix = kf_ids
        eff = block_size - len(prefix)
        chunk = seq[start : start + eff]
        row = np.zeros(block_size, dtype=np.int32)
        row[: len(prefix)] = prefix
        row[len(prefix) : len(prefix) + len(chunk)] = chunk
        rows.append(row)
        apos += sum(_alen(int(i)) for i in chunk)
        start += eff
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
