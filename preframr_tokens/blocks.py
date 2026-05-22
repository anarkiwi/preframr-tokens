"""Block-iteration helpers shared between tokenize-time and train-time. Torch-free; consumers (`preframr.train.regdataset.RegDataset`, etc.) wrap the outputs in torch tensors / DataLoaders at the boundary."""

import glob as _glob
import os
import random as _random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from preframr_tokens.macros import iter_self_contained_row_blocks, self_contain_slice
from preframr_tokens.reglog_helpers import load_palettes_attrs
from preframr_tokens.reglogparser import RegLogParser, remove_voice_reg
from preframr_tokens.stfconstants import (
    DELAY_REG,
    DUMP_SUFFIX,
    FRAME_REG,
    PARSED_SUFFIX,
)

__all__ = [
    "LEGACY_EVAL_SUBSET_NAME",
    "SeqMeta",
    "parse_eval_reglogs",
    "reg_widths_path",
    "glob_dumps",
    "iter_voiced_blocks",
    "materialize_block_array",
    "parser_worker",
    "self_contained_prompt_df",
]

LEGACY_EVAL_SUBSET_NAME = "val"


@dataclass
class SeqMeta:
    """Per-rotation metadata carried alongside a .blocks.npy file. Pure data; consumers (BlockMapper in main repo) keep their own torch buffers."""

    irq: int
    df_file: str
    i: int
    l: Optional[int] = None
    npy_path: Optional[str] = None


def parse_eval_reglogs(value):
    """Parse the --eval-reglogs string into an OrderedDict of {subset_name: glob}. Legacy single-subset form (no `name=` prefix) maps to the LEGACY_EVAL_SUBSET_NAME bucket; multi-subset form is `name1=glob1;name2=glob2`."""
    value = (value or "").strip()
    if not value:
        return OrderedDict()
    if "=" not in value:
        return OrderedDict([(LEGACY_EVAL_SUBSET_NAME, value)])
    out = OrderedDict()
    for entry in value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"--eval-reglogs entry {entry!r} is missing name=glob form "
                f"(multi-subset mode is name1=glob1;name2=glob2)"
            )
        name, glob = entry.split("=", 1)
        name = name.strip()
        glob = glob.strip()
        if not name or not glob:
            raise ValueError(f"--eval-reglogs entry {entry!r} has empty name or glob")
        if any(c in name for c in "/ \t"):
            raise ValueError(
                f"--eval-reglogs subset name {name!r} contains forbidden char"
            )
        if name in out:
            raise ValueError(
                f"--eval-reglogs subset name {name!r} appears more than once"
            )
        out[name] = glob
    return out


def reg_widths_path(df_map_csv_path):
    """Sidecar JSON path next to df-map.csv carrying corpus-wide reg_widths. Persisted at tokenize time so train + predict don't have to re-derive it via a parser_worker pass over the dumps."""
    base, _ = os.path.splitext(df_map_csv_path)
    return base + "_reg_widths.json"


def glob_dumps(
    reglogs,
    max_files,
    require_pq,
    seed=0,
    exclude_digi=False,
    irq_range=None,
    require_meta=False,
):
    """Glob dump paths from a comma-separated globs spec; optionally filter via dump_meta sidecar (exclude_digi / irq_range / require_meta)."""
    from preframr_tokens.dump_meta import (
        filter_dump_paths,
    )  # pylint: disable=import-outside-toplevel

    _random.seed(seed)
    dump_files = []
    for r in reglogs.split(","):
        max_globbed = max_files - len(dump_files)
        if max_globbed <= 0:
            break
        pre_globbed = _glob.glob(r, recursive=True)
        _random.shuffle(pre_globbed)
        globbed = []
        for f in pre_globbed:
            if not require_pq or _glob.glob(f.replace(DUMP_SUFFIX, PARSED_SUFFIX)):
                globbed.append(f)
                if len(globbed) >= max_globbed:
                    break
        dump_files.extend(globbed[:max_globbed])
    _random.seed()
    if exclude_digi or irq_range is not None or require_meta:
        dump_files, _dropped = filter_dump_paths(
            dump_files,
            exclude_digi=exclude_digi,
            irq_range=irq_range,
            require_meta=require_meta,
        )
    return dump_files


def iter_voiced_blocks(
    raw_df, seq_len, parser, reg_widths, frames_per_block=None, stride=None
):
    """Yield each self-contained block as a post-voice-reg row df. Used by RegDataset.make_tokens to build the token alphabet from blocks (so the alphabet covers exactly the (op, reg, subreg, val) tuples training will see), and by parser_worker for the per-dump block list."""
    if frames_per_block is None:
        frames_per_block = max(1, seq_len // 2)
    abs_df, _ = remove_voice_reg(raw_df.copy(), reg_widths)
    for block_df in iter_self_contained_row_blocks(
        abs_df, frames_per_block, args=parser.args, stride=stride
    ):
        if block_df.empty:
            continue
        try:
            voiced = parser._add_voice_reg(  # pylint: disable=protected-access
                block_df.copy(), zero_voice_reg=True
            )
        except Exception:  # pylint: disable=broad-except
            continue
        yield voiced


def materialize_block_array(
    tokenizer,
    raw_df,
    seq_len,
    parser,
    reg_widths,
    frames_per_block=None,
    stride=None,
):
    """Materialise the encoded raw_df (post-voice-reg, post-LoopPass) into a fixed-size 2D numpy array of self-contained blocks; pads short tails with zeros."""
    block_size = seq_len + 1
    blocks = []
    for voiced in iter_voiced_blocks(
        raw_df,
        seq_len,
        parser,
        reg_widths,
        frames_per_block=frames_per_block,
        stride=stride,
    ):
        merged = tokenizer.merge_token_df(tokenizer.tokens, voiced.copy())
        if merged is None or "n" not in merged.columns:
            raise RuntimeError(
                "merge_token_df returned no 'n' column; alphabet does "
                "not cover block tokens"
            )
        n = merged["n"].astype(np.int32).to_numpy()
        seq = tokenizer.encode(n).astype(np.int32)
        if len(seq) >= block_size:
            blocks.append(seq[:block_size])
        else:
            padded = np.zeros(block_size, dtype=np.int32)
            padded[: len(seq)] = seq
            blocks.append(padded)
    if not blocks:
        return np.zeros((0, block_size), dtype=np.int32)
    return np.stack(blocks)


def parser_worker(args, logger, dump_file, max_perm):
    """Parse one dump_file via RegLogParser; for each rotation, materialise the post-voice-reg block list. Returns (dump_file, [(rotation_df, [voiced_block_dfs]), ...]); on parser error returns (dump_file, [])."""
    reg_log_parser = RegLogParser(args, logger)
    block_parser = RegLogParser(args, logger)
    stride = getattr(args, "block_stride", None)
    seq_len = args.seq_len
    out = []
    try:
        for df in reg_log_parser.parse(
            dump_file, max_perm=max_perm, require_pq=args.require_pq
        ):
            blocks = list(
                iter_voiced_blocks(df, seq_len, block_parser, {}, stride=stride)
            )
            out.append((df, blocks))
    except (AssertionError, ValueError, KeyError) as e:
        logger.warning("parser_worker: dropping %s: %s", dump_file, e)
        return dump_file, []
    return dump_file, out


def self_contained_prompt_df(
    loader, dataset, seq, seq_meta, start, prompt_seq_len, irq
):
    """Return a row-level prompt df where BACK_REF / GATE_REPLAY rows whose targets fall before the prompt have been materialised into literal frames. Decoders can then expand the df without the preamble in scope."""
    full_states = dataset.tokenizer.decode(seq.numpy())
    full_df = loader._state_df(  # pylint: disable=protected-access
        full_states, dataset, irq
    )
    full_df, _ = remove_voice_reg(full_df, dataset.reg_widths)
    if "op" not in full_df.columns:
        return full_df.iloc[start : start + prompt_seq_len].reset_index(drop=True)
    parsed_pq = f"{seq_meta.df_file.replace(DUMP_SUFFIX, '')}.{seq_meta.i}.parquet"
    full_df.attrs.update(load_palettes_attrs(parsed_pq))
    is_marker = full_df["reg"].isin({FRAME_REG, DELAY_REG})
    slice_lo = int(is_marker.iloc[:start].sum())
    slice_hi = slice_lo + int(is_marker.iloc[start : start + prompt_seq_len].sum())
    return self_contain_slice(full_df, slice_lo, slice_hi, args=dataset.args)
