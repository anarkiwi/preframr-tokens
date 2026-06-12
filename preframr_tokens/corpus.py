"""Torch-free corpus orchestration: parse + tokenize + disk-cache .blocks.npy + load metadata routing. Owns the RegTokenizer + corpus-wide state (reg_widths, n_vocab, n_words, tokenize metadata). Main-repo torch.utils.data.Dataset adapters compose a Corpus + a BlockMapper to expose the train-side interface."""

import collections
import concurrent.futures
import json
import multiprocessing
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from preframr_tokens.alphabet_projection import build_projection_table, project_df
from preframr_tokens.blocks import (
    LEGACY_EVAL_SUBSET_NAME,
    SeqMeta,
    glob_dumps,
    parse_eval_reglogs,
    parser_worker,
    reg_widths_path,
)
from preframr_tokens.dump_meta import raw_is_digi, read_meta
from preframr_tokens.events import dataset as events_dataset
from preframr_tokens.events import oracle as events_oracle
from preframr_tokens.events import stream as events_stream
from preframr_tokens.regtokenizer import (
    RegTokenizer,
    TOKEN_PDTYPE,
    VAL_PDTYPE,
)
from preframr_tokens.stfconstants import (
    DUMP_SUFFIX,
    OP_PDTYPE,
    REG_PDTYPE,
    SUBREG_PDTYPE,
)

__all__ = ["Corpus", "TokenizeMeta"]


def _collect_atoms(df, sink):
    """Accumulate ``(op, reg, subreg, val)`` tuples from ``df`` into ``sink``. No-op if df is empty or missing required columns."""
    if df is None or len(df) == 0:
        return
    if not {"op", "reg", "subreg", "val"}.issubset(df.columns):
        return
    sub = df[["op", "reg", "subreg", "val"]].drop_duplicates()
    for op, reg, subreg, val in sub.itertuples(index=False, name=None):
        sink.add((int(op), int(reg), int(subreg), int(val)))


@dataclass
class TokenizeMeta:
    """Per-corpus tokenize-stage metadata snapshot. Populated by ``Corpus.make_tokens``; consumed by the dataset-map writer + the iter-block-seqs fast path."""

    irq_by_file: dict = field(default_factory=dict)
    rotations_by_file: dict = field(default_factory=dict)
    kind_by_file: dict = field(default_factory=dict)
    reg_widths: dict = field(default_factory=dict)
    val_subsets: list = field(default_factory=list)
    format_version: int = 0


class Corpus:
    """Torch-free corpus orchestration. See module docstring."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.tokenizer = RegTokenizer(args, tokens=None, logger=logger)
        self.reg_widths = {}
        self.n_vocab = 0
        self.n_words = 0
        self.val_subset_names = []
        self._tokenize_meta: Optional[TokenizeMeta] = None

    def load_dfs(self, reglogs=None, dump_files=None, max_perm=99, encode=True):
        """Yield (dump_file, i, df, seq, irq, blocks) tuples; parallel parser_worker; tokenizer.encode if tokens are already built and encode=True."""
        if not dump_files:
            if not reglogs:
                raise ValueError("need reglogs or dump_files")
            irq_lo = getattr(self.args, "meta_irq_lo", 0)
            irq_hi = getattr(self.args, "meta_irq_hi", 0)
            irq_range = (int(irq_lo), int(irq_hi)) if (irq_lo or irq_hi) else None
            dump_files = glob_dumps(
                reglogs,
                int(self.args.max_files * 1.25),
                self.args.require_pq,
                seed=0,
                exclude_digi=getattr(self.args, "meta_exclude_digi", False),
                irq_range=irq_range,
                require_meta=getattr(self.args, "meta_require", False),
            )
        output_dumps = set()
        max_workers = min(multiprocessing.cpu_count(), len(dump_files))
        max_files = min(self.args.max_files, len(dump_files))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    parser_worker, self.args, self.logger, dump_file, max_perm
                )
                for dump_file in dump_files[:max_workers]
            ]
            dump_files = dump_files[max_workers:]
            with tqdm(total=max_files) as pbar:
                while futures and len(output_dumps) < max_files:
                    new_futures = []
                    for future in concurrent.futures.as_completed(futures):
                        dump_file, dfs_with_blocks = future.result()
                        if dump_files:
                            new_futures.append(
                                executor.submit(
                                    parser_worker,
                                    self.args,
                                    self.logger,
                                    dump_files[0],
                                    max_perm,
                                )
                            )
                            dump_files = dump_files[1:]
                        for i, (df, blocks) in enumerate(dfs_with_blocks):
                            seq = None
                            if self.tokenizer.tokens is not None:
                                df = self.tokenizer.merge_token_df(
                                    self.tokenizer.tokens, df
                                )
                                if encode:
                                    n = df["n"].astype(np.int32).to_numpy()
                                    seq = self.tokenizer.encode(n).astype(np.int32)
                                    min_seq = getattr(self.args, "min_song_tokens", 256)
                                    if len(seq) < min_seq:
                                        self.logger.info(
                                            "rejecting sequence from %s too short %u (< %u)",
                                            dump_file,
                                            len(seq),
                                            min_seq,
                                        )
                                        break
                            output_dumps.add(dump_file)
                            irq = df["irq"].iloc[0]
                            yield dump_file, i, df, seq, irq, blocks
                        pbar.n = len(output_dumps)
                        pbar.refresh()
                        if len(output_dumps) == max_files:
                            break
                    futures = new_futures
            executor.shutdown(wait=True, cancel_futures=True)

    def make_tokens(self, reglogs, eval_reglogs=""):
        """Parse each song, materialise its self-contained blocks, build the token alphabet. Mutates self.tokenizer; populates self._tokenize_meta; returns (train_files, val_files, cached_blocks)."""
        train_files = []
        val_files = []
        cached_blocks = {}
        irq_by_file = {}
        rotations_by_file = {}
        reg_max = {}
        kind_by_file = {}

        train_atom_tuples = set()
        project_eval = bool(getattr(self.args, "project_eval_to_train", True))

        def walk(reglogs_glob, kind, files_out, projection=None, atom_sink=None):
            for df_file, i, df, _seq, irq, blocks in self.load_dfs(
                reglogs=reglogs_glob, max_perm=self.args.max_perm
            ):
                if projection is not None:
                    df = project_df(df, projection)
                    blocks = [project_df(b, projection) for b in blocks]
                if atom_sink is not None:
                    _collect_atoms(df, atom_sink)
                    for voiced in blocks:
                        _collect_atoms(voiced, atom_sink)
                self.tokenizer.accumulate_tokens(df, df_file)
                cached_blocks[(df_file, i, kind)] = blocks
                for voiced in blocks:
                    self.tokenizer.accumulate_tokens(voiced, df_file)
                if df_file not in irq_by_file:
                    irq_by_file[df_file] = int(irq)
                if df_file not in kind_by_file:
                    kind_by_file[df_file] = kind
                rotations_by_file[df_file] = max(
                    rotations_by_file.get(df_file, 0), i + 1
                )
                self.tokenizer.get_reg_max(df, reg_max)
                try:
                    if files_out[-1] == df_file:
                        continue
                except IndexError:
                    pass
                files_out.append(df_file)

        walk(
            reglogs,
            "train",
            train_files,
            atom_sink=train_atom_tuples if project_eval else None,
        )
        projection_table = (
            build_projection_table(train_atom_tuples) if project_eval else None
        )
        eval_subsets = (
            parse_eval_reglogs(eval_reglogs)
            if isinstance(eval_reglogs, str)
            else OrderedDict(eval_reglogs)
        )
        val_kinds = OrderedDict()
        for subset_name, subset_glob in eval_subsets.items():
            if subset_name == "train":
                raise ValueError(
                    "--eval-reglogs subset name 'train' collides with the "
                    "training set; pick a different name."
                )
            kind = subset_name
            val_kinds[kind] = subset_name
            walk(subset_glob, kind, val_files, projection=projection_table)

        tokens = self.tokenizer.make_tokens()
        self.tokenizer.tokens = tokens
        assert self.tokenizer.tokens[tokens["val"].isna()].empty, tokens[
            tokens["val"].isna()
        ]
        self._tokenize_meta = TokenizeMeta(
            irq_by_file=irq_by_file,
            rotations_by_file=rotations_by_file,
            kind_by_file=kind_by_file,
            reg_widths=self.tokenizer.get_reg_width_from_max(reg_max),
            val_subsets=list(eval_subsets.keys()),
        )
        return train_files, val_files, cached_blocks

    def _build_df_map_frame(self, train_files, val_files):
        """Build the dump_file → kind/irq/n_rotations dataframe written to ``args.df_map_csv``. Uses ``_tokenize_meta`` for kind / irq / n_rotations lookups; falls back to "train" / LEGACY_EVAL_SUBSET_NAME if meta is absent."""
        meta = self._tokenize_meta
        kind_lookup = meta.kind_by_file if meta else {}
        rows = []
        for p in train_files:
            rows.append((p, kind_lookup.get(p, "train")))
        for p in val_files:
            rows.append((p, kind_lookup.get(p, LEGACY_EVAL_SUBSET_NAME)))
        df = pd.DataFrame(rows, columns=["dump_file", "kind"])
        if meta:
            df["irq"] = df["dump_file"].map(meta.irq_by_file).astype("Int64")
            df["n_rotations"] = (
                df["dump_file"].map(meta.rotations_by_file).astype("Int64")
            )
        return df

    def _write_reg_widths_sidecar(self, df_map_csv_path):
        """Write the reg_widths JSON sidecar alongside ``df_map_csv_path``. No-op if no tokenize-stage meta or no df-map path."""
        meta = self._tokenize_meta
        if not meta or not df_map_csv_path:
            return
        sidecar = reg_widths_path(df_map_csv_path)
        self.logger.info("writing reg_widths to %s", sidecar)
        data = {str(k): int(v) for k, v in meta.reg_widths.items()}
        if meta.format_version:
            data["_event_format_version"] = int(meta.format_version)
        with open(sidecar, "w") as f:
            json.dump(data, f)

    def _load_reg_widths_sidecar(self, sidecar):
        """Read a reg-widths sidecar into ``{reg: width}``; when the artifact carries an event-format
        version, reject a mismatch with the running codec. Parse-domain sidecars omit the key and load
        unchanged (pre-versioning artifacts proceed)."""
        with open(sidecar) as f:
            raw = json.load(f)
        fmt = raw.pop("_event_format_version", None)
        if fmt is not None and int(fmt) != events_stream.EVENT_FORMAT_VERSION:
            raise ValueError(
                f"event-format version mismatch: artifact {int(fmt)} != code "
                f"{events_stream.EVENT_FORMAT_VERSION}; re-run the tokenize stage"
            )
        return {int(k): int(v) for k, v in raw.items()}

    def encode_and_save_cached_blocks(self, cached_blocks):
        """Encode each cached voiced-block df via the now-finalised tokenizer and write .blocks.npy. Failures (alphabet doesn't cover a row's (op, reg, subreg, val)) propagate; this is the catch point for any bug in the alphabet-building pipeline."""
        block_size = self.args.seq_len + 1
        for (df_file, i, _kind), blocks in cached_blocks.items():
            block_arrs = []
            for voiced in blocks:
                merged = self.tokenizer.merge_token_df(
                    self.tokenizer.tokens, voiced.copy()
                )
                if merged is None or "n" not in merged.columns:
                    raise RuntimeError(
                        f"merge_token_df returned no 'n' column for "
                        f"{df_file} rotation {i}"
                    )
                n = merged["n"].astype(np.int32).to_numpy()
                self.tokenizer.validate_encoding(df_file, n)
                seq = self.tokenizer.encode(n).astype(np.int32)
                if len(seq) >= block_size:
                    block_arrs.append(seq[:block_size])
                else:
                    padded = np.zeros(block_size, dtype=np.int32)
                    padded[: len(seq)] = seq
                    block_arrs.append(padded)
            if not block_arrs:
                continue
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            np.save(blocks_path, np.stack(block_arrs))

    def try_preload_from_disk(self):
        """Hydrate tokenizer from args.token_csv (+ args.tkmodel if tkvocab>0). Returns True iff both files exist and have content."""
        token_csv = getattr(self.args, "token_csv", None)
        if (
            not token_csv
            or not os.path.exists(token_csv)
            or os.path.getsize(token_csv) == 0
        ):
            return False
        tkmodel_path = getattr(self.args, "tkmodel", None)
        tkmodel_str = None
        if self.args.tkvocab:
            if (
                not tkmodel_path
                or not os.path.exists(tkmodel_path)
                or os.path.getsize(tkmodel_path) == 0
            ):
                return False
            with open(tkmodel_path) as f:
                tkmodel_str = f.read()
        try:
            tokens = pd.read_csv(token_csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return False
        if tokens.empty or "reg" not in tokens.columns:
            return False
        cast = {
            "reg": REG_PDTYPE,
            "val": VAL_PDTYPE,
            "n": TOKEN_PDTYPE,
            "subreg": SUBREG_PDTYPE,
            "op": OP_PDTYPE,
        }
        cast = {k: v for k, v in cast.items() if k in tokens.columns}
        if cast:
            tokens = tokens.astype(cast)
        self.tokenizer.load(tkmodel_str, tokens)
        self.logger.info(
            "preload (cached): %u tokens from %s%s",
            len(tokens),
            token_csv,
            f", tkmodel from {tkmodel_path}" if tkmodel_str else "",
        )
        return True

    def _read_dump(self, df_file):
        return pd.read_parquet(
            df_file, columns=["clock", "irq", "chipno", "reg", "val"]
        )

    def _events_glob(self, reglogs, eval_reglogs):
        """``(train_files, val_files, kind_by_file)`` over the raw-dump corpus (events owns parsing, so
        no ``parser_worker``): glob train + each eval subset, first-seen kind wins."""
        require_pq = getattr(self.args, "require_pq", False)
        irq_lo = getattr(self.args, "meta_irq_lo", 0)
        irq_hi = getattr(self.args, "meta_irq_hi", 0)
        irq_range = (int(irq_lo), int(irq_hi)) if (irq_lo or irq_hi) else None

        def _glob(g):
            return list(
                glob_dumps(
                    g,
                    self.args.max_files,
                    require_pq,
                    seed=0,
                    exclude_digi=getattr(self.args, "meta_exclude_digi", False),
                    irq_range=irq_range,
                    require_meta=getattr(self.args, "meta_require", False),
                )
            )

        train = _glob(reglogs)
        kind_by_file = {f: "train" for f in train}
        val = []
        if eval_reglogs:
            for name, g in parse_eval_reglogs(eval_reglogs).items():
                for f in _glob(g):
                    if f not in kind_by_file:
                        kind_by_file[f] = name
                        val.append(f)
        return train, val, kind_by_file

    def _encode_and_save_events(self, df_files):
        """Write each dump's ``.0.blocks.npy`` of BPE-encoded event token blocks (the model input). Thread
        pool: the shared tokenizer's Rust encode/decode, the zstd ``.atoms.zst`` reads and ``np.save`` all
        release the GIL, so this parallelises without pickling the tokenizer (mirrors ``train_tokenizer``'s
        uni-write pass)."""
        block_size = self.args.seq_len + 1

        def _encode_one(df_file):
            try:
                df = self._read_dump(df_file)
            except Exception:  # pylint: disable=broad-except
                return
            arr = events_dataset.encode_block_array(
                self.tokenizer, df, block_size, df_file=df_file
            )
            if arr.shape[0]:
                np.save(df_file.replace(DUMP_SUFFIX, ".0.blocks.npy"), arr)

        workers = min(8, (os.cpu_count() or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_encode_one, df_files))

    def preload(self, tokens=None, tkmodel=None):
        """Events-native tokenize-stage orchestrator: the raw dump is encoded by
        the factored codec, replacing parse -> (op,reg,subreg,val) alphabet -> merge_token_df. (a) explicit
        tokens -> load; (b) preload-from-disk; (c) glob raw dumps, set the fixed event alphabet, train BPE
        over the event token streams (tkvocab>0), write tokens.csv / df-map / reg-widths, and save the
        per-dump ``.0.blocks.npy`` event-token blocks."""
        if tokens is not None:
            self.tokenizer.load(tkmodel, tokens)
            return
        if self.try_preload_from_disk():
            return
        self.logger.info("preload making tokens (events-native)")
        eval_reglogs = getattr(self.args, "eval_reglogs", "") or ""
        train_files, val_files, kind_by_file = self._events_glob(
            self.args.reglogs, eval_reglogs
        )
        df_files = train_files + val_files
        self.tokenizer.tokens = events_dataset.events_alphabet()
        if getattr(self.args, "bpe_isolate_boundaries", False):
            self.tokenizer.isolation_ns = events_dataset.BOUNDARY_ISOLATION_NS

        irq_by_file = {}
        in_scope = set()
        skipped = collections.Counter()
        for df_file in df_files:
            try:
                df = self._read_dump(df_file)
            except Exception:  # pylint: disable=broad-except
                skipped["unreadable"] += 1
                continue
            ow = events_oracle.ordered_writes(df)
            if len(ow) == 0:
                skipped["empty"] += 1
                continue
            if not events_stream.single_speed(ow):
                skipped["multispeed"] += 1
                continue
            meta = read_meta(df_file)
            if meta is not None and not meta.stale:
                digi = meta.is_digi
            else:
                digi = raw_is_digi(df)
            if digi:
                skipped["digi"] += 1
                continue
            in_scope.add(df_file)
            irq_by_file[df_file] = int(df["irq"].min()) if len(df) else 0
        if skipped:
            self.logger.info(
                "events scope filter (single-speed, non-digi): kept %u of %u dumps, skipped %s",
                len(in_scope),
                len(df_files),
                dict(skipped),
            )
        train_files = [f for f in train_files if f in in_scope]
        val_files = [f for f in val_files if f in in_scope]
        kind_by_file = {f: k for f, k in kind_by_file.items() if f in in_scope}
        df_files = train_files + val_files
        self._tokenize_meta = TokenizeMeta(
            irq_by_file=irq_by_file,
            rotations_by_file={f: 1 for f in df_files},
            kind_by_file=kind_by_file,
            reg_widths={},
            val_subsets=(
                list(parse_eval_reglogs(eval_reglogs).keys()) if eval_reglogs else []
            ),
            format_version=events_stream.EVENT_FORMAT_VERSION,
        )

        if self.args.token_csv:
            self.logger.info("writing tokens to %s", self.args.token_csv)
            self.tokenizer.tokens.to_csv(self.args.token_csv, index=False)

        df_map_csv = self.args.df_map_csv

        def _write_map():
            if df_map_csv:
                self.logger.info("writing dataset map to %s", df_map_csv)
                self._build_df_map_frame(train_files, val_files).to_csv(
                    df_map_csv, index=False
                )
                self._write_reg_widths_sidecar(df_map_csv)

        if not self.args.tkvocab and not self.args.dataset_csv:
            _write_map()
            return

        if self.args.tkvocab:

            def worker():
                for i, df_file in enumerate(df_files):
                    try:
                        df = self._read_dump(df_file)
                    except Exception:  # pylint: disable=broad-except
                        continue
                    ids = events_dataset.dump_token_ids(df, df_file)
                    if ids:
                        yield df_file, pd.DataFrame(
                            {"n": np.asarray(ids, dtype=np.int64)}
                        ), i
                _write_map()

            self.tokenizer.train_tokenizer(worker())
        else:
            _write_map()

        if getattr(self.args, "write_blocks", True):
            self._encode_and_save_events(df_files)

    def iter_block_seqs(self):
        """Top-level train-stage iterator. Yields (kind, blocks_path, seq_meta) tuples. Tries metadata-fast-path first (df-map.csv with `irq`/`n_rotations` columns + _reg_widths.json sidecar); falls back to re-parsing the corpus. Sets self.n_vocab, self.n_words, self.reg_widths as side effects."""
        assert self.tokenizer.tokens is not None
        self.n_vocab = len(self.tokenizer.tokens["n"])
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        df_map_csv = self.args.df_map_csv
        if df_map_csv and os.path.exists(df_map_csv):
            df_map_df = pd.read_csv(df_map_csv)
            required = {"irq", "n_rotations", "kind", "dump_file"}
            if required.issubset(df_map_df.columns):
                sidecar = reg_widths_path(df_map_csv)
                if os.path.exists(sidecar):
                    self.reg_widths = self._load_reg_widths_sidecar(sidecar)
                    df_map_df = df_map_df.drop_duplicates("dump_file")
                    n_seq = 0
                    for _, row in df_map_df.iterrows():
                        df_file = row["dump_file"]
                        kind = row["kind"]
                        n_rot = (
                            int(row["n_rotations"])
                            if pd.notna(row["n_rotations"])
                            else 0
                        )
                        if not n_rot:
                            continue
                        irq = int(row["irq"])
                        for i in range(n_rot):
                            blocks_path = df_file.replace(
                                DUMP_SUFFIX, f".{i}.blocks.npy"
                            )
                            if not os.path.exists(blocks_path):
                                continue
                            yield kind, blocks_path, SeqMeta(
                                irq=irq, df_file=df_file, i=i
                            )
                            n_seq += 1
                    self.logger.info(
                        "load (cached): n_vocab=%u, %u sequences, reg widths %s",
                        self.n_vocab,
                        n_seq,
                        sorted(self.reg_widths.items()),
                    )
                    return
        dump_files = None
        reglogs = None
        kind_by_dump = {}
        if getattr(self.args, "reglog", None):
            self.logger.info("loading data from %s", self.args.reglog)
            reglogs = self.args.reglog
        elif df_map_csv and os.path.exists(df_map_csv):
            df_map_df = pd.read_csv(df_map_csv)
            if "kind" not in df_map_df.columns:
                df_map_df = df_map_df.assign(kind="train")
            df_map_df = df_map_df.drop_duplicates("dump_file")
            dump_files = df_map_df["dump_file"].tolist()
            kind_by_dump = dict(zip(df_map_df["dump_file"], df_map_df["kind"]))
            self.logger.info(
                "loading data from %s - %u files", df_map_csv, len(dump_files)
            )
        elif self.args.reglogs:
            self.logger.info("loading data from %s", self.args.reglogs)
            reglogs = self.args.reglogs
        self.n_words = 0
        n_seq = 0
        n_words = 0
        reg_max = {}
        for df_file, i, df, seq, irq, _blocks in self.load_dfs(
            reglogs=reglogs,
            dump_files=dump_files,
            max_perm=self.args.max_perm,
            encode=True,
        ):
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += len(seq) if seq is not None else 0
            n_words += len(df)
            n_seq += 1
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            if os.path.exists(blocks_path):
                kind = kind_by_dump.get(df_file, "train")
                yield kind, blocks_path, SeqMeta(irq=irq, df_file=df_file, i=i)
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        n_frac = 0
        if n_words:
            n_frac = round(self.n_words / n_words, 2)
        self.logger.info(
            "n_vocab: %u, n_words %u, n_encoded_words %u (%s), reg widths %s, %u sequences",
            self.n_vocab,
            n_words,
            self.n_words,
            n_frac,
            sorted(self.reg_widths.items()),
            n_seq,
        )

    def iter_predict_block_seqs(self):
        """Lean predict-time iterator. Yields (kind, blocks_path, seq_meta) for the single target file selected by --predict-set / --start-seq. May parse just that one file if no cached blocks exist."""
        assert self.tokenizer.tokens is not None
        assert os.path.exists(
            self.args.df_map_csv
        ), "iter_predict_block_seqs needs df-map-csv"
        df_map = pd.read_csv(self.args.df_map_csv)
        if "kind" not in df_map.columns:
            df_map = df_map.assign(kind="train")
        df_map = df_map.drop_duplicates("dump_file").reset_index(drop=True)
        predict_set = getattr(self.args, "predict_set", "train")
        kind_df = df_map[df_map["kind"] == predict_set].reset_index(drop=True)
        if kind_df.empty and predict_set == LEGACY_EVAL_SUBSET_NAME:
            present = [k for k in df_map["kind"].unique() if k != "train"]
            if present:
                fallback = "eval_a" if "eval_a" in present else sorted(present)[0]
                self.logger.info(
                    "predict_load: no '%s' rows; aliasing --predict-set to '%s' "
                    "(held-out subsets present: %s)",
                    predict_set,
                    fallback,
                    present,
                )
                predict_set = fallback
                self.args.predict_set = fallback
                kind_df = df_map[df_map["kind"] == predict_set].reset_index(drop=True)
        if kind_df.empty:
            raise ValueError(
                f"df_map.csv has no '{predict_set}' files; check --predict-set"
            )
        start_seq = getattr(self.args, "start_seq", 0)
        if start_seq >= len(kind_df):
            raise ValueError(
                f"--start-seq {start_seq} out of range "
                f"({len(kind_df)} {predict_set} files)"
            )
        target_row = kind_df.iloc[start_seq]
        target_file = target_row["dump_file"]
        self.n_vocab = len(self.tokenizer.tokens["n"])
        if self.args.tkvocab:
            self.n_vocab = self.args.tkvocab
        sidecar = reg_widths_path(self.args.df_map_csv)
        cached = (
            "irq" in target_row.index
            and pd.notna(target_row.get("irq"))
            and os.path.exists(sidecar)
        )
        if cached:
            self.reg_widths = self._load_reg_widths_sidecar(sidecar)
            irq = int(target_row["irq"])
            blocks_path = target_file.replace(DUMP_SUFFIX, ".0.blocks.npy")
            assert os.path.exists(
                blocks_path
            ), f"missing {blocks_path} -- did the tokenize stage run?"
            yield predict_set, blocks_path, SeqMeta(irq=irq, df_file=target_file, i=0)
            self.args.start_seq = 0
            self.logger.info(
                "predict_load (cached): %s rotation 0, n_vocab=%u, reg widths %s",
                target_file,
                self.n_vocab,
                sorted(self.reg_widths.items()),
            )
            return
        self.logger.info(
            "predict_load: parsing only %s (1/%u %s files in df_map.csv)",
            target_file,
            len(kind_df),
            predict_set,
        )
        self.n_words = 0
        n_seq = 0
        reg_max = {}
        for df_file, i, df, seq, irq, _blocks in self.load_dfs(
            dump_files=[target_file], max_perm=1, encode=True
        ):
            reg_max = self.tokenizer.get_reg_max(df, reg_max)
            self.n_words += len(seq) if seq is not None else 0
            n_seq += 1
            blocks_path = df_file.replace(DUMP_SUFFIX, f".{i}.blocks.npy")
            if os.path.exists(blocks_path):
                yield predict_set, blocks_path, SeqMeta(irq=irq, df_file=df_file, i=i)
        self.reg_widths = self.tokenizer.get_reg_width_from_max(reg_max)
        self.args.start_seq = 0
        self.logger.info(
            "predict-only lean load: n_vocab=%u, n_words=%u, %u sequences, reg widths %s",
            self.n_vocab,
            self.n_words,
            n_seq,
            sorted(self.reg_widths.items()),
        )
