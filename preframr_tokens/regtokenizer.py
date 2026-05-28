import concurrent.futures
import difflib
import logging
import multiprocessing
import string
import zstandard as zstd
from tqdm import tqdm
from tokenizers import Tokenizer
import numpy as np
import pandas as pd
from preframr_tokens.macros.loops import (
    EXTRA_ISOLATION_HEAD_OPS,
    MULTI_ROW_MACRO_HEAD_OPS,
)
from preframr_tokens.stfconstants import (
    DUMP_SUFFIX,
    FRAME_REG,
    FREQ_NUDGE_OP,
    FREQ_NUDGE_SUBREG_HI,
    FREQ_NUDGE_SUBREG_LO,
    FREQ_ONSET_OP,
    FREQ_TRAJ_OP,
    FREQ_TRAJ_REGS,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
    OP_PDTYPE,
    PAD_REG,
    SET_OP,
    SUBREG_PDTYPE,
    TOKEN_KEYS,
    TOKEN_PDTYPE,
    UNICODE_BASE,
    UNI_SUFFIX,
    VAL_PDTYPE,
)

_FREQ_REGS_FROZEN = frozenset(FREQ_TRAJ_REGS)
_FT_V0_SUBREGS = frozenset({FT_SUBREG_V0_HI, FT_SUBREG_V0_LO})
_NUDGE_PITCH_SUBREGS = frozenset({FREQ_NUDGE_SUBREG_HI, FREQ_NUDGE_SUBREG_LO})


def is_melody_pitch_atom(op, reg, subreg) -> bool:
    """True for a melodic-pitch atom: op45 V0 (FT_SUBREG_V0_HI/LO) or op48 FREQ_ONSET or
    op47 FREQ_NUDGE HI/LO -- all restricted to freq regs (FREQ_TRAJ_REGS = 0/7/14). The
    melody-merge-split rule uses this predicate to detect cross-boundary Unigram merges.
    """
    if int(reg) not in _FREQ_REGS_FROZEN:
        return False
    op = int(op)
    subreg = int(subreg)
    if op == FREQ_TRAJ_OP and subreg in _FT_V0_SUBREGS:
        return True
    if op == FREQ_ONSET_OP:
        return True
    if op == FREQ_NUDGE_OP and subreg in _NUDGE_PITCH_SUBREGS:
        return True
    return False


def is_freq_onset_atom(op, reg, subreg) -> bool:
    """A FREQ_TRAJ V0 onset atom: op45, freq reg (0/7/14), V0_HI/V0_LO subreg.
    Strict subset of ``is_melody_pitch_atom``; the onset-loss-weight buffer uses this
    narrower predicate to up-weight only the FREQ_TRAJ V0 class."""
    return (
        int(op) == FREQ_TRAJ_OP
        and int(reg) in _FREQ_REGS_FROZEN
        and int(subreg) in _FT_V0_SUBREGS
    )


def split_cross_boundary_merges(
    seq, decode_to_base_ids, base_to_unigram_id, is_melody, n_atoms, dtype=np.int32
):
    """Expand any merged Unigram token whose decoded base atoms cross the melody/non-melody
    boundary back into its base atoms. Pure-melody and pure-non-melody merges (and single
    base atoms) are kept. Pure: takes ``decode_to_base_ids(uid) -> list[int]``,
    ``base_to_unigram_id(bid) -> int|None`` (the single-atom Unigram id for that base id),
    and ``is_melody(bid) -> bool``. Returns a possibly-longer 1-D numpy array of ids."""
    out = []
    for tid in seq:
        tid = int(tid)
        base_ids = [int(b) for b in decode_to_base_ids(tid)]
        valid = [b for b in base_ids if 0 <= b < n_atoms]
        if len(valid) <= 1:
            out.append(tid)
            continue
        melody_n = sum(1 for b in valid if is_melody(b))
        if melody_n == 0 or melody_n == len(valid):
            out.append(tid)
            continue
        for b in valid:
            u = base_to_unigram_id(b)
            out.append(int(u) if u is not None else tid)
    return np.asarray(out, dtype=dtype)


from preframr_tokens.train_worker import train_worker

__all__ = ["RegTokenizer"]

UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"
SPLITCHS = [ord(i) for i in string.punctuation]
SPLITTERS = len(SPLITCHS)


class RegTokenizer:
    def __init__(self, args, tokens, logger=logging):
        self.args = args
        self.logger = logger
        self.tokens = tokens
        self.tkmodel = None
        self.frame_tokens = []
        self.splitters = SPLITTERS
        self.splitchs = SPLITCHS

    def _resync_splitters_from_tokens(self):
        """Single source of truth for the ``splitters`` invariant:
        ``splitters = min(SPLITTERS, frame_tokens)``.
        """
        if self.tokens is None or not len(self.tokens):
            return
        frame_tokens = len(self.tokens[self.tokens["reg"] == FRAME_REG])
        self.splitters = min(self.splitters, frame_tokens)

    def load(self, tkmodel, tokens):
        self.tokens = tokens
        self._resync_splitters_from_tokens()
        if tkmodel:
            self.logger.info("loading tokenizer model")
            self.tkmodel = Tokenizer.from_str(tkmodel)

    def encode_unicode(self, tokens, dtype=np.uint32):
        t = np.array(tokens, dtype=dtype)
        m = t >= self.splitters
        t[m] += UNICODE_BASE
        surr_mask = t >= 0xD800
        t[surr_mask] += 0x800
        for c in range(self.splitters):
            t = np.where(t == c, self.splitchs[c], t)
        encoded = "".join([chr(int(i)) for i in t])
        assert len(encoded) == t.shape[0]
        return encoded

    def decode_unicode(self, encoded_tokens, dtype=np.uint32):
        encoded = [ord(i) for i in encoded_tokens]
        t = np.array(encoded, dtype=dtype)
        assert len(encoded_tokens) == len(encoded)
        assert len(encoded_tokens) == t.shape[0]
        for c in range(self.splitters):
            t = np.where(t == self.splitchs[c], c, t)
        surr_mask = t >= 0xE000
        t[surr_mask] -= 0x800
        m = t >= self.splitters
        t[m] -= UNICODE_BASE
        return t

    def encode(self, tokens, dtype=np.uint32):
        if self.tkmodel:
            encoded = self.tkmodel.encode(self.encode_unicode(tokens, dtype=dtype))
            return np.array(encoded.ids, dtype=dtype)
        return tokens

    def decode(self, encoded_tokens, dtype=np.uint32):
        if self.tkmodel:
            return self.decode_unicode(self.tkmodel.decode(encoded_tokens), dtype=dtype)
        return encoded_tokens

    def split_melody_merges(self, seq):
        """Expand Unigram merges that cross the melody/non-melody atom boundary; pure-melody
        and pure-non-melody merges are kept. Opt-in (``args.melody_merge_split``); byte-exact
        (decode is the inverse of merge in the Unigram vocab). Caches per-tokenizer maps on
        first call so subsequent block encodes are O(seq)."""
        if not self.tkmodel or self.tokens is None or not len(self.tokens):
            return seq
        if not hasattr(self, "_melody_atom_unigram_id"):
            n_atoms = len(self.tokens)
            atom_ns = self.tokens["n"].astype(np.int64).to_numpy()
            uni_ids = [None] * n_atoms
            for i in range(n_atoms):
                ch = chr(UNICODE_BASE + int(atom_ns[i]))
                uni_ids[i] = self.tkmodel.token_to_id(ch)
            self._melody_atom_unigram_id = uni_ids
            self._melody_atom_mask = [
                is_melody_pitch_atom(
                    self.tokens.iloc[i]["op"],
                    self.tokens.iloc[i]["reg"],
                    self.tokens.iloc[i]["subreg"],
                )
                for i in range(n_atoms)
            ]
        n_atoms = len(self.tokens)
        return split_cross_boundary_merges(
            seq,
            decode_to_base_ids=lambda uid: self.decode([uid]),
            base_to_unigram_id=lambda b: (
                self._melody_atom_unigram_id[b] if 0 <= b < n_atoms else None
            ),
            is_melody=lambda b: (
                self._melody_atom_mask[b] if 0 <= b < n_atoms else False
            ),
            n_atoms=n_atoms,
            dtype=seq.dtype,
        )

    def _all_atom_chars(self):
        """Every unique atom char that ``encode_unicode`` can emit for
        this tokens table.
        """
        if self.tokens is None or not len(self.tokens):
            return []
        atom_ids = self.tokens["n"].astype(np.int64).to_numpy()
        unicode_str = self.encode_unicode(atom_ids)
        return sorted(set(unicode_str))

    def _isolation_chars_for_ops(self, op_filters):
        """Unicode chars for atomic ids matching any of ``op_filters``."""
        if self.tokens is None or not len(self.tokens):
            return ""
        op_arr = self.tokens["op"].astype(np.int64).to_numpy()
        sr_arr = self.tokens["subreg"].fillna(-1).astype(np.int64).to_numpy()
        match = np.zeros(len(self.tokens), dtype=bool)
        for entry in op_filters:
            if isinstance(entry, tuple):
                op, subreg = entry
                match |= (op_arr == int(op)) & (sr_arr == int(subreg))
            else:
                match |= op_arr == int(entry)
        atomic_ids = self.tokens.loc[match, "n"].astype(np.int64).to_numpy()
        if not atomic_ids.size:
            return ""
        unicode_str = self.encode_unicode(atomic_ids)
        return "".join(sorted(set(unicode_str)))

    def train_tokenizer(self, dfs):
        frame_tokens = 1
        if self.tokens is not None and len(self.tokens):
            frame_tokens = len(self.tokens[self.tokens["reg"] == FRAME_REG])
        self.logger.info(
            f"feeding {self.args.tokenizer} tokenizer with {frame_tokens} frame tokens"
        )
        self._resync_splitters_from_tokens()
        isolation_chars = self._isolation_chars_for_ops(
            MULTI_ROW_MACRO_HEAD_OPS + EXTRA_ISOLATION_HEAD_OPS
        )
        if isolation_chars:
            self.logger.info(
                "isolating %u atomic chars (head-row macro landmarks) "
                "from unigram merges",
                len(isolation_chars),
            )

        def write_uni(t):
            df_file, df, i = t
            uni_file = df_file.replace(DUMP_SUFFIX, f".{i}{UNI_SUFFIX}")
            orig_seq = df["n"].to_numpy()
            encoded = self.encode_unicode(orig_seq)
            with zstd.open(uni_file, "w") as f:
                f.write(encoded)
            return uni_file

        uni_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
            futures = [p.submit(write_uni, df) for df in dfs]
            for future in concurrent.futures.as_completed(futures):
                uni_files.append(future.result())

        initial_alphabet = self._all_atom_chars()
        self.logger.info(
            "seeding initial_alphabet with %u atom chars",
            len(initial_alphabet),
        )
        self.logger.info("running tokenizer")
        ctx = multiprocessing.get_context("spawn")
        p = ctx.Process(
            target=train_worker,
            args=(
                self.args.tokenizer,
                self.args.tkvocab,
                self.args.tkmodel,
                uni_files,
                initial_alphabet,
                isolation_chars,
            ),
        )
        p.start()
        p.join()
        assert p.exitcode == 0, p.exitcode

        self.tkmodel = Tokenizer.from_file(self.args.tkmodel)
        assert self.tkmodel.get_vocab_size() == self.args.tkvocab, (
            self.tkmodel.get_vocab_size(),
            self.args.tkvocab,
        )

    def token_metadata(self):
        if self.tkmodel:
            metadata = []
            for t in range(self.args.tkvocab):
                decoded = [self.tokens.iloc[x] for x in self.decode([t])]
                metadata.append(
                    ",".join((f"{x.op} {x.reg} {x.subreg} {x.val}" for x in decoded))
                )
            return metadata
        metadata = [
            f"{x.op} {x.reg} {x.subreg} {x.val}" for x in self.tokens.itertuples()
        ]
        return metadata

    def crunch_tokens(self):
        frame_tokens = pd.concat(self.frame_tokens, ignore_index=True)
        frame_tokens["count"] = frame_tokens.groupby(TOKEN_KEYS)["count"].transform(
            "sum"
        )
        self.frame_tokens = [
            frame_tokens.drop_duplicates(TOKEN_KEYS)
            .sort_values(TOKEN_KEYS)
            .reset_index(drop=True)
        ]

    def accumulate_tokens(self, df, df_file):
        frame_tokens = df[TOKEN_KEYS].copy().reset_index(drop=True)
        for col in TOKEN_KEYS:
            frame_tokens[col] = frame_tokens[col].astype("Int64")
        frame_tokens = frame_tokens.join(
            frame_tokens.value_counts(), on=TOKEN_KEYS
        ).drop_duplicates(TOKEN_KEYS)
        self.frame_tokens.append(frame_tokens)
        if len(self.frame_tokens) > 64:
            self.crunch_tokens()

    def make_tokens(self):
        self.logger.info("making tokens")
        self.crunch_tokens()
        tokens = self.frame_tokens[0]
        pad_row = pd.DataFrame(
            [{"op": 0, "reg": PAD_REG, "subreg": -1, "val": 0, "count": 0}]
        )
        tokens = pd.concat([pad_row, tokens], ignore_index=True)
        tokens["n"] = tokens.index
        tokens = tokens.sort_values(["n"])
        tokens = tokens.astype(
            {
                "val": VAL_PDTYPE,
                "n": TOKEN_PDTYPE,
                "subreg": SUBREG_PDTYPE,
                "op": OP_PDTYPE,
            }
        )
        assert tokens["reg"].max() < 256
        return tokens

    def _merged_and_missing(self, tokens, df):
        m = df["reg"] == FRAME_REG
        irq = df[m]["diff"].iloc[0]
        df.loc[m, "diff"] = 0
        df_norm = df.copy()
        tokens_norm = tokens.copy()
        for col in TOKEN_KEYS:
            df_norm[col] = df_norm[col].astype("Int64")
            tokens_norm[col] = tokens_norm[col].astype("Int64")
        df = df_norm.merge(tokens_norm, on=TOKEN_KEYS, how="left")
        df.loc[m, "diff"] = irq
        missing_tokens = (
            df[df["n"].isna()].drop_duplicates().sort_values(["reg", "val"])
        )
        return df, missing_tokens

    def merge_tokens(self, tokens, dfs):
        self.logger.info("merging tokens")
        merged_dfs = []
        for df in tqdm(dfs, ascii=True):
            merged_df = self.merge_token_df(tokens, df)
            if merged_df is None:
                return None
            merged_dfs.append(merged_df)
        return merged_dfs

    def merge_token_df(self, tokens, df):
        orig_cols, orig_dtypes = df.columns, df.dtypes
        df, missing_tokens = self._merged_and_missing(tokens, df)
        if not missing_tokens.empty:
            df = self._decompose_missing_via_registry(df, tokens, missing_tokens)
            df = df[orig_cols].astype(orig_dtypes)
            df, missing_tokens = self._merged_and_missing(tokens, df)
        if not missing_tokens.empty:
            for missing_token in missing_tokens.itertuples():
                reg = missing_token.reg
                val = missing_token.val
                op = getattr(missing_token, "op", SET_OP)
                subreg = getattr(missing_token, "subreg", -1)
                key_tokens = tokens[
                    (tokens["op"] == op)
                    & (tokens["reg"] == reg)
                    & (tokens["subreg"] == subreg)
                ]
                if key_tokens.empty:
                    self.logger.error(
                        "no token for op=%u reg=%d subreg=%d val=%u; "
                        "alphabet has no rows for this (op, reg, subreg)",
                        int(op),
                        int(reg),
                        int(subreg),
                        int(val),
                    )
                    raise KeyError(
                        f"missing token op={int(op)} reg={int(reg)} "
                        f"subreg={int(subreg)} val={int(val)}"
                    )
                compare_tokens = key_tokens.copy()
                compare_tokens["diff_val"] = (compare_tokens["val"] - val).abs()
                best_token = compare_tokens[
                    compare_tokens["diff_val"] == compare_tokens["diff_val"].min()
                ].iloc[0]
                best_val = best_token.val
                self.logger.info(
                    "substitute op=%u reg=%d subreg=%d val=%u with val=%u",
                    int(op),
                    int(reg),
                    int(subreg),
                    int(val),
                    int(best_val),
                )
                df.loc[
                    (df["op"] == op)
                    & (df["reg"] == reg)
                    & (df["subreg"] == subreg)
                    & (df["val"] == val),
                    "val",
                ] = best_val
            df = df[orig_cols].astype(orig_dtypes)
            df, missing_tokens = self._merged_and_missing(tokens, df)
            assert missing_tokens.empty, missing_tokens
            return df
        return df

    def _decompose_missing_via_registry(self, df, tokens, missing_tokens):
        from preframr_tokens.macros.transform import (
            _REGISTRY,
            ensure_default_transforms_registered,
        )

        ensure_default_transforms_registered()
        if df.empty or not _REGISTRY:
            return df
        by_op = {}
        for transform_cls in _REGISTRY.values():
            if not transform_cls.DECOMPOSES_TO_ATOMS:
                continue
            for op in transform_cls.OP_CODES:
                by_op[int(op)] = transform_cls
        if not by_op:
            return df
        token_keys = tokens[TOKEN_KEYS].drop_duplicates()
        token_set = set(map(tuple, token_keys.itertuples(index=False, name=None)))
        rows_out = []
        replacement_log = 0
        op_arr = df["op"].to_numpy()
        reg_arr = df["reg"].to_numpy()
        subreg_arr = df["subreg"].to_numpy() if "subreg" in df.columns else None
        val_arr = df["val"].to_numpy()
        missing_keys = {
            (int(t.op), int(t.reg), int(getattr(t, "subreg", -1)), int(t.val))
            for t in missing_tokens.itertuples()
        }
        cols = list(df.columns)
        for i, row in enumerate(df.itertuples(index=False)):
            op = int(op_arr[i])
            reg = int(reg_arr[i])
            subreg = int(subreg_arr[i]) if subreg_arr is not None else -1
            val = int(val_arr[i])
            key = (op, reg, subreg, val)
            if key not in missing_keys or op not in by_op:
                rows_out.append({c: getattr(row, c) for c in cols})
                continue
            transform = by_op[op]()
            synth = pd.DataFrame([{c: getattr(row, c) for c in cols}]).astype(
                df.dtypes.to_dict()
            )
            expanded = transform.inverse(synth)
            if expanded.equals(synth):
                rows_out.append({c: getattr(row, c) for c in cols})
                continue
            all_expanded_in_alphabet = True
            expanded_rows = expanded.to_dict("records")
            for er in expanded_rows:
                ek = (
                    int(er.get("op", SET_OP)),
                    int(er["reg"]),
                    int(er.get("subreg", -1)),
                    int(er["val"]),
                )
                if ek not in token_set:
                    all_expanded_in_alphabet = False
                    break
            if not all_expanded_in_alphabet:
                rows_out.append({c: getattr(row, c) for c in cols})
                continue
            rows_out.extend(expanded_rows)
            replacement_log += 1
        if replacement_log:
            self.logger.info(
                "registry-decomposed %u missing-token rows", replacement_log
            )
        rebuilt = pd.DataFrame(rows_out, columns=cols).astype(df.dtypes.to_dict())
        return rebuilt

    def validate_encoding(self, df_file, seq):
        if not self.args.tkvocab:
            return seq
        orig_seq = seq.copy()
        seq = self.encode(orig_seq, dtype=np.int64)
        decoded_seq = self.decode(seq, dtype=np.int64)
        if not np.array_equal(orig_seq, decoded_seq):
            for i, (orig, decoded) in enumerate(zip(orig_seq, decoded_seq)):
                if orig == decoded:
                    continue
                a = [str(i) for i in orig_seq]
                b = [str(i) for i in decoded_seq]
                d = "\n".join(difflib.context_diff(a, b))
                print(d)
                assert False, (
                    df_file,
                    i,
                    orig,
                    decoded,
                    self.tokens.iloc[int(orig)],
                )
        return seq

    def get_reg_max(self, df, reg_max):
        df_max = df.groupby("reg")["val"].max().to_dict()
        for reg, val_max in df_max.items():
            reg_max[reg] = max(val_max, reg_max.get(reg, 0))
        return reg_max

    def get_reg_width_from_max(self, reg_max):
        reg_widths = {}
        for reg, val in reg_max.items():
            for width in range(1, 8):
                if val < 2 ** (8 * width):
                    reg_widths[int(reg)] = width
                    break
            assert reg_widths[int(reg)]
        return reg_widths
