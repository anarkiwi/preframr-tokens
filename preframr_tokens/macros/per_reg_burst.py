"""Per-reg burst layer: detect DIFF / FLIP / REPEAT runs on a single
register's value sequence.
"""

__all__ = ["PerRegBurstPass"]

import numpy as np
import pandas as pd

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.reg_match import filter_match, freq_match, pcm_match
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FC_LO_REG,
    FLIP_OP,
    MODEL_PDTYPE,
    SET_OP,
    STAMP_REF_OP,
    STAMP_REL_REF_OP,
    VOICE_REG_SIZE,
    VOICES,
)

_STAMP_REF_OPS = (STAMP_REF_OP, STAMP_REL_REF_OP)


def _stamp_barrier_sets(reg, op, frame):
    """(reg, frame) of voice freq/PWM SETs whose previous write on that voice was a STAMP_REF/REL:
    the stamp replays its own value at decode, so delta-encoding the SET against the prior literal
    SET (which the stamp consumed) mis-bases it -- keep these absolute so the value lands exactly.
    """
    reg = np.asarray(reg)
    op = np.asarray(op)
    frame = np.asarray(frame)
    barrier = set()
    for v in range(VOICES):
        base = v * VOICE_REG_SIZE
        is_stamp = (reg == base) & np.isin(op, _STAMP_REF_OPS)
        for off in (0, 2):
            is_set = (reg == base + off) & (op == SET_OP)
            idx = np.nonzero(is_stamp | is_set)[0]
            if idx.size == 0:
                continue
            prev_stamp = np.concatenate(([False], is_stamp[idx][:-1]))
            for i in idx[is_set[idx] & prev_stamp]:
                barrier.add((base + off, int(frame[i])))
    return barrier


def _add_change_reg(df, change_df, minchange, opcodes):
    change_dfs = []
    change_df["val"] -= change_df["pval"]
    change_df = change_df.drop("pval", axis=1)
    for reg in change_df["reg"].unique():
        v_df = change_df[change_df["reg"] == reg].copy()
        v_df = v_df.sort_values(["n", "val"])
        v_df["cpf"] = v_df.groupby("f").transform("size")
        v_df["aval"] = v_df["val"].abs()
        v_df = v_df[
            (v_df["aval"] > 0) & (v_df["aval"] <= minchange) & (v_df["cpf"] == 1)
        ]
        df = df[~df["n"].isin(v_df["n"])]
        m_df = v_df[v_df["f"].diff().fillna(1) > 1].copy()
        m_df["op"] = DIFF_OP
        v_df = v_df[~v_df["n"].isin(m_df["n"])]
        change_dfs.append(m_df)
        v_df["cf"] = (
            (
                (v_df["f"].diff().fillna(1) > 1)
                .astype(MODEL_PDTYPE)
                .cumsum()
                .astype(MODEL_PDTYPE)
            )
            * 255
        ) + 1
        v_df[["repeat", "flip", "begin", "end"]] = 0
        rcols = ["cf", "val"]
        v_df.loc[
            (v_df[rcols] == v_df[rcols].shift(1)).all(axis=1)
            | (v_df[rcols] == v_df[rcols].shift(-1)).all(axis=1),
            "repeat",
        ] = (
            v_df["cf"] * v_df["val"]
        )
        same_cf_prev = v_df["cf"] == v_df["cf"].shift(1)
        same_cf_next = v_df["cf"] == v_df["cf"].shift(-1)
        alt = (same_cf_prev & (v_df["val"] == -v_df["val"].shift(1))) | (
            same_cf_next & (v_df["val"] == -v_df["val"].shift(-1))
        )
        v_df.loc[alt, "flip"] = v_df["cf"] * v_df["aval"]
        for f, of in (("repeat", "flip"), ("flip", "repeat")):
            m = v_df[f] != 0
            v_df.loc[m & (v_df[f] != v_df[f].shift(1)), "begin"] = v_df["n"]
            v_df.loc[m & (v_df[f] != v_df[f].shift(-1)), "end"] = v_df["n"]
            for shift in (0, 1):
                v_df.loc[
                    ((v_df["end"] != 0) & (v_df["begin"].shift(shift) != 0))
                    | ((v_df["begin"] != 0) & (v_df["end"].shift(-shift) != 0)),
                    ["begin", "end", f],
                ] = 0
            v_df.loc[v_df[f] != 0, of] = 0

        for f, op in (("flip", FLIP_OP),):
            if op in opcodes:
                d_df = v_df[
                    (v_df[f] != 0) & ((v_df["begin"] != 0) | (v_df["end"] != 0))
                ].copy()
                v_df = v_df[v_df[f] == 0]
                if d_df.empty:
                    continue
                assert d_df["begin"].iloc[0] != 0, d_df
                assert d_df["end"].iloc[-1] != 0, d_df
                d_df.loc[d_df["end"] != 0, "val"] = 0
                d_df["op"] = op
                change_dfs.append(d_df.copy())

        v_df["op"] = DIFF_OP
        change_dfs.append(v_df)

    df = df.drop("pval", axis=1)
    return df, change_dfs


class PerRegBurstPass(MacroPass):
    """Detect DIFF / FLIP / REPEAT runs across consecutive frames on
    the freq / pcm / filter reg families. Emits the macro op rows the
    rest of the encoder pipeline depends on; rows the detector skips
    stay literal SETs.
    """

    GATE_FLAGS = frozenset({"freq_trajectory_pass"})

    DEFAULT_OPCODES = (DIFF_OP, FLIP_OP)

    def __init__(self, opcodes=None):
        self.opcodes = (
            list(opcodes) if opcodes is not None else list(self.DEFAULT_OPCODES)
        )

    def apply(self, df, args=None):
        from preframr_tokens.reglogparser import last_reg_val_frame, norm_df

        if args is not None and getattr(args, "freq_trajectory_pass", True):
            if df is not None and "op" not in df.columns:
                df = df.copy()
                df["op"] = int(SET_OP)
            return df
        cents = getattr(args, "cents", 50) if args is not None else 50
        orig_df = df
        had_op = "op" in orig_df.columns
        df = norm_df(orig_df)
        if not had_op:
            df["op"] = SET_OP
        barrier = _stamp_barrier_sets(
            df["reg"].to_numpy(), df["op"].to_numpy(), df["f"].to_numpy()
        )

        if had_op:
            pivot_src = orig_df[orig_df["op"] == SET_OP].reset_index(drop=True)
        else:
            pivot_src = orig_df
        freq_df, pcm_df, filter_df = last_reg_val_frame(pivot_src, [0, 2, FC_LO_REG])
        freq_df["reg"] = freq_df["v"] * VOICE_REG_SIZE
        pcm_df["reg"] = pcm_df["v"] * VOICE_REG_SIZE + 2
        filter_df["reg"] = FC_LO_REG

        all_change_dfs = []
        for xdf, matcher, minchange in (
            (freq_df, freq_match, int((2 * 12) * 100 / cents)),
            (pcm_df, pcm_match, 64),
            (filter_df, filter_match, 512),
        ):
            df = df.merge(xdf[["reg", "f", "pval"]], how="left", on=["f", "reg"])
            xdf = df[matcher(df) & (df["op"] == SET_OP)].copy()
            if barrier:
                xdf = xdf[
                    [
                        (int(r), int(fr)) not in barrier
                        for r, fr in zip(xdf["reg"], xdf["f"])
                    ]
                ]
            df, change_dfs = _add_change_reg(
                df, xdf, minchange=minchange, opcodes=self.opcodes
            )
            all_change_dfs.extend(change_dfs)

        df = (
            pd.concat([df] + all_change_dfs, ignore_index=True)
            .sort_values(["n"])
            .reset_index(drop=True)
        )
        out_cols = list(orig_df.columns) + ([] if had_op else ["op"])
        df = df[out_cols].reset_index(drop=True)
        return df
