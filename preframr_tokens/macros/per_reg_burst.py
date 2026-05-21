"""Per-reg burst layer: detect DIFF / FLIP / REPEAT runs on a single
register's value sequence.
"""

import pandas as pd

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.reglog_helpers import filter_match, freq_match, pcm_match
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FC_LO_REG,
    FLIP_OP,
    MODEL_PDTYPE,
    SET_OP,
    VOICE_REG_SIZE,
)


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
        for f, c in (("repeat", "val"), ("flip", "aval")):
            cols = ["cf", c]
            v_df.loc[
                (v_df[cols] == v_df[cols].shift(1)).all(axis=1)
                | (v_df[cols] == v_df[cols].shift(-1)).all(axis=1),
                f,
            ] = (
                v_df["cf"] * v_df[c]
            )
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

    DEFAULT_OPCODES = (DIFF_OP, FLIP_OP)

    def __init__(self, opcodes=None):
        self.opcodes = (
            list(opcodes) if opcodes is not None else list(self.DEFAULT_OPCODES)
        )

    def apply(self, df, args=None):
        from preframr_tokens.reglogparser import last_reg_val_frame, norm_df

        cents = getattr(args, "cents", 50) if args is not None else 50
        orig_df = df
        had_op = "op" in orig_df.columns
        df = norm_df(orig_df)
        if not had_op:
            df["op"] = SET_OP

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
