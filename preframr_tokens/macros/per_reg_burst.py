"""Per-reg burst layer: detect DIFF / FLIP / REPEAT runs on a single
register's value sequence.
"""

__all__ = ["PerRegBurstPass"]

import os

import numpy as np
import pandas as pd

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.reg_match import filter_match, freq_match, pcm_match
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FC_LO_REG,
    FLIP_OP,
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


def _classify_runs(frames, vals, use_flip):
    """Classify a per-reg signed-delta sequence (frames ascending) into DIFF / FLIP. A maximal
    contiguous-frame strictly-alternating run of length >= 3 (d,-d,d,...) becomes FLIP(d) at its start
    and FLIP(0) at its end; the intermediate frames auto-flip at decode and are dropped. Every other
    delta is DIFF. Lossless by construction. Returns (ops, out_vals, drop) aligned to the input rows.
    """
    n = len(frames)
    ops = [int(DIFF_OP)] * n
    out_vals = [int(v) for v in vals]
    drop = [False] * n
    i = 0
    while i < n:
        j = i
        if use_flip:
            while (
                j + 1 < n
                and int(frames[j + 1]) == int(frames[j]) + 1
                and int(vals[j + 1]) == -int(vals[j])
            ):
                j += 1
        if use_flip and j - i + 1 >= 3:
            ops[i] = int(FLIP_OP)
            ops[j] = int(FLIP_OP)
            out_vals[j] = 0
            for k in range(i + 1, j):
                drop[k] = True
            i = j + 1
        else:
            i += 1
    return ops, out_vals, drop


def _fallback_enabled(args):
    """The per-register lossless fallback is a tool, enabled on demand (env PREFRAMR_PERREG_FALLBACK or
    args.perreg_lossless_fallback). Off by default: a register the delta encoder cannot round-trip (a
    rare pval mis-base) then stays as a low-residual lossy encoding the audit flags; on, it is re-encoded
    as literal SET so the pass is byte-exact (at a higher residual, since literal freq is uncaptured).
    """
    if os.environ.get("PREFRAMR_PERREG_FALLBACK"):
        return True
    return bool(args is not None and getattr(args, "perreg_lossless_fallback", False))


def _lossy_regs(orig_df, result, had_op):
    """Registers whose decoded per-frame state the re-encoding failed to reproduce (a rare pval
    mis-base). PerRegBurst re-encodes only these as literal SET, keeping every other register's delta
    compression -- a per-register lossless guarantee that doesn't nuke a whole tune's freq encoding.
    """
    from preframr_tokens.audit_primitives import register_state

    before = register_state(orig_df if had_op else orig_df.assign(op=int(SET_OP)))
    after = register_state(result)
    if before.shape != after.shape:
        return frozenset(range(int(before.shape[1])))
    diff = (before != after).any(axis=0)
    return frozenset(int(r) for r in np.nonzero(diff)[0])


def _add_change_reg(df, change_df, minchange, opcodes, skip_regs=frozenset()):
    change_dfs = []
    change_df["val"] -= change_df["pval"]
    change_df = change_df.drop("pval", axis=1)
    use_flip = FLIP_OP in opcodes
    for reg in change_df["reg"].unique():
        if int(reg) in skip_regs:
            continue
        v_df = change_df[change_df["reg"] == reg].copy()
        v_df = v_df.sort_values(["n", "val"])
        v_df["cpf"] = v_df.groupby("f").transform("size")
        v_df["aval"] = v_df["val"].abs()
        v_df = v_df[
            (v_df["aval"] > 0) & (v_df["aval"] <= minchange) & (v_df["cpf"] == 1)
        ]
        if v_df.empty:
            continue
        df = df[~df["n"].isin(v_df["n"])]
        ops, out_vals, drop = _classify_runs(
            v_df["f"].to_numpy(), v_df["val"].to_numpy(), use_flip
        )
        v_df = v_df.assign(op=ops, val=out_vals)
        v_df = v_df[[not d for d in drop]]
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
        from preframr_tokens.reglogparser import norm_df

        if args is not None and getattr(args, "freq_trajectory_pass", True):
            if df is not None and "op" not in df.columns:
                df = df.copy()
                df["op"] = int(SET_OP)
            return df
        cents = getattr(args, "cents", 50) if args is not None else 50
        orig_df = df
        had_op = "op" in orig_df.columns
        nd = norm_df(orig_df)
        if not had_op:
            nd["op"] = SET_OP
        barrier = _stamp_barrier_sets(
            nd["reg"].to_numpy(), nd["op"].to_numpy(), nd["f"].to_numpy()
        )
        result = self._encode(orig_df, nd, had_op, cents, barrier, frozenset())
        if _fallback_enabled(args):
            lossy = _lossy_regs(orig_df, result, had_op)
            if lossy:
                result = self._encode(orig_df, nd, had_op, cents, barrier, lossy)
        return result

    def _encode(self, orig_df, nd, had_op, cents, barrier, skip_regs):
        from preframr_tokens.reglogparser import last_reg_val_frame

        pivot_src = (
            orig_df[orig_df["op"] == SET_OP].reset_index(drop=True)
            if had_op
            else orig_df
        )
        freq_df, pcm_df, filter_df = last_reg_val_frame(pivot_src, [0, 2, FC_LO_REG])
        freq_df["reg"] = freq_df["v"] * VOICE_REG_SIZE
        pcm_df["reg"] = pcm_df["v"] * VOICE_REG_SIZE + 2
        filter_df["reg"] = FC_LO_REG
        df = nd.copy()
        all_change_dfs = []
        for xdf, matcher, minchange in (
            (freq_df, freq_match, int((2 * 12) * 100 / cents)),
            (pcm_df, pcm_match, 64),
            (filter_df, filter_match, 512),
        ):
            df = df.merge(xdf[["reg", "f", "pval"]], how="left", on=["f", "reg"])
            cand = df[matcher(df) & (df["op"] == SET_OP)].copy()
            if barrier:
                cand = cand[
                    [
                        (int(r), int(fr)) not in barrier
                        for r, fr in zip(cand["reg"], cand["f"])
                    ]
                ]
            df, change_dfs = _add_change_reg(
                df, cand, minchange=minchange, opcodes=self.opcodes, skip_regs=skip_regs
            )
            all_change_dfs.extend(change_dfs)
        df = (
            pd.concat([df] + all_change_dfs, ignore_index=True)
            .sort_values(["n"])
            .reset_index(drop=True)
        )
        out_cols = list(orig_df.columns) + ([] if had_op else ["op"])
        return df[out_cols].reset_index(drop=True)
