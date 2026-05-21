"""Move frame-lonely SLOPE/PRESET ops back one frame when the previous
frame flips gate on the same voice; rewrite the op to its SHIFTED
variant whose decoder defers execution by one frame. Audio-identical;
BPE/Unigram observes the (gate-flip, shifted-op) bigram in one frame."""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.stfconstants import (
    BASE_TO_SHIFTED_OP,
    DELAY_REG,
    FC_PRESET_OP,
    FRAME_REG,
    PRESET_OPS,
    SET_OP,
    SLOPE_OPS,
    VOICE_REG_SIZE,
    VOICES,
)

SHIFTABLE_OPS = frozenset(set(SLOPE_OPS) | set(PRESET_OPS))
CTRL_REGS = tuple(4 + v * VOICE_REG_SIZE for v in range(VOICES))


def _voice_of_shiftable(reg, op):
    if op == FC_PRESET_OP:
        return None
    r = int(reg)
    if 0 <= r < VOICE_REG_SIZE * VOICES:
        return r // VOICE_REG_SIZE
    return None


class GateSlopeShiftPass(MacroPass):
    def apply(self, df, args=None):
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        if args is not None and not getattr(args, "gate_slope_shift_pass", True):
            return df
        df = df.reset_index(drop=True).copy()
        regs = df["reg"].to_numpy()
        if not ((regs == FRAME_REG) | (regs == DELAY_REG)).any():
            return df
        vals = df["val"].to_numpy()
        ops = df["op"].to_numpy()
        n = len(df)
        frame_idx = np.flatnonzero((regs == FRAME_REG) | (regs == DELAY_REG)).tolist()
        if len(frame_idx) < 3:
            return df
        frame_idx.append(n)

        last_ctrl = [None] * VOICES
        moves = []
        for fi in range(len(frame_idx) - 2):
            lo = frame_idx[fi]
            mid = frame_idx[fi + 1]
            hi = frame_idx[fi + 2]
            gate_voices = set()
            for r in range(lo + 1, mid):
                reg = int(regs[r])
                op = int(ops[r])
                if reg in CTRL_REGS and op == SET_OP:
                    v = reg // VOICE_REG_SIZE
                    val = int(vals[r])
                    if last_ctrl[v] is not None and (last_ctrl[v] & 1) != (val & 1):
                        gate_voices.add(v)
                    last_ctrl[v] = val
            if not gate_voices:
                continue
            body_len = hi - (mid + 1)
            if body_len not in (1, 3):
                continue
            body_ops = {int(ops[r]) for r in range(mid + 1, hi)}
            if len(body_ops) != 1:
                continue
            op = next(iter(body_ops))
            if op not in SHIFTABLE_OPS:
                continue
            if op not in BASE_TO_SHIFTED_OP:
                continue
            if op in PRESET_OPS and body_len != 1:
                continue
            if op in SLOPE_OPS and body_len != 3:
                continue
            shift_voice = _voice_of_shiftable(int(regs[mid + 1]), op)
            if shift_voice is None or shift_voice not in gate_voices:
                continue
            if int(regs[mid - 1]) == DELAY_REG:
                continue
            for r in range(mid + 1, hi):
                moves.append((r, mid, op))

        if not moves:
            return df

        ranks = np.arange(n, dtype=np.float64)
        new_ops = ops.copy()
        for src, dst, base_op in moves:
            ranks[src] = dst - 0.5
            new_ops[src] = BASE_TO_SHIFTED_OP[base_op]
        df["op"] = new_ops
        order = np.argsort(ranks, kind="stable")
        return df.iloc[order].reset_index(drop=True)
