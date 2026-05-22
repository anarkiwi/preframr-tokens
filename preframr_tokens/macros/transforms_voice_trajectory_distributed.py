"""VoiceTrajectoryDistributedTransform: distribute trajectory bits across existing FRAME_REG and VOICE_REG markers (no new rows; bounded alphabet inflation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.macros.transforms_voice_trajectory import (
    _compute_trajectory_byte,
    compute_frame_stats,
)
from preframr_tokens.stfconstants import (
    FC_LO_REG,
    FC_PRESET_OP,
    FRAME_REG,
    SLOPE_FC_LO_OP,
    VOICE_REG,
)

_PER_VOICE_TRAJ_BITS = 4
_FRAME_TRAJ_BITS_SHIFT = 6
_FRAME_SVT_MASK = 0x3F


@register("voice_trajectory_distributed")
class VoiceTrajectoryDistributedTransform(Transform):
    """Pack 4 per-voice trajectory bits (gate_on / pitch_dir / arp) into VOICE_REG.val (currently zeroed) and 2 frame-level trajectory bits (any-gate-transition this frame / filter section active) into FRAME_REG.val bits 6-7 (svt occupies bits 0-5; remove_voice_reg already masks & 63). Adds no new rows; alphabet inflation bounded to ~+76 atoms vs ~+4200 for full-byte replacement. Audio-bit-exact: renderer ignores both fields. See `voice_trajectory_distributed` refuted entry in preframr-experiments for the A/B verdict."""

    TIER = "audio_bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    LOSS_TIER = "structural"
    REQUIRES_ARGS = frozenset({"voice_trajectory_distributed_pass"})
    MUST_FOLLOW = frozenset({"voice_block_order"})
    IDEMPOTENT = True
    DEFAULT_PARAMS = {"window": 8}
    PARAM_VALIDATORS = {"window": lambda v: isinstance(v, int) and v >= 1}

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if args is None or not getattr(
            args, "voice_trajectory_distributed_pass", False
        ):
            return df
        if df.empty or "reg" not in df.columns:
            return df
        if not (df["reg"] == VOICE_REG).any():
            return df
        return _distribute_voice_trajectory(df, int(self.params["window"]))

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if df.empty or "reg" not in df.columns:
            return df
        return _strip_distributed_trajectory(df)


def _distribute_voice_trajectory(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    n = len(out)
    reg = out["reg"].astype(np.int64).to_numpy()
    val = out["val"].fillna(0).astype(np.int64).to_numpy()
    op = out["op"].astype(np.int64).to_numpy()
    is_frame = reg == int(FRAME_REG)
    is_voice = reg == int(VOICE_REG)
    f_idx = np.cumsum(is_frame.astype(np.int64))

    frame_stats, voice_at_marker, _voice_per_row, _f_idx_inner = compute_frame_stats(
        out
    )
    if not frame_stats:
        return out

    filter_active_frames = _frames_with_filter_activity(reg, op, f_idx, n)
    gate_transition_frames = _frames_with_any_gate_transition(frame_stats)

    new_vals = val.copy()
    for i in range(n):
        if is_frame[i]:
            f = int(f_idx[i])
            svt = int(val[i]) & _FRAME_SVT_MASK
            frame_bits = 0
            if f in gate_transition_frames:
                frame_bits |= 0b01
            if f in filter_active_frames:
                frame_bits |= 0b10
            new_vals[i] = svt | ((frame_bits & 0b11) << _FRAME_TRAJ_BITS_SHIFT)
        elif is_voice[i]:
            v = int(voice_at_marker[i])
            if v < 0 or v > 2:
                continue
            f = int(f_idx[i])
            new_vals[i] = _compact_voice_byte(frame_stats, f, v, window)

    out["val"] = pd.array(new_vals, dtype=out["val"].dtype)
    return out


def _strip_distributed_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    if "val" not in out.columns or "reg" not in out.columns:
        return out
    reg = out["reg"].astype(np.int64).to_numpy()
    val = out["val"].fillna(0).astype(np.int64).to_numpy()
    new_vals = val.copy()
    is_frame = reg == int(FRAME_REG)
    is_voice = reg == int(VOICE_REG)
    new_vals[is_frame] = val[is_frame] & _FRAME_SVT_MASK
    new_vals[is_voice] = 0
    out["val"] = pd.array(new_vals, dtype=out["val"].dtype)
    return out


def _frames_with_filter_activity(reg, op, f_idx, n):
    fc_lo_int = int(FC_LO_REG)
    fc_hi_int = fc_lo_int + 1
    fc_preset_op_int = int(FC_PRESET_OP)
    slope_fc_op_int = int(SLOPE_FC_LO_OP)
    frames: set[int] = set()
    for i in range(n):
        r = int(reg[i])
        op_i = int(op[i])
        if r in (fc_lo_int, fc_hi_int) or op_i in (fc_preset_op_int, slope_fc_op_int):
            frames.add(int(f_idx[i]))
    return frames


def _frames_with_any_gate_transition(frame_stats):
    prev_gate = {0: 0, 1: 0, 2: 0}
    cur_gate = dict(prev_gate)
    transitions: set[int] = set()
    max_f = max((f for f, _v in frame_stats.keys()), default=0)
    for f in range(1, max_f + 1):
        for v in (0, 1, 2):
            s = frame_stats.get((f, v))
            if s is not None and s["ctrl_val"] is not None:
                cur_gate[v] = int(s["ctrl_val"]) & 1
            if cur_gate[v] != prev_gate[v]:
                transitions.add(f)
            prev_gate[v] = cur_gate[v]
    return transitions


def _compact_voice_byte(frame_stats, f, v, window):
    full = _compute_trajectory_byte(frame_stats, f, v, window)
    gate_on = full & 0b1
    pitch_dir = (full >> 1) & 0b11
    arp = (full >> 3) & 0b1
    return gate_on | (pitch_dir << 1) | (arp << 3)
