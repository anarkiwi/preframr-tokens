"""VoiceTrajectoryTransform: insert VOICE_TRAJ_REG annotations after each VOICE_REG."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preframr_tokens.macros.passes_base import _int64_cols
from preframr_tokens.macros.roles import slope_subreg_role
from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    DIFF_OP,
    FLIP_OP,
    FLIP2_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    SET_OP,
    SLOPE_FREQ_LO_OP,
    SLOPE_FREQ_LO_SHIFTED_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICE_REG,
    VOICE_REG_SIZE,
    VOICE_TRAJ_REG,
)

_FREQ_LO_OFFSET = 0
_CTRL_OFFSET = 4
_AD_OFFSET = 5
_SR_OFFSET = 6
_GATE_BIT_MASK = 0b1
_LEGATO_CLUSTER_NIBBLE_OPS = frozenset(
    {
        int(LEGATO_OP_CLUSTER_2),
        int(LEGATO_OP_CLUSTER_3),
        int(LEGATO_OP_CLUSTER_4),
    }
)
_GATE_RESTART_OPS = frozenset(
    {int(HARD_RESTART_OP), int(LEGATO_OP_CLUSTER_7)} | _LEGATO_CLUSTER_NIBBLE_OPS
)
_SLOPE_FREQ_OPS = frozenset({int(SLOPE_FREQ_LO_OP), int(SLOPE_FREQ_LO_SHIFTED_OP)})


def _is_voice_marker(reg, op):
    """A row delimits a voice block iff it is VOICE_REG, VOICE_TRAJ_REG (in replacement mode), PWM_SUSTAIN_OP, or WAVETABLE_SUSTAIN_OP. Each of those sits in the voice slot that svt indexes via (svt >> fn*2) & 3."""
    from preframr_tokens.stfconstants import PWM_SUSTAIN_OP, WAVETABLE_SUSTAIN_OP

    r = int(reg)
    if r == int(VOICE_REG) or r == int(VOICE_TRAJ_REG):
        return True
    o = int(op)
    return o == int(PWM_SUSTAIN_OP) or o == int(WAVETABLE_SUSTAIN_OP)


def _sign_extend_8(v):
    v = int(v) & 0xFF
    return v - 256 if v >= 128 else v


def _sign_extend_16(v):
    v = int(v) & 0xFFFF
    return v - 65536 if v >= 32768 else v


def _decoded_gate_restart_ctrl(op_i, val_raw):
    v = int(val_raw)
    if op_i == int(HARD_RESTART_OP):
        return v & 0xFF
    if op_i in _LEGATO_CLUSTER_NIBBLE_OPS:
        return (v & 0x0F) << 4
    if op_i == int(LEGATO_OP_CLUSTER_7):
        return v & 0xFF
    return 0


@register("voice_trajectory")
class VoiceTrajectoryTransform(Transform):
    """Insert VOICE_TRAJ_REG rows alongside each VOICE_REG carrying a backward-window-derived trajectory byte. Disabled by default; opt in via pipeline_spec. See design/voice_trajectory_design.md."""

    TIER = "audio_bit_exact"
    OP_CODES = frozenset()
    OPERATES_ON_VOICE_REGS = False
    LOSS_TIER = "mid"
    REQUIRES_ARGS = frozenset({"voice_trajectory_pass"})
    MUST_FOLLOW = frozenset({"voice_block_order"})
    IDEMPOTENT = True
    DEFAULT_PARAMS = {"window": 8, "replace_voice_reg": False}
    PARAM_VALIDATORS = {
        "window": lambda v: isinstance(v, int) and v >= 1,
        "replace_voice_reg": lambda v: isinstance(v, bool),
    }

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if args is None or not getattr(args, "voice_trajectory_pass", False):
            return df
        if df.empty or "reg" not in df.columns:
            return df
        if not (df["reg"] == VOICE_REG).any():
            return df
        return _annotate_voice_trajectory(
            df,
            int(self.params["window"]),
            replace=bool(self.params["replace_voice_reg"]),
        )

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if df.empty or "reg" not in df.columns:
            return df
        if not (df["reg"] == VOICE_TRAJ_REG).any():
            return df
        return df[df["reg"] != VOICE_TRAJ_REG].reset_index(drop=True)


def compute_frame_stats(df: pd.DataFrame):
    """Walk df once and build (frame_stats, voice_at_marker, voice_per_row, f_idx). frame_stats is a dict keyed by (frame, voice) of per-voice activity summaries; voice_at_marker[i] is voice id at VOICE_REG rows (-1 elsewhere); voice_per_row[i] is the ffilled voice for any row. Shared between insertion-mode and distributed-mode trajectory transforms."""
    out = df.reset_index(drop=True)
    n = len(out)
    reg, val, op = _int64_cols(out, "reg", "val", "op")
    is_frame = reg == int(FRAME_REG)
    is_voice = reg == int(VOICE_REG)
    f_idx = np.cumsum(is_frame.astype(np.int64))
    voice_at_marker = np.full(n, -1, dtype=np.int64)
    voice_per_row = np.full(n, -1, dtype=np.int64)
    cur_fn = -1
    cur_svt = 0
    cur_voice = -1
    for i in range(n):
        if is_frame[i]:
            cur_fn = 0
            cur_svt = int(val[i])
            cur_voice = -1
        elif is_voice[i]:
            cur_fn += 1
            slot = max(cur_fn - 1, 0)
            v_marker = ((cur_svt >> (slot * 2)) & 0b11) - 1
            if 0 <= v_marker <= 2:
                cur_voice = int(v_marker)
            else:
                v_from_val = int(val[i]) - 1 if int(val[i]) > 0 else int(val[i])
                cur_voice = v_from_val if 0 <= v_from_val <= 2 else -1
            voice_at_marker[i] = cur_voice
        voice_per_row[i] = cur_voice
    subreg_col = (
        out["subreg"].fillna(-1).astype(np.int64).to_numpy()
        if "subreg" in out.columns
        else np.full(n, -1, dtype=np.int64)
    )
    stat_keys = (
        "gate_on",
        "freq_lo",
        "ad",
        "sr",
        "ctrl_val",
        "had_write",
        "slope_term_hi",
        "slope_term_lo",
        "freq_delta_sum",
        "arp_marker",
    )
    frame_stats: dict[tuple[int, int], dict] = {}

    def _ensure_stats(key):
        s = frame_stats.get(key)
        if s is None:
            s = {
                k: (
                    False
                    if k in ("gate_on", "had_write", "arp_marker")
                    else (0 if k == "freq_delta_sum" else None)
                )
                for k in stat_keys
            }
            frame_stats[key] = s
        s["had_write"] = True
        return s

    set_op_int = int(SET_OP)
    diff_op_int = int(DIFF_OP)
    flip_op_int = int(FLIP_OP)
    flip2_op_int = int(FLIP2_OP)
    transpose_op_int = int(TRANSPOSE_OP)
    ctrl_bigram_op_int = int(CTRL_BIGRAM_OP)
    subreg_flush_op_int = int(SUBREG_FLUSH_OP)
    for i in range(n):
        v = int(voice_per_row[i])
        if v < 0:
            continue
        r = int(reg[i])
        op_i = int(op[i])
        v_signed_16 = _sign_extend_16(int(val[i]))
        if op_i == transpose_op_int and r == _FREQ_LO_OFFSET:
            voice_mask = int(subreg_col[i]) if subreg_col[i] >= 0 else 0
            delta = v_signed_16
            f_here = int(f_idx[i])
            for tv in range(3):
                if (voice_mask >> tv) & 1:
                    s = _ensure_stats((f_here, tv))
                    s["freq_delta_sum"] = int(s["freq_delta_sum"]) + delta
            continue
        if r < 0 and op_i not in _GATE_RESTART_OPS:
            continue
        stats = _ensure_stats((int(f_idx[i]), v))
        if op_i == set_op_int and r >= 0:
            if r == _FREQ_LO_OFFSET:
                stats["freq_lo"] = int(val[i])
            elif r == _CTRL_OFFSET:
                stats["ctrl_val"] = int(val[i])
                if int(val[i]) & _GATE_BIT_MASK:
                    stats["gate_on"] = True
            elif r == _AD_OFFSET:
                stats["ad"] = int(val[i])
            elif r == _SR_OFFSET:
                stats["sr"] = int(val[i])
        elif op_i == diff_op_int and r == _FREQ_LO_OFFSET:
            stats["freq_delta_sum"] = int(stats["freq_delta_sum"]) + v_signed_16
        elif op_i == flip_op_int and r == _FREQ_LO_OFFSET:
            stats["freq_delta_sum"] = int(stats["freq_delta_sum"]) + v_signed_16
            if int(val[i]) != 0:
                stats["arp_marker"] = True
        elif op_i == flip2_op_int and r == _FREQ_LO_OFFSET:
            stats["arp_marker"] = True
            a = (int(val[i]) >> 8) & 0xFF
            b = int(val[i]) & 0xFF
            if a >= 128:
                a -= 256
            if b >= 128:
                b -= 256
            stats["freq_delta_sum"] = int(stats["freq_delta_sum"]) + a + b
        elif op_i in _SLOPE_FREQ_OPS and r == _FREQ_LO_OFFSET:
            role = slope_subreg_role(op_i, int(subreg_col[i]))
            if role == "terminal_hi":
                stats["slope_term_hi"] = int(val[i])
            elif role == "terminal_lo":
                stats["slope_term_lo"] = int(val[i])
        elif op_i == ctrl_bigram_op_int and r == _CTRL_OFFSET:
            idx = int(val[i])
            if 0 <= idx < len(CTRL_BIGRAM_TABLE):
                _prev_byte, cur_byte = CTRL_BIGRAM_TABLE[idx]
                stats["ctrl_val"] = int(cur_byte)
                if int(cur_byte) & _GATE_BIT_MASK:
                    stats["gate_on"] = True
        elif op_i == subreg_flush_op_int and r >= 0:
            from preframr_tokens.stfconstants import VOICE_REG_SIZE

            base = r % VOICE_REG_SIZE if r >= 0 else r
            if base == _AD_OFFSET:
                stats["ad"] = int(val[i])
            elif base == _SR_OFFSET:
                stats["sr"] = int(val[i])
        elif op_i in _GATE_RESTART_OPS:
            new_ctrl = _decoded_gate_restart_ctrl(op_i, val[i])
            stats["ctrl_val"] = int(new_ctrl) if new_ctrl else _GATE_BIT_MASK
            stats["gate_on"] = True
    return frame_stats, voice_at_marker, voice_per_row, f_idx


def _annotate_voice_trajectory(
    df: pd.DataFrame, window: int, replace: bool = False
) -> pd.DataFrame:
    if window <= 0:
        return df
    if (df["reg"] == VOICE_TRAJ_REG).any():
        df = df[df["reg"] != VOICE_TRAJ_REG].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    n = len(out)
    reg, val = _int64_cols(out, "reg", "val")
    is_voice = reg == int(VOICE_REG)
    frame_stats, voice_at_marker, _voice_per_row, f_idx = compute_frame_stats(out)
    if not np.any(voice_at_marker >= 0):
        return out

    trajectory_byte_at_row: dict[int, int] = {}
    last_emitted: dict[int, int] = {}
    for i in range(n):
        if not is_voice[i]:
            continue
        v = int(voice_at_marker[i])
        if v < 0 or v > 2:
            continue
        f = int(f_idx[i])
        byte = _compute_trajectory_byte(frame_stats, f, v, window)
        if replace:
            trajectory_byte_at_row[i] = byte
        elif last_emitted.get(v) != byte:
            trajectory_byte_at_row[i] = byte
            last_emitted[v] = byte

    if not trajectory_byte_at_row:
        return out

    base_records = out.to_dict(orient="records")
    template = {col: 0 for col in out.columns}
    if "diff" in template:
        template["diff"] = 0
    if "subreg" in template:
        template["subreg"] = -1
    if "irq" in template:
        template["irq"] = 0
    new_rows: list[dict] = []
    for i, row in enumerate(base_records):
        if i in trajectory_byte_at_row and replace:
            traj_row = dict(template)
            for col in out.columns:
                if col in ("reg", "val", "op"):
                    continue
                if col in row:
                    traj_row[col] = row[col]
            traj_row["reg"] = int(VOICE_TRAJ_REG)
            traj_row["op"] = int(SET_OP)
            traj_row["val"] = int(trajectory_byte_at_row[i])
            if "diff" in traj_row:
                traj_row["diff"] = 0
            if "subreg" in traj_row:
                traj_row["subreg"] = -1
            new_rows.append(traj_row)
            continue
        new_rows.append(row)
        if i in trajectory_byte_at_row:
            traj_row = dict(template)
            for col in out.columns:
                if col in ("reg", "val", "op"):
                    continue
                if col in row:
                    traj_row[col] = row[col]
            traj_row["reg"] = int(VOICE_TRAJ_REG)
            traj_row["op"] = int(SET_OP)
            traj_row["val"] = int(trajectory_byte_at_row[i])
            if "diff" in traj_row:
                traj_row["diff"] = 0
            if "subreg" in traj_row:
                traj_row["subreg"] = -1
            new_rows.append(traj_row)

    result = pd.DataFrame(new_rows, columns=out.columns)
    return result.astype(out.dtypes.to_dict()).reset_index(drop=True)


def _compute_trajectory_byte(frame_stats: dict, f: int, v: int, window: int) -> int:
    window_frames = range(max(1, f - window + 1), f + 1)
    freqs: list[int] = []
    gate_on_now = False
    last_ctrl: int | None = None
    has_recent_ad = False
    has_recent_sr = False
    last_write_frame = None
    ups = 0
    downs = 0
    any_arp_marker = False
    for g in window_frames:
        s = frame_stats.get((g, v))
        if s is None:
            continue
        if s["had_write"]:
            last_write_frame = g
        if s["freq_lo"] is not None:
            freqs.append(int(s["freq_lo"]))
        slope_hi = s.get("slope_term_hi")
        slope_lo = s.get("slope_term_lo")
        if slope_hi is not None and slope_lo is not None:
            freqs.append(((int(slope_hi) & 0xFF) << 8) | (int(slope_lo) & 0xFF))
        d = int(s.get("freq_delta_sum") or 0)
        if d > 0:
            ups += 1
        elif d < 0:
            downs += 1
        if s.get("arp_marker"):
            any_arp_marker = True
        if s["ctrl_val"] is not None:
            last_ctrl = s["ctrl_val"]
            if g == f:
                gate_on_now = bool(s["gate_on"])
        if g >= f - 1:
            if s["ad"] is not None:
                has_recent_ad = True
            if s["sr"] is not None:
                has_recent_sr = True

    for a, b in zip(freqs[:-1], freqs[1:]):
        d = b - a
        if d > 0:
            ups += 1
        elif d < 0:
            downs += 1

    if ups == 0 and downs == 0:
        pitch_dir = 1
    elif ups > 0 and downs > 0:
        pitch_dir = 3
    elif ups > 0:
        pitch_dir = 2
    else:
        pitch_dir = 0

    arp = 1 if any_arp_marker else 0
    if arp == 0 and len(freqs) >= 4:
        even = freqs[0::2]
        odd = freqs[1::2]
        if len(set(even)) == 1 and len(set(odd)) == 1 and even[0] != odd[0]:
            arp = 1

    if last_ctrl is None:
        adsr_phase = 0
    else:
        gate_on_in_ctrl = (last_ctrl & _GATE_BIT_MASK) != 0
        if not gate_on_in_ctrl:
            adsr_phase = 3
        elif has_recent_ad:
            adsr_phase = 0
        elif has_recent_sr:
            adsr_phase = 1
        else:
            adsr_phase = 2

    if last_write_frame is None:
        activity_level = 3
    else:
        gap = f - last_write_frame
        if gap <= 0:
            activity_level = 0
        else:
            activity_level = min(int(np.floor(np.log2(gap + 1))), 3)

    byte = (
        (1 if gate_on_now else 0)
        | ((pitch_dir & 0b11) << 1)
        | ((arp & 0b1) << 3)
        | ((adsr_phase & 0b11) << 4)
        | ((activity_level & 0b11) << 6)
    )
    return int(byte) & 0xFF
