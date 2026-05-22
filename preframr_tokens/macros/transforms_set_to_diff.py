"""SetToDiffTransform: convert bare SETs to DIFFs on motion regs only, anchored at first-write and gate-transition frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from preframr_tokens.macros.transform import Transform, register
from preframr_tokens.utils import to_int64_arrays
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    DIFF_OP,
    FC_LO_REG,
    FC_PRESET_OP,
    FC_PRESET_TABLE,
    FLIP_OP,
    FLIP2_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    PWM_PRESET_OP,
    PWM_PRESET_TABLE,
    PWM_SUSTAIN_OP,
    SET_OP,
    VOICE_REG,
    VOICE_REG_SIZE,
    VOICE_TRAJ_REG,
    WAVETABLE_SUSTAIN_OP,
)

_FILTER_VOICE_KEY = -1
_VOICE_MOTION_OFFSETS = frozenset({0, 1, 2, 3})
_FILTER_MOTION_REGS = frozenset({int(FC_LO_REG), int(FC_LO_REG) + 1})
_CTRL_OFFSET = 4
_GATE_BIT = 0b1
_LEGATO_NIBBLE_OPS = frozenset(
    {
        int(LEGATO_OP_CLUSTER_2),
        int(LEGATO_OP_CLUSTER_3),
        int(LEGATO_OP_CLUSTER_4),
    }
)


@register("set_to_diff")
class SetToDiffTransform(Transform):
    """Two passes on the post-_add_voice_reg stream: (1) motion-reg normalization (anchor frames SET, non-anchor frames DIFF, with anchor-frame DIFF/FLIP/FLIP2 materialized to SET; non-motion regs pass through unchanged); (2) sustain-frame collapse (single-voice non-anchor frames whose only voice writes are PWM_PRESET on reg=2 become a PWM_SUSTAIN_OP atom replacing the VOICE_REG marker + PWM_PRESET pair; if the frame also has a single global FC_PRESET, both collapse into one WAVETABLE_SUSTAIN_OP). Audio render path's DiffDecoder + new PwmSustain/WavetableSustain decoders re-materialize absolutes. Disabled by default; opt in via pipeline_spec or --set-to-diff-pass. See design/set_to_diff_design.md."""

    TIER = "audio_bit_exact"
    OP_CODES = frozenset({int(PWM_SUSTAIN_OP), int(WAVETABLE_SUSTAIN_OP)})
    OPERATES_ON_VOICE_REGS = False
    LOSS_TIER = "content"
    REQUIRES_ARGS = frozenset({"set_to_diff_pass"})
    MUST_FOLLOW = frozenset({"voice_block_order"})
    IDEMPOTENT = False
    PROVIDES_OPS = frozenset({int(PWM_SUSTAIN_OP), int(WAVETABLE_SUSTAIN_OP)})
    EMITS_NON_SET_REGS = frozenset({0, 1, 2, 3, int(FC_LO_REG), int(FC_LO_REG) + 1})
    EXPECTS_SET_ON_REGS = frozenset({0, 1, 2, 3, int(FC_LO_REG), int(FC_LO_REG) + 1})
    HANDLES_NON_SET_ON_REGS = frozenset({0, 2, int(FC_LO_REG)})
    DEFAULT_PARAMS = {"convert_regs": [0, 1, 2, 3, int(FC_LO_REG), int(FC_LO_REG) + 1]}
    PARAM_VALIDATORS = {
        "convert_regs": lambda v: (
            isinstance(v, (list, tuple))
            and all(isinstance(r, int) and r >= 0 for r in v)
        ),
    }

    def forward(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if args is None or not getattr(args, "set_to_diff_pass", False):
            return df
        if df.empty or "reg" not in df.columns:
            return df
        convert_regs = frozenset(int(r) for r in self.params["convert_regs"])
        df = _convert_sets_to_diffs(df, convert_regs)
        df = _collapse_sustain_frames(df)
        return df

    def inverse(self, df: pd.DataFrame, args=None) -> pd.DataFrame:
        if df.empty or "reg" not in df.columns:
            return df
        df = _expand_sustain_frames(df)
        df = _materialize_diffs(df)
        return df


def _voice_per_row(reg, val, n, op=None):
    is_frame = reg == int(FRAME_REG)
    is_voice = reg == int(VOICE_REG)
    is_voice_traj = reg == int(VOICE_TRAJ_REG)
    from preframr_tokens.stfconstants import PWM_SUSTAIN_OP, WAVETABLE_SUSTAIN_OP

    sustain_marker_ops = {int(PWM_SUSTAIN_OP), int(WAVETABLE_SUSTAIN_OP)}
    voice_per_row = np.full(n, -1, dtype=np.int64)
    cur_fn = -1
    cur_svt = 0
    cur_voice = -1
    for i in range(n):
        if is_frame[i]:
            cur_fn = 0
            cur_svt = int(val[i])
            cur_voice = -1
        else:
            is_marker = (
                is_voice[i]
                or is_voice_traj[i]
                or (op is not None and int(op[i]) in sustain_marker_ops)
            )
            if is_marker:
                cur_fn += 1
                slot = max(cur_fn - 1, 0)
                v_marker = ((cur_svt >> (slot * 2)) & 0b11) - 1
                cur_voice = int(v_marker) if 0 <= v_marker <= 2 else -1
        voice_per_row[i] = cur_voice
    return voice_per_row


def _decoded_ctrl_val(op_i, val_raw):
    v = int(val_raw)
    if op_i == int(SET_OP):
        return v & 0xFF
    if op_i == int(HARD_RESTART_OP):
        return v & 0xFF
    if op_i in _LEGATO_NIBBLE_OPS:
        return (v & 0x0F) << 4
    if op_i == int(LEGATO_OP_CLUSTER_7):
        return v & 0xFF
    if op_i == int(CTRL_BIGRAM_OP):
        idx = v
        if 0 <= idx < len(CTRL_BIGRAM_TABLE):
            _prev, cur = CTRL_BIGRAM_TABLE[idx]
            return int(cur)
    return None


def _gate_anchor_set(reg, val, op, voice_per_row, f_idx, n):
    max_f = int(f_idx.max()) if n else 0
    gate_writes: dict[tuple[int, int], int] = {}
    for i in range(n):
        v = int(voice_per_row[i])
        if v < 0:
            continue
        r = int(reg[i])
        op_i = int(op[i])
        decoded = None
        if op_i == int(SET_OP) and r == _CTRL_OFFSET:
            decoded = _decoded_ctrl_val(op_i, val[i])
        elif (
            op_i
            in (
                int(HARD_RESTART_OP),
                int(LEGATO_OP_CLUSTER_7),
            )
            or op_i in _LEGATO_NIBBLE_OPS
        ):
            decoded = _decoded_ctrl_val(op_i, val[i])
        elif op_i == int(CTRL_BIGRAM_OP) and r == _CTRL_OFFSET:
            decoded = _decoded_ctrl_val(op_i, val[i])
        if decoded is None:
            continue
        gate_writes[(int(f_idx[i]), v)] = int(decoded) & _GATE_BIT

    transitions: set[tuple[int, int]] = set()
    prev_gate = {0: 0, 1: 0, 2: 0}
    cur_gate = dict(prev_gate)
    for f in range(1, max_f + 1):
        for v in (0, 1, 2):
            new_gate = gate_writes.get((f, v))
            if new_gate is not None:
                cur_gate[v] = int(new_gate)
            if cur_gate[v] != prev_gate[v]:
                transitions.add((f, v))
            prev_gate[v] = cur_gate[v]

    anchor_set: set[tuple[int, int]] = set()
    for f, v in transitions:
        anchor_set.add((f, v))
        anchor_set.add((f + 1, v))
    return anchor_set


def _convert_sets_to_diffs(
    df: pd.DataFrame, convert_regs: frozenset[int] | None = None
) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    n = len(out)
    reg, val, op = to_int64_arrays(out, "reg", "val", "op", fillna={"val": 0})
    voice_per_row = _voice_per_row(reg, val, n, op=op)
    f_idx = np.cumsum((reg == int(FRAME_REG)).astype(np.int64))
    anchor_set = _gate_anchor_set(reg, val, op, voice_per_row, f_idx, n)
    if convert_regs is None:
        convert_voice_offsets = _VOICE_MOTION_OFFSETS
        convert_filter_regs = _FILTER_MOTION_REGS
    else:
        convert_voice_offsets = frozenset(
            r for r in convert_regs if r in _VOICE_MOTION_OFFSETS
        )
        convert_filter_regs = frozenset(
            r for r in convert_regs if r in _FILTER_MOTION_REGS
        )

    running: dict[tuple[int, int], int] = {}
    new_ops = op.copy()
    new_vals = val.copy()
    for i in range(n):
        r = int(reg[i])
        if r < 0:
            continue
        v = int(voice_per_row[i])
        op_i = int(op[i])
        cur_val = int(val[i])
        is_voice_motion = v >= 0 and r in convert_voice_offsets
        is_filter_motion = r in convert_filter_regs
        is_motion = is_voice_motion or is_filter_motion
        is_anchor = is_voice_motion and (int(f_idx[i]), v) in anchor_set
        key_v = v if v >= 0 else _FILTER_VOICE_KEY
        key = (key_v, r)
        if op_i == int(SET_OP):
            prev = running.get(key)
            if not is_motion or prev is None:
                running[key] = cur_val
                continue
            if is_anchor:
                running[key] = cur_val
                continue
            delta = cur_val - prev
            new_ops[i] = int(DIFF_OP)
            new_vals[i] = int(delta)
            running[key] = cur_val
        elif op_i == int(DIFF_OP):
            prev = running.get(key, 0)
            new_running = prev + int(val[i])
            running[key] = new_running
            if is_anchor:
                new_ops[i] = int(SET_OP)
                new_vals[i] = int(new_running)
        elif op_i == int(FLIP_OP):
            prev = running.get(key, 0)
            new_running = prev + int(val[i])
            running[key] = new_running
            if is_anchor:
                new_ops[i] = int(SET_OP)
                new_vals[i] = int(new_running)
        elif op_i == int(FLIP2_OP):
            a = (int(val[i]) >> 8) & 0xFF
            b = int(val[i]) & 0xFF
            if a >= 128:
                a -= 256
            if b >= 128:
                b -= 256
            prev = running.get(key, 0)
            new_running = prev + a + b
            running[key] = new_running
            if is_anchor:
                new_ops[i] = int(SET_OP)
                new_vals[i] = int(new_running)

    out["op"] = pd.array(new_ops, dtype=out["op"].dtype)
    out["val"] = pd.array(new_vals, dtype=out["val"].dtype)
    return out


def _collapse_sustain_frames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    n = len(out)
    reg, val, op = to_int64_arrays(out, "reg", "val", "op", fillna={"val": 0})
    voice_per_row = _voice_per_row(reg, val, n, op=op)
    f_idx = np.cumsum((reg == int(FRAME_REG)).astype(np.int64))
    anchor_set = _gate_anchor_set(reg, val, op, voice_per_row, f_idx, n)
    frame_to_rows: dict[int, list[int]] = {}
    for i in range(n):
        f = int(f_idx[i])
        if f <= 0:
            continue
        frame_to_rows.setdefault(f, []).append(i)
    keep = [True] * n
    new_ops = op.copy()
    new_vals = val.copy()
    new_regs = reg.copy()
    pwm_preset_id_by_val = {int(v): i for i, v in enumerate(PWM_PRESET_TABLE)}
    fc_preset_id_by_val = {int(v): i for i, v in enumerate(FC_PRESET_TABLE)}
    for f, rows in frame_to_rows.items():
        voice_blocks: list[tuple[int, int, list[int]]] = []
        cur_block: tuple[int, int, list[int]] | None = None
        global_writes: list[int] = []
        for i in rows:
            r = int(reg[i])
            if r == int(FRAME_REG):
                continue
            if r == int(VOICE_REG):
                cur_voice = int(voice_per_row[i])
                cur_block = (i, cur_voice, [])
                voice_blocks.append(cur_block)
                continue
            if r >= 0:
                if cur_block is not None and r < VOICE_REG_SIZE:
                    cur_block[2].append(i)
                else:
                    global_writes.append(i)
        single_voice_frame = len(voice_blocks) == 1 and len(global_writes) == 0
        single_voice_with_fc = (
            len(voice_blocks) == 1
            and len(global_writes) == 1
            and int(op[global_writes[0]]) == int(FC_PRESET_OP)
            and int(reg[global_writes[0]]) == int(FC_LO_REG)
        )
        for marker_idx, voice, writes in voice_blocks:
            if (f, voice) in anchor_set:
                continue
            if len(writes) != 1:
                continue
            w = writes[0]
            if int(op[w]) != int(PWM_PRESET_OP) or int(reg[w]) != 2:
                continue
            pwm_id = int(val[w])
            if single_voice_with_fc and marker_idx == voice_blocks[0][0]:
                g = global_writes[0]
                fc_id = int(val[g])
                keep[marker_idx] = False
                keep[g] = False
                new_ops[w] = int(WAVETABLE_SUSTAIN_OP)
                new_vals[w] = int(((pwm_id & 0xFF) << 8) | (fc_id & 0xFF))
                break
            keep[marker_idx] = False
            new_ops[w] = int(PWM_SUSTAIN_OP)
            new_vals[w] = int(pwm_id)
    if all(keep) and (new_ops == op).all() and (new_vals == val).all():
        return out
    keep_arr = np.array(keep, dtype=bool)
    out["op"] = pd.array(new_ops, dtype=out["op"].dtype)
    out["val"] = pd.array(new_vals, dtype=out["val"].dtype)
    out["reg"] = pd.array(new_regs, dtype=out["reg"].dtype)
    return out[keep_arr].reset_index(drop=True)


def _expand_sustain_frames(df: pd.DataFrame) -> pd.DataFrame:
    if not (
        (df["op"] == int(PWM_SUSTAIN_OP)).any()
        or (df["op"] == int(WAVETABLE_SUSTAIN_OP)).any()
    ):
        return df
    out = df.reset_index(drop=True).copy()
    n = len(out)
    op_col, val_col, reg_col = to_int64_arrays(out, "op", "val", "reg")
    records = out.to_dict(orient="records")
    new_rows: list[dict] = []
    template = {col: 0 for col in out.columns}
    if "diff" in template:
        template["diff"] = 0
    if "subreg" in template:
        template["subreg"] = -1
    if "irq" in template:
        template["irq"] = 0
    for i, row in enumerate(records):
        op_i = int(op_col[i])
        if op_i == int(PWM_SUSTAIN_OP):
            voice_marker = dict(template)
            for col in out.columns:
                if col in row and col not in ("op", "reg", "val", "subreg", "diff"):
                    voice_marker[col] = row[col]
            voice_marker["reg"] = int(VOICE_REG)
            voice_marker["val"] = 0
            voice_marker["op"] = int(SET_OP)
            new_rows.append(voice_marker)
            restored = dict(row)
            restored["op"] = int(PWM_PRESET_OP)
            restored["val"] = int(val_col[i])
            new_rows.append(restored)
        elif op_i == int(WAVETABLE_SUSTAIN_OP):
            voice_marker = dict(template)
            for col in out.columns:
                if col in row and col not in ("op", "reg", "val", "subreg", "diff"):
                    voice_marker[col] = row[col]
            voice_marker["reg"] = int(VOICE_REG)
            voice_marker["val"] = 0
            voice_marker["op"] = int(SET_OP)
            new_rows.append(voice_marker)
            packed = int(val_col[i])
            pwm_id = (packed >> 8) & 0xFF
            fc_id = packed & 0xFF
            pwm_row = dict(row)
            pwm_row["op"] = int(PWM_PRESET_OP)
            pwm_row["val"] = int(pwm_id)
            new_rows.append(pwm_row)
            fc_row = dict(template)
            for col in out.columns:
                if col in row and col not in ("op", "reg", "val", "subreg", "diff"):
                    fc_row[col] = row[col]
            fc_row["op"] = int(FC_PRESET_OP)
            fc_row["reg"] = int(FC_LO_REG)
            fc_row["val"] = int(fc_id)
            new_rows.append(fc_row)
        else:
            new_rows.append(row)
    result = pd.DataFrame(new_rows, columns=out.columns)
    return result.astype(out.dtypes.to_dict()).reset_index(drop=True)


def _materialize_diffs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    n = len(out)
    reg, val, op = to_int64_arrays(out, "reg", "val", "op", fillna={"val": 0})
    voice_per_row = _voice_per_row(reg, val, n, op=op)

    running: dict[tuple[int, int], int] = {}
    new_ops = op.copy()
    new_vals = val.copy()
    for i in range(n):
        r = int(reg[i])
        if r < 0:
            continue
        v = int(voice_per_row[i])
        key_v = v if v >= 0 else _FILTER_VOICE_KEY
        key = (key_v, r)
        op_i = int(op[i])
        if op_i == int(SET_OP):
            running[key] = int(val[i])
        elif op_i == int(DIFF_OP) and key in running:
            running[key] = int(running[key]) + int(val[i])
            new_ops[i] = int(SET_OP)
            new_vals[i] = int(running[key])
        elif op_i == int(FLIP_OP):
            prev = running.get(key, 0)
            running[key] = prev + int(val[i])
        elif op_i == int(FLIP2_OP):
            a = (int(val[i]) >> 8) & 0xFF
            b = int(val[i]) & 0xFF
            if a >= 128:
                a -= 256
            if b >= 128:
                b -= 256
            prev = running.get(key, 0)
            running[key] = prev + a + b

    out["op"] = pd.array(new_ops, dtype=out["op"].dtype)
    out["val"] = pd.array(new_vals, dtype=out["val"].dtype)
    return out
