"""Per-op decoder dispatch."""

from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    DIFF_OP,
    FC_LO_REG,
    FC_PRESET_TABLE,
    FLIP2_OP,
    FLIP_OP,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    PRESET_OPS,
    PRESET_SHIFTED_OPS,
    PWM_PRESET_OP,
    PWM_PRESET_TABLE,
    PWM_SUSTAIN_OP,
    SET_OP,
    SHIFTED_TO_BASE_OP,
    SLOPE_OPS,
    SLOPE_SHIFTED_OPS,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICES,
    WAVETABLE_SUSTAIN_OP,
)


class MacroDecoder:
    """Base class for op decoders dispatched from ``expand_ops``."""

    op_code = -1

    def expand(self, row, state):
        """Update ``state`` and return a list of writes (or None for no write)."""
        raise NotImplementedError


class SetDecoder(MacroDecoder):
    op_code = SET_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        if row.subreg == 0:
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0xF0) | int(row.val)
            state.last_diff[row.reg] = row.diff
            state.pending_subreg_reg = row.reg
            state.pending_subreg_nibbles.add(0)
            return pre or None
        if row.subreg == 1:
            assert row.val < 16
            state.last_val[row.reg] = (state.last_val[row.reg] & 0x0F) | (
                int(row.val) << 4
            )
            state.last_diff[row.reg] = row.diff
            state.pending_subreg_reg = row.reg
            state.pending_subreg_nibbles.add(1)
            return pre or None
        state.last_val[row.reg] = row.val
        own = (row.reg, int(state.last_val[row.reg]), row.diff, row.description)
        return pre + [own]


class DiffDecoder(MacroDecoder):
    op_code = DIFF_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        assert row.subreg == -1
        state.last_val[row.reg] += row.val
        own = (row.reg, int(state.last_val[row.reg]), row.diff, row.description)
        return pre + [own]


class FlipDecoder(MacroDecoder):
    op_code = FLIP_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, row.subreg)
        assert row.subreg == -1
        if row.val == 0:
            state.last_val[row.reg] += state.last_flip[row.reg]
            state.last_flip[row.reg] = 0
            state.active_flip_regs.discard(row.reg)
            own = (row.reg, int(state.last_val[row.reg]), row.diff, row.description)
            return pre + [own]
        if state.strict:
            assert row.reg not in state.active_flip_regs, (
                row.reg,
                state.active_flip_regs,
            )
        state.last_flip[row.reg] = row.val
        state.active_flip_regs.add(row.reg)
        return pre or None


class Flip2Decoder(MacroDecoder):
    """Asymmetric ±a/±b alternation across N frames."""

    op_code = FLIP2_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        length = int(row.subreg)
        assert length >= 2, row
        a = (int(row.val) >> 8) & 0xFF
        b = int(row.val) & 0xFF
        if a >= 128:
            a -= 256
        if b >= 128:
            b -= 256
        state.last_diff[row.reg] = row.diff
        for k in range(length):
            state.pending_diffs[row.reg].append(a if k % 2 == 0 else b)
        return pre or None


class TransposeDecoder(MacroDecoder):
    """Single-frame: apply same delta to multiple voices' freq regs."""

    op_code = TRANSPOSE_OP

    def expand(self, row, state):
        delta = int(row.val)
        if delta >= 0x8000:
            delta -= 0x10000
        mask = int(row.subreg)
        pre = []
        for v in range(VOICES):
            if mask & (1 << v):
                pre.extend(state.maybe_flush_for(FREQ_REGS_BY_VOICE[v], -1))
        writes = []
        for v in range(VOICES):
            if mask & (1 << v):
                reg = FREQ_REGS_BY_VOICE[v]
                state.last_val[reg] += delta
                state.last_diff[reg] = row.diff
                writes.append(
                    (reg, int(state.last_val[reg]), row.diff, row.description)
                )
        return (pre + writes) if (pre or writes) else None


class HardRestartDecoder(MacroDecoder):
    """Expand the hard-restart 2-write CTRL pair."""

    op_code = HARD_RESTART_OP

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        ctrl_reg = int(row.reg)
        packed = int(row.val) & 0xFFFF
        a = (packed >> 8) & 0xFF
        b = packed & 0xFF
        writes = list(pre)
        state.last_val[ctrl_reg] = a
        writes.append((ctrl_reg, a, row.diff, row.description))
        state.last_val[ctrl_reg] = b
        writes.append((ctrl_reg, b, row.diff, row.description))
        return writes


class _LegatoClusterNibbleDecoder(MacroDecoder):
    """Per-cluster nibble-form decoder. ``op_code`` is bound at construction so the same class serves every nibble-form cluster (2/3/4). Semantics: val = waveform nibble; low nibble inherited from prev CTRL byte. Used by ``LegatoPerClusterPass``."""

    def __init__(self, op_code):
        self.op_code = op_code

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        ctrl_reg = int(row.reg)
        prev = int(state.last_val[ctrl_reg]) & 0xFF
        new_byte = ((int(row.val) & 0x0F) << 4) | (prev & 0x0F)
        writes = list(pre)
        state.last_val[ctrl_reg] = new_byte
        writes.append((ctrl_reg, new_byte, row.diff, row.description))
        return writes


class _LegatoClusterByteDecoder(MacroDecoder):
    """Per-cluster byte-form decoder. ``val`` is the full CTRL byte (handles Hubbard's gate-byte $FE / $FC sub-case where the low nibble changes). Used by cluster 7 in ``LegatoPerClusterPass``."""

    def __init__(self, op_code):
        self.op_code = op_code

    def expand(self, row, state):
        pre = state.maybe_flush_for(row.reg, -1)
        ctrl_reg = int(row.reg)
        new_byte = int(row.val) & 0xFF
        writes = list(pre)
        state.last_val[ctrl_reg] = new_byte
        writes.append((ctrl_reg, new_byte, row.diff, row.description))
        return writes


class SlopeDecoder(MacroDecoder):
    op_code = -1

    def expand(self, row, state):
        reg = int(row.reg)
        subreg = int(row.subreg)
        if subreg == SLOPE_SUBREG_TERMINAL_HI:
            state.pending_slope_terminal_hi = int(row.val) & 0xFF
            state.pending_slope_terminal_lo = 0
            return None
        if subreg == SLOPE_SUBREG_TERMINAL_LO:
            state.pending_slope_terminal_lo = int(row.val) & 0xFF
            return None
        assert subreg == SLOPE_SUBREG_RUNTIME, row
        pre = state.maybe_flush_for(reg, -1)
        terminal_u = (
            (state.pending_slope_terminal_hi << 8) | state.pending_slope_terminal_lo
        ) & 0xFFFF
        terminal = terminal_u if terminal_u < 0x8000 else terminal_u - 0x10000
        runtime = int(row.val)
        assert runtime > 0, row
        start_val = int(state.last_val[reg])
        delta = terminal - start_val
        state.last_diff[reg] = row.diff
        for k in range(1, runtime + 1):
            target = start_val + (delta * k) // runtime
            state.pending_set_writes[reg].append(int(target))
        state.pending_slope_terminal_hi = 0
        state.pending_slope_terminal_lo = 0
        return pre or None


class PresetDecoder(MacroDecoder):
    """Decode PRESET_OP rows: emit a SET-equivalent write with table-snapped val."""

    op_code = -1

    def expand(self, row, state):
        op = int(row.op)
        reg = int(row.reg)
        preset_id = int(row.val)
        if op == PWM_PRESET_OP:
            val = int(PWM_PRESET_TABLE[preset_id])
        else:
            val = int(FC_PRESET_TABLE[preset_id])
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = val
        own = (reg, val, row.diff, row.description)
        return pre + [own]


class ShiftedDecoder(MacroDecoder):
    """Defer a slope/preset op by one frame: stash a rewritten row with
    base op into pre-unroll queue (slope, queue-style writes) or
    post-marker queue (preset, inline SET). FrameWalker drains at the
    next FRAME or DELAY marker."""

    op_code = -1

    def expand(self, row, state):
        base_op = SHIFTED_TO_BASE_OP[int(row.op)]
        deferred = _FastRowProxy(row, op=base_op)
        if base_op in SLOPE_OPS:
            state.pending_deferred_pre_unroll.append((base_op, deferred))
        else:
            state.pending_deferred_post_marker.append((base_op, deferred))
        return None


class _FastRowProxy:
    __slots__ = ("reg", "val", "op", "subreg", "diff", "description", "Index")

    def __init__(self, src, op):
        self.reg = int(src.reg)
        self.val = int(src.val)
        self.op = int(op)
        self.subreg = int(src.subreg)
        self.diff = int(src.diff)
        self.description = int(src.description)
        self.Index = int(src.Index)


class SubregFlushDecoder(MacroDecoder):
    """Force-flush deferred subreg state. Inserted by SubregPass between two
    consecutive subreg rows that are on the same reg, touch different
    nibbles, AND came from different baseline SETs (so they would otherwise
    coalesce and lose the intermediate write)."""

    op_code = SUBREG_FLUSH_OP

    def expand(self, row, state):
        return state.flush_pending_subreg() or None


class PwmSustainDecoder(MacroDecoder):
    """Lonely-PWM sustain-frame macro: decoder emits the PWM_PRESET-equivalent SET on the voice's PW reg. Voice is recovered upstream by remove_voice_reg via FRAME_REG svt (frame is single-voice-only by construction; no VOICE_REG marker)."""

    op_code = PWM_SUSTAIN_OP

    def expand(self, row, state):
        reg = int(row.reg)
        preset_id = int(row.val)
        val = int(PWM_PRESET_TABLE[preset_id])
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = val
        own = (reg, val, row.diff, row.description)
        return pre + [own]


class WavetableSustainDecoder(MacroDecoder):
    """Lonely-PWM-plus-FC sustain-frame macro: decoder emits both a PWM_PRESET-equivalent SET on the voice's PW reg and an FC_PRESET-equivalent SET on the filter cutoff lo reg. Voice recovered upstream by remove_voice_reg; FC reg is global."""

    op_code = WAVETABLE_SUSTAIN_OP

    def expand(self, row, state):
        reg = int(row.reg)
        packed = int(row.val)
        pwm_preset_id = (packed >> 8) & 0xFF
        fc_preset_id = packed & 0xFF
        pwm_val = int(PWM_PRESET_TABLE[pwm_preset_id])
        fc_val = int(FC_PRESET_TABLE[fc_preset_id])
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = pwm_val
        state.last_val[int(FC_LO_REG)] = fc_val
        return pre + [
            (reg, pwm_val, row.diff, row.description),
            (int(FC_LO_REG), fc_val, row.diff, row.description),
        ]


class CtrlBigramDecoder(MacroDecoder):
    op_code = CTRL_BIGRAM_OP

    def expand(self, row, state):
        ctrl_reg = int(row.reg)
        idx = int(row.val)
        prev_byte, cur_byte = CTRL_BIGRAM_TABLE[idx]
        pre = state.maybe_flush_for(ctrl_reg, -1)
        state.last_val[ctrl_reg] = int(prev_byte)
        own = (ctrl_reg, int(prev_byte), row.diff, row.description)
        state.pending_set_writes[ctrl_reg].append(int(cur_byte))
        return list(pre) + [own]


DECODERS = {
    d.op_code: d
    for d in (
        SetDecoder(),
        DiffDecoder(),
        FlipDecoder(),
        TransposeDecoder(),
        Flip2Decoder(),
        SubregFlushDecoder(),
        HardRestartDecoder(),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_2),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_3),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_4),
        _LegatoClusterByteDecoder(LEGATO_OP_CLUSTER_7),
        CtrlBigramDecoder(),
        PwmSustainDecoder(),
        WavetableSustainDecoder(),
    )
}
_SLOPE_DECODER = SlopeDecoder()
for _op in SLOPE_OPS:
    DECODERS[_op] = _SLOPE_DECODER
_PRESET_DECODER = PresetDecoder()
for _op in PRESET_OPS:
    DECODERS[_op] = _PRESET_DECODER
_SHIFTED_DECODER = ShiftedDecoder()
for _op in SLOPE_SHIFTED_OPS + PRESET_SHIFTED_OPS:
    DECODERS[_op] = _SHIFTED_DECODER
