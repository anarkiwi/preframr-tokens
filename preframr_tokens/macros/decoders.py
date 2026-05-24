"""Per-op decoder dispatch."""

__all__ = [
    "DECODERS",
    "MacroDecoder",
    "SetDecoder",
    "DiffDecoder",
    "FlipDecoder",
    "Flip2Decoder",
    "TransposeDecoder",
    "HardRestartDecoder",
    "SlopeDecoder",
    "OscillationEnvelopeDecoder",
    "TrackRefDecoder",
    "FreqNudgeDecoder",
    "FreqRunDecoder",
    "ReleaseUpdateDecoder",
    "CtrlUpdateDecoder",
    "CtrlTripleDecoder",
    "PresetDecoder",
    "ShiftedDecoder",
    "SubregFlushDecoder",
    "PwmSustainDecoder",
    "WavetableSustainDecoder",
    "CtrlBigramDecoder",
]

from preframr_tokens.macros.envelope import cycle_multipliers
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    CTRL_TRIPLE_OP,
    CTRL_UPDATE_OP,
    CTRL_TRIPLE_SUBREG_0,
    CTRL_TRIPLE_SUBREG_1,
    CTRL_TRIPLE_SUBREG_2,
    DIFF_OP,
    FC_LO_REG,
    FC_PRESET_TABLE,
    FLIP2_OP,
    FLIP_OP,
    FREQ_NUDGE_MODE_DELTA,
    FREQ_NUDGE_OP,
    FREQ_NUDGE_SUBREG_HI,
    FREQ_NUDGE_SUBREG_LO,
    FREQ_NUDGE_SUBREG_MODE,
    FREQ_RUN_OP,
    FREQ_RUN_SUBREG_COUNT,
    FREQ_RUN_SUBREG_HI,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    OSC_SUBREG_AMP_HI,
    OSC_SUBREG_AMP_LO,
    OSC_SUBREG_ANCHOR_HI,
    OSC_SUBREG_ANCHOR_LO,
    OSC_SUBREG_FAMILY,
    OSC_SUBREG_NCYCLES,
    OSC_SUBREG_PARAM,
    OSC_SUBREG_PERIOD,
    OSC_NCYCLES_MASK,
    OSC_START_DOWN_BIT,
    OSC_FAMILY_MASK,
    OSC_STEP_MODE_BIT,
    OSCILLATE_ENV_OP,
    PRESET_OPS,
    PRESET_SHIFTED_OPS,
    PWM_PRESET_OP,
    PWM_PRESET_TABLE,
    PWM_SUSTAIN_OP,
    RELEASE_UPDATE_OP,
    SET_OP,
    SHIFTED_TO_BASE_OP,
    SLOPE_OPS,
    SLOPE_SHIFTED_OPS,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
    SUBREG_FLUSH_OP,
    TRACK_INTERVAL_RATIOS,
    TRACK_REF_OP,
    TRACK_REF_SUBREG_DETUNE,
    TRACK_REF_SUBREG_DURATION,
    TRACK_REF_SUBREG_INTERVAL,
    TRACK_REF_SUBREG_LEAD,
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


class OscillationEnvelopeDecoder(MacroDecoder):
    """Decode an OSCILLATE_ENV atom (8 subreg rows) back into the per-frame
    ramp sequence the collapsed slope chain produced. Subregs accumulate into
    ``pending_osc_fields``; on the final (PARAM) subreg the whole oscillation
    is reconstructed and queued into ``pending_set_writes`` (drained one value
    per frame by ``tick_frame``), exactly as ``SlopeDecoder`` queues a ramp."""

    op_code = OSCILLATE_ENV_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        state.pending_osc_fields[subreg] = int(row.val) & 0xFF
        if subreg != OSC_SUBREG_PARAM:
            return None

        f = state.pending_osc_fields
        state.pending_osc_fields = {}
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)

        anchor = (f.get(OSC_SUBREG_ANCHOR_HI, 0) << 8) | f.get(OSC_SUBREG_ANCHOR_LO, 0)
        amp = (f.get(OSC_SUBREG_AMP_HI, 0) << 8) | f.get(OSC_SUBREG_AMP_LO, 0)
        slope_frames = max(1, int(f.get(OSC_SUBREG_PERIOD, 1)))
        ncycles_byte = int(f.get(OSC_SUBREG_NCYCLES, 0))
        n_slopes = ncycles_byte & OSC_NCYCLES_MASK
        start_down = bool(ncycles_byte & OSC_START_DOWN_BIT)
        fam_byte = int(f.get(OSC_SUBREG_FAMILY, 0))
        step_mode = bool(fam_byte & OSC_STEP_MODE_BIT)
        family = fam_byte & OSC_FAMILY_MASK
        param = int(f.get(OSC_SUBREG_PARAM, 0))

        mult = cycle_multipliers(family, param, n_slopes)
        sign0 = -1 if start_down else 1
        cur = int(state.last_val[reg])
        state.last_diff[reg] = row.diff
        for h in range(n_slopes):
            s = sign0 if h % 2 == 0 else -sign0
            amp_h = int(round(amp * mult[h]))
            terminal = anchor + s * amp_h
            if step_mode:
                reps = slope_frames if h < n_slopes - 1 else 1
                state.pending_set_writes[reg].extend([terminal] * reps)
            else:
                delta = terminal - cur
                for j in range(1, slope_frames + 1):
                    state.pending_set_writes[reg].append(
                        int(cur + (delta * j) // slope_frames)
                    )
            cur = terminal
        return pre or None


class TrackRefDecoder(MacroDecoder):
    """Decode a TRACK_REF atom (4 subreg rows): the tracker voice's FREQ is
    ``round(lead_freq * interval_ratio) + detune`` for ``duration`` frames.
    The first frame is written directly; later frames are reconstructed by a
    ``pending_track_links`` entry drained per frame by ``tick_frame``."""

    op_code = TRACK_REF_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        state.pending_track_fields[subreg] = int(row.val)
        if subreg != TRACK_REF_SUBREG_DURATION:
            return None
        f = state.pending_track_fields
        state.pending_track_fields = {}
        tracker_reg = int(row.reg)
        lead_reg = int(FREQ_REGS_BY_VOICE[f.get(TRACK_REF_SUBREG_LEAD, 0)])
        ratio = TRACK_INTERVAL_RATIOS[f.get(TRACK_REF_SUBREG_INTERVAL, 0)]
        detune = f.get(TRACK_REF_SUBREG_DETUNE, 0) & 0xFF
        if detune >= 128:
            detune -= 256
        duration = max(1, int(f.get(TRACK_REF_SUBREG_DURATION, 1)))
        pre = state.maybe_flush_for(tracker_reg, -1)
        state.last_diff[tracker_reg] = row.diff
        state.pending_track_links.append(
            {
                "src": lead_reg,
                "tgt": tracker_reg,
                "ratio": ratio,
                "detune": detune,
                "remaining": duration,
            }
        )
        return pre or None


class FreqNudgeDecoder(MacroDecoder):
    """Decode a FREQ_NUDGE atom (mode, hi, lo): one isolated FREQ event that
    unifies DIFF (mode=delta, add signed payload) and absolute-SET (mode=
    absolute, set payload). One write on the final (LO) subreg."""

    op_code = FREQ_NUDGE_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        state.pending_nudge_fields[subreg] = int(row.val) & 0xFF
        if subreg != FREQ_NUDGE_SUBREG_LO:
            return None
        f = state.pending_nudge_fields
        state.pending_nudge_fields = {}
        reg = int(row.reg)
        payload = (f.get(FREQ_NUDGE_SUBREG_HI, 0) << 8) | f.get(FREQ_NUDGE_SUBREG_LO, 0)
        pre = state.maybe_flush_for(reg, -1)
        if f.get(FREQ_NUDGE_SUBREG_MODE, 0) == FREQ_NUDGE_MODE_DELTA:
            delta = payload if payload < 0x8000 else payload - 0x10000
            state.last_val[reg] += delta
        else:
            state.last_val[reg] = payload
        own = (reg, int(state.last_val[reg]), row.diff, row.description)
        return pre + [own]


class FreqRunDecoder(MacroDecoder):
    """Decode a FREQ_RUN atom: a count subreg then ``count`` (hi, lo) value
    pairs replaying a run of consecutive-frame FREQ SETs. Value 0 is written at
    the atom's frame; the rest queue into ``pending_set_writes`` per frame."""

    op_code = FREQ_RUN_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        reg = int(row.reg)
        val = int(row.val) & 0xFF
        if subreg == FREQ_RUN_SUBREG_COUNT:
            state.pending_run = {"reg": reg, "count": val, "vals": [], "hi": None}
            return None
        run = state.pending_run
        if run is None or run["reg"] != reg:
            return None
        if subreg == FREQ_RUN_SUBREG_HI:
            run["hi"] = val
            return None
        run["vals"].append((run["hi"] << 8) | val)
        if len(run["vals"]) < run["count"]:
            return None
        vals = run["vals"]
        state.pending_run = None
        pre = state.maybe_flush_for(reg, -1)
        state.last_diff[reg] = row.diff
        state.last_val[reg] = vals[0]
        for v in vals[1:]:
            state.pending_set_writes[reg].append(int(v))
        return pre + [(reg, int(vals[0]), row.diff, row.description)]


class ReleaseUpdateDecoder(MacroDecoder):
    """Decode a RELEASE_UPDATE atom: a single isolated SR/AD envelope write,
    equivalent to a SET on that register but tagged as a recognised op."""

    op_code = RELEASE_UPDATE_OP

    def expand(self, row, state):
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(row.val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(row.val), row.diff, row.description)]


class CtrlUpdateDecoder(MacroDecoder):
    """Decode a CTRL_UPDATE atom: a single residual CTRL write the bigram/triple
    passes did not take, equivalent to a SET on that register but tagged as a
    recognised op so it is not a lonely write."""

    op_code = CTRL_UPDATE_OP

    def expand(self, row, state):
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(row.val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(row.val), row.diff, row.description)]


class CtrlTripleDecoder(MacroDecoder):
    """Decode a CTRL_TRIPLE atom (3 byte subregs): three consecutive adjacent-
    frame CTRL writes. Byte 0 is written at the atom frame; bytes 1 and 2 queue
    into ``pending_set_writes`` for the following two frames (like CTRL_BIGRAM
    extended by one)."""

    op_code = CTRL_TRIPLE_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        reg = int(row.reg)
        state.pending_ctrl_triple[subreg] = int(row.val) & 0xFF
        if subreg != CTRL_TRIPLE_SUBREG_2:
            return None
        f = state.pending_ctrl_triple
        state.pending_ctrl_triple = {}
        pre = state.maybe_flush_for(reg, -1)
        b0 = f.get(CTRL_TRIPLE_SUBREG_0, 0)
        state.last_val[reg] = b0
        state.last_diff[reg] = row.diff
        state.pending_set_writes[reg].append(f.get(CTRL_TRIPLE_SUBREG_1, 0))
        state.pending_set_writes[reg].append(f.get(CTRL_TRIPLE_SUBREG_2, 0))
        return pre + [(reg, b0, row.diff, row.description)]


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
        OscillationEnvelopeDecoder(),
        TrackRefDecoder(),
        FreqNudgeDecoder(),
        FreqRunDecoder(),
        ReleaseUpdateDecoder(),
        CtrlUpdateDecoder(),
        CtrlTripleDecoder(),
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
