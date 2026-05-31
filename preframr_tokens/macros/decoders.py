"""Per-op decoder dispatch."""

__all__ = [
    "DECODERS",
    "MacroDecoder",
    "SetDecoder",
    "DiffDecoder",
    "FlipDecoder",
    "TransposeDecoder",
    "HardRestartDecoder",
    "FreqTrajectoryDecoder",
    "TrackRefDecoder",
    "FreqNudgeDecoder",
    "ReleaseUpdateDecoder",
    "CtrlUpdateDecoder",
    "CtrlTripleDecoder",
    "PresetDecoder",
    "ShiftedDecoder",
    "SubregFlushDecoder",
    "PwmSustainDecoder",
    "WavetableSustainDecoder",
    "CtrlBigramDecoder",
    "SkeletonDecoder",
    "OrnamentDecoder",
    "StampDecoder",
]

from preframr_tokens.macros.skeleton_pass import (
    LUT as SKEL_LUT,
    cycle_frame_offsets,
    slide_frame_offsets,
    vib_frame_offsets,
)
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_VIB,
)
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
    FLIP_OP,
    FREQ_NUDGE_DELTA_ESCAPE,
    FREQ_NUDGE_MODE_ABSOLUTE,
    FREQ_NUDGE_MODE_DELTA,
    FREQ_NUDGE_OP,
    FREQ_ONSET_OP,
    FREQ_NUDGE_SUBREG_DELTA,
    FREQ_NUDGE_SUBREG_HI,
    FREQ_NUDGE_SUBREG_MODE,
    FREQ_TRAJ_OP,
    FT_DELTA_ESCAPE,
    FT_PERIODIC_BIT,
    FT_SUBREG_COUNT_HI,
    FT_SUBREG_COUNT_LO,
    FT_SUBREG_DELTA,
    FT_SUBREG_FLAGS,
    FT_SUBREG_PERIOD,
    FT_SUBREG_RUNTIME,
    FT_SUBREG_TERMINAL_HI,
    FT_SUBREG_TERMINAL_LO,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
    FT_SUBTYPE_MASK,
    FT_SUBTYPE_MONOTONE_RAMP,
    FT_V0_INTERVAL_BIT,
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
    RELEASE_UPDATE_OP,
    SET_OP,
    SHIFTED_TO_BASE_OP,
    SKEL_OP,
    SKEL_SUBREG_ABS,
    STAMP_DEF_OP,
    STAMP_END_OP,
    STAMP_REF_OP,
    STAMP_STEP_OP,
    STAMP_SUBREG_FRAME,
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
    """Decode a FREQ_NUDGE atom: mode then a signed-byte delta (delta mode,
    escape FREQ_NUDGE_DELTA_ESCAPE -> 16-bit hi/lo) or hi/lo absolute (absolute
    mode); one write on the final subreg."""

    op_code = FREQ_NUDGE_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        val = int(row.val) & 0xFF
        reg = int(row.reg)
        if subreg == FREQ_NUDGE_SUBREG_MODE:
            state.pending_nudge_fields = {"mode": val, "esc": False}
            return None
        f = state.pending_nudge_fields
        if not f:
            return None
        if subreg == FREQ_NUDGE_SUBREG_DELTA:
            if val == FREQ_NUDGE_DELTA_ESCAPE:
                f["esc"] = True
                return None
            state.pending_nudge_fields = {}
            return self._apply(
                row, state, reg, FREQ_NUDGE_MODE_DELTA, val if val < 128 else val - 256
            )
        if subreg == FREQ_NUDGE_SUBREG_HI:
            f["hi"] = val
            return None
        payload = (f.get("hi", 0) << 8) | val
        delta_mode = f.get("mode", 0) == FREQ_NUDGE_MODE_DELTA or f.get("esc", False)
        state.pending_nudge_fields = {}
        if delta_mode:
            return self._apply(
                row,
                state,
                reg,
                FREQ_NUDGE_MODE_DELTA,
                payload if payload < 0x8000 else payload - 0x10000,
            )
        return self._apply(row, state, reg, FREQ_NUDGE_MODE_ABSOLUTE, payload)

    @staticmethod
    def _apply(row, state, reg, mode, payload):
        pre = state.maybe_flush_for(reg, -1)
        if mode == FREQ_NUDGE_MODE_DELTA:
            state.last_val[reg] += payload
        else:
            state.last_val[reg] = payload
        own = (reg, int(state.last_val[reg]), row.diff, row.description)
        return pre + [own]


class FreqTrajectoryDecoder(MacroDecoder):
    """Decode a FREQ_TRAJ atom (op 45) for any slope reg: FLAGS selects SUBTYPE,
    MONOTONE_RAMP replays SLOPE's terminal+runtime ramp, OSCILLATE/RUN replay a
    lossless v0 + cumulative-delta run; all queue into pending_set_writes so one
    value drains per frame tick (the 0.14.1 multi-frame-drain rule)."""

    op_code = FREQ_TRAJ_OP

    def expand(self, row, state):
        subreg = int(row.subreg)
        reg = int(row.reg)
        if subreg == FT_SUBREG_FLAGS:
            flags = int(row.val) & 0xFF
            state.pending_ft = {
                "reg": reg,
                "subtype": flags & FT_SUBTYPE_MASK,
                "periodic": bool(flags & FT_PERIODIC_BIT),
                "v0_interval": bool(flags & FT_V0_INTERVAL_BIT),
                "fields": {},
                "steps": [],
                "esc": [],
                "in_esc": False,
                "count": 0,
            }
            return None
        ft = state.pending_ft
        if ft is None or ft["reg"] != reg:
            return None
        if ft["subtype"] == FT_SUBTYPE_MONOTONE_RAMP:
            return self._ramp(row, state, ft, subreg)
        return self._delta_run(row, state, ft, subreg)

    @staticmethod
    def _ramp(row, state, ft, subreg):
        if subreg == FT_SUBREG_TERMINAL_HI:
            ft["fields"]["thi"] = int(row.val) & 0xFF
            return None
        if subreg == FT_SUBREG_TERMINAL_LO:
            ft["fields"]["tlo"] = int(row.val) & 0xFF
            return None
        if subreg != FT_SUBREG_RUNTIME:
            return None
        reg = ft["reg"]
        state.pending_ft = None
        pre = state.maybe_flush_for(reg, -1)
        terminal_u = (
            (ft["fields"].get("thi", 0) << 8) | ft["fields"].get("tlo", 0)
        ) & (0xFFFF)
        signed = terminal_u if terminal_u < 0x8000 else terminal_u - 0x10000
        if ft.get("v0_interval"):
            terminal = int(state.last_freq_v0.get(reg, 0)) + signed
        else:
            terminal = signed
        state.last_freq_v0[reg] = terminal
        runtime = max(1, int(row.val))
        start_val = int(state.last_val[reg])
        delta = terminal - start_val
        state.last_diff[reg] = row.diff
        for k in range(1, runtime + 1):
            state.pending_set_writes[reg].append(
                int(start_val + (delta * k) // runtime)
            )
        return pre or None

    def _delta_run(self, row, state, ft, subreg):
        val = int(row.val) & 0xFF
        if subreg == FT_SUBREG_V0_HI:
            ft["fields"]["v0hi"] = val
            return None
        if subreg == FT_SUBREG_V0_LO:
            ft["fields"]["v0lo"] = val
            return None
        if subreg == FT_SUBREG_COUNT_HI:
            ft["fields"]["chi"] = val
            return None
        if subreg == FT_SUBREG_COUNT_LO:
            ft["count"] = (ft["fields"].get("chi", 0) << 8) | val
            return None
        if subreg == FT_SUBREG_PERIOD:
            ft["period"] = max(1, val)
            return None
        if subreg != FT_SUBREG_DELTA:
            return None
        if ft["in_esc"]:
            ft["esc"].append(val)
            if len(ft["esc"]) == 2:
                ft["steps"].append(("abs", (ft["esc"][0] << 8) | ft["esc"][1]))
                ft["esc"] = []
                ft["in_esc"] = False
        elif val == FT_DELTA_ESCAPE:
            ft["in_esc"] = True
        else:
            ft["steps"].append(("rel", val if val < 128 else val - 256))
        return self._maybe_finish(row, state, ft)

    @staticmethod
    def _maybe_finish(row, state, ft):
        if ft["in_esc"]:
            return None
        periodic = ft["periodic"]
        if periodic and "period" not in ft:
            return None
        target = ft["period"] if periodic else ft["count"]
        if len(ft["steps"]) < target:
            return None
        reg = ft["reg"]
        state.pending_ft = None
        pre = state.maybe_flush_for(reg, -1)
        raw = (ft["fields"].get("v0hi", 0) << 8) | ft["fields"].get("v0lo", 0)
        if ft.get("v0_interval"):
            signed = raw if raw < 0x8000 else raw - 0x10000
            v0 = int(state.last_freq_v0.get(reg, 0)) + signed
        else:
            v0 = raw
        state.last_freq_v0[reg] = v0
        steps = ft["steps"]
        period = ft["period"] if periodic else max(1, len(steps))
        state.last_diff[reg] = row.diff
        state.last_val[reg] = v0
        state.pending_set_writes[reg].append(int(v0))
        cur = v0
        for i in range(ft["count"]):
            kind, sv = steps[i % period]
            cur = cur + sv if kind == "rel" else sv
            state.pending_set_writes[reg].append(int(cur))
        return list(pre)


class FreqOnsetDecoder(MacroDecoder):
    """Decode a FREQ_ONSET atom: an isolated TRAJ_REG (freq/PW/filter) write, equivalent to a
    SET on that register but tagged as a melodic/timbral onset (so it lives in the onset
    channel, not op0 SET)."""

    op_code = FREQ_ONSET_OP

    def expand(self, row, state):
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(row.val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(row.val), row.diff, row.description)]


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
    frame CTRL writes. All three bytes queue into ``pending_set_writes``, one
    drained per frame tick from the atom's frame (like CTRL_BIGRAM extended by
    one)."""

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
        state.pending_set_writes[reg].append(b0)
        state.pending_set_writes[reg].append(f.get(CTRL_TRIPLE_SUBREG_1, 0))
        state.pending_set_writes[reg].append(f.get(CTRL_TRIPLE_SUBREG_2, 0))
        return list(pre)


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
    """Defer a preset op by one frame: stash a rewritten row with the base op
    into the post-marker queue (inline SET); FrameWalker drains it at the next
    FRAME or DELAY marker."""

    op_code = -1

    def expand(self, row, state):
        base_op = SHIFTED_TO_BASE_OP[int(row.op)]
        deferred = _FastRowProxy(row, op=base_op)
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
        state.last_diff[ctrl_reg] = row.diff
        state.last_val[ctrl_reg] = int(prev_byte)
        state.pending_set_writes[ctrl_reg].append(int(prev_byte))
        state.pending_set_writes[ctrl_reg].append(int(cur_byte))
        return list(pre)


class SkeletonDecoder(MacroDecoder):
    """Decode a SKEL atom (op54): one clean held freq note. The note is absolute
    (subreg=SKEL_SUBREG_ABS) for the first claimed note on a reg, else a signed semitone
    interval (two's-complement byte) from the prior note on that reg; the decoded freq is
    LUT[note] (content-tier cents snap)."""

    op_code = SKEL_OP

    def expand(self, row, state):
        reg = int(row.reg)
        if int(row.subreg) == SKEL_SUBREG_ABS:
            note = int(row.val)
        else:
            v = int(row.val) & 0xFF
            signed = v if v < 128 else v - 256
            note = int(state.last_skel_note.get(reg, 0)) + signed
        state.last_skel_note[reg] = note
        freq = int(SKEL_LUT[max(0, min(127, note))])
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = freq
        state.last_diff[reg] = row.diff
        return pre + [(reg, freq, row.diff, row.description)]


class OrnamentDecoder(MacroDecoder):
    """Decode an ORN atom (op55): replay one note's driver-native pitch ornament onto the
    skeleton freq at the semitone floor. TYPE selects the primitive; PLAIN replays nothing;
    OCTAVE/ARP/SLIDE/VIB carry a small signed-P1 param list (cycle period / slide target+rate
    / vib depth+rate) terminated by a P2 length; RESID is a P2 count then one P1 offset/frame.
    Each offset becomes LUT[skel_note+off], queued after a base re-assert, one per frame tick.
    """

    op_code = ORN_OP

    def expand(self, row, state):
        reg = int(row.reg)
        subreg = int(row.subreg)
        val = int(row.val) & 0xFF
        if subreg == ORN_SUBREG_TYPE:
            state.pending_orn = {"reg": reg, "type": val, "params": [], "length": None}
            if val == ORN_TYPE_PLAIN:
                state.pending_orn = None
            return None
        orn = state.pending_orn
        if orn is None or orn["reg"] != reg:
            return None
        if orn["type"] == ORN_TYPE_RESID:
            return self._resid(state, orn, subreg, val, int(row.val))
        if subreg == ORN_SUBREG_P1:
            orn["params"].append(val if val < 128 else val - 256)
            return None
        orn["length"] = int(row.val) & 0xFFFF
        return self._queue(state, orn, self._offsets(orn))

    def _resid(self, state, orn, subreg, val, raw):
        if orn["length"] is None and subreg == ORN_SUBREG_P2:
            orn["length"] = raw & 0xFFFF
            if orn["length"] == 0:
                state.pending_orn = None
            return None
        if subreg == ORN_SUBREG_P1:
            orn["params"].append(val if val < 128 else val - 256)
            if len(orn["params"]) >= (orn["length"] or 0):
                return self._queue(state, orn, list(orn["params"]))
        return None

    @staticmethod
    def _offsets(orn):
        t, params, length = orn["type"], orn["params"], orn["length"] or 0
        if t in (ORN_TYPE_OCTAVE, ORN_TYPE_ARP):
            return cycle_frame_offsets(params, length)
        if t == ORN_TYPE_SLIDE and len(params) >= 2:
            return slide_frame_offsets(params[0], params[1], length)
        if t == ORN_TYPE_VIB and len(params) >= 2:
            return vib_frame_offsets(params[0], params[1], length)
        return [0] * length

    @staticmethod
    def _queue(state, orn, offsets):
        reg = orn["reg"]
        state.pending_orn = None
        note = int(state.last_skel_note.get(reg, 0))
        queue = state.pending_set_writes[reg]
        queue.append(int(SKEL_LUT[max(0, min(127, note))]))
        for off in offsets:
            queue.append(int(SKEL_LUT[max(0, min(127, note + int(off)))]))
        return None


class StampDecoder(MacroDecoder):
    """Decode the inline-redefinable percussion stamp ops (design/percussion_stamp_encoding.md):
    STAMP_DEF/STEP/END buffer a voice-relative write-series into a live id->frames table (a later
    DEF id rebinds), and STAMP_REF (reg=target-voice freq reg, val=id) replays it on that voice via
    pending_set_writes -- one forward-filled value per song frame -- reproducing the exact series.
    """

    op_code = -1

    def expand(self, row, state):
        op = int(row.op)
        if op == STAMP_DEF_OP:
            state.pending_stamp_def = {"id": int(row.val), "frames": [{}]}
            return None
        if op == STAMP_STEP_OP:
            self._step(row, state)
            return None
        if op == STAMP_END_OP:
            stamp = state.pending_stamp_def
            if stamp is not None:
                state.stamp_table[int(stamp["id"])] = stamp["frames"]
                state.pending_stamp_def = None
            return None
        return self._ref(row, state)

    @staticmethod
    def _step(row, state):
        stamp = state.pending_stamp_def
        if stamp is None:
            return
        if int(row.subreg) == STAMP_SUBREG_FRAME:
            stamp["frames"].append({})
        else:
            stamp["frames"][-1][int(row.subreg)] = int(row.val)

    @staticmethod
    def _ref(row, state):
        frames = state.stamp_table.get(int(row.val))
        if not frames:
            return None
        base = int(row.reg)
        offsets = sorted({o for fr in frames for o in fr})
        pre = state.maybe_flush_for(base, -1)
        cur = {}
        for fr in frames:
            cur.update(fr)
            for off in offsets:
                if off in cur:
                    state.pending_set_writes[base + off].append(int(cur[off]))
        return pre or None


DECODERS = {
    d.op_code: d
    for d in (
        SetDecoder(),
        DiffDecoder(),
        FlipDecoder(),
        TransposeDecoder(),
        SubregFlushDecoder(),
        HardRestartDecoder(),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_2),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_3),
        _LegatoClusterNibbleDecoder(LEGATO_OP_CLUSTER_4),
        _LegatoClusterByteDecoder(LEGATO_OP_CLUSTER_7),
        CtrlBigramDecoder(),
        PwmSustainDecoder(),
        WavetableSustainDecoder(),
        FreqTrajectoryDecoder(),
        TrackRefDecoder(),
        FreqNudgeDecoder(),
        FreqOnsetDecoder(),
        ReleaseUpdateDecoder(),
        CtrlUpdateDecoder(),
        CtrlTripleDecoder(),
        SkeletonDecoder(),
        OrnamentDecoder(),
    )
}
_STAMP_DECODER = StampDecoder()
for _op in (STAMP_DEF_OP, STAMP_STEP_OP, STAMP_END_OP, STAMP_REF_OP):
    DECODERS[_op] = _STAMP_DECODER
_PRESET_DECODER = PresetDecoder()
for _op in PRESET_OPS:
    DECODERS[_op] = _PRESET_DECODER
_SHIFTED_DECODER = ShiftedDecoder()
for _op in PRESET_SHIFTED_OPS:
    DECODERS[_op] = _SHIFTED_DECODER
