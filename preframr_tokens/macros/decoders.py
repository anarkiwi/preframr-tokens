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
    "PatchDecoder",
    "SweepDecoder",
    "CtrlOscDecoder",
    "NoteOffDecoder",
    "CtrlWtDecoder",
    "WavetableDecoder",
]

from preframr_tokens.macros.skeleton_pass import (
    LUT as SKEL_LUT,
    cycle_frame_offsets,
    held_cycle_offsets,
    slide_frame_offsets,
    slide2_frame_offsets,
    vib_frame_offsets,
)
from preframr_tokens.stfconstants import (
    ORN_OP,
    ORN_SUBREG_HOLD,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_SUBREG_TYPE,
    ORN_TYPE_ARP,
    ORN_TYPE_HELD_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_SLIDE2,
    ORN_TYPE_VIB,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_BIGRAM_OP,
    CTRL_BIGRAM_TABLE,
    CTRL_OSC_OP,
    CTRL_OSC_SUBREG_LEN,
    CTRL_OSC_SUBREG_PERIOD,
    CTRL_OSC_SUBREG_STATE_BASE,
    CTRL_TRIPLE_OP,
    CTRL_UPDATE_OP,
    CTRL_WT_DEF_OP,
    CTRL_WT_SET_OP,
    CTRL_WT_STEP_OP,
    CTRL_WT_SUBREG_VAL,
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
    NOTE_OFF_OP,
    NOTE_ON_OP,
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
    PATCH_AD_OFFSET,
    PATCH_DEF_OP,
    PATCH_SET_OP,
    PATCH_SR_OFFSET,
    PATCH_STEP_OP,
    PATCH_SUBREG_AD,
    PATCH_SUBREG_SR,
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
    STAMP_REL_REF_OP,
    STAMP_REL_SUBREG_BASE_HI,
    STAMP_REL_SUBREG_BASE_LO,
    STAMP_REL_SUBREG_ID,
    STAMP_STEP_OP,
    STAMP_SUBREG_FRAME,
    SUBREG_FLUSH_OP,
    SWEEP_OP,
    SWEEP_SUBREG_DELTA_HI,
    SWEEP_SUBREG_DELTA_LO,
    SWEEP_SUBREG_LEN,
    SWEEP_SUBREG_PERIOD,
    SWEEP_SUBREG_START_HI,
    SWEEP_SUBREG_START_LO,
    TRACK_INTERVAL_RATIOS,
    TRACK_REF_OP,
    TRACK_REF_SUBREG_DETUNE,
    TRACK_REF_SUBREG_DURATION,
    TRACK_REF_SUBREG_INTERVAL,
    TRACK_REF_SUBREG_LEAD,
    TRANSPOSE_OP,
    VOICES,
    WAVETABLE_DEF_OP,
    WAVETABLE_END_OP,
    WAVETABLE_ONESHOT_OP,
    WAVETABLE_REF_OP,
    WAVETABLE_STEP_OP,
    WAVETABLE_SUSTAIN_OP,
    WT_ONESHOT_SUBREG_END,
    WT_ONESHOT_SUBREG_HOLD,
    WT_ONESHOT_SUBREG_LEN_HI,
    WT_ONESHOT_SUBREG_LEN_LO,
    WT_ONESHOT_SUBREG_OFFSET,
    WT_REF_SUBREG_ID,
    WT_REF_SUBREG_LEAD,
    WT_REF_SUBREG_LEADOFF,
    WT_REF_SUBREG_LEN_HI,
    WT_REF_SUBREG_LEN_LO,
    WT_STEP_SUBREG_HOLD,
    WT_STEP_SUBREG_LOOP,
    WT_STEP_SUBREG_OFFSET,
)
from preframr_tokens.macros.wavetable import unroll as wt_unroll
from preframr_tokens.macros.codebook import codebook_decoders


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


class _SetEquivalentDecoder(MacroDecoder):
    """Base for ops that decode exactly like a plain SET on ``row.reg`` (flush pending, update
    last_val/last_diff, emit the single write) but carry a distinct op so the atom channels as a
    recognised, non-lonely write. Subclasses set ``op_code`` only."""

    op_code = -1

    def expand(self, row, state):
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(row.val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(row.val), row.diff, row.description)]


class FreqOnsetDecoder(_SetEquivalentDecoder):
    """Decode a FREQ_ONSET atom: an isolated TRAJ_REG (freq/PW/filter) write, equivalent to a
    SET on that register but tagged as a melodic/timbral onset (so it lives in the onset
    channel, not op0 SET)."""

    op_code = FREQ_ONSET_OP


class ReleaseUpdateDecoder(_SetEquivalentDecoder):
    """Decode a RELEASE_UPDATE atom: a single isolated SR/AD envelope write,
    equivalent to a SET on that register but tagged as a recognised op."""

    op_code = RELEASE_UPDATE_OP


class CtrlUpdateDecoder(_SetEquivalentDecoder):
    """Decode a CTRL_UPDATE atom: a single residual CTRL write the bigram/triple
    passes did not take, equivalent to a SET on that register but tagged as a
    recognised op so it is not a lonely write."""

    op_code = CTRL_UPDATE_OP


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
            state.pending_orn = {
                "reg": reg,
                "type": val,
                "params": [],
                "holds": [],
                "length": None,
            }
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
        if subreg == ORN_SUBREG_HOLD:
            orn["holds"].append(int(row.val) & 0xFF)
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
        if t == ORN_TYPE_HELD_ARP and params and orn["holds"]:
            return held_cycle_offsets(params, orn["holds"])
        if t in (ORN_TYPE_OCTAVE, ORN_TYPE_ARP):
            return cycle_frame_offsets(params, length)
        if t == ORN_TYPE_SLIDE and len(params) >= 2:
            return slide_frame_offsets(params[0], params[1], length)
        if t == ORN_TYPE_SLIDE2 and len(params) >= 2:
            return slide2_frame_offsets(params[0], params[1], length)
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
            state.pending_stamp_def = {"id": int(row.val), "frames": [[]]}
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
        if op == STAMP_REL_REF_OP:
            return self._rel_ref(row, state)
        return self._ref(row, state)

    @staticmethod
    def _step(row, state):
        stamp = state.pending_stamp_def
        if stamp is None:
            return
        if int(row.subreg) == STAMP_SUBREG_FRAME:
            stamp["frames"].append([])
        else:
            stamp["frames"][-1].append((int(row.subreg), int(row.val)))

    @staticmethod
    def _offsets_in_order(frames):
        """Voice-relative offsets in first-write order across the buffered frames -- preserves the
        drum's intra-frame freq<->ctrl order (the per-frame drain in State.tick_frame emits regs in
        pending_set_writes insertion order, so the first-seen offset writes first)."""
        offsets, seen = [], set()
        for fr in frames:
            for off, _val in fr:
                if off not in seen:
                    seen.add(off)
                    offsets.append(off)
        return offsets

    @staticmethod
    def _ref(row, state):
        frames = state.stamp_table.get(int(row.val))
        if not frames:
            return None
        base = int(row.reg)
        offsets = StampDecoder._offsets_in_order(frames)
        pre = state.maybe_flush_for(base, -1)
        cur = {}
        for fr in frames:
            for off, val in fr:
                cur[off] = val
            for off in offsets:
                if off in cur:
                    state.pending_set_writes[base + off].append(int(cur[off]))
        return pre or None

    def _rel_ref(self, row, state):
        """A transpose-relative hit: buffer id + base-freq hi/lo, then replay the stamp adding the
        (sign-extended) freq delta at offset 0 to the per-hit base, leaving ctrl (offset 4) raw.
        """
        subreg = int(row.subreg)
        if subreg == STAMP_REL_SUBREG_ID:
            state.pending_stamp_rel = {
                "id": int(row.val),
                "reg": int(row.reg),
                "base": 0,
            }
            return None
        pend = state.pending_stamp_rel
        if pend is None:
            return None
        if subreg == STAMP_REL_SUBREG_BASE_HI:
            pend["base"] |= (int(row.val) & 0xFF) << 8
            return None
        if subreg != STAMP_REL_SUBREG_BASE_LO:
            return None
        pend["base"] |= int(row.val) & 0xFF
        state.pending_stamp_rel = None
        return self._replay_rel(pend, state)

    @staticmethod
    def _replay_rel(pend, state):
        frames = state.stamp_table.get(int(pend["id"]))
        if not frames:
            return None
        base = int(pend["base"])
        voice = int(pend["reg"])
        offsets = StampDecoder._offsets_in_order(frames)
        pre = state.maybe_flush_for(voice, -1)
        cur = {}
        for fr in frames:
            for off, val in fr:
                cur[off] = val
            for off in offsets:
                if off not in cur:
                    continue
                val = int(cur[off])
                if off == 0:
                    signed = val if val < 0x8000 else val - 0x10000
                    val = (base + signed) & 0xFFFF
                state.pending_set_writes[voice + off].append(val)
        return pre or None


class PatchDecoder(MacroDecoder):
    """Decode the inline melodic-instrument patch ops (design/patch_preamble_encoding.md):
    PATCH_DEF + two PATCH_STEP rows buffer a full (AD,SR) envelope into a live id->state table (a
    later DEF id rebinds), emitting both register writes on the def's voice; PATCH_SET (val=id)
    re-emits a defined patch's AD/SR on the ref's voice -- reproducing the exact envelope writes.
    """

    op_code = -1

    def expand(self, row, state):
        op = int(row.op)
        if op == PATCH_DEF_OP:
            state.pending_patch_def = {
                "id": int(row.val),
                "freq_reg": int(row.reg),
                "ad": None,
                "sr": None,
            }
            return None
        if op == PATCH_STEP_OP:
            return self._step(row, state)
        return self._ref(row, state)

    def _step(self, row, state):
        pend = state.pending_patch_def
        if pend is None:
            return None
        if int(row.subreg) == PATCH_SUBREG_AD:
            pend["ad"] = int(row.val)
        elif int(row.subreg) == PATCH_SUBREG_SR:
            pend["sr"] = int(row.val)
        if pend["ad"] is None or pend["sr"] is None:
            return None
        state.pending_patch_def = None
        state.patch_table[int(pend["id"])] = (int(pend["ad"]), int(pend["sr"]))
        return self._emit(int(pend["freq_reg"]), pend["ad"], pend["sr"], row, state)

    def _ref(self, row, state):
        patch = state.patch_table.get(int(row.val))
        if patch is None:
            return None
        return self._emit(int(row.reg), patch[0], patch[1], row, state)

    @staticmethod
    def _emit(freq_reg, ad, sr, row, state):
        writes = []
        for reg, val in (
            (freq_reg + PATCH_AD_OFFSET, int(ad)),
            (freq_reg + PATCH_SR_OFFSET, int(sr)),
        ):
            writes.extend(state.maybe_flush_for(reg, -1))
            state.last_val[reg] = val
            state.last_diff[reg] = row.diff
            writes.append((reg, val, row.diff, row.description))
        return writes


class SweepDecoder(MacroDecoder):
    """Decode a SWEEP atom (design SoundMonitor/skydive): START_HI/LO, signed-16 DELTA_HI/LO, LEN
    buffer a constant-raw-freq-delta ramp; on LEN the per-frame freqs (start + k*delta) queue into
    pending_set_writes for the reg -- one drained per song frame, reproducing the exact ramp.
    """

    op_code = SWEEP_OP

    def expand(self, row, state):
        sub = int(row.subreg)
        if sub == SWEEP_SUBREG_START_HI:
            state.pending_sweep = {"reg": int(row.reg), "fields": {}}
        pend = state.pending_sweep
        if pend is None:
            return None
        pend["fields"][sub] = int(row.val)
        if sub != SWEEP_SUBREG_LEN:
            return None
        state.pending_sweep = None
        f = pend["fields"]
        start = ((f.get(SWEEP_SUBREG_START_HI, 0) & 0xFF) << 8) | (
            f.get(SWEEP_SUBREG_START_LO, 0) & 0xFF
        )
        raw = ((f.get(SWEEP_SUBREG_DELTA_HI, 0) & 0xFF) << 8) | (
            f.get(SWEEP_SUBREG_DELTA_LO, 0) & 0xFF
        )
        delta = raw if raw < 0x8000 else raw - 0x10000
        period = int(f.get(SWEEP_SUBREG_PERIOD, 0))
        reg = int(pend["reg"])
        pre = state.maybe_flush_for(reg, -1)
        for k in range(int(f.get(SWEEP_SUBREG_LEN, 0))):
            step = (k % period) if period else k
            state.pending_set_writes[reg].append((start + step * delta) & 0xFFFF)
        return pre or None


class CtrlOscDecoder(MacroDecoder):
    """Decode a CTRL_OSC atom (CtrlOscPass): PERIOD opens the buffer, STATE_BASE+m the P cycle bytes,
    LEN (terminal) queues cycle[k % P] for LEN frames into pending_set_writes -- one drained per song
    frame, reproducing the exact per-frame ctrl oscillation. A SWEEP twin with an explicit byte cycle.
    """

    op_code = CTRL_OSC_OP

    def expand(self, row, state):
        sub = int(row.subreg)
        if sub == CTRL_OSC_SUBREG_PERIOD:
            state.pending_ctrl_osc = {"reg": int(row.reg), "fields": {}}
        pend = state.pending_ctrl_osc
        if pend is None:
            return None
        pend["fields"][sub] = int(row.val)
        if sub != CTRL_OSC_SUBREG_LEN:
            return None
        state.pending_ctrl_osc = None
        f = pend["fields"]
        reg = int(pend["reg"])
        period = int(f.get(CTRL_OSC_SUBREG_PERIOD, 0))
        length = int(f.get(CTRL_OSC_SUBREG_LEN, 0))
        if period <= 0:
            return None
        cycle = [
            int(f.get(CTRL_OSC_SUBREG_STATE_BASE + m, 0)) & 0xFF for m in range(period)
        ]
        pre = state.maybe_flush_for(reg, -1)
        for k in range(length):
            state.pending_set_writes[reg].append(cycle[k % period])
        return pre or None


class NoteOffDecoder(MacroDecoder):
    """Decode a NOTE_OFF atom (NoteOffPass): re-emit the stored gate-clear ctrl byte inline on its reg,
    byte-identical to the literal SET it re-labels (same value, frame and intra-frame position).
    """

    op_code = NOTE_OFF_OP

    def expand(self, row, state):
        reg = int(row.reg)
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(row.val)
        state.last_diff[reg] = row.diff
        own = (reg, int(state.last_val[reg]), row.diff, row.description)
        return pre + [own]


class CtrlWtDecoder(MacroDecoder):
    """Decode the inline ctrl-state codebook ops (CtrlWavetablePass): CTRL_WT_DEF + CTRL_WT_STEP buffer
    a ctrl byte into a live id->byte table (a later DEF id rebinds), emitting the write on the def's
    voice; CTRL_WT_SET (val=id) re-emits the defined byte on the ref's voice -- byte-identical to the
    literal ctrl SET it re-labels."""

    op_code = -1

    def expand(self, row, state):
        op = int(row.op)
        if op == CTRL_WT_DEF_OP:
            state.pending_ctrl_wt_def = {"id": int(row.val), "reg": int(row.reg)}
            return None
        if op == CTRL_WT_STEP_OP:
            return self._step(row, state)
        return self._ref(row, state)

    def _step(self, row, state):
        pend = state.pending_ctrl_wt_def
        if pend is None or int(row.subreg) != CTRL_WT_SUBREG_VAL:
            return None
        state.pending_ctrl_wt_def = None
        val = int(row.val)
        state.ctrl_wt_table[int(pend["id"])] = val
        return self._emit(int(pend["reg"]), val, row, state)

    def _ref(self, row, state):
        val = state.ctrl_wt_table.get(int(row.val))
        if val is None:
            return None
        return self._emit(int(row.reg), int(val), row, state)

    @staticmethod
    def _emit(reg, val, row, state):
        pre = state.maybe_flush_for(reg, -1)
        state.last_val[reg] = int(val)
        state.last_diff[reg] = row.diff
        return pre + [(reg, int(val), row.diff, row.description)]


class WavetableDecoder(MacroDecoder):
    """Decode the inline-redefinable wavetable codebook ops (design/wavetable_codebook_encoding):
    DEF/STEP/END buffer a note-relative offset program into a live id->program table (a later DEF id
    rebinds); REF (reg=voice freq reg) unrolls the program to LEN frames after the per-hit onset-strip
    LEAD and replays base + LUT[note+off] per frame onto last_skel_note via pending_set_writes,
    byte-identical to the OrnamentDecoder RESID queue it replaces."""

    op_code = -1

    def expand(self, row, state):
        op = int(row.op)
        if op == WAVETABLE_DEF_OP:
            state.pending_wavetable_def = {"id": int(row.val), "steps": [], "loop": 0}
            return None
        if op == WAVETABLE_STEP_OP:
            self._step(row, state)
            return None
        if op == WAVETABLE_END_OP:
            wt = state.pending_wavetable_def
            if wt is not None:
                state.wavetable_table[int(wt["id"])] = (wt["steps"], int(wt["loop"]))
                state.pending_wavetable_def = None
            return None
        if op == WAVETABLE_ONESHOT_OP:
            return self._oneshot(row, state)
        return self._ref(row, state)

    def _oneshot(self, row, state):
        """Decode the self-contained inline one-shot (no codebook id): LEN_HI opens a pending
        program on the voice freq reg, OFFSET/HOLD buffer its verbatim RLE steps, END unrolls and
        replays base + LUT[note+off] per frame -- byte-identical to the ORN RESID queue it replaces.
        """
        subreg = int(row.subreg)
        if subreg == WT_ONESHOT_SUBREG_LEN_HI:
            state.pending_wavetable_oneshot = {
                "reg": int(row.reg),
                "len": (int(row.val) & 0xFF) << 8,
                "steps": [],
            }
            return None
        pend = state.pending_wavetable_oneshot
        if pend is None:
            return None
        if subreg == WT_ONESHOT_SUBREG_LEN_LO:
            pend["len"] |= int(row.val) & 0xFF
            return None
        if subreg == WT_ONESHOT_SUBREG_OFFSET:
            v = int(row.val) & 0xFF
            pend["steps"].append([v if v < 128 else v - 256, 1])
            return None
        if subreg == WT_ONESHOT_SUBREG_HOLD:
            if pend["steps"]:
                pend["steps"][-1][1] = int(row.val) & 0xFFFF
            return None
        return self._replay_oneshot(pend, state)

    @staticmethod
    def _replay_oneshot(pend, state):
        state.pending_wavetable_oneshot = None
        steps = pend["steps"]
        reg = int(pend["reg"])
        note = int(state.last_skel_note.get(reg, 0))
        offsets = wt_unroll(steps, len(steps), int(pend["len"]), [])
        queue = state.pending_set_writes[reg]
        queue.append(int(SKEL_LUT[max(0, min(127, note))]))
        for off in offsets:
            queue.append(int(SKEL_LUT[max(0, min(127, note + int(off)))]))
        return None

    @staticmethod
    def _step(row, state):
        wt = state.pending_wavetable_def
        if wt is None:
            return
        subreg = int(row.subreg)
        if subreg == WT_STEP_SUBREG_OFFSET:
            v = int(row.val) & 0xFF
            wt["steps"].append([v if v < 128 else v - 256, 1])
        elif subreg == WT_STEP_SUBREG_HOLD:
            if wt["steps"]:
                wt["steps"][-1][1] = int(row.val) & 0xFFFF
        elif subreg == WT_STEP_SUBREG_LOOP:
            wt["loop"] = int(row.val) & 0xFFFF

    def _ref(self, row, state):
        subreg = int(row.subreg)
        if subreg == WT_REF_SUBREG_ID:
            state.pending_wavetable_ref = {
                "id": int(row.val),
                "reg": int(row.reg),
                "len": 0,
                "lead": [],
                "lead_n": 0,
            }
            return None
        pend = state.pending_wavetable_ref
        if pend is None:
            return None
        if subreg == WT_REF_SUBREG_LEN_HI:
            pend["len"] |= (int(row.val) & 0xFF) << 8
            return None
        if subreg == WT_REF_SUBREG_LEN_LO:
            pend["len"] |= int(row.val) & 0xFF
            return None
        if subreg == WT_REF_SUBREG_LEAD:
            pend["lead_n"] = int(row.val) & 0xFFFF
            if pend["lead_n"] == 0:
                return self._replay(pend, state)
            return None
        if subreg == WT_REF_SUBREG_LEADOFF:
            v = int(row.val) & 0xFF
            pend["lead"].append(v if v < 128 else v - 256)
            if len(pend["lead"]) >= pend["lead_n"]:
                return self._replay(pend, state)
            return None
        return None

    @staticmethod
    def _replay(pend, state):
        state.pending_wavetable_ref = None
        program = state.wavetable_table.get(int(pend["id"]))
        if program is None:
            return None
        steps, loop = program
        reg = int(pend["reg"])
        note = int(state.last_skel_note.get(reg, 0))
        offsets = wt_unroll(steps, loop, int(pend["len"]), pend["lead"])
        queue = state.pending_set_writes[reg]
        queue.append(int(SKEL_LUT[max(0, min(127, note))]))
        for off in offsets:
            queue.append(int(SKEL_LUT[max(0, min(127, note + int(off)))]))
        return None


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
        SweepDecoder(),
        CtrlOscDecoder(),
        NoteOffDecoder(),
    )
}
DECODERS[NOTE_ON_OP] = DECODERS[NOTE_OFF_OP]
DECODERS.update(codebook_decoders())
_PRESET_DECODER = PresetDecoder()
for _op in PRESET_OPS:
    DECODERS[_op] = _PRESET_DECODER
_SHIFTED_DECODER = ShiftedDecoder()
for _op in PRESET_SHIFTED_OPS:
    DECODERS[_op] = _SHIFTED_DECODER
