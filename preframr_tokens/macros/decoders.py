"""Per-op decoder dispatch."""

__all__ = [
    "DECODERS",
    "MacroDecoder",
    "SetDecoder",
    "DiffDecoder",
    "FlipDecoder",
    "TransposeDecoder",
    "HardRestartDecoder",
    "SubregFlushDecoder",
    "SweepDecoder",
    "GenTriDecoder",
    "GenTuningDecoder",
    "MelodyIntervalDecoder",
]

from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    MELODY_INTERVAL_OP,
    MELODY_INTERVAL_SUBREG_DELTA_HI,
    MELODY_INTERVAL_SUBREG_DELTA_LO,
    MELODY_INTERVAL_SUBREG_FIRST,
    MELODY_INTERVAL_SUBREG_INTERVAL_HI,
    MELODY_INTERVAL_SUBREG_INTERVAL_LO,
    MELODY_INTERVAL_SUBREG_LEN,
    MELODY_INTERVAL_SUBREG_RESID_HI,
    MELODY_INTERVAL_SUBREG_RESID_LO,
    MELODY_INTERVAL_SUBREG_VOICE,
    GEN_TRI_OP,
    GEN_TRI_SUBREG_DIR,
    GEN_TRI_SUBREG_HI_HI,
    GEN_TRI_SUBREG_HI_LO,
    GEN_TRI_SUBREG_LEN,
    GEN_TRI_SUBREG_LO_HI,
    GEN_TRI_SUBREG_LO_LO,
    GEN_TRI_SUBREG_START_HI,
    GEN_TRI_SUBREG_START_LO,
    GEN_TRI_SUBREG_STEP_HI,
    GEN_TRI_SUBREG_STEP_LO,
    GEN_TUNING_OP,
    GEN_TUNING_SUBREG_FREQ_HI,
    GEN_TUNING_SUBREG_FREQ_LO,
    GEN_TUNING_SUBREG_NOTE,
    GEN_TUNING_SUBREG_REF,
    GEN_TUNING_SUBREG_VOICE,
    DIFF_OP,
    FLIP_OP,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_3,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    SET_OP,
    SUBREG_FLUSH_OP,
    SWEEP_OP,
    SWEEP_SUBREG_DELTA_HI,
    SWEEP_SUBREG_DELTA_LO,
    SWEEP_SUBREG_LEN,
    SWEEP_SUBREG_PERIOD,
    SWEEP_SUBREG_START_HI,
    SWEEP_SUBREG_START_LO,
    TRANSPOSE_OP,
    VOICES,
)
from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.codebook import codebook_decoders
from preframr_tokens.macros.generator_fit import _tri_seq, recon, unzig


def _s16(raw):
    raw &= 0xFFFF
    return raw if raw < 0x8000 else raw - 0x10000


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


class SubregFlushDecoder(MacroDecoder):
    """Force-flush deferred subreg state. Inserted by SubregPass between two
    consecutive subreg rows that are on the same reg, touch different
    nibbles, AND came from different baseline SETs (so they would otherwise
    coalesce and lose the intermediate write)."""

    op_code = SUBREG_FLUSH_OP

    def expand(self, row, state):
        return state.flush_pending_subreg() or None


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


class GenTuningDecoder(MacroDecoder):
    """Decode a GEN_TUNING atom (GeneratorPass): store the per-tune semitone-LUT offset
    ``ref = ref_q / 256`` on the decode state for the note-relative freq TABLE codebook, emit no
    register write."""

    op_code = GEN_TUNING_OP

    def expand(self, row, state):
        sub = int(row.subreg)
        val = int(row.val)
        if sub == GEN_TUNING_SUBREG_VOICE:
            state.pending_gen_tuning_voice = val
            state.pending_gen_tuning_note = None
            return None
        if sub == GEN_TUNING_SUBREG_REF:
            voice = state.pending_gen_tuning_voice
            if voice is None:
                state.gen_ref = (val & 0xFF) / 256.0
            else:
                state.gen_ref_by_voice[voice] = pitch_grid.q_to_tuning(val)
            return None
        if sub == GEN_TUNING_SUBREG_NOTE:
            state.pending_gen_tuning_note = val - 256 if val >= 128 else val
            return None
        if sub == GEN_TUNING_SUBREG_FREQ_LO:
            state.pending_gen_tuning_flo = val & 0xFF
            return None
        if sub == GEN_TUNING_SUBREG_FREQ_HI:
            voice = state.pending_gen_tuning_voice
            note = state.pending_gen_tuning_note
            if voice is not None and note is not None:
                freq = ((val & 0xFF) << 8) | state.pending_gen_tuning_flo
                state.gen_table_by_voice.setdefault(voice, {})[note] = freq
        return None


class GenTriDecoder(MacroDecoder):
    """Decode a GEN_TRI atom (GeneratorPass): START/STEP/LO/HI/DIR/LEN replay the bounded reversing
    zigzag ``_tri_seq`` (vibrato / PW auto-reverse / filter triangle) and queue its LEN per-frame values
    onto the reg, one drained per song frame -- the byte-exact triangle generator."""

    op_code = GEN_TRI_OP

    def expand(self, row, state):
        sub = int(row.subreg)
        if sub == GEN_TRI_SUBREG_START_HI:
            state.pending_gen_tri = {"reg": int(row.reg), "fields": {}}
        pend = state.pending_gen_tri
        if pend is None:
            return None
        pend["fields"][sub] = int(row.val)
        if sub != GEN_TRI_SUBREG_LEN:
            return None
        state.pending_gen_tri = None
        f = pend["fields"]
        reg = int(pend["reg"])
        start = ((f.get(GEN_TRI_SUBREG_START_HI, 0) & 0xFF) << 8) | (
            f.get(GEN_TRI_SUBREG_START_LO, 0) & 0xFF
        )
        step = ((f.get(GEN_TRI_SUBREG_STEP_HI, 0) & 0xFF) << 8) | (
            f.get(GEN_TRI_SUBREG_STEP_LO, 0) & 0xFF
        )
        lo = ((f.get(GEN_TRI_SUBREG_LO_HI, 0) & 0xFF) << 8) | (
            f.get(GEN_TRI_SUBREG_LO_LO, 0) & 0xFF
        )
        hi = ((f.get(GEN_TRI_SUBREG_HI_HI, 0) & 0xFF) << 8) | (
            f.get(GEN_TRI_SUBREG_HI_LO, 0) & 0xFF
        )
        d = 1 if int(f.get(GEN_TRI_SUBREG_DIR, 1)) >= 1 else -1
        length = int(f.get(GEN_TRI_SUBREG_LEN, 0))
        pre = state.maybe_flush_for(reg, -1)
        for v in _tri_seq(start, step, lo, hi, d, length):
            state.pending_set_writes[reg].append(int(v) & 0xFFFF)
        return pre or None


class MelodyIntervalDecoder(MacroDecoder):
    """Decode a MELODY_INTERVAL atom (melody-skeleton layer 2): a freq note-onset re-keyed off the
    voice's previous note. FIRST sets ``cur_note`` absolute, else ``cur_note += unzig(interval)``; the
    onset freq is ``recon(cur_note, gen_ref) + resid`` and the carried HOLD/ACCUM delta replays LEN
    per-frame freqs onto the voice's freq reg -- byte-exact, the running interval sum reproducing the
    absolute note."""

    op_code = MELODY_INTERVAL_OP

    def expand(self, row, state):
        sub = int(row.subreg)
        if sub == MELODY_INTERVAL_SUBREG_VOICE:
            state.pending_melody_interval = {"fields": {}}
        pend = state.pending_melody_interval
        if pend is None:
            return None
        pend["fields"][sub] = int(row.val)
        if sub != MELODY_INTERVAL_SUBREG_LEN:
            return None
        state.pending_melody_interval = None
        f = pend["fields"]
        voice = int(f.get(MELODY_INTERVAL_SUBREG_VOICE, 0)) % VOICES
        reg = int(FREQ_REGS_BY_VOICE[voice])
        raw_iv = ((f.get(MELODY_INTERVAL_SUBREG_INTERVAL_HI, 0) & 0xFF) << 8) | (
            f.get(MELODY_INTERVAL_SUBREG_INTERVAL_LO, 0) & 0xFF
        )
        if int(f.get(MELODY_INTERVAL_SUBREG_FIRST, 0)):
            cur_note = raw_iv
        else:
            cur_note = state.melody_interval_cur_note[voice] + unzig(raw_iv)
        state.melody_interval_cur_note[voice] = cur_note
        resid = _s16(
            ((f.get(MELODY_INTERVAL_SUBREG_RESID_HI, 0) & 0xFF) << 8)
            | (f.get(MELODY_INTERVAL_SUBREG_RESID_LO, 0) & 0xFF)
        )
        delta = _s16(
            ((f.get(MELODY_INTERVAL_SUBREG_DELTA_HI, 0) & 0xFF) << 8)
            | (f.get(MELODY_INTERVAL_SUBREG_DELTA_LO, 0) & 0xFF)
        )
        tbl = state.gen_table_by_voice.get(voice)
        if tbl is not None and cur_note in tbl:
            base = tbl[cur_note]
        elif voice in state.gen_ref_by_voice:
            base = pitch_grid.note_freq_at(cur_note, state.gen_ref_by_voice[voice])
        else:
            base = recon(cur_note, state.gen_ref)
        start = (base + resid) & 0xFFFF
        pre = state.maybe_flush_for(reg, -1)
        for k in range(int(f.get(MELODY_INTERVAL_SUBREG_LEN, 0))):
            state.pending_set_writes[reg].append((start + k * delta) & 0xFFFF)
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
        SweepDecoder(),
        GenTriDecoder(),
        GenTuningDecoder(),
        MelodyIntervalDecoder(),
    )
}
DECODERS.update(codebook_decoders())
