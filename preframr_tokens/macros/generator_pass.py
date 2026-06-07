"""GeneratorPass (design/generator_mdl_representation.md): ONE uniform generative model of every SID
value-channel write. Channelizes ``register_state`` into the generator-owned channels (freq x3 +
pw/cut/res/modevol; ctrl/AD/SR stay with InstrumentProgramPass), decomposes each with the self-verifying
longest-wins fitter, and replaces every per-frame raw SET with a generator atom (HOLD/ACCUM -> SWEEP_OP,
TRIANGLE -> GEN_TRI_OP, TABLE -> the GEN_TABLE codebook). Default OFF (``generator_pass``).
"""

__all__ = ["GeneratorPass"]

import numpy as np

from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.generator_fit import (
    _NOTES,
    _lut,
    all_freqs,
    channels,
    fit_run,
    note_of,
    recon,
    tune_ref,
    zig,
)
from preframr_tokens.macros.melody_segment import note_onsets
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    GEN_TABLE_DEF_OP,
    GEN_TABLE_END_OP,
    GEN_TABLE_MODE_ABS,
    GEN_TABLE_MODE_NOTE,
    GEN_TABLE_MODE_NOTE_UNIV,
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_BASE_NOTE,
    GEN_TABLE_REF_SUBREG_ID,
    GEN_TABLE_REF_SUBREG_LEN_HI,
    GEN_TABLE_REF_SUBREG_LEN_LO,
    GEN_TABLE_REF_SUBREG_RESID_HI,
    GEN_TABLE_REF_SUBREG_RESID_LO,
    GEN_TABLE_STEP_OP,
    GEN_TABLE_SUBREG_ABS_HI,
    GEN_TABLE_SUBREG_ABS_LO,
    GEN_TABLE_SUBREG_BASE_NOTE,
    GEN_TABLE_SUBREG_MODE,
    GEN_TABLE_SUBREG_OFFSET,
    GEN_TABLE_SUBREG_PERIOD,
    GEN_TABLE_SUBREG_RESID_HI,
    GEN_TABLE_SUBREG_RESID_LO,
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
    GEN_TUNING_SUBREG_REF,
    GEN_TUNING_SUBREG_VOICE,
    GEN_FREQ_REGS,
    GEN_SCALAR_REGS,
    INSTR_OFF_CTRL,
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
    SET_OP,
    SWEEP_OP,
    SWEEP_SUBREG_DELTA_HI,
    SWEEP_SUBREG_DELTA_LO,
    SWEEP_SUBREG_LEN,
    SWEEP_SUBREG_START_HI,
    SWEEP_SUBREG_START_LO,
)

_GEN_PRIORITY = -8


def _row(reg, op, subreg, val, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=irq, irq=irq)


class GeneratorPass(MacroPass):
    """Decompose every generator-owned channel into the {HOLD, ACCUM, TABLE, TRIANGLE} generator set and
    replace each per-frame raw SET with one byte-exact generator atom. Default OFF (``generator_pass``);
    arbitrated with ``validate=True`` (the self-verifying fitter makes every claim byte-exact, so the
    arbiter never drops -- the guard stays)."""

    GATE_FLAGS = frozenset({"generator_pass"})
    REQUIRES_ARGS = frozenset(
        {"melody_skeleton", "universal_pitch", "universal_freq", "table_resid_split"}
    )

    def apply(self, df, args=None):
        from preframr_tokens.audit_primitives import register_state

        if args is None or not getattr(args, "generator_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        if df["op"].isin((GEN_TABLE_DEF_OP, GEN_TRI_OP, GEN_TUNING_OP)).any():
            return df
        irq = _first_irq(df)
        state = register_state(df)
        regs = tuple(int(r) for r in GEN_FREQ_REGS + GEN_SCALAR_REGS)
        writes = self._collect_writes(df, regs)
        frame_rows = self._frame_marker_rows(df)
        ref_q = max(0, min(255, int(round(tune_ref(all_freqs(state)) * 256.0)))) & 0xFF
        ref = ref_q / 256.0
        mel_ctx = (
            self._melody_context(
                state,
                ref,
                bool(getattr(args, "universal_pitch", False)),
                bool(getattr(args, "universal_freq", False)),
            )
            if getattr(args, "melody_skeleton", False)
            else {}
        )
        split = bool(getattr(args, "table_resid_split", False))
        claims = [self._tuning_claim(ref_q, irq)]
        for m in mel_ctx.values():
            if m.get("tuning_q") is not None:
                claims.append(self._voice_tuning_claim(m["voice"], m["tuning_q"], irq))
        bank = {}
        def_rows = []
        for reg, is_freq, series in channels(state):
            claims.extend(
                self._channel_claims(
                    reg,
                    is_freq,
                    series,
                    writes[reg],
                    frame_rows,
                    ref,
                    bank,
                    def_rows,
                    irq,
                    mel_ctx.get(int(reg)),
                    split,
                )
            )
        if def_rows:
            claims.append(
                Claim(
                    writes=(), tokens=def_rows, priority=_GEN_PRIORITY, label="gen_defs"
                )
            )
        if len(claims) <= 1:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _tuning_claim(ref_q, irq):
        """One head GEN_TUNING atom carrying ``ref_q = round(ref*256)`` (0..255); decoded first (``__pos``
        before every row) so the note-relative freq TABLE replays resolve against it. Consumes no rows.
        The encoder keys note_of/recon off ``ref_q/256`` (the SAME value the decoder reads) so the stored
        residuals are bit-exact under the 8-bit ref quantization. ``__pos`` -2 puts it before the head
        DEFs (-1) and all content, so the tuning ref is set before any DEF buffers or any REF replays.
        """
        row = _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_REF, ref_q, irq)
        row["__pos"] = -2
        return Claim(
            writes=(), tokens=[row], priority=_GEN_PRIORITY, label="gen_tuning"
        )

    @staticmethod
    def _voice_tuning_claim(voice, tuning_q, irq):
        """A per-voice GEN_TUNING atom (``universal_pitch``): VOICE then REF=tuning_q, head-decoded so
        the voice's ``note_freq`` recon is set before any MELODY_INTERVAL replay -- the per-voice tuning
        makes the melody onset residual ~0 (pure note) under chorus/detune. Contiguous (one Claim).
        """
        rows = [
            _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_VOICE, voice, irq),
            _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_REF, tuning_q, irq),
        ]
        for r in rows:
            r["__pos"] = -2
        return Claim(
            writes=(), tokens=rows, priority=_GEN_PRIORITY, label="gen_voice_tuning"
        )

    @classmethod
    def _melody_context(cls, state, ref, universal=False, freq_bulk=False):
        """Per freq reg: ``{voice, onsets, melodic, note, tuning, tuning_q, universal, universal_freq}`` for
        the interval re-keying. ``universal`` (the ``universal_pitch`` flag) sets the PER-VOICE pitch_grid
        ``tuning`` so the interval keys off the universal note index (transferable under chorus/detune);
        ``freq_bulk`` (the ``universal_freq`` probe) extends the re-keying from melodic ONSETS to every
        sounding HOLD/ACCUM atom on the melodic voices -- the bulk pitched-freq stream. Byte-exact either way.
        """
        ctx = {}
        n = int(state.shape[0])
        for b in GEN_FREQ_REGS:
            freqarr = state[:, b].astype("int64") + 256 * state[:, b + 1].astype(
                "int64"
            )
            freq = freqarr.tolist()
            ctrl = (state[:, b + INSTR_OFF_CTRL].astype("int64") & 1).tolist()
            gate_on = [i for i in range(n) if ctrl[i] and (i == 0 or not ctrl[i - 1])]
            per_frame = [int(f) if f > 0 else None for f in freq]
            if universal:
                tq = pitch_grid.tuning_to_q(pitch_grid.voice_tuning(freqarr))
                tuning = pitch_grid.q_to_tuning(tq)
            else:
                tq, tuning = None, None
            ctx[int(b)] = {
                "voice": int(b) // 7,
                "onsets": set(note_onsets(per_frame, gate_on)),
                "melodic": cls._stability(freq, ref),
                "note": None,
                "tuning": tuning,
                "tuning_q": tq,
                "universal": universal,
                "universal_freq": freq_bulk,
            }
        return ctx

    @staticmethod
    def _stability(freq, ref):
        """True iff most sounding frames sit near a LUT note (small fraction of the local semitone gap),
        i.e. the voice settles to a stable note grid -- the cheap, waveform-agnostic melodic test.
        """
        f = np.asarray([int(x) for x in freq if x > 8], dtype=np.int64)
        if len(f) < 8:
            return False
        lut = _lut(ref)
        idx = np.clip(np.searchsorted(lut, f), 1, _NOTES - 1)
        nt = np.where(f - lut[idx - 1] <= lut[idx] - f, idx - 1, idx)
        base = lut[np.clip(nt, 0, _NOTES - 1)]
        span = np.maximum(1, lut[np.clip(nt + 1, 0, _NOTES - 1)] - base)
        good = int(np.sum(np.abs(f - base) <= 0.3 * span))
        return good / len(f) >= 0.6

    @staticmethod
    def _collect_writes(df, target_regs):
        """Per target reg, ordered ``(real_frame, row_idx, val)`` for plain SETs (subreg -1). real_frame
        counts a FRAME_REG as 1 frame and a DELAY_REG as its ``val`` frames -- the decode-frame index the
        per-frame drain unrolls against."""
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        out = {int(r): [] for r in target_regs}
        rf = 0
        for i in range(len(df)):
            reg = int(regs[i])
            if reg == FRAME_REG:
                rf += 1
            elif reg == DELAY_REG:
                rf += int(vals[i])
            elif int(ops[i]) != SET_OP or int(subregs[i]) != -1:
                continue
            elif reg in out:
                out[reg].append((rf, int(i), int(vals[i])))
        return out

    @staticmethod
    def _frame_marker_rows(df):
        """``real_frame -> the marker row a spliced atom decodes inside for that frame``. A generator
        segment that ends INTO a held value starts on a write-less frame, so it has no SET-row anchor;
        it splices at that frame's marker row instead. A marker-group's body dispatches in its LAST
        unrolled frame (the walker runs a DELAY's unroll ticks first, then the body), so a FRAME owns the
        next frame and a DELAY(v) owns its span's LAST frame -- mapping that, mirroring the walker.
        """
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        out = {0: 0}
        rf = 0
        for i in range(len(df)):
            reg = int(regs[i])
            if reg == FRAME_REG:
                rf += 1
                out[rf] = int(i)
            elif reg == DELAY_REG:
                rf += int(vals[i])
                out[rf] = int(i)
        return out

    @classmethod
    def _channel_claims(
        cls,
        reg,
        is_freq,
        series,
        writes,
        frame_rows,
        ref,
        bank,
        def_rows,
        irq,
        mel=None,
        split=False,
    ):
        """Walk one channel's timeline left-to-right (first write -> the global final frame), re-fitting
        the longest generator from each ANCHORED frame and emitting one Claim for it. A write-less frame
        with no marker row (a held value mid-DELAY-span) carries no write to drop and its value is already
        held by the prior drain, so it is skipped; every actual write sits on an anchored frame and is
        therefore covered. The bank (shared across channels) interns TABLE cycles for DEF->REF reuse.
        """
        if not writes:
            return []
        f0 = int(writes[0][0])
        f1 = len(series) - 1
        anchor = {}
        for frame, ri, _v in writes:
            anchor.setdefault(int(frame), int(ri))
        drop_at = {}
        for frame, ri, _v in writes:
            drop_at.setdefault(int(frame), []).append(int(ri))
        claims = []
        i = f0
        while i <= f1:
            pos = anchor.get(i)
            if pos is None:
                pos = frame_rows.get(i)
            if pos is None:
                i += 1
                continue
            kind, length, params = fit_run(series, i)
            length = max(1, length)
            b = i + length - 1
            seg = [int(series[f]) for f in range(i, b + 1)]
            drop = tuple(ri for fr in range(i, b + 1) for ri in drop_at.get(fr, ()))
            rows = cls._atom_rows(
                reg,
                is_freq,
                kind,
                params,
                seg,
                length,
                ref,
                bank,
                def_rows,
                irq,
                i,
                mel,
                split,
            )
            if rows is not None:
                for r in rows:
                    r["__pos"] = pos
                claims.append(
                    Claim(
                        writes=drop,
                        tokens=rows,
                        priority=_GEN_PRIORITY,
                        label="generator",
                    )
                )
            i = b + 1
        return claims

    @classmethod
    def _atom_rows(
        cls,
        reg,
        is_freq,
        kind,
        params,
        seg,
        length,
        ref,
        bank,
        def_rows,
        irq,
        i=0,
        mel=None,
        split=False,
    ):
        """The token rows (REF/atom) for one generator segment. HOLD/ACCUM -> SWEEP_OP (delta 0 for
        HOLD), TRI -> GEN_TRI_OP, TABLE -> a GEN_TABLE REF (the matching DEF is appended to ``def_rows``
        at the stream head on first sight of its bank key, so no REF can precede its DEF). With the
        melody-skeleton ``mel`` context, a HOLD/ACCUM onset on a melodic voice is re-keyed to a
        MELODY_INTERVAL atom (note-relative), the writes unchanged.
        """
        if mel is not None and is_freq and kind in ("HOLD", "ACCUM") and mel["melodic"]:
            onset = int(i) in mel["onsets"]
            bulk = bool(mel.get("universal_freq")) and int(seg[0]) > 8
            if onset or bulk:
                delta = int(params) if kind == "ACCUM" else 0
                return cls._melody_rows(mel, int(seg[0]), delta, length, ref, irq)
        if kind == "HOLD":
            return cls._sweep_rows(reg, seg[0], 0, length, irq)
        if kind == "ACCUM":
            return cls._sweep_rows(reg, seg[0], int(params), length, irq)
        if kind == "TRI":
            step, lo, hi, dir0 = params
            return cls._tri_rows(reg, seg[0], step, lo, hi, dir0, length, irq)
        if kind == "TABLE":
            return cls._table_rows(
                reg,
                is_freq,
                int(params),
                seg,
                length,
                ref,
                bank,
                def_rows,
                irq,
                split,
                mel,
            )
        return None

    @staticmethod
    def _melody_rows(mel, onset_freq, delta, length, ref, irq):
        """The MELODY_INTERVAL atom for a re-keyed freq onset: the note as FIRST-absolute or a zig-zag
        interval from the voice's previous keyed note, the exact residual, and the HOLD/ACCUM delta. The
        decoder's running interval sum + residual reproduces ``onset_freq`` bit-exact.
        """
        if mel.get("universal"):
            note = int(
                pitch_grid.note_index(np.asarray([onset_freq]), mel["tuning"])[0]
            )
            resid = (onset_freq - pitch_grid.note_freq_at(note, mel["tuning"])) & 0xFFFF
        else:
            note = note_of(onset_freq, ref)
            resid = (onset_freq - recon(note, ref)) & 0xFFFF
        prev = mel["note"]
        if prev is None:
            first, token = 1, note & 0xFFFF
        else:
            first, token = 0, zig(note - prev) & 0xFFFF
        mel["note"] = note
        d = delta & 0xFFFF
        voice = mel["voice"]
        reg = voice * 7
        return [
            _row(reg, MELODY_INTERVAL_OP, MELODY_INTERVAL_SUBREG_VOICE, voice, irq),
            _row(reg, MELODY_INTERVAL_OP, MELODY_INTERVAL_SUBREG_FIRST, first, irq),
            _row(
                reg,
                MELODY_INTERVAL_OP,
                MELODY_INTERVAL_SUBREG_INTERVAL_HI,
                (token >> 8) & 0xFF,
                irq,
            ),
            _row(
                reg,
                MELODY_INTERVAL_OP,
                MELODY_INTERVAL_SUBREG_INTERVAL_LO,
                token & 0xFF,
                irq,
            ),
            _row(
                reg,
                MELODY_INTERVAL_OP,
                MELODY_INTERVAL_SUBREG_RESID_HI,
                (resid >> 8) & 0xFF,
                irq,
            ),
            _row(
                reg,
                MELODY_INTERVAL_OP,
                MELODY_INTERVAL_SUBREG_RESID_LO,
                resid & 0xFF,
                irq,
            ),
            _row(
                reg,
                MELODY_INTERVAL_OP,
                MELODY_INTERVAL_SUBREG_DELTA_HI,
                (d >> 8) & 0xFF,
                irq,
            ),
            _row(
                reg, MELODY_INTERVAL_OP, MELODY_INTERVAL_SUBREG_DELTA_LO, d & 0xFF, irq
            ),
            _row(reg, MELODY_INTERVAL_OP, MELODY_INTERVAL_SUBREG_LEN, length, irq),
        ]

    @staticmethod
    def _sweep_rows(reg, start, delta, length, irq):
        d = delta & 0xFFFF
        return [
            _row(reg, SWEEP_OP, SWEEP_SUBREG_START_HI, (start >> 8) & 0xFF, irq),
            _row(reg, SWEEP_OP, SWEEP_SUBREG_START_LO, start & 0xFF, irq),
            _row(reg, SWEEP_OP, SWEEP_SUBREG_DELTA_HI, (d >> 8) & 0xFF, irq),
            _row(reg, SWEEP_OP, SWEEP_SUBREG_DELTA_LO, d & 0xFF, irq),
            _row(reg, SWEEP_OP, SWEEP_SUBREG_LEN, length, irq),
        ]

    @staticmethod
    def _tri_rows(reg, start, step, lo, hi, dir0, length, irq):
        s = int(step) & 0xFFFF
        lo = int(lo) & 0xFFFF
        hi = int(hi) & 0xFFFF
        return [
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_START_HI, (start >> 8) & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_START_LO, start & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_STEP_HI, (s >> 8) & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_STEP_LO, s & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_LO_HI, (lo >> 8) & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_LO_LO, lo & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_HI_HI, (hi >> 8) & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_HI_LO, hi & 0xFF, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_DIR, 1 if dir0 > 0 else 0, irq),
            _row(reg, GEN_TRI_OP, GEN_TRI_SUBREG_LEN, length, irq),
        ]

    @classmethod
    def _table_rows(
        cls,
        reg,
        is_freq,
        period,
        seg,
        length,
        ref,
        bank,
        def_rows,
        irq,
        split=False,
        mel=None,
    ):
        """A GEN_TABLE REF for a periodic TABLE cycle; the DEF appends to ``def_rows`` on first sight of
        its bank key. ``split`` keys the freq DEF on OFFSETS ALONE and moves the residual to the
        per-instance REF (de-fragments); ``split``+universal computes offsets/residual off the PER-VOICE
        ``note_index`` (item #4, mode NOTE_UNIV) so static residuals go ~0, decoded via the voice tuning.
        Scalar keys absolutely. All paths byte-exact."""
        cycle = seg[:period]
        univ = bool(
            split
            and is_freq
            and mel is not None
            and mel.get("universal")
            and mel.get("tuning") is not None
        )
        mode = GEN_TABLE_MODE_ABS
        if univ:
            tuning = mel["tuning"]
            notes = [
                int(pitch_grid.note_index(np.asarray([int(c)]), tuning)[0])
                for c in cycle
            ]
            base_note = notes[0]
            offs = [n - base_note for n in notes]
            resids = [
                int(c) - pitch_grid.note_freq_at(n, tuning)
                for c, n in zip(cycle, notes)
            ]
            if 0 <= base_note <= 255 and all(-128 <= o <= 127 for o in offs):
                key = ("noteU", tuple(offs))
                mode = GEN_TABLE_MODE_NOTE_UNIV
            else:
                univ = False
        if is_freq and not univ:
            base_note = note_of(cycle[0], ref)
            offs, resids = [], []
            for c in cycle:
                nt = note_of(c, ref)
                offs.append(nt - base_note)
                resids.append(int(c) - recon(nt, ref))
            key = (
                ("note", tuple(offs)) if split else ("note", tuple(offs), tuple(resids))
            )
            mode = GEN_TABLE_MODE_NOTE
        elif not is_freq:
            base_note = 0
            offs = resids = None
            key = ("abs", reg, tuple(int(c) for c in cycle))
        if key not in bank:
            cb_id = len(bank)
            bank[key] = cb_id
            def_resids = None if (split and is_freq) else resids
            for r in cls._def_rows(
                cb_id, is_freq, period, base_note, cycle, offs, def_resids, irq, mode
            ):
                r["__pos"] = -1
                def_rows.append(r)
        else:
            cb_id = bank[key]
        ref_resids = resids if (split and is_freq) else None
        return cls._ref_rows(reg, cb_id, is_freq, base_note, length, irq, ref_resids)

    @staticmethod
    def _def_rows(cb_id, is_freq, period, base_note, cycle, offs, resids, irq, mode):
        rows = [_row(0, GEN_TABLE_DEF_OP, -1, cb_id, irq)]

        def step(subreg, val):
            rows.append(_row(0, GEN_TABLE_STEP_OP, subreg, val, irq))

        split = resids is None and mode != GEN_TABLE_MODE_ABS
        step(GEN_TABLE_SUBREG_PERIOD, period)
        if is_freq:
            step(GEN_TABLE_SUBREG_MODE, mode)
            if not split:
                step(GEN_TABLE_SUBREG_BASE_NOTE, base_note & 0xFF)
            for m in range(period):
                step(GEN_TABLE_SUBREG_OFFSET, offs[m] & 0xFF)
                if resids is not None:
                    r = resids[m] & 0xFFFF
                    step(GEN_TABLE_SUBREG_RESID_LO, r & 0xFF)
                    step(GEN_TABLE_SUBREG_RESID_HI, (r >> 8) & 0xFF)
        else:
            step(GEN_TABLE_SUBREG_MODE, GEN_TABLE_MODE_ABS)
            for m in range(period):
                v = int(cycle[m]) & 0xFFFF
                step(GEN_TABLE_SUBREG_ABS_LO, v & 0xFF)
                step(GEN_TABLE_SUBREG_ABS_HI, (v >> 8) & 0xFF)
        rows.append(_row(0, GEN_TABLE_END_OP, -1, cb_id, irq))
        return rows

    @staticmethod
    def _ref_rows(reg, cb_id, is_freq, base_note, length, irq, resids=None):
        rows = [_row(reg, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_ID, cb_id, irq)]
        if is_freq:
            rows.append(
                _row(
                    reg,
                    GEN_TABLE_REF_OP,
                    GEN_TABLE_REF_SUBREG_BASE_NOTE,
                    base_note & 0xFF,
                    irq,
                )
            )
        if resids is not None:
            for r in resids:
                rq = int(r) & 0xFFFF
                rows.append(
                    _row(
                        reg,
                        GEN_TABLE_REF_OP,
                        GEN_TABLE_REF_SUBREG_RESID_LO,
                        rq & 0xFF,
                        irq,
                    )
                )
                rows.append(
                    _row(
                        reg,
                        GEN_TABLE_REF_OP,
                        GEN_TABLE_REF_SUBREG_RESID_HI,
                        (rq >> 8) & 0xFF,
                        irq,
                    )
                )
        rows.append(
            _row(
                reg,
                GEN_TABLE_REF_OP,
                GEN_TABLE_REF_SUBREG_LEN_HI,
                (length >> 8) & 0xFF,
                irq,
            )
        )
        rows.append(
            _row(reg, GEN_TABLE_REF_OP, GEN_TABLE_REF_SUBREG_LEN_LO, length & 0xFF, irq)
        )
        return rows
