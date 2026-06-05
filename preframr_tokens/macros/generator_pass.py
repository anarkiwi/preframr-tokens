"""GeneratorPass (design/generator_mdl_representation.md): ONE uniform generative model of every SID
value-channel write. Channelizes ``register_state`` into the generator-owned channels (freq x3 +
pw/cut/res/modevol; ctrl/AD/SR stay with InstrumentProgramPass), decomposes each with the self-verifying
longest-wins fitter, and replaces every per-frame raw SET with a generator atom (HOLD/ACCUM -> SWEEP_OP,
TRIANGLE -> GEN_TRI_OP, TABLE -> the GEN_TABLE codebook). Default OFF (``generator_pass``).
"""

__all__ = ["GeneratorPass"]

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.generator_fit import (
    all_freqs,
    channels,
    decompose,
    note_of,
    recon,
    tune_ref,
)
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
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_BASE_NOTE,
    GEN_TABLE_REF_SUBREG_ID,
    GEN_TABLE_REF_SUBREG_LEN_HI,
    GEN_TABLE_REF_SUBREG_LEN_LO,
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
    GEN_FREQ_REGS,
    GEN_SCALAR_REGS,
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
        ref_q = max(0, min(255, int(round(tune_ref(all_freqs(state)) * 256.0)))) & 0xFF
        ref = ref_q / 256.0
        claims = [self._tuning_claim(ref_q, irq)]
        bank = {}
        for reg, is_freq, series in channels(state):
            claims.extend(
                self._channel_claims(reg, is_freq, series, writes[reg], ref, bank, irq)
            )
        if len(claims) <= 1:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _tuning_claim(ref_q, irq):
        """One head GEN_TUNING atom carrying ``ref_q = round(ref*256)`` (0..255); decoded first (``__pos``
        before every row) so the note-relative freq TABLE replays resolve against it. Consumes no rows.
        The encoder keys note_of/recon off ``ref_q/256`` (the SAME value the decoder reads) so the stored
        residuals are bit-exact under the 8-bit ref quantization."""
        row = _row(0, GEN_TUNING_OP, GEN_TUNING_SUBREG_REF, ref_q, irq)
        row["__pos"] = -1
        return Claim(
            writes=(), tokens=[row], priority=_GEN_PRIORITY, label="gen_tuning"
        )

    @staticmethod
    def _collect_writes(df, target_regs):
        """Per target reg, ordered ``(real_frame, row_idx, val)`` for plain SETs (subreg -1). real_frame
        counts a FRAME_REG as 1 frame and a DELAY_REG as its ``val`` frames -- the decode-frame index the
        per-frame drain unrolls against (mirrors GlobalOscPass)."""
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

    @classmethod
    def _channel_claims(cls, reg, is_freq, series, writes, ref, bank, irq):
        """Decompose one channel's ``[first_write, last_write]`` timeline and emit one Claim per
        generator. Every segment starts on a value-change frame (a written frame), so each atom anchors
        on a real SET row; the bank (shared across channels) interns TABLE cycles for DEF->REF reuse.
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
        sub = series[f0 : f1 + 1]
        claims = []
        for kind, rel, length, params in decompose(sub):
            a = f0 + rel
            b = a + length - 1
            pos = anchor.get(a)
            if pos is None:
                continue
            drop = tuple(ri for fr in range(a, b + 1) for ri in drop_at.get(fr, ()))
            seg = [int(series[f]) for f in range(a, b + 1)]
            rows = cls._atom_rows(
                reg, is_freq, kind, params, seg, length, ref, bank, irq
            )
            if rows is None:
                continue
            for r in rows:
                r["__pos"] = pos
            claims.append(
                Claim(
                    writes=drop, tokens=rows, priority=_GEN_PRIORITY, label="generator"
                )
            )
        return claims

    @classmethod
    def _atom_rows(cls, reg, is_freq, kind, params, seg, length, ref, bank, irq):
        """The token rows for one generator segment. HOLD/ACCUM -> SWEEP_OP (delta 0 for HOLD), TRI ->
        GEN_TRI_OP, TABLE -> a GEN_TABLE DEF (first sight of its bank key) + REF, else a bare REF.
        """
        if kind == "HOLD":
            return cls._sweep_rows(reg, seg[0], 0, length, irq)
        if kind == "ACCUM":
            return cls._sweep_rows(reg, seg[0], int(params), length, irq)
        if kind == "TRI":
            step, lo, hi, dir0 = params
            return cls._tri_rows(reg, seg[0], step, lo, hi, dir0, length, irq)
        if kind == "TABLE":
            return cls._table_rows(
                reg, is_freq, int(params), seg, length, ref, bank, irq
            )
        return None

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
    def _table_rows(cls, reg, is_freq, period, seg, length, ref, bank, irq):
        """DEF+REF (first sight of the bank key) or a bare REF for a periodic TABLE cycle. Scalar channels
        key absolutely on the raw cycle bytes; freq keys note-relative on (offset cycle, residual cycle)
        so transposed arps share one entry, the per-instance base_note riding on the REF.
        """
        cycle = seg[:period]
        if is_freq:
            base_note = note_of(cycle[0], ref)
            offs, resids = [], []
            for c in cycle:
                nt = note_of(c, ref)
                offs.append(nt - base_note)
                resids.append(int(c) - recon(nt, ref))
            key = ("note", tuple(offs), tuple(resids))
        else:
            base_note = 0
            offs = resids = None
            key = ("abs", reg, tuple(int(c) for c in cycle))
        rows = []
        if key not in bank:
            cb_id = len(bank)
            bank[key] = cb_id
            rows.extend(
                cls._def_rows(
                    cb_id, is_freq, period, base_note, cycle, offs, resids, irq
                )
            )
        else:
            cb_id = bank[key]
        rows.extend(cls._ref_rows(reg, cb_id, is_freq, base_note, length, irq))
        return rows

    @staticmethod
    def _def_rows(cb_id, is_freq, period, base_note, cycle, offs, resids, irq):
        rows = [_row(0, GEN_TABLE_DEF_OP, -1, cb_id, irq)]

        def step(subreg, val):
            rows.append(_row(0, GEN_TABLE_STEP_OP, subreg, val, irq))

        step(GEN_TABLE_SUBREG_PERIOD, period)
        if is_freq:
            step(GEN_TABLE_SUBREG_MODE, GEN_TABLE_MODE_NOTE)
            step(GEN_TABLE_SUBREG_BASE_NOTE, base_note & 0xFF)
            for m in range(period):
                step(GEN_TABLE_SUBREG_OFFSET, offs[m] & 0xFF)
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
    def _ref_rows(reg, cb_id, is_freq, base_note, length, irq):
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
