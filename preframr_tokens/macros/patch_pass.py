"""PatchPass: melodic-instrument ADSR patch codebook (design/patch_preamble_encoding.md). Mines
per-voice frames writing BOTH AD (reg+5) and SR (reg+6) (a full envelope load); any (ad,sr) state
recurring >= PATCH_MINREP times drains to an inline PATCH_DEF + one-atom PATCH_SET reuses (a Claim),
byte-exact, factoring timbre out of the per-note stream. ADSR only (waveform-cycle/PW/MUT deferred);
opt-in (``patch_pass``), default OFF. Separability/token-budget, not RESID -- the twin of StampPass.
"""

__all__ = ["PatchPass"]

from collections import defaultdict

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.codebook_emit import emit_recurring
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    make_row,
    MacroPass,
)
from preframr_tokens.stfconstants import (
    _MIN_DIFF,
    FREQ_TRAJ_REGS,
    PATCH_AD_OFFSET,
    PATCH_DEF_OP,
    PATCH_MINREP,
    PATCH_SET_OP,
    PATCH_SR_OFFSET,
    PATCH_STEP_OP,
    PATCH_SUBREG_AD,
    PATCH_SUBREG_ID,
    PATCH_SUBREG_SR,
    SET_OP,
)

_PATCH_PRIORITY = -5


def _row(reg, op, subreg, val, irq):
    return make_row(reg, val, op=op, subreg=subreg, diff=_MIN_DIFF, irq=irq)


class PatchPass(MacroPass):
    """Mine recurring full (AD,SR) envelope loads per tune and replace them with an inline
    PATCH_DEF + per-reuse PATCH_SET, consuming the raw AD/SR writes. Default OFF."""

    GATE_FLAGS = frozenset({"patch_pass"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "patch_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        events = self._events(df)
        drop_idx, new_rows = self._emit(events, irq)
        if not new_rows:
            return df
        return arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_PATCH_PRIORITY,
                    label="patch",
                )
            ],
        )

    @staticmethod
    def _events(df):
        """Per voice, frames writing BOTH AD (reg+5) and SR (reg+6) as plain SETs -- a full envelope
        load. Returns dicts {freq_reg, ad, sr, ad_row, sr_row, pos} (pos = earliest consumed row, so
        the DEF can be ordered strictly before every reuse)."""
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        ad_regs = {int(reg) + PATCH_AD_OFFSET: int(reg) for reg in FREQ_TRAJ_REGS}
        sr_regs = {int(reg) + PATCH_SR_OFFSET: int(reg) for reg in FREQ_TRAJ_REGS}
        ad_at, sr_at = {}, {}
        ad_count, sr_count = defaultdict(int), defaultdict(int)
        for i in range(len(df)):
            if int(ops[i]) != SET_OP or int(subregs[i]) != -1:
                continue
            reg = int(regs[i])
            if reg in ad_regs:
                key = (ad_regs[reg], int(f_idx[i]))
                ad_at[key] = (int(i), int(vals[i]))
                ad_count[key] += 1
            elif reg in sr_regs:
                key = (sr_regs[reg], int(f_idx[i]))
                sr_at[key] = (int(i), int(vals[i]))
                sr_count[key] += 1
        events = []
        for key in set(ad_at) & set(sr_at):
            if ad_count[key] != 1 or sr_count[key] != 1:
                continue
            freq_reg, _frame = key
            ad_row, ad = ad_at[key]
            sr_row, sr = sr_at[key]
            events.append(
                {
                    "freq_reg": freq_reg,
                    "ad": ad,
                    "sr": sr,
                    "ad_row": ad_row,
                    "sr_row": sr_row,
                    "pos": min(ad_row, sr_row),
                }
            )
        return events

    @staticmethod
    def _emit(events, irq):
        """Group events by (freq_reg, ad, sr) -- per-voice, so a def and all its PATCH_SET reuses stay in
        one voice and keep def-before-ref under the voice-major _norm_pr_order (a global cross-voice
        codebook let a reuse sort ahead of its def -> "id not live"; trades cross-voice sharing for
        byte-exactness). Each (ad,sr) recurring >= PATCH_MINREP becomes a PATCH_DEF on its earliest
        occurrence and PATCH_SET backrefs after, via the shared recurring-codebook skeleton.
        """
        groups = defaultdict(list)
        for ev in events:
            groups[(ev["freq_reg"], ev["ad"], ev["sr"])].append(ev)

        def emit_first(cb_id, occ):
            ev = occ[0]
            return [
                _row(ev["freq_reg"], PATCH_DEF_OP, PATCH_SUBREG_ID, cb_id, irq),
                _row(ev["freq_reg"], PATCH_STEP_OP, PATCH_SUBREG_AD, ev["ad"], irq),
                _row(ev["freq_reg"], PATCH_STEP_OP, PATCH_SUBREG_SR, ev["sr"], irq),
            ]

        def emit_ref(cb_id, ev):
            return [_row(ev["freq_reg"], PATCH_SET_OP, -1, cb_id, irq)]

        drop_idx, new_rows, _ = emit_recurring(
            groups,
            minrep=PATCH_MINREP,
            group_sort=lambda kv: (min(e["pos"] for e in kv[1]), kv[0]),
            occ_sort=lambda e: e["pos"],
            pos_of=lambda e: e["pos"],
            rows_of=lambda e: (e["ad_row"], e["sr_row"]),
            emit_first=emit_first,
            emit_ref=emit_ref,
        )
        return drop_idx, new_rows
