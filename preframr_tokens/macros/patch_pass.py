"""PatchPass: melodic-instrument ADSR patch codebook (design/patch_preamble_encoding.md). Mines
per-voice frames writing BOTH AD (reg+5) and SR (reg+6) (a full envelope load); any (ad,sr) state
recurring >= PATCH_MINREP times drains to an inline PATCH_DEF + one-atom PATCH_SET reuses (a Claim),
byte-exact, factoring timbre out of the per-note stream. ADSR only (waveform-cycle/PW/MUT deferred);
opt-in (``patch_pass``), default OFF. Separability/token-budget, not RESID -- the twin of StampPass.
"""

__all__ = ["PatchPass"]

from collections import defaultdict

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
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
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(_MIN_DIFF),
        "op": int(op),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


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
        """Group events by (ad,sr); for each state recurring >= PATCH_MINREP, the earliest-positioned
        occurrence becomes the PATCH_DEF and the rest become PATCH_SET backrefs. ids are assigned in
        DEF-position order so every backref's def precedes it in the row stream."""
        groups = defaultdict(list)
        for ev in events:
            groups[(ev["ad"], ev["sr"])].append(ev)
        recurring = []
        for key, occ in groups.items():
            if len(occ) >= PATCH_MINREP:
                occ = sorted(occ, key=lambda e: e["pos"])
                recurring.append((occ[0]["pos"], key, occ))
        recurring.sort()
        drop_idx, new_rows = [], []
        for patch_id, (_pos, (ad, sr), occ) in enumerate(recurring):
            for j, ev in enumerate(occ):
                if j == 0:
                    rows = [
                        _row(
                            ev["freq_reg"], PATCH_DEF_OP, PATCH_SUBREG_ID, patch_id, irq
                        ),
                        _row(ev["freq_reg"], PATCH_STEP_OP, PATCH_SUBREG_AD, ad, irq),
                        _row(ev["freq_reg"], PATCH_STEP_OP, PATCH_SUBREG_SR, sr, irq),
                    ]
                else:
                    rows = [_row(ev["freq_reg"], PATCH_SET_OP, -1, patch_id, irq)]
                for r in rows:
                    r["__pos"] = ev["pos"]
                    new_rows.append(r)
                drop_idx.append(ev["ad_row"])
                drop_idx.append(ev["sr_row"])
        return drop_idx, new_rows
