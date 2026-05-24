"""Collapse periodic FREQ vibrato into one ``FREQ_VIBRATO`` atom (validation-
phase rework), behind ``vibrato_env_pass`` (default OFF): a consecutive-frame
FREQ SET run whose values repeat with a small period becomes ``(period, count,
v0, delta-cycle)`` and is replayed EXACTLY on decode (v0 + cyclic deltas). No
parametric envelope fit, no cycle cap; non-periodic runs are left for FREQ_RUN.
"""

__all__ = ["RawVibratoEnvelopePass", "VIB_MIN_LEN"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    FREQ_VIBRATO_OP,
    SET_OP,
    VIB_MAX_DELTA,
    VIB_MAX_PERIOD,
    VIB_MIN_LEN,
    VIB_SUBREG_COUNT_HI,
    VIB_SUBREG_COUNT_LO,
    VIB_SUBREG_DELTA,
    VIB_SUBREG_PERIOD,
    VIB_SUBREG_V0_HI,
    VIB_SUBREG_V0_LO,
)

_FREQ_REGS = tuple(FREQ_REGS_BY_VOICE)
_MAX_COUNT = 0xFFFF


def _value_period(vals):
    """Smallest period (2..VIB_MAX_PERIOD) the value sequence repeats with, or 0."""
    n = len(vals)
    for p in range(2, min(VIB_MAX_PERIOD, n // 2) + 1):
        if all(vals[i] == vals[i - p] for i in range(p, n)):
            return p
    return 0


def _vib_rows(reg, period, count, v0, deltas, diff, irq):
    v0u = int(v0) & 0xFFFF
    fields = [
        (VIB_SUBREG_PERIOD, int(period)),
        (VIB_SUBREG_COUNT_HI, (int(count) >> 8) & 0xFF),
        (VIB_SUBREG_COUNT_LO, int(count) & 0xFF),
        (VIB_SUBREG_V0_HI, (v0u >> 8) & 0xFF),
        (VIB_SUBREG_V0_LO, v0u & 0xFF),
    ]
    fields += [(VIB_SUBREG_DELTA, int(d) & 0xFF) for d in deltas]
    return [
        {
            "reg": int(reg),
            "val": int(val),
            "diff": int(diff),
            "op": int(FREQ_VIBRATO_OP),
            "subreg": int(subreg),
            "irq": int(irq),
            "description": 0,
        }
        for subreg, val in fields
    ]


class RawVibratoEnvelopePass(MacroPass):
    def apply(self, df, args=None):
        if args is None or not getattr(args, "vibrato_env_pass", False):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq_default = _first_irq(df)
        drop_idx = []
        new_rows = []
        for reg in _FREQ_REGS:
            sets = [
                (int(f_idx[i]), i)
                for i in range(len(df))
                if int(regs[i]) == reg
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ]
            self._collapse_reg(reg, sets, vals, diffs, irq_default, drop_idx, new_rows)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _collapse_reg(self, reg, sets, vals, diffs, irq_default, drop_idx, new_rows):
        k = 0
        while k < len(sets):
            j = k
            while j + 1 < len(sets) and sets[j + 1][0] == sets[j][0] + 1:
                j += 1
            run = sets[k : j + 1]
            if len(run) >= VIB_MIN_LEN:
                self._emit(reg, run, vals, diffs, irq_default, drop_idx, new_rows)
            k = j + 1

    @staticmethod
    def _emit(reg, run, vals, diffs, irq_default, drop_idx, new_rows):
        seq = [int(vals[i]) for _, i in run]
        period = _value_period(seq)
        if period == 0 or len(seq) - 1 > _MAX_COUNT:
            return
        deltas = [seq[(j + 1) % period] - seq[j % period] for j in range(period)]
        if any(abs(d) > VIB_MAX_DELTA for d in deltas) or not any(deltas):
            return
        first_idx = run[0][1]
        diff = int(diffs[first_idx]) if diffs is not None else 0
        atom = _vib_rows(reg, period, len(seq) - 1, seq[0], deltas, diff, irq_default)
        for nr in atom:
            nr["__pos"] = first_idx
        new_rows.extend(atom)
        drop_idx.extend(i for _, i in run)
