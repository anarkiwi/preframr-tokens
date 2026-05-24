"""Collapse pure-SET FREQ runs into one ``FREQ_RUN`` atom (spec #4), behind the
``freq_run_pass`` arg flag (default OFF): a maximal run of consecutive-frame
FREQ SETs (length >= ``FREQ_RUN_MIN_LEN``, not an arithmetic slope, which ran
earlier) becomes a count + value-list atom replayed exactly on decode."""

__all__ = ["FreqRunPass"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    FREQ_RUN_MAX_LEN,
    FREQ_RUN_MIN_LEN,
    FREQ_RUN_OP,
    FREQ_RUN_SUBREG_COUNT,
    FREQ_RUN_SUBREG_HI,
    FREQ_RUN_SUBREG_LO,
    SET_OP,
)

_FREQ_REGS = frozenset(FREQ_REGS_BY_VOICE)


def _run_rows(reg, values, diff, irq):
    rows = [(FREQ_RUN_SUBREG_COUNT, len(values))]
    for v in values:
        rows.append((FREQ_RUN_SUBREG_HI, (int(v) >> 8) & 0xFF))
        rows.append((FREQ_RUN_SUBREG_LO, int(v) & 0xFF))
    return [
        {
            "reg": int(reg),
            "val": int(val),
            "diff": int(diff),
            "op": int(FREQ_RUN_OP),
            "subreg": int(subreg),
            "irq": int(irq),
            "description": 0,
        }
        for subreg, val in rows
    ]


class FreqRunPass(MacroPass):
    def apply(self, df, args=None):
        if args is None or not getattr(args, "freq_run_pass", False):
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
            self._collapse(reg, sets, vals, diffs, irq_default, drop_idx, new_rows)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _collapse(self, reg, sets, vals, diffs, irq_default, drop_idx, new_rows):
        k = 0
        while k < len(sets):
            j = k
            while j + 1 < len(sets) and sets[j + 1][0] == sets[j][0] + 1:
                j += 1
            run = sets[k : j + 1]
            if len(run) >= FREQ_RUN_MIN_LEN:
                self._emit(reg, run, vals, diffs, irq_default, drop_idx, new_rows)
            k = j + 1

    @staticmethod
    def _emit(reg, run, vals, diffs, irq_default, drop_idx, new_rows):
        for chunk_start in range(0, len(run), FREQ_RUN_MAX_LEN):
            chunk = run[chunk_start : chunk_start + FREQ_RUN_MAX_LEN]
            if len(chunk) < FREQ_RUN_MIN_LEN:
                break
            first_idx = chunk[0][1]
            diff = int(diffs[first_idx]) if diffs is not None else 0
            values = [int(vals[i]) for _, i in chunk]
            atom = _run_rows(reg, values, diff, irq_default)
            for nr in atom:
                nr["__pos"] = first_idx
            new_rows.extend(atom)
            drop_idx.extend(i for _, i in chunk)
