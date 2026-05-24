"""Unify isolated FREQ events into one ``FREQ_NUDGE`` op (spec #3 / item 11),
behind the ``freq_nudge_pass`` arg flag (default OFF): every residual FREQ DIFF
becomes a delta-mode nudge (so strict-no-diff can pass) and every truly isolated
FREQ SET becomes an absolute-mode nudge, leaving SET runs for FREQ_RUN."""

__all__ = ["FreqNudgePass"]

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    DIFF_OP,
    FREQ_NUDGE_ISOLATION_GAP,
    FREQ_NUDGE_MODE_ABSOLUTE,
    FREQ_NUDGE_MODE_DELTA,
    FREQ_NUDGE_OP,
    FREQ_NUDGE_SUBREG_HI,
    FREQ_NUDGE_SUBREG_LO,
    FREQ_NUDGE_SUBREG_MODE,
    SET_OP,
)

_FREQ_REGS = frozenset(FREQ_REGS_BY_VOICE)


def _nudge_rows(reg, mode, payload, diff, irq):
    payload &= 0xFFFF
    fields = [
        (FREQ_NUDGE_SUBREG_MODE, int(mode)),
        (FREQ_NUDGE_SUBREG_HI, (payload >> 8) & 0xFF),
        (FREQ_NUDGE_SUBREG_LO, payload & 0xFF),
    ]
    return [
        {
            "reg": int(reg),
            "val": int(val),
            "diff": int(diff),
            "op": int(FREQ_NUDGE_OP),
            "subreg": int(subreg),
            "irq": int(irq),
            "description": 0,
        }
        for subreg, val in fields
    ]


class FreqNudgePass(MacroPass):
    def apply(self, df, args=None):
        if args is None or not getattr(args, "freq_nudge_pass", False):
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
        irq_default = (
            int(df["irq"].iloc[0])
            if "irq" in df.columns and len(df) and df["irq"].notna().any()
            else -1
        )

        catch_all = getattr(args, "lonely_catch_all", False)
        drop_idx = []
        new_rows = []
        for reg in _FREQ_REGS:
            same_reg = [i for i in range(len(df)) if int(regs[i]) == reg]
            reg_frames = [int(f_idx[i]) for i in same_reg]
            for pos, i in enumerate(same_reg):
                op = int(ops[i])
                diff = int(diffs[i]) if diffs is not None else 0
                if op == DIFF_OP and int(subregs[i]) == -1:
                    self._convert(
                        reg,
                        i,
                        FREQ_NUDGE_MODE_DELTA,
                        int(vals[i]),
                        diff,
                        irq_default,
                        drop_idx,
                        new_rows,
                    )
                elif (
                    op == SET_OP
                    and int(subregs[i]) == -1
                    and (catch_all or self._isolated(reg_frames, pos))
                ):
                    self._convert(
                        reg,
                        i,
                        FREQ_NUDGE_MODE_ABSOLUTE,
                        int(vals[i]),
                        diff,
                        irq_default,
                        drop_idx,
                        new_rows,
                    )
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    @staticmethod
    def _isolated(reg_frames, pos):
        fr = reg_frames[pos]
        prev_ok = pos == 0 or fr - reg_frames[pos - 1] >= FREQ_NUDGE_ISOLATION_GAP
        next_ok = (
            pos == len(reg_frames) - 1
            or reg_frames[pos + 1] - fr >= FREQ_NUDGE_ISOLATION_GAP
        )
        return prev_ok and next_ok

    @staticmethod
    def _convert(reg, i, mode, payload, diff, irq, drop_idx, new_rows):
        rows = _nudge_rows(reg, mode, payload, diff, irq)
        for nr in rows:
            nr["__pos"] = i
        new_rows.extend(rows)
        drop_idx.append(i)
