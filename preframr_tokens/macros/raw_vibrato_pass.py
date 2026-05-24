"""Raw-stream vibrato collapser (TOKEN_IMPROVEMENTS.md item 0 rework), behind
``vibrato_env_pass`` (default OFF): alternating short FREQ SET runs that
SlopePass's >=5-frame gate never makes into SLOPE atoms collapse into step-mode
``OSCILLATE_ENV`` atoms. Each maximal uniform-frame-gap run that alternates
about its midline and fits an envelope family becomes one atom, else kept raw."""

__all__ = ["RawVibratoEnvelopePass", "VIBRATO_MIN_SLOPES"]

from preframr_tokens.macros import envelope as env
from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    OSC_MAX_NCYCLES,
    OSC_START_DOWN_BIT,
    OSC_STEP_MODE_BIT,
    OSC_SUBREG_AMP_HI,
    OSC_SUBREG_AMP_LO,
    OSC_SUBREG_ANCHOR_HI,
    OSC_SUBREG_ANCHOR_LO,
    OSC_SUBREG_FAMILY,
    OSC_SUBREG_NCYCLES,
    OSC_SUBREG_PARAM,
    OSC_SUBREG_PERIOD,
    OSCILLATE_ENV_OP,
    SET_OP,
)

VIBRATO_MIN_SLOPES = 3
_FREQ_REGS = tuple(FREQ_REGS_BY_VOICE)
_MAX_GAP = 0xFF


def _osc_step_rows(
    reg, anchor, amp, period, n_slopes, start_down, family, param, diff, irq
):
    anchor_u = int(anchor) & 0xFFFF
    amp_u = int(amp) & 0xFFFF
    ncycles_byte = (int(n_slopes) & OSC_MAX_NCYCLES) | (
        OSC_START_DOWN_BIT if start_down else 0
    )
    fields = [
        (OSC_SUBREG_ANCHOR_HI, (anchor_u >> 8) & 0xFF),
        (OSC_SUBREG_ANCHOR_LO, anchor_u & 0xFF),
        (OSC_SUBREG_AMP_HI, (amp_u >> 8) & 0xFF),
        (OSC_SUBREG_AMP_LO, amp_u & 0xFF),
        (OSC_SUBREG_PERIOD, int(period) & 0xFF),
        (OSC_SUBREG_NCYCLES, ncycles_byte & 0xFF),
        (OSC_SUBREG_FAMILY, (int(family) | OSC_STEP_MODE_BIT) & 0xFF),
        (OSC_SUBREG_PARAM, int(param) & 0xFF),
    ]
    return [
        {
            "reg": int(reg),
            "val": int(val),
            "diff": int(diff),
            "op": int(OSCILLATE_ENV_OP),
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
        irq_default = (
            int(df["irq"].iloc[0])
            if "irq" in df.columns and len(df) and df["irq"].notna().any()
            else -1
        )
        drop_idx = []
        new_rows = []
        for reg in _FREQ_REGS:
            events = [
                (i, int(f_idx[i]), int(vals[i]))
                for i in range(len(df))
                if int(regs[i]) == reg
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ]
            self._collapse_reg(reg, events, diffs, irq_default, drop_idx, new_rows)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _collapse_reg(self, reg, events, diffs, irq_default, drop_idx, new_rows):
        k = 0
        while k < len(events) - 1:
            gap = events[k + 1][1] - events[k][1]
            if gap < 1 or gap > _MAX_GAP:
                k += 1
                continue
            j = k
            while j + 1 < len(events) and events[j + 1][1] - events[j][1] == gap:
                j += 1
            self._collapse_run(
                reg, events[k : j + 1], gap, diffs, irq_default, drop_idx, new_rows
            )
            k = j + 1

    def _collapse_run(self, reg, run, gap, diffs, irq_default, drop_idx, new_rows):
        m = len(run)
        if m > OSC_MAX_NCYCLES:
            m = OSC_MAX_NCYCLES
            run = run[:m]
        if m < VIBRATO_MIN_SLOPES:
            return
        terminals = [e[2] for e in run]
        anchor = round(sum(terminals) / len(terminals))
        signs = [t - anchor for t in terminals]
        if any(s == 0 for s in signs):
            return
        for a, b in zip(signs, signs[1:]):
            if (a > 0) == (b > 0):
                return
        base = abs(signs[0])
        if base <= 0:
            return
        family, param, residual = env.fit_family([abs(s) / base for s in signs])
        if residual > env.FIT_TOLERANCE:
            return
        first_idx = run[0][0]
        diff = int(diffs[first_idx]) if diffs is not None else 0
        atom = _osc_step_rows(
            reg, anchor, base, gap, m, signs[0] < 0, family, param, diff, irq_default
        )
        for nr in atom:
            nr["__pos"] = first_idx
        new_rows.extend(atom)
        for e in run:
            drop_idx.append(int(e[0]))
