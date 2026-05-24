"""Collapse alternating-sign SLOPE chains into ``OSCILLATE_ENV`` atoms, post
``SlopePass`` + ``PresetPass``: a maximal run of same-reg SLOPE atoms is split
into uniform-runtime sub-runs, and each sub-run whose terminals alternate about
their midline and fit an envelope family becomes one 8-subreg atom, else the raw
slopes are left as a fallback."""

__all__ = ["OscillationEnvelopePass", "OSC_MIN_SLOPES"]

from preframr_tokens.macros import envelope as env
from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _splice_rows,
)
from preframr_tokens.stfconstants import (
    OSC_MAX_NCYCLES,
    OSC_START_DOWN_BIT,
    OSC_SUBREG_AMP_HI,
    OSC_SUBREG_AMP_LO,
    OSC_SUBREG_ANCHOR_HI,
    OSC_SUBREG_ANCHOR_LO,
    OSC_SUBREG_FAMILY,
    OSC_SUBREG_NCYCLES,
    OSC_SUBREG_PARAM,
    OSC_SUBREG_PERIOD,
    OSCILLATE_ENV_OP,
    SLOPE_OPS,
    SLOPE_REG_TO_OP,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
)

OSC_MIN_SLOPES = 3
_SLOPE_OPS = frozenset(SLOPE_OPS)


def _signed16(v):
    v &= 0xFFFF
    return v if v < 0x8000 else v - 0x10000


def _osc_rows(
    reg, anchor, amp, slope_frames, n_slopes, start_down, family, param, diff, irq
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
        (OSC_SUBREG_PERIOD, int(slope_frames) & 0xFF),
        (OSC_SUBREG_NCYCLES, ncycles_byte & 0xFF),
        (OSC_SUBREG_FAMILY, int(family) & 0xFF),
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


class OscillationEnvelopePass(MacroPass):
    def apply(self, df, args=None):
        if args is not None and not getattr(args, "oscillate_env_pass", True):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq_default = _first_irq(df)
        n = len(df)

        per_reg = {reg: [] for reg in SLOPE_REG_TO_OP}
        i = 0
        while i < n:
            reg = int(regs[i])
            if reg not in per_reg:
                i += 1
                continue
            op = int(ops[i])
            if (
                op in _SLOPE_OPS
                and int(subregs[i]) == SLOPE_SUBREG_TERMINAL_HI
                and i + 2 < n
                and int(ops[i + 1]) == op
                and int(ops[i + 2]) == op
                and int(regs[i + 1]) == reg
                and int(regs[i + 2]) == reg
                and int(subregs[i + 1]) == SLOPE_SUBREG_TERMINAL_LO
                and int(subregs[i + 2]) == SLOPE_SUBREG_RUNTIME
            ):
                terminal = _signed16(
                    ((int(vals[i]) & 0xFF) << 8) | (int(vals[i + 1]) & 0xFF)
                )
                runtime = int(vals[i + 2])
                per_reg[reg].append(("slope", (i, i + 1, i + 2), terminal, runtime))
                i += 3
            else:
                per_reg[reg].append(("other",))
                i += 1

        drop_idx = []
        new_rows = []
        for reg, events in per_reg.items():
            slopes = [e for e in events]
            k = 0
            while k < len(slopes):
                if slopes[k][0] != "slope":
                    k += 1
                    continue
                j = k
                while j < len(slopes) and slopes[j][0] == "slope":
                    j += 1
                run = slopes[k:j]
                self._maybe_collapse(reg, run, diffs, irq_default, drop_idx, new_rows)
                k = j

        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _maybe_collapse(self, reg, run, diffs, irq_default, drop_idx, new_rows):
        a = 0
        while a < len(run):
            b = a + 1
            while b < len(run) and run[b][3] == run[a][3]:
                b += 1
            self._collapse_uniform(
                reg, run[a:b], diffs, irq_default, drop_idx, new_rows
            )
            a = b

    def _collapse_uniform(self, reg, run, diffs, irq_default, drop_idx, new_rows):
        m = len(run)
        if m > OSC_MAX_NCYCLES:
            m = OSC_MAX_NCYCLES
            run = run[:m]
        if m < OSC_MIN_SLOPES:
            return
        terminals = [e[2] for e in run]
        runtime = run[0][3]
        if runtime <= 0:
            return
        anchor = round(sum(terminals) / len(terminals))
        signs = [t - anchor for t in terminals]
        if any(s == 0 for s in signs):
            return
        for a, b in zip(signs, signs[1:]):
            if (a > 0) == (b > 0):
                return
        amps = [abs(s) for s in signs]
        base = amps[0]
        if base <= 0:
            return
        norm = [a / base for a in amps]
        family, param, residual = env.fit_family(norm)
        if residual > env.FIT_TOLERANCE:
            return
        start_down = signs[0] < 0
        first_hi_idx = run[0][1][0]
        diff = int(diffs[first_hi_idx]) if diffs is not None else 0
        atom = _osc_rows(
            reg,
            anchor,
            base,
            runtime,
            m,
            start_down,
            family,
            param,
            diff,
            irq_default,
        )
        insert_pos = first_hi_idx
        for nr in atom:
            nr["__pos"] = insert_pos
        new_rows.extend(atom)
        for entry in run:
            drop_idx.extend(int(x) for x in entry[1])
