"""Detect per-register arithmetic-progression runs and emit per-reg SLOPE atoms."""

__all__ = ["SlopePass", "quantise_slope_runtime"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.stfconstants import (
    MAX_REG,
    SET_OP,
    SLOPE_EXCLUDED_REGS,
    SLOPE_MAX_RUNTIME,
    SLOPE_REG_TERMINAL_GRID,
    SLOPE_REG_TO_OP,
    SLOPE_SUBREG_RUNTIME,
    SLOPE_SUBREG_TERMINAL_HI,
    SLOPE_SUBREG_TERMINAL_LO,
)

_RUNTIME_BUCKETS = (32, 64, 128, 256)


def quantise_slope_runtime(n):
    if n <= 0:
        return None
    if n <= 16:
        return int(n)
    if n <= SLOPE_MAX_RUNTIME:
        for b in _RUNTIME_BUCKETS:
            if n <= b:
                prev = b // 2
                return int(b) if (n - prev) >= (b - n) else int(prev)
        return int(SLOPE_MAX_RUNTIME)
    return None


SLOPE_MIN_RUN_LEN = 5


def _quantise_terminal(reg, val):
    grid = int(SLOPE_REG_TERMINAL_GRID.get(int(reg), 1))
    if grid <= 1:
        return int(val)
    v = int(val)
    if v >= 0:
        return ((v + grid // 2) // grid) * grid
    return -(((-v) + grid // 2) // grid) * grid


def _detect_runs(values):
    n = len(values)
    if n < SLOPE_MIN_RUN_LEN:
        return []
    runs = []
    i = 0
    while i <= n - SLOPE_MIN_RUN_LEN:
        s = values[i + 1] - values[i]
        j = i + 2
        while j < n:
            expected = values[i] + s * (j - i)
            if abs(int(values[j]) - int(expected)) <= 1:
                j += 1
                continue
            break
        run_len = j - i
        if run_len >= SLOPE_MIN_RUN_LEN:
            runs.append((i, run_len, int(s)))
            i = j - 1
        else:
            i += 1
    return runs


def _split_runtime(n):
    if n <= SLOPE_MAX_RUNTIME:
        return [n]
    k = (n + SLOPE_MAX_RUNTIME - 1) // SLOPE_MAX_RUNTIME
    base = n // k
    rem = n - base * k
    out = []
    for i in range(k):
        out.append(base + (1 if i < rem else 0))
    return out


def _slope_rows(reg, terminal, runtime, diff_default, irq_default):
    op = int(SLOPE_REG_TO_OP[int(reg)])
    terminal_q = _quantise_terminal(reg, terminal)
    terminal_u = int(terminal_q) & 0xFFFF
    return [
        {
            "reg": int(reg),
            "val": (terminal_u >> 8) & 0xFF,
            "diff": int(diff_default),
            "op": op,
            "subreg": int(SLOPE_SUBREG_TERMINAL_HI),
            "irq": int(irq_default),
            "description": 0,
        },
        {
            "reg": int(reg),
            "val": terminal_u & 0xFF,
            "diff": int(diff_default),
            "op": op,
            "subreg": int(SLOPE_SUBREG_TERMINAL_LO),
            "irq": int(irq_default),
            "description": 0,
        },
        {
            "reg": int(reg),
            "val": int(runtime),
            "diff": int(diff_default),
            "op": op,
            "subreg": int(SLOPE_SUBREG_RUNTIME),
            "irq": int(irq_default),
            "description": 0,
        },
    ]


class SlopePass(MacroPass):
    def apply(self, df, args=None):
        if args is not None and not getattr(args, "slope_pass", True):
            return df
        if df is None or len(df) == 0:
            return df
        orig_df = df
        df = df.reset_index(drop=True).copy()
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        df = _ensure_subreg(df)
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        irq_default = _first_irq(df)

        drop_idx = []
        new_rows = []
        for reg in range(MAX_REG + 1):
            if reg in SLOPE_EXCLUDED_REGS:
                continue
            if reg not in SLOPE_REG_TO_OP:
                continue
            row_mask = (regs == reg) & (ops == SET_OP) & (subregs == -1)
            indices = [int(i) for i in range(len(df)) if row_mask[i]]
            if len(indices) < 3:
                continue
            row_frames = [int(f_idx[i]) for i in indices]
            row_vals = [int(vals[i]) for i in indices]
            row_diffs = (
                [int(diffs[i]) for i in indices]
                if diffs is not None
                else [0] * len(indices)
            )
            k = 0
            while k < len(indices):
                j = k + 1
                while j < len(indices) and row_frames[j] == row_frames[j - 1] + 1:
                    j += 1
                seg_lo, seg_hi = k, j
                if seg_hi - seg_lo >= 3:
                    seg_vals = row_vals[seg_lo:seg_hi]
                    runs = _detect_runs(seg_vals)
                    for ofs, run_len, _slope in runs:
                        start = seg_lo + ofs
                        n = run_len - 1
                        chunks = _split_runtime(n)
                        cur_frame_in_run = 0
                        for chunk_n in chunks:
                            q = quantise_slope_runtime(chunk_n)
                            if q is None:
                                q = SLOPE_MAX_RUNTIME
                            terminal_idx = cur_frame_in_run + chunk_n
                            terminal_val = row_vals[start + terminal_idx]
                            insert_pos = int(indices[start + 1 + cur_frame_in_run])
                            chunk_rows = _slope_rows(
                                reg,
                                terminal_val,
                                q,
                                row_diffs[start + 1 + cur_frame_in_run],
                                irq_default,
                            )
                            for nr in chunk_rows:
                                nr["__pos"] = insert_pos
                            new_rows.extend(chunk_rows)
                            cur_frame_in_run += chunk_n
                        for off in range(1, run_len):
                            drop_idx.append(int(indices[start + off]))
                k = j
        if not new_rows and not drop_idx:
            return orig_df
        return _splice_rows(df, drop_idx, new_rows)
