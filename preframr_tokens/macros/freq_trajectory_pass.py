"""Unified FREQ/PW/FC trajectory pass: segment each slope-able register's
per-frame SET trajectory and emit one ``FREQ_TRAJ`` op (45) per segment, in
precedence MONOTONE_RAMP -> OSCILLATE -> RUN; isolated writes fall through."""

__all__ = ["FreqTrajectoryPass", "quantise_slope_runtime"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_OP,
    FT_DELTA_ESCAPE,
    FT_DELTA_MAX,
    FT_MAX_PERIOD,
    FT_PERIODIC_BIT,
    FT_SUBREG_COUNT_HI,
    FT_SUBREG_COUNT_LO,
    FT_SUBREG_DELTA,
    FT_SUBREG_FLAGS,
    FT_SUBREG_PERIOD,
    FT_SUBREG_RUNTIME,
    FT_SUBREG_TERMINAL_HI,
    FT_SUBREG_TERMINAL_LO,
    FT_SUBREG_V0_HI,
    FT_SUBREG_V0_LO,
    FT_SUBTYPE_MONOTONE_RAMP,
    FT_SUBTYPE_OSCILLATE,
    FT_SUBTYPE_RUN,
    OSC_MAX_GAP,
    OSC_MIN_ALTERNATION,
    OSC_MIN_HALFCYCLES,
    SET_OP,
    SLOPE_MAX_RUNTIME,
    SLOPE_REG_TERMINAL_GRID,
    TRAJ_REGS,
)

_RUNTIME_BUCKETS = (32, 64, 128, 256)
SLOPE_MIN_RUN_LEN = 5


def quantise_slope_runtime(n):
    """Snap a ramp runtime to the exact small values or the nearest bucket."""
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
    return [base + (1 if i < rem else 0) for i in range(k)]


def _is_oscillation(wdeltas):
    nz = [d for d in wdeltas if d != 0]
    if len(nz) < 2:
        return False
    changes = sum(1 for a, b in zip(nz, nz[1:]) if (a > 0) != (b > 0))
    return (changes / (len(nz) - 1)) >= OSC_MIN_ALTERNATION and (
        changes >= OSC_MIN_HALFCYCLES
    )


def _periodic_period(deltas):
    n = len(deltas)
    for p in range(2, min(FT_MAX_PERIOD, n // 2) + 1):
        if all(deltas[i] == deltas[i % p] for i in range(n)):
            return p
    return None


def _row(reg, subreg, val, diff, irq):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(diff),
        "op": int(FREQ_TRAJ_OP),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


def _ramp_rows(reg, terminal, runtime, diff, irq):
    tu = int(_quantise_terminal(reg, terminal)) & 0xFFFF
    fields = [
        (FT_SUBREG_FLAGS, FT_SUBTYPE_MONOTONE_RAMP),
        (FT_SUBREG_TERMINAL_HI, (tu >> 8) & 0xFF),
        (FT_SUBREG_TERMINAL_LO, tu & 0xFF),
        (FT_SUBREG_RUNTIME, int(runtime)),
    ]
    return [_row(reg, sr, v, diff, irq) for sr, v in fields]


def _delta_run_rows(reg, subtype, v0, deltas, period, diff, irq):
    v0u = int(v0) & 0xFFFF
    count = len(deltas)
    flags = int(subtype)
    if period is not None:
        flags |= FT_PERIODIC_BIT
    fields = [
        (FT_SUBREG_FLAGS, flags),
        (FT_SUBREG_V0_HI, (v0u >> 8) & 0xFF),
        (FT_SUBREG_V0_LO, v0u & 0xFF),
        (FT_SUBREG_COUNT_HI, (count >> 8) & 0xFF),
        (FT_SUBREG_COUNT_LO, count & 0xFF),
    ]
    if period is not None:
        fields.append((FT_SUBREG_PERIOD, int(period)))
        for d in deltas[:period]:
            fields.append((FT_SUBREG_DELTA, int(d) & 0xFF))
        return [_row(reg, sr, v, diff, irq) for sr, v in fields]
    cur = int(v0)
    for d in deltas:
        cur += int(d)
        if -FT_DELTA_MAX <= d <= FT_DELTA_MAX:
            fields.append((FT_SUBREG_DELTA, int(d) & 0xFF))
        else:
            cu = cur & 0xFFFF
            fields.append((FT_SUBREG_DELTA, FT_DELTA_ESCAPE))
            fields.append((FT_SUBREG_DELTA, (cu >> 8) & 0xFF))
            fields.append((FT_SUBREG_DELTA, cu & 0xFF))
    return [_row(reg, sr, v, diff, irq) for sr, v in fields]


class FreqTrajectoryPass(MacroPass):
    """Replace SlopePass/OscillationEnvelopePass/RawVibratoEnvelopePass/FreqRunPass
    with one trajectory primitive over every slope-able register."""

    GATE_FLAGS = frozenset({"freq_trajectory_pass"})

    def apply(self, df, args=None):
        if args is not None and not getattr(args, "freq_trajectory_pass", True):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        f_idx = _frame_index(df).to_numpy()
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        diffs = df["diff"].to_numpy() if "diff" in df.columns else None
        irq = _first_irq(df)
        last_frame = int(f_idx.max()) if len(f_idx) else 0
        drop_idx = []
        new_rows = []
        for reg in TRAJ_REGS:
            sets = [
                (
                    int(f_idx[i]),
                    int(i),
                    int(vals[i]),
                    int(diffs[i]) if diffs is not None else 0,
                )
                for i in range(len(df))
                if int(regs[i]) == reg
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ]
            if len(sets) < 2:
                continue
            claimed = self._emit_ramps(reg, sets, last_frame, irq, drop_idx, new_rows)
            self._emit_osc_run(reg, sets, claimed, last_frame, irq, drop_idx, new_rows)
        if not new_rows:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    def _emit_ramps(self, reg, sets, last_frame, irq, drop_idx, new_rows):
        claimed = set()
        n = len(sets)
        k = 0
        while k < n:
            j = k + 1
            while j < n and sets[j][0] == sets[j - 1][0] + 1:
                j += 1
            seg = sets[k:j]
            if len(seg) >= 3:
                for ofs, run_len, _ in _detect_runs([s[2] for s in seg]):
                    if seg[ofs + run_len - 1][0] == last_frame:
                        continue
                    for off in range(run_len):
                        claimed.add(k + ofs + off)
                    self._emit_ramp_run(reg, seg, ofs, run_len, irq, drop_idx, new_rows)
            k = j
        return claimed

    @staticmethod
    def _emit_ramp_run(reg, seg, start, run_len, irq, drop_idx, new_rows):
        cur = 0
        for chunk_n in _split_runtime(run_len - 1):
            q = quantise_slope_runtime(chunk_n) or SLOPE_MAX_RUNTIME
            anchor = seg[start + 1 + cur]
            rows = _ramp_rows(reg, seg[start + cur + chunk_n][2], q, anchor[3], irq)
            for nr in rows:
                nr["__pos"] = anchor[1]
            new_rows.extend(rows)
            cur += chunk_n
        for off in range(1, run_len):
            drop_idx.append(seg[start + off][1])

    def _emit_osc_run(self, reg, sets, claimed, last_frame, irq, drop_idx, new_rows):
        n = len(sets)
        k = 0
        while k < n:
            if k in claimed:
                k += 1
                continue
            cluster = [sets[k]]
            j = k + 1
            while (
                j < n
                and j not in claimed
                and sets[j][0] - cluster[-1][0] <= OSC_MAX_GAP
            ):
                cluster.append(sets[j])
                j += 1
            if len(cluster) >= 2 and cluster[-1][0] != last_frame:
                self._emit_cluster(reg, cluster, irq, drop_idx, new_rows)
            k = j
        return None

    @staticmethod
    def _emit_cluster(reg, cluster, irq, drop_idx, new_rows):
        frames = [c[0] for c in cluster]
        cvals = [c[2] for c in cluster]
        dense = []
        prev = cvals[0]
        fi = 0
        for f in range(frames[0], frames[-1] + 1):
            if fi < len(frames) and frames[fi] == f:
                prev = cvals[fi]
                fi += 1
            dense.append(prev)
        deltas = [dense[i + 1] - dense[i] for i in range(len(dense) - 1)]
        wdeltas = [cvals[i + 1] - cvals[i] for i in range(len(cvals) - 1)]
        subtype = FT_SUBTYPE_OSCILLATE if _is_oscillation(wdeltas) else FT_SUBTYPE_RUN
        fits = all(-FT_DELTA_MAX <= d <= FT_DELTA_MAX for d in deltas)
        period = _periodic_period(deltas) if fits else None
        rows = _delta_run_rows(
            reg, subtype, cvals[0], deltas, period, cluster[0][3], irq
        )
        for nr in rows:
            nr["__pos"] = cluster[0][1]
        new_rows.extend(rows)
        drop_idx.extend(c[1] for c in cluster)
