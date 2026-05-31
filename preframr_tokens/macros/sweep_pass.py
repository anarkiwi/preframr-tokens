"""SweepPass: freq-domain SWEEP primitive (design/sid_driver_ornament_reference.md SoundMonitor/
skydive). A wide pitch sweep is linear in RAW freq (constant -Δ/frame) so it accelerates in semitones
and the skeleton's semitone SLIDE misses it -> RESID; this proposer mines constant-raw-freq-delta runs
skeleton can't fit and replaces each with one byte-exact SWEEP atom (start, delta, length), consuming
the ramp's freq writes. Runs after StampPass, before SkeletonPass; opt-in (``sweep_pass``), default OFF.
"""

__all__ = ["SweepPass"]

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    MacroPass,
)
from preframr_tokens.macros.skeleton_pass import (
    fit_descriptor,
    fn_to_note_resid,
    is_fast_melodic_run,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FRAME_REG,
    FREQ_TRAJ_REGS,
    ORN_TYPE_RESID,
    SET_OP,
    SWEEP_MAX_SPAN,
    SWEEP_MIN_LEN,
    SWEEP_OP,
    SWEEP_SUBREG_DELTA_HI,
    SWEEP_SUBREG_DELTA_LO,
    SWEEP_SUBREG_LEN,
    SWEEP_SUBREG_PERIOD,
    SWEEP_SUBREG_START_HI,
    SWEEP_SUBREG_START_LO,
)

_SWEEP_PRIORITY = -5


def _row(reg, subreg, val, irq):
    return {
        "reg": int(reg),
        "val": int(val),
        "diff": int(irq),
        "op": int(SWEEP_OP),
        "subreg": int(subreg),
        "irq": int(irq),
        "description": 0,
    }


class SweepPass(MacroPass):
    """Mine constant-raw-freq-delta ramps and replace each with a byte-exact SWEEP atom, consuming
    the per-frame freq writes. Default OFF."""

    GATE_FLAGS = frozenset({"sweep_pass", "sweep_loop"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "sweep_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        sweep_loop = bool(getattr(args, "sweep_loop", False))
        freq_sets, gate_on = self._freq_sets(df)
        drop_idx, new_rows = [], []
        for reg in FREQ_TRAJ_REGS:
            self._claim_reg(
                int(reg),
                freq_sets[int(reg)],
                gate_on[int(reg)],
                irq,
                drop_idx,
                new_rows,
                sweep_loop,
            )
        if not new_rows:
            return df
        return arbitrate(
            df,
            [
                Claim(
                    writes=tuple(drop_idx),
                    tokens=new_rows,
                    priority=_SWEEP_PRIORITY,
                    label="sweep",
                )
            ],
        )

    @staticmethod
    def _freq_sets(df):
        """Per voice-freq reg, ordered (real_frame, row_idx, val) for plain freq SETs, plus the set
        of real-frames where that voice's gate (ctrl reg+4 bit0) rises. real_frame counts a FRAME_REG
        as 1 frame and a DELAY_REG as its ``val`` frames, so a sweep run requires writes exactly one
        REAL frame apart -- the decoder drains one value per unrolled frame, so a DELAY (held frame)
        must break a run; a gate-on retrigger is a note boundary that must break it too.
        """
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        ctrl_to_freq = {int(r) + 4: int(r) for r in FREQ_TRAJ_REGS}
        out = {int(r): [] for r in FREQ_TRAJ_REGS}
        gate_on = {int(r): set() for r in FREQ_TRAJ_REGS}
        gate_state = {int(r) + 4: 0 for r in FREQ_TRAJ_REGS}
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
            elif reg in ctrl_to_freq:
                g = int(vals[i]) & 1
                if g and not gate_state[reg]:
                    gate_on[ctrl_to_freq[reg]].add(rf)
                gate_state[reg] = g
        return out, gate_on

    @classmethod
    def _claim_reg(cls, reg, sets, gate_on, irq, drop_idx, new_rows, sweep_loop=False):
        claimed = set()
        if sweep_loop:
            for i, j, delta, period in cls._loop_runs(sets, gate_on):
                if not cls._skeleton_resids(sets, i, j):
                    continue
                cls._emit_run(reg, sets, i, j, delta, period, irq, drop_idx, new_rows)
                claimed.update(range(i, j + 1))
        for i, j, delta in cls._runs(sets, gate_on):
            if claimed.intersection(range(i, j + 1)):
                continue
            if not cls._skeleton_resids(sets, i, j):
                continue
            cls._emit_run(reg, sets, i, j, delta, 0, irq, drop_idx, new_rows)

    @classmethod
    def _emit_run(cls, reg, sets, i, j, delta, period, irq, drop_idx, new_rows):
        """Emit a run as one or more SWEEP atoms each spanning <= SWEEP_MAX_SPAN frames, anchored at
        its own first frame. A single long atom leaves a > 16-frame DELAY that ``_cap_delay`` coarsens
        (dropping frames while LEN stays), so the per-frame replay must re-anchor before the cap bites;
        loop chunks split on whole periods so each chunk's start value is the program's start phase.
        """
        length = j - i + 1
        span = period * (SWEEP_MAX_SPAN // period) if period else SWEEP_MAX_SPAN
        off = 0
        while off < length:
            clen = min(span, length - off)
            start = int(sets[i + off][2])
            chunk_pos = int(sets[i + off][1])
            for r in cls._sweep_rows(reg, start, delta, clen, period, irq):
                r["__pos"] = chunk_pos
                new_rows.append(r)
            off += clen
        drop_idx.extend(int(sets[k][1]) for k in range(i, j + 1))

    @staticmethod
    def _skeleton_resids(sets, i, j):
        """True iff skeleton would dump this run to RESID and could NOT rescue it: its content-tier
        ornament fitter rejects it (fit_descriptor -> RESID) AND it is not a fast-melodic-run (which
        skeleton splits into per-semitone notes). That is exactly the wide/fast raw-freq skydive its
        semitone machinery can't represent; ordinary slides and slow runs are left to skeleton.
        """
        first = fn_to_note_resid(int(sets[i][2]))
        if first is None:
            return True
        base = first[0]
        offs = []
        for k in range(i + 1, j + 1):
            res = fn_to_note_resid(int(sets[k][2]))
            if res is None:
                return True
            offs.append(res[0] - base)
        orn_type, _params = fit_descriptor(
            base, [int(sets[k][2]) for k in range(i + 1, j + 1)]
        )
        if orn_type != ORN_TYPE_RESID:
            return False
        return not is_fast_melodic_run(offs)

    @staticmethod
    def _runs(sets, gate_on):
        """Maximal runs of >= SWEEP_MIN_LEN freq SETs one REAL frame apart with a constant non-zero
        delta and NO gate-on retrigger between steps (a retrigger is a note boundary). Returns
        (start_idx, end_idx, delta) into ``sets`` (sorted (real_frame,row,val))."""

        def step_ok(k, delta):
            return (
                int(sets[k + 1][0]) == int(sets[k][0]) + 1
                and int(sets[k + 1][2]) - int(sets[k][2]) == delta
                and int(sets[k + 1][0]) not in gate_on
            )

        runs = []
        n = len(sets)
        i = 0
        while i < n - 1:
            delta = int(sets[i + 1][2]) - int(sets[i][2])
            if delta == 0 or not step_ok(i, delta):
                i += 1
                continue
            j = i + 1
            while j < n - 1 and step_ok(j, delta):
                j += 1
            if j - i + 1 >= SWEEP_MIN_LEN:
                runs.append((i, j, delta))
                i = j + 1
            else:
                i += 1
        return runs

    @staticmethod
    def _loop_runs(sets, gate_on):
        """Maximal periodic-sawtooth runs: freq one REAL frame apart with ``freq[k] = start +
        ((k-i) % P) * delta`` for a period ``2 <= P``, spanning >= 2 full periods and no gate-on
        retrigger mid-run. The looping freq-domain arp (SoundMonitor: constant -delta/frame, reset
        every P) the linear run-finder shatters at each reset jump. Returns (i, j, delta, period).
        """
        n = len(sets)
        runs = []
        i = 0
        while i < n - 1:
            start = int(sets[i][2])
            delta = int(sets[i + 1][2]) - start
            period = None
            j = i
            while j + 1 < n:
                nxt = sets[j + 1]
                if int(nxt[0]) != int(sets[j][0]) + 1 or int(nxt[0]) in gate_on:
                    break
                pos = j + 1 - i
                if period is None:
                    if int(nxt[2]) == start:
                        period = pos
                    elif int(nxt[2]) - int(sets[j][2]) != delta:
                        break
                    else:
                        j += 1
                        continue
                if (int(nxt[2]) & 0xFFFF) != (
                    (start + (pos % period) * delta) & 0xFFFF
                ):
                    break
                j += 1
            if (
                period is not None
                and 2 <= period <= SWEEP_MAX_SPAN
                and (j - i + 1) >= 2 * period
            ):
                runs.append((i, j, delta, period))
                i = j + 1
            else:
                i += 1
        return runs

    @staticmethod
    def _sweep_rows(reg, start, delta, length, period, irq):
        d = delta & 0xFFFF
        rows = [
            _row(reg, SWEEP_SUBREG_START_HI, (start >> 8) & 0xFF, irq),
            _row(reg, SWEEP_SUBREG_START_LO, start & 0xFF, irq),
            _row(reg, SWEEP_SUBREG_DELTA_HI, (d >> 8) & 0xFF, irq),
            _row(reg, SWEEP_SUBREG_DELTA_LO, d & 0xFF, irq),
        ]
        if period:
            rows.append(_row(reg, SWEEP_SUBREG_PERIOD, period & 0xFFFF, irq))
        rows.append(_row(reg, SWEEP_SUBREG_LEN, length, irq))
        return rows
