"""SweepPass (design/sid_driver_ornament_reference.md SoundMonitor/skydive): mine constant-delta ramps
and replace each with one byte-exact SWEEP atom (start, delta, length), consuming the per-frame writes.
Mines freq (skydive pitch sweeps the skeleton's semitone SLIDE can't fit), per-voice PW (``pw_sweep``)
and the global filter cutoff (``filter_sweep``); PW/filter sweeps PERSIST ACROSS NOTES so they are NOT
note-aligned (a gate-on retrigger does not break the run) and skip the skeleton RESID gate. Default OFF.
"""

__all__ = ["SweepPass"]

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.skeleton_pass import (
    fit_descriptor,
    fn_to_note_resid,
    is_fast_melodic_run,
)
from preframr_tokens.macros.state import PWM_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
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
    return make_row(reg, val, op=SWEEP_OP, subreg=subreg, diff=irq, irq=irq)


class SweepPass(MacroPass):
    """Mine constant-delta ramps and replace each with a byte-exact SWEEP atom, consuming the per-frame
    writes. Mines freq (always, when ``sweep_pass``), per-voice PW (``pw_sweep``) and the global filter
    cutoff (``filter_sweep``). Default OFF."""

    GATE_FLAGS = frozenset({"sweep_pass", "sweep_loop", "pw_sweep", "filter_sweep"})

    def apply(self, df, args=None):
        """One Claim per sweep run, arbitrated with validate=True: a sweep value drains at the frame
        TICK, so a later same-reg atom in that frame would clobber it -- validate drops only the
        offending run (kept literal). freq runs are note-aligned + skeleton-gated; PW/filter neither.
        """
        if args is None or not getattr(args, "sweep_pass", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        sweep_loop = bool(getattr(args, "sweep_loop", False))
        pw_regs = (
            tuple(int(r) for r in PWM_REGS_BY_VOICE)
            if getattr(args, "pw_sweep", False)
            else ()
        )
        filter_regs = (int(FC_LO_REG),) if getattr(args, "filter_sweep", False) else ()
        freq_regs = tuple(int(r) for r in FREQ_TRAJ_REGS)
        sets, gate_on = self._collect_sets(df, freq_regs + pw_regs + filter_regs)
        claims = []
        for reg in freq_regs:
            claims.extend(
                self._claim_reg(
                    reg, sets[reg], gate_on[reg], irq, sweep_loop, True, True
                )
            )
        for reg in pw_regs + filter_regs:
            claims.extend(
                self._claim_reg(reg, sets[reg], set(), irq, sweep_loop, False, False)
            )
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _collect_sets(df, target_regs):
        """Per target reg, ordered (real_frame, row_idx, val) for plain SETs, plus the set of real-frames
        where each voice's gate (freq ctrl reg+4 bit0) rises. real_frame counts a FRAME_REG as 1 frame
        and a DELAY_REG as its ``val`` frames, so a sweep run requires writes exactly one REAL frame
        apart -- the decoder drains one value per unrolled frame, so a DELAY (held frame) must break a
        run. Gate tracking is keyed to the freq regs only (note-aligned freq runs); PW/filter ignore it.
        """
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        ctrl_to_freq = {int(r) + 4: int(r) for r in FREQ_TRAJ_REGS}
        out = {int(r): [] for r in target_regs}
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
    def _claim_reg(
        cls, reg, sets, gate_on, irq, sweep_loop, note_aligned, skeleton_gated
    ):
        """Return one Claim per qualifying sweep run on ``reg``. ``note_aligned`` (freq) breaks runs on a
        gate-on retrigger; ``skeleton_gated`` (freq) restricts to runs the skeleton would dump to RESID.
        PW/filter pass both False (persist across notes, no skeleton)."""
        claims = []
        claimed = set()
        if sweep_loop:
            for i, j, delta, period in cls._loop_runs(sets, gate_on, note_aligned):
                if skeleton_gated and not cls._skeleton_resids(sets, i, j):
                    continue
                claims.append(cls._emit_run(reg, sets, i, j, delta, period, irq))
                claimed.update(range(i, j + 1))
        for i, j, delta in cls._runs(sets, gate_on, note_aligned):
            if claimed.intersection(range(i, j + 1)):
                continue
            if skeleton_gated and not cls._skeleton_resids(sets, i, j):
                continue
            claims.append(cls._emit_run(reg, sets, i, j, delta, 0, irq))
        return claims

    @classmethod
    def _emit_run(cls, reg, sets, i, j, delta, period, irq):
        """One Claim for a run, emitted as one or more SWEEP atoms each spanning <= SWEEP_MAX_SPAN
        frames, anchored at its own first frame. A single long atom leaves a > 16-frame DELAY that
        ``_cap_delay`` coarsens (dropping frames while LEN stays), so the per-frame replay must re-anchor
        before the cap bites; loop chunks split on whole periods so each chunk's start value is the
        program's start phase.
        """
        length = j - i + 1
        span = period * (SWEEP_MAX_SPAN // period) if period else SWEEP_MAX_SPAN
        off = 0
        new_rows = []
        while off < length:
            clen = min(span, length - off)
            start = int(sets[i + off][2])
            chunk_pos = int(sets[i + off][1])
            for r in cls._sweep_rows(reg, start, delta, clen, period, irq):
                r["__pos"] = chunk_pos
                new_rows.append(r)
            off += clen
        drop_idx = tuple(int(sets[k][1]) for k in range(i, j + 1))
        return Claim(
            writes=drop_idx, tokens=new_rows, priority=_SWEEP_PRIORITY, label="sweep"
        )

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
    def _runs(sets, gate_on, note_aligned=True):
        """Maximal runs of >= SWEEP_MIN_LEN SETs one REAL frame apart with a constant non-zero delta.
        When ``note_aligned`` (freq) a gate-on retrigger between steps breaks the run (a note boundary);
        PW/filter pass ``note_aligned=False`` (they persist across notes). Returns (start_idx, end_idx,
        delta) into ``sets`` (sorted (real_frame,row,val))."""

        def step_ok(k, delta):
            return (
                int(sets[k + 1][0]) == int(sets[k][0]) + 1
                and int(sets[k + 1][2]) - int(sets[k][2]) == delta
                and not (note_aligned and int(sets[k + 1][0]) in gate_on)
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
    def _loop_runs(sets, gate_on, note_aligned=True):
        """Maximal periodic-sawtooth runs: values one REAL frame apart with ``val[k] = start +
        ((k-i) % P) * delta`` for a period ``2 <= P``, spanning >= 2 full periods. When ``note_aligned``
        (freq) a gate-on retrigger mid-run breaks it; PW/filter pass False (persist across notes). The
        looping freq-domain arp (SoundMonitor: constant -delta/frame, reset every P) the linear
        run-finder shatters at each reset jump. Returns (i, j, delta, period).
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
                if int(nxt[0]) != int(sets[j][0]) + 1 or (
                    note_aligned and int(nxt[0]) in gate_on
                ):
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
