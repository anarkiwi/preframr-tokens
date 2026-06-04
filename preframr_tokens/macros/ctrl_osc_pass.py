"""CtrlOscPass: mine per-frame ctrl-register oscillation -- a run of song frames whose ffilled ctrl
value cycles with period 2 <= P <= CTRL_OSC_MAX_PERIOD -- and replace each with one register-exact
CTRL_OSC atom (PERIOD, the P cycle bytes, LEN) that drains ``cycle[k % P]`` per frame. A SWEEP twin on
the ctrl reg mined on the FFILLED timeline (DedupSetPass strips held repeats), NOT note-aligned (the
gate bit IS the oscillating value). Default OFF (``ctrl_osc``); decode in ``decoders.py``.
"""

__all__ = ["CtrlOscPass"]

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import CTRL_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    CTRL_OSC_MAX_PERIOD,
    CTRL_OSC_MAX_SPAN,
    CTRL_OSC_MIN_LEN,
    CTRL_OSC_OP,
    CTRL_OSC_SUBREG_LEN,
    CTRL_OSC_SUBREG_PERIOD,
    CTRL_OSC_SUBREG_STATE_BASE,
    DELAY_REG,
    FRAME_REG,
    SET_OP,
)

_CTRL_OSC_PRIORITY = -5


def _row(reg, subreg, val, irq):
    return make_row(reg, val, op=CTRL_OSC_OP, subreg=subreg, diff=irq, irq=irq)


class CtrlOscPass(MacroPass):
    """Mine per-frame ctrl oscillation runs on the ffilled timeline and replace each with a CTRL_OSC
    atom, consuming the per-frame ctrl writes. Default OFF (``ctrl_osc``)."""

    GATE_FLAGS = frozenset({"ctrl_osc"})

    def apply(self, df, args=None):
        """One Claim per oscillation run, arbitrated with validate=True: a ctrl value drains at the
        frame TICK, so a later same-reg atom in that frame would clobber it -- validate drops only the
        offending run (kept literal)."""
        if args is None or not getattr(args, "ctrl_osc", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        writes = self._collect_writes(df, CTRL_REGS_BY_VOICE)
        claims = []
        for reg in CTRL_REGS_BY_VOICE:
            claims.extend(self._claim_reg(reg, writes[reg], irq))
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _collect_writes(df, target_regs):
        """Per target ctrl reg, ordered (real_frame, row_idx, val) for plain SETs. real_frame counts a
        FRAME_REG as 1 frame and a DELAY_REG as its ``val`` frames -- the decode-frame index the per-frame
        drain unrolls against."""
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy()
        vals = df["val"].to_numpy()
        out = {int(r): [] for r in target_regs}
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
        return out

    @classmethod
    def _claim_reg(cls, reg, writes, irq):
        """One Claim per periodic run on ``reg``'s ffilled per-frame timeline. Each run is mined from a
        starting WRITE frame (so the atom anchors on a real row) and consumes every ctrl write whose
        frame falls in the run span; ``timeline[k]`` is the ffilled value at real-frame ``f0 + k``.
        """
        if len(writes) < CTRL_OSC_MIN_LEN:
            return []
        f0 = int(writes[0][0])
        f_last = int(writes[-1][0])
        length = f_last - f0 + 1
        timeline = [0] * length
        anchor_row = {}
        wi = 0
        cur = int(writes[0][2])
        for k in range(length):
            frame = f0 + k
            while wi < len(writes) and int(writes[wi][0]) == frame:
                cur = int(writes[wi][2])
                anchor_row.setdefault(frame, int(writes[wi][1]))
                wi += 1
            timeline[k] = cur
        claims = []
        for pos, end, period in cls._runs(timeline, anchor_row, f0):
            claims.append(
                cls._emit_run(
                    reg, timeline, anchor_row, writes, f0, pos, end, period, irq
                )
            )
        return claims

    @staticmethod
    def _runs(timeline, anchor_row, f0):
        """Maximal periodic runs on the dense per-frame ``timeline``: positions whose value cycles with
        the SMALLEST period ``2 <= P <= CTRL_OSC_MAX_PERIOD`` (``V[pos+k] == cycle[k % P]``), spanning
        >= max(CTRL_OSC_MIN_LEN, 2*P) frames with >= 2 distinct cycle bytes (period 1 is a held state,
        left to the ctrl codebook). A run must START on a written frame so its atom anchors on a real
        row. Returns (pos, end, period)."""
        n = len(timeline)
        runs = []
        pos = 0
        while pos < n - 1:
            if (f0 + pos) not in anchor_row:
                pos += 1
                continue
            best = None
            for period in range(2, CTRL_OSC_MAX_PERIOD + 1):
                if pos + period > n:
                    break
                cycle = timeline[pos : pos + period]
                if len(set(cycle)) < 2:
                    continue
                end = pos + period
                while end < n and timeline[end] == cycle[(end - pos) % period]:
                    end += 1
                run_len = end - pos
                if run_len >= 2 * period and run_len >= CTRL_OSC_MIN_LEN:
                    best = (end - 1, period)
                    break
            if best is not None:
                end, period = best
                runs.append((pos, end, period))
                pos = end + 1
            else:
                pos += 1
        return runs

    @classmethod
    def _emit_run(cls, reg, timeline, anchor_row, writes, f0, pos, end, period, irq):
        """One Claim of one or more CTRL_OSC atoms tiling the run in <= CTRL_OSC_MAX_SPAN-frame chunks,
        each boundary on a WRITTEN frame (a >=2-state cycle changes within every <= P frames, so write
        frames are <= P apart) so each atom anchors on a real row. Per-chunk re-anchoring keeps every
        consolidated empty-frame DELAY short, the SweepPass invariant a single long atom violates; the
        cycle is recomputed per chunk-start so its phase is 0 regardless of the boundary.
        """
        run_start = f0 + pos
        run_end = f0 + end
        write_frames = sorted(fr for fr in anchor_row if run_start <= fr <= run_end)
        starts = cls._chunk_starts(write_frames)
        run_cycle = [int(timeline[pos + m]) for m in range(period)]
        new_rows = []
        for ci, cs in enumerate(starts):
            ce = (starts[ci + 1] - 1) if ci + 1 < len(starts) else run_end
            phase = (cs - f0 - pos) % period
            cycle = [run_cycle[(phase + m) % period] for m in range(period)]
            chunk_pos = anchor_row[cs]
            for r in cls._osc_rows(reg, cycle, period, ce - cs + 1, irq):
                r["__pos"] = chunk_pos
                new_rows.append(r)
        drop_idx = tuple(int(i) for (rf, i, _v) in writes if pos <= int(rf) - f0 <= end)
        return Claim(
            writes=drop_idx,
            tokens=new_rows,
            priority=_CTRL_OSC_PRIORITY,
            label="ctrl_osc",
        )

    @staticmethod
    def _chunk_starts(write_frames):
        """Chunk-start frames tiling ``write_frames``' span: from each start, advance to the FARTHEST
        write frame still within CTRL_OSC_MAX_SPAN, so each chunk [start, next_start-1] spans <=
        CTRL_OSC_MAX_SPAN frames and begins on a written frame (writes are <= P < MAX_SPAN apart, so a
        boundary always exists)."""
        starts = [write_frames[0]]
        within = write_frames[0]
        for fr in write_frames[1:]:
            if fr - starts[-1] <= CTRL_OSC_MAX_SPAN:
                within = fr
            else:
                starts.append(within)
                within = fr
        return starts

    @staticmethod
    def _osc_rows(reg, cycle, period, length, irq):
        """The atom rows: PERIOD (resets pending), the P cycle bytes at STATE_BASE+m, then LEN (terminal,
        emitted last so the decoder expands on it)."""
        rows = [_row(reg, CTRL_OSC_SUBREG_PERIOD, period, irq)]
        for m in range(period):
            rows.append(_row(reg, CTRL_OSC_SUBREG_STATE_BASE + m, cycle[m] & 0xFF, irq))
        rows.append(_row(reg, CTRL_OSC_SUBREG_LEN, length, irq))
        return rows
