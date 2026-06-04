"""GradientPass: mine staged value-domain automation -- a run of plain SETs on a modulation register
whose values step through a held curve -- and replace each run with GRADIENT atoms carrying
(value, hold_frames) stages that drain one ffilled value per song frame. The Galway gradient envelope
generalised across volume (``modevol_gradient``), envelope (``env_gradient``), filter
(``filter_gradient``) and ctrl (``ctrl_gradient``); an aperiodic CTRL_OSC sibling. Default OFF.
"""

__all__ = ["GradientPass"]

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
)
from preframr_tokens.stfconstants import (
    DELAY_REG,
    FC_LO_REG,
    FILTER_REG,
    FRAME_REG,
    GRADIENT_MAX_DUR,
    GRADIENT_MAX_STAGES,
    GRADIENT_MIN_STAGES,
    GRADIENT_OP,
    GRADIENT_SUBREG_DUR_BASE,
    GRADIENT_SUBREG_END,
    GRADIENT_SUBREG_NSTAGES,
    GRADIENT_SUBREG_VAL_BASE,
    MODE_VOL_REG,
    SET_OP,
)

_GRADIENT_PRIORITY = -6
_FILTER_REGS = (int(FC_LO_REG), int(FC_LO_REG) + 1, int(FILTER_REG))


def _row(reg, subreg, val, irq):
    return make_row(reg, val, op=GRADIENT_OP, subreg=subreg, diff=irq, irq=irq)


class GradientPass(MacroPass):
    """Mine staged held-value automation runs on the ffilled per-frame timeline of a modulation reg and
    replace each with GRADIENT atoms. Runs after the sweep/osc/codebook passes so constant-delta sweeps,
    periodic oscillation and recurring states claim first; gradient takes the aperiodic held remainder.
    Default OFF."""

    GATE_FLAGS = frozenset(
        {"modevol_gradient", "env_gradient", "filter_gradient", "ctrl_gradient"}
    )

    @staticmethod
    def _target_regs(args):
        """Reg set to mine, unioned from the per-domain flags (empty if none enabled)."""
        target = []
        if getattr(args, "modevol_gradient", False):
            target.append(int(MODE_VOL_REG))
        if getattr(args, "env_gradient", False):
            target.extend(int(r) for r in AD_REGS_BY_VOICE)
            target.extend(int(r) for r in SR_REGS_BY_VOICE)
        if getattr(args, "filter_gradient", False):
            target.extend(_FILTER_REGS)
        if getattr(args, "ctrl_gradient", False):
            target.extend(int(r) for r in CTRL_REGS_BY_VOICE)
        return tuple(target)

    def apply(self, df, args=None):
        """One Claim per gradient run, arbitrated with validate=True: a value drains at the frame TICK,
        so a later same-reg atom in that frame would clobber it -- validate drops only the offending run
        (kept literal)."""
        if args is None:
            return df
        target = self._target_regs(args)
        if not target or df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        writes = self._collect_writes(df, target)
        claims = []
        for reg in target:
            claims.extend(self._claim_reg(reg, writes[reg], irq))
        if not claims:
            return df
        return arbitrate(df, claims, validate=True)

    @staticmethod
    def _collect_writes(df, target_regs):
        """Per target reg, ordered (real_frame, row_idx, val) for plain SETs. real_frame counts a
        FRAME_REG as 1 frame and a DELAY_REG as its ``val`` frames -- the decode-frame index the staged
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
        """One Claim per maximal run of >= GRADIENT_MIN_STAGES consecutive writes whose inter-write gap
        is <= GRADIENT_MAX_DUR (a held automation curve; a larger gap ends the curve).
        """
        claims = []
        for i, j in cls._runs(writes):
            claims.append(cls._emit_run(reg, writes, i, j, irq))
        return claims

    @staticmethod
    def _runs(writes):
        """Maximal index ranges [i, j] into ``writes`` (sorted (real_frame, row, val)) with every
        consecutive gap <= GRADIENT_MAX_DUR and >= GRADIENT_MIN_STAGES writes. Returns (i, j).
        """
        runs = []
        n = len(writes)
        i = 0
        while i < n:
            j = i
            while (
                j + 1 < n
                and int(writes[j + 1][0]) - int(writes[j][0]) <= GRADIENT_MAX_DUR
            ):
                j += 1
            if j - i + 1 >= GRADIENT_MIN_STAGES:
                runs.append((i, j))
            i = j + 1
        return runs

    @classmethod
    def _emit_run(cls, reg, writes, i, j, irq):
        """One Claim tiling the run into <= GRADIENT_MAX_STAGES-stage atoms, each anchored on its first
        written row. Stage k holds ``writes[k]`` value for ``frame[k+1] - frame[k]`` frames (the run's
        final write holds 1 frame -- its value persists by ffill after). Chunk boundaries fall on written
        frames so the next atom re-anchors on a real row."""
        new_rows = []
        a = i
        while a <= j:
            b = min(a + GRADIENT_MAX_STAGES - 1, j)
            chunk_pos = int(writes[a][1])
            stages = []
            for k in range(a, b + 1):
                val = int(writes[k][2])
                if k < j:
                    dur = int(writes[k + 1][0]) - int(writes[k][0])
                else:
                    dur = 1
                stages.append((val, dur))
            for r in cls._grad_rows(reg, stages, irq):
                r["__pos"] = chunk_pos
                new_rows.append(r)
            a = b + 1
        drop_idx = tuple(int(writes[k][1]) for k in range(i, j + 1))
        return Claim(
            writes=drop_idx,
            tokens=new_rows,
            priority=_GRADIENT_PRIORITY,
            label="gradient",
        )

    @staticmethod
    def _grad_rows(reg, stages, irq):
        """The atom rows: NSTAGES (resets pending), each stage's VAL_BASE+k / DUR_BASE+k, then END
        (terminal, emitted last so the decoder expands on it)."""
        rows = [_row(reg, GRADIENT_SUBREG_NSTAGES, len(stages), irq)]
        for k, (val, dur) in enumerate(stages):
            rows.append(_row(reg, GRADIENT_SUBREG_VAL_BASE + k, val & 0xFF, irq))
            rows.append(_row(reg, GRADIENT_SUBREG_DUR_BASE + k, dur, irq))
        rows.append(_row(reg, GRADIENT_SUBREG_END, 0, irq))
        return rows
