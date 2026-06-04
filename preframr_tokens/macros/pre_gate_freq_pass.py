"""PreGateFreqPass: a freq written BEFORE a voice's first gate-on is inaudible (the un-gated voice emits
nothing; proven in preframr-audio ``test_freq_write_audibility``). If the first gated note sets its own
freq, DROP the pre-gate freq; else RELOCATE it into the gate-on frame for the onset macros. AUDIO-exact,
not register-state-exact; default OFF (``pre_gate_freq``), in ``parse_audit._LOSSY_RESETS`` so the
auditor re-baselines after it."""

__all__ = ["PreGateFreqPass"]

from preframr_tokens.macros.passes_base import (
    _ensure_subreg,
    _first_irq,
    _frame_index,
    make_row,
    MacroPass,
)
from preframr_tokens.macros.state import (
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    VOICES,
)
from preframr_tokens.stfconstants import SET_OP


class PreGateFreqPass(MacroPass):
    """Drop or relocate a voice's pre-first-gate-on freq write (inaudible while un-gated). Default OFF
    (``pre_gate_freq``); audio-exact, not register-state-exact."""

    GATE_FLAGS = frozenset({"pre_gate_freq"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "pre_gate_freq", False):
            return df
        if df is None or len(df) == 0:
            return df
        df = _ensure_subreg(df.reset_index(drop=True).copy())
        if "op" not in df.columns:
            df["op"] = int(SET_OP)
        irq = _first_irq(df)
        regs = df["reg"].to_numpy()
        vals = df["val"].to_numpy()
        ops = df["op"].to_numpy()
        subs = df["subreg"].to_numpy()
        frame = _frame_index(df).to_numpy()
        drop_idx = []
        new_rows = []
        for v in range(VOICES):
            creg = int(CTRL_REGS_BY_VOICE[v])
            freg = int(FREQ_REGS_BY_VOICE[v])
            first_on = self._first_gate_on_frame(regs, ops, subs, vals, frame, creg)
            if first_on is None:
                continue
            pre = [
                i
                for i in range(len(df))
                if int(regs[i]) == freg
                and int(ops[i]) == SET_OP
                and int(subs[i]) == -1
                and int(frame[i]) < first_on
            ]
            if not pre:
                continue
            drop_idx.extend(pre)
            gate_has_freq = any(
                int(regs[i]) == freg
                and int(ops[i]) == SET_OP
                and int(frame[i]) == first_on
                for i in range(len(df))
            )
            if not gate_has_freq:
                anchor = next(i for i in range(len(df)) if int(frame[i]) == first_on)
                row = make_row(freg, int(vals[pre[-1]]), op=SET_OP, subreg=-1, irq=irq)
                row["__pos"] = int(anchor)
                new_rows.append(row)
        if not drop_idx:
            return df
        if new_rows:
            from preframr_tokens.macros.passes_base import _splice_rows

            return _splice_rows(df, drop_idx, new_rows)
        return df.drop(index=drop_idx).reset_index(drop=True)

    @staticmethod
    def _first_gate_on_frame(regs, ops, subs, vals, frame, creg):
        """Decoded frame of the first 0->1 gate transition on ``creg`` (its first full-byte ctrl SET
        with bit 0 set), or None -- the boundary below which freq is pre-gate (inaudible).
        """
        gate = 0
        for i in range(len(regs)):
            if int(regs[i]) == creg and int(ops[i]) == SET_OP and int(subs[i]) == -1:
                new_gate = int(vals[i]) & 1
                if new_gate and not gate:
                    return int(frame[i])
                gate = new_gate
        return None
