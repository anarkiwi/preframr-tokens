"""FreqOnsetPass: re-tag every residual op0 SET on TRAJ_REGS (freq/PW/filter-cutoff) to
FREQ_ONSET (op48), a 1-token tagged write (reg/val unchanged) -- so op0 SET only carries
control/ADSR/routing and all of melody lives in the onset channel. Opt-in
(``freq_onset_pass``); byte-exact (decode is SET-equivalent)."""

__all__ = ["FreqOnsetPass"]

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.stfconstants import FREQ_ONSET_OP, SET_OP, TRAJ_REGS

_TRAJ_REGS = frozenset(TRAJ_REGS)


class FreqOnsetPass(MacroPass):
    """Re-tag residual op0 SET on TRAJ_REGS to FREQ_ONSET (op48)."""

    GATE_FLAGS = frozenset({"freq_onset_pass"})

    def apply(self, df, args=None):
        if args is not None and not getattr(args, "freq_onset_pass", False):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        mask = (
            (df["op"] == SET_OP) & (df["reg"].isin(_TRAJ_REGS)) & (df["subreg"] == -1)
        )
        if mask.any():
            df = df.copy()
            df.loc[mask, "op"] = int(FREQ_ONSET_OP)
        return df
