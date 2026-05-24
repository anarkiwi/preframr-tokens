"""Tag isolated SR/AD envelope writes as ``RELEASE_UPDATE`` (spec #5), behind
the ``release_update_pass`` arg flag (default OFF): a full SR or AD SET with no
same-register neighbour within ``RELEASE_UPDATE_ISOLATION_GAP`` frames becomes a
RELEASE_UPDATE op so it is no longer a lonely SET for the validator."""

__all__ = ["ReleaseUpdatePass"]

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _splice_rows,
)
from preframr_tokens.macros.state import AD_REGS_BY_VOICE, SR_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    RELEASE_UPDATE_ISOLATION_GAP,
    RELEASE_UPDATE_OP,
    SET_OP,
)

_ENV_REGS = frozenset(SR_REGS_BY_VOICE) | frozenset(AD_REGS_BY_VOICE)


class ReleaseUpdatePass(MacroPass):
    def apply(self, df, args=None):
        if args is None or not getattr(args, "release_update_pass", False):
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

        catch_all = getattr(args, "lonely_catch_all", False)
        drop_idx = []
        new_rows = []
        for reg in _ENV_REGS:
            sets = [
                (int(f_idx[i]), i)
                for i in range(len(df))
                if int(regs[i]) == reg
                and int(ops[i]) == SET_OP
                and int(subregs[i]) == -1
            ]
            frames = [fr for fr, _ in sets]
            for pos, (_fr, i) in enumerate(sets):
                if not (catch_all or self._isolated(frames, pos)):
                    continue
                diff = int(diffs[i]) if diffs is not None else 0
                new_rows.append(
                    {
                        "reg": int(reg),
                        "val": int(vals[i]),
                        "diff": diff,
                        "op": int(RELEASE_UPDATE_OP),
                        "subreg": -1,
                        "irq": int(irq_default),
                        "description": 0,
                        "__pos": i,
                    }
                )
                drop_idx.append(i)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)

    @staticmethod
    def _isolated(frames, pos):
        fr = frames[pos]
        prev_ok = pos == 0 or fr - frames[pos - 1] >= RELEASE_UPDATE_ISOLATION_GAP
        next_ok = (
            pos == len(frames) - 1
            or frames[pos + 1] - fr >= RELEASE_UPDATE_ISOLATION_GAP
        )
        return prev_ok and next_ok
