"""Tag isolated SR/AD envelope writes as ``RELEASE_UPDATE`` (spec #5), behind
the ``release_update_pass`` arg flag (default OFF): a full SR or AD SET with no
same-register neighbour within ``RELEASE_UPDATE_ISOLATION_GAP`` frames becomes a
RELEASE_UPDATE op so it is no longer a lonely SET for the validator."""

__all__ = ["ReleaseUpdatePass"]

from preframr_tokens.macros.passes_base import (
    _first_irq,
    MacroPass,
    _ensure_subreg,
    _frame_index,
    _frame_isolated,
    _splice_rows,
    make_row,
)
from preframr_tokens.macros.state import AD_REGS_BY_VOICE, SR_REGS_BY_VOICE
from preframr_tokens.stfconstants import (
    RELEASE_UPDATE_ISOLATION_GAP,
    RELEASE_UPDATE_OP,
    SET_OP,
)

_ENV_REGS = frozenset(SR_REGS_BY_VOICE) | frozenset(AD_REGS_BY_VOICE)


class ReleaseUpdatePass(MacroPass):
    GATE_FLAGS = frozenset({"release_update_pass", "lonely_catch_all"})

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
        irq_default = _first_irq(df)

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
                if not (
                    catch_all
                    or _frame_isolated(frames, pos, RELEASE_UPDATE_ISOLATION_GAP)
                ):
                    continue
                diff = int(diffs[i]) if diffs is not None else 0
                row = make_row(
                    int(reg),
                    int(vals[i]),
                    op=RELEASE_UPDATE_OP,
                    subreg=-1,
                    diff=diff,
                    irq=int(irq_default),
                )
                row["__pos"] = i
                new_rows.append(row)
                drop_idx.append(i)
        if not drop_idx:
            return df
        return _splice_rows(df, drop_idx, new_rows)
