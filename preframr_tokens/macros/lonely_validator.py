"""Strict-lonely / strict-no-diff validator: behind the ``strict_lonely`` arg
flag (default OFF) it raises ``UnmodelledLonelyWriteError`` for any full SET off
the carveout allow-list and for any DIFF op on any register; flag-off it is an
identity no-op."""

__all__ = [
    "LonelyWriteValidatorPass",
    "UnmodelledLonelyWriteError",
    "classify_carveout",
    "TRAJECTORY_ANCHOR_WINDOW",
]

from preframr_tokens.macros.passes_base import MacroPass
from preframr_tokens.reg_match import reg_class
from preframr_tokens.stfconstants import (
    DELAY_REG,
    DIFF_OP,
    FILTER_REG,
    FLIP_OP,
    FRAME_REG,
    FREQ_TRAJ_OP,
    MODE_VOL_REG,
    SET_OP,
    TRANSPOSE_OP,
)

TRAJECTORY_ANCHOR_WINDOW = 5
_TRAJECTORY_ANCHOR_OPS = {
    FREQ_TRAJ_OP,
    FLIP_OP,
    TRANSPOSE_OP,
}


class UnmodelledLonelyWriteError(Exception):
    """A full SET survived every macro pass without matching a carveout."""


def _is_anchor_op(op):
    """True for the leading row of any trajectory primitive a SET can anchor."""
    return int(op) in _TRAJECTORY_ANCHOR_OPS


def _adjacent_trajectory(regs, ops, k, reg, step, n):
    """True if the nearest same-reg neighbour of ``k`` in direction ``step``
    (within the window) is a trajectory primitive this SET can anchor."""
    j = k + step
    seen = 0
    while 0 <= j < n and seen < TRAJECTORY_ANCHOR_WINDOW:
        if int(regs[j]) == reg:
            return _is_anchor_op(ops[j])
        j += step
        seen += 1
    return False


def classify_carveout(regs, ops, subregs, vals, k, first_seen, last_ctrl_val):
    """Return a carveout id for the full SET at row ``k``, or ``None``;
    ``first_seen`` and ``last_ctrl_val`` are mutable per-walk caches the caller
    threads across rows, matching the reference probe classifier."""
    reg = int(regs[k])
    cls = reg_class(reg)
    if reg == FILTER_REG:
        return "filter_route"
    if reg == MODE_VOL_REG:
        return "master_volume"
    if cls is None:
        return None
    kind, voice = cls
    key = (voice, reg)
    if key not in first_seen:
        first_seen[key] = k
        return "first_voice_write"
    n = len(regs)
    if _adjacent_trajectory(regs, ops, k, reg, 1, n) or _adjacent_trajectory(
        regs, ops, k, reg, -1, n
    ):
        return "trajectory_anchor"
    return _gate_off(kind, vals, k, last_ctrl_val, voice)


def _gate_off(kind, vals, k, last_ctrl_val, voice):
    """gate_off_terminal carveout: CTRL write with gate clear now and before."""
    if kind != "CTRL":
        return None
    new_val = int(vals[k])
    prev = last_ctrl_val.get(voice)
    if (new_val & 0x01) == 0 and prev is not None and (prev & 0x01) == 0:
        return "gate_off_terminal"
    return None


class LonelyWriteValidatorPass(MacroPass):
    GATE_FLAGS = frozenset({"strict_lonely"})

    def apply(self, df, args=None):
        if args is None or not getattr(args, "strict_lonely", False):
            return df
        if df is None or len(df) == 0 or "op" not in df.columns:
            return df
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        subregs = df["subreg"].to_numpy() if "subreg" in df.columns else [-1] * len(df)
        vals = df["val"].to_numpy()
        path = df.attrs.get("source_path", "<unknown>")
        first_seen = {}
        last_ctrl_val = {}
        for k in range(len(df)):
            reg = int(regs[k])
            op = int(ops[k])
            subreg = int(subregs[k])
            cls = reg_class(reg)
            if op == SET_OP and cls is not None and cls[0] == "CTRL":
                last_ctrl_val[cls[1]] = int(vals[k])
            if reg in (FRAME_REG, DELAY_REG) or reg < 0:
                continue
            if op == DIFF_OP:
                raise UnmodelledLonelyWriteError(
                    self._msg(path, k, reg, op, vals[k], "DIFF op under strict-no-diff")
                )
            if op != SET_OP or subreg != -1:
                continue
            carveout = classify_carveout(
                regs, ops, subregs, vals, k, first_seen, last_ctrl_val
            )
            if carveout is None:
                raise UnmodelledLonelyWriteError(
                    self._msg(path, k, reg, op, vals[k], "no carveout")
                )
        return df

    @staticmethod
    def _msg(path, k, reg, op, val, why):
        """One-line actionable failure summary for the raised error."""
        return f"{path}:row{k} reg={reg} op={op} val={int(val)}: {why}"
