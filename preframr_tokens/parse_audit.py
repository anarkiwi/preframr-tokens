"""Toggleable parse-time consistency audit: after each transform verify the invariants a lossless
re-encoding must hold and pinpoint the transform that breaks one, so inconsistencies surface wholesale
rather than bug-by-bug. Enable via env PREFRAMR_PARSE_AUDIT=raise|warn or args.parse_audit (off by
default; it decodes after every pass). Checks declared-lossless passes preserve per-frame
register_state, the elapsed-frame budget is conserved, and forward/back-refs resolve (codebook + loops).
"""

import logging
import os

import numpy as np

__all__ = ["make_pass_audit", "PassAudit"]

_LOSSY_RESETS = frozenset({"SkeletonPass"})
_FRAME_REBASE = frozenset({"_consolidate_frames"})


def make_pass_audit(args=None):
    """Build a PassAudit from env ``PREFRAMR_PARSE_AUDIT`` (raise|warn) or ``args.parse_audit``; an
    off auditor (mode None) is a cheap no-op so the parser can call it unconditionally.
    """
    mode = os.environ.get("PREFRAMR_PARSE_AUDIT")
    if not mode and args is not None:
        mode = getattr(args, "parse_audit", None)
    return PassAudit(mode or None)


class PassAudit:
    """Tracks the expected per-frame register_state and elapsed-frame budget across the pass chain and
    flags the first transform that violates losslessness, the frame budget, or reference integrity.
    Lossy-by-design passes (SkeletonPass RESID-snap; consolidation/cap timeline) re-baseline instead.
    """

    def __init__(self, mode):
        self.mode = mode
        self.on = mode is not None
        self._rs = None
        self._frames = None

    def _state(self, df):
        from preframr_tokens.audit_primitives import register_state
        from preframr_tokens.stfconstants import SET_OP

        return register_state(df if "op" in df.columns else df.assign(op=int(SET_OP)))

    def start(self, df):
        from preframr_tokens.reglogparser import elapsed_frames

        if not self.on:
            return
        self._rs = self._state(df)
        self._frames = elapsed_frames(df)

    def after(self, df, label, lossless=True):
        from preframr_tokens.reglogparser import elapsed_frames

        if not self.on:
            return
        problems = []
        frames = elapsed_frames(df)
        if label in _FRAME_REBASE:
            self._frames = frames
        elif frames != self._frames:
            problems.append(f"elapsed frames {self._frames} -> {frames}")
        if label in _LOSSY_RESETS or label in _FRAME_REBASE:
            self._rs = self._state(df)
        elif lossless:
            problems.extend(self._lossless_problems(df))
        problems.extend(self._ref_problems(df))
        if problems:
            self._report(label, problems)

    def _lossless_problems(self, df):
        rs = self._state(df)
        if rs.shape != self._rs.shape:
            return [f"register_state shape {self._rs.shape} -> {rs.shape}"]
        diff = rs != self._rs
        if not diff.any():
            return []
        fi = int(np.argmax(diff.any(axis=1)))
        r = int(np.where(diff[fi])[0][0])
        return [
            f"register_state diverged: frame {fi} reg {r} "
            f"{int(self._rs[fi, r])} -> {int(rs[fi, r])}"
        ]

    def _ref_problems(self, df):
        from preframr_tokens.macros.validators import validate_stream

        try:
            validate_stream(df.copy())
        except AssertionError as err:
            return [f"reference integrity: {err}"]
        return []

    def _report(self, label, problems):
        msg = f"PARSE AUDIT: {label} broke consistency -- " + "; ".join(problems)
        if self.mode == "raise":
            raise AssertionError(msg)
        logging.warning(msg)
