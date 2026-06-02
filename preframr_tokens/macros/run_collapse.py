"""Shared register-run collapse: find runs of ``run_len`` consecutive same-register SET writes each
one real frame apart (the shape behind CtrlBigram=2 and CtrlTriple=3) and replace each with one atom
via the Claim/arbiter pipeline. Centralises the two byte-exactness-critical predicates -- one-frame
adjacency and the frame-final guard (a run may only collapse if its last write ends a frame, else the
atom would clobber a later same-frame write) -- so they live once instead of re-inlined per pass.
"""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.arbiter import Claim, arbitrate
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, SET_OP

__all__ = ["one_frame_apart", "frame_final", "collapse_runs"]


def one_frame_apart(regs, i, j):
    """True iff rows ``i`` and ``j`` are exactly one real frame apart: no DELAY between them and
    exactly one FRAME boundary (a DELAY, or zero/multiple frames, is not an adjacent step).
    """
    between = regs[i + 1 : j]
    return not (between == DELAY_REG).any() and int((between == FRAME_REG).sum()) == 1


def frame_final(regs, positions, k, run_len):
    """True iff the run's last write (``positions[k+run_len-1]``) is frame-final: either no further
    same-register write follows, or at least one FRAME separates it from the next one. A non-final
    last write means a later write lands in the same frame and the collapsed atom would clobber it.
    """
    last = int(positions[k + run_len - 1])
    nxt_k = k + run_len
    if nxt_k >= len(positions):
        return True
    nxt = int(positions[nxt_k])
    return int((regs[last + 1 : nxt] == FRAME_REG).sum()) >= 1


def collapse_runs(df, *, run_len, target_regs, build_atom, label):
    """Per register in ``target_regs``, walk its SET writes (subreg=-1) and collapse each maximal-
    advancing window of ``run_len`` writes that are pairwise one frame apart and frame-final into the
    atom ``build_atom(reg, idxs)`` returns (rows get ``__pos`` = the run's first row; a ``None`` return
    declines the run, e.g. an unmappable pair). Each accepted run is a Claim; the arbiter selects a
    non-overlapping byte-exact subset. Returns the rewritten df (or df unchanged when nothing collapses).
    """
    if "op" not in df.columns or "reg" not in df.columns:
        return df
    regs = df["reg"].to_numpy()
    ops = df["op"].to_numpy()
    subregs = (
        df["subreg"].to_numpy() if "subreg" in df.columns else np.full(len(df), -1)
    )
    if FRAME_REG not in regs:
        return df
    claims = []
    for reg in target_regs:
        positions = np.flatnonzero((regs == reg) & (ops == SET_OP) & (subregs == -1))
        k = 0
        while k + run_len - 1 < len(positions):
            idxs = [int(positions[k + o]) for o in range(run_len)]
            adjacent = all(
                one_frame_apart(regs, idxs[o], idxs[o + 1]) for o in range(run_len - 1)
            )
            if adjacent and frame_final(regs, positions, k, run_len):
                atom = build_atom(reg, idxs)
                if atom is not None:
                    for row in atom:
                        row["__pos"] = idxs[0]
                    claims.append(Claim(writes=tuple(idxs), tokens=atom, label=label))
                    k += run_len
                    continue
            k += 1
    if not claims:
        return df
    return arbitrate(df, claims, validate=True)
