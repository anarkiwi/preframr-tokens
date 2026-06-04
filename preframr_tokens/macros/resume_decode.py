"""Resumable per-frame decode: capture the full ``DecodeState`` at chosen frame boundaries during a
walk, then resume the walk at a later frame from a restored snapshot. Lets the arbiter validate a
candidate by re-decoding only the suffix after the changed frame instead of the whole df
(``design/parse_perf_block_reencode.md``). The suffix snaps are byte-identical to a full walk's tail.
"""

from __future__ import annotations

import copy

import numpy as np

from preframr_tokens.macros.walker import FrameWalker

__all__ = ["decode_with_state_snaps", "resume_suffix_state"]


class _SnapWalker(FrameWalker):
    """Full register-state walk that also deep-copies ``self.state`` at the END of each requested
    FRAME-loop index (post-tick), so a later walk can resume from frame ``fi + 1``."""

    def __init__(self, df, state, snap_at):
        super().__init__(df, state)
        self.snaps = [np.zeros(25, dtype=np.int64)]
        self._snap_at = snap_at
        self.state_snaps = {}

    def on_frame_end(self):
        if self.cur_frame == -1:
            self.snaps[0] = self.state.last_val[:25].copy()
        else:
            self.snaps.append(self.state.last_val[:25].copy())
        if self.cur_frame in self._snap_at:
            self.state_snaps[self.cur_frame] = copy.deepcopy(self.state)


class _ResumeWalker(FrameWalker):
    """Resume the per-frame walk at ``start_frame`` from an already-restored ``self.state``; collects
    one ``last_val[:25]`` snap per walked frame into ``suffix_snaps``."""

    def __init__(self, df, state):
        super().__init__(df, state)
        self.suffix_snaps = []

    def on_frame_end(self):
        self.suffix_snaps.append(self.state.last_val[:25].copy())

    def walk_from(self, start_frame):
        n_total = len(self.df)
        n_frames = len(self.frame_starts)
        for fi in range(start_frame, n_frames):
            start = int(self.frame_starts[fi])
            end = int(self.frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            self.cur_frame = fi
            self._walk_frame(start, end)


def _expanded(xdf):
    from preframr_tokens.macros.loops import expand_loops
    from preframr_tokens.reglogparser import remove_voice_reg

    df, _ = remove_voice_reg(xdf.copy(), {})
    return expand_loops(df.copy())


def decode_with_state_snaps(xdf, snap_frames):
    """Decode ``xdf`` (post voice-reg + loop expansion, as ``register_state`` does); return
    ``(snaps, state_snaps)`` where ``snaps`` is the ``(n_frames+1, 25)`` array (snaps[0] = lead frame)
    and ``state_snaps[fi]`` is a deep copy of the decode state at the end of FRAME-loop index ``fi``
    for each ``fi`` in ``snap_frames``."""
    from preframr_tokens.macros.state import _build_decode_state

    df = _expanded(xdf)
    state = _build_decode_state(df, strict=False)
    walker = _SnapWalker(df, state, set(int(f) for f in snap_frames))
    walker.walk()
    return np.stack(walker.snaps), walker.state_snaps, df


def resume_suffix_state(df, start_frame, state_snapshot):
    """Resume a decode on the already-expanded ``df`` at FRAME-loop index ``start_frame``, starting
    from ``state_snapshot`` (the end-of-frame ``start_frame - 1`` state). Returns the suffix snaps
    ``(n_frames - start_frame, 25)`` aligned to ``full_snaps[1 + start_frame:]``."""
    walker = _ResumeWalker(df, copy.deepcopy(state_snapshot))
    walker.walk_from(start_frame)
    return (
        np.stack(walker.suffix_snaps)
        if walker.suffix_snaps
        else np.empty((0, 25), dtype=np.int64)
    )
