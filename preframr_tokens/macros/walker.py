"""FrameWalker -- the per-frame DECODERS dispatch driver shared by
encoder passes, validators, the canonical decode walk, and the
redundancy reporter.
"""

__all__ = ["FrameWalker"]

from preframr_tokens.macros.decoders import DECODERS
from preframr_tokens.macros.state import _FastRow, _df_arrays_and_frames
from preframr_tokens.stfconstants import DELAY_REG, FRAME_REG, SET_OP


class FrameWalker:
    """Per-frame DECODERS dispatch over a row DataFrame, with hooks for
    encoder/validator/decoder use cases.
    """

    emit_synthetic_frame_marker = False
    set_fastpath = False

    def __init__(self, df, state):
        """Hoists the column arrays out of ``arrs`` (one attribute lookup per row,
        not a dict lookup per field) and reuses a single ``_FastRow`` buffer across
        rows: decoders read row fields by value and never retain the row, so per-row
        reallocation is pure churn on the hottest loop."""
        self.df = df
        self.state = state
        self.arrs, self.frame_starts = _df_arrays_and_frames(df)
        self.cur_frame = 0
        self.f_writes = []
        self.marker_index = 0
        self._c_reg = self.arrs["reg"]
        self._c_val = self.arrs["val"]
        self._c_op = self.arrs["op"]
        self._c_subreg = self.arrs["subreg"]
        self._c_diff = self.arrs["diff"]
        self._c_desc = self.arrs["description"]
        self._c_idx = self.arrs["Index"]
        self._row_buf = _FastRow(0, 0, 0, 0, 0, 0, 0)

    def walk(self):
        if self.set_fastpath:
            ops = self.arrs["op"]
            subregs = self.arrs["subreg"]
            split_set = (ops == SET_OP) & ((subregs == 0) | (subregs == 1))
            assert not split_set.any(), (
                "FrameWalker.set_fastpath requires no subreg-split SET rows; "
                f"input has {int(split_set.sum())}"
            )
        n_total = len(self.df)
        n_frames = len(self.frame_starts)
        lead_end = int(self.frame_starts[0]) if n_frames else n_total
        if lead_end > 0:
            self._walk_lead(lead_end)
        for fi in range(n_frames):
            start = int(self.frame_starts[fi])
            end = int(self.frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            self.cur_frame = fi
            self._walk_frame(start, end)

    def _walk_lead(self, end):
        """Content before the first frame marker is a real frame (a tune's note-on / initial register
        load): the first marker ENDS it. Render it from row 0 so its writes -- and any atom anchored
        there (a SWEEP/wavetable consolidated into frame 0) -- are decoded, instead of dropped.
        """
        self.f_writes = []
        self.cur_frame = -1
        for i in range(0, end):
            self._dispatch_row(i)
        self.on_body_end()
        tick = self.state.tick_frame()
        self.on_frame_tick(tick)
        self.f_writes.extend(tick)
        self.on_pre_observe(self.f_writes)
        self.on_frame_end()

    def _walk_frame(self, start, end):
        self.f_writes = []
        marker_reg = int(self.arrs["reg"][start])
        marker_val = int(self.arrs["val"][start])
        marker_diff = int(self.arrs["diff"][start])
        marker_desc = int(self.arrs["description"][start])
        self.marker_index = int(self.arrs["Index"][start])
        self.on_marker(marker_reg, marker_val, marker_diff, marker_desc)
        self._drain_deferred(self.state.pending_deferred_pre_unroll)
        if marker_reg == DELAY_REG:
            for _ in range(marker_val - 1):
                self._unroll_delay(marker_desc)
        self._emit_marker_writes(marker_reg, marker_val, marker_diff, marker_desc)
        self._drain_deferred(self.state.pending_deferred_post_marker)
        for i in range(start + 1, end):
            self._dispatch_row(i)
        self.on_body_end()
        tick = self.state.tick_frame()
        self.on_frame_tick(tick)
        self.f_writes.extend(tick)
        self.on_pre_observe(self.f_writes)
        self.on_frame_end()

    def _emit_marker_writes(self, reg, val, diff, desc):
        if not self.emit_synthetic_frame_marker:
            return
        if reg == FRAME_REG:
            self.f_writes.append((reg, val, diff, desc))
        elif reg == DELAY_REG:
            self.f_writes.append((FRAME_REG, 0, self.state.frame_diff, desc))
        else:
            raise AssertionError(f"unknown marker reg {reg}")

    def _unroll_delay(self, marker_desc):
        unroll_writes = []
        if self.emit_synthetic_frame_marker:
            unroll_writes.append((FRAME_REG, 0, self.state.frame_diff, marker_desc))
        tick = self.state.tick_frame()
        self.on_unroll_tick(tick, marker_desc)
        unroll_writes.extend(tick)
        self.on_pre_observe(unroll_writes)

    def _drain_deferred(self, pending):
        if not pending:
            return
        rows = list(pending)
        pending.clear()
        for base_op, row in rows:
            decoder = DECODERS.get(base_op)
            if decoder is None:
                continue
            writes = decoder.expand(row, self.state)
            self.after_row(-1, int(row.reg), base_op, writes)

    def _dispatch_row(self, i):
        reg = int(self._c_reg[i])
        if reg < 0:
            return
        op = int(self._c_op[i])
        if not self.before_row(i, reg, op):
            return
        if self.set_fastpath and op == SET_OP and int(self._c_subreg[i]) == -1:
            val = int(self._c_val[i])
            diff = int(self._c_diff[i])
            self.state.last_val[reg] = val
            self.state.last_diff[reg] = diff
            self.after_row(i, reg, op, [(reg, val, diff, 0)])
            return
        decoder = DECODERS.get(op)
        if decoder is None:
            return
        row = self._row_buf
        row.reg = reg
        row.val = int(self._c_val[i])
        row.op = op
        row.subreg = int(self._c_subreg[i])
        row.diff = int(self._c_diff[i])
        row.description = int(self._c_desc[i])
        row.Index = int(self._c_idx[i])
        writes = decoder.expand(row, self.state)
        self.after_row(i, reg, op, writes)

    def on_marker(self, reg, val, diff, desc):
        pass

    def on_unroll_tick(self, tick_writes, marker_desc):
        pass

    def on_pre_observe(self, writes):
        pass

    # pylint: disable=unused-argument
    def before_row(self, i, reg, op):
        return True

    def after_row(self, i, reg, op, writes):
        if writes:
            self.f_writes.extend(writes)

    # pylint: enable=unused-argument

    def on_body_end(self):
        pass

    def on_frame_tick(self, tick_writes):
        pass

    def on_frame_end(self):
        pass
