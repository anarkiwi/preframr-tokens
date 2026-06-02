"""Encode-side macro passes."""

import numpy as np
import pandas as pd

__all__ = [
    "TransposePass",
    "HardRestartPass",
    "LegatoPerClusterPass",
    "SubregPass",
    "VoiceBlockOrderPass",
    "DedupSetPass",
]

from preframr_tokens.macros.passes_base import (
    MacroPass,
    _frame_index,
    _splice_rows,
    requires_state,
)
from preframr_tokens.macros.state import (
    AD_REGS_BY_VOICE,
    CTRL_REGS_BY_VOICE,
    FREQ_REGS_BY_VOICE,
    SR_REGS_BY_VOICE,
    SUBREG_REGS,
    _df_arrays_and_frames,
)
from preframr_tokens.macros.walker import FrameWalker
from preframr_tokens.stfconstants import (
    DELAY_REG,
    DIFF_OP,
    FRAME_REG,
    HARD_RESTART_OP,
    LEGATO_OP_CLUSTER_2,
    LEGATO_OP_CLUSTER_4,
    LEGATO_OP_CLUSTER_7,
    SET_OP,
    SUBREG_FLUSH_OP,
    TRANSPOSE_OP,
    VOICES,
    VOICE_REG_SIZE,
)


class TransposePass(MacroPass):
    """Within one frame, collapse same-delta freq DIFFs across >=2 voices."""

    target_regs = FREQ_REGS_BY_VOICE

    def apply(self, df, args=None):
        if (
            "op" not in df.columns
            or not (df["op"].eq(DIFF_OP) & df["reg"].isin(self.target_regs)).any()
        ):
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)

        target_set = set(self.target_regs)
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        target_mask = np.isin(regs, list(target_set)) & (ops == DIFF_OP)
        if not target_mask.any():
            return df
        sub = df[target_mask].copy()
        sub["mf"] = f_idx[target_mask]

        drop_idx = []
        new_rows = []
        target_reg_to_voice = {r: v for v, r in enumerate(self.target_regs)}
        for (_, val), grp in sub.groupby(["mf", "val"], sort=False):
            if len(grp) < 2:
                continue
            grp_regs = grp["reg"].to_numpy()
            grp_idx = grp.index.to_numpy()
            voice_mask = 0
            for r in grp_regs:
                voice_mask |= 1 << target_reg_to_voice[int(r)]
            drop_idx.extend(int(i) for i in grp_idx)
            first_reg = int(grp_regs.min())
            new_rows.append(
                {
                    "reg": first_reg,
                    "val": int(val) & 0xFFFF,
                    "diff": int(grp["diff"].iloc[0]),
                    "op": int(TRANSPOSE_OP),
                    "subreg": int(voice_mask),
                    "__pos": int(grp_idx.min()),
                }
            )
        return _splice_rows(df, drop_idx, new_rows)


class HardRestartPass(MacroPass):
    """Collapse the universal SID hard-restart two-write CTRL pair into one
    ``HARD_RESTART_OP`` token.
    """

    GATE_FLAGS = frozenset({"hard_restart_pass"})
    target_regs = CTRL_REGS_BY_VOICE
    _ctrl_reg_to_voice = {r: v for v, r in enumerate(CTRL_REGS_BY_VOICE)}

    @staticmethod
    def _is_valid_pair(a, b):
        """True iff (a, b) is a recognised hard-restart pair."""
        if (b & 0x09) != 0x01:
            return False
        if a == b:
            return False
        if a == 0x08 or a == 0x09:
            return True
        if a == (b & 0xFE):
            return True
        return False

    def apply(self, df, args=None):
        if args is None or not getattr(args, "hard_restart_pass", False):
            return df
        if "op" not in df.columns:
            return df
        ctrl_mask = df["reg"].isin(self.target_regs) & (df["op"] == SET_OP)
        if "subreg" in df.columns:
            ctrl_mask = ctrl_mask & (df["subreg"] == -1)
        if not ctrl_mask.any():
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        drop_idx = []
        new_rows = []
        for ctrl_reg in self.target_regs:
            sub_mask = (df["reg"] == ctrl_reg) & (df["op"] == SET_OP)
            if "subreg" in df.columns:
                sub_mask = sub_mask & (df["subreg"] == -1)
            sub = df[sub_mask]
            if len(sub) < 2:
                continue
            indices = sub.index.tolist()
            frames = sub["mf"].tolist()
            vals = [int(v) & 0xFF for v in sub["val"].tolist()]
            diffs = sub["diff"].tolist()
            n = len(indices)
            i = 0
            while i + 1 < n:
                a = vals[i]
                b = vals[i + 1]
                if frames[i] != frames[i + 1] or not self._is_valid_pair(a, b):
                    i += 1
                    continue
                packed = ((a & 0xFF) << 8) | (b & 0xFF)
                drop_idx.append(indices[i])
                drop_idx.append(indices[i + 1])
                new_rows.append(
                    {
                        "reg": int(ctrl_reg),
                        "val": int(packed),
                        "diff": int(diffs[i]),
                        "op": int(HARD_RESTART_OP),
                        "subreg": -1,
                        "__pos": int(indices[i]),
                    }
                )
                i += 2

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


_LEGATO_PER_CLUSTER_OP_MAP = {
    2: LEGATO_OP_CLUSTER_2,
    4: LEGATO_OP_CLUSTER_4,
    7: LEGATO_OP_CLUSTER_7,
}
_LEGATO_PER_CLUSTER_FLAG_ATTRS = {
    2: "legato_pass_c2",
    4: "legato_pass_c4",
    7: "legato_pass_c7",
}
_LEGATO_PER_CLUSTER_BYTE_VAL = frozenset({7})

_LEGATO_GATE_BIT = 0x01
_LEGATO_TEST_BIT = 0x08
_LEGATO_PULSE_BIT = 0x40
_LEGATO_WAVEFORM_NIBBLE_MASK = 0xF0


def _legato_c2_rule(prev, cur, adsr_changed):
    """Cluster 2 (Mibri) -- generic gate-retained waveform-only legato."""
    return (
        (prev & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT
        and (cur & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT
        and (cur & _LEGATO_TEST_BIT) == 0
        and (prev & _LEGATO_TEST_BIT) == 0
        and (prev & _LEGATO_WAVEFORM_NIBBLE_MASK)
        != (cur & _LEGATO_WAVEFORM_NIBBLE_MASK)
        and not adsr_changed
    )


def _legato_c3_rule(prev, cur, _adsr_changed):
    """Cluster 3 (Whittaker) -- pulse-bit-clear after a gate-1 CTRL.
    Models the gate_byte & $bf -> waveform-only update idiom."""
    return (prev & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT and (
        cur & _LEGATO_PULSE_BIT
    ) == 0


def _legato_c4_rule(prev, cur, adsr_changed):
    """Cluster 4 (Jammer / Daglish) -- gate-1 retained, ADSR unchanged,
    test-bit unchanged."""
    return (
        (prev & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT
        and (cur & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT
        and (cur & _LEGATO_TEST_BIT) == (prev & _LEGATO_TEST_BIT)
        and not adsr_changed
    )


def _legato_c7_rule(prev, cur, adsr_changed):
    """Cluster 7 (Hubbard) -- pulse-clear after gate-1 (cluster-3 rule)
    or gate-byte $FE / $FC after gate-1 (Hubbard's gate-byte memory
    idiom)."""
    if _legato_c3_rule(prev, cur, adsr_changed):
        return True
    if cur in (0xFE, 0xFC) and (prev & _LEGATO_GATE_BIT) == _LEGATO_GATE_BIT:
        return True
    return False


_LEGATO_PER_CLUSTER_RULES = {
    2: _legato_c2_rule,
    4: _legato_c4_rule,
    7: _legato_c7_rule,
}


class LegatoPerClusterPass(MacroPass):
    """Cluster-specific legato encoder gated on ``df.attrs["engine_fp_cluster"]``."""

    GATE_FLAGS = frozenset(_LEGATO_PER_CLUSTER_FLAG_ATTRS.values())
    target_regs = CTRL_REGS_BY_VOICE

    def apply(self, df, args=None):
        if args is None:
            return df
        if "op" not in df.columns:
            return df
        cluster_id = int(df.attrs.get("engine_fp_cluster", -1))
        op_code = _LEGATO_PER_CLUSTER_OP_MAP.get(cluster_id)
        if op_code is None:
            return df
        flag_attr = _LEGATO_PER_CLUSTER_FLAG_ATTRS[cluster_id]
        if not getattr(args, flag_attr, False):
            return df
        rule = _LEGATO_PER_CLUSTER_RULES[cluster_id]
        byte_val = cluster_id in _LEGATO_PER_CLUSTER_BYTE_VAL

        ctrl_set_mask = df["reg"].isin(self.target_regs) & (df["op"] == SET_OP)
        if "subreg" in df.columns:
            ctrl_set_mask = ctrl_set_mask & (df["subreg"] == -1)
        if not ctrl_set_mask.any():
            return df
        df = df.reset_index(drop=True).copy()
        f_idx = _frame_index(df)
        df["mf"] = f_idx

        adsr_block = {}
        hr_block = {}
        for v in range(VOICES):
            adsr_regs = (AD_REGS_BY_VOICE[v], SR_REGS_BY_VOICE[v])
            adsr_block[v] = set(
                df.loc[df["reg"].isin(adsr_regs), "mf"].astype(int).tolist()
            )
            hr_mask = (df["reg"] == CTRL_REGS_BY_VOICE[v]) & (
                df["op"] == HARD_RESTART_OP
            )
            hr_block[v] = set(df.loc[hr_mask, "mf"].astype(int).tolist())

        drop_idx = []
        new_rows = []
        for v, ctrl_reg in enumerate(self.target_regs):
            sub_mask = (df["reg"] == ctrl_reg) & df["op"].isin(
                (SET_OP, HARD_RESTART_OP)
            )
            if "subreg" in df.columns:
                sub_mask = sub_mask & (df["subreg"] == -1)
            sub = df[sub_mask].sort_index()
            if sub.empty:
                continue
            indices = sub.index.tolist()
            ops = sub["op"].tolist()
            vals = sub["val"].tolist()
            diffs = sub["diff"].tolist()
            frames = sub["mf"].astype(int).tolist()
            prev = 0
            for k in range(len(indices)):
                op = int(ops[k])
                val = int(vals[k])
                if op == HARD_RESTART_OP:
                    prev = val & 0xFF
                    continue
                cur = val & 0xFF
                f = frames[k]
                adsr_changed = f in adsr_block[v] or f in hr_block[v]
                if rule(prev, cur, adsr_changed):
                    emit_val = cur if byte_val else (cur & 0xF0) >> 4
                    drop_idx.append(indices[k])
                    new_rows.append(
                        {
                            "reg": int(ctrl_reg),
                            "val": int(emit_val),
                            "diff": int(diffs[k]),
                            "op": int(op_code),
                            "subreg": -1,
                            "__pos": int(indices[k]),
                        }
                    )
                prev = cur

        df = df.drop(columns=["mf"])
        return _splice_rows(df, drop_idx, new_rows)


class SubregPass(MacroPass):
    """Always-split byte-to-nibble for subreg-eligible registers, with
    SUBREG_FLUSH inserted to preserve byte-equality across intra-frame
    multi-write sequences.
    """

    target_regs = SUBREG_REGS

    @staticmethod
    def _split_byte(cur, prev):
        """Return ``[(subreg, nibble_val)]`` for the nibbles that
        differ. Empty when no nibble changed OR both nibbles changed."""
        cur_lo = cur & 0x0F
        cur_hi = (cur & 0xF0) >> 4
        prev_lo = prev & 0x0F
        prev_hi = (prev & 0xF0) >> 4
        lo_changed = cur_lo != prev_lo
        hi_changed = cur_hi != prev_hi
        if lo_changed and not hi_changed:
            return [(0, cur_lo)]
        if hi_changed and not lo_changed:
            return [(1, cur_hi)]
        return []

    @staticmethod
    def _splice_payloads(
        emitted, reg, row_idx, row_diff, last_emitted_reg, last_emitted_nib
    ):
        """Yield row dicts for one SUBREG_REG full-byte SET split.
        Inserts a SUBREG_FLUSH if the previous emitted row was on the
        same reg but a different nibble."""
        rows = []
        if last_emitted_reg == reg and emitted[0][0] != last_emitted_nib:
            rows.append(
                {
                    "reg": reg,
                    "val": 0,
                    "diff": row_diff,
                    "op": int(SUBREG_FLUSH_OP),
                    "subreg": -1,
                    "__pos": row_idx,
                }
            )
        for subr, sval in emitted:
            rows.append(
                {
                    "reg": reg,
                    "val": int(sval),
                    "diff": row_diff,
                    "op": int(SET_OP),
                    "subreg": subr,
                    "__pos": row_idx,
                }
            )
        return rows

    def apply(self, df, args=None):
        if not df["reg"].isin(self.target_regs).any():
            return df
        assert (df["reg"] == FRAME_REG).any(), (
            "SubregPass requires a FRAME_REG marker; got a slice with "
            "subreg-eligible writes but no frame markers"
        )
        # pylint: disable-next=no-value-for-parameter
        return self._apply_with_state(df, args=args)

    @requires_state
    # pylint: disable-next=unused-argument
    def _apply_with_state(self, df, state, args):
        target_regs = self.target_regs

        class _SubregWalker(FrameWalker):
            def __init__(self, df_, state_):
                super().__init__(df_, state_)
                self.drop_idx = []
                self.new_rows = []
                self.last_emitted_reg = None
                self.last_emitted_nib = None

            def on_marker(self, reg, val, diff, desc):
                self.last_emitted_reg = None
                self.last_emitted_nib = None

            def before_row(self, i, reg, op):
                subreg = int(self.arrs["subreg"][i])
                if reg in target_regs and op == SET_OP and subreg == -1:
                    self._handle_full_byte(i, reg)
                    return False
                return True

            def after_row(self, i, reg, op, writes):
                super().after_row(i, reg, op, writes)
                subreg = int(self.arrs["subreg"][i])
                if reg in target_regs and op == SET_OP and subreg != -1:
                    self.last_emitted_reg = reg
                    self.last_emitted_nib = subreg
                else:
                    self.last_emitted_reg = None
                    self.last_emitted_nib = None

            def _handle_full_byte(self, i, reg):
                row_val = int(self.arrs["val"][i])
                row_idx = int(self.arrs["Index"][i])
                row_diff = int(self.arrs["diff"][i])
                row_desc = int(self.arrs["description"][i])
                prev = self.state.peek(reg)
                emitted = SubregPass._split_byte(row_val, prev)
                if not emitted:
                    self.state.last_val[reg] = row_val
                    self.f_writes.append(
                        (reg, row_val, self.state.diff_for(reg), row_desc)
                    )
                    self.last_emitted_reg = None
                    self.last_emitted_nib = None
                    return
                self.drop_idx.append(row_idx)
                self.new_rows.extend(
                    SubregPass._splice_payloads(
                        emitted,
                        reg,
                        row_idx,
                        row_diff,
                        self.last_emitted_reg,
                        self.last_emitted_nib,
                    )
                )
                self.state.last_val[reg] = row_val
                self.f_writes.append((reg, row_val, self.state.diff_for(reg), row_desc))
                self.last_emitted_reg = reg
                self.last_emitted_nib = emitted[-1][0]

        walker = _SubregWalker(df, state)
        walker.walk()
        return _splice_rows(df, walker.drop_idx, walker.new_rows)


class VoiceBlockOrderPass(MacroPass):
    """Per-frame voice-block reorder by content key. Voice ordering is
    captured by FRAME_REG svt naturally, so no PERM_REG / reg-rewrite
    needed; tokens within each canonical slot collapse across voice
    rotations because the slot's content is voice-invariant."""

    GATE_FLAGS = frozenset({"voice_canonical_block_order"})

    @staticmethod
    def _voice_key(prev_ctrl, prev_freq, v):
        return (
            -(prev_ctrl & 0x01),
            -(prev_ctrl & 0xF0),
            -prev_freq,
            v,
        )

    def apply(self, df, args=None):
        if args is None or not getattr(args, "voice_canonical_block_order", False):
            return df
        if "op" not in df.columns or "reg" not in df.columns:
            return df
        if not df["reg"].isin({FRAME_REG, DELAY_REG}).any():
            return df
        # pylint: disable-next=no-value-for-parameter
        return self._apply_with_state(df, args=args)

    @requires_state
    # pylint: disable-next=unused-argument
    def _apply_with_state(self, df, state, args):
        arrs, frame_starts = _df_arrays_and_frames(df)
        n_total = len(df)
        n_frames = len(frame_starts)
        snapshots = []

        class _SnapshotWalker(FrameWalker):
            track_instruments = False

            def _walk_frame(self, start, end):
                snap = tuple(
                    (
                        int(self.state.last_val[CTRL_REGS_BY_VOICE[v]]),
                        int(self.state.last_val[FREQ_REGS_BY_VOICE[v]])
                        | (int(self.state.last_val[FREQ_REGS_BY_VOICE[v] + 1]) << 8),
                    )
                    for v in range(VOICES)
                )
                snapshots.append(snap)
                super()._walk_frame(start, end)

        _SnapshotWalker(df, state).walk()
        assert len(snapshots) == n_frames

        regs = arrs["reg"]
        cols = list(df.columns)
        df_iloc = df.values
        col_index = {c: cols.index(c) for c in cols}

        from preframr_tokens.macros.op_contracts import non_atom_ops

        exempt_ops = non_atom_ops()
        out_rows = []
        for fi in range(n_frames):
            start = int(frame_starts[fi])
            end = int(frame_starts[fi + 1]) if fi + 1 < n_frames else n_total
            out_rows.append({c: df_iloc[start, col_index[c]] for c in cols})

            if any(
                int(df_iloc[i, col_index["op"]]) in exempt_ops
                for i in range(start + 1, end)
            ):
                for i in range(start + 1, end):
                    out_rows.append({c: df_iloc[i, col_index[c]] for c in cols})
                continue

            prev_states = snapshots[fi]
            keys = tuple(
                self._voice_key(prev_states[v][0], prev_states[v][1], v)
                for v in range(VOICES)
            )
            perm_tuple = tuple(sorted(range(VOICES), key=keys.__getitem__))

            per_voice = {v: [] for v in range(VOICES)}
            non_voice = []
            negative = []
            for i in range(start + 1, end):
                reg = int(regs[i])
                if reg < 0:
                    negative.append(i)
                    continue
                v = reg // VOICE_REG_SIZE
                if 0 <= v < VOICES:
                    per_voice[v].append(i)
                else:
                    non_voice.append(i)

            for slot in range(VOICES):
                phys_v = perm_tuple[slot]
                for i in per_voice[phys_v]:
                    out_rows.append({c: df_iloc[i, col_index[c]] for c in cols})
            for i in non_voice:
                out_rows.append({c: df_iloc[i, col_index[c]] for c in cols})
            for i in negative:
                out_rows.append({c: df_iloc[i, col_index[c]] for c in cols})

        out_df = pd.DataFrame(out_rows, columns=cols)
        for c in cols:
            out_df[c] = out_df[c].astype(df[c].dtype)
        if df.attrs:
            out_df.attrs.update(df.attrs)
        return out_df.reset_index(drop=True)


class DedupSetPass(MacroPass):
    """Drop full-byte SET tokens whose value already matches the running
    register state. The burst passes (PWM/FILTER_SWEEP/FLIP2/INTERVAL) and
    the DIFF/REPEAT/FLIP rewriter can together walk a register's running
    value to exactly the byte the next surviving SET intends to write.
    Without this pass, the encoder emits that SET unchanged and the
    """

    @requires_state
    def apply(self, df, state, args):
        class _DedupWalker(FrameWalker):
            def __init__(self, df_, state_):
                super().__init__(df_, state_)
                self.drop_idx = []

            def before_row(self, i, reg, op):
                if (
                    op == SET_OP
                    and int(self.arrs["subreg"][i]) == -1
                    and self.state.peek(reg) == int(self.arrs["val"][i])
                ):
                    self.drop_idx.append(int(self.arrs["Index"][i]))
                    return False
                return True

        walker = _DedupWalker(df, state)
        walker.walk()
        if not walker.drop_idx:
            return df
        return df.drop(index=walker.drop_idx).reset_index(drop=True)
