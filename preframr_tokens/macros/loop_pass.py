"""LoopPass and the LZ77 / state-walk machinery it drives."""

__all__ = ["LoopPass"]

from collections import defaultdict

import numpy as np

from preframr_tokens.macros.loops import (
    OVERLAY_BODY_FREQ_DELTA,
    OVERLAY_BODY_FREQ_DELTA_BIN,
    _FREQ_REGS_VOICED,
    _bin_body_freq_delta,
    _pattern_overlay_rows,
    _pattern_replay_rows,
)
from preframr_tokens.macros.passes_base import MacroPass, _ensure_subreg, _rows_to_df
from preframr_tokens.macros.state import _FRAME_MARKER_REGS, _build_decode_state
from preframr_tokens.macros.walker import FrameWalker
from preframr_tokens.stfconstants import (
    DO_LOOP_OP,
    LOOP_OP_REG,
    SET_OP,
    VOICES,
    VOICE_REG_SIZE,
)


def _slice_into_frames(df):
    """Return a list of (start_row_idx, end_row_idx) per frame. Frame
    boundaries are FRAME_REG / DELAY_REG rows -- those rows belong to the
    frame they start. The final frame extends to the end of df."""
    starts = df.index[df["reg"].isin(_FRAME_MARKER_REGS)].tolist()
    if not starts:
        return []
    ends = starts[1:] + [len(df)]
    return list(zip(starts, ends))


def _frame_contents_batch(df, frames):
    """Hashable, comparable content tuple per frame -- ignores diff and irq
    columns so that sequential identical-content frames at different stream
    times still match.
    """
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    if "subreg" in df.columns:
        subregs = df["subreg"].to_numpy()
    else:
        subregs = np.full(len(df), -1, dtype=np.int64)
    out = []
    for s, e in frames:
        out.append(
            tuple(
                zip(
                    regs[s:e].tolist(),
                    vals[s:e].tolist(),
                    ops[s:e].tolist(),
                    subregs[s:e].tolist(),
                )
            )
        )
    return out


def _frame_stripped_contents_batch(df, frames):
    """Like ``_frame_contents_batch`` but freq SET vals are replaced by a
    placeholder so that stripped content matches across transpositions.
    Returned alongside per-frame freq-SET position lists for the
    transposed-match step.
    """
    regs = df["reg"].to_numpy()
    vals = df["val"].to_numpy()
    ops = df["op"].to_numpy()
    if "subreg" in df.columns:
        subregs = df["subreg"].to_numpy()
    else:
        subregs = np.full(len(df), -1, dtype=np.int64)
    is_freq_set = (
        np.isin(regs, list(_FREQ_REGS_VOICED)) & (ops == SET_OP) & (subregs == -1)
    )
    stripped = []
    for s, e in frames:
        rs = regs[s:e].tolist()
        vs = vals[s:e].tolist()
        os = ops[s:e].tolist()
        ss = subregs[s:e].tolist()
        is_fs = is_freq_set[s:e]
        stripped.append(
            tuple((rs[k], 0 if is_fs[k] else vs[k], os[k], ss[k]) for k in range(e - s))
        )
    return stripped


try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

    def njit(*args, **kwargs):
        def _inner(f):
            return f

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _inner


@njit(cache=True)
def _best_lz_njit(
    i,
    cands_arr,
    frame_data,
    frame_starts,
    frame_lens,
    sizes_cumsum,
    max_lz_length,
    n_frames,
    min_lz_match,
    emit_rows,
):
    """Inner loop of LoopPass.best_lz, JIT-compiled."""
    best_save = 0
    best_dist = 0
    best_len = 0
    n_cands = cands_arr.shape[0]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        while length < max_lz_length and i + length < n_frames and cand + length < i:
            ai = i + length
            ac = cand + length
            la = frame_lens[ai]
            lc = frame_lens[ac]
            if la != lc:
                break
            sa = frame_starts[ai]
            sc = frame_starts[ac]
            ok = True
            for k in range(la):
                if (
                    frame_data[sa + k, 0] != frame_data[sc + k, 0]
                    or frame_data[sa + k, 1] != frame_data[sc + k, 1]
                    or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                    or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                ):
                    ok = False
                    break
            if not ok:
                break
            length += 1
        if length < min_lz_match:
            continue
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - emit_rows
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
    return best_save, best_dist, best_len


@njit(cache=True)
def _best_lz_fuzzy_match_njit(
    i,
    cands_arr,
    fps_arr,
    snap_arr,
    sizes_cumsum,
    max_fuzzy_length,
    n_frames,
    min_fuzzy_match,
    head_rows,
    per_overlay_rows,
):
    """Inner walk + overlay-count for LoopPass.best_lz_fuzzy."""
    best_save = 0
    best_dist = 0
    best_len = 0
    n_cands = cands_arr.shape[0]
    n_regs = snap_arr.shape[1]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        while (
            length < max_fuzzy_length
            and i + length < n_frames
            and cand + length < i
            and fps_arr[i + length] == fps_arr[cand + length]
        ):
            length += 1
        if length < min_fuzzy_match:
            continue
        n_overlays = 0
        for k in range(length):
            for r in range(n_regs):
                if snap_arr[i + k, r] != snap_arr[cand + k, r]:
                    n_overlays += 1
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - (head_rows + per_overlay_rows * n_overlays)
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
    return best_save, best_dist, best_len


@njit(cache=True)
def _best_lz_transposed_njit(
    i,
    cands_arr,
    frame_data,
    frame_starts,
    frame_lens,
    freq_set_mask,
    sizes_cumsum,
    max_lz_length,
    n_frames,
    min_lz_match,
    emit_rows,
    bin_w,
):
    """Inner loop of LoopPass.best_lz_transposed, JIT-compiled. Only accepts a run whose uniform freq
    delta bins EXACTLY (``delta % bin_w == 0``): the emit stores ``_bin_body_freq_delta(delta)`` and the
    decode replays source+binned, so a delta that doesn't land on a bin would replay the wrong pitch.
    Lossy-delta runs are left for the literal / non-transposed encoders -- byte-exact over compression.
    """
    best_save = 0
    best_dist = 0
    best_len = 0
    best_delta = 0
    n_cands = cands_arr.shape[0]
    for c_idx in range(n_cands - 1, -1, -1):
        cand = cands_arr[c_idx]
        if cand >= i:
            continue
        length = 0
        delta = 0
        delta_set = False
        while length < max_lz_length and i + length < n_frames and cand + length < i:
            ai = i + length
            ac = cand + length
            la = frame_lens[ai]
            lc = frame_lens[ac]
            if la != lc:
                break
            sa = frame_starts[ai]
            sc = frame_starts[ac]
            ok = True
            for k in range(la):
                fa = freq_set_mask[sa + k]
                fc = freq_set_mask[sc + k]
                if fa != fc:
                    ok = False
                    break
                if fa:
                    if (
                        frame_data[sa + k, 0] != frame_data[sc + k, 0]
                        or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                        or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                    ):
                        ok = False
                        break
                    d = frame_data[sa + k, 1] - frame_data[sc + k, 1]
                    if not delta_set:
                        delta = d
                        delta_set = True
                    elif d != delta:
                        ok = False
                        break
                else:
                    if (
                        frame_data[sa + k, 0] != frame_data[sc + k, 0]
                        or frame_data[sa + k, 1] != frame_data[sc + k, 1]
                        or frame_data[sa + k, 2] != frame_data[sc + k, 2]
                        or frame_data[sa + k, 3] != frame_data[sc + k, 3]
                    ):
                        ok = False
                        break
            if not ok:
                break
            length += 1
        if length < min_lz_match or not delta_set or delta == 0:
            continue
        if bin_w > 1 and delta % bin_w != 0:
            continue
        body_rows = sizes_cumsum[i + length] - sizes_cumsum[i]
        save = body_rows - emit_rows
        if save > best_save:
            best_save = save
            best_dist = i - cand
            best_len = length
            best_delta = delta
    return best_save, best_dist, best_len, best_delta


class LoopPass(MacroPass):
    """Hybrid encoder for repeated frame sequences."""

    GATE_FLAGS = frozenset(
        {"loop_pass", "loop_transposed", "fuzzy_loop_pass", "fuzzy_fp_adsr"}
    )
    min_lz_match = 2
    min_do_repeat = 2
    max_lz_length = 64
    max_do_body = 32
    max_do_repeat = 255
    lz_emit_rows = 2
    pattern_replay_head_rows = 3
    pattern_overlay_rows = 3
    transposed_emit_rows = pattern_replay_head_rows + pattern_overlay_rows
    do_wrap_cost = 2

    def apply(self, df, args=None):
        if args is not None and not getattr(args, "loop_pass", True):
            return df
        loop_transposed = (
            getattr(args, "loop_transposed", True) if args is not None else True
        )
        fuzzy_loop = (
            getattr(args, "fuzzy_loop_pass", False) if args is not None else False
        )
        fuzzy_fp_adsr = (
            getattr(args, "fuzzy_fp_adsr", False) if args is not None else False
        )
        loop_lookahead = (
            max(1, int(getattr(args, "loop_lookahead", 1))) if args is not None else 1
        )
        df = df.reset_index(drop=True).copy()
        df = _ensure_subreg(df)
        frames = _slice_into_frames(df)
        n_frames = len(frames)
        if n_frames < self.min_lz_match:
            return df
        contents = _frame_contents_batch(df, frames)
        stripped = (
            _frame_stripped_contents_batch(df, frames) if loop_transposed else None
        )
        sizes = [e - s for s, e in frames]
        sizes_cumsum = np.zeros(n_frames + 1, dtype=np.int64)
        if sizes:
            sizes_cumsum[1:] = np.cumsum(np.asarray(sizes, dtype=np.int64))
        regs_full = df["reg"].to_numpy()
        vals_full = df["val"].to_numpy()
        ops_full = df["op"].to_numpy()
        subregs_full = (
            df["subreg"].to_numpy()
            if "subreg" in df.columns
            else np.full(len(df), -1, dtype=np.int64)
        )
        frame_data = np.column_stack(
            (
                regs_full.astype(np.int64),
                vals_full.astype(np.int64),
                ops_full.astype(np.int64),
                subregs_full.astype(np.int64),
            )
        )
        frame_starts_arr = np.asarray([s for s, _ in frames], dtype=np.int64)
        frame_lens_arr = np.asarray(sizes, dtype=np.int64)
        freq_set_mask = (
            np.isin(regs_full, list(_FREQ_REGS_VOICED))
            & (ops_full == SET_OP)
            & (subregs_full == -1)
        )

        fps = []
        snapshots = []
        if fuzzy_loop:
            try:
                fps, snapshots = _per_frame_state_walk(df, include_adsr=fuzzy_fp_adsr)
            except Exception:
                fps, snapshots = [], []
            if len(fps) != n_frames:
                fps, snapshots = [], []

        seed = defaultdict(list)
        seed_stripped = defaultdict(list)
        seed_fp = defaultdict(list)
        out_rows = []
        sample_row = df.iloc[0]
        diff_default = int(sample_row["diff"]) if "diff" in df.columns else 0
        irq_default = int(df["irq"].iloc[0]) if "irq" in df.columns else -1
        _rec_cols = list(df.columns)
        _rec_arrs = [df[c].to_numpy() for c in _rec_cols]
        all_records = [dict(zip(_rec_cols, vals)) for vals in zip(*_rec_arrs)]
        snapshot_regs = sorted(snapshots[0].keys()) if snapshots else []
        snapshot_regs_arr = np.asarray(snapshot_regs, dtype=np.int64)
        if snapshots:
            snap_arr = np.zeros((len(snapshots), len(snapshot_regs)), dtype=np.int64)
            for fi, snap in enumerate(snapshots):
                for ri, r in enumerate(snapshot_regs):
                    snap_arr[fi, ri] = int(snap.get(r, 0))
        else:
            snap_arr = np.zeros((0, 0), dtype=np.int64)
        fps_arr = (
            np.asarray(fps, dtype=np.int64) if fps else np.zeros(0, dtype=np.int64)
        )

        def best_do(i):
            best_save = 0
            best_body = 0
            best_n = 0
            for body_len in range(1, min(self.max_do_body, (n_frames - i) // 2) + 1):
                n = 1
                j = i + body_len
                while (
                    j + body_len <= n_frames
                    and n < self.max_do_repeat
                    and contents[i : i + body_len] == contents[j : j + body_len]
                ):
                    n += 1
                    j += body_len
                if n < self.min_do_repeat:
                    continue
                body_rows = int(sizes_cumsum[i + body_len] - sizes_cumsum[i])
                save = (n - 1) * body_rows - self.do_wrap_cost
                if save > best_save:
                    best_save, best_body, best_n = save, body_len, n
            return best_save, best_body, best_n

        def best_lz(i):
            if i + 1 >= n_frames:
                return 0, 0, 0
            cands = seed.get((contents[i], contents[i + 1]))
            if not cands:
                return 0, 0, 0
            cands_arr = np.asarray(cands, dtype=np.int64)
            return _best_lz_njit(
                i,
                cands_arr,
                frame_data,
                frame_starts_arr,
                frame_lens_arr,
                sizes_cumsum,
                self.max_lz_length,
                n_frames,
                self.min_lz_match,
                self.lz_emit_rows,
            )

        def best_lz_transposed(i):
            """Like ``best_lz`` but matches frames whose freq SET vals
            differ from the source by a uniform delta. Returns
            ``(save, dist, length, delta)``. delta=0 implies exact match
            -- defer to ``best_lz`` for that case."""
            if i + 1 >= n_frames:
                return 0, 0, 0, 0
            cands = seed_stripped.get((stripped[i], stripped[i + 1]))
            if not cands:
                return 0, 0, 0, 0
            cands_arr = np.asarray(cands, dtype=np.int64)
            return _best_lz_transposed_njit(
                i,
                cands_arr,
                frame_data,
                frame_starts_arr,
                frame_lens_arr,
                freq_set_mask,
                sizes_cumsum,
                self.max_lz_length,
                n_frames,
                self.min_lz_match,
                self.transposed_emit_rows,
                int(OVERLAY_BODY_FREQ_DELTA_BIN),
            )

        def _seed_pair(i):
            seed[(contents[i], contents[i + 1])].append(i)
            if loop_transposed:
                seed_stripped[(stripped[i], stripped[i + 1])].append(i)
            if fuzzy_loop and fps:
                seed_fp[(fps[i], fps[i + 1])].append(i)

        def emit_literal(i):
            s, e = frames[i]
            out_rows.extend(all_records[s:e])
            if i + 1 < n_frames:
                _seed_pair(i)

        def emit_back_ref(i, dist, length):
            out_rows.extend(
                _pattern_replay_rows(
                    dist,
                    length,
                    overlay_count=0,
                    diff_default=diff_default,
                    irq_default=irq_default,
                )
            )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def emit_back_ref_transposed(i, dist, length, delta):
            out_rows.extend(
                _pattern_replay_rows(
                    dist,
                    length,
                    overlay_count=1,
                    diff_default=diff_default,
                    irq_default=irq_default,
                )
            )
            out_rows.extend(
                _pattern_overlay_rows(
                    frame_offset=-1,
                    target_reg=OVERLAY_BODY_FREQ_DELTA,
                    new_val=_bin_body_freq_delta(int(delta)),
                    diff_default=diff_default,
                    irq_default=irq_default,
                )
            )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def emit_do_loop(i, body, n):
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": int(n),
                    "diff": diff_default,
                    "op": int(DO_LOOP_OP),
                    "subreg": 0,
                    "irq": irq_default,
                    "description": 0,
                }
            )
            for k in range(body):
                s, e = frames[i + k]
                out_rows.extend(all_records[s:e])
            out_rows.append(
                {
                    "reg": int(LOOP_OP_REG),
                    "val": 0,
                    "diff": diff_default,
                    "op": int(DO_LOOP_OP),
                    "subreg": 1,
                    "irq": irq_default,
                    "description": 0,
                }
            )
            covered = body * n
            for k in range(covered):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        def compute_overlays(src_idx, dst_idx, length):
            src_view = snap_arr[src_idx : src_idx + length]
            dst_view = snap_arr[dst_idx : dst_idx + length]
            diff_mask = src_view != dst_view
            ks, ridxs = np.nonzero(diff_mask)
            if ks.size == 0:
                return []
            regs_out = snapshot_regs_arr[ridxs]
            vals_out = dst_view[ks, ridxs]
            return list(zip(ks.tolist(), regs_out.tolist(), vals_out.tolist()))

        max_fuzzy_length = 16
        min_fuzzy_match = 2

        def best_lz_fuzzy(i):
            if not fuzzy_loop or not fps:
                return 0, 0, 0, []
            if i + 1 >= n_frames:
                return 0, 0, 0, []
            cands = seed_fp.get((fps[i], fps[i + 1]))
            if not cands:
                return 0, 0, 0, []
            cands_arr = np.asarray(cands, dtype=np.int64)
            save, dist, length = _best_lz_fuzzy_match_njit(
                i,
                cands_arr,
                fps_arr,
                snap_arr,
                sizes_cumsum,
                max_fuzzy_length,
                n_frames,
                min_fuzzy_match,
                self.pattern_replay_head_rows,
                self.pattern_overlay_rows,
            )
            if save <= 0:
                return 0, 0, 0, []
            overlays = compute_overlays(i - dist, i, length)
            return save, dist, length, overlays

        def emit_pattern_replay(i, dist, length, overlays):
            out_rows.extend(
                _pattern_replay_rows(
                    dist,
                    length,
                    overlay_count=len(overlays),
                    diff_default=diff_default,
                    irq_default=irq_default,
                )
            )
            for frame_offset, reg, val in overlays:
                out_rows.extend(
                    _pattern_overlay_rows(
                        frame_offset=int(frame_offset),
                        target_reg=int(reg),
                        new_val=int(val),
                        diff_default=diff_default,
                        irq_default=irq_default,
                    )
                )
            for k in range(length):
                if i + k + 1 < n_frames:
                    _seed_pair(i + k)

        i = 0
        while i < n_frames:
            do_save, do_body, do_n = best_do(i)
            lz_save, lz_dist, lz_len = best_lz(i)
            if loop_transposed:
                tr_save, tr_dist, tr_len, tr_delta = best_lz_transposed(i)
            else:
                tr_save = tr_dist = tr_len = tr_delta = 0
            fz_save, fz_dist, fz_len, fz_overlays = best_lz_fuzzy(i)
            best_now = max(do_save, lz_save, tr_save, fz_save)
            la_bumped = False
            if best_now > 0:
                for la_step in range(1, loop_lookahead + 1):
                    j = i + la_step
                    if j >= n_frames:
                        break
                    la_do, _, _ = best_do(j)
                    la_lz, _, _ = best_lz(j)
                    if loop_transposed:
                        la_tr, _, _, _ = best_lz_transposed(j)
                    else:
                        la_tr = 0
                    la_fz, _, _, _ = best_lz_fuzzy(j)
                    if max(la_do, la_lz, la_tr, la_fz) > best_now + 2:
                        emit_literal(i)
                        i += 1
                        la_bumped = True
                        break
            if la_bumped:
                continue
            if (
                do_save > 0
                and do_save >= lz_save
                and do_save >= tr_save
                and do_save >= fz_save
            ):
                emit_do_loop(i, do_body, do_n)
                i += do_body * do_n
            elif lz_save > 0 and lz_save >= tr_save and lz_save >= fz_save:
                emit_back_ref(i, lz_dist, lz_len)
                i += lz_len
            elif tr_save > 0 and tr_save >= fz_save:
                emit_back_ref_transposed(i, tr_dist, tr_len, tr_delta)
                i += tr_len
            elif fz_save > 0:
                emit_pattern_replay(i, fz_dist, fz_len, fz_overlays)
                i += fz_len
            else:
                emit_literal(i)
                i += 1

        if not out_rows:
            return df
        orig_dtypes = df.dtypes.to_dict()
        new_df = _rows_to_df(out_rows, df.columns, defaults={"description": 0})
        for col, dt in orig_dtypes.items():
            try:
                new_df[col] = new_df[col].astype(dt)
            except (TypeError, ValueError):
                pass
        new_df = new_df.reset_index(drop=True)
        if df.attrs:
            new_df.attrs.update(df.attrs)
        return new_df


def _musical_fingerprint(state, include_adsr=False):
    """Compact musical-state fingerprint from a ``DecodeState``."""
    fp = 0
    for v in range(VOICES):
        base = v * VOICE_REG_SIZE
        ctrl = state.peek(base + 4)
        freq = state.peek(base + 0)
        gate = ctrl & 0x01
        wave = (ctrl & 0xF0) >> 4
        note = freq & 0xFF if gate else 0xFF
        if include_adsr:
            ad = state.peek(base + 5)
            attack = (ad & 0xF0) >> 4
            fp = (fp << 17) | (note << 9) | (wave << 5) | (gate << 4) | attack
        else:
            fp = (fp << 13) | (note << 5) | (wave << 1) | gate
    cutoff = (state.peek(22) >> 4) & 0x0F
    modevol = state.peek(24) & 0x0F
    fp = (fp << 8) | (cutoff << 4) | modevol
    return fp


def _per_frame_state_walk(df, include_adsr=False):
    """Walk df via ``_simulate_palette``-style dispatch and capture, per
    logical frame, (fingerprint, state.last_val snapshot). The state
    snapshot is the dict of register → end-of-frame byte value, used by
    ``LoopPass``'s fuzzy matcher to compute state-level overlay diffs
    between a candidate source body and the target. Only voice-bound
    """
    state = _build_decode_state(df)
    if state is None:
        return [], []
    snapshot_regs = list(range(VOICES * VOICE_REG_SIZE)) + [22, 23, 24]

    class _StateSnapshotWalker(FrameWalker):
        emit_synthetic_frame_marker = True
        set_fastpath = False

        def __init__(self, df_, state_):
            super().__init__(df_, state_)
            self.fps = []
            self.snapshots = []

        def on_frame_end(self):
            self.fps.append(_musical_fingerprint(self.state, include_adsr=include_adsr))
            self.snapshots.append({r: self.state.peek(r) for r in snapshot_regs})

    walker = _StateSnapshotWalker(df, state)
    walker.walk()
    return walker.fps, walker.snapshots
