"""Shared primitives for generalization audits and tokenizer profiling: tier
accuracy, tail-cycle, distinct-n, decoded register state, per-op atom profile,
and FREQ/PW/FC trajectory coverage."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd

__all__ = [
    "distinct_n",
    "detect_tail_cycle",
    "tier_accuracy",
    "register_state",
    "op_atom_profile",
    "trajectory_coverage",
]


def distinct_n(tokens, n: int = 4) -> int:
    """Number of distinct n-grams in ``tokens``."""
    if len(tokens) < n:
        return 0
    return len({tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)})


def detect_tail_cycle(
    tokens, tail_window: int = 128, max_period: int = 32, min_repeats: int = 3
):
    """Return shortest period whose ``min_repeats`` copies cover ``tail_window`` tail tokens, else None."""
    n = len(tokens)
    if n < tail_window:
        return None
    tail = list(tokens[n - tail_window :])
    for period in range(1, max_period + 1):
        if tail_window < period * min_repeats:
            continue
        unit = tail[:period]
        if all(tail[i] == unit[i % period] for i in range(tail_window)):
            return {"period": period, "repeats": tail_window // period}
    return None


def tier_accuracy(predicted, ground_truth, tier_map):
    """Bucket (predicted, gt) by gt's tier; return per-class + per-tier + content/structural."""
    n = min(len(predicted), len(ground_truth))
    per_class_hit: dict[int, int] = defaultdict(int)
    per_class_total: dict[int, int] = defaultdict(int)
    per_tier_hit: dict[str, int] = defaultdict(int)
    per_tier_total: dict[str, int] = defaultdict(int)
    UNKNOWN = "_unknown"
    for i in range(n):
        gt = ground_truth[i]
        hit = 1 if predicted[i] == gt else 0
        per_class_total[gt] += 1
        per_class_hit[gt] += hit
        tier = tier_map.get(gt, UNKNOWN)
        per_tier_total[tier] += 1
        per_tier_hit[tier] += hit
    per_class = {
        cls: {
            "n": per_class_total[cls],
            "hits": per_class_hit[cls],
            "acc": per_class_hit[cls] / per_class_total[cls],
            "tier": tier_map.get(cls, UNKNOWN),
        }
        for cls in sorted(per_class_total)
    }
    per_tier = {
        t: {
            "n": per_tier_total[t],
            "hits": per_tier_hit[t],
            "acc": per_tier_hit[t] / per_tier_total[t],
        }
        for t in sorted(per_tier_total)
    }
    struct = per_tier.get("structural", {}).get("acc", 0.0)
    content = per_tier.get("content", {}).get("acc", 0.0)
    return {
        "per_class": per_class,
        "per_tier": per_tier,
        "content_over_structural": content / struct if struct > 0 else 0.0,
        "n_positions": n,
    }


_RS_WALKER_CLS = None


def _register_state_walker_cls():
    """Lazily build the snapshot walker (keeps the heavy decode import off ``import preframr_tokens``):
    a FrameWalker that records regs 0-24 (``state.last_val``) once per decoded frame, accumulating the
    ``(n_frames, 25)`` array directly rather than materialising every literal write into a DataFrame.
    The pre-frame zero row plus one post-tick snapshot per frame (unroll ticks included) reproduce
    ``expand_ops``'s per-marker state, so the result is identical to the old DataFrame reduction.
    """
    global _RS_WALKER_CLS  # pylint: disable=global-statement
    if _RS_WALKER_CLS is None:
        from preframr_tokens.macros.walker import FrameWalker

        class _RegisterStateWalker(FrameWalker):
            def __init__(self, df, state):
                super().__init__(df, state)
                self.snaps = [np.zeros(25, dtype=np.int64)]

            def on_unroll_tick(self, tick_writes, marker_desc):
                self.snaps.append(self.state.last_val[:25].copy())

            def on_frame_end(self):
                """The lead frame (cur_frame -1, content before the first marker) IS frame_reg frame 0,
                so its state REPLACES the pre-frame zero placeholder instead of adding a snapshot --
                keeping the frame budget and every frame_reg-indexed pass aligned (no off-by-one).
                """
                if self.cur_frame == -1:
                    self.snaps[0] = self.state.last_val[:25].copy()
                else:
                    self.snaps.append(self.state.last_val[:25].copy())

        _RS_WALKER_CLS = _RegisterStateWalker
    return _RS_WALKER_CLS


_RS_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_RS_CACHE_MAX = 8


def _rs_cache_key(xdf):
    """Exact content fingerprint of ``xdf`` (all columns + index + row order). register_state is a
    pure function of that content, so identical content -> identical decode."""
    row_hash = pd.util.hash_pandas_object(xdf, index=True).to_numpy()
    return (len(xdf), hashlib.blake2b(row_hash.tobytes(), digest_size=16).digest())


def register_state(xdf):
    """Decoded per-frame SID register state (regs 0-24) as ``(n_frames, 25)`` -- the canonical
    atoms->writes reduction the fidelity oracle + profiler share. Memoized on an exact content
    fingerprint (bounded LRU): the arbiter decodes a pass's input then its output, and the next
    pass's input IS that output, so the source decode repeats it. Self-correcting; the returned
    array is shared and read-only (callers must not mutate)."""
    key = _rs_cache_key(xdf)
    cached = _RS_CACHE.get(key)
    if cached is not None:
        _RS_CACHE.move_to_end(key)
        return cached

    from preframr_tokens.macros.loops import expand_loops
    from preframr_tokens.macros.state import _build_decode_state
    from preframr_tokens.reglogparser import remove_voice_reg

    df, _ = remove_voice_reg(xdf.copy(), {})
    df = expand_loops(df.copy())
    walker = _register_state_walker_cls()(df, _build_decode_state(df, strict=False))
    walker.walk()
    out = np.stack(walker.snaps)

    _RS_CACHE[key] = out
    _RS_CACHE.move_to_end(key)
    while len(_RS_CACHE) > _RS_CACHE_MAX:
        _RS_CACHE.popitem(last=False)
    return out


def op_atom_profile(xdf):
    """Per-op atom histogram, %atoms, total, atoms/frame, per-tier budget (via
    the public op->tier map), and FREQ payload byte-widths for one parsed df."""
    from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE
    from preframr_tokens.macros.transform import collect_op_loss_tiers
    from preframr_tokens.macros.transform_registry import (
        ensure_default_transforms_registered,
    )
    from preframr_tokens.stfconstants import FRAME_REG

    ops = xdf["op"].to_numpy() if "op" in xdf.columns else np.zeros(len(xdf), dtype=int)
    regs = xdf["reg"].to_numpy()
    hist = Counter(int(o) for o in ops)
    total = int(sum(hist.values()))
    n_frames = max(1, int((regs == FRAME_REG).sum()))
    ensure_default_transforms_registered()
    tier_map = collect_op_loss_tiers()
    per_tier = Counter()
    for op, count in hist.items():
        per_tier[tier_map.get(int(op), "other")] += count
    state = register_state(xdf)
    freq = state[:, list(FREQ_REGS_BY_VOICE)]
    freq_nz = freq[freq > 0]
    deltas = np.diff(freq, axis=0).ravel()
    deltas = deltas[deltas != 0]
    return {
        "total_atoms": total,
        "n_frames": n_frames,
        "atoms_per_frame": total / n_frames,
        "op_hist": {int(o): int(c) for o, c in hist.items()},
        "op_pct": {int(o): c / total for o, c in hist.items()} if total else {},
        "per_tier": {t: int(c) for t, c in per_tier.items()},
        "freq_hi_byte_pct": float(np.mean(freq_nz > 0xFF)) if freq_nz.size else 0.0,
        "delta_fits_byte_pct": (
            float(np.mean(np.abs(deltas) <= 127)) if deltas.size else 1.0
        ),
    }


def _record_segment(seq, seg, run_lengths, gaps, alternations):
    run_lengths.append(len(seg))
    gaps.extend(b - a for a, b in zip(seg, seg[1:]))
    vals = [int(seq[f]) for f in seg]
    nz = [b - a for a, b in zip(vals, vals[1:]) if b != a]
    if len(nz) >= 2:
        changes = sum(1 for x, y in zip(nz, nz[1:]) if (x > 0) != (y > 0))
        alternations.append(changes / (len(nz) - 1))


def trajectory_coverage(xdf, tier="freq"):
    """Segment each register's per-frame motion gap-tolerantly (from register_state)
    and report structural FREQ_TRAJ coverage vs mop-ups plus run-length / gap /
    alternation distributions for the ``freq`` / ``pw`` / ``fc`` register family."""
    from preframr_tokens.macros.state import FREQ_REGS_BY_VOICE, PWM_REGS_BY_VOICE
    from preframr_tokens.stfconstants import (
        FC_LO_REG,
        FREQ_TRAJ_OP,
        FT_SUBREG_FLAGS,
        OSC_MAX_GAP,
        SET_OP,
    )

    regs = {
        "freq": tuple(FREQ_REGS_BY_VOICE),
        "pw": tuple(PWM_REGS_BY_VOICE),
        "fc": (int(FC_LO_REG),),
    }[tier]
    state = register_state(xdf)
    run_lengths, gaps, alternations = [], [], []
    for reg in regs:
        seq = state[:, reg]
        changed = [f for f in range(1, len(seq)) if int(seq[f]) != int(seq[f - 1])]
        if not changed:
            continue
        seg = [changed[0]]
        for f in changed[1:]:
            if f - seg[-1] <= OSC_MAX_GAP:
                seg.append(f)
                continue
            _record_segment(seq, seg, run_lengths, gaps, alternations)
            seg = [f]
        _record_segment(seq, seg, run_lengths, gaps, alternations)
    ops = xdf["op"].to_numpy()
    xregs = xdf["reg"].to_numpy()
    subregs = xdf["subreg"].to_numpy() if "subreg" in xdf.columns else None
    in_regs = np.isin(xregs, np.asarray(regs, dtype=np.int64))
    head = (
        (subregs == FT_SUBREG_FLAGS) if subregs is not None else np.ones_like(in_regs)
    )
    structural = int(np.sum(in_regs & (ops == FREQ_TRAJ_OP) & head))
    mopup = int(np.sum(in_regs & (ops == SET_OP)))
    denom = structural + mopup
    return {
        "tier": tier,
        "n_segments": len(run_lengths),
        "run_length_mean": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "gap_mean": float(np.mean(gaps)) if gaps else 0.0,
        "alternation_mean": float(np.mean(alternations)) if alternations else 0.0,
        "structural_atoms": structural,
        "mopup_atoms": mopup,
        "captured_frac": structural / denom if denom else 0.0,
    }
