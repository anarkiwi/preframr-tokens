"""Shared primitives for generalization audits: tier accuracy, tail-cycle, distinct-n."""

from __future__ import annotations

from collections import defaultdict


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
