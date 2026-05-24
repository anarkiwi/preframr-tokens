"""Parametric envelope families for the ``OSCILLATE_ENV`` primitive: the
shared source of truth for the encode-side fitter and decode-side
reconstructor of per-cycle amplitude."""

from __future__ import annotations

__all__ = [
    "ENV_CONSTANT",
    "ENV_LINEAR",
    "ENV_EXP_DECAY",
    "ENV_EXP_GROWTH",
    "ENV_STEP_2",
    "ENV_STEP_3",
    "ENV_STEP_4",
    "ENV_TRIANGULAR",
    "ENV_FAMILY_NAMES",
    "STEP_LEVEL_TABLE",
    "FIT_TOLERANCE",
    "FIT_PRIORITY",
    "cycle_multipliers",
    "fit_family",
]

ENV_CONSTANT = 0
ENV_LINEAR = 1
ENV_EXP_DECAY = 2
ENV_EXP_GROWTH = 3
ENV_STEP_2 = 4
ENV_STEP_3 = 5
ENV_STEP_4 = 6
ENV_TRIANGULAR = 7

ENV_FAMILY_NAMES = {
    ENV_CONSTANT: "CONSTANT",
    ENV_LINEAR: "LINEAR",
    ENV_EXP_DECAY: "EXP_DECAY",
    ENV_EXP_GROWTH: "EXP_GROWTH",
    ENV_STEP_2: "STEP_2",
    ENV_STEP_3: "STEP_3",
    ENV_STEP_4: "STEP_4",
    ENV_TRIANGULAR: "TRIANGULAR",
}

STEP_LEVEL_TABLE = (0.18, 0.37, 0.61, 0.94)
FIT_TOLERANCE = 0.10
FIT_PRIORITY = (
    ENV_CONSTANT,
    ENV_LINEAR,
    ENV_EXP_DECAY,
    ENV_EXP_GROWTH,
    ENV_STEP_2,
    ENV_STEP_3,
    ENV_STEP_4,
    ENV_TRIANGULAR,
)


def _q_ratio_to_param(ratio):
    """Encode a non-negative ratio as Q2.6 in 0..255 (64 == 1.0)."""
    return max(0, min(255, int(round(ratio * 64.0))))


def _param_to_q_ratio(param):
    """Decode a Q2.6 param byte to a ratio (64 == 1.0)."""
    return (int(param) & 0xFF) / 64.0


def _q07_to_param(ratio):
    """Encode a ratio in (0, 1] as Q0.7-ish in 0..255 (128 == 1.0)."""
    return max(0, min(255, int(round(ratio * 128.0))))


def _param_to_q07(param):
    """Decode a Q0.7-ish param byte to a ratio (128 == 1.0)."""
    return (int(param) & 0xFF) / 128.0


def _pack_step_levels(levels):
    """Pack up to four 2-bit level indices low-bits-first into one byte."""
    param = 0
    for k, lvl in enumerate(levels):
        param |= (int(lvl) & 0x3) << (2 * k)
    return param & 0xFF


def _unpack_step_levels(param, count):
    """Unpack ``count`` 2-bit level indices from a packed param byte."""
    return [(int(param) >> (2 * k)) & 0x3 for k in range(count)]


def _nearest_step_index(value):
    """Index of the closest ``STEP_LEVEL_TABLE`` entry to ``value``."""
    return min(
        range(len(STEP_LEVEL_TABLE)),
        key=lambda i: abs(STEP_LEVEL_TABLE[i] - value),
    )


def cycle_multipliers(family, param, n_cycles):
    """Per-cycle envelope multipliers; cycle ``i`` amplitude is base * mult[i]."""
    n = int(n_cycles)
    if n <= 0:
        return []
    family = int(family)
    param = int(param) & 0xFF

    if family == ENV_CONSTANT:
        return [1.0] * n
    if family == ENV_LINEAR:
        end_ratio = _param_to_q_ratio(param)
        if n == 1:
            return [1.0]
        return [1.0 + (end_ratio - 1.0) * (i / (n - 1)) for i in range(n)]
    if family == ENV_EXP_DECAY:
        decay = min(max(_param_to_q07(param), 0.0), 0.999)
        return [decay**i for i in range(n)]
    if family == ENV_EXP_GROWTH:
        growth = max(_param_to_q_ratio(param), 1.0)
        return [growth**i for i in range(n)]
    if family == ENV_STEP_2:
        level_ratio = _param_to_q07(param)
        return [1.0 if i % 2 == 0 else level_ratio for i in range(n)]
    if family == ENV_STEP_3:
        tbl = [STEP_LEVEL_TABLE[lvl] for lvl in _unpack_step_levels(param, 3)]
        return [tbl[i % 3] for i in range(n)]
    if family == ENV_STEP_4:
        tbl = [STEP_LEVEL_TABLE[lvl] for lvl in _unpack_step_levels(param, 4)]
        return [tbl[i % 4] for i in range(n)]
    if family == ENV_TRIANGULAR:
        return _triangular_multipliers(param, n)
    raise ValueError(f"unknown envelope family {family}")


def _triangular_multipliers(param, n):
    """Rise to a crest at ``peak_index`` then decay back toward baseline."""
    peak_index = min(max((param >> 4) & 0x0F, 0), max(n - 1, 0))
    peak_amp = ((param & 0x0F) / 8.0) or (1.0 / 8.0)
    out = []
    for i in range(n):
        if i <= peak_index:
            out.append(
                1.0 if peak_index == 0 else 1.0 + (peak_amp - 1.0) * (i / peak_index)
            )
        else:
            tail = n - 1 - peak_index
            out.append(
                peak_amp
                if tail <= 0
                else peak_amp + (1.0 - peak_amp) * ((i - peak_index) / tail)
            )
    return out


def _residual(norm_amps, multipliers):
    """Mean relative error between normalised amplitudes and multipliers."""
    if not norm_amps:
        return float("inf")
    total = 0.0
    for a, m in zip(norm_amps, multipliers):
        denom = abs(a) if abs(a) > 1e-9 else 1e-9
        total += abs(a - m) / denom
    return total / len(norm_amps)


def _fit_one(family, norm_amps):
    """Best ``(param, residual)`` fitting ``family`` to ``norm_amps``."""
    n = len(norm_amps)
    if family == ENV_CONSTANT:
        param = 0
    elif family == ENV_LINEAR:
        param = _q_ratio_to_param(norm_amps[-1])
    elif family in (ENV_EXP_DECAY, ENV_EXP_GROWTH):
        ratios = [
            norm_amps[i + 1] / norm_amps[i]
            for i in range(n - 1)
            if abs(norm_amps[i]) > 1e-9
        ]
        ratio = (sum(ratios) / len(ratios)) if ratios else 1.0
        if family == ENV_EXP_DECAY:
            param = _q07_to_param(min(max(ratio, 0.0), 0.999))
        else:
            param = _q_ratio_to_param(max(ratio, 1.0))
    elif family == ENV_STEP_2:
        odd = [norm_amps[i] for i in range(1, n, 2)]
        level = (sum(odd) / len(odd)) if odd else 1.0
        param = _q07_to_param(min(max(level, 0.0), 1.0))
    elif family in (ENV_STEP_3, ENV_STEP_4):
        period = 3 if family == ENV_STEP_3 else 4
        levels = []
        for k in range(period):
            members = [norm_amps[i] for i in range(k, n, period)]
            avg = (sum(members) / len(members)) if members else 1.0
            levels.append(_nearest_step_index(avg))
        param = _pack_step_levels(levels)
    elif family == ENV_TRIANGULAR:
        peak_index = max(range(n), key=lambda i: norm_amps[i])
        peak_amp = norm_amps[peak_index]
        param = ((peak_index & 0x0F) << 4) | (
            max(0, min(15, int(round(peak_amp * 8.0)))) & 0x0F
        )
    else:
        raise ValueError(f"unknown envelope family {family}")
    return param, _residual(norm_amps, cycle_multipliers(family, param, n))


def fit_family(norm_amps):
    """Best ``(family, param, residual)`` for amplitudes scaled so amps[0]==1."""
    if not norm_amps:
        return ENV_CONSTANT, 0, float("inf")
    best = None
    for family in FIT_PRIORITY:
        param, res = _fit_one(family, norm_amps)
        if res <= FIT_TOLERANCE:
            return family, param, res
        if best is None or res < best[2]:
            best = (family, param, res)
    return best
