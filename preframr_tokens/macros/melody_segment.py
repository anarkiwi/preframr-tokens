"""Note-onset segmenter for the melody-skeleton (layer 2, ``melody_skeleton``): the landed
TrajectoryAnchorPass pass-1 detector (sustained pitch-level change) unioned with gate-on retriggers,
so a held-gate legato line that moves pitch under one sustained gate segments by its intrinsic level
changes (not raw gate) while a re-struck same-pitch note still onsets on its gate edge. Segmentation
only -- it emits no ops; the GeneratorPass re-keys the freq atoms that start on these frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["SegmentParams", "note_onsets", "pass1_origins"]


@dataclass(frozen=True)
class SegmentParams:
    """Pass-1 detector knobs (semitone band over a median-smoothed pitch line); validated starting
    values from the upstream trajectory-anchor prototype."""

    freq_w: int = 5
    freq_band: float = 1.0
    freq_min_hold: int = 3
    match_w: int = 2


def _to_semitones(freq_per_frame) -> np.ndarray:
    """Per-frame 16-bit freq -> semitone ``round(12*log2 f)`` (NaN when silent ``f<=0``)."""
    out = np.full(len(freq_per_frame), np.nan, dtype=float)
    for i, f in enumerate(freq_per_frame):
        if f is not None and f > 0:
            out[i] = float(round(12.0 * math.log2(float(f))))
    return out


def _smooth(val: np.ndarray, w: int) -> np.ndarray:
    """Median-filter (window ``w``) over a NaN-interpolated copy of ``val`` to suppress vibrato/PWM
    jitter so a held-with-modulation level reads flat."""
    valid = ~np.isnan(val)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros(len(val), dtype=float)
    series = np.interp(np.arange(len(val)), idx, val[idx])
    n = len(series)
    half = w // 2
    if half == 0 or n < 2 * half + 1:
        return np.array(
            [np.median(series[max(0, i - half) : i + half + 1]) for i in range(n)],
            dtype=float,
        )
    out = np.empty(n, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(series, 2 * half + 1)
    out[half : n - half] = np.median(windows, axis=1)
    for i in range(half):
        out[i] = np.median(series[0 : i + half + 1])
    for i in range(n - half, n):
        out[i] = np.median(series[i - half : n])
    return out


def pass1_origins(val: np.ndarray, band: float, w: int, min_hold: int) -> list[int]:
    """Sustained level-change origins: walk the smoothed pitch tracking a reference level, emit an
    origin where the value leaves the reference by more than ``band`` and the new level holds
    ``>= min_hold`` frames (a transient returning sooner is modulation). Deliberately over-segments.
    """
    valid = ~np.isnan(val)
    if int(valid.sum()) < 8:
        return []
    med = _smooth(val, w)
    frames: list[int] = []
    ref = float(med[0])
    cand: float | None = None
    cstart = 0
    crun = 0
    for t in range(1, len(med)):
        if not valid[t]:
            cand = None
            continue
        if abs(med[t] - ref) <= band:
            cand = None
            continue
        if cand is None or abs(med[t] - cand) > band:
            cand, cstart, crun = float(med[t]), t, 1
        else:
            crun += 1
        if crun >= min_hold:
            ref = float(np.median(med[cstart : t + 1]))
            frames.append(cstart)
            cand = None
    return frames


def _dedup(frames, match_w: int) -> list[int]:
    """Sorted, de-duplicated frames merging any within ``match_w`` (keep first)."""
    out: list[int] = []
    for f in sorted({int(x) for x in frames}):
        if not out or f - out[-1] > match_w:
            out.append(f)
    return out


def note_onsets(
    freq_per_frame, gate_on, params: SegmentParams | None = None
) -> list[int]:
    """Note-onset frames for one voice's freq channel: pass-1 sustained-pitch-change origins unioned
    with the voice's gate-on retrigger frames, de-duplicated. ``freq_per_frame`` is the per-frame
    carry-forward 16-bit freq (None/0 when silent); ``gate_on`` the gate 0->1 frames."""
    params = params or SegmentParams()
    semis = _to_semitones(freq_per_frame)
    origins = pass1_origins(
        semis, params.freq_band, params.freq_w, params.freq_min_hold
    )
    return _dedup(list(origins) + [int(g) for g in gate_on], params.match_w)
