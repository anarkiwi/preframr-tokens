"""ALL-AXIS diversity-collapse probe (UNTRACKED, throwaway) for NEXT_BUILD_additive_instrument_model.md.

Across every SID value axis at once -- pitch, pulse-width, filter (cutoff+res), volume -- recover the
gestures by OPTIMAL MDL PARSE (mdl_parse: shortest path on the description-length cost DAG, the
LZ-optimal-parse / Knuth-Plass / Viterbi DP) rather than greedy thresholds, and measure the collapse:

  modDV   = distinct MODULATED values a naive per-frame tokenizer mints (the diversity)
  modGEN  = distinct gesture SHAPES the parse recovers (the structural alphabet; reused DEFs collapse)
  bits    = naive per-frame description length / structured (parsed) description length
  palette = set-point/note-index content (not diversity to collapse)
  literal = frames no gesture compresses = genuine per-frame sample data

Run:  python freq_collapse_probe.py [N] [seed]
"""

from __future__ import annotations

import glob
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from preframr_tokens.macros import pitch_grid
from mdl_parse import channel_report
from freq_residual_instrument_spike import per_frame_state

_POOL = sorted(glob.glob("/scratch/preframr/hvsc/**/*.dump.parquet", recursive=True))
_WIN = 3000  # frames analysed per tune
_SKIP = 150  # skip the silent/intro lead-in

_FREQ = {0: (0, 1), 1: (7, 8), 2: (14, 15)}
_PW = {0: (2, 3), 1: (9, 10), 2: (16, 17)}
_CTRL = {0: 4, 1: 11, 2: 18}
_KEYS = ("naive_bits", "struct_bits", "modDV", "modGEN", "palette", "literal")


def _freq_report(freq, ctrl):
    """Pitch axis: optimal-parse the raw voiced freq (wrap-aware). HOLD tokens are notes (folded to a
    note-index palette = content); RAMP tokens are slides/vibrato legs (the modulation that collapses
    to a few step shapes, however many freqs they sweep through)."""
    f = np.asarray(freq, dtype=np.int64)
    voiced = (f > 8) & ((np.asarray(ctrl, dtype=np.int64) & 1) == 1)
    if not voiced.any():
        return None
    tuning = pitch_grid.voice_tuning(f)
    pmap = lambda v: int(pitch_grid.note_index(np.array([v]), tuning)[0])  # noqa: E731
    return channel_report(np.where(voiced, f, 0), wrap=True, palette_map=pmap)


def analyse_tune(path):
    state = per_frame_state(pd.read_parquet(path))
    if len(state) < _SKIP + 300:
        return None
    state = state[_SKIP : _SKIP + _WIN]
    axes = defaultdict(lambda: {k: 0 for k in _KEYS})

    def add(axis, rep):
        if rep:
            for k in _KEYS:
                axes[axis][k] += rep[k]

    for v, (lo, hi) in _FREQ.items():
        add("pitch", _freq_report((state[:, hi] << 8) | state[:, lo], state[:, _CTRL[v]]))
    for v, (lo, hi) in _PW.items():
        add("pulse", channel_report(((state[:, hi] & 0x0F) << 8) | state[:, lo]))
    add("filter", channel_report((state[:, 22].astype(np.int64) << 3) | (state[:, 21] & 7)))
    add("filter", channel_report(state[:, 23]))
    add("volume", channel_report(state[:, 24] & 0x0F))
    return {k: dict(v) for k, v in axes.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sample = random.Random(seed).sample(_POOL, min(n, len(_POOL)))

    AXES = ["pitch", "pulse", "filter", "volume"]
    tot = {a: {k: 0 for k in _KEYS} for a in AXES}
    bit_ratios = []
    ok = 0
    for path in sample:
        try:
            res = analyse_tune(path)
        except Exception:  # noqa: BLE001
            continue
        if res is None:
            continue
        ok += 1
        tn = ts = 0
        for a in AXES:
            r = res.get(a)
            if r:
                for k in _KEYS:
                    tot[a][k] += r[k]
                tn += r["naive_bits"]
                ts += r["struct_bits"]
        bit_ratios.append(tn / max(1, ts))

    print(f"\nsampled {len(sample)} tunes, analysed {ok} (seed={seed}, window={_WIN}f)")
    print("MDL optimal-parse collapse  (palette = note/set-point content, not diversity)\n")
    print(f"{'axis':7} {'modDV':>7} {'modGEN':>7} {'divX':>6} {'bitsX':>6} {'palette':>8} {'literal':>8}")
    gdv = ggen = gn = gs = 0
    for a in AXES:
        t = tot[a]
        gdv += t["modDV"]; ggen += t["modGEN"]; gn += t["naive_bits"]; gs += t["struct_bits"]
        print(
            f"{a:7} {t['modDV']:>7} {t['modGEN']:>7} {t['modDV']/max(1,t['modGEN']):>5.1f}x "
            f"{t['naive_bits']/max(1,t['struct_bits']):>5.1f}x {t['palette']:>8} {t['literal']:>8}"
        )
    print(
        f"{'TOTAL':7} {gdv:>7} {ggen:>7} {gdv/max(1,ggen):>5.1f}x {gn/max(1,gs):>5.1f}x"
    )
    if bit_ratios:
        rs = sorted(bit_ratios)
        print(
            f"\nper-tune bits collapse: median {rs[len(rs)//2]:.1f}x  "
            f"min {rs[0]:.1f}x  max {rs[-1]:.1f}x"
        )


if __name__ == "__main__":
    main()
