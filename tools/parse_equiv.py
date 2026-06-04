"""Parse-path equivalence harness.

Runs ``RegLogParser.parse`` over a fixed fixture set (real cached HVSC driver
dumps + deterministic synthetic edge-case dumps) under several encoding
configs, and prints one stable hash per (fixture, config, rotation). Diff the
output across a code change to prove the parse output is byte-identical (or to
characterise exactly what moved).

Usage:
    PYTHONPATH=. PREFRAMR_SID_FIXTURE_CACHE=/scratch/preframr/sid_fixture_cache \
        python tools/parse_equiv.py            # all fixtures, all configs
    ... python tools/parse_equiv.py --synthetic-only

The real dumps are read directly from the fixture cache (no Docker needed when
already cached); missing ones are skipped with a SKIP line so a partial cache
still yields a usable diff.
"""

import argparse
import hashlib
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.parse_probes import DumpBuilder, parse_args, write_dump  # noqa: E402
from preframr_tokens.macros.skeleton_pass import LUT  # noqa: E402
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402

CACHE = os.environ.get(
    "PREFRAMR_SID_FIXTURE_CACHE", "/scratch/preframr/sid_fixture_cache"
)

# Real cached driver dumps: diverse SID drivers (Hubbard / DRAX / Daglish /
# Goto80 / Jammer), the strongest available exercise of the parse path.
REAL_DUMPS = [
    "commando_20_20s.dump.parquet",
    "camerock_20_20s.dump.parquet",
    "trap_20_20s.dump.parquet",
    "baggis_20_20s.dump.parquet",
    "grid_runner_26s.dump.parquet",
    "grid_runner_head.dump.parquet",
]

# Encoding configs spanning distinct pass chains.
CONFIGS = {
    "minimal": dict(),
    "skeleton": dict(skeleton_pass=True, trajectory_anchor_pass=True),
    "freqfam": dict(
        freq_trajectory_pass=True, freq_onset_pass=True, trajectory_anchor_pass=True
    ),
    "loop": dict(skeleton_pass=True, trajectory_anchor_pass=True, loop_pass=True),
}


def _synthetic_dumps(tmp):
    """Deterministic edge-case dumps targeting the strip / rotation / combine
    paths. Each returns a written ``.dump.parquet`` path."""
    out = {}

    # Plain held melody across 3 voices' worth of pitch motion (rotation path).
    b = DumpBuilder().adsr(ad=0x00, sr=0xF0).pw(0x800)
    for note in (60, 62, 64, 65, 67, 64, 60, 62, 65, 69, 72, 71, 69, 67):
        b.note([LUT[note]] * 6)
    out["melody"] = write_dump(b, os.path.join(tmp, "melody.dump.parquet"))

    # Vibrato + slide + arp: exercises freq combine + cent-quantize + trajectory.
    import numpy as np

    b = DumpBuilder().adsr(ad=0x10, sr=0xC0).pw(0x400)
    for note in (60, 62, 64):
        b.note([LUT[note]] * 5)
    b.note([LUT[60 + (0 if f % 2 == 0 else 12)] for f in range(8)])  # arp
    b.note([cents_fn(67, 20.0 * float(np.sin(f))) for f in range(8)])  # vibrato
    b.note([LUT[60 + min(7, f)] for f in range(8)])  # slide
    out["ornament"] = write_dump(b, os.path.join(tmp, "ornament.dump.parquet"))

    # Long held tail -> trailing empty frames feed the trailing-marker strip.
    b = DumpBuilder().adsr(ad=0x00, sr=0xF0).pw(0x800)
    for note in (60, 64, 67, 72):
        b.note([LUT[note]] * 8)
    b.ctrl(0x40)  # gate off, then long silence
    for _ in range(40):
        b.frame()
    out["longtail"] = write_dump(b, os.path.join(tmp, "longtail.dump.parquet"))

    return out


def cents_fn(note, cents):
    from tests.parse_probes import cents_to_fn

    return cents_to_fn(note, cents)


def _hash_df(df):
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True)
    return hashlib.md5(h.values.tobytes()).hexdigest()[:16]


def _run(path, label):
    for cfg_name, over in CONFIGS.items():
        try:
            parser = RegLogParser(args=parse_args(**over))
            outs = list(
                parser.parse(path, max_perm=99, require_pq=False, reparse=True)
            )
        except Exception as e:  # noqa: BLE001
            print(f"{label:18s} {cfg_name:9s} ERROR {type(e).__name__}: {e}")
            continue
        if not outs:
            print(f"{label:18s} {cfg_name:9s} (filtered, 0 rotations)")
            continue
        for i, df in enumerate(outs):
            print(
                f"{label:18s} {cfg_name:9s} rot{i} "
                f"{df.shape[0]:5d}x{df.shape[1]} {_hash_df(df)}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--real-only", action="store_true")
    a = ap.parse_args()

    if not a.synthetic_only:
        for name in REAL_DUMPS:
            p = os.path.join(CACHE, name)
            if not os.path.exists(p):
                print(f"{name:18s} SKIP (not in cache {CACHE})")
                continue
            _run(p, name.replace(".dump.parquet", ""))

    if not a.real_only:
        with tempfile.TemporaryDirectory() as tmp:
            for label, p in _synthetic_dumps(tmp).items():
                _run(p, "syn:" + label)


if __name__ == "__main__":
    main()
