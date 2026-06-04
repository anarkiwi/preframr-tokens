"""Bulk parse-equivalence over many real HVSC dumps (parallel).

Parses each dump path (from a file, one per line) through the FULL
RegLogParser.parse with reparse=True (no .parsed sidecar shortcut) under a
couple of configs, and writes one stable hash per (dump, config, rotation).
Diff the output across old/new reglogparser.py to prove byte-identical parse
on the real corpus.

Usage:
  PYTHONPATH=. python tools/bulk_equiv.py run <paths.txt> <out.txt> [workers]

Output lines (tab-separated): "<relpath>\t<config>\t<rotN|FILTERED|ERROR ...>\t..."
Order is non-deterministic (parallel); sort before diffing.
"""

import concurrent.futures
import hashlib
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])

from tests.parse_probes import parse_args  # noqa: E402
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402

ROOT = "/scratch/preframr/hvsc/"
CONFIGS = {
    "minimal": {},
    "rich": dict(
        skeleton_pass=True,
        trajectory_anchor_pass=True,
        freq_trajectory_pass=True,
        freq_onset_pass=True,
        loop_pass=True,
    ),
}
MAX_PERM = 1


def _hash(df):
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True)
    return hashlib.md5(h.values.tobytes()).hexdigest()[:16]


def parse_one(path):
    rel = path[len(ROOT) :] if path.startswith(ROOT) else path
    lines = []
    for cfg, over in CONFIGS.items():
        try:
            outs = list(
                RegLogParser(args=parse_args(**over)).parse(
                    path, max_perm=MAX_PERM, require_pq=False, reparse=True
                )
            )
        except Exception as e:  # noqa: BLE001
            lines.append(f"{rel}\t{cfg}\tERROR {type(e).__name__}")
            continue
        if not outs:
            lines.append(f"{rel}\t{cfg}\tFILTERED")
            continue
        for i, df in enumerate(outs):
            lines.append(
                f"{rel}\t{cfg}\trot{i}\t{df.shape[0]}x{df.shape[1]}\t{_hash(df)}"
            )
    return lines


def main():
    _, mode, paths_file, out_file = sys.argv[:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    assert mode == "run"
    paths = [ln.strip() for ln in open(paths_file) if ln.strip()]
    done = 0
    with open(out_file, "w") as out:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for lines in ex.map(parse_one, paths, chunksize=1):
                for ln in lines:
                    out.write(ln + "\n")
                done += 1
                if done % 50 == 0:
                    sys.stderr.write(f"{done}/{len(paths)}\n")
                    sys.stderr.flush()
    sys.stderr.write(f"done {done}/{len(paths)}\n")


if __name__ == "__main__":
    main()
