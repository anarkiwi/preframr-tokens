"""CLI: profile per-op atom usage of a parsed corpus under a tokenizer config.
Run as ``python -m preframr_tokens.tokenizer_profile``; ``--compare A B`` prints
the net per-op atom delta between two named configs on the same sample."""

from __future__ import annotations

import argparse
from collections import Counter

from preframr_tokens.audit_primitives import op_atom_profile
from preframr_tokens.blocks import glob_dumps
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import default_tokenizer_args, named_config

DEFAULT_CORPUS = "/scratch/preframr/training-dumps/**/*.dump.parquet"


def _coerce(value):
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _build_args(config, sets):
    overrides = {}
    for spec in sets or []:
        key, val = spec.split("=", 1)
        overrides[key] = _coerce(val)
    if config:
        return named_config(config, **overrides)
    return default_tokenizer_args(**overrides)


def profile_corpus(args_ns, corpus, sample, seed=0):
    """Aggregate op_atom_profile over a sampled corpus parsed with ``args_ns``."""
    dumps = glob_dumps(corpus, max_files=sample, require_pq=False, seed=seed)
    parser = RegLogParser(args=args_ns)
    op_hist = Counter()
    per_tier = Counter()
    total_atoms = 0
    total_frames = 0
    songs = 0
    skipped = 0
    for path in dumps:
        try:
            xdf = next(
                parser.parse(path, max_perm=1, require_pq=False, reparse=True), None
            )
        except Exception:  # pylint: disable=broad-except
            skipped += 1
            continue
        if xdf is None or not len(xdf):
            continue
        prof = op_atom_profile(xdf)
        op_hist.update(prof["op_hist"])
        per_tier.update(prof["per_tier"])
        total_atoms += prof["total_atoms"]
        total_frames += prof["n_frames"]
        songs += 1
    return {
        "songs": songs,
        "skipped": skipped,
        "total_atoms": total_atoms,
        "atoms_per_song": total_atoms / songs if songs else 0.0,
        "atoms_per_frame": total_atoms / total_frames if total_frames else 0.0,
        "op_hist": dict(op_hist),
        "per_tier": dict(per_tier),
    }


def _print_profile(prof):
    print(
        f"songs={prof['songs']} skipped={prof.get('skipped', 0)} "
        f"atoms={prof['total_atoms']} "
        f"atoms/song={prof['atoms_per_song']:.1f} "
        f"atoms/frame={prof['atoms_per_frame']:.3f}"
    )
    total = prof["total_atoms"] or 1
    for op, count in sorted(prof["op_hist"].items(), key=lambda kv: -kv[1]):
        print(f"  op {op:>3}: {count:>8} ({100 * count / total:5.1f}%)")
    print("  per-tier:", prof["per_tier"])


def _print_compare(prof_a, prof_b, name_a, name_b):
    print(f"{name_a}: {prof_a['total_atoms']} atoms over {prof_a['songs']} songs")
    print(f"{name_b}: {prof_b['total_atoms']} atoms over {prof_b['songs']} songs")
    print(f"net atom delta: {prof_b['total_atoms'] - prof_a['total_atoms']:+d}")
    ops = set(prof_a["op_hist"]) | set(prof_b["op_hist"])
    for op in sorted(ops):
        da = prof_a["op_hist"].get(op, 0)
        db = prof_b["op_hist"].get(op, 0)
        if da != db:
            print(f"  op {op:>3}: {da:>8} -> {db:>8} ({db - da:+d})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("-n", "--sample", type=int, default=20)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args(argv)
    if opts.compare:
        name_a, name_b = opts.compare
        prof_a = profile_corpus(
            named_config(name_a), opts.corpus, opts.sample, opts.seed
        )
        prof_b = profile_corpus(
            named_config(name_b), opts.corpus, opts.sample, opts.seed
        )
        _print_compare(prof_a, prof_b, name_a, name_b)
        return
    _print_profile(
        profile_corpus(
            _build_args(opts.config, opts.set), opts.corpus, opts.sample, opts.seed
        )
    )


if __name__ == "__main__":
    main()
