"""CLI: FREQ/PW/FC trajectory coverage over a parsed corpus — how much motion the
structural FREQ_TRAJ op captures vs mop-ups. Run as
``python -m preframr_tokens.trajectory_coverage``."""

from __future__ import annotations

import argparse

from preframr_tokens.audit_primitives import trajectory_coverage as _coverage
from preframr_tokens.blocks import glob_dumps
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_profile import DEFAULT_CORPUS, _build_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("-n", "--sample", type=int, default=20)
    parser.add_argument("--config", default="full_macros")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--tier", default="freq", choices=("freq", "pw", "fc"))
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args(argv)
    dumps = glob_dumps(
        opts.corpus, max_files=opts.sample, require_pq=False, seed=opts.seed
    )
    rlp = RegLogParser(args=_build_args(opts.config, opts.set))
    structural = 0
    mopup = 0
    segments = 0
    run_len_acc = 0.0
    alt_acc = 0.0
    songs = 0
    for path in dumps:
        xdf = next(rlp.parse(path, max_perm=1, require_pq=False, reparse=True), None)
        if xdf is None or not len(xdf):
            continue
        cov = _coverage(xdf, tier=opts.tier)
        structural += cov["structural_atoms"]
        mopup += cov["mopup_atoms"]
        segments += cov["n_segments"]
        run_len_acc += cov["run_length_mean"] * cov["n_segments"]
        alt_acc += cov["alternation_mean"] * cov["n_segments"]
        songs += 1
    denom = structural + mopup
    print(f"tier={opts.tier} songs={songs} segments={segments}")
    print(f"  structural_atoms={structural} mopup_atoms={mopup}")
    print(f"  captured_frac={structural / denom if denom else 0.0:.3f}")
    print(f"  run_length_mean={run_len_acc / segments if segments else 0.0:.2f}")
    print(f"  alternation_mean={alt_acc / segments if segments else 0.0:.3f}")


if __name__ == "__main__":
    main()
