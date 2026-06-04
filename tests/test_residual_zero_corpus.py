"""ACCEPTANCE GATE: residual raw SETs must be ZERO across a real-corpus sample (each is an
unmodeled mechanism; nothing is "irreducible"). Parses a deterministic corpus sample through
the full parse() (reparse=True, never the stale cache), digi-excluded, asserts 0 residual SETs,
reports outliers on failure. Strict -- NO threshold/xfail/skip-on-residual, human-only to weaken.
Corpus: $PREFRAMR_RESID_CORPUS (default /scratch/preframr/hvsc)."""

import glob
import os
import unittest

from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.tokenizer_config import default_tokenizer_args
from preframr_tokens.stfconstants import SET_OP

_RESIDUAL_ARM = (
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "skeleton_pass",
    "held_arp",
    "zero_plain",
    "slide_wide",
    "slide_landing",
    "stamp_pass",
    "sweep_pass",
    "sweep_loop",
    "pw_sweep",
    "filter_sweep",
    "wavetable_pass",
    "wt_short",
    "wt_oneshot",
    "patch_pass",
    "ctrl_osc",
    "modevol_gradient",
    "env_gradient",
    "filter_gradient",
    "ctrl_gradient",
    "note_off",
    "note_on",
    "ctrl_wavetable",
    "env_wavetable",
    "filter_wavetable",
    "modevol_wavetable",
    "freq_wavetable",
    "pw_wavetable",
    "onset_instrument",
)

_CORPUS = os.environ.get("PREFRAMR_RESID_CORPUS", "/scratch/preframr/hvsc")
_STRIDE = int(os.environ.get("PREFRAMR_RESID_STRIDE", "1500"))


def _is_digi(path):
    try:
        from preframr_tokens.dump_meta import meta_path_for, read_meta

        return bool(getattr(read_meta(meta_path_for(path)), "is_digi", False))
    except Exception:  # noqa: BLE001
        return False


def _residual_sets(df):
    ops = df["op"].to_numpy()
    regs = df["reg"].to_numpy()
    subs = df["subreg"].to_numpy() if "subreg" in df.columns else [-1] * len(df)
    vals = df["val"].to_numpy()
    out = []
    for i in range(len(df)):
        if int(ops[i]) == SET_OP and 0 <= int(regs[i]) < 25:
            out.append((int(regs[i]), int(subs[i]), int(vals[i])))
    return out


class TestResidualZeroCorpus(unittest.TestCase):
    def test_no_residual_sets_across_corpus_sample(self):
        allf = sorted(
            glob.glob(os.path.join(_CORPUS, "**", "*.dump.parquet"), recursive=True)
        )
        self.assertTrue(
            allf,
            f"residual-zero gate requires the corpus at {_CORPUS} "
            f"(set PREFRAMR_RESID_CORPUS); found none.",
        )
        sample = allf[::_STRIDE]
        args = default_tokenizer_args(seq_len=4096, **{f: True for f in _RESIDUAL_ARM})
        parser = RegLogParser(args)
        total = 0
        outliers = []
        for path in sample:
            if _is_digi(path):
                continue
            name = path.rsplit("/", 1)[-1]
            try:
                df = next(
                    parser.parse(path, max_perm=1, require_pq=False, reparse=True)
                )
            except StopIteration:
                continue
            except Exception as e:  # noqa: BLE001
                total += 1
                outliers.append((name, -1, f"PARSE CRASH: {type(e).__name__}: {e}"))
                continue
            resid = _residual_sets(df)
            if resid:
                total += len(resid)
                outliers.append((name, len(resid), resid[:6]))
        if total:
            outliers.sort(key=lambda x: -x[1])
            detail = "\n".join(
                f"  {n:6d}  {name}  sample(reg,subreg,val)={s}"
                for name, n, s in outliers[:25]
            )
            self.fail(
                f"{total} residual SETs (unmodeled mechanisms) across "
                f"{len(outliers)}/{len(sample)} sampled tunes -- model them, do not mask:\n{detail}"
            )


if __name__ == "__main__":
    unittest.main()
