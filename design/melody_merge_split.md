# Melody-merge split — expose pitch as a standalone prediction target

**Status:** Design + impl in progress (branch `feat/melody-merge-split`). Tokens-side post-
encode pass; opt-in `--melody-merge-split` (default OFF); byte-exact round-trip preserved
(it only splits a Unigram merge into its already-decodable base atoms).

## Problem (measured on `freq_onset_channel_mini`)

After `FreqOnsetPass` re-tagged op0-SET-on-freq → FREQ_ONSET (op48), the freq-pitch tokens
**mostly did not land in op48 as standalone classes** (only 0.4% of stream) — instead they
**got absorbed into Unigram merged pieces** (merged share 33% → **54%**, +21 pp). Unigram
greedily merges frequent adjacent pairs (`FREQ_ONSET + DELAY`, `op45-V0 + FRAME`, …) into
compound vocab classes that the model **nails as compound shapes** (content-tier acc
0.076 → 0.249, op0 SET 0.154 → 0.831). But the *pitch* inside those compounds is no longer
a separable target — the model never has to predict it. So Unigram is optimising for byte
compression and **against melody learnability**: the pitch becomes invisible as a discrete
prediction class.

## The principled rule

Two kinds of merges containing melody atoms:
- **GOOD (keep):** merges whose decoded base atoms are *all* melody-pitch (consecutive
  `V0_HI + V0_LO`, a `FREQ_ONSET + FREQ_ONSET` melodic step), or *all* non-melody
  (`DELAY + FRAME` timing runs). These compress without hiding the target.
- **BAD (split):** merges that **cross the melody/non-melody boundary** (any merge whose
  decoded atoms contain *both* a melody-pitch atom and any other atom). Splitting these
  re-exposes pitch as a standalone class.

Defining "melody-pitch atom" precisely:
- `op == FREQ_TRAJ_OP (45)` and `reg in FREQ_TRAJ_REGS (0/7/14)` and
  `subreg in {FT_SUBREG_V0_HI(1), FT_SUBREG_V0_LO(2)}`, OR
- `op == FREQ_ONSET_OP (48)` and `reg in FREQ_TRAJ_REGS`, OR
- `op == FREQ_NUDGE_OP (47)` and `reg in FREQ_TRAJ_REGS` and
  `subreg in {FREQ_NUDGE_SUBREG_HI(2), FREQ_NUDGE_SUBREG_LO(3)}`.

Pure non-pitch op45 atoms (FLAGS / COUNT / PERIOD / DELTA shape) are **not** melody under
this rule — they are trajectory structure, and merging them with surrounding non-melody
tokens is fine.

## Mechanism (post-encode split, simplest + trainer-agnostic)

A new pure function and a `RegTokenizer` method:

```python
def split_cross_boundary_merges(seq, decode_fn, is_melody_fn, n_atoms) -> np.ndarray:
    """For each merged id (>= n_atoms), decode to base ids; if mixed melody + non-melody,
    emit the base ids. Pure-melody and pure-non-melody merges are kept."""
```

Wire into `Corpus.encode_and_save_cached_blocks` immediately after `seq =
self.tokenizer.encode(n)`, gated by `args.melody_merge_split`. Default OFF ⇒ identity ⇒
byte-identical to today. Cost: per-block, one decode of each merge (only at tokenize time;
not at training). Stream-length cost depends on how many merges are cross-boundary —
expected to be a meaningful chunk (the 21 pp jump in merged-share that hid the pitch).

**Why this is the right seam:**
- Tokenize-time (not load-time): the cached `.blocks.npy` already contains the split — no
  repeat work in the dataloader / no model-side change.
- Post-Unigram (not vocab-time): we keep Unigram's compression for melody-only and
  non-melody-only runs; we only un-merge the boundary-crossing pieces. No trainer change.
- Lossless: decode is the inverse of merge in the Unigram vocab. Round-trip oracle stays
  green.

## Why this should produce a signal at mini

Diagnosis so far on melody at mini (six A/Bs, V0-onset ≈ 0):
- **Predictable** (trigram 0.79–0.82) — not aleatoric.
- **Capacity is there** (freq_core SET 0.42; freq_onset_channel SET 0.83).
- **Not rare** (~13.4% of stream).

If the model has capacity and the target is predictable but the model still scores 0, the
remaining candidate cause is **the pitch isn't a separable target** — it's hidden inside
compound vocab classes the model nails as shapes. De-merging directly tests that. If V0-onset
moves off 0 at mini, the merge-hiding diagnosis is correct and the lever is real (sweep at
prodlike). If it stays flat — the structural-blocker story is wrong and we pivot the success
metric (distributional / perceptual; `music_llm_landscape_and_fail_fast_plan.md`).

## A/B

`melody_merge_split_mini`: anchored + interval + freq-onset-channel + onset-loss-weight
(the current best stack from `freq_onset_channel_mini`'s target) **+ `--melody-merge-split`**
vs the same stack without. 3 seeds, mini body=large, `:0.2.10`. The arms tokenize
differently (the split changes the stream), so cache keys split as designed. Read via
`audit.content_tier_report --onset` on the unified `melodic_onset_bucket` (op45 V0 + op48 +
op47 pitch). Decisive number = V0-onset acc on the split arm.

## Files
- **`regtokenizer.py`**: `split_cross_boundary_merges` (pure) + `RegTokenizer.split_melody_merges`
  (instance wrapper using `self.tkmodel.decode` + `self.tokens.iloc`).
- **`stfconstants.py`**: optional `MELODY_PITCH_OP_SUBREGS` constant (or compute in regtokenizer).
- **`corpus.py`**: call `split_melody_merges` in `encode_and_save_cached_blocks` when
  `args.melody_merge_split`.
- **`preframr/args.py`** (framework): `--melody-merge-split` BooleanOptionalAction default
  False; NOT in `_PIPELINE_NAME_TO_FLAG` (a tokenize-step modifier, per-arm via extra_cargs).
- Bump tokens `fallback_version` 0.27.0 → 0.28.0; framework VERSION 0.2.9 → 0.2.10; framework
  floor `preframr-tokens>=0.28.0`.

## Tests
- **Pure-function unit:** synthetic decode/is_melody stubs; cross-boundary merge → split,
  pure-melody → kept, pure-non-melody → kept, base atom → kept.
- **Byte-exact round-trip:** with `melody_merge_split` ON, a stream containing isolated
  freq writes round-trips identical per-frame values.
- **Default OFF ⇒ identity:** byte-identical stream to today.
