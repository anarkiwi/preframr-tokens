# AGENTS.md — resume the residual-SET drain

Goal (user directive, firm): drive residual raw SETs to **ZERO** across the HVSC
corpus, every mechanism modelled by a real named abstraction. **NO escape hatches**
(no lonely catch-all, no "irreducible" floor — uniqueness just means the generator
is unidentified). Work on branch `resid/ornament-and-digi`. Commit + merge
incremental PRs, but **do NOT release** (no PyPI tag, no version bump).

## Read first
- `RESIDUAL_SET_DRAIN_PLAN.md` — the mechanism map, the SubregPass nibble-lane root
  cause, the GRADIENT/INIT specs, and the live STATUS (drain numbers + what remains).
- `design/sid_driver_ornament_reference.md` (in the sibling `preframr-xpt` repo) —
  ground every outlier pattern here before inventing an encoding.

## Acceptance gate (never weaken)
`tests/test_residual_zero_corpus.py` parses a digi-excluded corpus sample through the
full `parse()` and asserts **0** residual SETs. It is kept **UNTRACKED** (listed in
`.gitignore`) so incremental PRs merge green while it still enforces locally — it is
currently RED (work in progress). Do NOT xfail/skip/mask it; do NOT commit it until
the count is genuinely 0. The work is done only when it passes.

## How to measure / iterate (baked image + local source)
```
# build the digi-excluded stride sample once (writes /tok/.resid_sample.txt)
docker run --rm -v $PWD:/tok -v /scratch/preframr:/scratch/preframr -v /tmp:/tmp:ro \
  -w /tok -e PYTHONPATH=/tok -e PREFRAMR_RESID_CORPUS=/scratch/preframr/hvsc \
  anarkiwi/preframr-xpt:0.2.18 bash -c \
  'python /tmp/build_sample.py; python /tmp/resid_precise.py $(cat /tok/.resid_sample.txt)'
```
- `/tmp/resid_precise.py` — precise per-shape bucketer (TRUST THIS).
- `residual_mechanism.py` (repo root, untracked) — mechanism census (its heuristic
  mislabels sparse held automation as `periodic_table`; cross-check with the bucketer).
- Tests: baked image, `PYTHONPATH=/tok`, `pytest tests/ -p no:cacheprovider
  --ignore=tests/test_residual_zero_corpus.py`. Lint forbids narrative `#` comments
  and docstrings >5 lines / with blank-line breaks. `black` before committing.

## What is landed (default-OFF research flags, register-state-exact, unit-tested)
- **GRADIENT** (`GRADIENT_OP=76`, `gradient_pass.py`, `test_gradient_pass.py`): staged
  (value,duration) automation curve — the Galway gradient envelope; flags
  modevol_/env_/filter_/ctrl_gradient. Drains MODEVOL fades + per-frame oscillation.
- **INIT** (`INIT_OP=77`, `init_pass.py`, `test_init_pass.py`): relabels
  pre-first-note-on SETs on the single-byte value regs as INIT; flag init_preamble.
  Drains the driver init routine + its SubregPass nibble-pairs.
- **ONSET_DEF** (no new op — reuses the CTRL_WT codebook; `ctrl_wavetable_pass.py`
  `_onset_def_claims`, flag `onset_def`): define-on-first. A single-reg instrument SET
  (ctrl/AD/SR/freq/PW/filter/modevol) written ONCE and unclaimed by the CTRL_WT
  recurrence phases is still a codebook entry — emit a lone CTRL_WT_DEF+STEP (the STEP
  re-emits byte-exactly). Scoped to one-write-per-frame writes at/after the first
  gate-rise onset (HARD_RESTART owns multiwrites, INIT the pre-onset preamble). The
  held_step bucket was NOT nibble-lane runs (the prior hypothesis) — diagnostics showed
  117/117 at note-ons, 83/117 single-reg, 68/86 runlen-1 singletons; the CTRL_WT
  recurrence floor (MINREP=2) structurally couldn't claim count-1 values. Lowering that
  floor to 1 for onset writes is the fix.
- **ENV_MULTILOAD** (no new op — reuses `HARD_RESTART_OP`; `passes.py` `HardRestartPass`,
  flag `env_multiload`): collapse a same-frame double write of one AD/SR envelope reg
  (gate-off + reload, the ADSR-bug workaround) into one HARD_RESTART_OP packing both
  bytes. The decoder is reg-generic, so it re-emits both writes in order (raw-write-order-
  exact). Runs in `PASSES` right before SubregPass (full bytes). Default OFF.

Result: sample 444 -> 215 -> 20 -> **11** residual SETs. Suite green (962 + 4 onset_def +
5 env_multiload passed, gate excluded; black/pylint 10.00/pyright clean).

## Next, in priority order (all real abstractions, NO catch-all)
The remaining 11 (digi-excluded stride sample): ~5 FREQ pre-onset preamble, ~6 nibble-lane
/ misc stragglers (held_step 4 / per_frame 2 on subreg 0/1).
1. **FREQ pre-onset preamble** (~5): reg0 freq written once at frames 1-4, BEFORE the first
   gate-on. User insight (2026-06-04): a freq change before any gate-on is INAUDIBLE — so
   these may be droppable as an AUDIO-exact (not register-state-exact) normalisation rather
   than needing combined-reg INIT. Audio unit test in `preframr-audio` pending to confirm.
2. **Nibble-lane stragglers** (~6): SR/AD held automation on subreg 0/1 (post-SubregPass).
   Every CTRL_WT phase filters subreg==-1, so they are unseen. Needs nibble-lane mining
   (decoder must re-emit the subreg write) — surface the design before building.

## Invariants
- Mine full bytes PRE-SubregPass (inline pass list, `reglogparser.py` ~988) where you
  can, so one logical write isn't counted as two nibble residuals.
- Every new atom: register-state-exact via `register_state` round-trip; new op across
  all 9 touchpoints (stfconstants/op_contracts/macro_contracts/state/decoders/
  reglogparser/pass/test); default-OFF; OUT of REGISTERED_MACROS (research arm — they
  are register-state/audio-exact but NOT raw-write-order-exact).
- Re-measure after each mechanism; expect downstream buckets to shift as nibble pairs
  collapse. Commit green increments; push; do not release.
