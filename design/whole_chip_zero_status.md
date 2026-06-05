# Whole-chip residual-zero: status + the irregular 23/24 tail finding

**Audience:** an engineer/agent working entirely inside `preframr-tokens` (+ the HVSC corpus at
`/scratch/preframr/hvsc`). Records what the driver-mechanism residual-zero arc landed (PRs #59, #60), what
remains, and the design decision blocking literal zero on Res/Filt(23) + Mode/Vol(24).

## One-line state

Every **non-freq driver register class is at raw-`SET` zero** in the deployed default
(`REGISTERED_MACROS`) **except the irregular one-off tail on Res/Filt(23) + Mode/Vol(24)**; the
escape-hatch op family is gone (soft residual `0`); freq is left as raw `SET` (the pitch slate). Driving
the 23/24 tail to *literal* zero needs a ref-per-write codebook that re-introduces the relabel anti-pattern
PR #59 deleted, so it is deferred to a deliberate decision (below), not shipped silently.

## What landed

### PR #59 -- drains folded in + escape-hatch family deleted
- Added `sweep_pass`, `pw_sweep`, `filter_sweep`, `filter_gradient` to `REGISTERED_MACROS`
  (all `arbitrate(validate=True)`), draining **PW / CTRL / AD / SR / filter-cutoff(21/22)** raw `SET` to `0`.
- Deleted the `freq_nudge` / `freq_onset` / `release_update` / `ctrl_update` / `lonely_validator` passes and
  the `lonely_catch_all` / `strict_lonely` flags -- byte-exact relabels of un-modelled writes that made a
  raw-`SET` residual gate lie. ctrl/AD/SR catch-alls were already dead (`InstrumentProgramPass` owns them);
  `freq_nudge` reverts to raw `SET`, exposing freq as the honest slate. Ops 47/48/49/51 retired as holes.

### PR #60 -- `GlobalOscPass` (op `GLOBAL_OSC=82`, flag `global_osc`)
- A `SWEEP` twin (recovered from the deleted `CtrlOscPass`) generalised to regs **21/22/23/24**: mines a
  per-frame period-`P` (`2<=P<=8`) cycle into one register-exact atom (`PERIOD` + P cycle bytes + `LEN`)
  draining `cycle[k % P]` per frame, chunked to `<= GLOBAL_OSC_MAX_SPAN` so `_cap_delay`/consolidation can't
  break the replay. Byte-exact via `arbitrate(validate=True)`; runs **after** `GradientPass` (takes only its
  leftover). Drains the periodic(P=2..8) Mode/Vol + Res/Filt wobble.

### `modevol_gradient` deliberately NOT enabled
The work order's A1 listed `modevol_gradient`, but it is a **pre-existing `GradientPass` byte-exactness
gap**: large-`DUR` gradient stages leave empty-frame gaps that `_cap_delay` coarsens, breaking the per-frame
drain (`SweepPass`/`GlobalOscPass` chunk to `MAX_SPAN` to avoid exactly this; `GradientPass` does not). It
corrupts the **gated** MODE_VOL/CTRL/AD/SR registers on baseline-clean tunes (e.g. `Superhero.7`) despite
`validate=True` (validate runs pre-consolidation). Verified identical on `origin/main` -- not introduced
here. The hard "byte-exact always" guardrail overrode the literal flag list; MODE_VOL drainage moved to the
oscillation/codebook track instead.

## Verified residual (census, corpus, `reparse=True`, step-50 = 1609 tunes)

```
prod arm (REGISTERED_MACROS):  raw_SET soft_residual=0
  FREQ      422306   (the slate -- raw SET, intentionally NOT drained)
  MODEVOL    25294   (24 tail)
  RESFILT     2303   (23 tail)
  PW/CTRL/AD/SR/FC = 0
```

23/24 residual by mechanism: the bulk is **`step_hold` + `irregular`** (isolated / short single writes),
not periodic. `GlobalOscPass` removes the periodic share; the periodic-classified remainder is short runs
that fail the `>= GLOBAL_OSC_MIN_LEN` / start-on-written-frame / `validate` guards.

## Byte-exactness

The project's lossless contract is `sid_frame_diff.diff_dump_vs_pipeline`: **CTRL/AD/SR/RES_FILT(23)/
MODE_VOL(24) byte-exact + FREQ within cent tolerance, frame-offset aligned**. PW and filter-cutoff(21/22)
are **not** in `EXACT_REGS` -- not gated. Against that oracle the merged default has **0 gated-register
regressions on baseline-clean tunes** and improves losslessness (334 vs 208 fully-ok of 1599 sampled).
`pw_sweep`/`filter_sweep` do shift PW/FC register-state decode (pre-existing sweep validate-gap under
consolidation) but those regs are ungated, and draining them is the explicit goal. CI green py3.10/3.11/3.12.

## The blocker for literal 23/24 zero (decision needed)

The `step_hold`/`irregular` 23/24 tail is genuine **one-off** writes (a single Mode/Vol or Res/Filt change
held until the next, no run to mine). The only way to take them to raw-`SET` zero is a **define-on-first
global-reg codebook**: intern each distinct (reg,val) and emit a DEF on first sight + a REF thereafter.

That is exactly the **relabel anti-pattern PR #59 just deleted**: a ref-per-write op that makes a raw-`SET`
gate read clean while modelling nothing (the `lonely_catch_all` critique -- "a gate that lies"). For
genuinely recurring global-reg states a codebook is legitimate (it captures structure, like
`InstrumentProgramPass`); for non-recurring one-offs it is pure relabelling. So the call is:

1. **Accept the current floor:** all *structured* (periodic + run + staged) non-freq automation drained
   byte-exact; the irregular one-off 23/24 writes stay as honest raw `SET`. Driver-mechanism zero is met;
   the residue is content-like single writes, not a driver mechanism.
2. **Add the codebook** to reach literal zero, accepting it relabels one-off writes (and re-opens the
   "gate that lies" critique for the non-recurring fraction). Template: `InstrumentProgramPass` (a
   register_state-guarded define-on-first codebook with literal fallback).

Recommendation: option 1, unless a downstream consumer specifically needs the 23/24 raw-`SET` count at
literal zero. If option 2, scope the codebook to **recurring** (`>= 2` occurrences) global-reg states only,
so it stays a structural model, not a per-write relabel, and leave the truly-singleton writes literal.
