# Residual-SET drain: mechanism map and model plan

Goal: ZERO residual raw SETs (op==SET_OP, reg 0..24) across the corpus, every
mechanism named by a real abstraction. Gate: `tests/test_residual_zero_corpus.py`
(non-maskable). Census tool: `residual_mechanism.py` (heuristic, mislabels sparse
held automation as `periodic_table`). Precise bucketer: `/tmp/resid_precise.py`.

## Pipeline fact that governs everything: SubregPass nibble-split

`SubregPass` (last pass in `macros.PASSES` → `run_passes`, reglogparser ~1043)
ALWAYS splits registers 4,5,6,23,24 (`SUBREG_REGS` = ctrl/AD/SR/res-filt/modevol)
byte→nibble: low nibble → subreg=0, high nibble → subreg=1 (SUBREG_FLUSH op keeps
intra-frame byte-equality). AFTER SubregPass these registers have NO `subreg==-1`
SETs — only nibble lanes.

Consequence: every per-reg model that filters `subreg == -1` (the state codebook
`CtrlWavetablePass`, the oscillation `CtrlOscPass`, `SweepPass`) is BLIND to these
registers once SubregPass has run. The inline pre-rotation pass list (reglogparser
~984) runs codebook/osc/sweep BEFORE SubregPass on full bytes, so it catches the
*recurring full-byte* states there (e.g. reg24 init values become CTRL_WT_DEF/SET).
What survives to residual on regs 4,5,6,23,24 is whatever those full-byte models
did NOT claim — then it gets nibble-split, so one surviving logical write becomes
TWO residual SETs (lo+hi lane). This inflates `multiwrite_sameframe` counts.

A post-SubregPass model MUST mine the nibble lanes (key on `(reg, subreg)`), or run
pre-SubregPass on full bytes. The decoder is already reg/subreg-generic.

## Precise residual breakdown (57-tune digi-excluded stride-1500 sample, 444 SETs)

| bucket | n | % | mechanism |
|--------|---|---|-----------|
| multiwrite_sameframe | 143 | 32% | nibble-pair duplicates of unmodeled ctrl/env writes (1 logical → 2 counted) + true hard-restart double AD/SR load |
| held_step_automation | 137 | 31% | sparse held value curves on SR/MODEVOL/AD/CTRL/filter (writes many frames apart, value held via DELAY) |
| init_startup | 130 | 29% | one-time driver init routine (RESFILT/MODEVOL/FREQ at startup, value set once) |
| per_frame_dense | 31 | 7% | real MODEVOL per-frame oscillation/sweep (gap=1) |
| recurring_value | 3 | 0.7% | codebook near-miss |

`recurring_value` ~0 confirms the inline full-byte codebook already drains true
recurrence. The work is the three mechanisms below — NONE is an escape hatch.

## What was tried and REVERTED

Generalizing `CtrlOscPass` to env/filter/modevol (`env_osc`/`filter_osc`/
`modevol_osc`) drained ~0: MODEVOL/env residual is held-step automation (ramps/
fades, e.g. Winter_Events reg24 13→1), not periodic — osc correctly rejects it.
The post-SubregPass run also can't see these regs (nibble lanes). Reverted.

## Models to build (priority by leverage, all real abstractions)

### A. Held-step automation curve (covers held_step 137 + per_frame_dense 31 = 168)
A run of held-step SETs on a modulation lane (modevol/env/filter/ctrl) IS an
automation curve the driver steps through. Model as a curve codebook: DEF a
`(value, hold_frames)` step sequence once, REF it (reused curves share a DEF →
learnable induction, generalizable). The periodic case (`per_frame_dense`, hold=1)
is the degenerate sub-case CTRL_OSC already handles; this generalizes it to
arbitrary held steps. Mine pre-SubregPass on full bytes (inline list) so a logical
curve is one atom, not nibble pairs. SWEEP requires writes 1 frame apart and so
cannot model held curves; this is the missing twin.

### B. INIT preamble atom (covers init_startup 130)
The driver's init routine sets filter/master-vol/freq once at startup and never
recurs. Bundle the song's initial one-time register configuration into a named
INIT atom (a preamble), not a catch-all — it is the driver init mechanism. One-time
so the codebook (MINREP=2) cannot claim it; needs its own atom.

### C. hard-restart multiwrite (the non-nibble remainder of multiwrite 143)
Double AD/SR (and ctrl) write in one frame = hard-restart gate-off + reload.
`HardRestartPass` exists; extend it to bundle the multiload. Much of bucket C is
nibble-pair duplicates that A/B will collapse — re-measure after A/B before sizing.

## Execution order
1. Build A (held-step automation codebook). Re-measure.   [DONE — GRADIENT atom]
2. Build B (INIT preamble). Re-measure.                   [DONE — INIT atom]
4. Held_step singletons-at-onsets (NOT nibble-lane runs). [DONE — ONSET_DEF define-on-first]
3. Extend C (hard-restart multiload) for the remainder.   [DONE — ENV_MULTILOAD, reuses HARD_RESTART_OP]
6. FREQ pre-onset preamble (inaudible).                   [DONE — PRE_GATE_FREQ drop/relocate, audio-exact]
7. Nibble-lane SR/AD held (subreg 0/1).                   [DONE — NIBBLE_WAVETABLE, CTRL_WT lane on DEF subreg]
5. Gate green → PR (NO release).

## STATUS (2026-06-04): 444 -> 215 -> 20 -> 11 -> 6 -> 0 residual SETs on the sample (DONE)

NIBBLE_WAVETABLE (CtrlWavetableNibblePass `nibble_wavetable`) drained the final 6 -> 0. Post-SubregPass,
it mines surviving subreg-0/1 nibble SETs into the CTRL_WT codebook keyed on (reg, subreg, val):
recurring -> DEF + SET reuse, once-only -> lone define-on-first DEF. The nibble lane rides on the DEF
subreg (new CTRL_WT_SUBREG_ID_NIB0/NIB1); _CtrlWtCodec stores (lane, val) and re-emits the nibble via
the SetDecoder merge. Register-state-exact (arbiter validates). The work order's sample acceptance gate
(residual SETs == 0) is MET; next is the full-corpus census + xpt RUNBOOK release. The drain stack:
GRADIENT, INIT (prior) + ONSET_DEF, ENV_MULTILOAD, PRE_GATE_FREQ (audio-exact), NIBBLE_WAVETABLE.

## STATUS (prior): 444 -> 215 -> 20 -> 11 -> 6 residual SETs on the sample (-98.6%)

PRE_GATE_FREQ (PreGateFreqPass `pre_gate_freq` flag) drained 11 -> 6. A freq written before a
voice's first gate-on is inaudible (preframr-audio test); the user-directed rule DROPs it when the
first gated note sets its own freq, else RELOCATEs it into the gate-on frame for the onset/skeleton
macros. The FIRST audio-exact (not register-state-exact) drain atom -- in parse_audit._LOSSY_RESETS,
default OFF; the audible region (gate-on onward) is preserved (unit-tested). Remaining 6: all SR/AD
held on subreg 0/1 nibble lanes (CTRL_WT phases filter subreg==-1, so unseen) -- nibble-lane mining,
design to be surfaced.

## STATUS (prior): 444 -> 215 -> 20 -> 11 residual SETs on the sample (-97.5%)

## STATUS (prior): 444 -> 215 -> 20 residual SETs on the sample (-95%)

ONSET_DEF (CTRL_WT define-on-first, flag `onset_def`) drained 215 -> 20. The held_step
bucket (117) was NOT post-SubregPass nibble-lane runs as item 4 hypothesised: diagnostics
showed 117/117 at note-ons, 83/117 single-reg, 68/86 runlen-1 singletons. The CTRL_WT
recurrence phases all gate on MINREP=2, so a value written once is never claimed;
define-on-first lowers that floor to 1 for onset-co-located single-reg instrument writes
(lone DEF, no REF). It also collapsed the nibble-pair-inflated multiwrites (46->6) and
post-onset init writes (44->5). Remaining 20: ~5 FREQ pre-onset preamble (combined-reg
INIT), ~6 true same-frame double-loads (hard-restart), ~9 stragglers. No new op/family/
decoder; register-state-exact via the arbiter; default-OFF; OUT of REGISTERED_MACROS.

## STATUS (prior): 444 -> 215 residual SETs on the sample (-52%)

Landed on `resid/ornament-and-digi` (default-OFF research flags, NOT in
REGISTERED_MACROS, register-state-exact, unit-tested, suite green):
- **GRADIENT atom** (`GRADIENT_OP=76`, `gradient_pass.py`): drains MODEVOL fades +
  per-frame oscillation (held_step MODEVOL 30->7, per_frame 31->8). Flags
  modevol_/env_/filter_/ctrl_gradient. Mines full bytes pre-SubregPass.
- **INIT atom** (`INIT_OP=77`, `init_pass.py`): relabels pre-first-note-on SETs on
  the single-byte value regs (`SUBREG_REGS`) as INIT; note-on boundary from
  register_state (robust to note atoms); runs LAST over surviving SETs only (running
  first drifts frame consolidation -> lossy). Flag init_preamble. Drains
  init_startup 124->44 AND multiwrite 135->46 (init nibble-pairs vanish: INIT
  relabels before SubregPass, so SubregPass skips them).

Remaining 215: held_step 117 (SR/AD/CTRL sparse + post-SubregPass NIBBLE LANES,
subreg 0/1 -- gradient is subreg==-1 only, needs a nibble-lane variant keyed on
(reg,subreg), OR the writes are isolated singletons not in runs of >=3), init 44
(mostly FREQ 30 -- combined 16-bit regs excluded from INIT; need combined-reg-aware
init), multiwrite 46 (hard-restart double AD/SR load -> extend HardRestartPass),
per_frame_dense 8.

Acceptance gate `tests/test_residual_zero_corpus.py` is kept UNTRACKED (in
.gitignore) so incremental PRs merge green while it still enforces 0 locally; do
NOT commit it until the count is actually 0.

## Driver grounding (design/sid_driver_ornament_reference.md)

Model A is NOT a foreign mechanism. The reference's **mechanism (B) — parametric/
table sweep in the value domain** covers vibrato/PW/filter/volume, and states
"**one bounded-sweep / gradient primitive family generalizes across them, as
Galway's shared envelope shows**." Galway uses ONE structure — `G0..G3` values +
`D0..D3` durations + delay — for vibrato, PW AND filter. A held-step automation
curve (volume fade, filter gradient, env automation) is that **gradient envelope**:
a sequence of `(value, duration)` stages. `SweepPass` already implements the
per-frame (delta/frame) form for freq/PW/filter (reg 21 cutoff only); the gap is
(a) the STAGED/held form and (b) the volume (reg24) and env (reg5/6) domains.
RESID in the reference = "lossless escape for content no primitive models YET ...
not a floor" — matches the zero-residual directive exactly.

## A. GRADIENT atom — executable spec

New default-OFF research atom modeled on CTRL_OSC (the closest analog: per-frame
drain into `pending_set_writes`). Mine pre-SubregPass on full bytes (inline pass
list, reglogparser ~984, so a logical curve is ONE atom, not nibble pairs). Keep
default-OFF and OUT of `_RESIDUAL_ARM`/REGISTERED_MACROS until its own tests pass
AND it drains in the census — so every intermediate commit stays green.

Touchpoints (mirror every CTRL_OSC site):
- `stfconstants.py`: `GRADIENT_OP = 76` (76 is free; NOTE_ON_OP=75 is current max —
  re-verify). Subregs: `GRADIENT_SUBREG_NSTAGES=0`, `GRADIENT_SUBREG_END=1`,
  `GRADIENT_SUBREG_VAL_BASE=2`, `GRADIENT_SUBREG_DUR_BASE=2+MAX_STAGES`. Limits:
  `GRADIENT_MAX_STAGES` (e.g. 16), `GRADIENT_MIN_STAGES=2`, `GRADIENT_MAX_SPAN` for
  per-atom DELAY-cap re-anchoring (see CTRL_OSC `_chunk_starts`).
- `macros/gradient_pass.py` (new, mirror `ctrl_osc_pass.py`): `GATE_FLAGS =
  {"modevol_gradient","env_gradient","filter_gradient","ctrl_gradient"}`;
  `_target_regs(args)` unions reg classes (MODE_VOL_REG; AD/SR_REGS_BY_VOICE;
  `_FILTER_REGS`; CTRL_REGS_BY_VOICE). `_collect_writes` = CTRL_OSC's (plain SETs,
  subreg==-1, real-frame index). Mining: a maximal run of >= MIN_STAGES SETs on one
  reg, NON-note-aligned (gradients persist across notes), each stage = (val, hold)
  where hold = frames until the next write (>=1; a per-frame oscillation degenerates
  to hold=1, subsuming per_frame_dense). Emit one Claim per run (priority below
  SWEEP so true constant-delta sweeps win first), `validate=True`. Tile to
  GRADIENT_MAX_SPAN/MAX_STAGES chunks each re-anchored on a written row.
- `macros/decoders.py` (new `GradientDecoder`, mirror `CtrlOscDecoder`): NSTAGES
  opens `pending_gradient={reg,fields:{}}`; collect VAL_BASE+i / DUR_BASE+i; END
  drains `for i in range(nstages): pending_set_writes[reg] += [val_i]*dur_i` after
  `maybe_flush_for(reg,-1)`.
- `macros/op_contracts.py`: `OpContract(GRADIENT_OP, MaskRole.ATOM)` + import.
- `macros/macro_contracts.py`: add `int(GRADIENT_OP)` to the ATOM set (~line 314) +
  import; add to `reg_class`/role handling if CTRL_OSC is there.
- `macros/__init__.py` + `reglogparser.py` ~984: add `GradientPass()` to the inline
  list AFTER SweepPass/CtrlOscPass (so constant-delta sweeps and periodic osc claim
  first; gradient takes the aperiodic held remainder).
- `macros/flag_registry.py`: no edit needed (GATE_FLAGS auto-registers); add any
  FLAG_REQUIRES if a gradient flag should imply another (none expected).
- Tests: `tests/test_gradient_pass.py` through full `RegLogParser.parse()`
  (test-through-real-parse: synthetic dfs ship false-green). Assert byte-exact
  round-trip via the decoder, and that a fade like reg24 13..1 becomes ONE gradient
  atom. Run under xdist; NO narrative `#` comments (lint).
- Codebook reuse (phase 2): once the structural atom drains, add DEF/REF so
  identical gradients across voices/subtunes share one DEF (Galway reuses one
  structure) — generalizable + learnable, per "codebooks always".

Then add the four `*_gradient` flags to `_RESIDUAL_ARM` (gate) and `_CODEBOOK`
(residual_mechanism.py) and re-measure.

## B. INIT preamble atom — spec sketch

`init_startup` (130) = the driver's one-time init routine (filter/master-vol/freq
set once at startup, never recurs). One-time so the codebook (MINREP=2) can't claim
it. Model: an INIT/preamble atom bundling the leading one-time register
configuration (the writes before the first note-on / within the first ~3% of
frames that never recur) into one named atom. NOT a catch-all — it is the named
init mechanism. Build after A; re-measure first (A may absorb some startup curves).
