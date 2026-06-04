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
1. Build A (held-step automation codebook). Re-measure.
2. Build B (INIT preamble). Re-measure (expect multiwrite to fall as nibble pairs go).
3. Extend C (hard-restart multiload) for the remainder.
4. Gate green → PR `resid/ornament-and-digi` (NO release).
