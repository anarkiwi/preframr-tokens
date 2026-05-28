# FREQ_ONSET — route all TRAJ_REG writes into the onset channel, clean SET

**Status:** Design, ready to implement (preframr-tokens; opt-in `--freq-onset-pass`, default OFF;
byte-exact round-trip). Cross-ref: `preframr-xpt/design/trajectory_anchoring.md` (melody arc) +
`design/freq_v0_interval.md`.

## Problem (measured)

The continuous registers (freq 0/7/14, PW 2/9/16, filter-cutoff 21 = `TRAJ_REGS`) only become
trajectory/onset tokens when they form runs of **≥2** writes; isolated/short runs fall through
`FreqTrajectoryPass` and stay as **op0 SET**. Measured on the anchored+interval mini stream:
**op0 SET on freq regs = 12.1% of the stream, acc 0.0013** — i.e. ~12% of the stream is
*melodic onsets hiding in SET*, fragmented away from the op45 V0-onset channel (0.9%). So:
- the melody metric **undercounts** (only counts op45 V0; misses the SET-hidden onsets),
- the *same* musical event (a note onset) takes **two different token forms** depending on
  whether it clustered — fragmenting a learnable pattern,
- SET is **heterogeneous** (control/ADSR mixed with these continuous-reg leftovers).

## The change (invariant: TRAJ_REGS never SET)

Add `FREQ_ONSET_OP` (48). After `FreqTrajectoryPass`, re-tag every **remaining op0 SET on a
`TRAJ_REG`** to `FREQ_ONSET` — a **1-token** tagged write `(op=48, reg, subreg=-1, val)`,
**cost-neutral** vs the SET it replaces. Result: every TRAJ_REG write is now either a
`FREQ_TRAJ` (op45, clustered) or a `FREQ_ONSET` (op48, isolated); **op0 SET carries only
control/ADSR/routing** (ctrl 4/11/18, AD 5/12/19, SR 6/13/20, filter mode 23, volume 24).

v1 keeps the FREQ_ONSET `val` **absolute** (a pure re-tag). Interval-coding the FREQ_ONSET val
(matching trajectory V0) + decoupling onset-from-shape inside trajectories are **follow-ups**
(noted below) — v1's job is the channel invariant + the metric/SET cleanup.

## Encoding / decode (byte-exact)

- `FREQ_ONSET_OP = 48` (free). Token = the SET row with `op` swapped 0→48; `reg/subreg/val`
  unchanged. 1 token, no format change.
- `FreqOnsetDecoder` (op48): emit a SET on `reg` with `val` (set `state.last_val[reg]`), i.e.
  decode is identical to a plain SET. Round-trip: SET→FREQ_ONSET→SET is the identity on
  `(reg, val)`, so the per-frame fidelity oracle stays green.
- **Tier:** classify op48 as **content** in `tier_classify` (it is a melodic/timbral onset, like
  FREQ_TRAJ), so the content-tier audit + loss see it correctly.

## Files

- **`stfconstants.py`**: `FREQ_ONSET_OP = 48`.
- **`macros/freq_onset_pass.py`** (new): `class FreqOnsetPass(MacroPass)` — gated by
  `freq_onset_pass`; after FreqTrajectoryPass, set `op = FREQ_ONSET_OP` on rows where
  `op == SET_OP and reg in TRAJ_REGS and subreg == -1`. Annotation-only (no new rows).
- **`reglogparser.py`**: call `FreqOnsetPass().apply(df, args=self.args)` immediately after
  `FreqTrajectoryPass`.
- **`macros/decoders.py`**: `FreqOnsetDecoder` (op48 → SET-equivalent).
- **`tier_classify.py`**: op48 → content tier (in `_op_tier_map`).
- **Framework `preframr/args.py`**: `--freq-onset-pass` (BooleanOptionalAction, default False),
  not in `_PIPELINE_NAME_TO_FLAG` (a freq_trajectory-family modifier, per-arm via extra_cargs).
- **Bump** `fallback_version`.

## Tests

- **Round-trip byte-exact** with `freq_onset_pass` on: a stream with isolated freq/PW/filter
  writes → expand == raw per-frame values (the FREQ_ONSET re-tag is identity on decode).
- **Invariant:** after the pass, **no op0 SET remains on any TRAJ_REG**; control/ADSR regs are
  untouched.
- **Default off** ⇒ byte-identical to today (no FREQ_ONSET emitted).
- Tier: `vocab_id_tier` on a FREQ_ONSET vid returns `content`.

## A/B + gate

`freq_onset_channel_mini`: anchored+interval base, arm `onset_chan` adds `--freq-onset-pass`
vs baseline. Read with `content_tier_report --onset` **grouping op45-V0 + op48-FREQ_ONSET on
freq regs** as the unified melodic onset (the reader must include op48): does the unified
onset acc beat the split baseline's op45-only onset (de-fragmentation), and is SET cleaner
(op0 share drops ~12pp)? Mini is scale-bound for melody, so the primary wins are the metric
de-confound + SET cleanup + the de-fragmentation signal; the real melody test is the prodlike
with this channel in place.

## Melody is fragmented across THREE freq ops (measured)
op0 SET freq-reg 12.1% / op45 V0-onset 0.9% / **op47 FREQ_NUDGE pitch 0.4%** — all acc ~0.
FREQ_ONSET v1 consolidates the op0-SET fragment (the big one). FREQ_NUDGE is a small pitch
*delta* (nudge), semantically NOT a fresh onset, so it is **not** re-tagged (would corrupt its
delta decode) — but the unified-melody read must group {op45 V0, op48 FREQ_ONSET, op47 pitch
subregs} so total freq-pitch content (~13.4%) is measured, not just the onset channel.

## Follow-ups (not v1)
- **Interval-code the FREQ_ONSET val** (freq regs) to match trajectory V0 — full representation
  unification (transposition-invariant single onsets).
- **Decouple onset from shape** inside FREQ_TRAJ (emit FREQ_ONSET for the pitch, a separate
  shape token for the modulation) — one consistent melodic-onset form for *all* notes.
