# Interval-coded freq V0 (transposition-invariant melodic onsets)

**Status:** Design + implementation in progress (branch `feat/freq-v0-interval`). Self-contained
in preframr-tokens; opt-in (`--freq-v0-interval`, default OFF); FREQ_TRAJ byte-exact round-trip
must stay green. Target: next minor.

## Problem (evidence-backed)

The FREQ_TRAJ melodic line is the sequence of trajectory **onset pitches** — the anchored base
value `V0` (osc/run) or `terminal` (ramp), on the freq regs 0/7/14. Today these are stored
**absolute** (cents). Consequences, measured on `trajectory_anchor_mini` (2026-05-27):

- The onset line is **highly predictable in-sample** — conditional entropy k=2 ≈ 2.2 bits, a
  trigram predicts the next onset at **0.79** (`audit.melody_predictability`).
- Yet the model predicts the **V0 onset exactly 0.000** of the time (subreg-split audit; 0
  hits / 7332 anchored, 0 / 10416 unanchored) — neither aleatoric nor a metric artifact.

The gap is **generalization**: an in-sample trigram memorises each song's *absolute* pitches,
but a model trained to predict held-out songs sees the *same motif at a different key/octave as
a completely different absolute-cent token sequence* — so melodic structure does not transfer
across songs/keys/composers, and the model learns nothing transferable. Absolute pitch is the
wrong representation for a generalising melodic predictor.

(The trajectory **shape** — the per-frame DELTA samples — is already relative/cumulative, so
e.g. a ±2-semitone vibrato already encodes identically at any base. The shape is fine; the
**onset** is the defect. PW/filter onsets stay absolute — timbre genuinely is absolute.)

## The change

Encode each freq trajectory's characteristic value as a **signed interval from the previous
freq trajectory's characteristic value on the same voice**, instead of absolute cents:

- `char = v0` for OSCILLATE/RUN, `char = terminal` for MONOTONE_RAMP.
- Per voice (freq regs 0/7/14), maintain `prev_char`. The **first** trajectory is stored
  **absolute** (FLAGS `FT_V0_INTERVAL_BIT` clear); every subsequent one stores
  `char − prev_char` as a signed 16-bit value in the existing `V0_HI/V0_LO`
  (= `TERMINAL_HI/LO`) bytes, with `FT_V0_INTERVAL_BIT` (0x08) set. Then `prev_char ← char`.
- **Freq regs only.** PW (2/9/16) and filter (21) keep absolute V0 (encoder never sets the bit
  for them; decoder simply honours the bit, so no reg special-casing is needed downstream).

A melody transposed by +X cents shifts every `char` by X, leaving every interval unchanged →
**identical interval-token sequence at any pitch**. The onset alphabet also collapses (melodic
intervals cluster tightly around 0 / ±a few semitones vs ~200 distinct absolute pitches), so the
onset becomes a low-entropy, transferable, learnable target.

## Encoding details (byte-exact)

- New flag bit `FT_V0_INTERVAL_BIT = 0x08` in the FLAGS byte (free; SUBTYPE=0x03, PERIODIC=0x04).
- Signed interval stored two's-complement 16-bit in `V0_HI`(<<8)|`V0_LO`. Range ±32768 cents =
  ±327 semitones ≫ any musical interval, so **no escape needed** (same 2-byte budget as today;
  the decoder already interprets the ramp terminal as signed 16-bit — reuse that).
- Default OFF ⇒ the bit is never set ⇒ decode path is **identical to today, byte-for-byte**.

## Decode

Add `state.last_freq_v0[reg]` (per-reg onset reference, distinct from `last_val` so it is
unaffected by sample draining — encoder and decoder track the same chain by construction):

- OSC/RUN: `raw = (v0hi<<8)|v0lo`; if FLAGS has the interval bit, `v0 = last_freq_v0[reg] +
  signed16(raw)`, else `v0 = raw`. Then `last_freq_v0[reg] = v0`. (Shape deltas still apply on
  top of the reconstructed absolute `v0`, unchanged.)
- RAMP: same for `terminal`; `last_freq_v0[reg] = terminal`. `start_val = last_val[reg]` and the
  ramp interpolation are unchanged.

The first trajectory per reg has the bit clear (absolute) and seeds `last_freq_v0[reg]`.

## Files

- **`stfconstants.py`**: `FT_V0_INTERVAL_BIT = 0x08`; freq-reg subset constant (0,7,14).
- **`macros/freq_trajectory_pass.py`**: gate `freq_v0_interval`; per-reg `prev_char` in `apply`;
  `_delta_run_rows` / `_ramp_rows` emit signed interval + set the bit for freq regs.
- **`macros/decoders.py`**: `state.last_freq_v0`; `FreqTrajectoryDecoder` honours the bit.
- **`preframr/args.py`** (framework): `--freq-v0-interval` BooleanOptionalAction default False,
  NOT in `_PIPELINE_NAME_TO_FLAG` (a freq_trajectory modifier, toggled per-arm via extra_cargs —
  same pattern as `--trajectory-anchor-pass`).

## Tests

- **Round-trip byte-exact** with `freq_v0_interval` ON: existing FREQ_TRAJ oracle must stay green
  (expand(encode(x)) == x). Default-OFF path unchanged.
- **Transposition invariance** (the point): build a freq SET stream, transpose it by a constant
  cent offset, encode both with interval ON; assert the emitted FREQ_TRAJ rows are **identical
  except the first (absolute anchor) V0 row**. With interval OFF they differ everywhere.
- **PW/filter untouched**: a PW trajectory encodes byte-identically with the flag on/off.

## A/B + gate

Mini A/B `freq_v0_interval` on top of anchoring (anchored+interval vs anchored-absolute), read by
`audit.content_tier_report` **split to the V0-onset subreg** (the decisive number is V0-onset
acc, not aggregate op45) + `audit.melody_predictability` (the interval onset line should now be
low-entropy AND the model should finally capture it). Escalate to a prodlike interval-vs-absolute
A/B if the onset acc moves. See `preframr-xpt/design/trajectory_anchoring.md` for why this is the
top lever.
