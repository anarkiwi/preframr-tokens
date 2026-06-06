# WORK ORDER (SELF-DIRECTING): melody learnability — interval-skeleton (layer 2) + cross-voice de-mux (layer 3)

**Status:** Pending impl — **auto-gated on the generator-MDL pipeline landing** (the other in-flight agent's
`preframr-tokens/AGENT_TASK_generator_pipeline.md`). **This file is self-directing: an agent told only "execute
this .md" must run §A's start-gate first — wait (autonomously, by re-checking on an interval) until the
generator pipeline is the deployed default on tokens `origin/main`, then start §1 with NO further help or
decisions.** Do not change the other agent's instructions; do not start partial work before the gate passes.
You operate ENTIRELY inside `/scratch/anarkiwi/preframr-tokens`; this file is your spec. Background reference
docs (in `/scratch/anarkiwi/preframr-xpt/design/`): `generator_mdl_representation.md` (the substrate this
extends), `learnability_token_ordering_theory.md` "Compatibility with the generator-MDL pipeline" (why this is
the melody fix), `encoding_principles.md` (P4.2/P5/P6). Read them for grounding; everything you must DO is here.

## §A. START-GATE — run this FIRST; it is the entire "wait for the other agent" mechanism

Executing this file means: **(1) run the landing check; (2) if WAITING, schedule a re-check and STOP this turn;
(3) if LANDED, start §1 immediately.** No human decision is needed at any step — the check is the decision.

**The landing check (copy-paste; LANDED only when the generator pipeline is merged AND is the default AND the
subsumed zoo is gone on tokens `origin/main`):**
```bash
T=/scratch/anarkiwi/preframr-tokens
git -C "$T" fetch origin -q && \
  git -C "$T" show origin/main:preframr_tokens/tokenizer_config.py 2>/dev/null | grep -q '"generator_pass"' && \
  ! git -C "$T" cat-file -e origin/main:preframr_tokens/macros/freq_trajectory_pass.py 2>/dev/null && \
  ! git -C "$T" cat-file -e origin/main:preframr_tokens/macros/skeleton_pass.py 2>/dev/null && \
  echo LANDED || echo WAITING
```
- **`WAITING`** → the generator pipeline is not yet the default. **Schedule a wake-up ~30–60 min out
  (`ScheduleWakeup`, or a `/loop` re-running this file) and STOP — do not block, do not start.** On each
  wake-up, re-run the check. (Optional faster signal: also poll `gh -R anarkiwi/preframr-tokens pr list
  --state merged --search "generator"` — but the file check above is authoritative.)
- **`LANDED`** → proceed immediately to §1. The tokens 0.45.0 PyPI release is orthogonal and **not** required
  to start this in-repo work (you build on tokens source `origin/main`).

**On LANDED, set up and go (no decisions):**
```bash
cd /scratch/anarkiwi/preframr-tokens && git fetch origin -q && git switch -c melody-skeleton origin/main
```
Confirm the base is green first (`./run_tests.sh` passes on `origin/main`, including
`tests/test_whole_chip_no_singleton_set.py` — it is `xfail`'d until generator_pass is the default, and
un-`xfail`'d by PART D, so a LANDED base has it green). Then execute §1–§6 end-to-end and open a PR (merge on
green if the repo allows; else leave it open). Everything below is fully specified — implement it as written,
raise nothing back.

**Known implementation anchors (the LANDED generator pipeline you build ON — verify they still exist, they
were committed in PR #62 / `generator-pipeline`):** `preframr_tokens/macros/generator_fit.py` exposes
`note_of(f, ref)`, `recon(note, ref)`, `tune_ref(freqs)`, `channels(state)`, `decompose(series)`, the LUT; the
freq channels are `GEN_FREQ_REGS=(0,7,14)` (16-bit combined). A freq note-onset's pitch currently rides as the
**`SWEEP_OP` START** (HOLD/ACCUM, raw 16-bit) or the **`GEN_TABLE` base_note** (note-relative; the freq DEF key
is `("note", offsets, residuals)`). `GEN_TUNING_OP=84` carries the per-tune `ref_q`. The highest live op is 88
(`GEN_TABLE_REF_OP`); **your new `MELODY_INTERVAL_OP` = 89** (next free; never renumber). `GeneratorPass` runs
inline in `reglogparser.py`'s pass list and is gated by the `generator_pass` flag.

## 0. Why this exists (the gap the generator pipeline leaves)

The generator-MDL pipeline makes **structure** learnable (gesture type/shape, DEF→REF copy, counter
elimination) and **de-ornaments by construction** (arp/vibrato/slide are separate note-relative atoms, off the
melody line). But it encodes each freq note-onset as a **raw 16-bit absolute pitch** (`SWEEP_OP` start /
`GEN_TABLE` base_note). **Absolute onset pitch is high-entropy with no local determinant ⇒ ~0 next-token**
(the V0-onset≈0 result; Principle 4.2). So the generator pipeline alone does NOT make melody learnable — it
removes the *pollution* (ornament) but leaves the melody line in its *unlearnable absolute form*.

**Melody learnability is a layered stack; this work order builds layers 2 AND 3 (both REQUIRED); layer 4 is a
named deferred hypothesis.**
- **Layer 1 (done by the generator pipeline):** de-ornamentation — ornament off the melody line.
- **Layer 2 (§1–§4): the interval-skeleton** — encode each voice's note-onsets as **key-invariant intervals**
  (measured held-out next-interval ≈ 0.52 > cross-tune 2-gram ceiling 0.41 — genuine transfer).
- **Layer 3 (§4B, REQUIRED — the DOMINANT lever): de-multiplex AND causally order the lanes.** Layer 2 alone
  does NOT deliver deployed melody learnability: the 0.52 was measured on *extracted single-voice
  (de-multiplexed)* data; deployed, the voices are frame-interleaved, so deployed melody-onset ≈ 0 vs the ~0.34
  per-voice ceiling (the gap is cross-voice multiplexing; within-voice factoring was only ~+0.03). **But
  contiguity alone can backfire** — plain voice-lanes push the melody's *harmonic* determinant out of locality
  and may predict the melody *before its cause* (P4 violation). The mechanism is **causal-DAG-ordered ROLE
  lanes — accompaniment before melody** — so the melody is predicted with its harmonic context in-context; this
  REQUIRES role identification (role-lanes are the *mechanism*, not a follow-up). Lane order is a TESTED
  variable, gated hard.
- **Layer 4 (§4C, HYPOTHESIS, deferred):** front-load the remaining determinants P4 names — rhythmic/metric
  position + explicit harmonic context, and scale-degree-vs-interval anchoring. Lossy/hard; open only if layer
  3 plateaus.

Without layers 1–3 the generalization goal fails on melody; with them the line is learnable in the only sense
the data supports (a *plausible*, transferable next note — exact pitch caps ~0.51 even for a memorizer, P5/P6).
Layer-3 designs: [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md) (lane mechanics) +
[`role_lane_factorization.md`](role_lane_factorization.md) (the role/causal-order mechanism).

## 1. The interval-skeleton (LAYER 2 — three pieces; the generator already gives within-note ornament)

1. **Note segmentation — intrinsic level-change ∪ gate** (NOT raw gate). Held-gate/legato drivers move pitch
   under one sustained gate, so gate-on under-segments. Use the landed `TrajectoryAnchorPass` pass-1 detector
   (sustained pitch-level change ∪ gate-on for re-struck same-pitch notes). **NOTE:** the generator pipeline
   DELETES `trajectory_anchor.py` (§4 of its work order) — recover the pass-1 detector from git
   (`git show <pre-deletion>:preframr_tokens/macros/trajectory_anchor.py`) and re-introduce it as the
   note-onset segmenter (segmentation only; it does not emit ops).
2. **Interval onset encoding (THE melody token).** For each note onset on a voice's freq channel, encode the
   note's pitch as a **signed semitone interval from the previous note's pitch**, in the LUT note domain:
   `interval = note_of(f_onset, ref) − note_of(f_prev_onset, ref)`. The first note of a voice is absolute.
   Lossless (a running sum reconstructs absolute note; the exact freq residual rides as in the generator's
   freq encoding). Intervals are **key-invariant** (transfer across transposition) and **low-cardinality**
   (cluster at 0/±few) — exactly Principle 4.2's "anchor to a nearby reference."
3. **Within-note ornament = the generator's note-relative atoms** (already built): `GEN_TABLE` arps are
   note-relative; `ACCUM`/`TRI` *within a note span* should be keyed relative to **this note's** pitch (extend
   the generator's note-relative keying from TABLE to the within-note ACCUM/TRI start). De-ornamentation is
   thus complete: the melody line is the interval sequence; everything per-frame is a note-relative gesture.

## 2. Precise spec (builds on the generator pipeline's freq channel)

The generator pipeline encodes freq as one 16-bit channel → `SWEEP_OP`(HOLD/ACCUM, raw-16-bit START) /
`GEN_TRI` / `GEN_TABLE`(base_note + note-relative offsets+residuals) atoms, with `GEN_TUNING` carrying `ref_q`
and `note_of`/`recon` the LUT maps (in `macros/generator_fit.py`). This layer changes ONLY how the
**note-onset base pitch** of each voice's freq atoms is keyed:

- **Run the segmenter** (piece 1) per voice on the freq channel → note-onset frames.
- **A freq atom that STARTS on a note-onset frame is a melody onset.** Replace its absolute start/base note
  with a **`MELODY_INTERVAL` atom** (new op, free id in the generator's range; e.g. `MELODY_INTERVAL_OP=89`):
  subreg `INTERVAL` = signed semitone delta from the previous onset's note (zig-zag/bias-encoded to stay a
  small non-negative token), subreg `FIRST`=1 + `NOTE_ABS` for a voice's first note. The atom's residual +
  its generator kind (HOLD/ACCUM/TABLE/TRI) and length are unchanged — only the *base pitch* is re-keyed.
- **Decoder:** maintain per-voice `cur_note`; on a `MELODY_INTERVAL` atom, `cur_note += interval` (or
  `=NOTE_ABS` if FIRST); the freq value = `recon(cur_note, ref) + residual`. Within-note ornament atoms decode
  relative to `cur_note` (offsets added to `cur_note`). Byte-exact: the running sum reproduces the absolute
  note; residual is exact (the generator's losslessness is preserved — this is a re-keying, not a value change).
- **Non-melodic voices (percussion/swept/low — e.g. Facemorph v0):** a voice whose freq never settles to a
  stable note grid (the generator's note channel is degenerate) is NOT interval-segmented — leave it as the
  generator's raw freq atoms. Decide per voice by a cheap stability test (fraction of frames whose `|residual|`
  is small); this is waveform-AGNOSTIC (never read the waveform bit — Facemorph guardrail).

## 3. Losslessness + the byte-exact gate (unchanged contract)
The interval re-keying is a lossless bijection on the note index (running sum), and the residual is carried
exactly, so `decode == register_state` byte-exact is preserved. Gate exactly as the generator pipeline:
`arbitrate(validate=True)`; corpus `reparse=True` byte-exact vs the generator-pipeline default; raw `SET`==0
(unchanged). The segmentation/interval layer must not introduce any residual.

## 4. The learnability gate (THIS is the payoff — not byte-exactness)
Melody is multi-modal; exact next-note caps ~0.51 even for a memorizing n-gram (P5/P6), so **do NOT gate on
exact-token accuracy.** Gate on:
- **Held-out next-interval accuracy** (held-out-by-dump, ×3 seeds) **must beat the cross-tune 2-gram ceiling**
  (the transfer test; the prior interval-skeleton hit 0.52 vs ceiling 0.41 — match or beat). Reuse the
  generalization-probe harness style.
- **Distributional + audition** — ornament/interval emission at ~corpus rate (JS small), and the 12-SID WAV
  audition. The model emits a *plausible*, transferable melody; that is the success criterion.
- **Learnability triage** (`audit/learnability_triage.py --mode blocks --seq_len 8192`): the interval melody
  token must show fast MI-decay + high induction-copy vs the absolute-pitch baseline (the interval is the
  key-invariant, low-cardinality form the theory predicts learnable).

## 4B. LAYER 3 — de-multiplex AND causally order the lanes (REQUIRED; the dominant melody lever)
Layer 2 makes each voice's line clean + key-invariant but the stream is **frame-major** (voices interleaved),
so the next-melody-note horizon is polluted by the other voices' tokens. Layer 3 reorders into per-channel
lanes so each line is **contiguous** — BUT contiguity alone is not enough and can BACKFIRE.

**The load-bearing subtlety — CAUSAL ORDER, not just de-mux — is now MEASURED, not just argued
(`/scratch/tmp/measure_melody.py`, 415 corpus tunes, 132k melody steps; heuristic extraction, trust direction):**
the current **harmony adds 0.294 bits of information about the next melodic interval BEYOND the melody's own
history** (H(next|self) 0.956 → H(next|self+harmony) 0.662, a ~31% conditional-entropy cut). So the harmony is
a real, non-redundant determinant — and the next interval is **low-entropy** (1.54 marginal → 0.66 conditioned),
i.e. melody is *learnable once self-history + harmony are surfaced*; the deployed ≈0 is a representation failure,
not a data ceiling. Plain voice-major lanes make the melody's *own* history local but push its *harmonic*
determinant out of locality — and if the melody lane is emitted *before* the accompaniment, the model predicts
melody **before its cause** (P4 violation). So the mechanism is **causal-DAG-ordered lanes: emit the
conditioning roles (bass/harmony) BEFORE the melody role** — now evidence-backed, not a guess. **This REQUIRES
role identification**, so the **role-lane** form ([`role_lane_factorization.md`](role_lane_factorization.md)) is
the *mechanism*, not a follow-up — and **measured: 63% of tunes' lead voice HOPS** (only 37% single-lead
all-tune), so fixed physical voice-lanes are wrong for most tunes; **but the lead changes only ~2.5×/tune, so
role assignment can be COARSE (per-section/block), not per-note.** The triage still gates the *magnitude*; the
*existence* of the harmony→melody dependency is settled.

**Spec:**
- **A `voice_lane` reordering pass**, per self-contained block (reuse `block_refire` + the
  `voice_canonical_block_order` / `super_frame` scaffold — [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)).
  Group each lane's events contiguously (voice from the FRAME-header packing / `remove_voice_reg`'s `v`), with
  reg-class sub-lanes so PW/filter re-admit without re-fragmenting the line; the global filter/mode lane stays
  shared. The **block (superframe) is the harmonic window** — short enough that cross-lane harmonic context is
  in-window.
- **Lane ORDER is the lever; melody-last is the evidence-backed default, the MAGNITUDE is the TESTED variable.**
  Order **accompaniment→melody (melody-last)** so the harmony (measured: +0.294 bits) precedes the melody;
  triage melody-last vs melody-first vs frame-major to confirm the gain transfers to the tokenized stream.
  Identify the melody/lead lane by a **COARSE per-section role tracker** (the lead hops only ~2.5×/tune, so a
  per-block assignment suffices — do NOT track per-note): highest-register sustained-pitched line, control-aware,
  **NEVER by waveform** (Facemorph guardrail). Bass = lowest line (the harmonic anchor, emit first).
- **Lossless = a permutation with a recorded byte-exact inverse.** Decode MUST restore the canonical
  voice-respecting, reg-ascending, frame-major render order (intra-frame write order is audible — the ADSR
  bug; only the canonical order the dumps already use is inaudible — `sid_render_fidelity_contract.md`). The
  inverse is derivable from the FRAME header; gate byte-exact as everything else (`arbitrate(validate=True)`).

**Layer-3 gate — UNTESTED at deployment, so gate hard (do NOT assume "de-mux helps"):**
1. **Triage pre-screen (mandatory, cheap):** `learnability_triage.py --mode blocks --seq_len 8192` across the
   lane-ORDER variants above. The melody-onset token's **MI-decay must shorten and induction-copy must rise**
   vs frame-major — AND the causal order (melody-last) should beat melody-first (confirming the harmony-context
   mechanism). If no order beats frame-major, STOP and report (do not ship a de-mux that doesn't help).
2. **No regression on OTHER content** (de-mux is a GLOBAL reorder): the same triage must show the structural /
   timbre / instrument / sweep tokens do NOT get worse (h_k, induction-copy) under the reorder — melody gain
   must not come at their cost.
3. **One canonical run (the go/no-go):** deployed melody-onset accuracy must recover from ~0 **toward the
   ~0.34 per-voice ceiling**, AND content-tier (other content) must not regress, vs the frame-major default.
   Default-OFF flag (`voice_lane`) until this passes.
- Lossless is necessary but NOT sufficient; the learnability recovery (melody up, others flat) is the point.

## 4C. LAYER 4 — surface the unsurfaced determinants (HYPOTHESIS, deferred; the theory-ideal endpoint)
Layers 2–3 surface melodic **contour** (interval) and the melodic **line + its harmonic context order**, but
NOT the two remaining low-entropy determinants Principle 4 says to front-load. Open this ONLY if layer 3
plateaus below the per-voice ceiling; it is lossy/hard and must be content-tier + audition-gated:
- **Rhythmic / metric position** — a beat/meter token local to each note (strong beats take chord tones).
- **Explicit harmonic context** — a chord/scale token (the biggest determinant; layer 3 already orders it
  before the melody, this would also tag it).
- **Scale-degree vs interval anchoring** — scale-degree (relative to the tune's key) generalizes across keys
  AND across melodies; an interval only across keys. **Measured caveat (`measure_melody.py`): WITHIN a tune,
  abs-pc / interval / scale-degree have ~equal entropy (0.94/0.97/0.94) — they're a relabeling at fixed key,
  so per-tune entropy CANNOT distinguish them.** The only axis that can is **cross-tune transfer**, so resolve
  this with a held-out cross-tune probe (does a scale-degree model trained on tune A predict tune B better than
  interval?) before building it; it needs key detection (lossy), gated distributionally + by audition.
These need beat-tracking / key+chord detection (lossy) — name them now as the endpoint; do not build until the
measured layers 2–3 have landed and shown a plateau.

## 5. Composition with the generator pipeline (what changes, what doesn't)
- **Changes:** (layer 2) add the note-onset segmenter + the `MELODY_INTERVAL` re-keying on freq note-onsets +
  within-note ACCUM/TRI note-relative keying; (layer 3) add the `voice_lane` block reorder pass + its byte-exact
  inverse.
- **Unchanged:** the generator's `{SWEEP_OP, GEN_TRI, GEN_TABLE}` atoms, the LUT/`GEN_TUNING`, all non-freq
  channels, `InstrumentProgramPass` (ctrl/AD/SR), the residual-zero + byte-exact gates, the digi exclusion.
- **Two default-OFF flags:** `melody_skeleton` (layer 2) and `voice_lane` (layer 3). Each folds into
  `REGISTERED_MACROS` only after BOTH its byte-exact gate AND its learnability gate (§4 / §4B) pass. Ship
  layer 2 first if layer 3's triage/canonical gate isn't yet green — but the deployed melody win needs both.

## 6. Tests + PR (same discipline as the generator work order)
- Through the real `RegLogParser.parse()`; xdist-chunked; lint forbids non-directive `#` comments.
- Byte-exact corpus gate (`reparse=True`) + the held-out interval transfer test + the triage read.
- **Module↔macros round-trip:** the SWM/defMON round-trip (§7B of the generator work order) must still pass —
  a melody re-keyed to intervals must still render to the SAME OUTPUT.
- New tests: segmentation correctness (held-gate legato → one note line, not over-segmented), interval
  round-trip (running sum == absolute), non-melodic-voice passthrough (Facemorph v0 unchanged), **and layer 3:
  the `voice_lane` reorder + inverse is byte-exact render-order (canonical, not arbitrary) on the corpus, and
  the SWM/defMON round-trip still produces the SAME OUTPUT under reordering.**
- **Layer-3 learnability gate (§4B) is part of this work order:** run the triage pre-screen; if green, the one
  canonical run is the go/no-go for the deployed melody recovery. Report it in the PR.
- Stay in preframr-tokens; no release/tag without the cross-repo procedure; PR through to merge.

## 7. Honest non-claims (state in the PR; do not relitigate)
- **Within-context, melody is MORE learnable than the old "~0.51 ceiling" framing implied** — measured next
  interval is low-entropy (1.54 marginal → 0.66 conditioned on self+harmony; `measure_melody.py`), so the
  deployed ≈0 was a representation failure, not a data ceiling. The remaining hard limit is **cross-tune
  transfer** (the 0.30–0.41 ceilings), which no encoding removes; score that axis distributionally + by
  audition. So: surface the determinants (this work order) → reach the conditional ceiling; transfer is the
  separate, residual limit.
- **Interval, not absolute, is the lever for transfer** — measured (0.52 held-out beats the 0.41 cross-tune
  ceiling; and within-tune entropy can't distinguish them, so the justification IS transfer); the generator
  pipeline supplies the de-ornamentation that makes the interval line clean.
- **De-mux + causal ordering (layer 3) is the DOMINANT lever; its PREMISE is now measured** (harmony adds
  0.294 bits beyond melody history → melody-last ordering justified) but its **deployed MAGNITUDE is
  untested** — gate hard (triage + one canonical run + no other-content regression). Role identification is
  required (63% of tunes' lead hops) but coarse/per-section suffices (~2.5 changes/tune); voice-lanes first,
  role-lanes the follow-up.
