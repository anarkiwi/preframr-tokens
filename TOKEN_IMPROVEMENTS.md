# Token improvements — strategic backlog

**Status (2026-05-23):** living document iterated alongside main-repo design discussions. Documents tokenizer-side improvements identified during the model-side architecture refute arc (per_tier_heads MoS/entropy/mask/cluster/diffusion all hit a content-tier ceiling at prodlike). Empirical sparsity probes pin the bottleneck at the tokenizer level; this doc collects the candidate fixes ranked by signal strength.

The intended consumer is the tokenizer maintainer agent: each section has enough impl detail to either land directly or expand into a focused PR.

## Empirical baseline (prodlike tokenizer, hash `991929565a378250`)

Probes already run on the prodlike training corpus (see main repo `integration_tests/design/music_llm_landscape_and_fail_fast_plan.md` for full data):

- **Long-tail data sparsity:** 38.2% of 7376 base atoms appear <10 times in training. Worst content family: `(op=0, reg=0)` (SET freq_lo voice 0) — 65% of 1926 atoms are long-tail. Content tier specifically: 37.5% long-tail.

- **PWM mass disproportion:** PWM is 2.8% of vocab but 22.6% of training mass (= 8× over-represented). Top family `(op=35, reg=2)` (PWM voice 0) alone is 16.8M training occurrences (21.4% of all mass).

- **Slope oscillation prevalence:** 31.3% of all slope tokens are in opposite-sign chains. Per-reg pair rate:
  - reg 2 (PW V0): 38.3%
  - reg 21 (FC_LO): 19.9%
  - reg 0 (FREQ_LO V0): **0%** (cent quantization at 50 destroys FREQ vibrato before slope can see it)
  - Chain length distribution: 674 chains of L=3, 413 of L=4, 201 of L=5, descending. Max chain: 41 slopes (a sustained vibrato).

Three findings, one root cause: **the tokenizer compresses some patterns well (PRESET, SLOPE, FLIP2 for ±a/±b alternation) but misses others (slope chains, lonely PW context, sub-50-cent FREQ vibrato), leaving the long tail and PWM mass disproportion that the model fails to learn.**

### Per-family op-coverage probe (500 prodlike songs)

Post-macro-pass row counts per voice-register family:

| family | total rows | SET-only | macro-encoded | SET% | macro% | top non-SET op |
|---|---|---|---|---|---|---|
| FREQ_LO | 680039 | 231023 | 19163 | 34.0% | 2.8% | DIFF_OP (op=1) 412947 |
| PW_LO | 923047 | 0 | 921139 | 0.0% | **99.8%** | PWM_PRESET (op=35) 790651 |
| CTRL | 332944 | 161343 | 1530 | 48.5% | 0.5% | CTRL_BIGRAM (op=42) 169955 |
| **AD** | 130845 | 130845 | **0** | **100.0%** | **0.0%** | (none) |
| **SR** | 165465 | 165465 | **0** | **100.0%** | **0.0%** | (none) |

**Critical gaps:**
- **AD + SR are 100% raw SET**, zero macro coverage. ADSR programs (instrument envelope changes) are completely uncompressed despite following strong repeating patterns.
- **CTRL is split SET / CTRL_BIGRAM**. The SET half is the residual that no macro caught.
- **FREQ_LO is 34% SET**. The 4-row pattern probe on FREQ_LO SETs found:
  - 45 arithmetic 4-tuples (caught by SLOPE if longer)
  - 657 geometric 4-tuples (exponential frequency curves — **SLOPE can't capture these**)
  - **7027 damped-oscillation 4-tuples** (alternating sign with decreasing magnitude — **OSCILLATE with fixed amplitude can't capture these either; needs a decay parameter or DAMPED_OSCILLATE variant**)
- **Patch-swap windows** (AD + SR + CTRL all changing within 3 rows on same voice): **137887 occurrences** across 500 songs ≈ **275 per song**. The single most common multi-register coupled pattern, currently uncaptured.

## Architectural principle: zero "lonely" updates

**Mandate:** every voice-register write must be part of a larger trajectory primitive (SLOPE, OSCILLATE, PATCH_SET, ATTACK, RELEASE, etc.). The 40% long-tail-SET pattern is to be **eliminated, not compressed**. Carveouts are explicit (enumerated) rather than the default fallback.

Rationale: the long-tail SET atoms are the model's hardest tokens (each appears few times in training). Replacing them with structured trajectory atoms that compose from a smaller alphabet of "shapes" gives the model fewer-but-more-meaningful primitives to learn. The model's content-tier ceiling at prodlike (~13% eval_a acc across five architectures) appears to be a function of how many distinct rare SETs it has to memorize.

### Explicit carveout list (proposed)

Where lonely SETs remain acceptable:
1. **First write per voice per song** — establishes initial state; no preceding trajectory possible.
2. **Section-boundary state changes** — when CTRL gate flips off and stays off for ≥K frames, the trailing SET on that voice may carry the "off-state" carve-out (rare).
3. **Filter-routing changes (reg 23)** — discrete topology change; not a trajectory.
4. **Master volume (reg 24)** — single-channel global control; rarely-changing setpoints.
5. **Truly first-of-pattern atoms** — the first SET in a SLOPE/OSCILLATE chain IS the SLOPE/OSCILLATE atom's anchor; not a separate SET.

Every other voice-register write must be folded into a trajectory primitive. The doc's proposed primitives below collectively aim to satisfy this.

## Proposed improvements

Listed in recommended landing order. Each has an explicit fail-fast gate so non-viable changes get killed within hours.

### 1. OSCILLATE op (subsumes FLIP + FLIP2, generalizes both)

**What:** new op encoding an N-cycle oscillation around an anchor value. Subsumes the existing `FLIP_OP` (single excursion) and `FLIP2_OP` (2-frame ±a/±b alternation) as special cases, and additionally collapses chained-slope oscillations (currently encoded as N independent slope tokens).

**Schema (proposed):**
- `op = OSCILLATE_OP` (new)
- `reg = <oscillating SID reg>`
- Subregs (4-row group, like SLOPE's 3-row group):
  - `subreg 0`: anchor value (the baseline the oscillation centers around)
  - `subreg 1`: amplitude (peak excursion from anchor, signed)
  - `subreg 2`: period_frames (frames per full cycle)
  - `subreg 3`: n_cycles_or_runlen (number of complete cycles; for FLIP variants, half-cycles encoded here)

**Subsumes existing ops:**

| existing op | OSCILLATE equivalent | what changes |
|---|---|---|
| `FLIP_OP` (open/close bracket on one reg) | `OSCILLATE(anchor=V, amplitude=Δ, period=2·gap_frames, n_cycles=0.5)` | becomes a regular 4-row OSCILLATE atom with n_cycles encoding half-period as a marker value; saves 2 atoms per FLIP and keeps the same audio |
| `FLIP2_OP` (asymmetric ±a/±b for N frames) | `OSCILLATE(anchor=(a+b)/2, amplitude=|a-b|/2, period=2, n_cycles=N/2)` | symmetric a==b case is exact; asymmetric a≠b loses ≤1 LSB on individual rows but matches at slope-terminal grid; saves (N-1) atoms |
| chained slopes (N alternating-sign slopes) | `OSCILLATE(anchor=slope_baseline, amplitude=slope_amplitude, period=2·slope_runtime, n_cycles=N/2)` | reduces N×3 atoms to 4 atoms |

**Detector (proposed `OscillationCollapsePass`, runs after `slope` + `preset`):**

```
for each (reg, voice) per-frame slope-terminal sequence:
    find runs of L>=3 alternating-sign slopes with same |amplitude| (±10% tolerance)
        and same period (±1 frame tolerance)
    emit OSCILLATE atom; drop the L source slope rows.
```

After OSCILLATE landing, deprecate `FlipPass` and `Flip2Pass` (have them no-op, removed in a later cleanup).

**Compression estimate (from probe):**
- ~7500 chained slopes × 3 atoms = ~22,500 atoms today (the chains alone).
- Replaced by ~1500 OSCILLATE atoms × 4 subregs = 6000 atoms.
- **74% reduction in slope-chain atoms** (and likely 50%+ reduction in PWM training mass once combined with the FREQ unlock below).

**Fail-fast plan:**
1. Implement `OscillationCollapsePass` + `OscillationDecoder`. Tests: synthetic chained-slope sequence collapses correctly; round-trip decode reconstructs the original frame-by-frame writes within slope-terminal grid precision.
2. Run round-trip audio fidelity (`preframr_audio.fidelity.compare_renders`) on 100 sample songs: tokenize twice, once with OSCILLATE off, once on; render both; compare. **PASS** if ≥95% within default `FRAME_RMS_TOLERANCE`. **HARD KILL** if <90% pass or any `DRIFTING_DIVERGENCE` shape (suggests amplitude/period detection is too lossy).
3. On round-trip pass, main repo runs `oscillation_collapse_mini_body_large` model A/B (3 seeds). Pass if val_acc within 1σ of baseline AND content vocab shrinks ≥10%.

### 2. Move `quantize_freq_to_cents` AFTER `slope` (unlock FREQ vibrato)

**Current pipeline order** (`macros/default_pipeline.py`):
```
squeeze_changes → combine_regs → quantize_freq_to_cents → simplify_ctrl
→ simplify_pcm → squeeze_changes → add_frame_reg → filter
→ squeeze_frame_regs → slope → preset → ...
```

`_quantize_freq_to_cents` maps 16-bit FREQ → ~240 cent-bins via `FreqMapper.fi_map` with `cents=50` default. At cents=50, a half-semitone bin width destroys all sub-semitone FREQ activity:
- **Vibrato (±20 cents typical)** → entirely below the bin boundary; quantized to a constant; slope detector sees no oscillation.
- **Portamento (±30 cents/frame)** → 0 or 1 cent-bin step per frame; pattern looks random; slope detector fails to bind.
- **Empirical:** reg 0 (FREQ_LO V0) shows 0/1849 opposite-sign slope pairs. FREQ never oscillates at this resolution.

**Proposed reordering:**
```
squeeze_changes → combine_regs → simplify_ctrl → simplify_pcm
→ squeeze_changes → add_frame_reg → filter → squeeze_frame_regs
→ slope → quantize_freq_to_cents (slope-aware: only quantizes
  slope-terminal subregs and stationary SET rows; leaves slope INTERIOR
  cells alone since they were already discarded by slope encoding)
→ preset → ...
```

**OR** a two-stage cent quantization that's simpler to land:

```
slope_pass detection input: FreqMapper(cents=10)  # fine grid for detecting vibratos
slope_pass terminal output: FreqMapper(cents=50)  # coarse grid keeps vocab small
PRESET / stationary FREQ rows: FreqMapper(cents=50)  # unchanged
```

**Vocab impact:** Slope INTERIOR values are never stored (only terminal + runtime). Fine grid affects detection only, not vocab atoms. Slope TERMINAL vocab stays the same. Stationary FREQ SET vocab stays the same.

**Expected effect:**
- FREQ slopes detectable at sub-semitone resolution → catches vibrato and portamento.
- Combined with OSCILLATE, FREQ vibrato collapses into single atoms (currently invisible in the tokenized stream entirely).
- May increase total slope token count by 1.3-2× on FREQ regs (need probe to confirm).

**Fail-fast plan:**
1. Build `FreqMapper(cents=10)` in addition to existing 50-cent mapper.
2. Re-route slope detection through fine-cent mapper; emit terminals via coarse-cent mapper.
3. Probe: count FREQ slope tokens emitted on 100 sample songs before/after. **PASS** if FREQ slope count ≥1.3× baseline AND total token count not >1.1× baseline. **NEUTRAL** if FREQ slope count unchanged → 10-cent is still too coarse, try 5-cent. **HARD KILL** if total token count >1.2× baseline (vocab inflation).
4. Round-trip audio fidelity (same audit as OSCILLATE).
5. Mini A/B in main repo if both prior gates pass.

### 3. Lonely PW frame analysis + ANCHOR semantics

**What is a "lonely" PW frame?** A frame containing a SET-PW write where:
- The frame is not part of any detected slope, FLIP, or (future) OSCILLATE on the same reg.
- The neighbouring ±N frames (small window, propose N=3) contain no other PW activity for the same reg.

**Trajectory taxonomy (proposed analysis, needs probe):**

| pattern | what it implies | currently encoded as | better encoding |
|---|---|---|---|
| "Lonely set, then stationary for >>frames" | a step setpoint change (musically: "set PWM for the rest of this section") | SET_PW (handled by PRESET if value matches grid) | PRESET if not already; no change needed |
| "Lonely set just before a slope/chain" | the trajectory's STARTING ANCHOR (the value the slope sweeps away from) | SET_PW + SLOPE | embed anchor into the slope/OSCILLATE atom directly (proposed: SLOPE gets an explicit `anchor` subreg if lonely SET precedes it within k frames) |
| "Lonely set just after a slope/chain" | the trajectory's ENDING ANCHOR / overshoot correction (musically: "land here after the sweep") | SLOPE + SET_PW | same — fold into the slope/OSCILLATE atom as `final_anchor` |
| "Lonely set between two slopes/chains" | a linkage/transition between two trajectories | SET_PW between two SLOPEs | leave as SET_PW (no clean compression unless the gap is small enough to merge into one extended trajectory atom) |
| "Truly isolated lonely set" (no nearby PW context at all) | one-frame state change with no surrounding trajectory; possibly musically meaningless (interrupt jitter, redundant write missed by squeeze) | SET_PW | candidate for DROP if audio-fidelity-neutral |

**Probe needed:** for the 16974 PW slope tokens already counted, walk the tokenized stream and bucket each NON-slope SET_PW into the categories above. Count proportions. Then for the smallest-impact category ("truly isolated, no nearby context"), test whether dropping them is audio-fidelity-neutral on 100 sample songs.

**Why this matters:** if many lonely PWs are "starting/ending anchors" of detected trajectories, folding them into the trajectory atom both shrinks the token stream AND helps the model learn the trajectory + anchor as one unit (rather than chasing the anchor separately, then trying to predict the slope from a SET context).

**Fail-fast plan:**
1. Write `profile/audit_lonely_pw_trajectories.py` (new profile in main repo). Walks 200 sample songs, counts PW SET frames per category. ~30 min CPU.
2. If the "starting/ending anchor" category is ≥30% of lonely PWs: design the anchor-fold extension to SLOPE/OSCILLATE (~half-day eng).
3. If "truly isolated" is ≥10% of all PW writes: design a SET_PW dropping pass with audio-fidelity round-trip gate.
4. If both categories are <10%: lonely PWs aren't doing much; abandon this lane.

### 4. PCM_BITS calibration sweep

**Current:** `_combine_reg(reg=PW, bits=PCM_BITS=5)` zeros the bottom 5 bits of every PW write at pipeline step 2, before slope. This snaps PW to grid=32 (128 distinct values out of 4096).

**Question:** is grid=32 the right tradeoff? Finer grids preserve more sweep detail but inflate vocab. Coarser grids shrink vocab but may smooth real musical sweeps to monotonic blobs.

**Sweep parameters:** `PCM_BITS ∈ {3, 4, 5, 6, 7}` (grids 8, 16, 32, 64, 128).

**Measurement per setting (no model training needed):**
- Total PW vocab atoms.
- Total slope tokens detected on PW regs (after pipeline runs).
- Round-trip audio fidelity on 100 sample songs (`compare_renders` at default tolerance).

**Fail-fast gates:**
- **STRONG WIN** (lower PCM_BITS i.e., finer grid): slope token count rises ≥20% AND round-trip ≥95% AND PW vocab stays under 1.5× baseline.
- **STRONG WIN** (higher PCM_BITS i.e., coarser grid): slope token count rises (sweeps that were ±32 LSB now look ±0 after grid=64 quantization, become slope-detectable) AND round-trip ≥95% AND vocab shrinks.
- **NEUTRAL:** no meaningful change at any setting; PCM_BITS=5 is already at the local optimum.

~2 hr fogbank to run the full sweep + audit.

### 5. PWM elimination probe (most aggressive PWM-mass cut)

**Hypothesis:** PWM is 22.6% of training mass for 2.8% of vocab. If most of that mass is wasted on inaudible micro-modulation that the model can't generalize from anyway, replacing all PWM with a canonical 50% (PW=0x800) at song init AND dropping all subsequent PWM writes would free 22.6% of training compute for content the model can actually learn.

**Mechanism:** new `PWMCollapsePass` (inserted at pipeline start, before everything else):
1. For each song, find the first PW write per voice. Replace with `(voice_offset+2, val=0x800)` (50% duty cycle, neutral square-wave-like timbre).
2. Drop all subsequent PW writes on every voice.

**Cost / risk:**
- Implementation: ~40 LoC + tests.
- Audio impact: severe — songs that use PWM modulation as a featured effect (most C64 leads, many basses) lose that timbral character. Audio audition would FAIL on most songs.
- **NOT recommended as a default** — even if it boosts val_acc, the audio output is musically impoverished.

**Why even consider it:** as a DIAGNOSTIC probe. If PWM elimination boosts val_acc on content tier substantially, it tells us how much of the current ceiling is "model can't learn content because PWM dominates the loss." That confirms the audio-norm / OSCILLATE direction is high-value (since those preserve PWM modulation while compressing it).

**Fail-fast plan:**
1. Implement `PWMCollapsePass` behind `--pwm-collapse-diagnostic` flag (off by default, never recommend on for real models).
2. Mini A/B in main repo: baseline vs `--pwm-collapse-diagnostic`. 3 seeds, mini body=large, 60 epochs.
3. **DIAGNOSTIC OUTCOME** (no shipping decision either way):
   - val_acc up ≥0.5%: confirms PWM mass distortion is real → prioritize OSCILLATE + audio-norm.
   - val_acc unchanged: PWM mass isn't crowding out content learning; bottleneck is elsewhere.
   - val_acc down: PWM carries cross-content generalization signal we shouldn't lose. OSCILLATE compression is still safe (preserves audio), but raw elimination is wrong.

### 6. Audio-equivalence value canonicalization (per-register family)

Cross-link to main repo `integration_tests/design/audio_equivalence_normalization_design.md`. Tokenizer-side normalization that collapses (op, reg, val) tuples producing perceptually-equivalent SID output to a canonical representative per register family. Uses `preframr_audio.fingerprint` v0.3.0 for the offline equivalence-class build.

The OSCILLATE + FREQ-reorder + PCM_BITS sweep above are subsets of what audio normalization could achieve more generally, but they're simpler to land and validate. **Recommended landing order: 1 → 2 → 3 → 4 → 5 (probe), then 6 (general normalization) once the targeted improvements have established the round-trip audit + fail-fast gates as standard infrastructure.**

### 7. PATCH_SET op (ADSR + CTRL coupled programs)

**Problem (from per-family probe):** AD + SR are **100% raw SET** rows; CTRL is 48.5% raw SET. ADSR changes mostly happen as INSTRUMENT PROGRAM CHANGES — a coordinated AD + SR + CTRL-waveform change within ≤3 rows on the same voice. **137887 such windows detected in 500 songs ≈ 275/song.** Each window is currently 3-4 SET atoms; could be 1 PATCH_SET atom.

**Schema:**
- `op = PATCH_SET_OP` (new)
- `reg = <voice CTRL reg>` (4, 11, or 18 — anchors the patch to a voice)
- Subregs (packed 16-bit val):
  - `subreg 0`: AD byte (8 bits)
  - `subreg 1`: SR byte (8 bits)
  - `subreg 2`: CTRL waveform bits (bits 1-7 of CTRL, packed 7 bits; gate bit handled separately)
  - `subreg 3`: optional PWM preset id (or sentinel "no PWM change")

**Detector (`PatchSetPass`, runs before `slope`):** for each voice, scan rolling 3-row windows looking for `{AD, SR, CTRL-non-gate}` co-occurrence. Replace the 3-4 SET atoms with one PATCH_SET atom + (optional) gate-only CTRL SET preserved.

**Compression estimate:** 137k patch-swaps × ~3 atoms each = ~410k atoms replaced by ~137k PATCH_SET atoms × 4 subregs = ~548k. Wait, that's larger — but **the win is vocab compression, not atom count**: a single PATCH_SET vocab atom covers (AD, SR, CTRL) combinations that span thousands of distinct SET atoms today (each tuple is one atom). Long-tail collapse on AD/SR/CTRL would shrink content vocab substantially.

**Fail-fast:**
1. Implement `PatchSetPass` + `PatchSetDecoder`. Tests on synthetic patch-swap windows.
2. Round-trip audio fidelity on 100 sample songs (same audit as items 1-5).
3. Count residual SETs on AD/SR/CTRL after PatchSetPass; target ≥50% reduction.
4. Main-repo mini A/B if round-trip passes.

### 8. DAMPED_OSCILLATE (or OSCILLATE decay parameter)

**Problem:** **7027 damped-oscillation 4-tuples in FREQ_LO** (alternating-sign deltas with decreasing magnitude) currently encoded as 4+ SET atoms each. OSCILLATE with fixed amplitude doesn't fit.

**Two design options:**

**Option A (preferred):** add a decay parameter to OSCILLATE. Subreg 4 (new) = `decay_rate_per_cycle` (0 = constant amplitude = vanilla OSCILLATE; positive = amplitude shrinks). 1 atom for both vanilla and damped cases.

**Option B:** separate DAMPED_OSCILLATE op. Simpler decoder but doubles vocab footprint of oscillation atoms.

**Fail-fast:** measure how many damped-oscillation windows in real corpus have approximately-exponential decay (where Option A's single decay_rate captures it well) vs more complex envelopes (where neither option captures faithfully — fall back to chain-of-SETs). If ≥80% of damped-osc windows are well-fit by single decay_rate, Option A is sufficient.

### 9. GEOMETRIC slope variant (exponential progressions)

**Problem:** **657 geometric 4-tuples in FREQ_LO** (constant-ratio deltas) — exponential frequency curves like rapid pitch sweeps or portamento. Current SLOPE pass requires arithmetic progressions (constant additive step); geometric progressions fail the ±1-LSB tolerance check and emit as 4+ SET atoms.

**Schema:** new `SLOPE_GEOMETRIC_OP` (per-reg, like the existing SLOPE_REG_TO_OP table). Subregs: terminal_hi, terminal_lo, runtime, ratio_per_step (8-bit fixed-point, e.g., `0x80 = 1.0` baseline, `0x90 = 1.125` = 1/8 octave per step). Decoder reconstructs by multiplying each frame.

**Fail-fast:** count geometric 4-tuples in 500-song probe (already 657 in the residual SET probe; likely 3-10× more in raw FREQ before slope detection). Mini A/B if compression is ≥5% of FREQ_LO SET atoms.

### 10. Under/overshoot correction audio test (CPU-only)

**Motivation:** OSCILLATE collapses N chained slopes into one atom with `(anchor, amplitude, period, n_cycles)`. The reconstruction emits N slopes that approximate the original sequence. Real-music chains often end with a "correction" SET (a lonely write that fixes the final landing value when the chain's last cycle over/undershoots the intended end-state). The principled-zero-lonelies mandate wants those corrections folded INTO the OSCILLATE atom (e.g., subreg 5 = `terminal_correction`); the empirical question is whether dropping them outright is audibly equivalent in practice.

**Test design:** CPU-only audio diff on real songs.

```
for each chained-slope sequence in 100 sample songs:
    identify the "correction" lonely SET that follows the chain (if any)
    version_A_writes = original chain + correction
    version_B_writes = OSCILLATE-reconstructed chain (no correction)
    samples_A = render_writes_to_samples(version_A_writes, ...)
    samples_B = render_writes_to_samples(version_B_writes, ...)
    diff = mel_distance(mel_features(samples_A), mel_features(samples_B))
    correction_magnitude = abs(original_final_val - oscillate_predicted_final_val)
    emit (correction_magnitude, mel_diff, song_path)

plot/bucket: correction_magnitude vs mel_diff
```

**Output:** a chart with X = correction magnitude in slope-terminal-grid units, Y = mel feature distance. Buckets show:
- **Drop-safe zone:** correction_magnitude ≤ K AND mel_diff ≤ ε → OSCILLATE without correction is audibly indistinguishable.
- **Fold-into-OSCILLATE zone:** correction_magnitude in (K, M], mel_diff in (ε, δ] → noticeable but small; OSCILLATE should learn the correction via a `terminal_correction` subreg.
- **Keep-explicit zone:** correction_magnitude > M OR mel_diff > δ → drop a chain-of-SETs is wrong; keep correction as a follow-up SET (rare; ideally <10% of cases).

**Tooling:** uses only `preframr_audio.audio_driver.render_to_samples` + `preframr_audio.features.mel_features` (both shipped in preframr-audio v0.3.0). No GPU. ~30 min on fogbank for 100 songs.

**Decision logic from output:**
- **>80% drop-safe at K=3, ε=0.05:** the lonely-correction lookup table can be dropped entirely from OSCILLATE reconstruction. Saves a subreg slot.
- **40-80% drop-safe:** OSCILLATE needs the `terminal_correction` subreg to handle non-drop-safe cases. Drop-safe cases use a sentinel value.
- **<40% drop-safe:** chains have too-variable terminations for any fixed-form OSCILLATE encoding. Fall back: keep correction SETs but flag them in the macros pipeline as "explicit overshoot fix" with a special carveout op (so the model sees them as a structured class, not as raw long-tail SETs).

**This is the highest-priority empirical probe** because it directly validates whether the "zero lonely updates" mandate is achievable at acceptable audio fidelity. If it isn't, the mandate needs revision (e.g., "≤5% lonely updates with explicit overshoot-fix op").

## Cross-cutting infrastructure needed

Two preframr-audio additions (already drafted in main repo's
`cluster_conditional_content_head_design.md` "preframr-audio enhancements"
section) become load-bearing:

1. `preframr_audio.fingerprint` already shipped at v0.3.0 — provides `fingerprint_batch` for offline acoustic clustering (used by item 6).

2. `preframr_audio.fidelity.compare_renders` additive parameters:
   - `max_frame_drift=N` to tolerate ±N frames of cross-correlation lag (used by every round-trip audit in items 1-5).
   - `feature_diff_fn` + `feature_diff_tolerance` for feature-space comparison (used by item 6).
   - Defaults preserve current bit-exact behavior; only opt-in callers see the relaxation.

Both can land as preframr-audio v0.3.1 patch release; no breaking changes.

## Round-trip audit standard

Every tokenizer change in this doc shares a single fail-fast gate: the round-trip audio fidelity audit. Standardize on:

```
profile/audit_tokenizer_change.py (new, lives in main repo):
    for each of 100 sample songs:
        original_df = parse(dump.parquet, transforms=baseline_pipeline)
        modified_df = parse(dump.parquet, transforms=modified_pipeline)
        original_audio = render(original_df)
        modified_audio = render(modified_df)
        compare_renders(original_audio, modified_audio,
                        max_frame_drift=2,
                        feature_diff_fn=mel_distance,
                        feature_diff_tolerance=0.1)
    assert >=95% pass at default tolerance
    assert 0 DRIFTING_DIVERGENCE
```

Run on every PR that modifies the macro pipeline.

## Open questions for the maintainer

1. **OSCILLATE schema details:** is the 4-subreg layout (anchor/amplitude/period/n_cycles) the cleanest, or should anchor live elsewhere (e.g., implied from preceding SET context)?
2. **FREQ reordering vs two-stage cents:** which is cleaner to land — moving `quantize_freq_to_cents` after `slope` (touches pipeline ordering), or adding a second `FreqMapper(cents=10)` (touches `reglogparser.py` + slope_pass)?
3. **Lonely PW dropping:** do we have audio-side evidence that one-frame PW writes in isolation are audibly irrelevant, or do they affect attack transients in ways the mel-distance comparison won't catch?
4. **Deprecation timeline for FLIP / FLIP2:** simultaneous with OSCILLATE landing, or staged? Staged means model checkpoints trained on FLIP/FLIP2 vocab continue to work; simultaneous means a clean cut but invalidates all existing checkpoints.
5. **PCM_BITS sweep range:** is `{3, 4, 5, 6, 7}` the right span, or should it extend to `{2, 8}` to test the extremes?

## Empirical TODO (probes that haven't run yet)

1. **Under/overshoot correction audio test** (item 10 — highest priority; validates the zero-lonely-updates mandate). ~30 min fogbank.
2. Lonely PW trajectory classification probe (item 3 fail-fast step 1). ~30 min.
3. PCM_BITS slope-detection sweep (item 4 fail-fast). ~2 hr.
4. FREQ slope count at `cents=10` vs `cents=50` on 100 sample songs (item 2 fail-fast step 3). ~30 min.
5. Damped-oscillation decay-fit goodness probe (item 8 fail-fast). ~30 min.
6. Geometric slope ratio quantization probe (item 9 fail-fast). ~30 min.

All are ≤2 hr each on fogbank, no GPU needed. The under/overshoot test (#1) gates the entire OSCILLATE design.
