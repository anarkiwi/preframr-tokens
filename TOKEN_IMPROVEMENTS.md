# Token improvements — strategic backlog

**Status (2026-05-23):** living document iterated alongside main-repo design discussions. Documents tokenizer-side improvements identified during the model-side architecture refute arc (per_tier_heads MoS/entropy/mask/cluster/diffusion all hit a content-tier ceiling at prodlike). Empirical sparsity probes pin the bottleneck at the tokenizer level; this doc collects the candidate fixes ranked by signal strength.

The intended consumer is the tokenizer maintainer agent: each section has enough impl detail to either land directly or expand into a focused PR.

## Canonical implementer spec (2026-05-24)

This section is the **definitive list of work** to ship. Items 0-12 below contain design rationale and detector internals; this section is what to build and in what order.

**Goal:** zero strict-no-diff validator failures on the prodlike corpus (dataset hash `991929565a378250`). Day-1 baseline is 84,037 unmodelled non-INFRA rows per 200 songs (~13.8%). The six primitives + rules below absorb 100% of those rows; every signature has been empirically mapped.

### Locked detector defaults

All four are sweep-verified (2026-05-24):
- `OSCILLATE_ENV.vibrato_min_cycles = 2`
- `OSCILLATE_ENV.portamento_min_steps = 3`
- `NOTE_ON.coupling_window = 12 rows`
- `VOICE_TRACK.min_track_duration = 10 frames` — sweep knee; below this catches spurious coincidences (T=4→6 loses 17,218 absorptions of which most are 4-5 frame chord-change overlaps), above this misses real sustained tracking (T=12→16 loses 12,040 of actual musical intervals)

These four constants land directly in the default config — no further tuning needed.

### The six primitives / rules

Land in this order. Each row lists the primitive, its empirical absorption count from 200 prodlike songs, and the gap class(es) it closes from the 100% accounting.

| # | primitive / rule | absorbs (200 songs) | gap classes closed |
|---|---|---|---|
| 1 | **OSCILLATE_ENV** (item 0) with relaxed defaults | 95,182 rows already absorbed in baseline coverage sim | item 0's full coverage table |
| 2 | **VOICE_TRACK** (item 12, pre-per-voice-split, `MIN_TRACK_DURATION=10 frames`) | ~246,000 tracker FREQ writes across 200 songs (sweep-verified) — a large fraction of the 10,401 per-voice isolated SET/FREQ_LO would be absorbed pre-split | up to 10,401 isolated SET/FREQ_LO BOS/EOS in the strict-no-diff residual, plus equivalent fraction of FREQ_HI writes that aren't separately counted at the per-voice layer |
| 3 | **FREQ_NUDGE** (item 11) — unified op with delta-or-absolute payload | 16,072 isolated DIFFs + remainder of 10,401 isolated SETs not taken by VOICE_TRACK + 17,857 of the 2-event FREQ_LO runs (BOS/SAME and SAME/EOS) | "truly isolated" + 2-event FREQ_LO categories |
| 4 | **FREQ_RUN extension to pure-SET runs** (item 11) — drop the "DIFF must be present" requirement | 4,483 SET/SET interior signatures + co-absorbs the 2-event SET pairs in (3) | 3+ SET runs and SET-SET pairs |
| 5 | **RELEASE_UPDATE** (item 11) — covers SR and trivially AD | 5,378 SR + 33 AD = 5,411 | SET/SR + SET/AD isolated |
| 6 | **CTRL_TRIPLE** OR expanded CTRL_BIGRAM with adjacency rule (item 11) | 1,802 CTRL writes (854 SET/EOS + 486 BOS/EOS + 264 SET/SET + 52 BOS/SET + 146 CTRL_BIGRAM-adjacency) | all SET/CTRL signatures |
| 7 | **Universal trajectory-anchor extension rule** — every SLOPE/FLIP/FLIP2/TRANSPOSE primitive may absorb ±1 leading/trailing same-reg event into its anchor | 3,017 rows across DIFF + SET adjacency to existing trajectory primitives | OP3/OP5/OP7/SLOPE adjacency signatures |

(7 rows, the user asked for 6 — items 6 and 7 are both small enough that 7 can be merged into a generic "carveout" but the design rule is distinct enough to be enumerated separately. The implementer can choose to fold 7 into 6 if preferred; the absorption math is identical.)

### Per-primitive acceptance gates (no surprises at integration)

Each primitive must pass before the next one merges:

1. **Round-trip audio fidelity** on 100 sample songs via `preframr_audio.fidelity.compare_renders` (`max_frame_drift=2`). PASS if ≥95% within `FRAME_RMS_TOLERANCE`; HARD KILL if <90%.
2. **Validator delta**: re-run the strict-no-diff coverage simulator (`/scratch/tmp/probe_coverage_sim.py`) and confirm the expected absorption count is hit within ±15%. Smaller-than-expected absorption is a sign the detector is too narrow; larger may indicate false absorption.
3. **Mini A/B model spec** (`<primitive_name>_mini_body_large`, 3 seeds, in the main repo): val_acc within 1σ of baseline AND content vocab shrinks ≥5% AND no diversity_ratio regression at T=0.5.

### Final acceptance: WAV audition gate (post-all-primitives)

After all six primitives are merged, the implementer **must produce and submit a WAV audition set** for human review before declaring the work complete. This gate is non-negotiable — automated audio-fidelity metrics caught the obvious failures but cannot catch subtle musical quality degradation (envelope mis-fits affecting timbre, lost vibrato character, broken cross-voice harmony from VOICE_TRACK ordering bugs).

Audition spec:
- **Subset:** 12 SIDs covering breadth of the prodlike corpus. Recommended representative cohort: 2 each from Galway (Martin), Hubbard (Rob), Tel Jeroen, Daglish (Ben), Whittaker (David), and one each from Dane and Detert. These names appear in `dataset_cache/991929565a378250/train/` and span pre-1990 / post-1990 / chip-music / film-score idioms.
- **Renders per SID:** two WAVs each — baseline (current tokenizer) vs strict-no-diff (all six primitives applied). Both rendered via the same `preframr_audio.audio_driver.render_to_samples` pipeline.
- **Naming:** `audition/<composer>_<title>.baseline.wav` and `audition/<composer>_<title>.strict_no_diff.wav`.
- **Reporting:** alongside the WAV pairs, ship a `audition/report.md` with: (a) the strict-no-diff token compression ratio per SID, (b) any segments where automated mel-distance is >0.10 (highlighted as "review carefully"), (c) any cross-voice TRACK_REF spans that the implementer noticed during dev (areas the auditioner should focus on).
- **Pass criterion:** the user listens to all 12 pairs and notes "indistinguishable" or "minor difference, acceptable" for ≥10/12. If ≥3 pairs are flagged as "audibly degraded", the work goes back for rework on the specific primitive responsible.

### Re-analyze after work lands

Once the strict-no-diff validator passes on the prodlike corpus end-to-end and the WAV audition is signed off, the user will re-run the full content-tier model arc (mini A/B sweep + prodlike A/B) to measure whether the tokenizer rework unblocks the content-tier ceiling that the five model-side interventions (MoS, entropy, mask_structural, cluster_conditional, diffusion) couldn't break. This is the actual point of the entire token-improvements arc; the strict-no-diff cleanup is the precondition.

### What the implementer does NOT need to do before starting

All design parameters are empirically locked:
- Detector thresholds (vibrato, portamento, NOTE_ON window) — sweep-verified.
- Envelope family set + STEP level tables — probe-verified, in item 0.
- Carveout list — calibration data in item 11.
- Trajectory primitive ID space — exists in `stfconstants.py`; VOICE_TRACK and FREQ_NUDGE and RELEASE_UPDATE and CTRL_TRIPLE need new op codes assigned (next available slots).
- VOICE_TRACK MIN_TRACK_DURATION — sweep result tightening this is in item 12.

The implementer does not need to run any probes to start. All open questions remaining in the doc are tagged "open question for the implementer" and have a recommended default to use unless reason to deviate.

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

### 0. Envelope-modulated OSCILLATE (umbrella primitive — subsumes items 1, 8, 9 + ARPs)

**Status:** unification design — supersedes items 1 (vanilla OSCILLATE), 8 (DAMPED_OSCILLATE), 9 (GEOMETRIC), and addresses a large fraction of unmodelled FREQ_LO writes (item 11) by reframing FREQ arpeggios as oscillator instances. Empirically grounded by the envelope-fit probe (2026-05-23, 200 prodlike songs, 3039 chains).

**What:** one op `OSCILLATE_ENV` whose subreg layout adds an **envelope** that modulates per-cycle amplitude over the chain's duration. The envelope shape is selected from a small parametric family (closed-form, no library lookup). This collapses the previously-proposed vanilla / damped / geometric / step / arp ops into a single primitive parameterized over the envelope shape.

**Why one op instead of four:** the envelope-fit probe shows the chains divide naturally by *amplitude trajectory shape*, not by separate underlying phenomena — a damped chain and an arpeggio both modulate oscillator amplitude, just with different envelope shapes. Folding them into one op:
- collapses 4 op-ids → 1, shrinking the model's op-tier alphabet
- gives the model a single primitive to learn whose interior structure (envelope_id) is a small categorical
- composes cleanly with FREQ arpeggios without inventing an ARP op
- means future envelope shapes (AHDSR, triangular, etc.) extend the existing op rather than adding new ones

**Schema (6 subregs):**

| subreg | bits | meaning |
|---|---|---|
| 0 | 8-16 | `anchor` — baseline value (reg-width matched) |
| 1 | 8-16 | `base_amplitude` — peak excursion when envelope=1.0 |
| 2 | 4-6 | `period_frames` — frames per full cycle |
| 3 | 5 | `n_cycles` — 0-31 (saturate at 31; covers 99% of observed chains) |
| 4 | 4 | `envelope_family_id` — selects shape family (see table below) |
| 5 | 8 | `envelope_param` — signed fixed-point, family-specific meaning |

**Envelope families (envelope_family_id values):**

| id | name | envelope_param meaning | per-cycle amplitude formula |
|---|---|---|---|
| 0 | CONSTANT | unused (zero) | `base_amplitude` |
| 1 | LINEAR | `end_ratio_q6` — signed Q1.6 ratio of end/base | `base_amplitude * lerp(1.0, end_ratio, i/(n-1))` |
| 2 | EXP_DECAY | `decay_per_cycle_q6` — Q0.7 in (0, 1) | `base_amplitude * decay^i` |
| 3 | EXP_GROWTH | `growth_per_cycle_q6` — Q1.6 in (1, 4) | `base_amplitude * growth^i` |
| 4 | STEP_2 | `level_ratio_q7` — Q0.7 ratio of level_2/level_1 | `base_amplitude * (1.0 if i%2==0 else level_ratio)` |
| 5 | STEP_3 | 3 levels × 2 bits each (6 of 8 bits used); each 2-bit value indexes `{0.25, 0.5, 1.0, 0.75}` | `base_amplitude * levels[i % 3]` |
| 6 | STEP_4 | 4 levels × 2 bits each (8 bits) | `base_amplitude * levels[i % 4]` |
| 7 | TRIANGULAR | upper 4 bits = `peak_index` (0-15), lower 4 bits = `peak_amplitude_q3` | rise linearly to peak at `peak_index`, then linear decay to `base_amplitude * (peak_amp_q3 / 8)` |
| 8-15 | reserved | — | — |

**ARP unification:** a FREQ arpeggio is `OSCILLATE_ENV(reg=FREQ_LO_voice, anchor=root_freq_quantized, base_amplitude=primary_interval_in_cents, period_frames=per_step_framerate, n_cycles=N, envelope_family=STEP_3, envelope_param=interval_pattern_bits)`. A 3-note arp like C-E-G becomes one atom (vs ~6-12 raw SETs today). Major (4,7) and minor (3,7) interval patterns fit cleanly into 2-bit step levels with the 4-level table `{minor_3rd, major_3rd, 5th, octave}`.

**Empirical coverage (2026-05-23, 3039 chains, 200 prodlike songs, fit tolerance 10%):**

| envelope_family | chains fit | pct of all chains | pct of fittable (n≥4) |
|---|---|---|---|
| (3-cycle special case — see below) | 1,678 | 55.2% | — |
| CONSTANT | 262 | 8.6% | 19% |
| STEP_2 | 188 | 6.2% | 14% |
| STEP_3 | 98 | 3.2% | 7% |
| LINEAR | 94 | 3.1% | 7% |
| EXP_DECAY | 52 | 1.7% | 4% |
| EXP_GROWTH | 36 | 1.2% | 3% |
| STEP_4 | 32 | 1.1% | 2% |
| **uncovered** (raw SLOPE rows fallback) | 599 | 19.7% | 44% |

**3-cycle chains (55% of all chains)** have only 2 distinct amplitudes; every parametric family fits trivially. They become OSCILLATE_ENV with `n_cycles=3` and either `envelope_family=CONSTANT` (if amps match within 10%) or `envelope_family=STEP_2` (otherwise). No fitting needed; deterministic emit rule in the detector.

**TRIANGULAR fit verified (2026-05-23, v2 probe run):** TRIANGULAR family captures **67 chains (2.2% of all, 5% of fittable)** — the rise-then-decay shapes shown in the uncovered examples. With TRIANGULAR added the family-fit table becomes:

| envelope_family | chains | pct of all |
|---|---|---|
| (3-cycle special case) | 1,678 | 55.2% |
| CONSTANT | 262 | 8.6% |
| STEP_2 | 188 | 6.2% |
| STEP_3 | 98 | 3.2% |
| LINEAR | 94 | 3.1% |
| TRIANGULAR | 67 | 2.2% |
| EXP_DECAY | 52 | 1.7% |
| EXP_GROWTH | 36 | 1.2% |
| STEP_4 | 32 | 1.1% |
| **uncovered (raw SLOPE rows)** | **532** | **17.5%** |

**Final coverage:** 2,507 of 3,039 chains (**82.5%**) get an OSCILLATE_ENV atom; 17.5% remain as raw SLOPE rows. Raw SLOPE rows are a valid trajectory primitive (already on the validator's allow-list), so the residual does **not** produce validator failures. The fail-fast gate (`uncovered < 25% of all chains` HARD KILL line) **passes**. The stricter `uncovered < 12%` pass-threshold is NOT met; this is an acceptable trade-off given the "parametric only, no library" design constraint — pushing below 12% would require either a library escape hatch (rejected) or relaxing fit tolerance from 10% to ~25% (trades audio fidelity for compression).

**Detector (`OscillationEnvelopePass`, runs after `slope` + `preset`, replaces the proposed `OscillationCollapsePass`):**

```
for each detected slope chain (>=3 alternating-sign slopes on same reg):
    amps = per-cycle amplitudes
    if len(amps) == 2:  # 3-cycle chain
        if amp_cv <= 0.10: emit OSCILLATE_ENV(family=CONSTANT)
        else:              emit OSCILLATE_ENV(family=STEP_2, level_ratio=amps[1]/amps[0])
        continue
    best_family, best_param, best_residual = fit_all_families(amps)
    if best_residual <= 0.10:
        emit OSCILLATE_ENV(family=best_family, param=best_param)
    else:
        # Leave as raw SLOPE rows; LonelyWriteValidatorPass will flag the
        # trajectory_anchor on the residual SET if any, and the SLOPE rows
        # themselves pass the validator (SLOPE is a trajectory primitive).
        keep_raw_slopes()
```

Fit priority (when multiple families fit, prefer simpler):
`CONSTANT > LINEAR > EXP_DECAY > EXP_GROWTH > STEP_2 > STEP_3 > STEP_4 > TRIANGULAR`.

**Subsumes prior proposals:**

| previously proposed | now expressed as |
|---|---|
| Item 1 vanilla OSCILLATE | `OSCILLATE_ENV(family=CONSTANT)` |
| Item 8 DAMPED_OSCILLATE | `OSCILLATE_ENV(family=EXP_DECAY)` |
| Item 9 GEOMETRIC slope | `OSCILLATE_ENV(family=EXP_GROWTH)` on the slope chain whose amps form a geometric progression |
| Existing FLIP_OP | `OSCILLATE_ENV(family=CONSTANT, n_cycles=1, period=2·gap)` |
| Existing FLIP2_OP | `OSCILLATE_ENV(family=STEP_2, n_cycles=N/2, period=2)` |
| Hypothesized ARP op | `OSCILLATE_ENV(reg=FREQ_LO, family=STEP_3 or STEP_4)` |

Once OSCILLATE_ENV lands, deprecate FLIP / FLIP2 (no-op then remove). Items 1, 8, 9 should be **struck from the doc** — they are subsumed.

**Locked detector defaults (probe-verified 2026-05-24):** the relaxed-threshold sweep absorbed 30% additional rows at zero implementation cost. Lock these as the spec defaults:
- `vibrato_min_cycles = 2` (was 3) — catches 2-cycle wiggles that 3-cycle threshold missed
- `portamento_min_steps = 3` (was 5) — catches 3-step glides
- `chain_proximity_window = 8 frames` — unchanged

Combined effect on DIFF absorption: 49,792 → 24,996 unmodelled DIFFs on FREQ_LO across 200 prodlike songs (-50%). See item 11 for the post-relax coverage table.

**Fail-fast plan:**
1. ~~Re-run the envelope-fit probe with TRIANGULAR added.~~ **DONE 2026-05-23 (v2).** Uncovered = 17.5% of all chains (39% of fittable). Below the HARD-KILL 25% line; above the strict-pass 12% line. **Cautious PASS** under the parametric-only constraint.
2. Implement `OscillationEnvelopePass` + `OscillationEnvelopeDecoder`. Tests:
   - Each envelope family round-trips: synthetic chain in → atom out → decode → original chain back, frame-by-frame within ±1 LSB at slope-terminal grid.
   - Fallback path: chains that don't fit emit raw SLOPE rows unchanged.
3. Round-trip audio fidelity on 100 sample songs. **PASS** if ≥95% within `FRAME_RMS_TOLERANCE`. **HARD KILL** if <90% pass or any `DRIFTING_DIVERGENCE` (envelope reconstruction is off-by-one in time).
4. Main-repo mini A/B (`oscillation_envelope_mini_body_large`, 3 seeds). **PASS** if val_acc within 1σ AND content vocab shrinks ≥10% AND op-tier vocab loses FLIP, FLIP2 entries.
5. ARP-specific gate: tokenize a hand-curated set of 5 known-arpeggio SIDs (e.g., Galway *Comic Bakery*, Hubbard *Commando*); confirm the arp regions emit STEP_3/STEP_4 OSCILLATE_ENV atoms with sensible interval params; render and audition.

**STEP-level tables (probe-verified 2026-05-23):** the static `{0.25, 0.5, 0.75, 1.0}` table is **wrong for STEP_3 and STEP_4**. Use per-N empirical tables:

| family | n chains fit | n samples | empirical k-means centroids | implementer table |
|---|---|---|---|---|
| STEP_2 | 175 | 945 | `{0.43, 0.97}` | `{0.5, 1.0}` — 1 bit per level (matches within 10%) |
| STEP_3 | 108 | 623 | `{0.28, 0.64, 0.96}` | `{0.28, 0.64, 1.0}` (or pick a clean approximation like `{0.25, 0.625, 1.0}` within 5%) |
| STEP_4 | 32 | 236 | `{0.18, 0.37, 0.61, 0.94}` | `{0.18, 0.37, 0.61, 0.94}` — corpus-derived; the obvious geometric `{0.125, 0.25, 0.5, 1.0}` is **off by 30%** |

This changes the envelope_param encoding for STEP_3 and STEP_4:
- STEP_3: 3 levels indexing a 3-entry table → `⌈log2(3³)⌉ = 5` bits of envelope_param used to pick one of 27 STEP_3 patterns; or 3 × 2 bits = 6 bits if encoded per-level with a 4-entry shared table (the original spec).
- STEP_4: same logic with 4 levels — 4 × 2 = 8 bits packed in envelope_param.

The implementer should pick the encoding scheme that minimizes vocab without losing fidelity. The 2-bit-per-level scheme works if the 4-entry table is the union `{0.18, 0.37, 0.61, 0.94}` (STEP_4 empirical, which subsumes STEP_3 and STEP_2 within 10%).

Probe scripts: `/scratch/tmp/probe_envelope.py` (family fit) + `/scratch/tmp/probe_step_levels.py` (level verification). JSON outputs in `/scratch/tmp/probe_token_rework_out/`.

### 1. OSCILLATE op (subsumes FLIP + FLIP2, generalizes both)

> **SUPERSEDED by item 0** (envelope-modulated OSCILLATE). Retained below for historical context; the detector params table and empirical chain stats from the probe still apply to item 0's CONSTANT envelope family. Do **not** implement this item separately — implement item 0 instead.

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

**Empirical detector params (2026-05-23, 200 prodlike songs):** 3039 chained-slope sequences detected. Distribution:

| param | finding | implementer guidance |
|---|---|---|
| chain location | reg=2 (PW_LO V0): 84% · reg=21 (FC_LO): 16% | Detector wires up these two regs first; voices 1/2 PW chains were absorbed by PRESET. |
| `n_cycles` | top-10 max = 13; 55% are 3-cycle; 99% are ≤9 | Subreg width = 5 bits (range 0-31) is safe; saturate at 31 if chain longer. |
| amplitude variance (cv across chain) | 33% exact (cv=0.00), only 36.5% within ±10% (`amp+period_ok` bucket); 50% within tight period but loose amplitude; 13% neither | ±10% amplitude tolerance is **too tight** — only 36.5% of chains fit. Recommend **±20% amplitude tolerance** (captures ~70%) AND surface chains that don't fit as separate `DAMPED_OSCILLATE` (item 8) candidates rather than dropping them. |
| period variance | ~85% within tight period (cv ≤ 0.50 OR stdev ≤ 1 frame) | ±1 frame period tolerance is correct; period_only bucket is the largest single class. |

So OSCILLATE captures **36.5% strictly** + **50.3% with the amplitude relaxation** → ~87% of chains. The remaining 13% (~400) split between truly non-oscillatory chains and item 8's damped class. Probe script: `/scratch/tmp/probe_token_rework.py` (one-off, not committed).

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

**Empirical result (2026-05-23, 200 prodlike songs):** **ZERO lonely PW SETs found in 200 prodlike songs.** PRESET (op=35) has absorbed every single one (consistent with item 10's finding that PRESET also absorbs all lonely-SET corrections after slope chains). **Verdict: ABANDON THIS LANE.** Item 3 produces no compression in the current macro pipeline; the anchor-fold extension to SLOPE/OSCILLATE is unnecessary because there are no naked PW anchors left to fold. (Caveat: if a future tokenizer pipeline disables PRESET or replaces it, re-run the probe.) Probe script: `/scratch/tmp/probe_token_rework.py`.

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

**Empirical result (2026-05-23, 200 prodlike songs):** in the **post-macro 0.parquet** stream, only 815 patches at K=2 / 906 at K=3 / 1196 at K=5 ≈ **~6 patches/song remaining**, all participation pattern `{AD, CTRL, SR}`. The 275/song figure in the empirical baseline was measured on **pre-macro dump.parquet**; PRESET (op=35) + CTRL_BIGRAM (op=42) have **already absorbed ~97% of patch swaps** in the current pipeline. Validator-failure-count drop estimate: only ~1200 atoms over 200 songs (single-digit percent of the 20,691 unmodelled SETs).

**Revised verdict:** item 7 is **low-impact** as currently scoped (~6/song wins). Two options:

1. **DEPRIORITIZE.** Land items 1, 2, 8, 9, 11 first; revisit item 7 only if PATCH_SET emerges as a top-10 unmodelled signature after those primitives reduce the long tail.
2. **RE-SCOPE.** Move `PatchSetPass` to run **before** PRESET in the pipeline, replacing PRESET on patch-swap windows. This would cleanly attribute the compression to PATCH_SET and may shrink the PRESET sub-vocab, but requires reasoning about PRESET-to-PATCH_SET commutation and is a larger change.

Probe script: `/scratch/tmp/probe_token_rework.py`.

### 8. DAMPED_OSCILLATE (or OSCILLATE decay parameter)

> **SUPERSEDED by item 0.** The damped-oscillation case is the `EXP_DECAY` envelope family. Do not implement separately.

**Problem:** **7027 damped-oscillation 4-tuples in FREQ_LO** (alternating-sign deltas with decreasing magnitude) currently encoded as 4+ SET atoms each. OSCILLATE with fixed amplitude doesn't fit.

**Two design options:**

**Option A (preferred):** add a decay parameter to OSCILLATE. Subreg 4 (new) = `decay_rate_per_cycle` (0 = constant amplitude = vanilla OSCILLATE; positive = amplitude shrinks). 1 atom for both vanilla and damped cases.

**Option B:** separate DAMPED_OSCILLATE op. Simpler decoder but doubles vocab footprint of oscillation atoms.

**Fail-fast:** measure how many damped-oscillation windows in real corpus have approximately-exponential decay (where Option A's single decay_rate captures it well) vs more complex envelopes (where neither option captures faithfully — fall back to chain-of-SETs). If ≥80% of damped-osc windows are well-fit by single decay_rate, Option A is sufficient.

### 9. GEOMETRIC slope variant (exponential progressions)

> **SUPERSEDED by item 0.** Geometric progressions are the `EXP_GROWTH` envelope family. Do not implement separately.

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

**Empirical result (2026-05-23, 100 prodlike songs, dataset_cache hash `991929565a378250`):** **1826 chained-slope sequences found; ZERO followed by a raw lonely SET (op=0).** Chains concentrate on reg=2 (PW_LO v1, 85%) and reg=21 (FC_LO filter, 15%). Next-write-on-same-reg distribution:

| next op | count | pct |
|---|---|---|
| PRESET (op=35) | 1264 | 69.2% |
| next-chain SLOPE (op=33/34) | 296 | 16.2% |
| op=36 (CTRL_BIGRAM-class) | 248 | 13.6% |
| no follow-up within 200 rows | 17 | 0.9% |
| **raw SET (op=0)** | **0** | **0.00%** |

**Verdict:** in the current prodlike tokenizer, **lonely-SET corrections do not exist as a distinct class** — PRESET, follow-on chains, and bigram macros already absorb every would-be correction. OSCILLATE can adopt the **drop-safe path** (no `terminal_correction` subreg needed) with respect to lonely SETs.

Caveats:
- Does not measure whether PRESET / next-chain / op=36 followups themselves act as corrections that OSCILLATE-collapse could disturb. Those are *not* lonely (they fire alongside other coordinated writes), so they fall outside the zero-lonely mandate's scope, but a sequel probe should confirm OSCILLATE-reconstructed chains followed by PRESET render audibly equivalent to the originals.
- Probe operates on the post-macro-pass parsed `0.parquet`. Pre-macro raw SET corrections that PRESET absorbed are now part of PRESET's value — the probe correctly counts those as PRESET, not SET. A separate probe on `dump.parquet` would expose pre-tokenization correction rates if the mandate ever needs to be re-checked against a different macro pipeline.

Probe script: `integration_tests/profile/audit_oscillate_overshoot.py` (main preframr repo).

### 11. Lonely-write strict-mode validator (parser fails on unmodelled lonelies + all DIFF)

**What:** a terminal pipeline pass `LonelyWriteValidatorPass` that runs LAST (after every macro/trajectory pass has had its chance to absorb writes). It scans the parsed stream and **raises** if (a) any voice-register SET remains that is neither part of a trajectory primitive nor on the explicit carveout allow-list, OR (b) any DIFF op is emitted on any register. Behind a single `--strict-lonely` flag (default ON for new builds, OFF for legacy compatibility); flag-ON corpora cannot tokenize until the macro pipeline covers all observed patterns and the DIFF pass has been removed.

**Strict-no-diff rationale (2026-05-23):** an empirical scan showed 100% of DIFF emissions (104,949 in 200 prodlike songs) are on FREQ_LO, with median delta 6 LSB / p95 30 LSB — the territory of vibrato (regular oscillation) and portamento (small monotonic steps). These are properly the domain of OSCILLATE_ENV (item 0) and SLOPE post-cents=10 (item 2). DIFF is filling a gap that trajectory primitives should close. Forcing the validator to reject DIFF makes that gap visible at parse time rather than silently leaking smooth-motion atoms into the model's long tail. Once items 0, 2, and FREQ_RUN are in, the DIFF pass is deleted from the macro pipeline (vs. left running with the validator rejecting its output).

**Why:** the zero-lonely-updates mandate is currently aspirational — nothing in the pipeline enforces it. New trajectory primitives can land without coverage gaps being noticed, because uncovered patterns silently fall through to raw SET. A hard-fail validator turns silent coverage gaps into immediate tokenizer errors, forcing every new pattern observed in the corpus to either get its own primitive or be added to the allow-list with explicit justification.

**Why:** keeps the long-tail-SET pressure on the macro pipeline rather than the model. The model's content-tier ceiling is a function of how many distinct rare SETs it has to memorize; a strict validator guarantees that distribution shrinks monotonically as primitives land, with no quiet regression when a new corpus introduces an unmodelled pattern.

**How to apply:** any new preframr-tokens release with `--strict-lonely` ON must tokenize the prodlike corpus end-to-end without raising; if a class of lonely SETs is too small to deserve its own primitive, add it to the carveout allow-list with a comment citing the empirical fraction and the audio-fidelity audit that confirmed dropping/keeping it.

**Schema (per-write classification):**

```
for each row in parsed_df:
    if row.op in TRAJECTORY_OPS:    # SLOPE, OSCILLATE, PRESET, FLIP, FLIP2,
        ok                          # PATCH_SET, DAMPED_OSCILLATE, GEOMETRIC,
                                    # CTRL_BIGRAM, HARD_RESTART, DIFF
    elif row.op == SET_OP:
        carveout = classify_carveout(row, surrounding_window)
        if carveout is None:
            raise UnmodelledLonelyWriteError(row, surrounding_window)
        else:
            row.metadata["carveout"] = carveout  # tagged for downstream stats
    else:
        ok  # FRAME, DELAY, other infrastructure rows pass through
```

**Carveout classifier (matches the 5 carveouts from the mandate):**

| carveout id | matches | check |
|---|---|---|
| `first_voice_write` | first SET on (voice, reg) in the song | precompute first-seen index per (voice, reg) |
| `gate_off_terminal` | SET after CTRL gate-off + ≥K stationary frames | look back: prior CTRL write has gate=0 AND ≥K frames since |
| `filter_route` | SET to reg 23 (filter routing topology) | row.reg == 23 |
| `master_volume` | SET to reg 24 | row.reg == 24 |
| `trajectory_anchor` | first SET in a SLOPE/OSCILLATE atom's anchor position | flagged by SLOPE/OSCILLATE detector before validator runs |

Any SET not matching one of the above is an `UnmodelledLonelyWriteError`. The error message includes the song path, row index, (op, reg, val), a ±10-row context window, and a one-line summary of which trajectory primitives were checked and missed. This makes failures actionable: the engineer sees exactly which pattern is uncovered.

**Fail-fast plan:**
1. Implement `LonelyWriteValidatorPass` + `classify_carveout`. Tests: synthetic streams that should pass; streams that violate each carveout fire `UnmodelledLonelyWriteError` with the right field set; allow-listed patterns pass without raising.
2. Run `--strict-lonely` ON against the current prodlike corpus. **Expected outcome:** fails on day-1 because the current pipeline leaves ~40% of writes as raw SET. The validator emits a per-(op, reg, surrounding-context-signature) histogram of failures so the next primitives to build can be prioritized by failure-count.
3. Each subsequent primitive (OSCILLATE, PATCH_SET, DAMPED_OSCILLATE, etc.) lands paired with the validator failure-count drop it produces. The mandate is satisfied when `--strict-lonely` passes on the full prodlike corpus.
4. Once the corpus passes, flip `--strict-lonely` to default ON for all new tokenizer builds. The carveout allow-list becomes the canonical specification of "lonely writes that are by-design", and any future corpus that introduces a new violator forces explicit triage.

**Combines with item 10:** the under/overshoot probe verdict (no raw-SET corrections in prodlike) means the validator can treat the post-OSCILLATE-chain region as fully-covered without a `terminal_correction` carveout. If a future corpus DOES produce raw-SET corrections there, the validator will fail and force the question to be re-opened.

**Tooling cost:** one pipeline pass (~50 LOC) + carveout classifier (~150 LOC) + test fixtures. ~1 day eng. No runtime cost beyond an O(N) scan.

**Empirical day-1 calibration (2026-05-23, 200 prodlike songs, 79,624 raw SETs scanned):**

Carveout coverage (with the tight gate_off_terminal predicate: CTRL-reg write whose new val AND prev val both have gate=0):

| carveout | hits | pct of all SETs |
|---|---|---|
| `gate_off_terminal` | 57,608 | 72.4% |
| `first_voice_write` | 802 | 1.0% |
| `trajectory_anchor` (precedes SLOPE within 5 rows) | 401 | 0.5% |
| `filter_route` (reg=23) | 122 | 0.2% |
| `master_volume` (reg=24) | 0 | 0.0% |
| **UNMODELLED** | **20,691** | **26.0%** |

Unmodelled breakdown by reg-class: CTRL 9,830 (47.5%), FREQ_LO 9,768 (47.2%), SR 1,067 (5.2%), AD 26 (0.1%). **The two dominant unmodelled classes are CTRL (gate-on toggling that isn't bigram-absorbed) and FREQ_LO (frequency setpoints with no slope chain).** Items 1, 2, 7, 8, 9 between them target these two classes; the validator's failure-count drop after each primitive lands is the gating metric.

Top-15 unmodelled `(reg_class, prev_op, next_op)` context signatures (drives next-primitive prioritization):

| reg_class | prev_op | next_op | count |
|---|---|---|---|
| FREQ_LO | BOS | EOS | 4,740 |
| CTRL | SET | EOS | 3,204 |
| CTRL | BOS | EOS | 2,990 |
| CTRL | SET | SET | 1,344 |
| FREQ_LO | BOS | SET | 1,101 |
| FREQ_LO | SET | EOS | 1,011 |
| SR | BOS | EOS | 981 |
| FREQ_LO | BOS | DIFF | 860 |
| CTRL | BOS | SET | 787 |
| CTRL | BOS | CTRL_BIGRAM | 711 |
| FREQ_LO | DIFF | EOS | 611 |
| FREQ_LO | SET | SET | 603 |
| CTRL | SET | CTRL_BIGRAM | 424 |
| FREQ_LO | SET | DIFF | 397 |
| CTRL | CTRL_BIGRAM | EOS | 173 |

(BOS/EOS = "no same-reg row within ±4 rows" — i.e., this write is genuinely isolated on its reg.)

**Implementer takeaways:**
1. The day-1 validator failure rate is ~26% (20,691 / 79,624). Treat ~26% as the baseline that should monotonically decrease as primitives land.
2. **Highest single-pattern wins:** FREQ_LO/BOS/EOS (4,740) absorbed by item 2 (FREQ slope-aware quantize) and possibly a new FREQ_SETPOINT primitive for truly isolated values. CTRL/SET/EOS + CTRL/BOS/EOS (6,194 combined) need a CTRL primitive beyond CTRL_BIGRAM — the implementer should consider whether the CTRL writes are state-transition pairs that a CTRL_TRIPLE op could absorb, or whether they're already-isolated state changes that fold into the gate_off_terminal carveout with a relaxed predicate.
3. **`trajectory_anchor` is small (0.5%)** — confirms most SLOPE chains start at values that were absorbed by PRESET, not at lonely SET anchors. The anchor-fold extension to SLOPE/OSCILLATE (proposed in item 3) is **unnecessary** for the current pipeline.
4. **AD is essentially solved (26 unmodelled in 200 songs).** No new primitive needed for AD.
5. The probe script (`/scratch/tmp/probe_token_rework.py`) is the reference implementation of the carveout classifier the validator should ship — copy its `classify_carveout` logic into `LonelyWriteValidatorPass`.

### Day-N coverage under strict-no-diff (2026-05-24)

Whole-row simulation across all 1,116,032 rows in 200 prodlike songs, with proposed primitives (item 0 OSCILLATE_ENV at v=2/p=3, item 2 SLOPE post-cents=10, item 7 PATCH_SET revived, FREQ_RUN, NOTE_ON@w=12) folded in:

| bucket | rows | pct of non-INFRA (609,598) |
|---|---|---|
| PRESENT (existing primitives) | 425,025 | 69.7% |
| PROPOSED (would be absorbed) | 121,132 | 19.9% |
| **UNMODELLED (validator fails)** | **58,087** | **9.5%** |
| CARVEOUT | 5,354 | 0.9% |

Residual unmodelled breakdown (58,087 total / ~290 per song):

| class | count | dominant pattern |
|---|---|---|
| DIFF/FREQ_LO | 24,996 | isolated 1-DIFF events not in any chain or FREQ_RUN window |
| SET/FREQ_LO | 25,878 | isolated FREQ_LO writes (mostly BOS/EOS — no nearby same-reg context) |
| SET/SR | 5,378 | release-time changes without coupled AD or CTRL |
| SET/CTRL | 1,802 | single CTRL state changes outside any CTRL_BIGRAM |
| SET/AD | 33 | essentially solved |

### Four new primitives required to reach zero strict-no-diff failures

| primitive | proposed schema | absorbs | residual after |
|---|---|---|---|
| **`FREQ_NUDGE`** | op=FREQ_NUDGE, reg=FREQ_LO_voice, subreg 0: signed_delta (8-bit), subreg 1: is_absolute_or_delta_bit (so absolute-SET and DIFF use the same atom) | 24,996 isolated DIFFs + (possibly merged) 25,878 isolated FREQ_LO SETs | ~7,200 |
| **`RELEASE_UPDATE`** | op=RELEASE_UPDATE, reg=SR_voice, subreg 0: new SR byte | 5,378 isolated SR changes (release-time tweaks between notes) | ~1,800 |
| **`CTRL_TRIPLE`** or extended `CTRL_BIGRAM` adjacency | greedy-extend CTRL_BIGRAM to absorb a third adjacent CTRL within the window | 1,802 unabsorbed CTRL writes | ~0 (rest are carveouts) |
| TRANSPOSE/FLIP anchor extension | extend the existing TRANSPOSE and FLIP detectors to absorb a leading/trailing same-reg SET as the trajectory's anchor | ~500 anchor SETs adjacent to TRANSPOSE/FLIP atoms | residual hits the carveout list |

Recommended landing order (by absorption count): `FREQ_NUDGE` → `RELEASE_UPDATE` → `CTRL_TRIPLE` → TRANSPOSE/FLIP anchor extension.

**Open question for FREQ_NUDGE:** the empirical split is 24,996 DIFF + 25,878 SET ≈ 50k isolated FREQ_LO events. Two encodings possible: (a) one unified `FREQ_NUDGE` op with an is_absolute bit that accepts either form, (b) two separate ops `FREQ_DIFF` and `PITCH_SETPOINT`. Option (a) shrinks op-tier vocab by 1 entry and keeps the model's "isolated FREQ event" prediction as one categorical; option (b) is closer to existing op-style conventions (each op fixed-meaning). Recommend (a) unless the implementer finds a reason the model needs to discriminate the two cases at op-tier time. (Sub-probe: does the surrounding context predict which form fires? If yes, the model would benefit from the discrimination being explicit at op-tier — go with (b). If not, (a) is the clean win.)

**FREQ_NUDGE is intentionally per-voice (architectural constraint).** The prodlike tokenizer splits each song into three per-voice parquets with register numbers canonicalized to voice 0's layout. Each voice is an independent training stream. A FREQ_NUDGE atom on voice 1's parquet **cannot reference voice 0's FREQ value** — voice 0 isn't visible at that layer. Cross-voice tuning relationships (the unison/octave/fifth patterns covering ~30-40% of frames in raw dumps — see item 12) require a separate macro that fires before the per-voice split. FREQ_NUDGE handles the residual single-voice events that no cross-voice macro absorbs.

### 12. VOICE_TRACK macro (pre-per-voice-split cross-voice tuning)

> Empirically motivated by the cross-voice tracking probe (2026-05-24): raw `dump.parquet` shows voice pairs in stable musical-interval relationships (unison, octave, fifth, etc.) for 30-40% of frames; the per-voice tokenizer discards this signal at the split step. VOICE_TRACK is the macro that captures it before the split.

**What:** a new pre-split macro `VoiceTrackPass` that runs on `dump.parquet` BEFORE the per-voice canonicalization step. It detects spans where one voice's FREQ value is a stable musical-interval (or detuned-unison) multiple of another voice's FREQ value, and emits a `TRACK_REF` atom on the tracker voice's stream that replaces the tracker's redundant FREQ writes for that span.

**Schema:**

| op | reg | subreg 0 | subreg 1 | subreg 2 | subreg 3 |
|---|---|---|---|---|---|
| `TRACK_REF_OP` (new) | tracker voice's FREQ_LO (canonicalized to 0) | `lead_voice_id` (0-2) | `interval_id` (5 bits, table-indexed) | `detune_cents` (signed 8-bit, for unison-with-detune; 0 for non-unison intervals) | `duration_frames` (8 bits, saturate at 255) |

**Interval table (16 entries, 4 bits):**

| id | name | ratio |
|---|---|---|
| 0 | unison | 1.00 |
| 1 | minor 2nd | 1.06 |
| 2 | major 2nd | 1.12 |
| 3 | minor 3rd | 1.19 |
| 4 | major 3rd | 1.26 |
| 5 | fourth | 1.33 |
| 6 | tritone | 1.41 |
| 7 | fifth | 1.50 |
| 8 | minor 6th | 1.59 |
| 9 | major 6th | 1.68 |
| 10 | minor 7th | 1.78 |
| 11 | major 7th | 1.89 |
| 12 | octave | 2.00 |
| 13 | octave + 5th | 3.00 |
| 14 | two octaves | 4.00 |
| 15 | reserved / future | — |

**Detector (`VoiceTrackPass`, runs on dump.parquet):**

```
for each frame:
    for each ordered voice-pair (lead, tracker) where lead < tracker:
        compute lead_freq = (FREQ_HI[lead] << 8) | FREQ_LO[lead]
        compute tracker_freq = (FREQ_HI[tracker] << 8) | FREQ_LO[tracker]
        ratio = max(lead, tracker) / min(lead, tracker)
        match ratio to nearest interval (±2% tolerance)
        if match: extend the current TRACK_REF candidate for (lead, tracker, interval)
        else: close any open candidate; if duration >= MIN_TRACK_DURATION (default 10 frames),
              emit TRACK_REF on tracker voice's stream; otherwise discard candidate.
```

**Lead-voice selection rule:** when both `(v0, v1)` and `(v0, v2)` and `(v1, v2)` could fire, pick the lead with the most FREQ writes in the span (the "active melody voice"); break ties by lowest voice id. Avoids double-counting and keeps the encoding deterministic.

**Per-voice split treatment:** AFTER detection, when the dump is split into per-voice parquets:
- The **lead voice's parquet** contains its own FREQ writes unchanged. No marker needed (lead doesn't know about its trackers).
- The **tracker voice's parquet** contains the `TRACK_REF` atom **in place of** the redundant FREQ writes for the span. During inference, the tracker's FREQ values are reconstructed by reading the lead voice's parquet at the matching frame index and applying the interval.

**Decoder coordination at inference time:** voices must be decoded in **lead-first order**. The renderer reads all three per-voice parquets and resolves each `TRACK_REF` atom by looking up `(lead_voice_id, frame_offset)` in the already-decoded lead stream. This requires inference-time generation order: `voice0 → voice1 → voice2` (or per-section based on which is the active melody). The orchestrator must enforce this ordering; cannot generate voices in parallel.

**Model-side implication (the trade-off):** the model now sees `TRACK_REF` as a vocab atom but doesn't have access to the lead voice's stream during training of the tracker voice's stream. Two paths:

1. **Independent training (smaller change).** Train each voice's stream independently. The model learns "voice 1 often emits `TRACK_REF(lead=0, interval=octave)` after gate-on" as a statistical prior. At inference, `TRACK_REF` is sampled like any other atom; the renderer fills in the actual FREQ values from the lead voice's already-generated stream. The model never sees the lead's FREQ values, but it learns the *pattern* of tracking.

2. **Cross-voice context training (larger change).** Provide the lead voice's already-generated stream as a context prefix when training the tracker voice. Model has explicit access to what it's tracking. Bigger model-side change, may need cross-attention or stacked-voice context.

Recommend **(1)** for landing — keeps the model architecture unchanged. If diversity_ratio / generation quality on tracker voices remains weak, escalate to (2).

**Compression estimate (sweep-verified 2026-05-24):** at the locked threshold `MIN_TRACK_DURATION=10 frames`, the dump-side sweep across 50 prodlike dumps measured **61,527 tracker-voice FREQ writes absorbed in 5,967 tracked spans**, or **~1,231 per song**. Extrapolated to 200 songs: **~246,000 tracker FREQ writes absorbed**. Interval distribution at T=10:

| interval | spans |
|---|---|
| two_octaves | 1,131 |
| octave | 1,044 |
| fifth | 703 |
| unison | 688 |
| major 3rd | 358 |
| (others <300 each) | rest |

These distributions match standard SID composition idioms (octave doubling for thickness, unison-with-detune for chorus, fifth/octave-plus-fifth for the classic bass-+-harmony pattern). The interval table size of 16 entries is empirically validated.

**Knee analysis:** the sweep showed marginal absorption between thresholds:

| T | absorbed | Δ vs prev T |
|---|---|---|
| 4 | 99,088 | — |
| 6 | 81,870 | -17,218 |
| 8 | 67,586 | -14,284 |
| 10 | 61,527 | **-6,059 (knee)** |
| 12 | 52,049 | -9,478 |
| 16 | 40,009 | -12,040 |
| 20 | 34,268 | -5,741 |

T=10 sits on the knee: the absorption loss going T=8→T=10 (6,059) is half the loss going T=4→T=6 (17,218) and the loss going T=10→T=12 (9,478). Below T=10 the additional absorptions are likely 4-7 frame chord-change coincidences (false tracking); above T=10 the loss is real long-tracking that should have been absorbed.

**Interaction with FREQ_NUDGE:** VOICE_TRACK runs first (catches sustained tracking spans); FREQ_NUDGE absorbs the residual single-voice isolated events. No double-counting because TRACK_REF replaces tracker's FREQ writes for the span.

**Interaction with the validator (item 11):** `TRACK_REF` joins the trajectory-primitive allow-list; gate-on/-off transitions within a TRACK_REF span are still emitted normally (the gate doesn't track). The validator additionally requires that every `TRACK_REF` atom on a tracker voice has a matching frame range in the lead voice's stream — a cross-stream consistency check at validation time only (cheap).

**Fail-fast plan:**
1. Implement `VoiceTrackPass` operating on `dump.parquet`. Tests:
   - Synthetic 3-voice dump where voice 1 = voice 0 + octave for 50 frames → emit one `TRACK_REF` atom in voice 1's parquet covering the span; voice 0's parquet unchanged.
   - Synthetic dump where ratio is musical-interval for only 5 frames → no `TRACK_REF` (below `MIN_TRACK_DURATION`).
   - Synthetic dump with three-way unison → lead is voice 0 (lowest id); both v1 and v2 get `TRACK_REF` atoms referencing v0.
2. Run on 50 prodlike songs; confirm TRACK_REF emission count matches the probe's estimate (~825-1,100 per song; allow ±30%).
3. Round-trip audio fidelity audit (`compare_renders`): render the original vs the TRACK_REF-encoded version. **PASS** if ≥95% within `FRAME_RMS_TOLERANCE`. **HARD KILL** if <90% pass — means the interval table or `MIN_TRACK_DURATION` is wrong.
4. Validator consistency check: each emitted `TRACK_REF` resolves to a non-empty lead-voice FREQ slice. Hard error if not.
5. Mini A/B in main repo: baseline vs `voice_track_mini_body_large`, 3 seeds. **PASS** if val_acc within 1σ AND tracker-voice content vocab shrinks ≥15% AND no diversity_ratio regression on real-stream prompts.

**Out-of-scope follow-ups (queue for after this lands):**
- `VOICE_RHYTHMIC_LOCK`: same melody on two voices with a fixed frame offset (e.g., voice 1 plays voice 0's melody delayed by 4 frames — chorus effect). Different macro, similar shape.
- `VOICE_DETUNE_OSCILLATE`: when one voice tracks another with a *modulated* detune (slow vibrato in the offset). Probably needs an envelope on the `detune_cents` subreg; defer until the static `TRACK_REF` lands.
- Investigate whether the per-voice tokenizer split should be reconsidered at all: VOICE_TRACK recovers cross-voice signal at the macro layer but doesn't restore the temporal alignment in training. A full per-song multi-voice stream is a separate larger architectural question.

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

## Empirical TODO

**Done 2026-05-23 (200 prodlike songs from dataset_cache `991929565a378250`, ~5 min CPU):**

1. ~~Under/overshoot correction audio test (item 10)~~ — 1826 chains, 0 raw-SET corrections. OSCILLATE drop-safe. *Verdict in item 10.*
2. ~~Strict-lonely simulator (item 11)~~ — 79,624 SETs scanned, 26% UNMODELLED. Carveout coverage table + top-15 unmodelled signatures. *Calibration block in item 11.*
3. ~~OSCILLATE tolerance + n_cycles distribution (item 1, superseded by item 0)~~ — 3039 chains. ±10% amp tolerance too tight (36.5% fit); recommend ±20%. n_cycles ≤ 13; 5-bit subreg. *Detector params table in item 1 (historical).*
4. ~~PATCH_SET coupling window (item 7)~~ — only ~6 patches/song remaining post-PRESET; item 7 deprioritized. *Verdict in item 7.*
5. ~~Lonely PW trajectory classification (item 3)~~ — 0 lonely PW SETs in 200 songs (PRESET absorbed all). Item 3 abandoned. *Verdict in item 3.*
6. ~~Envelope-family fit (item 0 unification)~~ — 8 candidate families fit to all 3039 chains; v2 (with TRIANGULAR) covers 82.5% of chains; 17.5% remain raw SLOPE. Cautious PASS for parametric-only design. *Coverage table + family spec in item 0.*
7. ~~STEP level table verification (item 0)~~ — empirical centroids: STEP_2 `{0.43, 0.97}`, STEP_3 `{0.28, 0.64, 0.96}`, STEP_4 `{0.18, 0.37, 0.61, 0.94}`. The static `{0.25, 0.5, 0.75, 1.0}` table is wrong for STEP_3/4. Per-N empirical tables now in item 0.
8. ~~Strict-no-diff coverage simulation (items 0/2/7/11 + FREQ_RUN/NOTE_ON)~~ — **DONE 2026-05-23.** All ops classified; 84,037 day-1 unmodelled rows (13.8% of non-INFRA) under v=3/p=5/w=6 defaults. Five gap classes named; four new primitives sized. *Coverage table in item 11.*
9. ~~Relaxed-detector + NOTE_ON window sweep~~ — **DONE 2026-05-24.** Locked defaults (v=2, p=3, w=12) drop unmodelled to 58,087 (-31%); FREQ_NUDGE still needed (50% of DIFFs remain isolated). NOTE_ON knee at w=12 (marginal absorption halves between w=12→16). *Locked defaults + new-primitive recommendations in items 0 and 11.*
10. ~~Cross-voice tracking hypothesis (raw dump.parquet)~~ — **DONE 2026-05-24.** Voice pairs in raw dumps are in stable musical-interval relationships 30-40% of frames; the per-voice tokenizer split discards this signal. Spawned item 12 (VOICE_TRACK macro). Also surfaced architectural finding: prodlike pipeline emits one per-voice parquet with canonicalized regs — cross-voice references are impossible at the tokenizer layer post-split. *Architectural note in item 11, macro spec in item 12.*
11. ~~100% accounting of strict-no-diff residual (item 11)~~ — **DONE 2026-05-24.** All 58,087 unmodelled rows from the v=2/p=3/w=12 sweep mapped to 61 distinct signatures, each assigned to a primitive/extension/carveout. Zero genuine gaps. Six-primitive minimum set published as the canonical implementer spec at the top of the doc.
12. ~~VOICE_TRACK MIN_TRACK_DURATION sweep (item 12)~~ — **DONE 2026-05-24.** Knee at T=10 frames (61,527 absorbed in 50 songs); locked as default. Interval distribution validates the 16-entry table. *Sweep table in item 12.*

**Pending:**

6. PCM_BITS slope-detection sweep (item 4 fail-fast). ~2 hr.
7. FREQ slope count at `cents=10` vs `cents=50` on 100 sample songs (item 2 fail-fast step 3). ~30 min.
8. Damped-oscillation decay-fit goodness probe (item 8 fail-fast). ~30 min.
9. Geometric slope ratio quantization probe (item 9 fail-fast). ~30 min.

All pending probes are ≤2 hr each on CPU. The unified probe script is `/scratch/tmp/probe_token_rework.py` (one-off; rebuild for items 4/2/8/9 by adding new probe functions to the same harness).

**Implementer notes:** the empirical findings above resolve the major design uncertainties for items 1, 3, 7, 10, 11. The implementer can proceed directly to coding `LonelyWriteValidatorPass` (item 11) and `OscillationCollapsePass` (item 1) using the calibration tables in those sections; items 3 and 7 should be **skipped or re-scoped** based on the empirical results.

## Implementation log (2026-05-24, preframr-tokens repo)

Implementer agent landing the canonical spec in strict doc order. All corpus / audio-fidelity / model-A·B / WAV-audition gates are **postponed** per the maintainer until every primitive is coded; the notes below are the design decisions and issues found while coding, each backed by synthetic round-trip unit tests (the doc's prescribed per-primitive gate). `preframr-audio` 0.3.0 is installed; the prodlike `dataset_cache/991929565a378250` is **not present on this machine**, so no corpus probe was re-run — all empirical numbers in this doc are taken as given.

### OSCILLATE_ENV (item 0) — DONE, default ON

- New `OSCILLATE_ENV_OP = 45`; 8-subreg atom (anchor hi/lo, amplitude hi/lo, period, ncycles+start-direction bit, family, param). Byte-split 16-bit fields to bound vocab, mirroring SLOPE's terminal split.
- `macros/envelope.py` holds the 8 families + fitter, shared by encode and decode.
- **Issue / decision — per-slope, not per-full-cycle amplitude.** The doc schema describes a per-*cycle* envelope. I model each SLOPE atom's (half-cycle) excursion directly so every original slope round-trips *exactly* rather than via a lossy crest/trough average. `NCYCLES` field therefore holds the slope (half-cycle) count and `PERIOD` holds frames-per-slope.
- **Issue / decision — uniform-runtime requirement + raw-slope fallback.** Exact frame-count reconstruction needs equal per-slope runtimes. PW's grid-32 terminal quantization frequently yields non-uniform runtimes (e.g. `[4,5,4,5]`), in which case the pass declines and leaves raw SLOPE rows (a valid trajectory primitive — no validator failure). Consequence: PW chains compress less than the doc's headline numbers until this is relaxed; revisit against the corpus in the validation phase (a per-slope runtime correction sub-field, or a total-runtime even-split, would lift coverage at some round-trip cost).
- **Deferred — FLIP/FLIP2 deprecation.** The doc says deprecate FLIP/FLIP2 once OSCILLATE_ENV lands, but open question #4 flags the checkpoint-invalidation timing as undecided, so they are left in place.

### LonelyWriteValidatorPass (item 11) — DONE, default OFF (`strict_lonely`)

- `macros/lonely_validator.py`: raises `UnmodelledLonelyWriteError` for any full SET (subreg −1) not matching a carveout, and for any DIFF op. Carveout classifier ports the reference `classify_carveout` logic; `trajectory_anchor` recognises both SLOPE and OSCILLATE_ENV anchor-hi rows.
- **Issue / decision — default OFF.** The doc wants it default-ON for new builds, but the current pipeline still leaves ~26% of writes as raw SET, so ON would fail day-1 (and break the test suite). It is wired in `parse()` after the post-norm passes, gated off; flipping it ON is part of the validation phase once the residual-absorbing primitives have done their work.

### VOICE_TRACK (item 12) — DONE in this repo, default OFF (`voice_track_pass`)

- New `TRACK_REF_OP = 46`; 4-subreg atom (lead voice, interval id, signed detune, duration). 16-entry interval table per the doc. Reconstruction adds a `pending_track_links` queue to `DecodeState`, drained per frame by `tick_frame` (mirrors the existing `interval_links` machinery).
- **Issue / decision — single combined-voice stream.** This repo (unlike the main repo's per-voice parquet split) keeps all three voices in one stream through tokenization, so the TRACK_REF decoder *can* read the lead voice's FREQ directly from `state.last_val`. This is why VOICE_TRACK is implementable here at all. **The main-repo lead-first decode-ordering dependency the doc describes is NOT addressed here** (per maintainer instruction) — under the main repo's per-voice split this decoder would not see the lead voice and would need the cross-voice context the doc's design section discusses.
- **Issue / decision — runs post-quantize on combined FREQ, exact-match only.** The pass runs after `squeeze_frame_regs` (FREQ already combined to one 16-bit reg per voice and cent-quantized; frame markers present), not on the raw `dump.parquet` as the doc describes. It emits a TRACK_REF only where `tracker == round(lead*ratio) + constant_detune` holds for *every* frame of a consecutive-frame span (≥ `TRACK_MIN_DURATION=10`), giving exact round-trip; the doc's ±2% ratio tolerance is intentionally not used (it would trade audio fidelity for coverage). Requires the tracker to write FREQ every frame in the span and the lead's write to precede the tracker within each frame; both are validation-phase robustness items.
- **Issue / decision — default OFF.** Highest round-trip risk of the set plus the unaddressed main-repo ordering dependency, so it ships gated; the validation phase flips it on after corpus + audio-fidelity checks.

### FREQ_NUDGE (spec #3) — DONE, default OFF (`freq_nudge_pass`)

- New `FREQ_NUDGE_OP = 47`; 3-subreg atom (mode, hi, lo). Mode delta adds a signed 16-bit payload (replaces DIFF so strict-no-diff can pass); mode absolute sets a 16-bit value (isolated FREQ setpoint).
- **Issue / decision — 16-bit hi/lo payload, not the doc's 8-bit delta.** Post-`quantize_freq_to_cents` FREQ is a bin index (~240 at cents=50 but larger at finer `cents`), so an 8-bit field is unsafe; hi/lo round-trips at any setting at the cost of one extra subreg.
- **Issue / decision — wired post-`_apply_optional_transforms`.** DIFF is produced by the optional `set_to_diff` transform, which runs last, so FREQ_NUDGE must run after it. By then `add_voice_reg` has canonicalised regs to voice-0 layout (all FREQ on reg 0), so the live pass operates on reg-0 FREQ and its voice-aware isolation heuristic is approximate. Synthetic tests exercise the pre-voice 0/7/14 layout. Voice-aware placement is a validation-phase item.

### FREQ_RUN (spec #4) — DONE, default OFF (`freq_run_pass`)

- New `FREQ_RUN_OP = 48`. No prior FREQ_RUN existed, so this is a new primitive (not literally an "extension"): a maximal run of consecutive-frame FREQ SETs (length ≥ 2, capped at 16/atom) becomes a count + value-list atom, replayed exactly on decode (value 0 written at the atom frame, the rest queued per frame).
- **Issue / decision — stores all values (exact), not a pattern.** Arithmetic runs are already taken by SlopePass (≥5); the residual short/irregular runs have no compact pattern, so FREQ_RUN stores the literal values. The win is op-tier (one run op vs K SET ops) and removing lonely SETs, not raw atom count. Runs before FREQ_NUDGE so runs are taken first and only truly isolated SETs fall through to nudge.

### RELEASE_UPDATE (spec #5) — DONE, default OFF (`release_update_pass`)

- New `RELEASE_UPDATE_OP = 49`; single-row atom tagging an isolated SR or AD envelope write (SR + AD, per the doc's "covers SR and trivially AD"). Decode is SET-equivalent. Isolation uses the same ±2-frame gap rule as FREQ_NUDGE; reg-based so the legato/ADSR detectors (which key on reg, not op) are unaffected.

### CTRL_TRIPLE (spec #6) — DONE, default OFF (`ctrl_triple_pass`)

- New `CTRL_TRIPLE_OP = 50`; 3-byte atom for three consecutive CTRL writes each one frame apart (no DELAY between), mirroring `CtrlBigramPass`'s adjacency test. Runs before `CtrlBigramPass` in `run_passes` so triples win, leaving pairs to CTRL_BIGRAM. Decode writes byte 0 at the atom frame and queues bytes 1–2.
- **Issue / decision — triples only.** This implements the "CTRL_TRIPLE" half of the spec row. The residual single CTRL writes the doc also lists (SET/EOS, BOS/EOS) are handled by the validator's `gate_off_terminal` carveout and the trajectory-anchor extension below, not by this op.

### Trajectory-anchor extension (spec #7) — DONE, in the validator

- **Issue / decision — implemented as a carveout, not an encoding change.** The doc says the residual "hits the carveout list", so rather than re-encode SLOPE/FLIP/FLIP2/TRANSPOSE atoms to swallow an anchor (which would risk their round-trip and the existing audio-invariant tests), the `LonelyWriteValidatorPass` carveout classifier now tags any full SET immediately adjacent (leading or trailing, within the anchor window) to a SLOPE / OSCILLATE_ENV / FLIP / FLIP2 / TRANSPOSE primitive as `trajectory_anchor`. This absorbs the OP3/OP5/OP7/SLOPE adjacency signatures into the allow-list with zero round-trip risk. Literal encode-side anchor folding remains available as a future optimisation.

### Status summary (2026-05-24)

All seven primitives + the validator are implemented with synthetic round-trip unit tests; the full suite is green (713 passed, 3 skipped) at 86% coverage. OSCILLATE_ENV is default-ON; every residual-absorbing primitive (VOICE_TRACK, FREQ_NUDGE, FREQ_RUN, RELEASE_UPDATE, CTRL_TRIPLE) ships behind a default-OFF flag and the validator is default-OFF, so the existing pipeline is byte-for-byte unchanged until the validation phase enables them. **Postponed to the validation phase:** corpus coverage simulation, round-trip audio fidelity (needs preframr-audio v0.3.1 params), mini A/B, the 12-SID WAV audition, flipping the flags to default-ON, and relaxing the conservative exact-match thresholds (OSCILLATE_ENV uniform-runtime, VOICE_TRACK exact-interval) once measured against the real corpus.

## Validation-phase structural-primitive findings (2026-05-24, v0.11.0)

The coverage gate was run (main-repo `integration_tests/profile/audit_strict_no_diff_coverage.py`, real passes over 100 prodlike songs): all-primitives-on absorbs 93.7% of lonely-writes, but FREQ_NUDGE/FREQ_RUN/CTRL_TRIPLE/RELEASE_UPDATE carry all of it. The two **structural** primitives barely fired — OSCILLATE_ENV 256 rows, VOICE_TRACK 0 — so headroom probes were run (40 songs).

### OSCILLATE_ENV — starved, not just gated; uniform sub-run split landed
The pass detects on `SlopePass` output (≥3 consecutive same-reg SLOPE atoms), but SlopePass needs `SLOPE_MIN_RUN_LEN=5`-frame monotonic runs, and vibrato half-cycles are 2–4 frames, so most vibrato never becomes a SLOPE the pass can see. Only 122 ≥3-SLOPE chains exist per 40 songs, and the **uniform-runtime gate** rejected 91 of them (75%; e.g. PW grid-32 yields `[4,5,4,5]`).

**Landed (v0.11.0):** the gate is replaced by splitting each maximal SLOPE chain into maximal **uniform-runtime sub-runs** (≥3), each round-tripping exactly through the existing ramp decoder. OSCILLATE_ENV firing rose 88→224 rows / 40 songs (~2.5×). Note this does **not** move the lonely-write residual (SLOPE chains were never lonely) — its value is structural expression (oscillation as one semantic token), not residual reduction.

**Deferred (needs WAV audition):** the real headroom is raw-stream vibrato — a 40-song probe found 912 alternating-direction short-cycle FREQ runs / 7,142 rows (~18k rows/100 songs) that the SLOPE-chain path cannot reach. Capturing it needs (a) a detector on the forward-filled per-frame FREQ stream (the locked `vibrato_min_cycles=2` default implies this — never built), (b) a **step-mode** (jump-and-hold) decode path distinct from the ramp decoder, and (c) a shift from exact to **audio-equivalent** reconstruction — now possible via `preframr_audio.fidelity.compare_renders(max_frame_drift=2)` (preframr-audio 0.3.1). This changes OSCILLATE from lossless to within-tolerance, so it is gated on the non-negotiable 12-SID WAV audition before flip.

### VOICE_TRACK — REFUTED (disabled, kept default-OFF)
Zero qualifying spans across **three** detection models over 40 songs: per-frame-SET exact (the shipped pass), held-value multiplicative ±2%, and held-value additive-exact. Three compounding reasons: (1) FREQ here is a **cent-bin index (logarithmic)**, where a musical interval is a constant *additive* offset — the shipped `round(lead·ratio)+detune` *multiplicative* model is mismatched to the representation; (2) the consecutive-per-frame-SET requirement only ever catches a voice's own glissando (held harmony is squeezed to a single SET, invisible); (3) the spec's own `MIN_TRACK_DURATION=10` sweep notes most "tracking" is 4–5-frame chord-change overlaps — already absorbed by FREQ_RUN/FREQ_NUDGE. The 246k "tracker FREQ writes" the spec predicted are those short overlaps, not sustained cross-voice tracking. Pass left in place but marked REFUTED in its docstring and kept default-OFF.

### Next increment
Raw-stream OSCILLATE_ENV (detector + step-mode decoder + audio-equivalent round-trip), gated per-primitive by `compare_renders(max_frame_drift=2)` ≥95% over 100 songs, then the 12-SID WAV audition, before flipping on and re-cutting training data.
