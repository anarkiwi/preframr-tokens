# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.39.0]
### Fixed
- Render fidelity: PRESERVE within-voice register write ORDER. The SID ADSR bug
  (envelope prescaler equality-compare) makes intra-frame AD/SR/gate order audible, so
  `_norm_pr_order` now keeps each voice's input emit order instead of sorting by
  register; reg-sorting scrambled interleaved ADSR/CTRL frames (~17% of single-speed
  tunes). Proven in preframr-audio `test_register_canonicalization`.
- `patch_pass` emitted decoded AD/SR writes with `diff=irq` (a whole frame) instead of
  the nominal `_MIN_DIFF`, driving the per-frame time budget negative and dropping
  samples (-25% level, cadence break). Now emits `_MIN_DIFF`.
- skeleton: map skeleton-owned freq words to the renderer's cent-index domain and
  preserve audible noise-onset freq (the freq-domain render bug).
### Added
- `sid_frame_diff` (settled per-frame register diff) + `tests/test_sid_frame_diff.py`.
- `tests/test_register_order_fidelity.py`: strict register-LEVEL order+timing fidelity
  gate (decoded CTRL/AD/SR order+values vs the raw dump, plus nominal-`diff`), catching
  the order/timing classes `register_state` is blind to.

## [0.38.3]

### Added (continued)

- **W5.3 — SWEEP loop-period (`sweep_loop`, default OFF).** A looping freq-domain arp (constant
  −delta/frame reset every P, SoundMonitor) is shattered by the linear run-finder at each reset jump; with
  `sweep_loop` on, `SweepPass._loop_runs` mines the periodic sawtooth (`freq[k] = start + ((k−i) % P) ×
  delta`, ≥2 periods, period ≤ `SWEEP_MAX_SPAN`) and emits looping SWEEP atoms carrying a new
  `SWEEP_SUBREG_PERIOD` (the decoder replays `start + (k % period) × delta`). SWEEP_OP already has its ATOM
  contract, so no new op. Byte-exact through the full `RegLogParser.parse`.

### Fixed

- **Long SWEEP runs lost frames through `_cap_delay` (latent base-SWEEP bug).** A single SWEEP atom spanning
  N frames leaves a `DELAY(N−1)` after it, and `_cap_delay` coarsens delays > 16 to the nearest power of two
  — so a run longer than ~17 frames dropped playback frames while the atom's `LEN` stayed, overrunning the
  per-frame replay into the next note (a 22-frame run lost 4 frames; not byte-exact). `SweepPass._emit_run`
  now splits every run (linear and loop) into chunks of ≤ `SWEEP_MAX_SPAN` (17) frames, each re-anchored at
  its own first frame so the inter-chunk DELAY stays in `_cap_delay`'s exact (≤16) range; loop chunks split
  on whole periods so each chunk starts on the program's start phase. Only reachable with `sweep_pass` on
  (default OFF, no golden masters affected). New `tests/test_sweep_loop.py` guards both the loop drain and
  the long-run chunking, byte-exact.

- **W5.1 — exact-landing SLIDE2 primitive (`slide_landing`, default OFF).** The rate-only SLIDE only
  expresses unit steps (±1 per `rate` frames), so a constant per-frame delta ≠ 1 (e.g. a +2/frame ramp
  `[2,4,6,8]`) leaks to RESID even with W4's wide SLIDE on. A new `ORN_TYPE_SLIDE2` (op stays `ORN_OP`, so
  no new contract) carries `(target, duration)` and replays a linear ramp that reaches `target` after
  `duration` frames then holds (`slide2_frame_offsets`), landing exactly on any reachable target. With
  `slide_landing` on, `fit_descriptor` routes both narrow and wide constant-delta ramps the rate-only form
  misses to SLIDE2 (`_slide2_descriptor` derives the duration as the landing frame and verifies once;
  target/duration each bounded to a signed byte). Byte-exact via the `_orn_rows` reconstruct-verify and
  provenance-invariant (same shape → same SLIDE2 tokens at any base). Built without the corpus survey at
  the author's direction; the other §5 candidates (VIB delay+length, SWEEP loop-period, PERC) remain
  survey-gated. New parse-level guard `tests/test_slide_landing.py`.

### Changed

- **`register_state` decode (no output change).** The byte-exact oracle (run on every isolation check and
  the full-corpus hunt) previously expanded the stream with `expand_ops` into a literal-write DataFrame and
  then re-reduced it row-by-row into the `(n_frames, 25)` state. It now walks the **same** canonical
  FrameWalker/DECODERS decode but snapshots the live regs 0-24 (`state.last_val`) once per decoded frame —
  skipping the DataFrame materialisation/sort/ffill entirely (~18% faster end-to-end; the standalone
  reduction goes from a 1.4 ms Python loop to a direct accumulation). Output is identical: differential-
  checked against the old reduction over 20k synthetic frame streams and 30 diverse real parses, and the
  whole suite (which uses `register_state` as its oracle) stays green.

- **Hot-loop performance (no output change).** Four output-preserving speedups on the encode path:
  `SkeletonPass._ctrl_at` binary-searches the frame-ascending ctrl writes instead of scanning (the hot
  per-frame pitched-frame test); `_slide_rate` derives the only candidate rate from the leading-zero run
  and verifies once (O(n) vs O(n²), now that W4 routes longer ramps through it); `wavetable.factorise`
  uses a per-period last-mismatch pass (O(m²) vs the cubic position-by-position scan); and
  `WavetablePass._build_codebook` indexes programs by first-step offset so the fallback verify only
  visits plausible candidates. `factorise` and `_slide_rate` were differential-checked byte-identical to
  the brute-force versions over ~100k+ inputs (incl. exhaustive small cases); the suite (golden masters
  included) stays green.

### Added

- **W4 — route wide monotone ramps to SLIDE (`slide_wide`, default OFF).** A constant note-relative delta
  ramp (the SWEEP/monotone tail class) wider than `_OFFSET_LIMIT` currently leaks straight to RESID before
  the SLIDE check; with `slide_wide` on, `fit_descriptor` routes it to `ORN_TYPE_SLIDE` when the rate-only
  form reproduces it exactly and the target fits a signed byte (`_slide_descriptor`). This keeps a genuine
  glissando a real primitive rather than a W3 one-shot dumping ground, and makes it **provenance-invariant**
  — the same ramp shape encodes to the same SLIDE tokens at any base. Byte-exact: the existing `_orn_rows`
  reconstruct-verify reverts to RESID if `slide_frame_offsets` does not reproduce the floor, so a mismatch
  can never ship. Gate threaded only to the emit-site `fit_descriptor` call (resegmentation classification
  is unchanged). New parse-level guard `tests/test_slide_wide.py` (routes a wide ramp to SLIDE through the
  full `RegLogParser.parse`, asserts the isolation oracle and provenance-invariance). The PERIOD-survivor
  onset-strip alignment (a codebook-catch compression refinement, not a RESID=0 requirement — the W3
  one-shot already gives those notes a byte-exact home) is left for the survey-gated W5.

- **W3 — inline one-shot, the RESID=0 backstop (`wt_oneshot`, default OFF).** A dedicated self-contained
  `WAVETABLE_ONESHOT_OP` (op 69) that inlines a verbatim offset program at the note position with **no
  codebook id** (no DEF/REF indirection, no single-use id pollution): LEN_HI/LO then RLE OFFSET(/HOLD)
  atoms, terminated by END, replayed by `WavetableDecoder` as base + `LUT[note+off]` per frame —
  byte-identical to the ORN RESID queue it replaces. With `wt_oneshot` on, `WavetablePass` becomes **total
  over all residue**: it keeps every RESID note (dropping the pitched-core gate on the one-shot path —
  noise-interleaved and otherwise-unkeyable notes included) and any note that matched no codebook id emits
  a one-shot, so `ORN_TYPE_RESID` can no longer reach the deployed stream. The op is a single op-code (all
  rows share it), so it is same-frame-block safe without a `_BLOCK_SOP` entry. W6: contracted as a
  self-contained `MaskRole.ATOM` (absent from `CODEBOOK_SPECS`, unlike the REF path); the completeness
  test stays green. New parse-level guard `tests/test_wavetable_oneshot.py` drains FLAT and
  no-pitched-core residue to zero RESID through the full `RegLogParser.parse`, byte-exact.

- **W2 — short literal-tuple WAVETABLE codebook (`wt_short`, default OFF).** The dominant tail lever
  (RECUR+SHORT ≈55% of the post-wavetable residue). The existing codebook keys on `factorise(core)` after
  an onset-strip and a pitched-core gate — a key designed for long looping programs that is *stricter than
  exact-tuple recurrence*, so it misses short transients (length-1 like `[31]`, or noise-onset tuples like
  `[33,0]` whose non-pitched onset strips the core below `_MIN_CORE`). With `wt_short` on, residue of
  length ≤ `WT_SHORT_MAX` (4) is routed to a literal path in `WavetablePass`: keyed on the verbatim offset
  tuple (no factorise, no onset-strip, no pitched-core), stored as a loopless RLE program (`unroll` is the
  identity), reusing the existing `WAVETABLE_DEF/STEP/END/REF` ops and decoder. A tuple recurring ≥
  `WT_MINREP` drains to one DEF + N REFs; unique short tuples stay RESID (no single-use ids — W3 absorbs
  them). Byte-exact (the REF replays the same per-frame freqs the RESID did). New parse-level guard
  `tests/test_wavetable_short_codebook.py` (drains length-1 and noise-onset transients to a bounded
  codebook through the full `RegLogParser.parse`, asserts the isolation oracle and bounded vocab).

- **W1 — ZERO→PLAIN (`zero_plain`, default OFF).** A skeleton ORN note held at its base whose freq only
  moves on unresolvable (silent/noise/test) frames the content floor snaps to 0 is an all-offset-0 RESID
  escape (the ZERO survivor class, ~8% of the post-wavetable tail). With `zero_plain` on,
  `SkeletonPass._orn_rows` rewrites such an all-zero-target RESID to `ORN_TYPE_PLAIN` — byte-exact, because
  PLAIN replays the same held base the all-zero RESID did (`register_state` identical). Gated default-OFF;
  only fires when every offset in the snapped target is 0 (any non-zero offset stays RESID for W2/W3). New
  parse-level guard `tests/test_zero_plain_parse_wiring.py` (drains through the full `RegLogParser.parse`,
  asserts the isolation oracle).

## [0.38.2]

### Fixed

- **Codebook DEF blocks shattered when two share a frame (WAVETABLE/STAMP/PATCH byte-exactness).** The
  within-frame row sort `_norm_pr_order` (`["f","v","reg","op","n"]`) ordered by op-code, so when two
  variable-length codebook blocks (`DEF→STEP*→END`) landed in the same frame it grouped all DEFs, then all
  STEPs, then all ENDs — splitting both blocks so the decoder mis-parsed them and replayed garbage freqs.
  A full-corpus byte-exact survey (register_state OFF vs `wavetable_pass` ON) measured **~2.5% of tunes
  diverging**, concentrated in the dense-codebook engines (GoatTracker, DMC, JCH). Fix: each family's
  STEP/END now collapse to its DEF op for the sort (`_BLOCK_SOP`), so a block stays contiguous in emit
  order (`n`); a no-op when no codebook ops are present, so non-codebook streams and the golden masters are
  unchanged. New `tests/test_wavetable_multidef_frame.py` builds a 2-voice dump that puts two
  WAVETABLE_DEFs in one frame and asserts byte-exact through the full `RegLogParser.parse`. Latent for
  STAMP/PATCH (rarer per-frame); the wavetable codebook exposed it. Still default-OFF.

## [0.38.1]

### Fixed

- **WAVETABLE pass was dead in the real parse path (RESID_ZERO_PHASE3 §2 follow-up).** `WavetablePass`
  (0.38.0, #41) was registered in `macros.FREQ_BLOCK_PASSES` but never inserted into the parallel
  hand-listed freq-pass sequence inside `RegLogParser.parse`, so `wavetable_pass=True` was a no-op for
  every real parse — the codebook never materialised and recurring ORN-RESID never drained. The 8
  direct-apply unit tests stayed green because they call `WavetablePass().apply()` straight, never
  through `parse()`. Wired the pass in after `SkeletonPass` (mirroring `FREQ_BLOCK_PASSES` order) and
  added `tests/test_wavetable_parse_wiring.py`, a parse-level guard that drives a recurring RESID
  program through the full `RegLogParser.parse` and asserts the WAVETABLE_DEF/REF codebook appears
  byte-exactly (`register_state` OFF==ON) while default/OFF stays a no-op. Default golden stream
  unchanged (pass gated OFF). Measured post-fix byte-exact drain (3 tunes/engine): GoatTracker 83%,
  JCH 76%, DMC 72%, SidWizard 67%.

## [0.38.0]

### Added

- **WAVETABLE codebook primitive (RESID_ZERO_PHASE3 §2, #41).** The pitched twin of STAMP:
  `WavetablePass` (after `SkeletonPass`, gate `wavetable_pass`, default OFF) mines the skeleton
  ORN-RESID note-relative offset dumps into an inline-redefinable `WAVETABLE_DEF`/`WAVETABLE_REF`
  codebook (ops 65–68) — held-ARP generalised to a cross-note loop — or an inline-structured one-shot.
  Onset-strip, noise-inclusive detection, RLE+loop factorisation, recurrence codebook (`WT_MINREP=2`)
  + verify-match; byte-identical to the content-floor RESID it replaces (isolation oracle) or falls
  back to RESID. New `macros/wavetable.py` (`factorise`/`unroll`/`program_key`), `WavetableDecoder`,
  `DecodeState.wavetable_table`. Shared `macros/rle.py` run-length codec dedupes `skeleton_pass._rle`.
- **Constrained-decode OpContract registry + codebook inference safety (RESID_ZERO_PHASE3 §4 B0–B4,
  #42).** One `macros/op_contracts.py` `OP_CONTRACTS` registry (one contract per emittable op) with a
  completeness test that fails at unit-test time if any op lacks a contract. The atomic sampling mask
  (`precompute_vocab_arrays` structural arrays + the `StreamState` slot tables) is now generated from
  `STRUCTURAL_SUBREGS`, byte-identical (golden-master regression lock). Inline-codebook DEF→REF
  backrefs (STAMP/PATCH/WAVETABLE) are made inference-safe: the mask forbids a REF to a non-live id, a
  DEF/COMMIT make ids live, `expand_ops(codebook_seed=…)` / `StreamState(init_codebook_ids=…)` /
  `validate_stream(live_ids=…)` / `codebook_live_ids()` materialize out-of-window refs, and
  `validate_codebook_refs` replays legality. Cross-repo API extended additively only.
- **Provenance-invariance tests (#11.4, principle P7).** Test-only: `TestProvenanceInvariance`
  asserts the universal-driver property — the same musical gesture encodes to the SAME SKEL+ORN
  tokens regardless of register-level provenance: ORN is transposition- and duration-invariant, and
  the content-tier semitone floor is invariant to driver tuning (constant sub-semitone detune). New
  `inline_note_signature` probe helper. No library/vocab change.

## [0.35.0]

### Changed

- **Skeleton segmentation: fold fast-melodic-runs into notes (#13).** A held-gate fast melodic
  line (each pitch held < `MIN_HOLD`, no re-gate, not a periodic arp) previously collapsed into
  one note whose non-periodic offsets leaked to RESID — the dominant shared real-tune RESID
  source. `SkeletonPass._resegment_fast_run` now splits such a note (gated on `fit_descriptor`
  returning RESID, so genuine ARP/SLIDE/VIB/OCTAVE ornaments are untouched) into one SKEL note
  per semitone step. Measured RESID note-share: Trap 0.44→0.01, Camerock 0.17→0.06, Baggis
  0.66→0.26, Commando 0.25→0.24; the fast-melodic-run frame-fraction drops to ~0 (Trap) / 0.009
  (Baggis). Baggis's remainder is a *distinct* wide/aperiodic primitive (span 51–71 semitones),
  not the fast-run mechanism. Skeleton-mode token stream changes (more SKEL notes) — re-cut
  skeleton corpora. New `is_fast_melodic_run` helper; `test_driver_coverage` updates the Trap gap
  to a passing test, re-reasons the Baggis xfail, and adds a fast-run-closed regression guard.

## [0.34.0]

### Removed

- **BREAKING: op-code/vocab change, re-cut corpora/checkpoints, no metric transfer.**
  Removed refuted/dead-wood transforms and their full 3-layer (pass + decoder + transform)
  wiring. All removed work is refuted, so nothing to lose; any existing corpus/checkpoint
  cut before this release must be re-cut (the op alphabet shifts). Removed:
  - `flip2` — `Flip2Pass` (was in the default `PASSES`), `Flip2Transform`, `Flip2Decoder`.
    Frees op `7` (now `RESERVED_OP_7`).
  - `motif` — `MotifPass` / `MotifTransform` / `MotifDict` / `mine_motifs` /
    `get_motif_dict` (`macros/motif_pass.py`) and `mine_dict_from_dumps`
    (`motif_mine.py`), plus the `blocks.py` apply-site and the `vocab_signature`
    `MOTIF_ARG` tier branch. Frees ops `52`/`53` (now `RESERVED_OP_52`/`RESERVED_OP_53`).
  - `voice_trajectory` + `voice_trajectory_distributed` — both transforms and the
    `voice_trajectory_window` param. Frees reg `-123` (now `RESERVED_REG_NEG123`).
  - `super_frame` — `SuperFrameTransform` (the N>=2 pack was never implemented). Frees reg
    `-124` (now `RESERVED_REG_NEG124`).
  - `set_to_diff` — `SetToDiffTransform` and the `_OPTIONAL_TRANSFORMS` opt-in registry in
    `reglogparser.py`. Reused `DIFF_OP=1`, so frees no op.
  - Survivors are NOT renumbered; each freed op/reg number is held by a `RESERVED_*`
    sentinel constant in `stfconstants.py`. `ctrl_update` (`CTRL_UPDATE_OP=51`, live via
    `--lonely-catch-all`) and `TRACK_REF_OP=46` (live `VoiceTrackPass`) were verified
    in-use and **kept**.

## [0.33.0]

### Changed

- **Driver-native parametric ornament channel** (`ORN_OP=55`). The 0.32.0 ORN stored
  ornament params **inline per-frame** (one signed offset per frame for OCTAVE/ARP/SLIDE,
  a raw 16-bit freq per frame for VIB/RESID), which flooded op55 to ~13–55:1 over op54
  SKEL and starved the skeleton. `fit_descriptor` / `_orn_rows` (`macros/skeleton_pass.py`)
  and `OrnamentDecoder` (`macros/decoders.py`) now emit a **constant-size-per-note**
  descriptor independent of note duration, in the SID driver's own domain (no mined
  codebook / no top-N LUT / no per-composer banks / no mining step):
  - **PLAIN** — `TYPE` only.
  - **OCTAVE / ARP** — `TYPE` + the **canonical ordered offset cycle** (the minimal period
    that genuinely repeats, ≤`ARP_MAX_PERIOD`=8 signed note-relative semitone atoms) + a
    **length** atom; decode replays `LUT[note + period[k % period_len]]` per frame. A
    non-repeating one-shot run is residual, not a spurious whole-sequence "period".
  - **SLIDE** — `TYPE` + **target-interval** + **rate** + length; decode ramps one
    semitone per `rate` frames toward the target, clamped.
  - **VIB** — `TYPE` + **depth-bucket** + **rate** + length; a sub-semitone oscillator that
    reconstructs to the held semitone at the content-tier floor (the wobble is below the
    floor), carrying a learnable depth/rate signal, not one raw freq per frame.
  - **RESID** — minimized raw escape (one signed note-relative offset per frame, at the
    semitone floor) reserved for genuinely-unfittable notes; a parametric fit that does not
    reconstruct the floor exactly falls back to RESID.
- **Held-gate re-segmentation** (`SkeletonPass._resegment_holdgate` /
  `_split_holdgate_resid`, `macros/skeleton_pass.py`). On held-gate / legato drivers a
  sustained gate spans several melodic notes plus their connecting slides, so a phrase that
  opened with a stable `≥MIN_HOLD` plateau but then moved on (with no gate retrigger —
  Hubbard note-flag bit6 "appended, no attack") was read as ONE note whose internal melody
  could not be a parametric ornament and dumped to a long `RESID`. A RESID note that opens
  with such a plateau is now split at its first post-plateau moving frame into the held note
  + the trailing melody as its own note, recursively, de-merging the phrase into its
  constituent notes (the connecting motion becomes each note's ornament). Fast (`<MIN_HOLD`)
  arp/vibrato steps that never settle are NOT turned into notes (the fast-step ornament guard
  is kept), so clean tunes are unaffected. Decode stays floor-exact. This sharply cuts the
  op55:op54 ratio and RESID-note count on held-gate tunes (e.g. Camerock 10.1→4.7:1, RESID
  344→173) while clean tunes hold (Commando ~3:1).

### Added

- **Deterministic pre-training encoding test suite** — catches unmodelled-encoding /
  implementation errors deterministically before any training run. All synthetic checks
  are copyright-free and always run in the standard docker gate; real-tune layers
  regenerate-or-fail from a local cache (never `skipTest`, never commit a `.sid`/dump).
  - `tests/parse_probes.py` — torch-free parse-side helpers: a `DumpBuilder` that emits a
    raw `clock,irq,chipno,reg,val` dump with **separate lo+hi byte writes + per-frame freq
    writes**, so tests run the FULL `RegLogParser.parse` (`_combine_regs` +
    `_quantize_freq_to_cents`) instead of a hand-built `Pass.apply` df (the path that hid
    the cent-index no-op). Op counting over the deployed block stream, single-encode inline
    ORN classification, and a RESID-note classifier (fast-melodic-run / glissando / noise).
  - `tests/test_parse_pipeline_smoke.py` (#10) — per-config op matrix (skeleton →
    `op54>0,op55>0,op45==0,op48==0`; default → `op45>0`; freq_onset → `op48>0`), a skeleton
    round-trip onto the LUT semitone floor, and an `op55:op54 <= 6` channel-balance bound
    (catches channel-drowning). These FAIL if reverted onto the 0.31.0 cent-index no-op.
  - `tests/test_driver_coverage.py` (#11) — one synthetic generator per driver mechanism
    (octave arp → `ORN_TYPE_OCTAVE`, table arp → `ORN_TYPE_ARP` period, vibrato →
    `ORN_TYPE_VIB` depth, slide → `ORN_TYPE_SLIDE` target, held → `ORN_TYPE_PLAIN`), each
    asserting the correct primitive AND zero RESID; RESID is the completeness metric. The
    known real-tune RESID gap (Trap/Daglish, Baggis/Goto80) is `xfail(strict=True)` so the
    suite stays green and flips to XPASS when the missing segmentation mechanism lands; the
    gap is characterized (dominated by fast-melodic-run under-segmentation, not legit
    glissando). Real-driver fixtures resolve via `tests/sid_fixtures.py:ensure_driver_fixture`.

## [0.32.0]

### Added

- **Stage 1+2 unified pitch: dense skeleton + ornament channel** (`ORN_OP=55`).
  `SkeletonPass` was rewritten to segment each freq reg's settled per-frame values
  into NOTES via the validated `audit.unified_pitch` algorithm — semitone-run with
  `MIN_HOLD=3` (so fast arp steps are not notes) UNIONed with gate-on rising edges
  (ctrl bit0) — instead of `traj_anchor` chunks. Each note emits one `SKEL` atom
  (note→freq LUT index; absolute first per reg, signed semitone interval after)
  plus one `ORN` descriptor that collapses the note's intra-note arps/vibrato/slide
  into a single classified primitive (`PLAIN`/`OCTAVE`/`ARP`/`SLIDE`/`VIB`/`RESID`)
  with params stored **inline** (per-frame note-relative offset cycle for
  OCTAVE/ARP/SLIDE; raw per-frame 16-bit freq escape for VIB/RESID). So a note's
  dozens of sub-frame freq writes collapse to one SKEL + one ORN — arps decode AS
  arps (coherent), not raw op0 "clicks". (Future compression step: a mined
  per-composer ARP/offset codebook replacing the inline per-frame params.)
- `OrnamentDecoder` (`op_code=ORN_OP`) replays the ornament onto the skeleton freq
  via `pending_set_writes` (one drain per frame tick, the SKEL owns the onset
  frame): OCTAVE/ARP/SLIDE → `LUT[skel_note+off]` per frame; VIB/RESID → the raw
  16-bit escape. Registered in `DECODERS`. `SkeletonTransform` now owns both
  `{SKEL_OP, ORN_OP}` and dispatches ORN to `OrnamentDecoder`. `ORN_OP` added to
  `is_melody_pitch_atom` (pitch-ornament content). `SkeletonDecoder` clamps the
  decoded note to the LUT range.

### Notes

- Content-tier (semitone-snap is the deliberate pitch quantisation); the RESID/VIB
  raw escapes keep the per-frame freq byte-exact at that floor. On real dumps the
  skeleton owns essentially all freq content (raw op0 freq SET remaining ≈ 0).

## [0.31.1]

### Fixed

- `SkeletonPass` produced **no** `SKEL` atoms on real parsed dumps (0.31.0 was a
  no-op on real data): it read the `val` column, but `_quantize_freq_to_cents`
  had remapped freq `val` to a small cent-index (the 16-bit freq survives in
  `freq_unq`), so every value looked silent and nothing was claimed. Fixed by
  **skipping cent-quantization when `skeleton_pass` is on** (the skeleton is the
  freq quantisation — to semitones — so freq stays raw 16-bit and is not
  double-quantised). Also relaxed the PLAIN claim from exact single-value to
  "all values within `CENTS_THRESHOLD` of one semitone" (real held notes jitter a
  few cents). Regression test `test_skeleton_claims_jittery_within_semitone_note`.

## [0.31.0]

### Added

- `SkeletonPass` (op `SKEL_OP=54`) + `--skeleton-pass` flag (opt-in; reserves
  `ORN_OP=55` for the ornament channel): Stage 1 of the unified-pitch
  skeleton+ornament encoding. Collapses each clean held freq note (single settled
  value within `CENTS_THRESHOLD` of a semitone) into ONE atomic `SKEL` atom — a
  note→freq LUT index, absolute for the first claimed note per reg and a small
  signed semitone interval after — so a note is one token (the per-note-token
  generation-coherence axis). Notes with intra-note motion or off-semitone held
  values stay raw `op0` SET (byte-exact pass-through / RESID). When on,
  `freq_trajectory_pass` and `freq_onset_pass` must be OFF (skeleton owns
  `FREQ_TRAJ_REGS`). `SkeletonDecoder` decodes `SKEL` to `LUT[note]`
  (`audio_bit_exact`, `LOSS_TIER="content"`); `SKEL_OP` is a melodic-pitch atom.
  `combine_reg`-backed; reuses `TrajectoryAnchorPass` segmentation.
  `test_skeleton_pass` covers the round-trip; additive (default pipeline byte-exact).

## [0.30.0]

### Added

- Public `combine_reg` (and `combine_val`) in `preframr_tokens.reglogparser`,
  exported from the package root. These are the canonical lo+hi byte-coalescing
  the parser already used internally (`RegLogParser._combine_reg`/`_combine_val`
  now delegate to them, unchanged behaviour): forward-fill both bytes and keep
  the settled value per `clock // diffmax` bucket, so a coordinated lo+hi update
  is read as one 16-bit value and a half-updated pair is never seen. Exposed so
  the frequency audits reuse the parser's combine instead of re-deriving settled
  freq from raw lo/hi bytes (`test_combine_reg`).

## [0.20.0]

### Added

- Corpus-mined `motif` pass (`preframr_tokens.macros.motif_pass`): a
  boundary-constrained, cross-composer motif dictionary that losslessly
  collapses recurring atom sequences into single `MOTIF_OP` (52) atoms between
  the structural macros and Unigram. `mine_motifs` (frequency-greedy with a
  frame-boundary guard and a cross-composer floor) + `MotifDict`
  (`encode`/`expand`, JSON artifact) + `MotifTransform` (registered `motif`,
  `DECODES_VIA_DF` lossless `forward`/`inverse` over `(op,reg,subreg,val,diff)`)
  + `MotifPass`. Verified byte-exact forward→inverse on parsed SID fixtures
  (`test_motif_fidelity`). OFF by default (`motif_pass` flag, needs a mined
  `motif_dict`); opt-in only — not in `DEFAULT_PIPELINE_SPEC`. Generation-time
  constrained-decode metadata for motif atoms is not yet implemented (needs a
  model trained on the motif vocab to validate).

## [0.18.0]

### Fixed (BREAKING — re-cut corpora and checkpoints; no metric transfer)

- Frame timing now preserves playback duration through the parse pipeline. The
  base tokenizer was dropping wall-clock song duration — up to ~67% on some
  songs — independent of any macro; `FreqTrajectoryPass` only exposed it by
  emptying frames. Three fixes:
  - `read_initial_irq` and `macros.state._build_decode_state` take the first
    FRAME row with a **positive** `diff` as the frame period. A degenerate
    leading frame can carry `diff` 0 (song starts at t=0); using it as the
    period zeroed every DELAY-expanded frame's playback time.
  - `RegLogParser._consolidate_frames` rewritten to be cycle-preserving. The old
    consecutive-DELAY merge lost values on runs of ≥3 adjacent DELAY markers and
    counted frames rather than time; it now collapses each marker-only run into
    one DELAY + trailing frame while conserving total cycles (FRAME worth
    `round(diff / period)`, DELAY worth its val, the run's final marker kept
    verbatim so its voice-order survives).
  - `RegLogParser._add_frame_reg` rounds a marker's frame count
    (`round(irqdiff / irq)`) instead of truncating. After `_squeeze_changes`
    merges rows the surviving gaps are rarely integer multiples of the period,
    so truncation dropped a fractional frame on every marker (worst on dumps
    whose play rate is not a simple multiple of the period).
- Net effect on a 12-SID cohort: the macro round-trip now preserves duration vs
  the no-macro baseline to within 0–0.5% (was −37% to −44% on
  ramp/oscillation-heavy songs), and base tokenization tracks the true dump
  clock-span to within ~3% (was −30% to −67%). Output bytes change for affected
  songs, so corpora and checkpoints must be re-cut.

## [0.17.0]

### Changed

- `tokenizer_config.MACRO_FLAGS` is now *derived* from the passes rather than
  hand-listed. Each gated `MacroPass` declares the args it reads via a
  `GATE_FLAGS` frozenset (each `Transform` already declared its via
  `REQUIRES_ARGS`); the new `macros.flag_registry.macro_flag_names` glob-imports
  the `macros` package and unions the declarations. A renamed/added/removed pass
  flag can no longer drift out of sync with `MACRO_FLAGS`, and a new
  `test_flag_registry` drift guard fails CI if a pass reads a boolean arg that no
  declaration covers.
- Dropped two phantom flags that no pass read (`mode_vol_flip_pass`,
  `legato_pass_c3`) and surfaced four that passes read but the hand-list omitted
  (`gate_slope_shift_pass`, `voice_track_pass`, `strict_lonely`,
  `super_frame_pass`). The no-macro baseline (`default_tokenizer_args()`) now
  zeroes `gate_slope_shift_pass` too (previously it leaked its default-on state);
  pass output is unaffected because the pass is applied consistently across the
  baseline and every macro config the fidelity oracle compares.

## [0.16.0]

### Changed (BREAKING — re-cut corpora and checkpoints; no metric transfer)

- Unified FREQ/PW/FC trajectory primitive `FREQ_TRAJ` (op 45) replaces the four
  split passes (`SlopePass`, `OscillationEnvelopePass`, `RawVibratoEnvelopePass`,
  `FreqRunPass`) with one `FreqTrajectoryPass` over every slope-able register
  (FREQ 0/7/14, PW 2/9/16, FC 21). A `SUBTYPE` in the `FLAGS` byte selects the
  payload: `MONOTONE_RAMP` keeps SLOPE's terminal+runtime fit unchanged;
  `OSCILLATE` is recognised by a gap-tolerant sign-alternation gate
  (`OSC_MAX_GAP=2`, `OSC_MIN_ALTERNATION=0.5`, `OSC_MIN_HALFCYCLES=3`); `RUN`
  catches the rest. `OSCILLATE`/`RUN` share one lossless `v0` + cumulative-delta
  payload — signed-byte deltas with a `0x80` escape to a 16-bit absolute value
  (PW/FC exceed a signed-16 delta), plus optional periodic collapse.
- `FREQ_NUDGE` (op 47) delta mode is now a 2-atom `mode + signed-delta-byte`
  (escape to 16-bit), down from `mode + hi + lo`.
- Retired ops 32–34 (`SLOPE_*`), 37–38 (`SLOPE_*_SHIFTED`), 48 (`FREQ_RUN`),
  52 (`FREQ_VIBRATO`) and their `SLOPE_*` / `OSC_*` / `VIB_*` constants and
  decoders. `GateSlopeShiftPass` now shifts presets only; `PerRegBurstPass`
  skips its FREQ/PW/FC burst when `freq_trajectory_pass` is active (the unified
  pass owns those registers). The registered `slope` transform is renamed
  `freq_traj`.
- Op-code reuse and the payload change invalidate any pinned vocab/alphabet:
  corpora and checkpoints must be re-cut (no metric transfer).

### Changed

- `audio` extra floor bumped to `preframr-audio>=0.5.0`.

### Added

- Torch-free tokenizer profiling tools: `tokenizer_config`
  (`default_tokenizer_args` / `named_config` — one source of truth for the
  parser/macro args namespace, now consumed by the fidelity test),
  `register_state` / `op_atom_profile` / `trajectory_coverage` in
  `audit_primitives`, and the `python -m preframr_tokens.tokenizer_profile`
  (with `--compare`) and `python -m preframr_tokens.trajectory_coverage` CLIs.

## [0.15.0]

The public API now lives behind a curated `preframr_tokens` package façade:
import everything from the package root and rely on `__all__` as the
semver-promised surface, so internal module layout can change without breaking
consumers. `stfconstants` and `engine_fingerprint` remain public submodule
namespaces; all other submodule paths are now internal.

No copyrighted SID-derived song data is committed anymore. The two
`grid_runner_*.dump.parquet` fidelity fixtures were register dumps of HVSC
`MUSICIANS/J/Jammer/Grid_Runner.sid` and are now regenerated on demand from
HVSC and cached locally outside the source tree. History was rewritten so the
dumps never appear in any committed tree.

### Removed

- `preframr_tokens.reglog_helpers` — the back-compat re-export grab-bag is
  dissolved. Voice-relative reg matchers plus a new scalar `reg_class`
  classifier moved to `preframr_tokens.reg_match`; `read_initial_irq` to
  `preframr_tokens.reglogparser`; `tighten_persist_dtypes` to
  `preframr_tokens.utils`. (The `wrapbits` / palette-sidecar re-exports were
  already removed.)
- `LOSS_TIER_NAMES` is no longer importable from
  `preframr_tokens.macros.transform`; it now lives in
  `preframr_tokens.stfconstants`.
- `preframr_tokens.macros.lonely_validator._REG_CLASS` (private) — replaced by
  the public `preframr_tokens.reg_match.reg_class`.
- `tests/fixtures/grid_runner_head.dump.parquet` and
  `tests/fixtures/grid_runner_26s.dump.parquet` (and the now-empty
  `tests/fixtures/` directory). Purged from the whole branch history.
- `TOKEN_IMPROVEMENTS.md` (strategic-backlog narrative) — purged from the
  whole branch history; its "item N" citations were dropped from source
  docstrings and CHANGELOG entries.
- `API_SURFACE.md` — its durable content (design principle, decision-helper
  inventory, public surface, intentional-shape rationale, versioning policy)
  is folded into the README "API surface" / "Stability" sections; the
  standalone doc is deleted.

### Added

- `preframr_tokens/__init__.py` re-exports the full public surface as
  `__all__` (57 names): `from preframr_tokens import RegLogParser, reg_class, …`.
- `reg_class(reg) -> (kind, voice)` scalar register classifier in
  `preframr_tokens.reg_match`, the parse-domain sibling of `macros.roles`.
- `tests/sid_fixtures.py`: a `SidDumpSpec`-driven helper that downloads the
  `.sid` from HVSC, renders a register dump with `vsid` inside the
  `anarkiwi/headlessvice` image (a regular-file dump target, replicating
  vsiddump.py's post-processing byte-for-byte — no FIFO deadlock), slices the
  `head`/`26s` fixtures, and caches them under `$PREFRAMR_SID_FIXTURE_CACHE`
  (default `$XDG_CACHE_HOME/preframr-tokens/sid-fixtures`). `test_full_pipeline_fidelity.py`
  sources its fixtures through it and skips (`FixtureUnavailable`) when Docker,
  the image, or the network are absent — the same contract as the prior
  "fixture missing" skip.

### Changed

- **Breaking:** import from the `preframr_tokens` package root rather than
  submodule paths (e.g. `from preframr_tokens import RegLogParser`). Only
  `stfconstants` and `engine_fingerprint` stay importable as submodules; every
  other `preframr_tokens.*` path is internal and may move between releases.
- Centralised duplicated macro-pass logic into `macros/passes_base.py`:
  `_first_irq(df)` replaces the identical "first IRQ value else -1" ternary
  open-coded in 11 sites (10 passes + `_splice_rows`), and `_frame_isolated(
  frames, pos, gap)` replaces the byte-identical lonely-SET isolation predicate
  in `FreqNudgePass` and `ReleaseUpdatePass` (now parameterised by the gap
  constant). Behaviour-preserving; internal API only.

## [0.14.1]

Decode-only fix: the multi-frame collapse decoders drained one frame too early.
`CtrlBigramDecoder`, `CtrlTripleDecoder`, `FreqRunDecoder` and `FreqVibratoDecoder`
each emitted byte 0 immediately into frame N, then frame N's own `tick_frame()`
drained byte 1 from `pending_set_writes` into the SAME frame — clobbering byte 0
and shifting the whole run one frame early. For CTRL hard-restart runs this
corrupted gate/waveform timing (audible pitch-ups, broken percussion). The
encoder was already correct; only the decode placement was wrong, so tokenized
output (the train alphabet) is unchanged and existing corpora/checkpoints stay
valid — only audio reconstruction is fixed.

### Fixed

- All four decoders now queue ALL bytes into `pending_set_writes` and emit none
  immediately: frame N's tick drains byte 0, N+1 drains byte 1, N+2 drains byte 2
  — matching the raw (no-macro) stream exactly. Verified byte-exact against the
  raw decode across the full 314s Grid_Runner song, and audibly identical in
  render (`ctrl_triple`/`freq_run` isolated vs raw: PASS, worst rel-RMS 0.0002).

### Added

- `tests/test_full_pipeline_fidelity.py`: a per-frame register-STATE gate that
  parses fixtures through the WHOLE pipeline under each macro and asserts decoded
  per-frame state equals the no-macro baseline — catching frame-PLACEMENT bugs
  the single-pass synthetic round-trip tests (value-sequence only) miss. Two
  fixtures: `grid_runner_head` (~3s, ctrl_bigram/ctrl_triple) and
  `grid_runner_26s` (~26s, freq_run); each test asserts its decoder fired.

## [0.14.0]

Vibrato rework — lossless `FREQ_VIBRATO` replacing the lossy parametric
OSCILLATE-for-vibrato. A per-voice/per-frame fidelity probe found `vibrato_env_pass`
was the single biggest register divergence vs a no-macro render (~7-9k FREQ
frames on a prodlike song); root cause was NOT the fit tolerance (accepted fits
were exact) but (a) an absolute-anchor parametric atom capped at 31 cycles, and
(b) the pass running early, so later passes + frame consolidation misaligned its
multi-frame drain.

### Added

- `FreqVibratoPass` output op `FREQ_VIBRATO_OP` (52) + `FreqVibratoDecoder`: a
  consecutive-frame FREQ run whose values repeat with a small period encodes as
  `(period, 16-bit count, v0, signed delta-cycle)` and is replayed EXACTLY on
  decode (v0 + cyclic deltas). Lossless, uncapped, no envelope fit; non-periodic
  runs fall through to FREQ_RUN. (`RawVibratoEnvelopePass` rewritten.)
- `tests/macros/test_voice_agnostic.py`: enforces FREQ macros carry no
  out-of-band voice info — the same content on a different voice must yield
  byte-identical atom payloads (op, subreg, val), differing only by the register
  stride.

### Changed

- `RawVibratoEnvelopePass` now runs immediately before `FreqRunPass` (late,
  after the frame-altering passes) rather than early; this alone removed the
  multi-frame-drain phase swap + frame-count drift.

### Removed

- The OSCILLATE step-mode path (`OSC_STEP_MODE_BIT`, `OSC_FAMILY_MASK`, added in
  0.12.0) — superseded by `FREQ_VIBRATO`. `OscillationEnvelopeDecoder` is back to
  ramp-only.

### Result

- On the prodlike probe, removing `vibrato_env_pass` now clears **0** register
  divergence (was ~9550): vibrato matches the lossless FREQ_RUN baseline exactly.

## [0.13.0]

Fail-on-lonely: drive the strict-no-diff residual to zero. A v0.12.0 residual probe (100
prodlike songs) found 5,384 surviving lonely writes — FREQ (3,923) and SR (4)
that FREQ_NUDGE/RELEASE_UPDATE's isolation heuristic skipped, plus CTRL (1,457)
short runs CTRL_BIGRAM/TRIPLE missed — none of which are preset-anchor related.

### Added

- `lonely_catch_all` arg flag (default OFF). When on, the trailing absorbers
  become true catch-alls (each conversion is an exact single-write encoding):
  FREQ_NUDGE absorbs *every* residual FREQ SET (not just isolated ones),
  RELEASE_UPDATE absorbs every residual SR/AD SET, and the new `CtrlUpdatePass`
  (`CTRL_UPDATE_OP` = 51, SET-equivalent decode) absorbs every residual CTRL SET
  the control macros left.

### Changed

- `RegLogParser.parse`: `LonelyWriteValidatorPass` now runs **last** (after
  `add_voice_reg`, the optional transforms, FreqNudge and CtrlUpdate), as the
  spec intended — previously it ran before FreqNudge, so the catch-alls could
  not clear the residual it checked.

### Result

- With `lonely_catch_all` on, the strict-no-diff residual is **0** on 100
  prodlike songs (was 5,384), and `strict_lonely` parses **150/150** songs with
  zero `UnmodelledLonelyWriteError`. Fail-on-lonely is achievable. Default
  output is unchanged (`lonely_catch_all` and `strict_lonely` default OFF).

## [0.12.0]

Raw-stream OSCILLATE_ENV rework.

### Added

- `RawVibratoEnvelopePass` (`macros/raw_vibrato_pass.py`), behind the
  `vibrato_env_pass` arg flag (**default OFF**): collapses alternating short
  FREQ SET runs — vibrato whose 2–4-frame half-cycles SlopePass's
  `SLOPE_MIN_RUN_LEN=5` gate never turns into SLOPE atoms — into step-mode
  `OSCILLATE_ENV` atoms. Each maximal uniform-frame-gap run that alternates
  about its midline and fits an envelope family becomes one atom. On a 40-song
  prodlike probe this raised OSCILLATE_ENV firing 103× (240→24,848 atom rows),
  reaching the raw-vibrato headroom the SLOPE-chain path cannot.
- `OSC_STEP_MODE_BIT` / `OSC_FAMILY_MASK` (`stfconstants`): the FAMILY subreg
  high bit selects step-mode reconstruction.

### Changed

- `OscillationEnvelopeDecoder`: step-mode atoms reconstruct by holding each
  terminal for `period` frames (audio-exact; re-writing a held FREQ value is
  inaudible) rather than ramping. gap=1 collapses to the exact per-frame case
  and round-trips byte-for-byte; gap>1 is audio-equivalent (identical per-frame
  FREQ trajectory). Ramp-mode (existing SLOPE-sourced, default-ON) atoms are
  unchanged — the step bit is unset on them and FAMILY masking is a no-op.

### Notes

- `vibrato_env_pass` ships default-OFF: its amplitude fit shares
  `OscillationEnvelopePass`'s `FIT_TOLERANCE`, so it is no lossier than the
  already-default-ON pass, but flipping it on changes tokenizer output and is
  gated on the 12-SID WAV audition + a re-cut of training data.

## [0.11.0]

Validation-phase structural-primitive fixes:

### Changed

- `OscillationEnvelopePass`: the all-or-nothing uniform-runtime gate is
  replaced by splitting each maximal SLOPE chain into maximal uniform-runtime
  sub-runs (≥3 slopes), each collapsed and round-tripping exactly through the
  ramp decoder. OSCILLATE_ENV firing rose ~2.5× on a 40-song prodlike probe
  (88→224 atom rows) with no change to reconstruction fidelity. Default-ON
  behaviour now collapses more oscillation chains.

### Refuted

- `VoiceTrackPass` (`voice_track_pass`, default OFF) is marked REFUTED: a
  40-song headroom probe found zero ≥10-frame cross-voice tracking spans under
  the multiplicative, held-value, and additive-offset models. FREQ is a
  cent-bin index where intervals are additive, not the multiplicative
  `round(lead·ratio)+detune` the pass models; sustained tracking is absent and
  short chord-change overlaps are already absorbed by FREQ_RUN/FREQ_NUDGE. Code
  left in place, kept default-OFF.

## [0.10.0]

### Added

- Canonical-spec tokenizer primitives, each with synthetic round-trip
  tests:
  - `OSCILLATE_ENV_OP` (45, **default on**): envelope-modulated oscillation
    collapsing alternating-sign SLOPE chains; 8 parametric envelope families
    in `macros/envelope.py`; `OscillationEnvelopePass` + decoder.
  - `LonelyWriteValidatorPass` (`macros/lonely_validator.py`, behind
    `strict_lonely`, default off): raises `UnmodelledLonelyWriteError` for
    any full SET off the carveout allow-list and for any DIFF op. Carveout
    classifier includes the trajectory-anchor extension (a SET adjacent to a
    SLOPE/OSCILLATE_ENV/FLIP/FLIP2/TRANSPOSE primitive).
  - `TRACK_REF_OP` (46, behind `voice_track_pass`): cross-voice FREQ
    tracking via exact interval-ratio + constant detune.
  - `FREQ_NUDGE_OP` (47, `freq_nudge_pass`), `FREQ_RUN_OP` (48,
    `freq_run_pass`), `RELEASE_UPDATE_OP` (49, `release_update_pass`),
    `CTRL_TRIPLE_OP` (50, `ctrl_triple_pass`).
- `macros.envelope` module (parametric envelope fit + reconstruct).
- Per-op decoders for the new ops in `macros/decoders.py`; matching
  `pending_*` fields and a per-frame `pending_track_links` reconstructor in
  `DecodeState`.

### Changed

- `RegLogParser.parse()` pipeline gains the new passes. Every
  residual-absorbing primitive and the validator are gated behind
  default-off arg flags, so default tokenizer output is unchanged;
  `OSCILLATE_ENV` is default-on (additive, passes the audio-invariant
  suite).

## [0.9.0]

### Added

- `VocabArrays` (`constrained_decode`): dict subclass with attribute access
  alongside dict semantics. Returned by `precompute_vocab_arrays` and
  `precompute_subtoken_arrays`. External dict consumers are unaffected.
- `PassBackedTransform`, `RowExpandingTransform` (`macros.transform`):
  public bases for `Transform` subclasses whose `forward()` is a single
  `MacroPass.apply` and (optionally) whose `inverse()` decomposes
  `OP_CODES` rows. Hoisted from `transforms_bit_exact.py` so other
  transform files can reuse the pattern.
- `to_int64_arrays` (`utils`): public version of the private `_int64_cols`
  helper. Extracts named columns as int64 numpy arrays with explicit
  per-column `fillna={col: value}` mapping. Used by `constrained_decode`
  and several transforms.
- `MacroShape` IntEnum (`constrained_decode`): names the 23 macro shapes
  the sub-token classifier emits. Replaces the previous string-tag tuple.
- `OverlaySlot` IntEnum (`constrained_decode`): names the 3 overlay-slot
  positions for the `pending_overlay_slot` state machine.
- Internal `macros.transform_registry` module holding the shared
  pipeline-spec primitives (`_REGISTRY`, `PipelineEntry`,
  `PipelineConfigError`, `_normalize_spec`, `register`,
  `ensure_default_transforms_registered`). Lets `transform.py` and
  `pipeline_check.py` both depend on it without cycling. Consumers
  should keep importing from `macros.transform` (re-export).
- `__all__` declarations on every public top-level module and every
  `macros/*.py` (except `macros/__init__.py`, which is itself a
  re-export surface).

### Changed

- `constrained_decode.py` restructured:
  - `compute_invalid_mask` unifies the previous
    `_compute_invalid_atomic` / `_compute_invalid_subtoken` pair via
    per-mode gate tables (`_ATOMIC_SLOT_GATE` / `_SUBTOKEN_SLOT_GATE`).
  - `_update_atomic` is now table-driven via `_ATOMIC_SLOT_TRANSITION`
    and `_ATOMIC_NEW_PENDING`.
  - `_classify_macro_shape` lifted to module scope and rewritten as
    a `_HEAD_RULES` table + matcher (BR_LEN_WITH_TAIL stays
    out-of-table). Direct unit tests added.
  - `precompute_subtoken_arrays`' 23-arm shape-flag switch replaced
    by a declarative `_SHAPE_HANDLERS` table.
  - Per-sub-token frame-walking state machine extracted to
    `_walk_frame_aggregates` + `_FrameAggregates` dataclass. Direct
    unit tests added.
- `transforms_audio_bit_exact.py`: 7 of 8 Transform subclasses now
  inherit from `PassBackedTransform`. ~60 LoC of boilerplate removed.
- `Corpus` (`corpus.py`): two closures in `preload` promoted to methods
  (`_build_df_map_frame`, `_write_reg_widths_sidecar`); `_collect_atoms`
  lifted from `make_tokens` closure to module scope.
- Two import cycles broken:
  - `macros/__init__.py` → `coarsen_pass.py` (via lazy
    `_maybe_append_coarsen_pass`): `coarsen_pass.py` now imports
    `OVERLAY_BODY_FREQ_DELTA` from `macros.loops` directly; `CoarsenPass()`
    is a normal eager entry in `POST_NORM_PRE_VOICE_PASSES`.
  - `transform.py` → `pipeline_check.py` (via lazy import inside
    `TransformPipeline.from_spec`): `register` and
    `ensure_default_transforms_registered` moved into
    `transform_registry.py`; `pipeline_check.py` calls
    `ensure_default_transforms_registered()` at the top of
    `validate_pipeline_spec` instead of eager-importing
    `transforms_parser_stubs`. Top-level import of
    `validate_pipeline_spec` in `transform.py` now resolves cleanly.

### Removed

- Back-compat aliases now that consumers have cut over:
  `_frame_marker_count` (use `frame_marker_count`),
  `StreamState._compute_invalid` (use `compute_invalid_mask`),
  `_LOSS_TIER_NAMES` (use `LOSS_TIER_NAMES`).
- `MIN_DIFF` is now `_MIN_DIFF` in `stfconstants.py`. The re-export
  in `macros/__init__.py` is dropped. Internal callers and tests use
  the private name.
- Four `LegatoCluster{2,3,4,7}Decoder` subclasses dropped;
  `_LegatoClusterNibbleDecoder` / `_LegatoClusterByteDecoder` bases
  are now parameterised on `op_code` at construction.
- `macros/__init__.py` re-exports trimmed to what tests / main repo
  actually consume (163 → 83 LoC). Internal helpers
  (decoders, state internals, loops internals, walker, etc.) must be
  imported from their source modules.
- `macros.passes_base._int64_cols` removed; callers migrated to
  `utils.to_int64_arrays`.

### Known issues / outstanding work

- `reglog_helpers.{dump,load}_palettes_attrs` and
  `reglog_helpers.wrapbits` re-exports are still present, blocked on
  a main-repo `render_play.py` cutover.
- Several diminishing-returns refactors documented under "Outstanding
  work" — merging `_HEAD_RULES` / `_SHAPE_HANDLERS`, `Corpus` full
  split, `Transform.register` via `__init_subclass__`. Defer until
  there's a paired motivating change.

## [0.8.0]

API surface narrowing round 2: helper consolidations, `macros.roles`
predicates, `VocabSignature` single-pass classifier, new boundary
constants (`DEFAULT_IRQ_CYCLES`, `LOSS_TIER_NAMES`).

## [0.7.0]

First release after the parsing/tokenization extraction from the
main `preframr` research codebase.
