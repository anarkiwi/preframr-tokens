# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
