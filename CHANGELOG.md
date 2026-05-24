# Changelog

All notable changes to this project will be documented in this file.

## [0.10.0]

### Added

- TOKEN_IMPROVEMENTS.md canonical-spec tokenizer primitives, each with
  synthetic round-trip tests (see `TOKEN_IMPROVEMENTS.md` "Implementation
  log" for per-primitive design decisions and the postponed validation
  gates):
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

See `API_SURFACE.md`:

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
constants (`DEFAULT_IRQ_CYCLES`, `LOSS_TIER_NAMES`). See
`API_SURFACE.md` for the full inventory and the per-helper
"replaces" mapping.

## [0.7.0]

First release after the parsing/tokenization extraction from the
main `preframr` research codebase.
