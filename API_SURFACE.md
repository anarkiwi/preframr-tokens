# API surface: inventory + narrowing brief

This document is **for an agent / contributor managing the
preframr-tokens public API before a release**. It catalogues:

1. What the main `preframr` repo (the only first-class consumer
   today) imports from `preframr_tokens`.
2. Which of those imports are stable contracts vs candidates for
   further narrowing.
3. Concrete next steps for a smaller surface.

## Design principle

**Expose decisions, not facts.** If a consumer imports a raw
`stfconstants` value (a reg id, an op code, a subreg index) only
to plug it into a switch / classification / arithmetic, that's an
indicator the helper should live here. The helper rides the same
code as the parser/tokenizer that produces the data, so the
classification matches by construction.

Counter-example: `PAD_ID`, `DEFAULT_IRQ_CYCLES`, `MODEL_PDTYPE`,
`DUMP_SUFFIX`, `LEGACY_EVAL_SUBSET_NAME`, `LOSS_TIER_NAMES` are
genuine boundary constants — they name a single thing and have no
behaviour around them. Those stay imported by callers.

## What `preframr` consumes today

Snapshot from `grep -rn 'from preframr_tokens'` in the main repo
(post the helper additions in the version this doc ships with).

### Top-level public imports

| symbol | module | consumer | purpose |
|---|---|---|---|
| `parse_corpus` | `parse_runner` | `preframr/parse.py` | CLI entry shim |
| `Corpus` | `corpus` | `train/regdataset.py` | corpus orchestration |
| `RegLogParser` | `reglogparser` | `train/regdataset.py`, `inference/{predict,render_play}.py` | parser construction |
| `remove_voice_reg` | `reglogparser` | `train/regdataset.py`, `inference/predict.py` | voice-reg masking |
| `prepare_df_for_audio` | `reglogparser` | `inference/{predict,render_play}.py` | renderer-ready df |
| `DUMP_SUFFIX` | `reglogparser` (re-export) | `inference/render_play.py` | file-ext sniff |
| `RegTokenizer` | `regtokenizer` | `train/structural_loss.py` (+ helpers below pre-cutover) | tokenizer construction |
| `LEGACY_EVAL_SUBSET_NAME` | `blocks` | `train/regdataset.py` | eval-subset routing |
| `self_contained_prompt_df`, `iter_voiced_blocks`, `materialize_block_array`, `parser_worker`, `glob_dumps`, `reg_widths_path`, `SeqMeta`, `parse_eval_reglogs` | `blocks` | `train/regdataset.py` | block iteration |
| `tier_accuracy`, `detect_tail_cycle`, `distinct_n` | `audit_primitives` | `train/generalization_gate.py` | decision helpers |
| `StreamState`, `precompute_vocab_arrays`, `precompute_subtoken_arrays`, `_frame_marker_count` | `constrained_decode` | `inference/predict.py`, `train/structural_loss.py` | mask state machine |
| `validate_back_refs`, `validate_pattern_overlays` | `macros` | `inference/predict.py` | post-decode validation |
| `load_palettes_attrs` | `reglog_helpers` | `inference/render_play.py` | palette IO |
| `PAD_ID`, `MIN_DIFF`, `MODEL_PDTYPE`, `FRAME_REG` | `stfconstants` | various | see "Boundary constants" |

### New decision helpers (added this release)

| helper | module | replaces |
|---|---|---|
| `vocab_id_tier`, `build_vocab_tier_ids`, `build_vocab_tier_map`, `CONTENT_TIER` | `tier_classify` | preframr's `_vocab_id_to_class_tier` + `_build_vocab_tier_id` + `_build_vocab_class_weight` + `build_tier_map` (which imported `DELAY_REG`, `FRAME_REG`, `MODE_VOL_REG`, `VOICE_CTRL_REG`, `FILTER_REG`, `collect_op_loss_tiers` + the two `transforms_*` registration-side-effect modules) |
| `vocab_frame_weights` | `token_weighting` | preframr's `_build_vocab_frame_weight` (which imported `BACK_REF_OP`, `BACK_REF_SUBREG_LEN`, `DELAY_REG`, `DO_LOOP_OP`, `FRAME_REG`, `SLOPE_OPS`, `SLOPE_SUBREG_RUNTIME`) |
| `VocabSignature` | `vocab_signature` | single-pass class behind both wrappers above; carries `tier_ids`, `tier_names`, `frame_weights` |
| `read_initial_irq` | `reglog_helpers` | preframr's `df[df["reg"] == FRAME_REG]` + diff-lookup + magic `19656` default |
| `tail_charge_for_prompt` | `constrained_decode` | preframr's `is_real_reg_np[tail].sum() * MIN_DIFF` arithmetic |
| `frame_marker_count` | `constrained_decode` | was `_frame_marker_count`; consumer-imported as a leak in the previous round. Public name + back-compat alias. |
| `StreamState.compute_invalid_mask` | `constrained_decode` | was `_compute_invalid`; test surface stops needing `# pylint: disable=protected-access`. |
| `ensure_default_transforms_registered` | `macros.transform` | the duplicated `from preframr_tokens.macros import transforms_audio_bit_exact, transforms_bit_exact  # noqa` side-effect import that 3+ call sites carried. Idempotent. |
| `distance_pair_role`, `slope_subreg_role`, `frame_weight_role`, `DISTANCE_PAIR_OPS`, `DistancePairSpec` | `macros.roles` | central `(op, subreg) → role` predicates. Consumed by `validators._step_distance_pair`, `transforms_voice_trajectory`, and `vocab_signature`. |
| `LOSS_TIER_NAMES` (was `_LOSS_TIER_NAMES`) | `macros.transform` | preframr's local `_LOSS_TIER_ORDER` duplicate |
| `DEFAULT_IRQ_CYCLES` | `stfconstants` | preframr's hard-coded `19656` |
| `TokenizeMeta` | `corpus` | typed dataclass replacing the untyped `Corpus._tokenize_meta` dict |
| `PendingSlot` | `constrained_decode` | `IntEnum` consolidating the 7 mutually-exclusive `pending_*` booleans on `StreamState` (booleans remain as bidirectional `@property` shims for back-compat) |
| `VocabArrays` | `constrained_decode` | `dict` subclass with `__getattr__` for attribute access (`a.is_real_reg` alongside `a["is_real_reg"]`). Return type of `precompute_vocab_arrays` and `precompute_subtoken_arrays`. External consumers retain dict semantics. |
| `PassBackedTransform`, `RowExpandingTransform` | `macros.transform` | Public bases for `Transform` subclasses whose `forward()` is a single `MacroPass.apply` and (optionally) whose `inverse()` decomposes ``OP_CODES`` rows via a per-row `_expand_row` staticmethod. Hoisted from `transforms_bit_exact.py` so other transform files can reuse the pattern. |
| (internal) `transform_registry` module | `macros.transform_registry` | Holds the shared pipeline-spec primitives (``_REGISTRY``, ``PipelineEntry``, ``PipelineConfigError``, ``_normalize_spec``) so `transform.py` and `pipeline_check.py` can both depend on them without a cycle. Consumers import via `transform.py` re-exports. |
| `to_int64_arrays` | `utils` | Public version of the private `_int64_cols` helper. Extracts named columns from a df as int64 numpy arrays with explicit per-column ``fillna={col: value}`` map. Sweeps the 10+ ad-hoc `df[col].fillna(default).astype(np.int64).to_numpy()` triples that were scattered across the package. |

### Boundary constants (legitimate, no helper needed)

| constant | type | what it is |
|---|---|---|
| `PAD_ID` | int | the padding token id |
| `MODEL_PDTYPE` | pandas dtype | dtype for parquet IO of token-shape dfs |
| `DUMP_SUFFIX` | str | `.dump.parquet` file extension |
| `LEGACY_EVAL_SUBSET_NAME` | str | name of the legacy single-eval subset |
| `DEFAULT_IRQ_CYCLES` | int | PAL raster IRQ default (19656 cycles) |
| `LOSS_TIER_NAMES` | tuple[str, ...] | canonical tier ordering for partitioning heads |

## Leaks to clean up (next agent)

After the helpers in this release land in `preframr`, these become
the **remaining** raw-constant or private-symbol imports in the
main repo. Each is a candidate for further narrowing.

All consumer-side cutovers are now done as of `preframr` commit
`a7d8cf3` (`cutover to preframr-tokens 0.8.0 + preframr-audio
0.2.0 helpers`) and `preframr-audio` commit `efeaa45` (`fidelity:
route _irq_from_df through preframr_tokens.read_initial_irq`,
shipped in `preframr-audio==0.2.0`). The next pass is now about
**dropping the back-compat aliases** that those cutovers made
obsolete.

### Status (after cutover round 4)

| symbol | leak shape | resolution | status |
|---|---|---|---|
| `MIN_DIFF` (stfconstants) | Was re-exported and imported by `preframr/inference/predict.py:37` for inline tail-charge math. | Replaced by `tail_charge_for_prompt(prompt_ids, vocab_arrays)`; predict.py no longer imports `MIN_DIFF`. Renamed to `_MIN_DIFF` in `stfconstants.py`; internal callers + tests updated. The re-export in `macros/__init__.py` is also dropped. | **done** |
| `FRAME_REG` in `preframr/inference/render_play.py` | `df[df["reg"] == FRAME_REG]` + diff lookup. | Replaced by `read_initial_irq(df)`. | **done** |
| `FRAME_REG` in `preframr-audio/preframr_audio/fidelity.py:201` | `df[df["reg"] == FRAME_REG]` + diff lookup with explicit raise on empty. | Shipped in `preframr-audio==0.2.0` as `read_initial_irq(df, default=_IRQ_MISSING_SENTINEL)` with sentinel check preserving the raise contract. | **done** |
| `_LOSS_TIER_NAMES` | Underscore-prefixed alias kept for back-compat against pre-cutover preframr. | Main repo cuts over to public `LOSS_TIER_NAMES` in `tier_map.py`. The alias in `macros/transform.py` is now dropped. | **done** |
| `_frame_marker_count` (constrained_decode) | Private-prefix imported by `preframr/inference/predict.py`. | Renamed + main repo imports public `frame_marker_count` directly. Alias in `constrained_decode.py` is dropped. | **done** |
| `StreamState._compute_invalid` | Protected method called from tests via `# pylint: disable=protected-access`. | Public `compute_invalid_mask` landed; tests use the public name. The `_compute_invalid = compute_invalid_mask` class-level alias is dropped. | **done** |
| `vocab_id_tier` re-implementation in `preframr/train/model/tier_map.py` | Main repo open-coded the `(reg, op) → tier` switch. | `tier_map.py` is now a thin torch adapter over `vocab_id_tier` / `build_vocab_tier_ids` / `build_vocab_tier_map`; dropped imports of `DELAY_REG`, `FRAME_REG`, `MODE_VOL_REG`, `VOICE_CTRL_REG`, `FILTER_REG`, `collect_op_loss_tiers`, and the two `transforms_*` side-effect modules. | **done** |
| `vocab_frame_weights` re-implementation in `preframr/train/model/losses.py:140-185` | Main repo open-coded the frame-time weighting loop. | `_build_vocab_frame_weight` is now a 5-line `torch.from_numpy(vocab_frame_weights(...))` wrap; dropped imports of `BACK_REF_OP`, `BACK_REF_SUBREG_LEN`, `DELAY_REG`, `DO_LOOP_OP`, `FRAME_REG`, `SLOPE_OPS`, `SLOPE_SUBREG_RUNTIME`. | **done** |
| `Transform.round_trip_check` lazy-imports `preframr_audio` | Optional dep; only triggers for lossy macros. | Documented in README via `[audio]` extra; `preframr-audio>=0.2.0` declared in main repo's pins so the lazy import resolves cleanly. | intentional |

### Back-compat aliases scheduled to drop next release

The three internal aliases below have been dropped this round
(`_frame_marker_count`, `StreamState._compute_invalid`,
`_LOSS_TIER_NAMES`). What remains is the `reglog_helpers` re-export
set, which is **blocked on a main-repo cutover** of `render_play.py`
(currently imports `load_palettes_attrs` and `wrapbits` from
`reglog_helpers` rather than from their source modules).

Drop on the release after the main-repo cutover:

- `preframr_tokens.reglog_helpers.{dump_palettes_attrs,load_palettes_attrs}`
  re-exports from `preframr_tokens.palette_io` — verify no main-repo
  call site still imports these from the old path before dropping
  (`render_play.py` imports `load_palettes_attrs` from
  `reglog_helpers` today; cut it over to `palette_io` first).
- `preframr_tokens.reglog_helpers.wrapbits` re-export from
  `preframr_tokens.utils` — same drill; verify no caller before
  dropping.

### Audit pending

Things the next agent should grep for in main repo BEFORE cutover, to confirm scope:

- `from preframr_tokens.stfconstants import` — every import in
  `preframr/` should reduce to the "Boundary constants" set above
  after consumers move to helpers.
- `from preframr_tokens.macros import` — should reduce to
  `validate_back_refs`, `validate_pattern_overlays` (decision
  helpers) only. Any consumer importing `transforms_audio_bit_exact`
  or `transforms_bit_exact` directly (for registration side
  effects) is a leak — the helpers here lazy-import them.
- `from preframr_tokens.regtokenizer import` — should be just the
  `RegTokenizer` class. Direct imports of private helpers from
  inside `regtokenizer.py` (`_snap`, `_split_reg`, etc.) are
  leaks.
- `_compute_invalid` on `StreamState` — leaked as a protected
  method in `preframr-tokens/tests/test_constrained_decode.py`
  after the torch detach. Should be promoted to public
  `compute_invalid_mask()` (callable in numpy land without the
  `_` ceremony) so consumers stop poking through.

### Will not narrow (intentional shape)

| pattern | why kept |
|---|---|
| `precompute_vocab_arrays` returns a dict of numpy arrays | Dict shape is fast iteration; consumers index named keys. A typed `VocabArrays` wrapper would add ceremony without payoff. |
| `RegLogParser` takes an `argparse.Namespace` | Matches how main repo wires args. Detaching to a typed dataclass would force every consumer to translate. |
| `Corpus` carries `args` through methods | Same reason. |
| `BlockMapper` / DataLoader wrapping lives in main repo, not here | The torch-free guarantee is load-bearing; never accept torch deps here. |

## Surface reduction goals before v1.0

Tracked against the "Back-compat aliases scheduled to drop" list
above. After the main-repo cutover round, the bulk of these
are mechanical drops in `preframr_tokens` itself with no
consumer-side coordination required.

- `from preframr_tokens import *` shouldn't yield `_FOO` symbols at
  module top-level. Promote or hide. **done for**
  `_frame_marker_count`, `_compute_invalid`, `_LOSS_TIER_NAMES` (all
  aliases now removed). No `_FOO` re-exports remain at module-public
  level beyond the `reglog_helpers` back-compat set.
- `MIN_DIFF` should be a `_MIN_DIFF` module-private once preframr
  is on `tail_charge_for_prompt`. **done** — renamed in
  `stfconstants.py`; internal callers (`constrained_decode.py`,
  `macros/state.py`, `reglogparser.py`) and tests updated; the
  re-export in `macros/__init__.py` is dropped.
- `_frame_marker_count` should be `frame_marker_count`. **done**
  (rename + consumer cut over + alias removed).
- `StreamState.compute_invalid_mask()` should be public so tests
  don't have to call `_compute_invalid`. **done** (rename + tests
  cut over + alias removed).
- Cut over main-repo `losses.py` and `tier_map.py` to
  `vocab_frame_weights` / `vocab_id_tier` (or `VocabSignature`).
  **done** (`a7d8cf3`; `tier_map.py` is a 90-LoC torch adapter,
  `losses.py::_build_vocab_frame_weight` is 5 LoC).
- Cut over `preframr-audio/fidelity.py` to `read_initial_irq`.
  **done** (`preframr-audio==0.2.0`; main repo bumps the pin in
  `a7d8cf3`).
- Cut over `render_play.py` consumers of `reglog_helpers.{dump,load}_palettes_attrs`
  + `wrapbits` to the new homes (`palette_io.py` / `utils.py`).
  **next:** mechanical preframr-side change — drops the
  `reglog_helpers` re-exports without breaking anything.
- Document each module's `__all__` (currently most modules don't
  declare one). **done for the public modules** (`audit_primitives`,
  `palette_io`, `tier_classify`, `token_weighting`, `vocab_signature`,
  `alphabet_projection`, `reg_mappers`, `utils`, `parse_runner`,
  `macros/roles`, `macros/transform`, `constrained_decode`, `corpus`,
  `blocks`, `regtokenizer`, `reglogparser`, `reglog_helpers`).
  Helper-only / re-export modules (`engine_fingerprint`, `dump_meta`,
  `coarsen_pass`, `macros/__init__.py`) intentionally omitted.

## Next round (mechanical, no consumer coordination)

Suggested commit shape for the next preframr-tokens minor bump:

1. Cut over main repo's `render_play.py` to import
   `load_palettes_attrs` from `preframr_tokens.palette_io` directly
   (currently goes through `reglog_helpers` re-export). This is the
   prerequisite for dropping the `reglog_helpers` re-exports cleanly.
2. Drop the `reglog_helpers.{dump_palettes_attrs,load_palettes_attrs,wrapbits}`
   re-exports once step 1 is in place. Update the `__all__` in
   `reglog_helpers.py` accordingly.
3. Bump version, update CHANGELOG.

## How to verify "narrowed enough" before release

1. In the main repo, run:
   ```bash
   grep -rn '^from preframr_tokens\|^from preframr_tokens\..*import' \
       preframr/ | sort -u
   ```
   The result should be one of: a class, a public decision helper, or
   a boundary constant from the "legitimate" set.
2. In preframr-tokens, run:
   ```bash
   grep -rn '^def [^_]\|^class [^_]' preframr_tokens/*.py
   ```
   Every public symbol should either be in this doc or be a
   genuine new addition.
3. The torch-free regression test
   (`test_constrained_decode.py::TestModuleTorchFree`) must keep
   passing.

## Versioning hint

When a helper moves from main repo to here, bump minor (new
public surface). When `_LOSS_TIER_NAMES`-style aliases are dropped
or `_FOO` privatisations land, bump minor again (subtractive but
intentional). When token-alphabet shape or `precompute_vocab_arrays`
dict keys change, bump major (downstream checkpoints invalidated).
