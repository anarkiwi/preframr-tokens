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

### Confirmed leaks (need followup in `preframr_tokens`)

| symbol | leak shape | suggested fix | status |
|---|---|---|---|
| `MIN_DIFF` (stfconstants) | Used as a re-export inside `constrained_decode`. After `tail_charge_for_prompt` cutover preframr no longer imports it, but `predict.py:37` still does. | Hide as `_MIN_DIFF` once preframr cutover lands; verify no other reach. | open (preframr cutover pending; `predict.py:37` still imports) |
| `FRAME_REG` in `preframr/inference/render_play.py` | Only used by the `df[df["reg"] == FRAME_REG]` snippet that `read_initial_irq` replaces. | preframr-side cutover (this doc only flags; impl is in preframr). | open (still imported at `render_play.py:26`) |
| `FRAME_REG` in `preframr-audio/preframr_audio/fidelity.py:201` | Same shape as the `render_play.py` leak; not previously catalogued because `preframr-audio` wasn't audited in the previous round. | Add a small audio-side decision helper (e.g. `is_frame_row(df)`) or hoist `read_initial_irq` to the audio side. | open (new; flagged by the v0.8 narrowing pass) |
| `_LOSS_TIER_NAMES` | Underscore-prefixed alias kept for back-compat against pre-cutover preframr. Drop once main repo is on the LOSS_TIER_NAMES name. | Remove after one preframr release cycle. | open (alias retained; main-repo cutover pending) |
| `_frame_marker_count` (constrained_decode) | Private-prefix but imported by `preframr/inference/predict.py`. The helper is genuinely useful; rename to public `frame_marker_count`. | Rename + add back-compat private alias. | **done** (public name landed; alias retained) |
| `StreamState._compute_invalid` | Protected method called from tests via `# pylint: disable=protected-access`. | Promote to `compute_invalid_mask`; keep alias for one cycle. | **done** (public method landed; alias retained) |
| `vocab_id_tier` re-implementation in `preframr/train/model/tier_map.py:11-55` | Main repo open-codes the exact `(reg, op) → tier` switch that `vocab_id_tier` / `VocabSignature` exposes. Reaches into 5 raw `stfconstants` (`DELAY_REG`, `FRAME_REG`, `MODE_VOL_REG`, `VOICE_CTRL_REG`, `FILTER_REG`) plus does its own `transforms_audio_bit_exact` / `transforms_bit_exact` side-effect import dance. | preframr-side cutover: call `vocab_id_tier(vid, rt, tokens)` or build a `VocabSignature`. | open (new; helper exists, consumer hasn't cut over) |
| `vocab_frame_weights` re-implementation in `preframr/train/model/losses.py:140-185` | Same shape — main repo open-codes the frame-time weighting loop. Reaches into 7 raw `stfconstants` (`BACK_REF_OP`, `BACK_REF_SUBREG_LEN`, `DELAY_REG`, `DO_LOOP_OP`, `FRAME_REG`, `SLOPE_OPS`, `SLOPE_SUBREG_RUNTIME`). | preframr-side cutover: call `vocab_frame_weights(rt, tokens, n_vocab)` or build a `VocabSignature`. | open (new; helper exists, consumer hasn't cut over) |
| `Transform.round_trip_check` lazy-imports `preframr_audio` | Optional dep; only triggers for lossy macros. Documented in README via `[audio]` extra. | No fix needed; document only. | intentional |

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

- `from preframr_tokens import *` shouldn't yield `_FOO` symbols at
  module top-level. Promote or hide. **partial:** `_frame_marker_count`
  and `_compute_invalid` promoted; `_LOSS_TIER_NAMES` alias still
  present pending main-repo cutover.
- Every `_LOSS_TIER_NAMES`-style alias should be gone (one release
  cycle of grace, then drop). See README "Back-compat aliases scheduled
  to drop" for the full list.
- `MIN_DIFF` should be a `_MIN_DIFF` module-private once preframr
  is on `tail_charge_for_prompt`. **still open** — `predict.py:37`
  hasn't cut over.
- `_frame_marker_count` should be `frame_marker_count`. **done.**
- `StreamState.compute_invalid_mask()` should be public so tests don't
  have to call `_compute_invalid`. **done.**
- Document each module's `__all__` (currently most modules don't
  declare one). **partial:** `reglog_helpers.py` declares one;
  others still don't.
- Cut over main-repo `losses.py` and `tier_map.py` to
  `vocab_frame_weights` / `vocab_id_tier` (or `VocabSignature`).
  The helpers exist; main repo just hasn't switched. New goal flagged
  by the v0.8 external-consumer audit.

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
