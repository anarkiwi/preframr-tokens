# preframr-tokens

SID reglog parsing, tokenization, and macro transforms extracted from
the [preframr](https://github.com/anarkiwi/preframr) research codebase.

Torch-free. The training-side concerns (model, loss, DataLoader,
predict) live in the main `preframr` repo; this package contains the
stable parsing + encoding layer that produces the parsed parquets +
unigram tokenizer alphabet that downstream training consumes.

## Install

```bash
pip install preframr-tokens
```

Optional extras:

- `preframr-tokens[audio]` — pulls `preframr-audio` for the lossy-macro
  round-trip check (`Transform.round_trip_check` lazy-imports
  `preframr_audio.fidelity.assert_dfs_render_equivalent` for any
  `TIER != "bit_exact"` macro). Skip if you only use the parsing /
  tokenisation / constrained-decode paths.

## Importing

Import from the package root:

```python
from preframr_tokens import RegLogParser, RegTokenizer, Corpus, reg_class
```

`preframr_tokens.__all__` is the semver-promised surface. Two submodules
are also public, stable namespaces you import directly:
`preframr_tokens.stfconstants` (reg ids, op codes, dtypes, PAL clock) and
`preframr_tokens.engine_fingerprint` (feature-vector layout, `ClusterTable`,
`compute_fingerprint`). Every other `preframr_tokens.*` submodule path is
**internal and may move between releases** — depend on the root re-exports
instead. The module list below documents that internal structure.

## Modules

- `preframr_tokens.reglogparser` -- SID dump → parsed dataframe
  pipeline. `RegLogParser`, plus `read_initial_irq` (first-frame IRQ
  read off a parser-output df, with PAL default).
- `preframr_tokens.regtokenizer` -- alphabet build + unigram tokenizer
  fit. `RegTokenizer`.
- `preframr_tokens.macros.*` -- declarative `Transform` registry plus
  the macro / pre-norm passes (slope, preset, hard_restart,
  legato_per_cluster, voice_block_order, ctrl_bigram, loop, etc.).
  Macros declare `OP_CODES`, `LOSS_TIER`, `SUBSTITUTABLE_OPS`,
  `MUST_FOLLOW`, etc. on their classes; `pipeline_check.validate_pipeline_spec`
  validates a pipeline declaratively.
- `preframr_tokens.stfconstants` -- SID register IDs, op codes, pandas
  dtypes, PAL clock constants.
- `preframr_tokens.engine_fingerprint` -- engine clustering for
  cross-engine evaluation pinning.
- `preframr_tokens.coarsen_pass` -- tracker-export pass (lossy
  audio-domain bucketing).
- `preframr_tokens.dump_meta` -- per-dump metadata sidecar with code-hash
  staleness gate.
- `preframr_tokens.reg_match` -- voice-relative register classification:
  raw `reg` id → boolean row mask (`freq_match`, `pcm_match`,
  `ctrl_match`, `adsr_match`, `ad_match`, `sr_match`, `filter_match`,
  `frame_match`, built on `vreg_match`), plus `reg_class(reg) -> (kind,
  voice)` for scalar per-reg classification. Pure-`stfconstants`
  parse-domain sibling of `macros.roles`.
- `preframr_tokens.palette_io` -- JSON sidecar load/dump for the
  engine-fingerprint / engine-fp-cluster `df.attrs`.
- `preframr_tokens.macros.roles` -- single source of truth for macro
  `(op, subreg)` → role classification. `distance_pair_role`,
  `slope_subreg_role`, `frame_weight_role`, plus the
  `DISTANCE_PAIR_OPS` table and the `DistancePairSpec` dataclass. The
  parse-domain reg-id counterpart is `reg_match`.
- `preframr_tokens.vocab_signature` -- `VocabSignature` class. Single-
  pass per-vocab-id (loss-tier, frame-time-weight) computation. The
  `tier_classify` and `token_weighting` free functions are thin
  wrappers; consumers that need both should build a `VocabSignature`
  directly to avoid two passes over the vocab.
- `preframr_tokens.alphabet_projection` -- eval-set atom projection
  table.
- `preframr_tokens.reg_mappers` -- `FreqMapper` (PAL clock + cents
  quantization).
- `preframr_tokens.constrained_decode` -- per-step structural-validity
  mask for sampling-time logit guarding. Pure numpy state machine;
  consumers (torch users) apply the returned bool mask with a single
  `masked_fill` at the boundary.
- `preframr_tokens.blocks` -- block iteration + materialization
  helpers: `iter_voiced_blocks`, `materialize_block_array`,
  `parser_worker`, `glob_dumps`, `reg_widths_path`,
  `self_contained_prompt_df`, plus the `SeqMeta` dataclass and
  `parse_eval_reglogs` / `LEGACY_EVAL_SUBSET_NAME` for eval-subset
  routing. Torch-free; main repo's RegDataset wraps the outputs in
  DataLoaders.
- `preframr_tokens.audit_primitives` -- pure-Python token-level
  audit functions: `tier_accuracy` (per-tier hit-rate + content/
  structural ratio), `detect_tail_cycle` (loop-collapse detector),
  `distinct_n` (n-gram diversity). Used by the generalization-gate
  callback in main repo and by post-hoc audit scripts.
- `preframr_tokens.parse_runner` -- `write_df(args, logger, dump_file)`
  + `parse_corpus(args, logger)` parallel dump-parsing orchestrator.
  Main-repo `preframr/parse.py` is a thin argparse shim around this.
- `preframr_tokens.corpus` -- `Corpus` class: torch-free corpus
  orchestration owning the RegTokenizer + reg_widths +
  tokenize-stage metadata. Methods `load_dfs`, `make_tokens`,
  `encode_and_save_cached_blocks`, `try_preload_from_disk`,
  `preload`, `iter_block_seqs`, `iter_predict_block_seqs` cover
  the full parse → tokenize → load pipeline up to the point where
  blocks need to be routed into a torch `BlockMapper` (main repo's
  RegDataset is a thin adapter that does that routing).

## Library-only

No CLI entry points. Consumers build their own (the main `preframr`
repo's `parse.py` and `stftokenize.py` are simple wrappers that
construct `RegLogParser` / `RegTokenizer` from an `argparse.Namespace`).

## API surface

Design principle: **expose decisions, not facts.** Whenever a
consumer would otherwise import raw `stfconstants` (reg ids, op
codes, subreg constants) and re-implement a classification switch
on top of them, that's a sign the helper should live here instead.
Helpers ride the same code as the parser/tokenizer, so the
classification matches the data by construction.

The decision helpers below were added in that vein; each replaces an
ad-hoc reg/op classification or arithmetic that consumers used to
open-code on top of raw `stfconstants`:

- `preframr_tokens.tier_classify` — `vocab_id_tier`,
  `build_vocab_tier_ids`, `build_vocab_tier_map`, `CONTENT_TIER`.
  Replaces ad-hoc reg/op tier classification in consumers.
- `preframr_tokens.reg_match.reg_class` — scalar `reg -> (kind, voice)`
  classification (`"FREQ" | "PW" | "CTRL" | "AD" | "SR"`). Replaces the
  hand-built `{reg: (kind, voice)}` table consumers open-coded from the
  per-voice register layout.
- `preframr_tokens.token_weighting.vocab_frame_weights` — per-vocab
  audio-frame-time weighting. Replaces ad-hoc BACK_REF / DO_LOOP /
  SLOPE / DELAY / FRAME val accounting in consumers.
- `preframr_tokens.vocab_signature.VocabSignature` — single-pass
  bundle of both of the above. Consumers that need both `tier_ids`
  and `frame_weights` should construct this directly.
- `preframr_tokens.reglogparser.read_initial_irq` — first-frame
  diff lookup with PAL default. Replaces the `df[df["reg"] ==
  FRAME_REG]` dance in consumers.
- `preframr_tokens.constrained_decode.tail_charge_for_prompt` —
  cycle cost of real-reg writes after the last frame marker.
  Replaces the manual `is_real_reg[tail].sum() * MIN_DIFF`
  arithmetic + the matching `MIN_DIFF` import in consumers.
- `preframr_tokens.constrained_decode.frame_marker_count` —
  formerly `_frame_marker_count`; promoted (underscore alias dropped).
- `preframr_tokens.constrained_decode.StreamState.compute_invalid_mask`
  — formerly `_compute_invalid`; promoted (underscore alias dropped).
- `preframr_tokens.macros.transform.ensure_default_transforms_registered`
  — call before any `_REGISTRY` lookup to populate
  `transforms_audio_bit_exact` / `transforms_bit_exact` side effects.
  Idempotent. Replaces the duplicated import-and-cache dance.
- `preframr_tokens.corpus.TokenizeMeta` — typed snapshot of the
  tokenize-stage metadata previously carried as an untyped dict on
  `Corpus._tokenize_meta`.
- `preframr_tokens.constrained_decode.VocabArrays` — `dict` subclass
  with attribute access (`a.is_real_reg` alongside `a["is_real_reg"]`).
  Return type of `precompute_vocab_arrays` /
  `precompute_subtoken_arrays`; external dict consumers see no change.
- `preframr_tokens.macros.transform.PassBackedTransform`,
  `RowExpandingTransform` — public bases for `Transform` subclasses
  that wrap a `MacroPass` for `forward()` and (optionally) a decoder
  for `expand_atom()`. Hoisted from `transforms_bit_exact.py` so other
  transform files can reuse the pattern.
- `preframr_tokens.macros.transform_registry` (internal) — holds
  the shared pipeline-spec primitives (`_REGISTRY`, `PipelineEntry`,
  `PipelineConfigError`, `_normalize_spec`) so `transform.py` and
  `pipeline_check.py` can both depend on them without forming an
  import cycle. Consumers should keep importing from
  `preframr_tokens.macros.transform`, which re-exports.
- `preframr_tokens.utils.to_int64_arrays(df, *names, fillna={col: val})`
  — extract named columns as int64 numpy arrays with explicit per-column
  NaN fill values. Replaces 10+ ad-hoc
  `df[col].fillna(...).astype(np.int64).to_numpy()` triples.

## Stability

Library follows semver from v1.0. Pre-1.0 releases may break API as
the preframr codebase evolves. Token-alphabet shape changes bump
major version since they invalidate downstream checkpoints.

The authoritative promised surface is `preframr_tokens.__all__`
(importable from the package root), plus the `stfconstants` and
`engine_fingerprint` namespaces noted under "Importing". It groups as:

- **Classes**: `RegLogParser`, `RegTokenizer`, `Corpus`, `TokenizeMeta`,
  `StreamState`, `PendingSlot`, `VocabArrays`, `VocabSignature`,
  `Transform` (+ `register` decorator, `PipelineEntry`,
  `TransformPipeline`, `PassBackedTransform`, `RowExpandingTransform`),
  `DistancePairSpec`, and the pass classes `SlopePass`, `PresetPass`,
  `PerRegBurstPass`, `GateSlopeShiftPass`.
- **Decision helpers**: the `tier_classify` (`vocab_id_tier`,
  `build_vocab_tier_ids`, `build_vocab_tier_map`) / `token_weighting`
  (`vocab_frame_weights`) / `VocabSignature` / `read_initial_irq` /
  `reg_class` / `to_int64_arrays` family catalogued under "API surface".
- **Routines**: `parse_corpus`, `precompute_vocab_arrays`,
  `precompute_subtoken_arrays`, `prepare_df_for_audio`,
  `remove_voice_reg`, `validate_back_refs`, `validate_pattern_overlays`,
  `frame_marker_count`, `tail_charge_for_prompt`,
  `ensure_default_transforms_registered`, `get_transform_class`,
  `distance_pair_role`, `slope_subreg_role`, `frame_weight_role`,
  `classify_carveout`, `iter_voiced_blocks`, `reg_widths_path`,
  `self_contained_prompt_df`, `tier_accuracy`, `detect_tail_cycle`,
  `distinct_n`, `load_palettes_attrs`, `dump_palettes_attrs`.
- **Boundary constants**: `PAD_ID`, `MODEL_PDTYPE`, `DUMP_SUFFIX`,
  `LEGACY_EVAL_SUBSET_NAME`, `DEFAULT_IRQ_CYCLES`, `LOSS_TIER_NAMES`,
  `DISTANCE_PAIR_OPS`, `CONTENT_TIER`.

### Intentional shape (won't narrow)

- `precompute_vocab_arrays` / `precompute_subtoken_arrays` return a
  `dict` of numpy arrays (the `VocabArrays` subclass adds attribute
  access without breaking dict consumers) — fast iteration over named
  keys with no per-call wrapper ceremony.
- `RegLogParser` and `Corpus` take an `argparse.Namespace` and thread
  `args` through their methods — matches how the main repo wires them;
  a typed config object would force every consumer to translate.
- `BlockMapper` / DataLoader wrapping stays in the main repo — the
  torch-free guarantee here is load-bearing and never accepts a torch
  dependency.

### Removed back-compat aliases

All back-compat aliases have been removed. The internal aliases
(`_frame_marker_count`, `_compute_invalid`, `_LOSS_TIER_NAMES`) and the
public `MIN_DIFF` re-export went in the prior round; the `reglog_helpers`
re-export set (`dump_palettes_attrs`, `load_palettes_attrs`, `wrapbits`)
followed once the last main-repo consumer (`render_play.py`) cut over to
the source modules.

The `reglog_helpers` module itself was then dissolved (it had become a
grab-bag): the reg matchers moved to `reg_match`, `tighten_persist_dtypes`
to `utils` (beside `to_int64_arrays`), and `read_initial_irq` to
`reglogparser`. Import palette sidecar IO from `palette_io`, `wrapbits`
and dtype helpers from `utils`, reg matchers from `reg_match`, and
`read_initial_irq` from `reglogparser`.

Symbols prefixed `_` are package-internal and may change without
notice.

## License

Apache 2.0. See `LICENSE`.
