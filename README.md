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

## Modules

- `preframr_tokens.reglogparser` -- SID dump → parsed dataframe
  pipeline. `RegLogParser`.
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
- `preframr_tokens.reglog_helpers` -- voice-relative reg matchers, dtype
  tightening, and re-exports of `wrapbits` (now in `utils`) and the
  palette sidecar IO (now in `palette_io`) for back-compat. New consumers
  should import from the source modules.
- `preframr_tokens.palette_io` -- JSON sidecar load/dump for the
  engine-fingerprint / engine-fp-cluster `df.attrs` (extracted from
  `reglog_helpers`).
- `preframr_tokens.macros.roles` -- single source of truth for
  `(op, subreg)` → role classification. `distance_pair_role`,
  `slope_subreg_role`, `frame_weight_role`, plus the
  `DISTANCE_PAIR_OPS` table and the `DistancePairSpec` dataclass.
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

Full inventory of what the main `preframr` repo currently consumes
plus the narrowing opportunities still on the table is in
[`API_SURFACE.md`](API_SURFACE.md). Recent additions in that vein:

- `preframr_tokens.tier_classify` — `vocab_id_tier`,
  `build_vocab_tier_ids`, `build_vocab_tier_map`, `CONTENT_TIER`.
  Replaces ad-hoc reg/op tier classification in consumers.
- `preframr_tokens.token_weighting.vocab_frame_weights` — per-vocab
  audio-frame-time weighting. Replaces ad-hoc BACK_REF / DO_LOOP /
  SLOPE / DELAY / FRAME val accounting in consumers.
- `preframr_tokens.vocab_signature.VocabSignature` — single-pass
  bundle of both of the above. Consumers that need both `tier_ids`
  and `frame_weights` should construct this directly.
- `preframr_tokens.reglog_helpers.read_initial_irq` — first-frame
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

## Stability

Library follows semver from v1.0. Pre-1.0 releases may break API as
the preframr codebase evolves. Token-alphabet shape changes bump
major version since they invalidate downstream checkpoints.

Public surface (semver-promised once v1.0):

- **Classes**: `RegLogParser`, `RegTokenizer`, `Corpus`, `TokenizeMeta`,
  `StreamState`, `PendingSlot`, `VocabSignature`, `Transform`
  (+ `register` decorator, `PipelineEntry`, `TransformPipeline`),
  `DistancePairSpec`.
- **Decision helpers**: see `API_SURFACE.md` "Decision helpers"
  section.
- **Routines**: `parse_corpus`, `precompute_vocab_arrays`,
  `precompute_subtoken_arrays`, `prepare_df_for_audio`,
  `remove_voice_reg`, `validate_back_refs`,
  `validate_pattern_overlays`, `frame_marker_count`,
  `ensure_default_transforms_registered`, `distance_pair_role`,
  `slope_subreg_role`, `frame_weight_role`.
- **Boundary constants**: `PAD_ID`, `MODEL_PDTYPE`, `DUMP_SUFFIX`,
  `LEGACY_EVAL_SUBSET_NAME`, `DEFAULT_IRQ_CYCLES`, `LOSS_TIER_NAMES`,
  `DISTANCE_PAIR_OPS`, `CONTENT_TIER`.

### Back-compat aliases scheduled to drop

The internal aliases (`_frame_marker_count`, `_compute_invalid`,
`_LOSS_TIER_NAMES`) and the public `MIN_DIFF` re-export have been
removed this round; consumers have cut over to the public names.
What remains is the `reglog_helpers` re-export set, which is blocked
on a main-repo cutover of `render_play.py`.

| alias | replacement | location |
|---|---|---|
| `reglog_helpers.dump_palettes_attrs` | `palette_io.dump_palettes_attrs` | `preframr_tokens/reglog_helpers.py` |
| `reglog_helpers.load_palettes_attrs` | `palette_io.load_palettes_attrs` | `preframr_tokens/reglog_helpers.py` |
| `reglog_helpers.wrapbits` | `utils.wrapbits` | `preframr_tokens/reglog_helpers.py` |

Symbols prefixed `_` are package-internal and may change without
notice (current consumers that reach into them are tracked in
`API_SURFACE.md` as "leaks to clean up").

## License

Apache 2.0. See `LICENSE`.
