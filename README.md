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
- `preframr_tokens.reglog_helpers` -- palette IO + dtype tightening.
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

## Stability

Library follows semver from v1.0. Pre-1.0 releases may break API as
the preframr codebase evolves. Token-alphabet shape changes bump
major version since they invalidate downstream checkpoints.

## License

Apache 2.0. See `LICENSE`.
