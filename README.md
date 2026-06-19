# preframr-tokens

SID reglog parsing, tokenization, and macro transforms extracted from
the [preframr](https://github.com/anarkiwi/preframr) research codebase.

Torch-free. The training-side concerns (model, loss, DataLoader,
predict) live in the main `preframr` repo; this package contains the
stable parsing + encoding layer that produces the parsed parquets +
the token alphabet that downstream training consumes.

This README is the API reference for the package, including the
[input dump format](#the-input-dump-format), the
[inline-event token alphabet](#the-token-alphabet-inline-event-model) and its
[fidelity contract](#fidelity-contract), and the
[parse-domain output schema](#parse-domain-reglogparser). The
SID-chip behavior facts these encodings rest on are documented (and
unit-tested) in the
[preframr-audio README](https://github.com/anarkiwi/preframr-audio).

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

`preframr_tokens.__all__` is the semver-promised surface. Three submodules
are also public, stable namespaces you import directly:
`preframr_tokens.stfconstants` (reg ids, op codes, dtypes, PAL clock),
`preframr_tokens.engine_fingerprint` (feature-vector layout, `ClusterTable`,
`compute_fingerprint`), and `preframr_tokens.events` (the inline-event codec:
`events.inline`, `events.stream`, `events.oracle`, `events.pipeline`,
`events.dataset`, `events.generate`). Every other `preframr_tokens.*`
submodule path is **internal and may move between releases** — depend on
the root re-exports instead. The module list below documents that
internal structure.

## The input dump format

A raw tune is a `.dump.parquet` (`DUMP_SUFFIX`) of register writes
captured from a SID player:

| Column | Meaning |
|---|---|
| `clock` | absolute PAL φ2 clock cycle of the write |
| `irq` | IRQ counter; each unique value is one player frame (~19656 cycles, `DEFAULT_IRQ_CYCLES`, ≈50.1 Hz) |
| `chipno` | SID chip number — v1 scope is single-SID, `chipno != 0` is dropped |
| `reg` | register 0..24 |
| `val` | byte written |

Register map (`VOICE_REG_SIZE = 7`, base = `voice * 7`): +0/+1 freq
lo/hi, +2/+3 pulse-width lo/hi, +4 control (bit0 GATE, bit3 TEST,
bits4–7 waveform), +5 AD, +6 SR. Globals: 21/22 filter cutoff lo/hi,
23 resonance/routing, 24 mode/volume. A 16-bit frequency is always
`(hi << 8) | lo` — the parser settles lo/hi pairs (`combine_reg`)
so a half-updated pair is never read.

Scope: single-speed (one player call per IRQ frame; `events.stream.single_speed`),
non-digi (`dump_meta.is_digi`). Multi-speed (~5%) and digi (~3%) tunes
are rejected up front; an out-of-scope failure is a scope bug, not a
fidelity bug.

## The token alphabet (inline-event model)

The current tokenizer is the inline-event codec in `preframr_tokens.events`
(`inline.py` + `stream.py`). It is a fixed `stream.VOCAB_SIZE`-atom alphabet
(55 atoms); BPE over these atoms is the dictionary. There are no ids, no
literals table, no frozen-table DEF/REF, and no escape op.

Two fidelity classes merge into ONE stream. The **10 settled non-env lanes**
(freq16 × 3, pw12 × 3, filter cutoff-lo, cutoff-hi, resonance, mode-volume) are
taken from the settled per-frame `(n_frames, 25)` register grid and each encoded
with two ops over its running value; **ctrl / AD / SR** (the 9 env regs
4,5,6,11,12,13,18,19,20) are NOT settled — they are kept as the ORDERED write
stream so the audibly-significant envelope / hard-restart / gate order survives.

- **freq lanes** use `NOTE(interval)` (an inline relative pitch step, signed) and
  `MOD(deltas, n)` (an `n`-frame periodic delta run — vibrato / glide), both
  relative to the running freq so the lane is transposition-invariant.
- **every other non-env lane** uses `LOAD(value)` (a jump to a value — a note /
  wavetable / table entry) and `RUN(deltas, n)` (an `n`-frame periodic delta run —
  sweep / sustain / PWM).
- **env regs** emit a `WRITE(value)` event per source write (consecutive
  same-reg-same-val no-ops de-duped), preserving intra-frame write order.

All selectors merge into ONE time-ordered event stream `(start_frame, sub, payload)`
with implicit non-env holds dropped, sorted by `(start_frame, sub)`; within a frame
the non-env lane events come first (by lane) then the env writes in source order.
The stream is then a flat atom-id list. A non-env event is `[DT][LANE][OP][params]`;
an env event is `[DT][SELECTOR][value]` (no OP byte):

| Range | Tokens | Meaning |
|---|---|---|
| `LANE_BASE` 0–9 | 10 | non-env lane ids (freq × 3, pw × 3, cutoff-lo, cutoff-hi, res, vol) |
| `LANE_BASE` 10–18 | 9 | env reg selectors (ctrl/ad/sr × 3 voices); selector ≥ 10 is a `WRITE` |
| `OP_BASE` 19–22 | 4 | op: `NOTE`, `LOAD`, `MOD`, `RUN` |
| `DIGIT_BASE` 23–54 | 32 | self-delimiting base-16 LEB digits: low 16 = continue, high 16 = terminal; signed values are zig-zagged |

`DT` is the unsigned inter-event frame delta (a digit run). A non-env event carries
a `LANE` atom, an `OP` atom, then the op's params as digit runs (`NOTE` = one signed
value, `LOAD` = one unsigned value, `MOD`/`RUN` = unsigned period `p`, unsigned
length `n`, then `p` signed deltas). An env event carries an env selector then one
unsigned value. Every field family owns a disjoint position, so the stream is
self-delimiting with no separator or escape token.

**Inline streaming.** There is no SET op, no preamble, no forward declaration,
and no frozen table. Any prefix of the token stream (cut at an event boundary,
`stream.unit_starts`) is itself a valid, decodable, continuable song. Reuse is
backward-looking only (BPE over the atoms); the model emits new
pitches / ornaments / instruments inline at any time.

`is_content_atom(tok)` splits the alphabet into loss tiers: content = the varint
digits (the payload the model must predict — intervals, deltas, durations,
values); structural = the lane and op atoms.

### Dictionary segmentation

The unigram BPE that sits on top of the atom alphabet is a *dictionary*, and it
trains over **grammar-unit words**. The `.uni` training text is segmented at
every event start the codec emits (`stream.unit_starts` / `dataset.unit_starts`),
one whitespace-delimited word per event. The unigram pre-tokenizer splits on that
whitespace first, so **no learned piece can span an event boundary**. Runtime
`encode` is unchanged: real streams carry no spaces and the `WhitespaceSplit`
is a no-op there.

### Fidelity contract

The codec target is the **audio-faithful** `stream.canonical_writes(ow)` =
`oracle.corrected_writes(ow)`: per frame the settled NON-env register changes (in
ascending register order) interleaved with the ORDERED ctrl/AD/SR writes in their
source order. Only intra-frame non-env intermediates and env same-value rewrites
drop (both inaudible) — the env write ORDER is preserved, so a within-frame gate
toggle or hard-restart sequence round-trips write-for-write. This is NOT
byte-exact-to-settled-state (that would erase the load-bearing env order).
"Lossless" means `decode(encode(ow))` reproduces `corrected_writes(ow)`, exactly,
on every tune (100% of the HVSC music corpus).

`encode(ow, verify=True)` self-verifies the round trip against the corrected target
on every call and raises loudly on a miss; `roundtrip_ok(df)` is the one-call smoke
test. The entry points:

```python
from preframr_tokens.events import oracle, stream

ow = oracle.ordered_writes(dump_df)   # byte-exact ordered writes (clock-sorted, chipno 0)
tokens = stream.encode(ow)            # verified against corrected_writes (settled non-env + ordered env)
writes = stream.decode(tokens)        # [(frame, reg, val), ...]
```

`events.pipeline` / `events.dataset` build self-contained event-token blocks for
training (`Corpus.preload` drives them); `events.generate` decodes generated
token ids back to ordered writes and a render-ready dump DataFrame.


## Parse-domain (RegLogParser)

The pre-events parse pipeline is still the substrate for the macro
passes, audits and the constrained-decode mask. `RegLogParser(args)`
is constructed from an `argparse.Namespace`
(`tokenizer_config.default_tokenizer_args()` /
`named_config("full_macros")` provide the presets) and
`parse(name, max_perm=99, require_pq=False, reparse=False)` yields one
parsed DataFrame per voice rotation:

| Column | Meaning |
|---|---|
| `reg` | register id, plus marker registers below |
| `val` | value (post combine/quantize) |
| `diff` | clock delta (frame period on FRAME rows) |
| `op` | op code (`SET_OP = 0` for literal writes; macros emit their own) |
| `subreg` | sub-register index for multi-row macro atoms (−1 unused) |
| `irq` | frame period in cycles |

Marker registers (`stfconstants`): `FRAME_REG = -128` (frame boundary;
`val` packs the per-frame voice order base-4, see `remove_voice_reg` /
`VALID_VOICEORDERS`), `DELAY_REG = -127` (multi-frame gap, `val` =
frames), `VOICE_REG = -126` (voice delimiter, `val` = 0 in any trained
stream), `PAD_REG = -1`. `prepare_df_for_audio` converts a parsed df
back to the literal-write + marker form the
[preframr-audio](https://github.com/anarkiwi/preframr-audio) renderer
consumes.

## Modules

- `preframr_tokens.events` -- the inline-event codec (see
  [The token alphabet](#the-token-alphabet-inline-event-model)):
  `inline` (lane split, freq NOTE/MOD + generic LOAD/RUN + ordered env WRITE
  encode/decode, event merge, flat-atom serialization), `stream` (alphabet,
  `encode`/`decode`, `canonical_writes`, `unit_starts`, `roundtrip_ok`,
  `single_speed`, `is_content_atom`), `oracle` (`OrderedWrites`, `ordered_writes`,
  `settled_grid`, `env_writes`, `corrected_writes`), `pipeline` / `dataset` (frame-window blocking + training
  arrays), `generate` (token ids → ordered writes), `constrained` (per-step
  grammar-validity mask for sampling over the inline-event alphabet).
- `preframr_tokens.reglogparser` -- SID dump → parsed dataframe
  pipeline. `RegLogParser`, plus `read_initial_irq` (first-frame IRQ
  read off a parser-output df, with PAL default).
- `preframr_tokens.regtokenizer` -- alphabet build + unigram tokenizer
  fit. `RegTokenizer`.
- `preframr_tokens.bpe_audit` -- merge-table boundary audit
  (lane-crossing + multi-op-kind merges; run after any unigram train).
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
  `frame_weight_role`, plus the `DISTANCE_PAIR_OPS` table and the
  `DistancePairSpec` dataclass. The parse-domain reg-id counterpart is
  `reg_match`.
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
  and `DistancePairSpec`.
- **Decision helpers**: the `tier_classify` (`vocab_id_tier`,
  `build_vocab_tier_ids`, `build_vocab_tier_map`) / `token_weighting`
  (`vocab_frame_weights`) / `VocabSignature` / `read_initial_irq` /
  `reg_class` / `to_int64_arrays` family catalogued under "API surface".
- **Routines**: `parse_corpus`, `precompute_vocab_arrays`,
  `precompute_subtoken_arrays`, `prepare_df_for_audio`,
  `remove_voice_reg`, `validate_back_refs`, `validate_pattern_overlays`,
  `frame_marker_count`, `tail_charge_for_prompt`,
  `ensure_default_transforms_registered`, `get_transform_class`,
  `distance_pair_role`, `frame_weight_role`,
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

## License

Apache 2.0. See `LICENSE`.
