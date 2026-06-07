# MDL OPTIMAL-PARSER TRANSITION — design + audit note (UNTRACKED, working)

Working note accumulating the audit + locked design for the monolithic implementation doc that
replaces the greedy/threshold parser with the MDL optimal parser. Build + validate first (user's
choice), then the doc is written against working code. Subsume ctrl/AD/SR into MDL. Validate both
codebook scopes (per-tune vs corpus), recommend. No dead wood.

Companion: `NEXT_BUILD_additive_instrument_model.md` (the additive-model finding + defMON), the
`mdl_parse.py` prototype (HOLD/RAMP/PERIOD optimal parse), and `freq_collapse_probe.py` (the 1024-tune
collapse evidence: 13.9x bits, median 14.5x, all axes).

## The constrained-decode constraint SHAPES the token format (the key finding)

`constrained_decode.py` is a per-step logit mask applied during autoregressive generation. It is
**registry-driven** by `macros/op_contracts.py` — the single source of truth the mask, validators and
precompute arrays all dispatch on. `missing_contracts()` is a completeness test that goes RED if any
emittable op lacks a contract. So the new tokens must register there; if they do, the machinery
enforces their validity for free. Three validity classes it enforces:

1. **Codebook DEF->REF liveness** (`_apply_codebook_mask`, `_update_codebook`, constrained_decode.py:901-938).
   Driven by `CODEBOOK_TABLES` + `CODEBOOK_SPECS` (op_contracts.py:225-247, sourced from
   `codebook.codebook_spec_tuples()`). A REF whose id is not yet DEF'd-and-committed is masked. State
   tracks `codebook_live` / `codebook_pending_def` per table. **This already does exactly the streaming
   dictionary we need.**
2. **Multi-row slot sequencing** (`PendingSlot` state machine + `STRUCTURAL_SUBREGS`,
   op_contracts.py:172-216, constrained_decode.py:716-787). A multi-subreg op must emit its subregs in
   order and complete. GEN_TABLE_REF is already multi-row (ID, BASE_NOTE, RESID, LEN subregs) handled as
   `CODEBOOK_REF`.
3. **Frame-budget / timing** (`_walk_frame_aggregates`, frame_advance + cycle-charge, _MIN_DIFF). Any
   token emitting per-frame writes must carry correct frame_advance so the budget masks hold. Works as
   long as tokens decompose into the existing FRAME/DELAY + per-frame-queue model.

Both atomic (`precompute_vocab_arrays`) and Unigram sub-token (`precompute_subtoken_arrays`) modes
must be populated.

## NO RESIDUALS. The universal primitive is the driver's own: a replayed TABLE.

Residuals are forbidden — they signal a missing abstraction. The driver source is simple code: nested
accumulators and table lookups. The complete residual-free primitive basis is THREE composable things:

1. **Forward-differenced polynomial of degree N** (= constant N-th difference = N nested accumulators).
   The driver computes smooth curves with additions only via a difference table. degree 0 = HOLD,
   degree 1 = RAMP/slide (constant 1st diff), degree 2 = parabola/smooth-vibrato (`freq += vel; vel +=
   accel`, constant 2nd diff — directly observed in Luft: 2nd diff is a constant -2 with sign flips),
   degree N = deeper. This SUBSUMES HOLD/RAMP and was the missing generator behind the "literals".
2. **Table / wavetable replay** (looped period-p table) — arps, looped LFOs, the WGl/PWM tables.
3. **Additive note layer** — `freq = note_table[note] + modulation`, the two superimposed tables below.

These compose (a forward-diff curve can ride a note; a table can hold polynomial segments) and recur in
the corpus dictionary. Each "literal" I hit was one of these unmodeled — raw-domain slide, wrapping
ramp, looped table, forward-differenced vibrato. The driver never emits noise.

STATUS (validated): degree-N forward-differencing is confirmed (a 40-frame parabolic vibrato is ONE
degree-2 token; Baggis slide one degree-1). Byte-exact round-trip holds with the full HOLD/POLY(N)/
PERIOD codec (mdl_codec.py). CAVEAT corrected: an earlier "poly+table = 100% coverage" claim was a
degree-3-over-4-points OVERFIT (a cubic fits any 4 points); the honest MDL DP does not hit it and still
leaves ~30% length-1 holds when parsing the RAW per-channel freq single-pass. DIAGNOSED (Bambulino
100%, Hrabal 100%, Luft 72% of those holds): they are NOTE-JUMPS, i.e. note events that belong in the
NOTE-INDEX layer, not the freq-delta modulation layer; the rest are recurring values (dictionary). So
they are NOT residual and NOT a missing generator -- they are the two un-integrated ingredients
(two-layer note split + corpus dictionary). Path to zero is INTEGRATION, not discovery:
  1. split freq -> note_table[note-index] + freq-delta (note-jumps leave the modulation layer),
  2. parse the freq-delta layer with HOLD/POLY(N)/PERIOD (pure modulation, no jumps),
  3. corpus dictionary collapses recurring tables;
then re-measure: modulation-layer length-1 holds must go to ~0, byte-exact preserved.

VALIDATED (two-layer + degree-N, greedy re-anchor on instant jumps only): modulation-layer literals
drop from ~36% to ~1-8% on most tunes (Luft 36->8, Sweden 2->1, Heroes 10->5, Sky 9->5, Gruniozerca 0).
Two holdouts remain and were diagnosed by direct inspection -- BOTH are the un-built ingredients, NOT
residual: (a) Bambulino: all 242 e-literals are the IDENTICAL value 793 (a recurring one-frame note-on
transient) -> the corpus DICTIONARY collapses it to one DEF; (b) Hrabal reg7: a wrapping ramp
(+20788/frame mod 65536) that the greedy re-anchor SHATTERS (each wrap-delta looks like a note jump) ->
the JOINT MDL parse keeps it as one POLY(1) because that is far cheaper than 240 note-events. So
provable-zero requires the JOINT 2-D MDL parse: a single shortest-path DP over (note-index segmentation
x freq-delta gestures) sharing the corpus dictionary, where the COST decides note-event-vs-modulation
(a greedy heuristic cannot -- Hrabal proves it). The primitive basis (forward-diff POLY(N) + PERIOD +
additive note layer) is COMPLETE and byte-exact; the remaining work is this joint parser + dictionary,
both already specified above. No residuals anywhere -- confirmed three times independently.

- **Scalar channels** (PW x3, filter cutoff/res, vol, ctrl/AD/SR x3) = one value-or-delta table.
- **Freq channel** = TWO superimposed tables: a **note-index table** (integer semitone offsets selecting
  `note_table[]` via recover_table — handles arp/melody, recurs as interval patterns / zig-zag) PLUS a
  **freq-delta table** (sub-grid modulation in raw Hz — handles vibrato/slide/PWM, recurs base-invariant).
  Decode: `freq[t] = note_table[note_index[phase]] + freq_delta[phase]`. Both replayed EXACTLY.

Residual-free + collapsing rests on two ingredients, BOTH required:
1. **Base-relative anchoring** — modulation is measured from the onset/anchor (the note for freq), so a
   freq modulation table is base-invariant and recurs across notes/tunes. (Measured: Luft's len-28
   vibrato recurs 170x base-relative; 372 reusable tables over 10 tunes.)
2. **Corpus-global table dictionary** — a recurring table is ONE DEF + cheap REFs; the MDL cost prefers
   a shared table over L literals. The earlier "literals" (up to 86% of a tune) were just un-dictionaried
   recurring tables, NOT noise — they collapse once the dictionary holds arbitrary tables, exactly.

A truly-unique run is a length-1-table-per-frame (the literal fallback) and is RARE (driver is simple).
The note-index domain absorbs the arp/legato spans that don't recur in the freq-delta domain. No bytes
are ever approximated: a table is the exact content; the only "exactness mechanism" is the table itself.

### Consequence: the table dictionary IS a codebook family

The new MDL tokens express as the EXISTING codebook-family shape (DEF / STEP / END / REF), which
`op_contracts` + `constrained_decode` already enforce. Concretely:

- **One table dictionary** holding reusable TABLES (the wavetables above): a freq-delta table, a
  note-index/interval table, or a scalar value/delta table; the ctrl/AD/SR **program** is just a scalar
  table type (subsuming InstrumentProgramPass — its `"instrument"` table folds in; CODEBOOK_TABLES
  already supports N tables). A constant / constant-delta / looping table is the same machinery.
- **DEF/STEP/END** serialize a table into the dictionary (define-on-first, corpus-global = learned
  preamble), **REF** replays a table by id with per-instance ANCHOR fields on subregs: the anchor
  (note-index for freq, base value for scalars) and the replay length. Mirrors GEN_TABLE_REF
  (generator_pass.py:655-700). NO residual field — the table is the exact content.
- A frame with no recurring table is a length-1 literal table (rare; driver is simple). The note-index
  table absorbs arp/legato; the freq-delta table absorbs vibrato/slide/PWM.

Byte-exactness is structural: a REF replays the dictionary table exactly and `freq = note_table[note] +
freq_delta` reconstructs the log bit-for-bit. Verified by the `register_state` round-trip oracle
(audit_primitives.py:135, parse_audit.py `_lossless_problems`).

## Where it splices into the pipeline

`RegLogParser.parse` (reglogparser.py:900) runs, in order: read -> combine 16-bit regs ->
simplify ctrl/pcm -> `_add_frame_reg` -> `_filter` -> **`InstrumentProgramPass()` + `GeneratorPass()`**
(reglogparser.py:959-964) -> consolidate/cap delay -> voice rotate -> norm order -> `run_passes` ->
post-norm passes -> `_add_voice_reg` -> yield token df.

The MDL parser REPLACES the `InstrumentProgramPass()` + `GeneratorPass()` stage with one
`MdlGesturePass` that:
1. builds per-frame settled value channels (freq x3, pw x3, cutoff, res, vol, AND ctrl/AD/SR x3),
2. runs `mdl_parse` per channel (wrap-aware for freq) with the dictionary-aware cost,
3. emits the codebook-family token rows (DEF/STEP/END for new shapes, REF for instances) via the
   existing `make_row` schema (passes_base.py:26) and `emit_recurring`/codebook id assignment
   (codebook_emit.py).

Decode adds `MdlGestureDecoder`(s) to `DECODERS` (decoders.py:400) producing per-frame
`pending_set_writes` queues drained by `state.tick_frame()` (state.py) — identical timing model to
SWEEP/GEN_TRI/INSTR_REF today.

## Retirement map (from the dependency audit — all callers known)

REPLACE/RETIRE (replacement = the MDL pass + gesture codebook):
- `generator_fit.decompose`/`fit_run`/`channels` (greedy parse) -> MDL `mdl_parse`. Keep `recon`/
  `note_of`/`zig`/`unzig`/`tune_ref` (used by decoders/codebook/role_lane). Callers: generator_pass.py,
  decoders.py, codebook.py, role_lane.py, melody_audit.py, tests.
- generator_pass.py per-gesture emitters `_melody_rows`/`_sweep_rows`/`_tri_rows`/`_table_rows`/
  `_atom_rows` + the kind dispatch -> one uniform token emitter. Whole GeneratorPass retired.
- `pitch_grid.decompose_voice`/`reconstruct`/`pure_fraction` (cents domain — WRONG domain) -> deleted.
  Keep `recover_table`/`voice_tuning`/`note_index`/`note_freq_at`. Tests to rewrite:
  test_residual_decompose_target.py, test_pitch_grid.py (decompose_voice parts), test_tracker_pitch_recovery.py.
- `encoding_complexity.py` (uncalled in production) -> deleted; the guard becomes `struct_bits <=
  naive_bits` MDL comparison if wanted. Delete test_encoding_complexity.py.
- Flags `universal_freq`/`universal_pitch`/`melody_skeleton`/`table_resid_split` (flag_registry.py:19-25,
  generator_pass gating) + the unused `freq_trajectory_pass` -> removed. Update tokenizer_config.py
  REGISTERED_MACROS, flag_registry FLAG_REQUIRES, and tests (test_flag_registry, test_generator_residual_zero,
  test_melody_skeleton_emit, test_voice_lane_core, parse_probes).
- `InstrumentProgramPass` + `instrument_program` flag -> subsumed into the MDL dictionary (its
  `"instrument"` codebook table folds in). Retire the class + test_instrument_program_pass.py; rewrite
  test_codebook_machine_equivalence.py, test_parse_audit.py instrument scenarios.

KEEP: codebook machinery (codebook.py/codebook_emit.py — extend families), decoders dispatch
(decoders.py — add MDL decoders), constrained_decode.py (extend via op_contracts registry only),
op_contracts.py (add contracts for new ops — completeness test enforces), parse_audit/audit_primitives
(the byte-exact oracle), frame/timing model, loop/coarsen/transpose/dedup/subreg passes, regtokenizer
serialization, vocab_signature.

## Learnability constraints (a Llama3-arch model learns this output — research-backed)

Compression != learnability. The token stream must be LEARNABLE by a transformer, not just small.
Findings (sources below) and the guardrails they impose on the design:

1. **Don't serialize the MDL bitstream.** Near-optimal/arithmetic coding "severely degrades model
   performance" — bit-optimal codes obscure the local structure learning exploits. Target LOW
   CONDITIONAL ENTROPY (next-token ~<1 bit given context), not minimal bits. MDL parse RECOVERS the
   tables; serialize them as the regular codebook-family stream (fixed fields, predictable order). The
   stream is mostly table-REFs (low-entropy, dictionary ids) + a few anchor values — no dense
   high-entropy value blobs.
2. **Corpus-global dictionary (decisive for learnability, not just bits).**
   - Per-tune ids = singletons seen once -> under-trained "glitch tokens" -> worse perf/hallucination.
     Corpus-global shapes recur across thousands of tunes -> frequent -> well-trained.
   - DEF->REF over distance is the induction-head copy problem (degrades with distance). A global
     dictionary makes a REF id a LEARNED VOCAB EMBEDDING (known like a word), not an in-context copy
     from a distant DEF. => global codebook = a learned preamble / known vocabulary, NOT in-context
     DEFs. E2 still measures both but weights learnability; default recommendation = corpus-global.
3. **Cap/merge rare tables.** High-cardinality one-off tables are a glitch-token risk. The
   dictionary-iterate must keep the alphabet frequent-dominated: merge near-duplicate tables, else fall
   back to per-frame literal tables. No singleton table-ids in the learned vocabulary.
4. **Numerics: consistent fields, decoder does the arithmetic.** Model only selects/copies field
   values (step/len/note); decoder does exact recon/accumulate. Keep fixed byte-order fields; KEEP the
   zig-zag interval encoding (zig/unzig) — small, centered-near-zero, low-cardinality = learnable.
5. **Sequence length is a win.** MDL ~14x fewer tokens = longer effective musical context, cheaper
   attention. The Unigram/BPE layer (regtokenizer) re-compounds frequent multi-subreg REF sequences
   into single sub-tokens (compound-token benefit) — keep it; constrained_decode already handles
   sub-token mode (precompute_subtoken_arrays).

None of this conflicts with byte-exactness (a replayed table is exact content); it bounds HOW
aggressively we compress and scopes the dictionary corpus-global.

Sources: Information-Theoretic Perspective on LLM Tokenizers (arXiv:2601.09039); Unpacking
Tokenization (arXiv:2403.06265); Beyond Text Compression (arXiv:2506.03101); MidiTok / music
tokenization (arXiv:2310.17202, 2310.08497); In-context Learning & Induction Heads; retrieval-distance
(arXiv:2510.22752); number tokenization (arXiv:2410.11781, 2505.14178, 2411.02083); under-trained
tokens / Fishing for Magikarp (arXiv:2405.05417); Vocabulary Frequency Imbalance (arXiv:2508.15390).

## Open experiments (in build order)

- **E1 byte-exact round-trip**: table-parse -> codebook-family tokens (table DEF/REF, no residual) ->
  decode via register_state -> assert byte-exact on the 5 driver tunes + corpus sample. Pins the exact
  token layout. INCLUDES wiring op_contracts contracts + a constrained_decode smoke test (generate under the
  mask, assert only valid streams).
- **E3 register-ownership / ctrl-AD-SR subsumption**: prove the MDL value+program split covers 100% of
  writes byte-exactly; confirm InstrumentProgramPass can be retired.
- **E2 dictionary scope**: build dictionary-aware cost + iterate; measure per-tune vs corpus-global
  (bits, alphabet size, byte-exactness); recommend.

Acceptance for the doc: full suite green (725+ baseline) with the new pass as default, byte-exact via
parse_audit=raise on the 5 drivers + a corpus sample, constrained-decode completeness test green, and
the collapse numbers reproduced by the integrated encoder (not just the probe).
