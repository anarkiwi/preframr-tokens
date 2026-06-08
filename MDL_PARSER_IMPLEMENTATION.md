# MDL OPTIMAL-PARSER — monolithic implementation spec (execute end-to-end)

This document replaces the current greedy/threshold register-log parser with one **MDL optimal parser**
over the driver's own primitive basis, and **retires all code the new system does not need**. It
subsumes the prior `NEXT_BUILD_additive_instrument_model.md` (Part 1/Part 2) and the working note
`MDL_TRANSITION_DESIGN.md`. Everything needed to build it is here; no further research is required.

Author it against the validated throwaway prototypes in the repo root — they are the reference
implementations for the generator math and the byte-exact decode, already proven on real data:
- `mdl_parse.py` — the 1-D optimal parse (HOLD / POLY(N) forward-difference / PERIOD) via shortest path.
- `mdl_codec.py` — the byte-exact self-contained encode/decode (forward differencing, period replay,
  16-bit wrap). Proven byte-exact on the 5 driver tunes + 40 random corpus tunes (6 .. 220,971 frames).
- `freq_collapse_probe.py`, `freq_residual_instrument_spike.py` — the collapse measurements and
  per-frame register-state reconstruction.

Delete all four prototypes plus `NEXT_BUILD_additive_instrument_model.md` and
`MDL_TRANSITION_DESIGN.md` once this lands. The current parser is unlearnable and broken; keep no dead
wood.

---

## 0. Why, in one paragraph

The driver source (e.g. defMON `pydefmon`, Daglish/Hubbard register logs) shows the SID logs are emitted
by **simple code: nested accumulators + table lookups, on top of a per-note frequency LUT**. The
current parser approximates that with ad-hoc, threshold-driven, per-gesture passes that mint per-tune,
high-entropy, unlearnable tokens, and it carries **residuals** — which are forbidden, because a residual
means a missing abstraction. The new system recovers the driver's actual program by **optimal MDL
parsing over the exact primitive basis**, producing a **corpus-global dictionary of reusable gestures**
that is byte-exact, compresses ~10-14×, and is shaped for a Llama3-class transformer to learn.

---

## 1. The model (residual-free; this is the whole abstraction)

Recovering the gestures is the **optimal-parsing problem** (LZ-optimal parse / Knuth-Plass /
Bellman segmented-regression / Viterbi — one DP): build a cost DAG over frame positions, edge `i->j` =
"encode `s[i:j)` as generator g" weighted by its **description length in bits**; the shortest path
`0->n` is the globally optimal parse. There are **no residuals** — every frame is exactly the output of
one of three composable primitives, the driver's own:

1. **HOLD** — constant value (degree-0).
2. **POLY(N)** — forward-differenced polynomial, degree `N` = constant N-th difference = `N` nested
   accumulators. The driver computes smooth curves with additions only via a difference table.
   `N=1` = RAMP/slide (constant 1st diff), `N=2` = parabola / smooth vibrato (`f += v; v += a`, constant
   2nd diff), up to `MAXDEG=3`. **This subsumes HOLD/RAMP** and was the historically missing generator.
   Reference: `mdl_parse._poly_runs`, `mdl_codec._difftable` (encode) / `decode` "D" branch (forward
   differencing, 16-bit wrap on the value level only). Validated: a 40-frame parabolic vibrato is ONE
   degree-2 token; a `+20788/frame mod 65536` sweep is ONE wrapping degree-1 token.
3. **PERIOD(cell)** — looped delta-cell table (arp, looped LFO, PWM/WGl waveform table). Reference:
   `mdl_parse._period_edges`, `mdl_codec.decode` "P" branch.

### 1.1 The additive two-layer freq decomposition

The frequency channel is **two superimposed tables**, exactly as the driver computes it:

```
freq[t] = note_table[ note_index[t] ] + freq_delta[t]
```

- `note_table` = the per-tune note-frequency LUT, recovered EXACTLY (see §4.3). 81% of voiced frames
  sit exactly on it (`freq_delta = 0`), measured corpus-wide.
- `note_index[t]` = the **note-index layer**: piecewise-constant, changing only at **note events**
  (gate-on, arp step, legato note change). This is the melody/arp — musical content, low cardinality
  (~15 notes/voice). Encoded as its own token stream of intervals (zig-zag, see §3, §7).
- `freq_delta[t]` = the **freq-delta layer**: pure modulation (vibrato/slide/PWM), a small reusable
  table set. Sub-grid alphabet is tiny corpus-wide (top-16 values cover 87%; the rest are sample points
  of POLY/PERIOD gestures, not noise). Encoded as HOLD/POLY/PERIOD gestures.

The scalar channels (PW×3, filter cutoff+res, vol, and the ctrl/AD/SR program ×3) have **no note
layer** — each is one value/delta channel parsed directly into HOLD/POLY/PERIOD.

### 1.2 The joint 2-D parse — the one piece not yet prototyped

The note-event segmentation **cannot be done by a frame-by-frame heuristic** (proven: a wrapping ramp's
huge per-frame deltas are indistinguishable from arp note-jumps by any local rule — Hrabal). It MUST be
the **joint shortest-path DP** that decides note-event-vs-modulation **by cost**:

- DP state: `(frame i, current note-index m)`.
- Edge "modulation gesture": from `(i, m)` to `(j, m)` = a HOLD/POLY/PERIOD gesture over
  `freq[i:j] - note_table[m]`, cost = that gesture's dictionary-aware bits (§4.4).
- Edge "note event": from `(i, m)` to `(i, m')` for a candidate grid note `m'` near `freq[i]` (nearest,
  ±1, ±2 — small fan-out), cost = the note-index-layer token cost for the interval `m' - m` (zig-zag,
  dictionary-aware).
- Minimize total bits; shortest path gives the segmentation, the note-index stream, and the freq-delta
  gestures **simultaneously and optimally**. Greedy is forbidden — only the global cost keeps Hrabal's
  wrapping ramp as one POLY(1) (cheaper than 240 note-events) and lets Bambulino's recurring 793
  transient collapse to one dictionary REF.

Because the value at frame `i` is `note_table[m] + freq_delta[i]` and the gesture decode is exact,
**every (i,m) decomposition is byte-exact**; the DP only chooses the cheapest. Bound segment length and
note fan-out so the DP is `O(n · notes · maxlen)`; reuse the incremental run precompute from
`mdl_parse` (`_poly_runs`, `_period_edges`) per candidate anchor.

---

## 2. The corpus-global dictionary (decisive for learnability)

Gestures are interned in ONE **corpus-global dictionary** — the structural alphabet of the drivers —
NOT per tune. This is required for learnability, not just compression (research-backed, §7):

- **Two-pass / iterate (Zopfli-style):** (pass 1) parse every corpus tune with an empty/seed dictionary,
  accumulating gesture-shape frequencies; (pass 2) rebuild the dictionary as the shapes used ≥2× with
  bit-costs reflecting frequency, re-parse; iterate until description length converges.
- A gesture **shape** is the reusable key: POLY `(degree, N-th difference)`; PERIOD `(delta cell)`;
  ctrl/AD/SR `(per-frame field program)`. The per-instance anchor (`note_index`/base value, length,
  and POLY lower-order initial diffs) rides on the REF, never in the dictionary.
- **Validate both scopes, recommend corpus-global** (the user asked for the measurement): build the
  dictionary per-tune vs corpus-global, report bits, alphabet size, and the rare-shape tail; expect
  corpus-global to win on learnability (per-tune ids are singletons = under-trained glitch tokens).
- **Cap/merge rare shapes** so the learned vocabulary stays frequent-dominated: merge near-duplicate
  cells/coefficients; a genuinely one-off run is a length-1 HOLD (rare; the driver is simple). No
  singleton shape-ids.

---

## 3. Token format — a codebook family (so constrained decode works unchanged)

The new tokens are a **codebook family** in the EXISTING machinery (`macros/codebook.py`
`CODEBOOK_FAMILIES`, `op_contracts.py` `CODEBOOK_SPECS`/`CODEBOOK_TABLES`), so `constrained_decode.py`
enforces their validity with NO new mask logic — it is registry-driven and has a completeness test
(`op_contracts.missing_contracts`) that fails red if any emittable op lacks a contract.

Add ONE unified gesture codebook table (call it `"gesture"`) that **replaces both** the current
`"generator"` and `"instrument"` tables (ctrl/AD/SR programs are just scalar gesture shapes — this is
how `InstrumentProgramPass` is subsumed; `CODEBOOK_TABLES` already supports N tables).

New ops (assign integer values in `stfconstants.py`, following the existing block at lines 28-39;
register each in `op_contracts._CONTRACT_LIST` with the matching `MaskRole`):

| Op | MaskRole | Role |
|---|---|---|
| `GESTURE_DEF_OP` | `CODEBOOK_DEF` | open a dictionary shape (define-on-first; corpus-global = preamble) |
| `GESTURE_STEP_OP` | `CODEBOOK_STEP` | shape body: POLY `(degree, N-th diff)` / PERIOD `(p, cell deltas)` / program fields |
| `GESTURE_END_OP` | `CODEBOOK_END` | commit the shape id |
| `GESTURE_REF_OP` | `CODEBOOK_REF` | replay shape `id` with per-instance anchor + length |
| `NOTE_INTERVAL_OP` | `ATOM` | note-index-layer event: zig-zag interval to the next note (melody/arp) |

`GESTURE_REF_OP` subregs (each a fixed, consistent field — see learnability §7): `ID` (dictionary id),
`ANCHOR_LO/HI` (note-index for freq, base value for scalars), `LEN_LO/HI`, and for POLY the lower-order
initial diffs `D1..D{N-1}` (the N-th diff is the shape). Mirror `GEN_TABLE_REF` exactly
(`generator_pass.py:655-700`) for the row construction and the `codebook._GeneratorCodec` replay; there
is **no residual subreg**. Register the family in `codebook.CODEBOOK_FAMILIES` so `codebook_spec_tuples`
feeds `CODEBOOK_SPECS`; then `constrained_decode` masks a REF whose id is not yet live, sequences the
multi-subreg REF, and frame-budgets it, all for free. Populate both `precompute_vocab_arrays` (atomic)
and `precompute_subtoken_arrays` (Unigram) — they read the registry, so no per-op code is needed beyond
the contract.

Loss tiers: add the gesture ops to `op_contracts.MACRO_OP_LOSS_TIERS` — DEF/END/REF = `structural`,
STEP body + NOTE_INTERVAL = `content`.

---

## 4. The encoder pass — `MdlGesturePass`

Replaces `InstrumentProgramPass()` + `GeneratorPass()` at `reglogparser.py:959-964` with ONE pass.

### 4.1 Splice point
In `RegLogParser.parse` (`reglogparser.py:900`), the freq-block stage (lines 959-964) currently runs
`InstrumentProgramPass()` then `GeneratorPass()`. Replace that loop with a single `MdlGesturePass()`.
Keep the `assert_elapsed_frames` frame-conservation guard. Everything before (`_combine_regs`,
`_simplify_ctrl/pcm`, `_add_frame_reg`, `_filter`, `_squeeze_frame_regs`) and after
(`_consolidate_frames`, `_cap_delay`, voice rotation, `_norm_pr_order`, `run_passes`,
post-norm passes, `_add_voice_reg`) is unchanged. Note: with `generator_pass` gone, the
`_quantize_freq_to_cents` skip condition at `reglogparser.py:942-943` becomes unconditional-skip — the
MDL pass owns raw freq; delete the cents quantization path.

### 4.2 Input
Build the per-frame settled register state `(n_frames, 25)` for the tune (the same reduction
`audit_primitives.register_state` computes; `freq_residual_instrument_spike.per_frame_state` is the
reference). Partition the 25 regs into channels exactly as `mdl_codec` does:
freq words `(0,1)(7,8)(14,15)` wrap-16; PW `(2,3)(9,10)(16,17)`; cutoff `(21,22)`; singles
`4,5,6,11,12,13,18,19,20,23,24` (ctrl/AD/SR ×3, res, vol).

### 4.3 Recover the exact note table
For each freq voice recover the per-tune LUT (`pitch_grid.recover_table` / `voice_tuning` /
`note_index` — KEEP these). It is exact for 81% of voiced frames by construction; the rest is modulation
in the freq-delta layer. (Do NOT use `pitch_grid.decompose_voice` — cents domain, wrong, deleted in §8.)

### 4.4 Parse
- **Freq voices:** the joint 2-D MDL parse of §1.2 (note-index layer + freq-delta gestures), sharing the
  corpus dictionary. Emit `NOTE_INTERVAL_OP` per note event and `GESTURE_REF_OP` per freq-delta gesture.
- **Scalar channels:** the 1-D MDL parse (`mdl_parse`) directly into HOLD/POLY/PERIOD gestures; emit
  `GESTURE_REF_OP` (or a literal HOLD for a one-off). ctrl/AD/SR programs are scalar gesture shapes.
- **Cost model:** `mdl_parse.nbits` (Elias-gamma-ish; `nbits(0)=1`), `_HDR=6`. POLY cost = header +
  anchor + `sum nbits(D1..DN)` (so lower degree / longer run wins); PERIOD cost = header + anchor +
  `nbits(p)` + `sum nbits(cell)`; a dictionary REF costs `log2(|dict|)` (cheap) vs a new DEF's full
  shape bits — this is what rewards reuse and yields the collapse.
- Emit rows with `make_row` (`passes_base.py:26`) into the 7-field schema; assign dictionary ids with
  `codebook_emit.emit_recurring` (DEF-on-first, REF-after, deterministic id order). Decode-order
  invariant: a DEF strictly precedes its REFs (the corpus-global preamble emits all DEFs up front).

### 4.5 Byte-exact guard
After the pass, assert `register_state(before) == register_state(after)` (the existing
`parse_audit._lossless_problems` oracle, `PREFRAMR_PARSE_AUDIT=raise`). Byte-exactness is structural
(every token replays exactly), so unlike `InstrumentProgramPass._instr_is_lossless` there is **no
fall-back-to-literal-stream guard** — divergence is a bug, fail loudly.

---

## 5. The decoder

Add gesture decoders to `DECODERS` (`macros/decoders.py:400`), producing per-frame writes queued into
`state.pending_set_writes[reg]` and drained by `state.tick_frame()` — identical timing model to
`SWEEP`/`GEN_TRI`/`INSTR_REF` today (frame/DELAY model unchanged, `walker.py`/`state.py`).
- `GESTURE_REF_OP` → look up the shape; POLY = forward-difference from `(anchor + initial diffs)` for
  `LEN` frames (lift `mdl_codec.decode` "D" branch, 16-bit wrap on the value level for freq); PERIOD =
  replay the delta cell (the "P" branch); for freq, add `note_table[note_index]` per frame.
- `NOTE_INTERVAL_OP` → `note_index += unzig(interval)` (or absolute on the first), set the voice's
  current note; the freq-delta REF that follows adds onto it. Reuse `MelodyIntervalDecoder`'s
  interval-sum + `recon(note, ref)` logic (`decoders.py:347-397`) as the template.
- The unified gesture `_GestureCodec` replaces `_InstrumentCodec` and `_GeneratorCodec` in
  `codebook.py`. `register_state` then reconstructs `(n_frames,25)` exactly.

---

## 6. Pipeline integration checklist
1. `stfconstants.py`: add the 5 new ops + their subregs; remove the retired ops (§8).
2. `codebook.py`: add the `"gesture"` family + `_GestureCodec`; remove `"instrument"`/`"generator"`.
3. `op_contracts.py`: add the 5 contracts + loss tiers + `CODEBOOK_TABLES = ("gesture",)`; the
   completeness test now guards them.
4. `decoders.py`: add gesture decoders; remove retired decoders.
5. `reglogparser.py`: swap the §4.1 splice; delete the cents-quantization branch.
6. `tokenizer_config.py` / `flag_registry.py`: remove the retired flags (§8); the new pass is
   unconditional (no gate flag).
7. `macros/__init__.py`: swap `InstrumentProgramPass`/`GeneratorPass` for `MdlGesturePass` in the pass
   lists.

---

## 7. Learnability guardrails (a Llama3-class model learns this output — research-backed)

Do not over-compress; target LOW CONDITIONAL ENTROPY, not minimal bits (arithmetic-coding-style packing
*degrades* models). Concretely:
1. Serialize the regular codebook-family stream (fixed fields, predictable order); the stream is mostly
   table-REFs (dictionary ids) + a few anchor values — never dense high-entropy blobs.
2. **Corpus-global dictionary** (§2): recurring shapes = frequent, well-trained tokens; a global REF id
   is a learned vocabulary embedding, eliminating the long-range in-context DEF→REF copy (induction-head
   problem). Per-tune dictionaries make singletons = under-trained glitch tokens — avoid.
3. **Cap/merge rare shapes**; no singleton shape-ids in the learned vocabulary.
4. **Consistent numeric fields; the decoder does the arithmetic.** The model only selects/copies field
   values (degree, diffs, length, interval). KEEP the zig-zag interval encoding (`generator_fit.zig`/
   `unzig`) for note intervals — small, centered-near-zero, low-cardinality = learnable.
5. Sequence-length reduction (~10-14×) is a win (longer effective musical context). The Unigram/BPE
   layer (`regtokenizer`) re-compounds frequent multi-subreg REF sequences into single sub-tokens
   (compound-token benefit) — keep it; `constrained_decode` already handles sub-token mode.

Sources (for context, not needed to build): arXiv 2601.09039, 2403.06265, 2506.03101, 2310.17202,
2310.08497, 2510.22752, 2410.11781, 2505.14178, 2405.05417, 2508.15390.

---

## 8. Retirement — remove all dead wood (exhaustive; from the caller audit)

DELETE outright (definitions + all caller sites; replacement = `MdlGesturePass` + gesture codebook):

- `macros/generator_pass.py` — the whole `GeneratorPass` and its per-gesture emitters `_atom_rows`,
  `_melody_rows`, `_sweep_rows`, `_tri_rows`, `_table_rows`, `_def_rows`, `_ref_rows`.
- `macros/generator_fit.py` — `decompose`, `fit_run`, `channels`, `gen_hold/gen_accum/gen_tri/gen_table`
  (the greedy parse). **KEEP** `recon`, `note_of`, `tune_ref`, `zig`, `unzig`, `_tri_seq` (used by the
  decoder, `codebook`, `role_lane`, `melody_audit`) — move them to a small `pitch_math.py` if you prefer
  not to keep the file.
- `macros/pitch_grid.py` — DELETE `decompose_voice`, `reconstruct`, `pure_fraction` (cents domain,
  wrong). **KEEP** `recover_table`, `voice_tuning`, `note_index`, `note_freq_at`, `q_to_tuning`.
- `macros/instrument_program_pass.py` — DELETE `InstrumentProgramPass` (subsumed by the gesture codebook
  scalar shapes).
- `encoding_complexity.py` — DELETE entirely (uncalled in production; the MDL bits comparison
  `struct_bits <= naive_bits` is the principled guard if one is wanted at all).
- Old codebook codecs in `codebook.py`: `_InstrumentCodec`, `_GeneratorCodec`, the `"instrument"` /
  `"generator"` families. Old ops in `stfconstants.py`: `SWEEP_OP`, `GEN_TRI_OP`, `GEN_TUNING_OP`,
  `GEN_TABLE_*`, `MELODY_INTERVAL_OP`, `INSTR_*` and their subregs; old decoders in `decoders.py`
  (`GenTuningDecoder`, `GenTriDecoder`, `MelodyIntervalDecoder`, the instrument/generator codec decode).
- Flags: `generator_pass`, `instrument_program`, `melody_skeleton`, `universal_pitch`,
  `universal_freq`, `table_resid_split`, and the unused `freq_trajectory_pass` — remove from
  `tokenizer_config.PARSER_DEFAULTS`/`REGISTERED_MACROS`, `flag_registry.FLAG_REQUIRES`, and
  `GeneratorPass.REQUIRES_ARGS`.

TESTS to delete or rewrite to pin the new behavior (from the audit):
- DELETE: `test_encoding_complexity.py`, `test_residual_decompose_target.py`,
  `test_instrument_program_pass.py`, `test_melody_skeleton_emit.py`.
- REWRITE: `test_pitch_grid.py` (drop `decompose_voice`/`reconstruct`/`pure_fraction` cases, keep table
  recovery), `test_tracker_pitch_recovery.py`, `test_generator_residual_zero.py`,
  `test_codebook_machine_equivalence.py`, `test_parse_audit.py` (instrument scenarios → gesture),
  `test_flag_registry.py`, `test_voice_lane_core.py`, `test_tokenizer_config.py`, `test_reglogparser.py`,
  `tests/parse_probes.py`, `tests/macros/test_pipeline_check.py`. Add new pins: a `test_mdl_gesture.py`
  (byte-exact + zero-modulation-literal on the 5 drivers), `test_gesture_codebook.py` (DEF→REF
  constrained-decode liveness), and a corpus byte-exact audit.

KEEP (extend only): `codebook.py`/`codebook_emit.py` machinery, `decoders.py` dispatch, `walker.py`/
`state.py` frame model, `parse_audit.py`/`audit_primitives.py` oracle, `constrained_decode.py`
(registry-driven; no edits beyond the new contracts), `op_contracts.py`, `regtokenizer.py`/
`vocab_signature.py` serialization, the loop/coarsen/transpose/dedup/subreg/voice-block passes.

---

## 9. Acceptance recipe (the build is done when ALL pass)
1. **Byte-exact** via `PREFRAMR_PARSE_AUDIT=raise` on the 5 driver fixtures (`tests/sid_fixtures.py`:
   commando/camerock/trap/baggis + grid_runner) AND a ≥200-tune random corpus sample
   (`/scratch/preframr/hvsc/**/*.dump.parquet`). Zero divergences.
2. **Zero modulation-layer literals**: on every tune, the freq-delta layer and scalar channels parse to
   HOLD/POLY/PERIOD/REF with no one-off length-1 holds except a logged, near-zero tail; note-events live
   in the note-index layer, recurring shapes in the dictionary. (This is the joint-parse + dictionary
   target; the greedy two-layer already reaches ~1-8% — the joint DP + dictionary must reach ~0.)
3. **Collapse**: reproduce the prototype's corpus collapse from the INTEGRATED encoder (≈10-14× bits;
   `freq_collapse_probe` is the reference metric), and report the corpus dictionary size + rare-shape
   tail for both per-tune and corpus-global scopes, with a recommendation (expected: corpus-global).
4. **Constrained-decode completeness**: `op_contracts.missing_contracts()` empty; a generation smoke
   test under `StreamState.mask_logits` emits only decodable streams (no REF before its live DEF, no
   broken multi-subreg REF), atomic and Unigram modes.
5. **Full suite green** (725+ baseline) with the rewritten tests; `xdist` clean.

---

## 10. Build order
1. Land `stfconstants` ops/subregs + `op_contracts` contracts + `codebook` `"gesture"` family
   (completeness test green, no behavior yet).
2. Port the validated 1-D parse + byte-exact codec from `mdl_parse.py`/`mdl_codec.py` into the pass +
   `_GestureCodec` decoder; wire `MdlGesturePass` for SCALAR channels only; byte-exact audit.
3. Add the note-table recovery + the **joint 2-D freq parse** (§1.2) — the core new algorithm; byte-exact
   audit; drive modulation-layer literals to ~0.
4. Add the **corpus-global dictionary iterate** (§2); measure both scopes; recommend.
5. Execute the retirement (§8); rewrite/add tests; full suite green; acceptance recipe (§9).

---

## 11. EXECUTION STATUS / HANDOFF (in-progress, written for a /clear context reset)

Branch: `feat/mdl-optimal-parser` (off `main`). Steps 1-4 are **committed and were green**; Step 5
(production swap + retirement) is **in the working tree, UNCOMMITTED, and the suite is RED** — there is
one remaining correctness bug (below) plus residual test rewrites. Re-read this whole §11 before resuming.

### What is committed (green at the time of commit)
- `30c0ed6` Step 1 — gesture codebook scaffold (ops/contracts/family/`_GestureCodec`), no behavior change.
- `4132dda` Step 2 — `mdl_core.py` (ported optimal parse) + scalar-channel gesture encoder, byte-exact.
- `6280734` Step 3 — two-layer freq parse (NOTE_INTERVAL note-index layer + freq-delta gestures), byte-exact.
- `d9e82d0` Step 4 — `gesture_shape_uses` + corpus dictionary measurement. RESULT (50 tunes, 144,798
  gesture instances): per-tune alphabet 2370 vs corpus-global 1973; top 5% of shapes cover 94% of uses;
  93% singletons. RECOMMENDATION: corpus-global scope + cap/merge the rare tail (frequent core is the
  learnable vocab; singletons are the glitch-token tail). This satisfies §9.3's scope deliverable.

### What is in the working tree (Step 5, uncommitted)
Production swap DONE: `reglogparser.parse` now runs `MdlGesturePass()` only (lines ~957-959), and the
`_quantize_freq_to_cents` call was deleted (raw freq is owned by the gesture pass). `MdlGesturePass.apply`
routes to `.encode`. `FREQ_BLOCK_PASSES = [MdlGesturePass()]`.

Retirement DONE (deleted): `generator_pass.py`, `instrument_program_pass.py`, `encoding_complexity.py`;
the **instrument + generator codebook families** (CODEBOOK_TABLE_NAMES/CODEBOOK_TABLES now `("gesture",)`,
`_InstrumentCodec`/`_GeneratorCodec` gone, `_CODECS` gesture-only); the greedy parse fns in
`generator_fit.py` (kept `recon`/`note_of`/`zig`/`unzig`/`_tri_seq`/`_lut`); `pitch_grid.decompose_voice`/
`reconstruct`/`pure_fraction`; the flags `generator_pass`/`instrument_program`/`melody_skeleton`/
`universal_pitch`/`universal_freq`/`table_resid_split` (removed from `FLAG_REQUIRES`, `REGISTERED_MACROS`);
state.py `instrument_table`/`_SEED_TABLE_KEYS`/`_TABLE_IDX`; macro_contracts `GeneratorPass`/
`InstrumentProgramPass` entries; op_contracts INSTR_*/GEN_TABLE_* contracts+producers+tiers.
NOTE: the dead ATOM decoders SWEEP/GEN_TRI/GEN_TUNING/MELODY_INTERVAL are KEPT (GEN_TUNING is REUSED by the
freq path to carry the note table; the others are harmless dead atoms — retire later if desired).

Tests deleted (4, per §8): test_encoding_complexity, test_residual_decompose_target,
test_instrument_program_pass, test_melody_skeleton_emit.
Tests rewritten + GREEN already: test_pitch_grid (raw-domain), test_codebook_registry/consistency/
machine_equivalence (gesture-only golden + dead-ref via GESTURE_REF), test_flag_registry, test_tokenizer_config,
test_macro_contracts, test_block_refire_contract, test_op_contracts, test_register_order_fidelity (see below),
plus test_mdl_gesture.py (new pin: core round-trip, scalar/freq byte-exact, idempotence, dictionary scope).

### KEY DESIGN DECISION made during Step 5 (user-directed): ctrl/AD/SR = a PROGRAM gesture
register_state only checks END-OF-FRAME settled state; it MISSES intra-frame write order, which is audibly
significant for ctrl/AD/SR (gate timing, hard-restart ctrl pairs). A HOLD/replay gesture re-asserts a value
EVERY frame, changing the on-change write pattern -> breaks `test_register_order_fidelity`. Leaving ctrl/AD/SR
as raw SETs was REJECTED (no literal sets allowed). SOLUTION (built, working): a new gesture shape kind
`GESTURE_KIND_PROGRAM` that stores ctrl/AD/SR as the VERBATIM on-change ordered write-series (frame-delta,
field, val per write) captured from the raw df rows, decoded at the exact frames in order via
`state.pending_program` (dict frame->writes) drained in `tick_frame`. `MdlGesturePass._program_claims` mines
it per voice (split on >60000-frame gaps). `_SCALAR_REGS = (2,9,16,21,23,24)` (PW/cutoff/res/vol only);
ctrl/AD/SR go through the program path. `test_register_order_fidelity` PASSES with zero ctrl/AD/SR literals.

### THE CURRENT BLOCKER (must fix to make the suite usable)
`test_sid_frame_diff::test_deployed_and_stack_full_tune` and `::test_every_macro_pipeline_is_frame_exact`
FAIL — root cause is PERFORMANCE driven by a CORRECTNESS bug. The arbiter `_greedy_accept`
(macros/arbiter.py) has a fast path (ONE register_state decode if ALL claims byte-exact) but falls into an
**O(claims x rows) loop** if ANY claim diverges. On grid_runner (full = 15,685 frames) a gesture claim
diverges past ~frame 1200, so every large-tune parse hits the O(n^2) loop and effectively hangs (>120s for
3000 frames). A diverged claim ALSO means a dropped claim = a literal SET, which is forbidden. So: **every
gesture claim MUST be byte-exact** so the fast path always holds (this fixes BOTH correctness and speed).

Bugs found & FIXED so far (all in the working tree):
- `mdl_core.mdl_parse` HOLD scan was O(n^2) on constant runs -> added `_hold_runs` O(n) precompute.
- Freq-delta anchor overflow: a held note far from a fast slide/silent frame makes |delta| exceed signed-16.
  FIX: `_held_notes` now also re-anchors on a >=`_NOTE_REANCHOR`(3)-semitone jump, bounding |delta|.
- Scalar anchor was decoded with `_s16` (signed) but cutoff/PW words are UNSIGNED up to 65535. FIX: codec
  `_GestureCodec.ref`/`_replay` now store the anchor RAW and interpret it SIGNED only for freq regs (delta),
  UNSIGNED for scalar regs (value). Verified in isolation (reg21=65280 round-trips byte-exact).
- `pending_program` O(n^2) `pop(0)` -> dict keyed by absolute frame (`state.program_pos`).

REMAINING DIVERGENCE: on the REAL grid_runner slice the parse STILL hits the slow path (3000 frames times
out) even though the cutoff-HOLD case round-trips in isolation — so there is ANOTHER divergent claim (likely
a cutoff/PW POLY or PERIOD whose accumulated value exceeds 16-bit, or a freq span the re-anchor still misses,
or a PROGRAM edge case). NEXT STEP: re-run the divergence finder (it patches the arbiter to apply-ALL claims
and diffs register_state vs the decoded result, reporting (frame,reg,src,decoded)) on the smallest slice that
diverges, fix the channel, repeat until the full tune parses fast with `exact_fail==[]` `freq_fail==[]`.
Divergence-finder sketch (run as `python -c`):
  patch `arbiter.arbitrate` (and `mdl_gesture_pass.arbitrate`) to: sel=_select_nonoverlap(claims);
  src=_decoded_state(df); out=register_state(_apply(df,sel)); record where src!=out; return _apply(df,sel).
A robust belt-and-braces option: have `MdlGesturePass` itself VERIFY each claim byte-exact during encode and
re-anchor/split rather than emit a divergent claim (so the arbiter never needs the slow loop), since "no
literals" means a dropped claim is never acceptable anyway.

### Also still TODO before green
- Performance: even on the fast path, a 15,685-frame tune is slow (multiple full register_state decodes +
  per-frame `note_freq_at` in `_freq_claims`). Vectorize the freq base: replace the per-frame
  `[table.get(m) or note_freq_at(m,tuning) for m in note]` with a 256-entry note->freq LUT
  (`note_freq(np.arange(-128,128),tuning)` then overlay `table`) indexed by `note`. Confirm corpus-parsing
  tests (test_voice_lane_core parses 20 HVSC tunes) finish in reasonable wall-clock.
- Run the FULL suite (`PYTEST_WORKERS=auto python -m pytest -q -p no:cov`) and fix remaining failures.
  Known-likely: test_generator_residual_zero (DELETE or rewrite — it asserts retired GEN_TABLE/SWEEP ops),
  test_tracker_pitch_recovery (rewrite the `decompose_voice`/`reconstruct` calls to the raw note_index/
  note_freq split; both tests skip unless pydefmon/pysidwizard present), tests/parse_probes.py and
  tests/macros/test_pipeline_check.py (the `freq_trajectory_pass` kwarg + the `quantize_freq_to_cents`
  transform still in default_pipeline.py DEFAULT_PIPELINE_SPEC — decide keep-as-dead vs remove).
- `default_pipeline.py` still lists `quantize_freq_to_cents` in the transform spec (the imperative parse no
  longer calls it). Reconcile.
- Acceptance §9: byte-exact via PREFRAMR_PARSE_AUDIT=raise on the 5 drivers (PASSES today) + a >=200-tune
  corpus sample; constrained-decode completeness smoke test for the gesture family; reproduce the collapse.
- Cleanup (§ top): delete the 4 root prototypes (`mdl_parse.py`, `mdl_codec.py`, `freq_collapse_probe.py`,
  `freq_residual_instrument_spike.py`) + `NEXT_BUILD_additive_instrument_model.md` +
  `MDL_TRANSITION_DESIGN.md` + this doc, once green. (The AGENT_TASK_*.md are pre-existing, unrelated.)
- The joint 2-D MDL DP (§1.2) is NOT built — the greedy `_held_notes` re-anchor is the tractable
  approximation actually shipping; it reaches the ~1-8% literal range, not provable-zero. Optional follow-up.

### Quick orientation for the new context
- Encoder: `preframr_tokens/macros/mdl_gesture_pass.py` (`encode` -> scalar `_channel_claims`, freq
  `_freq_claims`, ctrl/AD/SR `_program_claims`; all arbitrated `validate=True`).
- Core math: `preframr_tokens/macros/mdl_core.py` (`mdl_parse`, `difftable`, `nbits`).
- Codec/decode: `preframr_tokens/macros/codebook.py` (`_GestureCodec`, `gesture_value_series`,
  `_replay_program`, `_gesture_note_base`). Decode timing: `preframr_tokens/macros/state.py` `tick_frame`.
- Ops/subregs: `preframr_tokens/stfconstants.py` (GESTURE_*, GESTURE_KIND_HOLD/POLY/PERIOD/PROGRAM,
  NOTE_INTERVAL_*).
- Reproduce the blocker: parse a >=1500-frame grid_runner slice via `tests.test_sid_frame_diff._slice_dump`
  + `RegLogParser(default_tokenizer_args(cents=50)).parse(...)`; it hangs (slow arbiter). The 300-frame
  slice is fast and `exact_fail==[]`/`freq_fail==[]`.
