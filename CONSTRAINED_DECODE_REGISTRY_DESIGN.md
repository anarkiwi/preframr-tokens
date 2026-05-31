# Constrained-decode OpContract registry — design (RESID_ZERO_PHASE3 §4 B0/B1)

**Untracked working doc** (not committed). Branch `feat/constrained-decode-registry` off `main`@v0.37.0.
Goal: collapse the **three hand-kept copies** of the op state machine into one `OP_CONTRACTS` registry
that generates decode + mask + validate + precompute, with a golden-master proving the refactor is
behaviour-preserving before any old code is deleted. Decision record per `feedback_reframe_at_impl`.

## 0. The three copies today (what we're unifying)

1. **`macros/decoders.py` `DECODERS`** — the real decode state machine: per-op `MacroDecoder.expand(row,
   state)` mutating `DecodeState`. ~24 decoders. The source of truth for *semantics*.
2. **`constrained_decode.py`** (1111 lines) — a **second** state machine for the sampling mask:
   `StreamState` (`pending_slot`/`pending_overlays`/`frame_count`/`frame_budget`/`current_sval`/
   `current_fn`/`current_dist_hi`) + `precompute_vocab_arrays` (atomic) + `precompute_subtoken_arrays`
   (Unigram, via `_classify_macro_shape`/`_walk_frame_aggregates`/`_SHAPE_HANDLERS`). Knows ONLY
   BACK_REF / PATTERN_REPLAY / PATTERN_OVERLAY (+ dead all-zero SLOPE arrays). **Blind to** STAMP/PATCH/
   SWEEP/SKEL/ORN/FREQ_TRAJ/… — the documented drift (STAMP/PATCH were added to DECODERS only).
3. **`macros/validators.py`** — a **third** walk: `validate_back_refs` (distance-pair reachability via
   `roles.DISTANCE_PAIR_OPS`) + `validate_pattern_overlays` (PR→overlay pairing).

## 1. Cross-repo API to preserve (framework calls only these — keep signatures)

- `StreamState(vocab_arrays, init_frame_count, irq, init_budget=None, init_sval=0, init_fn=0,
  remaining_steps=None, logger=None, disable_resource_masks=False)` with `.update(token_id)` and
  `.mask_logits(logits)->logits` (+ `.compute_invalid_mask()->np.bool_`).
- `precompute_vocab_arrays(tokens_df) -> VocabArrays` and
  `precompute_subtoken_arrays(tokens_df, regtokenizer, pad_id=0) -> VocabArrays`.
- `validate_stream(df)` (new single entry point) + **thin re-export shims** `validate_back_refs(df,
  prompt_frame_count=0)` / `validate_pattern_overlays(df)` so the framework keeps compiling.
- `frame_marker_count`, `tail_charge_for_prompt`, `VocabArrays`, `PendingSlot` stay exported.
Internals (`StreamState` walk, `MacroShape`/`_ShapeRule`/`_SHAPE_HANDLERS`, validators' bespoke walks)
are private and may be replaced once equivalence is green.

## 2. `OpContract` (B0) — one per op, co-located with its `MacroDecoder`

```
@dataclass(frozen=True)
class OpContract:
    op_code: int
    decoder: MacroDecoder          # the existing DECODERS[op] (decode unchanged)
    shape: Shape                   # ordered slot/value-class walk as DATA, not code
    legal_next: Callable[[AbsState], Predicate]   # which (op,subreg,value-class) may come next
    update: Callable[[AbsState, Token], None]      # advance the abstract state
```

`OP_CONTRACTS: dict[int, OpContract]` collected from the decoder modules (one registration site per
decoder — adding a decoder + contract is one edit). `shape` examples (data): `BACK_REF -> [DIST_HI,
DIST_LO, LEN]`; `PATTERN_REPLAY -> [DIST_HI, DIST_LO, LEN, OV_COUNT, then OV_COUNT×overlay-triple]`;
`STAMP_DEF -> [id] then STEP* until END`; `*_REF -> [id (+voice/base)]`; `SWEEP -> [START_HI, START_LO,
DELTA_HI, DELTA_LO, LEN]`. This replaces `MacroShape`/`_ShapeRule`/`_classify_macro_shape`.

`legal_next` rules (the masking semantics, expressed once):
- a mid-walk slot admits only its expected `(op,subreg)` (replaces `_ATOMIC_SLOT_GATE`/`_SUBTOKEN_SLOT_GATE`);
- a distance-pair `*_DIST_LO` is illegal if `full_dist > frame_count` (reach-before-frame-0);
- `PR_OV_COUNT` capped by `remaining_steps`;
- free position forbids pair-intermediates / orphan overlays / pad;
- resource masks (delay reg, `frame_budget < _MIN_DIFF`) gated by `disable_resource_masks`;
- **codebook ops (B2): a `*_REF`/`*_SET` is legal iff its id ∈ the relevant live table; a `*_DEF` is
  always legal and its `update` adds the id.** (Added in B2; B0 only needs the existing ops.)

## 3. `AbsState` (B1) — the mask-relevant projection of `DecodeState`

`AbsState` is exactly the fields the mask/validate need, advanced by the **same** `OpContract.update`
that the decoder's transitions imply (decode and mask = two views of one machine):
`frame_count, frame_budget, current_sval, current_fn, current_dist_hi, pending_slot, pending_overlays,
pending_overlay_slot, remaining_steps` + (B2) the live `stamp_table`/`patch_table`/`wavetable_table`
**id-sets** + back-ref `output_frame_count`. Document this projection so the equivalence test can assert
`AbsState` ⊆ `DecodeState`’s observable transitions (the mask⟺decode forward invariant: the mask never
forbids a token the decoder accepts; the validator rejects exactly the dangling refs the decoder would
silently drop).

`precompute_*` become **generated**: iterate `OP_CONTRACTS`, emit each op's per-vocab boolean/int
columns from its `shape` (atomic: one row per atomic id; subtoken: aggregate over the sub-token's atomic
decomposition via one generic walker that feeds assembled atomic ops into the shared `update`). The
GPU-hot arrays are derived from the registry, never hand-maintained, and covered by the equivalence test.

## 4. Migration & test order (golden-master FIRST — non-negotiable)

1. **Golden-master capture** (`tests/test_constrained_decode_golden.py` + committed fixtures under
   `tests/fixtures/constrained_decode_golden/`): on a corpus of streams (the hand vocab + constructed
   streams hitting every branch, atomic AND subtoken; plus a real parsed-tune stream when the fixture is
   available) freeze: (a) every `precompute_*` array; (b) the `compute_invalid_mask` at **every position**
   of every stream; (c) `validate_back_refs`/`validate_pattern_overlays` verdicts on valid + deliberately
   corrupted streams (truncated macro, orphan overlay, out-of-range distance). The test asserts the
   **current** code reproduces the frozen golden — a regression lock now, the equivalence oracle later.
2. **Build the registry (B0/B1) alongside** the old code, behind an internal switch; old stays default.
3. **Equivalence**: flip the generated path on inside the golden test; assert byte-identical (arrays
   element-wise, mask at every position, verdicts on valid+corrupted). Add the **mask⟺decode** forward
   invariant test (replay through `expand_ops`/`DecodeState` and `AbsState`; mask never forbids a decoder-
   accepted token; validator rejects exactly the decoder-mishandled streams).
4. **Completeness test** (the "fail at unit-test time" requirement): `set(OP_CONTRACTS) ⊇ set(DECODERS)`
   AND ⊇ every op any pass can emit (emit-set derived programmatically — a pass→ops registry or scanning
   the op constants passes reference, not a hand list). Demonstrate it bites: register a dummy op in
   DECODERS without a contract → completeness test goes red.
5. **Property tests**: random `AbsState`s — mask never permits a `*_REF`/`*_SET` to an id ∉ live table;
   `*_DEF` always permitted; after `DEF(id)`, `REF(id)` permitted; a `*_REF` to a just-rebound id
   resolves to the new def.
6. **Only when all green: delete** `validators.py` bespoke walks + `StreamState`/precompute internals,
   keeping the §1 public shims. `constrained_decode.py` collapses to registry + one generic walker +
   the Unigram sub-token assembler.

## 5. Scope / sequencing

B0/B1 cover the **current** op set (this branch is off main, pre-wavetable). B2 (codebook contracts for
STAMP_REF/STAMP_REL_REF/PATCH_SET/WAVETABLE_REF + live id-sets), B3 (materialization), B4 (validation =
legality replayed) land after B0/B1 is green and the wavetable branch merges — each forced by the
completeness test. This doc covers B0/B1 + the golden-master; B2–B4 get their own design pass.

## 6. Risk & checkpoint

Highest-risk piece in the plan: the generated `precompute_*` must be **byte-identical** to the current
hand-tuned arrays (subtle: `frame_budget` segment charging, sub-token aggregation, overlay counters,
distance reachability). The golden-master makes any divergence loud and safe (no silent behaviour change).
**Checkpoint with the author after the golden-master lands**, before the registry build + old-code
deletion — this is multi-day architectural work (`feedback_design_first`).

---

# As-built (shipped) — PR #42 `feat/constrained-decode-registry`

Status: **§4 Workstream B (B0–B4) complete.** 11 commits, `docker build` green (794 passed, coverage 89%,
pylint 10/10, pyright clean). Branch off `main`@v0.37.0. This section records what was actually built and
why it diverged from the plan above (per the design-as-hypothesis rule).

## What shipped, in order

1. **Golden-master** (`tests/test_constrained_decode_golden.py` + committed `tests/fixtures/
   constrained_decode_golden.json`) — froze the CURRENT `precompute_*` arrays, the `compute_invalid_mask`
   at every position of a corpus that was then **expanded to cover all 22 `MacroShape`s** + the pending
   dist_lo/len/ov_count/overlay chains, and validator verdicts on valid + corrupted streams. Built FIRST,
   as mandated, so every later change is proven byte-identical.
2. **`macros/op_contracts.py`** — `OP_CONTRACTS` (one `OpContract{op_code, role}` per emittable op: 34
   `DECODERS` + 4 loop ops; emit-set derived programmatically) + **completeness test that bites**
   (`tests/test_op_contracts.py`). This is the keystone: the "fail at unit-test time" guarantee that would
   have caught STAMP/PATCH being added to `DECODERS` only.
3. **Atomic mask generated from the registry, byte-identical**: enriched `STRUCTURAL_SUBREGS` with each
   slot's value-array + sub-token `consumes_gate`; `precompute_vocab_arrays` structural flags/values now
   come from `_structural_arrays()`, and the five `StreamState` slot tables (`_ATOMIC_SLOT_GATE`/
   `_SUBTOKEN_SLOT_GATE`/`_ATOMIC_SLOT_TRANSITION`/`_ATOMIC_NEW_PENDING`/overlay-gate tuples) from
   `_build_slot_tables()` iterating the registry. Dead SLOPE entries dropped (all-False flags → never
   fire; golden confirms unchanged).
4. **`validate_stream`** unified entry point (B4 API) + property + mask⟺decode forward-invariant tests.
5. **B2 codebook mask** (`CODEBOOK_SPECS` + `StreamState` live id-sets) — DEF stashes the table's pending
   id, COMMIT (`STAMP_END` / `PATCH` SR-step) makes it live, the mask **forbids any REF to a non-live id**.
6. **B3 materialization** — `expand_ops(codebook_seed=…)` (decoder snapshot-seed via `DecodeState(seed=…)`
   renders an out-of-window REF), `StreamState(init_codebook_ids=…)` (mask seed), `validate_stream(
   live_ids=…)` (validator seed), `codebook_live_ids(prior_df)` (compute what's live before a window).
7. **B4 validation** — `validate_codebook_refs`/`validate_stream` replay liveness and reject a REF to a
   non-live id (offline mirror of the B2 mask). Tests in `tests/test_codebook_mask.py`.

## Impl-time reframes / scope calls (the "why diverged")

- **The three copies are an ATOMIC-path problem.** Tracing it, the validators were ALREADY single-sourced
  via `roles.DISTANCE_PAIR_OPS`, and the Unigram **sub-token classifier** (`_classify_macro_shape` /
  `_SHAPE_HANDLERS`) is **mask-internal — not a triplication** (the decoder/`expand_loops` and the
  validators operate on ATOMIC streams; only the mask has a sub-token path). So the high-value collapse
  was the atomic `precompute` + `StreamState` slot tables (done, byte-identical), not the sub-token
  classifier. The classifier is left bespoke and golden-locked; regenerating it is table-relocation
  (bespoke `MacroShape` enums + irregularities like `BR_LEN_WITH_TAIL` and the non-chain overlay rules)
  with no dedup payoff and real risk to live-sampling code. **Deliberately not done.**
- **Sub-token-MODE codebook masking deferred.** B2/B3/B4 wire codebook liveness on the ATOMIC mask +
  decoder + validator. Sub-token-mode codebook refs are left to the decoder verify-or-RESID safety net
  (which §4 explicitly permits: "the safety net catches palette violations post-decode"). Atomic precompute
  carries the real codebook arrays; sub-token precompute carries all-none placeholders so `StreamState`
  keys always exist.
- **WAVETABLE not on this branch.** This branch is off `main` (pre-#41). STAMP/PATCH codebook contracts are
  wired; the WAVETABLE_REF contract is a one-line registry add that the **completeness test forces** once
  #41 merges — exactly the design intent.
- **Cross-repo API**: kept the §1 signatures stable; extended ONLY additively (`expand_ops(codebook_seed=)`,
  `StreamState(init_codebook_ids=)`, `validate_stream(live_ids=)`, new `codebook_live_ids`,
  `validate_codebook_refs`). The framework's existing hooks compile unchanged.

## Author-side remaining (per the work order)
PyPI tag + `preframr` bridge + 12-SID audition; the corpus per-engine `UNRESOLVED→0` re-profile that
decides whether any §3 frontier-tail primitive is needed (Workstream A2 is author-gated by design).
