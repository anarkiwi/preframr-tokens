# WAVETABLE codebook — encoding design (Phase 3 §2, full)

**Untracked working doc** (mirrors `RESID_ZERO_PHASE3.md`; not committed). Branch
`feat/wavetable-codebook` off `main` @ v0.37.0. Implements §2 of `RESID_ZERO_PHASE3.md` in full:
onset-strip + noise-inclusive detection + loop factorisation + recurring codebook + inline-structured
one-shots (§9). Decision record per `feedback_reframe_at_impl` — framings are hypotheses, updated here on shift.

## 0. Architecture (confirmed with author)

A **separate post-skeleton `MacroPass`** (`macros/wavetable_pass.py`), inserted immediately after
`SkeletonPass()` in `FREQ_BLOCK_PASSES`. It reads the skeleton's emitted **`ORN`-RESID atoms**
(op 55, `TYPE=RESID`/`P2=length`/`length`×`P1=offset`), mines a tune-global codebook of recurring
note-relative offset programs, and arbiter-splices each matching RESID atom into an inline-redefinable
`WAVETABLE_DEF`+`WAVETABLE_REF`. Mirrors `StampPass`/`StampDecoder` structurally; held-ARP generalised
from within-note period to a cross-note codebook with a loop point.

Why a separate pass works fully (not a compromise): the skeleton drops only **freq** writes, so the
surviving per-voice **ctrl** SETs are still in the df → `SkeletonPass._context(df)` recovers per-frame
waveform state for onset-strip / noise-inclusive grouping. The `SKEL` atom precedes each `ORN` and sets
`last_skel_note[reg]`; a `WAVETABLE_REF` spliced at the ORN's position decodes right after it, so the
replay base note is already set.

## 1. Correctness floor — what "byte-exact" means here (critical)

The skeleton/ORN path is **content-tier (deliberately lossy** — semitone floor; `_snap_offsets` maps
unresolvable/noise frames to whatever `fn_to_note_resid` returns, often 0). So WAVETABLE's correctness
target is **byte-identical to the ORN-RESID render it replaces** (§2: "Byte-identical to the RESID it
replaces, or fall back to RESID"), NOT to the raw register log. Decisive oracle = **isolation oracle**
(§1): `register_state(df)` with `wavetable_pass` OFF must equal `register_state(df)` with it ON.

**Consequence (the key impl-time reframe):** the replayed offsets must be the offsets **exactly as stored
in the ORN-RESID atom** — including any "garbage" semitone a noise/test frame snapped to. Therefore
ctrl-awareness (onset-strip, noise-inclusive) may shape **grouping/canonicalisation only**, never the
stored/replayed values. A per-step waveform marker is **unnecessary at the content floor** (a noise step
is already a fixed stored offset; it neither needs nor can take a different replay value). It would only
matter for a future bit-exact wavetable tier. Noise-inclusive detection is achieved by factorising the
**full** stored offset sequence (noise frames are just steps with their stored offset, never a break),
and by treating non-pitched frames as **wildcards in the grouping key** while still storing their exact
values for replay. This is logged in the handback as a deviation from §2's literal "per-step waveform
marker" wording.

The safety net (held-ARP's pattern): every claim is **verified** — `unroll(program, L) == stored_offsets`
exactly — before it is emitted; a non-verifying note is left as RESID. Mis-detection can only fail to
drain, never corrupt.

## 2. Op / subreg layout (`stfconstants.py`)

Next free op codes after `SWEEP_OP=64`:

- `WAVETABLE_DEF_OP   = 65`  — header, `reg=0`, `val=id`.
- `WAVETABLE_STEP_OP  = 66`  — one program step atom, `reg=0`.
- `WAVETABLE_END_OP   = 67`  — terminator, `reg=0`, `val=id`.
- `WAVETABLE_REF_OP   = 68`  — replay, `reg=voice freq reg`.

STEP subregs (small signed atoms, Unigram-clusterable — shared sub-sequences → shared sub-tokens):

- `WT_STEP_SUBREG_OFFSET = 0`  — note-relative semitone offset (`&0xFF`, sign-extended on decode).
- `WT_STEP_SUBREG_HOLD   = 1`  — run length of the current step (RLE; ≥1).
- `WT_STEP_SUBREG_LOOP   = 2`  — marks the loop-start **step index** (the body start); `val=index`.

REF subregs:

- `WT_REF_SUBREG_ID      = 0`  — codebook id.
- `WT_REF_SUBREG_LEN_HI  = 1`  — total frame length L hi byte.
- `WT_REF_SUBREG_LEN_LO  = 2`  — total frame length L lo byte.
- `WT_REF_SUBREG_LEAD    = 3`  — per-hit count of leading literal-offset atoms that follow (onset-strip).
- `WT_REF_SUBREG_LEADOFF = 4`  — one per-hit leading literal offset atom (`LEAD` of them).

`WT_MINREP = 2` (STAMP uses 3; held-ARP recurrence is 80–89% so 2 is justified — start here, tune later).

## 3. Program model & unroll

A program = `(prefix_steps, body_steps, body_repeatable)` where each step is `(offset, hold)`.
`prefix_steps` plays once; `body_steps` repeats. `WT_STEP_SUBREG_LOOP` carries the index where the body
begins (so prefix = steps `[0:loop)`, body = steps `[loop:]`). One-shot programs (§9) have `loop ==
len(steps)` → no body repeat, encoded inline (no codebook ref) when they do not recur.

**unroll(program, L, lead_offsets):** emit `lead_offsets` (the per-hit onset-strip head) verbatim, then
expand prefix once and body cyclically (step `(off,hold)` → `off` repeated `hold` frames), truncating at
exactly `L` total frames. Returns the per-frame offset list. Decode prepends the base note (see §5).

**Grouping key** (tune-global, transposition-invariant since offsets are note-relative):
`(tuple(prefix_steps), tuple(body_steps))` over the **pitched-core** of the stored sequence — leading
non-pitched (onset attack) frames are stripped into the per-hit `lead_offsets` (NOT in the key);
interior non-pitched frames stay in the steps with their stored offset but are marked wildcard for the
key comparison (so two hits whose only difference is interior noise garbage still group, yet each stores
its own exact value). Keys recurring `≥ WT_MINREP` notes → codebook ids in first-occurrence order.

## 4. Detector (`WavetablePass.apply`, gate `wavetable_pass`, default OFF)

1. `_ensure_subreg`; if `wavetable_pass` not set → return df unchanged.
2. `ctx = SkeletonPass._context(df)` (ctrl writes survive); `frames = _frame_index(df)`.
3. Walk rows in order; per freq reg, group each contiguous `ORN`-RESID atom → record
   `(reg, onset_fr, length L, stored_offsets[L], row_indices)`.
4. Per record: build per-frame pitched mask via `SkeletonPass._is_pitched_frame(ctx.ctrl[reg], onset_fr+1+k)`.
   Onset-strip leading non-pitched frames → `lead_offsets`; the remainder is the core.
   RLE-collapse core → steps; loop-factorise (minimal prefix p, minimal body b such that core tail is
   exactly period-b; `loop = p`). One-shot if no repeat (`body` empty / `loop=len`).
5. Canonical key from (prefix, body) with interior non-pitched offsets wildcarded. Group across all
   records. Assign codebook ids to keys with `≥ WT_MINREP` members.
6. **Verify** each member: `unroll(program, L, lead_offsets) == stored_offsets`. Drop non-verifying.
7. Emit via `arbitrate` (priority below skeleton): first occurrence of an id → `WAVETABLE_DEF id` +
   STEP rows (`OFFSET`,`HOLD` per step; one `LOOP` row) + `WAVETABLE_END id`; every occurrence →
   `WAVETABLE_REF` rows (`ID`,`LEN_HI`,`LEN_LO`,`LEAD`, `LEAD`×`LEADOFF`). Consume the ORN-RESID
   `row_indices`; keep the SKEL atom. `__pos` = the ORN atom's first row index (DEF sorts before its REF
   via the stable splice, exactly like `StampPass._emit_abs` j==0).
8. **Inline-structured one-shot (§9):** a non-recurring but loop-structured program is still emitted as a
   `WAVETABLE_DEF`+`WAVETABLE_REF` pair with a freshly-allocated id used once (structured token, no shared
   codebook), so the model predicts "structured wavetable" not a raw offset dump. Gated by the same flag.
   (A pure non-structured one-shot — no body, no recurrence — stays RESID: no regression.)

## 5. Decoder (`WavetableDecoder`, `macros/decoders.py`; `DecodeState.wavetable_table`)

Mirror `StampDecoder` + `OrnamentDecoder._queue`:

- `DEF` → `pending_wavetable_def = {id, steps:[], loop:None}`.
- `STEP` `OFFSET`→ append `(off, 1)`; `HOLD`→ set last step hold; `LOOP`→ record `loop` index.
- `END` → `wavetable_table[id] = (steps, loop)` (a later `DEF id` rebinds — streaming dict).
- `REF` → buffer `ID`/`LEN_HI`/`LEN_LO`/`LEAD`/`LEADOFF×LEAD`; on completion look up program; if absent
  return None (the §4-B out-of-window-DEF case — fixed by constrained-decode/materialization, step 2);
  else `note = last_skel_note[reg]`, `offsets = unroll(program, L, lead_offsets)`, then
  `queue.append(LUT[note]); for off: queue.append(LUT[clamp(note+off)])` into `pending_set_writes[reg]`
  — identical to `OrnamentDecoder._queue`, so byte-identical to the RESID it replaces.

`DecodeState`: add `wavetable_table = {}`, `pending_wavetable_def = None`, `pending_wavetable_ref = None`.
Register the 4 ops on one `WavetableDecoder()` instance in `DECODERS` (loop, like STAMP).

## 6. Wiring & scope boundaries

- `FREQ_BLOCK_PASSES`: insert `WavetablePass()` right after `SkeletonPass()` (re-runs per self-contained
  block via `run_freq_block_passes`; smaller per-block codebooks but still byte-exact).
- Flag auto-registers from `GATE_FLAGS={"wavetable_pass"}` (no `_PIPELINE_NAME_TO_FLAG` edit).
- **Out of scope for this PR (handback):** §4 B2/B3 — `WAVETABLE_REF` is a DEF→REF backref with the same
  out-of-window-DEF inference risk as STAMP; not inference-deployable until the constrained-decode
  registry + materialization land (build-order step 2). `blocks.py`/`expand_to_literal` self-containment
  for WT atoms to confirm. Author handles PyPI + `preframr` bridge + audio audition.

## 7. Test matrix (`tests/test_wavetable_pass.py`)

- Unit: loop-factoriser + `unroll` inverse (recurring, prefix+loop, one-shot, aperiodic→None).
- Unit: codebook grouping (≥WT_MINREP groups; transposition-invariance: same program at two base notes
  shares an id; bounded size).
- Isolation-oracle round-trip on a synthetic per-frame builder: run SkeletonPass(+held_arp) → record
  `register_state` (OFF) → `WavetablePass.apply` → `register_state` (ON); assert arrays equal.
- RESID-note-count drop: assert `ORN TYPE=RESID` count falls (and `WAVETABLE_REF` count rises) with the
  flag ON on a recurring-sequence fixture.
- Noise-inclusive: an interleaved noise-tik / onset attack still groups + round-trips byte-exact.
- Inline one-shot (§9): a structured non-recurring program emits DEF+REF, round-trips byte-exact.
- Real `parse()` fixture round-trip (skip on `FixtureUnavailable`).
- Gate: full `docker build -f Dockerfile .` (black, pytest, pylint-curated, pyright, coverage ≥85).

---

# As-built (shipped) — PR #41 `feat/wavetable-codebook`

Status: **§2 WAVETABLE codebook complete + RLE dedup.** `docker build` green (780 passed, coverage 89%,
pylint 10/10, pyright clean). This records what shipped and the impl-time reframes.

## What shipped
- **`macros/wavetable_pass.py` `WavetablePass`** (after `SkeletonPass` in `FREQ_BLOCK_PASSES`, gate
  `wavetable_pass` default OFF): mines the skeleton ORN-RESID note-relative offset dumps into an inline
  `WAVETABLE_DEF`/`WAVETABLE_REF` codebook (ops 65–68), or an inline-structured one-shot (§9). Full §2:
  onset-strip of leading non-pitched frames → per-hit `LEAD` on the REF; noise frames kept as program
  steps; RLE+loop factorisation; recurrence codebook (`WT_MINREP=2`) + verify-match for short/partial hits.
- **`macros/wavetable.py`** pure helpers (`factorise`/`unroll`/`program_key`) shared by the encoder verify
  and `WavetableDecoder` so they cannot disagree; `WavetableDecoder` in `DECODERS`; `DecodeState.
  wavetable_table`.
- **`macros/rle.py`** (`run_length_encode`/`run_length_decode`) — extracted shared run-length codec; both
  `skeleton_pass._rle` and `wavetable.factorise` now route through it (your no-duplicate-RLE directive).

## Impl-time reframes (the "why")
- **"Byte-exact" = byte-identical to the ORN-RESID content-floor render it replaces**, NOT to the raw log
  (the skeleton is content-tier, deliberately lossy). Decisive oracle = isolation oracle
  (`register_state` OFF == ON). Verify-or-fall-back-to-RESID is the safety net.
- **§2's "per-step waveform marker" is replay-inert at the content floor** — a noise step is already a
  fixed stored offset, so byte-exactness comes from replaying stored offsets exactly. Ctrl-awareness
  shapes GROUPING only (onset-strip / noise-as-step); no marker atom is carried in the byte-exact stream.
  Trivially reversible if a future bit-exact wavetable tier wants it.
- **Factorise on RLE STEPS, not frames** — a hold is not a loop (an early bug: a `(0,2)` hold read as a
  period-1 loop). The loop body is one period; `unroll` repeats it.
- **Recurring FLAT one-shots are codebooked too** — grouping is by program key over ALL records, not only
  looping ones; `has_body` only gates the inline-structured fallback.

## Not inference-deployable until §4 (now done)
`WAVETABLE_REF` is a DEF→REF backref with the out-of-window-DEF risk — fixed by the §4 B2/B3 constrained-
decode work in PR #42 (`CONSTRAINED_DECODE_REGISTRY_DESIGN.md`). The completeness test there forces the
WAVETABLE_REF contract once this branch merges. Author handles PyPI + `preframr` bridge + audio audition.
