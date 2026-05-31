# Implementation work order — RESID→0 Phase 4: drain the tail to zero

**Audience:** a preframr-tokens agent working in `/scratch/anarkiwi/preframr-tokens` (clean `main` @ ≥ v0.38.1).
**Author-side validation:** I (preframr-xpt) re-run the full-corpus survey on handback and gate the default flip + release.
**Scope:** tokens-only. Every change gated default-OFF, byte-exact, with parse-level tests, until the final flip (W7).

> **Operating rule (author directive, non-negotiable):** "If `resid > 0` it's because the engine that
> produced it isn't modelled yet." Every tune's engine is known and the dominant residue engines are
> **wavetable engines**. So `resid > 0` ⇒ a documented mechanism we haven't encoded. The endgame is:
> **`ORN_TYPE_RESID` never appears in any deployed token stream** — every residue is either a codebook
> `*_REF`, an inline wavetable one-shot, or routed to its proper parametric primitive (PLAIN / SLIDE /
> VIB / noise-hold) when that is what it actually is. RESID becomes an internal intermediate that is
> always rewritten before emission.

> **Process rule (just cost us a dead feature — do not repeat):** every encoding change MUST be tested
> through the full `RegLogParser.parse` path, NOT a direct `Pass().apply(df)`. The WAVETABLE pass shipped
> in 0.38.0 was **dead in production for an entire release** because it was wired into
> `macros.FREQ_BLOCK_PASSES` but omitted from the parallel hand-listed pass sequence in
> `RegLogParser.parse`, and the 8 unit tests called `.apply()` directly so they stayed green. See
> `tests/test_wavetable_parse_wiring.py` for the guard pattern; mirror it for every workstream here.

---

## 0. State on handback (what is already true)

- **0.38.1–0.38.2 (released, PyPI):** `WavetablePass` (ops 65–68) is wired into `RegLogParser.parse` after
  `SkeletonPass`, **default-OFF**. With `wavetable_pass=True` it drains the *recurring* and
  *structured-looping* RESID into a `WAVETABLE_DEF`/`REF` codebook + inline one-shots. 0.38.1 fixed the
  parse-path wiring (the pass was dead in 0.38.0); **0.38.2 fixed the same-frame codebook byte-exactness
  bug (W0 below) — the pass is now byte-exact corpus-wide.**
- **Constrained-decode `OpContract` registry** (`preframr_tokens/macros/op_contracts.py`,
  `OP_CONTRACTS`): one contract per emittable op with a `MaskRole`
  (`ATOM`/`CODEBOOK_DEF`/`CODEBOOK_STEP`/`CODEBOOK_END`/`CODEBOOK_REF`); a **completeness test fails at
  unit time if any op lacks a contract**. Every new op below MUST add one (W6).
- **Measured drain (post-0.38.1, wavetable ON):** dominant engines 67–93% byte-exact. The remaining
  residue — **the tail this work order closes** — is characterised below.

## 1. The tail, measured (full-corpus survey: `audit/probes/resid_corpus_survey.py`)

Survey = parse every canonical `.1.dump` (MUSICIANS+DEMOS+GAMES, ~60.5k tunes) once with `wavetable_pass`
ON; `drain% = REF/(REF+RESID)`; classify every surviving RESID note (offset-only). **Survivor classes and
where each is created in code:**

| class | share¹ | what it is | why it survives (code site) |
|---|---|---|---|
| **RECUR** | **52%** | exact offset seq recurs ≥2× in tune, mostly length-1/2 (e.g. `[31]`, `[33,0]`, `[-14,-31]`) | codebook keys on `factorise(core)` AFTER onset-strip + a **pitched-core** requirement, a stricter key than exact-tuple recurrence; `_MIN_CORE=2` drops len-1; `_make_record`→None drops no-pitched-core — `wavetable_pass.py:44,135,142-165` |
| **FLAT** | **36%** | unique, aperiodic, no loop body | inline one-shot only emits for `has_body` — `wavetable_pass.py:205` |
| **ZERO** | **8%** | all offsets 0 (held base / unresolvable-noise frames snapped to 0) | skeleton emits RESID-all-zero when `fn_to_note_resid` is None — `skeleton_pass.py:229` |
| **SHORT** | 3% | non-recurring single offset | `_MIN_CORE=2` (same as RECUR) |
| **SWEEP** | <1% | constant note-relative delta ramp | SweepPass is raw-freq domain; note-relative ramps slip through |
| **STRUCT/PERIOD** | <1% | loop-body-but-unique / period≤8 | `_make_record` reject (no pitched core) / onset-strip mismatch vs `classify` |

¹ Share of the post-wavetable tail, 1440-tune corpus survey (`design/resid_corpus_survey_1440.txt`; classes total 11,083
residual RESID). Structurally stable from 200 tunes. **The dominant lever is RECUR+SHORT (≈55%)**: the
codebook's factorise/onset-strip/pitched-core key is *more restrictive than exact-tuple recurrence* — this
is precisely the gap between the 89% exact-recurrence measured pre-build (SidWizard) and the 67% actual
drain. Re-run the full survey (`resid_corpus_survey.py all`) on handback only for **final rare-engine
coverage** (the W5 list); weights here are already decision-grade.

## 2. Architecture of the fix (do these in order; each its own branch, gated, byte-exact, parse-tested)

The tail is closed by making the WAVETABLE family total over residue. Three guards in `WavetablePass`
create most of it (`_MIN_CORE=2`, `has_body`-only inline, no-pitched-core reject); two classes belong to
*other* primitives (ZERO→PLAIN, SWEEP→SLIDE) and must be routed there, not absorbed, to preserve
**provenance-invariance** (same gesture → same tokens, principle P7).

### W0 — fix WAVETABLE byte-exactness — ✅ DONE (tokens 0.38.2, author-side)

A full-corpus byte-exact survey found **~2.5% of tunes failed the isolation oracle** (`register_state`
OFF≠ON) with `wavetable_pass` ON. **Root cause (confirmed):** the within-frame row sort `_norm_pr_order`
(`reglogparser.py`, key `["f","v","reg","op","n"]`) ordered by op-code, so when two variable-length
codebook blocks (`DEF→STEP*→END`) landed in the **same frame** it grouped all DEFs, then all STEPs, then
all ENDs — shattering both blocks; the decoder then mis-parsed them and replayed garbage freqs (often
clamping to 0xFFFF and desyncing every later frame). Concentrated in dense-codebook engines (GoatTracker,
DMC, JCH); latent for STAMP/PATCH too (rarer per-frame). **Fix:** each family's STEP/END collapse to its
DEF op for the sort (`_BLOCK_SOP`) so a block stays contiguous in emit order (`n`); a no-op when no
codebook ops are present, so non-codebook streams and the golden masters are unchanged. Guard:
`tests/test_wavetable_multidef_frame.py` (2-voice dump → two WAVETABLE_DEFs in one frame → byte-exact
through `parse`; fails pre-fix). Hunt tool: `resid_byte_exact_hunt.py` (full-verify every tune, logs
divergences). **This is why the DoD §5 byte-exact gate is now a FULL-verify pass, not 1-in-50** — keep it
green as W1–W5 land (each new op family is a new chance for a same-frame block collision).

### W1 — ZERO → PLAIN/noise-hold (skeleton, drains the ZERO class)

An all-offset-0 RESID note is a note **held at its base** whose freq either never moved or moved only on
unresolvable (noise/test) frames the content floor snapped to 0. That is a `PLAIN` held note, not a
wavetable. In `skeleton_pass.fit_descriptor` / the RESID-emit path, when the descriptor would emit RESID
with an all-zero offset tuple, emit `ORN_TYPE_PLAIN` instead (the content floor already replays base — so
the register bytes are identical). Keep the per-voice `ctrl` writes (they carry the noise/test timbre);
only the freq is held-base. **Isolation oracle must hold.** Do NOT collapse a note with any non-zero
offset here — that is W2/W3.

### W2 — short/literal codebook (drains RECUR+SHORT ≈55%, the dominant lever)

The RECUR survivors are NOT failing recurrence — they recur on their **exact verbatim offset tuple**; they
fail the codebook's *key*, which is `factorise(core)` after onset-strip + a pitched-core gate (designed for
long looping programs). For short sequences that key is wrong: a `[33,0]` or `[31]` transient has no loop
to factorise and may have a non-pitched onset that strips its core below `_MIN_CORE`.

Add a **literal-tuple short codebook** path in `WavetablePass`: for residue at or below a length threshold
(start `≤ 4`), key the codebook on the **verbatim offset tuple** (length ≥1, non-pitched frames included —
NO factorise, NO onset-strip, NO pitched-core requirement). A tuple recurring ≥ `WT_MINREP` drains to a
`DEF`/`REF`; the DEF stores the literal offsets, `unroll` is the identity for a loopless program (verify
`== offsets`). Longer residue keeps the existing factorise path. **Watch vocab growth:** only codebook
tuples that recur (unique short tuples fall through to W3's inline one-shot — do not mint single-use ids).
Parse-level test: recurring `[31]` and `[33,0]` transients across notes each drain to one DEF + N REFs,
byte-exact, including when the onset frame is noise.

### W3 — inline one-shot for EVERY non-recurring pitched residue (the RESID=0 guarantee)

Today inline one-shot emits only `if rec["has_body"]` (`wavetable_pass.py:205`), so FLAT stays RESID.
Make the WAVETABLE family **total over ALL residue**: any surviving RESID note that matched no codebook id
emits an inline one-shot (offsets stored **verbatim**; `unroll` reproduces them). **A one-shot needs NO
pitched core** — the pitched-core requirement in `_make_record` exists only to *key the codebook* (grouping
for recurrence); a one-shot just stores the literal offsets, so it must apply to noise-interleaved and
otherwise-unkeyable notes too. Drop the pitched-core gate on the one-shot path (keep it on the codebook
path). **Design decision (recommend):** add a dedicated `WAVETABLE_ONESHOT_OP` that inlines `STEP*` at the
note position with **no codebook id** (no DEF/REF indirection, no single-use id pollution, no
out-of-window materialization). Alternative (single-use DEF/REF) is simpler but bloats the codebook and the
constrained-decode live-id table — prefer the dedicated inline op.

**This is the RESID=0 backstop:** W1 (all-zero→PLAIN) and W4 (slides/sweeps→primitives) run FIRST as
*quality routing* (the correct, compressible model); whatever residue remains — flat, noise-interleaved,
unkeyable, any length ≥1 — is emitted verbatim as a one-shot here. After W3, `ORN_TYPE_RESID` cannot reach
the deployed stream. Gate it: assert `residual_resid == 0` on a corpus sample with all gates ON.

> **Why this is modelling, not renaming:** these are wavetable engines; a unique aperiodic note-relative
> program played once *is* a one-shot wavetable table. The vocabulary shift from "unresolved RESID escape"
> to "wavetable one-shot" is the modelling claim, and it is load-bearing for the LM: `WAVETABLE_STEP`
> atoms are small signed values that Unigram-cluster, so sub-sequences become shared sub-tokens across
> one-shots AND codebook entries — which a per-note `ORN P1` RESID dump does not get. Route genuine
> slides/sweeps to their primitives FIRST (W4) so the one-shot is a real mechanism, not a dumping ground.

### W4 — route mis-modelled residue to its primitive (quality / provenance-invariance)

Before W3's catch-all runs, reclassify and route:
- **SWEEP** (constant note-relative delta, monotone ramp) → `ORN_TYPE_SLIDE` (extend to target+duration if
  the rate-only form can't land exactly — see W5.1) or a wavetable sweep-step. A ramp must NOT encode as a
  one-shot in one tune and SLIDE in another.
- **PERIOD survivors** → align the onset-strip in `WavetablePass._make_record` with the `classify`/held-ARP
  strip so the codebook actually catches them.
- **monotone** runs → SLIDE.
Each routed class must produce the **same tokens** as the equivalent gesture from any other engine
(test #11.4 provenance-invariance pattern).

### W5 — frontier-tail parametric primitives (build ONLY what the FULL survey shows still leaks)

From the per-engine survey, add a primitive only where a *class* persists for an engine after W1–W4. Candidates
(carried from Phase-3 §3; **confirm against `design/resid_corpus_survey_full.txt` before building**):
1. **SLIDE target+duration.** Some engines (MoN/FutureComposer) compute `(target−cur)/duration` and land
   exactly after N frames; rate-only SLIDE can't. Add a target+duration variant (byte-exact landing).
2. **VIB delay+length(+shape).** Sine-LUT vibrato with onset delay + finite length (MoN); current VIB is
   depth+rate only.
3. **SWEEP loop-period.** A looping freq-domain arp (Soundmonitor: constant −Δ/frame, period N — see the
   survey's Soundmonitor RECUR/STRUCT examples `[-11,-44,...]`). Extend SWEEP with `loop_period`.
4. **PERC primitive.** No-pitched-frame drums (noise waveform + freq table/sweep) that STAMP didn't catch
   because they don't recur exactly — a parametric percussion primitive drains them. (Overlaps ZERO/noise
   residue not covered by W1.)
Each: own branch, gated default-OFF, byte-exact, own parse-level tests, own `OpContract`.

### W6 — constrained-decode contracts + completeness for every new op

Every op minted above (`WAVETABLE_ONESHOT_OP`, any single-offset codebook op, W5 primitives) gets an
`OpContract` co-located with its `MacroDecoder`, registered in `OP_CONTRACTS` with the right `MaskRole`,
and the completeness test stays green. Inline one-shots are **self-contained** (no out-of-window
materialization) — simpler than the codebook REF path; assert that in a mask⟺decode test. Re-use the
B0–B4 machinery from #42; do not add a fourth hand-kept copy of the decode walk.

### W7 — flip default-ON, re-cut, cross-repo release (author-gated)

Only after the survey shows **`residual_resid == 0` corpus-wide, byte-exact, 0 corruptions**: flip
`wavetable_pass` (and the new W1–W5 gates) default-ON. This is **breaking** (op-alphabet shift → re-cut
corpora/checkpoints, no metric transfer — see the 0.34.0 changelog convention). Sequence per
[[cross-repo-release-ordering]]: bump tokens minor (0.39.0), tag `vX` → PyPI OIDC, THEN bump preframr's
tokens floor + add the `wavetable_pass`/new-gate **args bridge** in preframr so its CLI mirrors the flags.
**I (author) run the corpus survey + audition and own this flip — hand back at the end of W6 with the
survey clean.**

## 3. Validation gates (every workstream, no exceptions)

1. **Parse-level test** through `RegLogParser.parse` (not `.apply`) — the wiring-bug guard. Build dumps via
   `tests/parse_probes.DumpBuilder`; assert the new op appears in the deployed stream AND RESID drops.
2. **Isolation oracle:** `register_state(off) == register_state(on)` on the fixture and on a corpus sample.
3. **Completeness:** `OP_CONTRACTS` covers the new op (W6) — test goes red otherwise.
4. **No vocab blow-up:** assert codebook/one-shot id counts stay bounded on a fixture (W2/W3 risk).
5. **Provenance-invariance (#11.4):** same gesture → same tokens regardless of provenance (W1/W4).
6. **Lint:** repo enforces black + no-narrative-comments + ≤5-line one-paragraph docstrings (see
   `tests/test_lint.py`). New tests/code must pass it.
7. **Full suite green in docker** (`anarkiwi/preframr-tokens-test:3.12`); host lacks `pytz`.

## 4. Author-side items (do NOT attempt from the tokens repo)

A few small engines are ~100% UNRESOLVED and need register-level reverse-engineering against driver
sources (GMC/Superiors, SynC, Jeff, SkyLine_Editor, LordsOfSonics, and any new ones the full survey
surfaces). The author traces these and hands you a primitive spec in the W5 shape if one is needed. Do
not guess them.

## 5. Definition of done

`residual_resid == 0` for every engine in `resid_corpus_survey.py all`, full suite green, every new op
contracted, default-OFF until W7. **Byte-exactness gate (hard): zero corruptions under a FULL-verify
corpus pass** (`resid_byte_exact_hunt.py all` — register_state OFF==ON for EVERY tune, not the 1-in-50
sample). The current pass already fails this (~3% of verified tunes diverge — see W0). Hand back with both
surveys clean; author flips and releases.

---
*Survey tooling (xpt `preframr_experiments/audit/probes/`): `resid_corpus_survey.py` (full-corpus drain +
tail classes + 1-in-50 byte-exact verify), `resid_wavetable_drain.py`, `resid_wavetable_recurrence.py`,
`resid_engine_profile.py`, `sidid_cache.py`.*
