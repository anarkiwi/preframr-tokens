# Event model (REDESIGN_optionB) — build status

Implements the escape-free, factored, corpus-global event/tracker token model of `REDESIGN_optionB.md`.

## v3 — the canonical contract (CURRENT; supersedes §2.8 byte-order fidelity)

The fidelity contract was **corrected** (2026-06-11): the oracle is `stream.canonical_writes(dump)` —
the dump's audibly-faithful canonical form — NOT the raw byte order. It is an exact **intra-frame
permutation** of the dump's writes: **zero writes are dropped** (enforced by a multiset assertion on
every `encode`). Concretely:

- **CTRL/ADSR**: ALL *change* activity preserved as ordered typed events at **sub-frame** resolution
  (hard-restart `CTRL,CTRL` and `SR,AD,NOTE_ON` onset sequences are events in driver order — the
  previously-literalized, most musically critical class). Gate 0→1 = typed `NOTE_ON` carrying the §4
  mixed-radix duration; **gate 1→0 is ALWAYS derived** at onset+duration. There is **no NOTE OFF
  token and no fallback path** — structurally sound: gate only reaches 1 via a NOTE_ON, so every
  gate-off pairs (measured: 0 unpaired across 7M in-scope writes). The DERIVE/VALUE release mode is
  decided at the gate-off's *canonical* slot (`_assign_gate_off_modes`), not its dump position.
- **freq/PW**: settled value per frame, canonically FIRST in the voice's frame group (reg-offset
  ascending). Transient same-frame pre-writes (0.13% of in-scope writes: the Commando freq re-fire —
  dominant payload = prev byte −1 — and player init wipes) are **`PRE` events**: `[PRE][reg][signed
  delta vs the reg's previous-frame byte]`, slotted at the head of their voice group in driver order.
  Nothing is dropped. (The freq/PW-between-two-CTRL-changes position exception measured 0.71%, all
  hard-restart onsets — position canonicalization only; Facemorph's noise→tonal instrument verified
  strictly inter-frame.)
- **globals** (filter/vol): settled, canonically LAST in the frame, reg ascending.
- **writes are implied, not transmitted**: the canonical write set = "bytes whose settled value
  changed" + PRE events + the cas write sequence (ALL cas writes kept, including the 10-in-1.6M
  same-value rewrites). The v1/v2 ORDER descriptor and ALL literal mechanisms are **gone** — every
  write value derives from modeled state.
- **voice-grouped frames**: `[DT]([VOICE][kind-led event bodies]*)*` — the VOICE token appears once
  per voice per frame, and event bodies are voice-free, so a patch (drum sequence, instrument) emits
  identical tokens on any voice; BPE learns voice-portable patch fragments (voice tokens fell
  18.2% → 11.1% of the stream).
- **Scope**: single-speed, non-digi tunes (`stream.single_speed`, `dump_meta.is_digi`). Multi-speed
  (~5%) and digi (~3%) are excluded from corpus builds/tests — raw corpus globs must filter.

Stream shape: `[n_frames][headers]([DT][voice events in canonical order])*` — headers on first use
(TUNING omitted at 0; NOTE_TABLE = nonzero grid deviations only, delta-coded; TICK omitted at 1,
recovered by exact-grid fit with groove offset ∈ {-1,0,+1} — the ±1/unconstrained criteria were
measured degenerate). Pitch is interval-coded (`NI_STEP/NI_RAMP`); freq residual, 12-bit PW and global
lanes are STEP/RAMP events with no-op suppression (held value = no event; HOLD = STEP with no length);
gesture covers use the emitted-token cost model (§8.6 completed — `mdl_parse(cost_model=...)`).
Vocab: 84 atoms (32 digits, 25 regs, 4 voices, headers/kinds/shapes). `encode` self-verifies
`decode == canonical_writes` (fail loudly).

**Verified**: canonical roundtrip on the 5 drivers + in-scope 200-sample corpus sweep; determinism;
no-NOTE-OFF, driver-order cas, freq-first/globals-last reorder, retrigger/blip chains pinned in
`tests/test_events_stream.py`. Full suite green except 2 pre-existing failures (GEN_TABLE tiering,
decompose_voice removal).

**Measured** (59 in-scope tunes + 5 drivers, 1.6M writes; canonical = **100.0%** of raw writes —
zero-drop, every tune):

| | atomic tok/write | atomic H1 bits/write | post-BPE tok/write (drivers, 1000 merges) | post-BPE H0 bits/write |
|---|---|---|---|---|
| v1 factored (byte-exact baseline) | 2.99 | 11.07 | 0.55 | 5.07 |
| v2 time-ordered (interim, byte-exact) | ~2.4 (5-driver) | ~6.7 (5-driver) | 0.51 | 4.67 |
| **v3 canonical (voice-grouped, zero-drop)** | **1.66** | **5.92** | **0.23** | **2.04** |

v3 collapse vs the 16-bit raw floor: **7.8×** at post-BPE order-0, **23×** at order-1 — past the §9
10-14× target. Per driver: trap 0.30, grid_runner 0.61, commando 1.45, camerock 1.65, baggis 1.75
tok/write atomic. Token families: digits 65.6%, kind 19.7%, voice 11.1%, shape 2.3%, reg 1.2%.

## Historical: v1 (columnar factored) and v2 (time-ordered, byte-exact) status below

`factored.py` remains in-tree as the byte-exact measurement baseline. The §2.8-era notes below predate
the v3 contract.

## Built & verified (green)

| module | role | doc |
|---|---|---|
| `oracle.py` | ordered write stream `(frame,reg,val)` = the fidelity oracle; settled grid (secondary) | §0, §2.8, §7 |
| `varint.py` | complete escape-free zig-zag/base-16 integer codec (one digit alphabet) | §2.2, §3.5 |
| `schema.py` | SID register map, voices, `Kind`/`Shape` enums, `Event` | §3, §3.5 |
| `gestures.py` | self-contained HOLD/POLY/PERIOD cover + lossless replay over `mdl_core` | §8.4-8.5 |
| `encoder.py`/`decoder.py`/`tokenize.py` | v0 verbatim skeleton (per-write, byte-exact safety net) | §11 step 3 |
| `factored.py` | **v1 factored codec**: per-register gesture lanes + per-frame **ORDER descriptor** + freq two-layer + **note/attack layer** | §7.0, §8.2-8.5, §13 |
| `measure.py` | corpus-global greedy-BPE collapse / bits measurement | §9, §8.6 |

**Verification:**
- Byte-exact ordered-stream roundtrip on all 5 driver fixtures (grid_runner, commando, camerock, trap,
  baggis) and a 200-tune corpus sample — both v0 and factored (`tests/test_events_roundtrip.py`).
- 300/300 corpus tunes byte-exact direct scan; **176 had multi-speed sub-frame repeats, all reconstructed
  exactly** via the ORDER descriptor + literal write-run path (§13's sanctioned primitive — not an escape).
- No-escape invariant, expansion guard, gesture-basis losslessness, determinism, strict-grammar rejection
  (`tests/test_events_acceptance.py`).

**Numbers (5 drivers, 1500 BPE merges):** post-BPE **3.77 bits/write (4.24×** vs the 16-bit raw dump
floor). Progression: byte-lane CTRL/AD/SR 4.55 → §8.3 note layer 4.12 → §4 duration layer 4.23 (a small
*within-tune* order-0 cost; its win is corpus-scale tempo-invariance, below) → **§8.5 PW-combine 3.77**. The
10–14× target is corpus-scale (BPE shares the ORDER descriptor + recurring gestures/attacks across tunes).

**§4 numbers (120 corpus tunes, the right scale):** encoding note durations as `q·tick + r` cuts their
pooled coding cost **H(raw frames)=3.83 → H(q)+H(r)=2.08+1.29=3.37 bits** (−12%) and collapses the alphabet
(423 distinct raw durations → 218 `q` + **14** `r`). A quarter note is the same `(q,r)` regardless of
tempo — the generalization the order-0 5-driver metric structurally cannot reward.

**§4 numbers (120 corpus tunes, the right scale):** encoding note durations as `q·tick + r` cuts their
pooled coding cost **H(raw frames)=3.83 → H(q)+H(r)=2.08+1.29=3.37 bits** (−12%) and collapses the alphabet
(423 distinct raw durations → 218 `q` + **14** `r`). A quarter note is the same `(q,r)` regardless of
tempo — the generalization the order-0 5-driver metric structurally cannot reward.

## Hard invariants (§2)

Met: §2.2 escape-free complete encodings · §2.3 no per-tune ids (BPE *is* the dictionary) · §2.4 factored
not fused (per-register lanes) · §2.8 byte-exact ordered stream · **§6 no-note-off** (gate-on is the only
explicit note marker `FLD_NOTE_ON`; the gate-off value is *derived* from the note's duration — there is no
note-off token in the alphabet) · **§4 mixed-radix tick** (per-voice `tick`/`offset` header; note durations
encode as `q·tick + r`, tempo-invariant — see §4 numbers below).

Partially met: **§2.7 one time encoding** — note durations are mixed-radix, but the ORDER-descriptor `DT`
and gesture `LEN` are still raw frame-delta varints; unifying those onto the one scheme is the remaining
step.

## Production swap — DONE (§7.1, the events codec IS the pipeline now)

The events codec replaced the old `parse → (op,reg,subreg,val) alphabet → merge_token_df` substrate, and
the old gesture/codebook machinery is retired.

- **Tokenizer/alphabet** (`events/dataset.py`): a fixed 68-atom alphabet (atom id +1 into n-space, 0=PAD);
  the alphabet-agnostic `RegTokenizer` unicode-serialize + BPE (`train_worker`) + encode/decode is reused.
- **Dataset build** (`corpus.preload`): globs raw dumps, sets the event alphabet, trains BPE over per-dump
  whole-tune event token streams, writes per-dump `.0.blocks.npy` (BPE-encoded stream chunked to seq_len+1)
  + tokens.csv + df-map + reg-widths; `iter_block_seqs` serves event blocks unchanged.
- **Generation** (`events/generate.py`): generated BPE ids → `tokenizer.decode` → `ids_to_writes` → ordered
  writes → render-ready dump df. The factored decoder is the strict grammar/completeness oracle.
- **Retired:** the gesture PASS (`mdl_gesture_pass`/`arbiter`/`codebook_emit`), the codebook op subsystem
  (gesture was the last family → `CODEBOOK_FAMILIES = {}`, GESTURE ops de-registered from
  `op_contracts`/`macro_contracts`), the prototypes + this-era docs, and the tests of all that.

Validated byte-exact end to end (synthetic corpus, identity + trained BPE): the reassembled block stream
decodes to the exact ordered writes. Full suite (minus the 6-min corpus test): **676 passed, 3 failed** —
all 3 pre-existing on this branch (decompose_voice removed earlier, GEN_TABLE tiering, repo-wide lint),
**zero events-swap regressions**. Model train/generate end-to-end is a downstream run (not done here).

*Residual cleanup (cosmetic, non-blocking):* unreachable dead codebook code still sits in
`constrained_decode`/`validators`/`regtokenizer`/`decoders`/`stfconstants` (no codebook ops reach it) and
the old `Corpus.make_tokens`/`encode_and_save_cached_blocks` + parse/blocks machinery are now unused;
deletable in a follow-up. The §2.7 ORDER-`DT`/gesture-`LEN` unification and the §7.1 generation grammar
mask remain as optimizations (the decoder already validates).

## Done since last status

- **§8.3 note/attack layer (step 2, task #4) — LANDED.** Each voice's CTRL/AD/SR settled series is owned by
  a per-voice note section (`NOTE_MARK voice n_edges (DT FLD VAL)*`), replacing its three byte lanes. A
  gate-on (CTRL bit0 0→1) is the typed `FLD_NOTE_ON` edge (§6); gate-off / body-waveform are `FLD_CTRL`;
  AD/SR ride `FLD_AD`/`FLD_SR`, so the *held* sustained envelope is the absence of edges between notes and
  hard-restart (AD/SR rewritten across the onset window) is reproduced edge-for-edge. **Byte-exactness is
  structural:** the section reconstructs `settled[:, ctrl/ad/sr]` exactly, and the unchanged ORDER + literal-
  run mechanism supplies write order and same-frame repeats — so no attack-form fallback is needed for the
  value series (intra-frame double-writes already route to the §13 literal run). Verified byte-exact on all
  5 drivers + the 200-tune corpus sample, and isolation-tested (`tests/test_events_acceptance.py::
  test_note_layer_gate_on_typed_and_byte_exact`).
  *Deferred refinements on this substrate (not blockers):* body-CTRL as an explicit `MOD_CTRL` gesture
  (currently inline `FLD_CTRL` edges); the structural (not fixed-W) attack window.

- **§4 mixed-radix tick + §6 gate-off derivation — LANDED.** Each voice's note section now carries a
  `tick`/`offset` header (§8.1 grid-fit on gate-on durations: largest unit in [2,32] with ≥90% on-grid
  mass, offset = mode residual). Each `NOTE_ON` carries a `mode` + its **duration** as `q·tick + r`, and
  the matched gate-off CTRL edge is **removed** and re-synthesized at `onset+duration`: `mode=DERIVE` (78%,
  value = held waveform with the gate bit cleared), `mode=VALUE` (22%, the explicit release byte), or
  `mode=NONE` (the 0.1% drone/tie where the gate runs past the next note). Edge `DT`s stay raw varint (only
  the duration is mixed-radix — applying it to every sub-tick edge gap regressed; the win is in durations).
  A per-voice encode-time replay verifies the reconstruction and falls back to plain edges on any mismatch,
  so byte-exactness is guaranteed. This honors §4 (tempo-invariant durations) and **completes §6** (the
  gate-off value is derived, no note-off token). Verified byte-exact on 5 drivers + 50-tune spot-check +
  the 200-tune corpus roundtrip; pinned by `test_note_duration_carried_and_gate_off_derived`.

- **§8.5 PW combined lane — LANDED.** PW lo (reg 7v+2) and hi (reg 7v+3) are adjacent bit ranges, so each
  voice's PW is encoded as one combined 12-bit value lane (`(hi<<8)|lo`, byte-exact split on decode) under
  a `PW_MARK` instead of two byte covers. A sweep is one ramp instead of two byte covers fighting at the
  lo-byte wrap: **5-driver post-BPE 4.23 → 3.77 bits/write (−10.8%), collapse 4.24×**, byte-exact on 5
  drivers + 50-tune spot-check + the 200-tune corpus. Pinned by `test_pw_combined_lane_byte_exact`.

## Measured dead-ends (investigated, not shipped — roadmap refinements)

- **§2.7 ORDER-descriptor `DT` as mixed-radix — REJECTED (measured harmful).** The inter-write-frame gap is
  **entropy ≈ 0: 100% of values are `1`** (consecutive frames), already one BPE-folded token. Mixed-radix
  would turn every `1` into `(q=0, r=1)` = 2 tokens, doubling the most common token in the stream. The
  strict "one time encoding" purity is counterproductive here; ORDER `DT` stays raw varint.
- **§8.4 joint freq/note path DP — REJECTED (measured regression, reverted).** A shortest-path parse over
  `(frame, note-anchor)` emitting NOTE_STEP events + per-segment delta gestures was implemented and is
  byte-exact, but cost **4.23 → 4.42 bits/write**. Root cause: the current two-tiling freq encoding covers
  the note-index and the delta as *separate* series, so `delta = 0` (static notes, via the recovered table)
  spans across note boundaries as one long HOLD; the interleaved path forces a delta-gesture boundary at
  every note event, fragmenting `delta = 0` into a per-note HOLD. The two-tiling greedy is the stronger
  representation for the common static-note case. (A variant that keeps the two-tiling but picks a
  piecewise-constant note-index via DP could still help fragmented tunes like baggis — 536 ni-gestures —
  without the delta fragmentation; left as a future option.)

## Remaining (doc build order §11)

- **§4.1 span inheritance:** a gesture whose span == the enclosing note duration emits no length token
  (needs note-aware gesture boundaries). The only remaining clearly-beneficial mixed-radix/time item.
- **§3.1 global lane semantics / §8.5 labels:** add the semantic `MOD_PW`/`MOD_CUTOFF`/`FILTER_CTL`/
  `MOD_VOL` typing + `VOICE=GLOBAL` grouping (cutoff/res/vol stay separate byte lanes — combining cutoff
  lo/hi was measured −50.6% because reg21's low-3-bits register is not contiguous when byte-packed).
- **§7.1 grammar mask (step 1/5):** `factored.from_tokens`/`decode` is already a strict parser (the
  grammar); add the generation-time finite-state mask + completeness test.
- **§9 corpus-scale bits measurement** and **§7.1 retirement** of the old gesture-codebook row machinery
  for these channels (step 6) — the note layer has landed, so this is unblocked.

## Known caveat

The repo-wide lint suite (`tests/test_lint.py`: black-88, no narrative comments, ≤5-line docstrings) is
RED and was so before this work — the whole `events/` module is authored in a heavily-commented, §-doc-
referenced style with ~100-col lines (101 narrative-comment offenders module-wide; black wants ~88). New
code matches the module's established style; conforming the lint is a deliberate module-wide decision, not
part of the §8.3 build.
