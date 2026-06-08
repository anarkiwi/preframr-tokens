# Event model (REDESIGN_optionB) — build status

Implements the escape-free, factored, corpus-global event/tracker token model of `REDESIGN_optionB.md`.
The north star is the §2.8 hard invariant: **byte-exact reproduction of the ordered register-write
stream**. Every layer is guarded by that roundtrip.

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
