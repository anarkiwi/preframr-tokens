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

**Numbers (5 drivers, 1500 BPE merges):** post-BPE ~**4.2 bits/write** (~3.8× vs the 16-bit raw dump
floor). The §8.3 note layer beat plain byte lanes here (4.55→4.12); the §4 duration layer adds a small
*within-tune* order-0 cost (the per-note mode token) and is ~neutral on 5 same-corpus drivers — by design,
because its win is **corpus-scale tempo-invariance**, which 5 drivers can't show. The 10–14× target is
corpus-scale (BPE shares the dominant ORDER descriptor + recurring gestures/attacks across many tunes).

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
  (currently inline `FLD_CTRL` edges); the structural (not fixed-W) attack window; the §8.4 joint freq/note
  DP that would let some onsets share a note-index.

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

## Remaining (doc build order §11)

- **§2.7 finish:** route the ORDER-descriptor `DT` and gesture `LEN` through the same mixed-radix scheme
  (+ §4.1 span inheritance: a gesture whose span == the note duration emits no length token).
- **§8.4 joint 2-D DP:** the freq lane currently uses nearest-grid note-index + recovered table (the
  greedy bootstrap). Upgrade to the optimal shortest-path joint parse for fewer note events.
- **§3.1 global lane semantics / §8.5 labels:** PW/cutoff/res/vol are already byte-exactly gesture-covered
  (mdl scalar = §8.5); add the semantic `MOD_PW`/`MOD_CUTOFF`/`FILTER_CTL`/`MOD_VOL` typing + `VOICE=GLOBAL`
  grouping.
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
