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

**Numbers (5 drivers, 1500 BPE merges):** with the §8.3 note layer, post-BPE **4.12 bits/write** (3.88×
vs the 16-bit raw dump floor) — up from 4.55 / 3.50× with CTRL/AD/SR on plain byte lanes. The note
structure costs ~3% more *atomic* tokens but folds far better under BPE (recurring attacks become single
tokens), the §2.3 thesis. The 10–14× target is corpus-scale (BPE shares the dominant ORDER descriptor + recurring
gestures across many tunes) and improves with the note/attack + joint-DP layers below.

## Hard invariants (§2)

Met: §2.2 escape-free complete encodings · §2.3 no per-tune ids (BPE *is* the dictionary) · §2.4 factored
not fused (per-register lanes) · §2.7 one time encoding (frame-delta varint everywhere) · §2.8 byte-exact
ordered stream · **§6 no-note-off** (gate-on is the only explicit note marker `FLD_NOTE_ON`; gate-off and
the sustained envelope are the *absence* of edges — there is no note-off token in the alphabet).

Not yet met: **§4 mixed-radix tick** — lives in the duration layer (below); the note layer currently
carries no explicit per-note duration (gate-off rides as a plain CTRL edge), so duration derivation is the
next refinement on top of the §8.3 substrate.

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
  *Deferred refinements on this substrate (not blockers):* explicit per-note **duration** with gate-off
  *derived* from it (§4/§6 compression — currently gate-off is a stored CTRL edge, lossless but not yet
  derived); body-CTRL as an explicit `MOD_CTRL` gesture (currently inline `FLD_CTRL` edges); the structural
  (not fixed-W) attack window; the §8.4 joint freq/note DP that would let some onsets share a note-index.

## Remaining (doc build order §11)

- **§4 mixed-radix tick:** replace the frame-delta `DT`/duration with `q·tick + r` over the tick header
  (tempo/key invariance). Pure compression/generalization; correctness unaffected.
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
