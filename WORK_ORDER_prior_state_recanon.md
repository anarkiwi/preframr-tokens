# Work order — prior-state-aware re-canonicalisation (Tier-4 DAgger P0)

**Goal:** make `events.generate.recanon` work on the streams DAgger actually rolls out — windowed,
keyframe-led, possibly mid-event — so a model rollout can be projected onto a valid SID state and
re-prompted. This is the binding P0 for Tier-4 (see
`preframr-xpt/design/generation/dagger_recanonicalization_design.md`). Until it lands, the recoverability
triage and the DAgger build cannot run on real eval-B rollouts.

## What already exists (do not redo)

`events.generate.recanon(n_ids)` (PR #84) is correct for **continuous, keyframe-free** streams: strip
PAD, `strip_keyframes`, `decode`, `ordered_writes(writes_to_dump_df(...))`, re-encode. Green unit tests
in `tests/test_events_recanon.py` (identity / idempotent / write-preserving / PAD-drop / keyframe-free).
Real DRAX ×12: reg/val content exact on all; exact identity 9/12.

## The three tangled problems (diagnosed; keep them separate)

1. **Mid-event truncation.** `.blocks.npy` rows are `seq_len` windows (8193 atoms) that end mid-event,
   so `decode` raises at the tail. A rollout is also fixed-length → mid-event. **Fix:** trim to the last
   whole frame before decode (the audition's `decode_tolerant` logic; `recover_triage.py` already has a
   `trim_decodable`). Cheap; fold a `trim` option into `recanon` or require the caller to trim.

2. **Leading-rest frame-base off-by-one** (the 3/12 non-identity tunes). `ordered_writes` makes `frame`
   a dense index over *occupied* frames, so a leading empty/rest frame (a DT before the first event) is
   normalised away → the whole tune shifts one frame earlier. Content-exact, musically negligible, but
   breaks strict round-trip. **Fix:** preserve the leading-rest by carrying the absolute first-frame
   offset through `writes_to_dump_df` → `ordered_writes` (or prepend the lead frame on re-encode).

3. **Keyframe prior state (THE CORE TASK).** Later windows carry a leading `[KEYFRAME … KEYFRAME]`
   conditioning segment (`stream.KEYFRAME = 126`). It encodes the decoder state at the window start —
   per-voice `TUNING`/`NOTE_TABLE`/`TICK` headers + current `NI_STEP`/`FD_STEP`/`PW_STEP`/`FLD_CTRL`/`AD`/
   `SR` + global `G_STEP` values (see `stream.chunk_keyframe`, lines ~1138–1188, which EMITS it from a
   `_Decoder` state `d`). `strip_keyframes` discards it, so the body's deltas decode from a zero state →
   wrong writes (recanon delta ≈ 1.0 on windowed rollouts, won't re-prompt). Removing just the marker
   atoms does not decode either (body starts with a header/VOICE token, not a DT).

   **Build the inverse of `chunk_keyframe`:** a keyframe-consuming decode that parses the
   `[KEYFRAME … KEYFRAME]` content and SEEDS a fresh `_Decoder`'s state fields (`q`, `devs`, `tick`,
   `ni[v]`, `fd[v]`, `pw[v]`, `_ctrl_state[v]`, `_ad_state[v]`, `_sr_state[v]`, `g[reg]`, and the
   `*_active` flags), then decodes the body from that seed. The trajectory fields (`ni[v]`, `fd[v]`,
   `pw[v]`, `g[reg]` are `.at(f)` interpolators) must be initialised to a constant at the snapshot value.
   Mirror `chunk_keyframe`'s emission order exactly to parse it back. `_Decoder.__init__` is at
   stream.py:836; the body parse handlers (NI_STEP/FD_STEP/PW_STEP/FLD_CTRL/AD/SR/G_STEP/TUNING/
   NOTE_TABLE/TICK) define the field semantics.

## Acceptance tests (the spec)

In `tests/test_events_recanon.py`, on **real windowed blocks** (build via
`dataset.dump_block_ids(df, frames_per_block, stride)` with a small window so later windows carry
keyframes — or a corpus-sample fixture if the suite has one):

1. **Round-trip identity on a keyframe-led window:** `recanon(window) == window` (after trim), including
   the keyframe-seeded prior state — the decisive test that was impossible before.
2. **Write-preserving:** `ids_to_writes`-equivalent (the seeded decode) of `recanon(window)` equals the
   window's intended writes (absolute, prior-state-correct).
3. **Idempotent** and **re-promptable** (an `EventConstraint` primes from `recanon(window)` and a
   continuation decodes). Keep the continuous-case tests green.

## Gotchas

- These windows do NOT decode standalone via `decode()` today (truncation + keyframe) — fix #1 and #3
  together before asserting anything on real blocks.
- Test through real `parse()`/real blocks, not synthetic dfs (synthetic ships false-green).
- Repo lint: no narrative `#` comments, docstrings ≤5 lines; the gate runs under xdist.
- Validate on the v2 corpus, then re-run the recoverability triage (`/scratch/tmp/recover_triage.py`,
  uses `generate.recanon` + `trim_decodable`) on real eval-B rollouts — recanon delta should drop from
  ≈1.0 to ≈0, and `empty_recanon` should become a real number.
