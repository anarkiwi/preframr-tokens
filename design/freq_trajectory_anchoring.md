# Trajectory anchoring (gate-/sweep-aware FREQ_TRAJ + PW/filter)

**Status:** Design, ready to implement. Self-contained: implement + test entirely
within preframr-tokens. No torch, no external tracker tools required (the validation
fixtures are synthetic). Target version: next minor (`fallback_version` bump).

## Problem

`FreqTrajectoryPass` (`preframr_tokens/macros/freq_trajectory_pass.py`) segments each
trajectory register (`TRAJ_REGS = (0, 2, 7, 9, 14, 16, 21)` = per-voice freq + PW, and
the global filter cutoff) into ramp/oscillation/run primitives **purely from the
register's own value runs and frame-contiguity** (`_emit_ramps`, `_emit_osc_run`). It
never consults the control/gate register or any note/sweep structure. Consequence:
a trajectory's start anchor (`V0`) coincides with a real note/sweep origin only by
accident — the melodic note-onset pitch gets buried as a delta inside a value-segmented
trajectory.

Downstream, a sequence model trained on these tokens predicts the FREQ_TRAJ stream at
~0.026 content accuracy (it cannot tie a frequency event to the note-on that triggered
it; it falls back to predicting the dominant SET register content instead). The melodic
information *is* present and lower-entropy when anchored correctly, but the encoding
discards the anchor. See the upstream investigation summary at the end.

We want a small, **annotation-only** pass that recovers the true trajectory **origin
frames** per register from the observable register dynamics, so `FreqTrajectoryPass` can
start each trajectory at its real anchor (`V0` = the anchored base value) instead of at
an arbitrary value-run boundary.

## Definition of an anchor (generator-agnostic)

An anchor is an **observable (re)initialization of a trajectory**, recovered from the
register stream alone. It is NOT "note-on" and must not assume any tracker convention —
arbitrary 6502 code generates SID music, and PW/filter sweeps are frequently armed
mid-note (verified: defMON cutoff re-arms on its own clock 72/127 times off-gate;
SID-Wizard bass filter table loops a sweep off-gate; *A Mind Is Born* sweeps the filter
across the whole tune with the gate irrelevant). Two observable events (re)initialize a
trajectory:

1. **Value-origin (intrinsic):** the register's value departs from its current level
   into a new sustained level, a new ramp, or reverses a ramp.
2. **Retrigger (gate):** the voice's control-register gate bit goes 0→1. This is the
   *only* observable for a repeated same-value note (no value change to detect). It is a
   per-voice event and applies to FREQ and PW; the **global filter is not voice-gated**,
   so it is intrinsic-only.

**Granularity = note/sweep level**, not every internal driver re-init. An arpeggio is
ONE textured note (one anchor at its onset), a vibrato'd held note is ONE note, a
continuous filter sweep is ONE trajectory. Driver re-inits that produce no observable
value change and no gate (e.g. re-arming a slide to the same rate) are **out of scope** —
unobservable from writes and irrelevant to a write-only model.

## Algorithm (two annotation passes)

Operate per register stream. For each, build a per-frame carry-forward value series and
(FREQ/PW) the gate-on frames. Register value encodings:
- FREQ (regs 0/7/14 lo, +1 hi): 16-bit; convert to semitone `round(12*log2(freq))`,
  NaN when freq==0 (voice silent → inactive frame).
- PW (regs 2/9/16 lo, +3 hi): 12-bit `((hi & 0x0F) << 8) | lo`.
- FILTER (reg 22 hi, 21 lo): 11-bit cutoff `(hi << 3) | (lo & 0x07)` (global).
Gate per voice: control reg `4/11/18` bit0; gate-on = 0→1 transition frame.

### Pass 1 — sustained-departure origins (over-segments; recall ~1.0)
Smooth the value with a median filter (window `W`, suppresses vibrato/PWM jitter). Walk
frames tracking a reference level `ref`; emit a candidate origin where the value leaves
`ref` by more than `band` and the new level **holds ≥ `min_hold` frames** (a transient
that returns within `min_hold` is modulation, not an origin); on emit, `ref` ← median of
the new level. Returns `(origin_frames, level_value_at_each)`. This deliberately
over-segments a slow ramp into its steps and an arp into its tones; pass 2 fixes that.

### Pass 2 — ramp + oscillator collapse (restores precision)
Group pass-1 origins into dense runs (consecutive gaps ≤ `P_MAX`). Collapse a run of
≥ `MIN_RUN` origins to **its onset only** if either:
- **RAMP:** the origin level values are monotonic (all-increasing or all-decreasing) —
  a sweep/staircase is one trajectory; or
- **OSCILLATOR:** the value *waveform* over the run's frame span is **periodic**
  (normalized autocorrelation has a peak > `AC_THRESH` at any lag in `[2, span/2]`) —
  vibrato/arp/PWM/filter-wobble is one trajectory.
Runs that are neither (aperiodic, non-monotonic → a genuine fast melodic line) are kept
intact. This is the key discriminator: autocorrelation cleanly separates an *arp*
(periodic value waveform → collapse) from a *fast melody* (aperiodic → keep), where a
distinct-value-count heuristic does not.

### Final anchors
`anchors = pass2(pass1(value)) ∪ gate_on` for FREQ and PW; `= pass2(pass1(value))` only
for FILTER (not voice-gated).

### Parameters (validated starting values — tune against a corpus)
| register | median `W` | `band` | `min_hold` |
|---|---|---|---|
| FREQ | 5 | 1.0 semitone | 3 |
| PW | 5 | 200 | 3 |
| FILTER | 5 | 64 | 3 |

Run-collapse: `P_MAX = 24`, `MIN_RUN = 4`, `AC_THRESH = 0.6`. The validated reference
implementation is `/scratch/tmp/anchor_val/anchor_probe_final.py` (functions
`stream` / `pass1_origins` / `pass2_collapse` / `_periodic`) — copy its logic; it is the
spec. These params are corpus-tuning knobs, not hard constants; expose them on the pass.

## Pipeline integration (`preframr_tokens/reglogparser.py::parse`)

Current order (call sites near the end of `parse`): `_squeeze_changes` → `_combine_regs`
→ `_quantize_freq_to_cents` → `_simplify_ctrl` → … → `_add_frame_reg` → … →
`VoiceTrackPass` → `FreqTrajectoryPass`.

1. **Stash unquantized freq.** Immediately after `_combine_regs` (which assembles the
   16-bit freq) and BEFORE `_quantize_freq_to_cents`, copy the freq value into a side
   column `freq_unq` so anchor detection uses full-precision pitch (cent-quantization is
   lossy; PW/filter are not cent-quantized, read directly). Carry `freq_unq` through the
   subsequent transforms (they operate row-wise; preserve the column).
2. **Run `TrajectoryAnchorPass` immediately before `FreqTrajectoryPass`** (so the frame
   index `_frame_index(df)`, the gate/control rows, and the per-reg SET rows are all
   present and aligned with what `FreqTrajectoryPass` sees). It writes a boolean column
   **`traj_anchor`** = True on each row that begins a trajectory for its register
   (the onset frame of each final anchor, mapped to that register's SET row at that
   frame). Annotation-only: no new atoms, no token-stream change by itself.
3. **`FreqTrajectoryPass` consumes `traj_anchor`.** In `apply` / `_emit_ramps` /
   `_emit_osc_run`, force a segment boundary at every `traj_anchor` frame for that reg:
   a trajectory may not span an anchor; each segment's `V0` (or ramp/osc origin) is the
   value at the anchor frame. Where `FreqTrajectoryPass` currently starts a run at a
   value-run boundary, it must instead start at the nearest preceding anchor. Keep the
   existing ramp/osc primitive emission otherwise — anchors only re-cut the boundaries.
   If `traj_anchor` is absent (column missing), behave exactly as today (safe default,
   for callers that don't run the new pass).

`TrajectoryAnchorPass` gating: a `GATE_FLAGS = frozenset({"trajectory_anchor_pass"})`
+ `if args is not None and not getattr(args, "trajectory_anchor_pass", True): return df`
(mirror `FreqTrajectoryPass`), so it is opt-out and existing specs are unaffected until
flipped on.

## Files to add / change
- **Add** `preframr_tokens/macros/trajectory_anchor.py`:
  - `detect_anchors(value: np.ndarray, gate_on: list[int], kind: str, *, params) -> list[int]`
    — the two-pass detector (pass1 + pass2 + gate union per `kind`).
  - helpers `pass1_origins`, `pass2_collapse`, `_periodic` (autocorrelation), `_smooth`.
  - `class TrajectoryAnchorPass(MacroPass)` with `apply(df, args=None)` that, per
    register in `TRAJ_REGS` (per voice for freq/PW, global for filter), extracts the
    value+gate series (use `freq_unq` for freq), calls `detect_anchors`, and sets
    `df["traj_anchor"]` True on the anchor rows.
- **Edit** `preframr_tokens/reglogparser.py`: stash `freq_unq` after `_combine_regs`;
  call `TrajectoryAnchorPass().apply(df, args=self.args)` before
  `FreqTrajectoryPass().apply(...)`.
- **Edit** `preframr_tokens/macros/freq_trajectory_pass.py`: honor `traj_anchor` as
  forced segment boundaries (default-safe when the column is absent).
- **Bump** `fallback_version` in `pyproject.toml`.

## How to test (pytest, self-contained — no external tunes)

Add `tests/macros/test_trajectory_anchor.py`. Build **synthetic per-register write
streams** with known anchors (each is a list of `(frame, reg, val)` plus control writes
for gate), one per case below, and assert `detect_anchors` returns the expected anchor
frames within ±2 frames, with precision and recall both 1.0 on the synthetic cases.

| # | fixture | expected anchors | what it locks |
|---|---|---|---|
| a | filter cutoff monotonic slow ramp 0→2000 over 4000f, gate firing on an unrelated voice | **1** (ramp onset), off-gate | ultra-slow sweep = one ramp; FILTER intrinsic-only |
| b | filter staircase: 30 steps of +60, no gate | **1** (ramp collapse) | discrete slow sweep ≠ 30 notes |
| c | freq 3-tone arp (e.g. C,E,G repeating every 3f) under ONE held gate | **1** (arp onset) | oscillator collapse via autocorrelation |
| d | freq held note + ±1-semitone vibrato (period ~6f), one gate-on | **1** (the note) | vibrato suppressed, no spurious origins |
| e | freq stepped melody: 8 distinct held pitches, gate-on each | **8** (one/note) | level-jumps = note anchors |
| f | freq legato slide from pitch A to B mid-note (gate stays on) | **2**: the initial note + the slide onset (off-gate) | intrinsic recovers an off-gate sweep |
| g | freq repeated SAME pitch 5×, gate-on each (no value change) | **5** (one/retrigger) | gate union catches value-invisible retriggers |
| h | freq fast APERIODIC line (12 distinct pitches, dense, irregular) under one gate | **~12** (kept, NOT collapsed) | melody is not mistaken for an oscillator |

Cases (c) vs (h) are the crucial pair: both are dense/fast; (c) is periodic (collapse),
(h) is aperiodic (keep). If your autocorrelation discriminator passes both, it
generalizes.

**Integration / fidelity test:** with `trajectory_anchor_pass` on, run a fixture through
`parse` (or directly through `TrajectoryAnchorPass` + `FreqTrajectoryPass`) and assert
(1) every emitted trajectory's `V0`/origin sits on a `traj_anchor` frame, and (2) the
existing FREQ_TRAJ **byte-exact round-trip oracle still passes** (anchors only re-cut
boundaries; expand must remain lossless). Reuse the existing freq-trajectory
round-trip/fidelity test as the gate — it must stay green.

**Acceptance:** synthetic cases (a)–(h) pass at P=R=1.0 (±2f); the FREQ_TRAJ
round-trip oracle stays byte-exact; `run_tests.sh` (black, pylint, pyright, pytest,
coverage) green.

## Known tradeoffs (for the implementer)
- The oscillator/fast-melody boundary is the one genuinely fuzzy decision; the
  autocorrelation test (case c vs h) is the principled separator — prefer it over any
  count/regularity heuristic. `AC_THRESH` is the dial.
- Gate-union is register-specific: FREQ/PW yes, FILTER no. Do not union gate for filter.
- Validation against an *external tracker* (defMON/SID-Wizard/non-tracker) was done in
  the upstream prototype and is **not** required here; note that tracker "ground truth"
  counts every driver re-init (e.g. 799 arp re-targets, 127 filter slide re-arms), which
  is *finer* than the observable note/sweep granularity this pass targets — so do not
  chase per-driver-reinit recall; target note/sweep level (the synthetic fixtures encode
  the correct granularity).

## Upstream context (why this matters; informational)
The melodic content a generalizing model must learn is the gate-/sweep-anchored base
trajectory; the current value-run segmentation discards it. Tracker models (defMON,
SID-Wizard) and a non-tracker tune (*A Mind Is Born*) confirmed: notes/sweeps are
anchored at observable value-origins and gate-retriggers, frequently **off-gate** for
PW/filter; an intrinsic value-origin detector ∪ gate recovers them (recall ~1.0,
including off-gate sweeps) once the two-pass ramp+oscillator collapse restores precision.
This pass is the tokenizer-side fix; the consuming model change is out of scope here.
