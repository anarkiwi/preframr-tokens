# WORK ORDER: the generator-MDL tokenizer pipeline (lossless, zero-residual, driver-agnostic)

**Audience:** an engineer/agent working **entirely inside `/scratch/anarkiwi/preframr-tokens`**. Self-contained
— everything you need is in this repo + the HVSC corpus at `/scratch/preframr/hvsc` + the prototype and
design refs below. Do **not** touch any other repo. **Do** take it all the way to a pushed PR.

**Start clean:** `git fetch origin && git switch -c generator-pipeline origin/main`. Do **not** commit this file.

**Run non-GPU work on `fogbank`** (`ssh fogbank`, 72 cores, shared `/scratch`, the baked images). Iterate the
corpus gate with a coarse `--step`, gate on the full run. **`reparse=True` on every residual/byte-exact
measurement** (stale `.pq` caches lie).

---

## 0. What you are building (one paragraph)

Replace the per-register macro zoo with **ONE uniform generative model of every SID write**, derived as an
MDL problem. Every value-channel's per-frame series is a sequence of **generators** drawn from a complete,
minimal set — **`{HOLD, ACCUM, SWEEP, TABLE}`** — reused via a **DEF→REF bank**, with pitch expressed over a
**unified per-tune semitone LUT** (transposition-invariant arp reuse with the exact freq residual carried in
the codebook entry, **no cents quantization**). The escapes are
typed, not unexplained: an **END** marker for the truncated final frame, and a **SAMPLE** modality for digi
PCM (detected at raw-write density, kept out of the per-frame model). The deliverable is **byte-exact,
residual-zero (no raw `SET`, no escape-hatch op), corpus-wide**, with the obsolete/conflicting freq+codebook
passes removed.

**This design is already prototyped and proven** (read these first — do not re-derive):
- Design spec + evidence: `/scratch/anarkiwi/preframr-xpt/design/generator_mdl_representation.md`.
- **The fitter is embedded VERBATIM in §1A** — implement that; you do NOT need any file outside this repo.
  (`/scratch/tmp/decompose5.py` is the validated original it was ported from, and `swm_suite.py`/`defmon_test.py`/
  `measure_lut.py` are the cross-driver proofs — background only, on shared `/scratch`, not required to build.)
- Driver grounding: `/scratch/anarkiwi/preframr-xpt/design/sid_driver_ornament_reference.md`,
  `digi_detection_reference.md`, `encoding_principles.md`.

**Proven (the bar you must meet or beat):** 100% byte-exact on 1580 corpus tunes; **0 truly-bare events**
(every EVENT is final-row truncation); 91/91 SID-Wizard example modules + 9/9 defMON player fixtures + 46/46
DefMon corpus tunes byte-exact, 0 bare; the historically-hard RESID engines (Baggis/JCH, SoundMonitor,
System6581, Commando, Camerock) all lossless. The note-relative LUT collapses distinct freq shapes by 64%.

---

## 1. The representation (precise spec)

### 1.1 Channelize (from `register_state`)
Build `S = register_state(df)` (`audit_primitives.register_state`, `(n_frames, 25)` per-frame settled regs).
The exact channel list + decode is §1.2 (it supersedes any other description); the 25 SID regs split into the
generator pass's 13 channels + ctrl/AD/SR (owned by `InstrumentProgramPass`, not the generator pass).

### 1.2 Channels (exact list + exact decode — no decisions left)
The generator pass owns these per-frame integer channels. **ctrl/AD/SR are NOT here** — they stay with
`InstrumentProgramPass`, which already drives them to residual-zero (do not touch them). For `b∈{0,7,14}`
(voice base):
- **freq_v** ×3 — 16-bit combined `reg[b] | reg[b+1]<<8`. Decode a value `f` to two writes per changed frame:
  `reg[b]=f&0xFF`, `reg[b+1]=(f>>8)&0xFF`.
- **pw_lo_v** (`reg b+2`) and **pw_hi_v** (`reg b+3`) ×3 — TWO separate 8-bit scalar channels per voice. Do
  NOT combine to 12-bit: `$D403`'s upper nibble is unused but `register_state` keeps the literal byte, so
  decode each reg as its own byte.
- **cut_lo** (`21`), **cut_hi** (`22`), **res** (`23`), **modevol** (`24`) — 8-bit scalar channels.

**13 channels.** Each is decomposed independently; each atom targets exactly its reg(s). Build them with the
exact `channels()` in §1A.

### 1.3 Generators — THREE ops, only ONE genuinely new
Decompose each channel left-to-right with the **self-verifying longest-wins** fitter (§1A — embed verbatim;
each candidate is accepted only for the longest prefix its OWN decoder reproduces, so reconstruction==source
by construction). **HOLD is ACCUM with Δ=0**, so the set is three ops — and **two already exist** (the
generator pass becomes their producer; the standalone passes are deleted in §4, the ops retained):

| generator | op | status | atom = subreg-tagged rows under the op (mirror SweepPass/GlobalOscPass exactly) |
|---|---|---|---|
| **HOLD / ACCUM** (Δ=0 ⇒ HOLD) | `SWEEP_OP=64` REUSE | keep `SweepDecoder` + `SweepPass._sweep_rows` (linear, no `period`) | `START_HI, START_LO, DELTA_HI, DELTA_LO, LEN` |
| **TABLE** (period-P cycle, 2≤P≤24) | **codebook (new, §2.2)** | DEF carries the cycle (mirror `GlobalOscPass._osc_rows`), REF carries id+base | DEF: `PERIOD, STATE_BASE+0..P-1, LEN`(+freq: base_note+offsets+resids); REF: `id, base_note, LEN` |
| **TRIANGLE** (bounded reversing zigzag) | `GEN_TRI_OP=83` **NEW** | new decoder + contract | `START_HI, START_LO, STEP, LO_HI, LO_LO, HI_HI, HI_LO, DIR, LEN` |

Value width: **freq is 16-bit** (use the `*_HI/*_LO` subregs); the **8-bit scalar channels set `*_HI=0` and
use `*_LO` only**. `STATE_BASE+m` for a 16-bit cycle entry uses two consecutive subregs (lo,hi). Retire
`GLOBAL_OSC_OP=82` (its job is the TABLE codebook now). **`OP_PRODUCER[SWEEP_OP]` → `"GeneratorPass"`.**

### 1.4 Pitch reuse via the unified LUT (the ONLY freq-specific rule; lossless, no cents quantization)
The LUT does exactly ONE job: make **`TABLE` cycles on freq channels transposition-invariant** so arps/chords
reuse one codebook entry across pitches (measured −64% distinct shapes). It does NOT split the freq stream and
NEVER reads the waveform.
- **TUNING atom** (`GEN_TUNING_OP=84` NEW; one per tune, emitted first): carries `ref_q = round(ref*256)`
  (0..255), `ref` = circular mean of `frac(12·log2(f))` over all freq frames with `f>8`, all 3 voices (§1A
  `tune_ref`). The decoder stores `ref_q` for the tune.
- **freq `TABLE` codebook key = note-relative.** For a freq cycle `c[0..P-1]` key on
  `(tuple(note(c[k])−note(c[0])), tuple(resid(c[k])))`, with `note(f)=clip(round(12·log2(f)−ref),0,95)`,
  `recon(n)=round(2^((n+ref)/12))`, `resid(f)=f−recon(note(f))` (§1A). DEF stores `base_note=note(c[0])`, the
  offset cycle, the residual cycle; REF stores the id + this instance's `base_note`. **Decode:**
  `c[k] = recon(base_note + offset[k]) + resid[k]` — exact. Two transposed clean arps (same offsets, resids
  all 0) share the entry, differing only by `base_note` in the REF.
- **HOLD/ACCUM/TRIANGLE on freq stay raw 16-bit** (slides/vibrato/holds are per-instance — no LUT, no split).
  **All non-freq channels: absolute keying** (raw cycle bytes), no LUT.
- **Learnability caveat (measure — the codebook reuse IS the learnability lever, not the byte count).** The
  −64% reuse was measured on note-offsets ALONE; keying on (offsets, **residuals**) refragments arps that
  share offsets but differ in residual (vibrato/detune). Clean driver arps write exact LUT values (resid=0) so
  they still collapse, but **measure the refragmentation** (distinct `GEN_TABLE` freq keys with vs without the
  residual in the key). If material, key on note-offsets only and carry the residual on a separate companion
  stream. See `preframr-xpt/design/learnability_token_ordering_theory.md` "Compatibility with the generator-MDL
  pipeline" — copy-fraction, not raw MDL, is the objective.

**Facemorph guardrail (load-bearing): nothing above reads the waveform bit.** Noise is used for onset accents
on *pitched* notes (Facemorph v0: a 1-frame freq≈213 NOISE burst INSIDE a pulse sweep) and pulse for
percussion — waveform is NOT a pitch signal and must never gate routing. A low/swept voice whose freqs fall
below the LUT floor (Facemorph v0, freq 54–213) simply yields a degenerate note range + large residuals —
still byte-exact, and its sweep is a clean raw `ACCUM`. Verified: the prototype reads the waveform bit nowhere
and is byte-exact + 0-bare on `MUSICIANS/W/Wiklund/Facemorph.1` incl. every noise-tik frame. (Add the property
test in §7.2: permuting a freq-only frame's waveform nibble must not change the emitted tokens.)

### 1.5 The two non-generator cases — fully resolved (no EVENT op, no SAMPLE op)
- **No END/EVENT op.** `ACCUM` fits ANY two consecutive points (a line through 2 points), so the fitter
  returns a length-≥2 generator at every position **except the final frame** `i=F−1`. A length-1 atom can
  therefore occur ONLY at the last frame — encode it as `SWEEP_OP` with `LEN=1` (a degenerate HOLD). The
  residual-zero gate asserts **every length-1 atom starts at frame F−1** and there are **no raw `SET`s**.
  That is the entire "tail" story — there is no `GEN_END`.
- **No SAMPLE op (this PR).** Digis are excluded upstream by `is_digi`/`filter_dump_paths` (§5 hardens the
  detector), so a digi never reaches the generator pass. SAMPLE-as-typed-PCM is a future modality, NOT built here.

### 1A. The canonical fitter — EMBED VERBATIM as `macros/generator_fit.py` (torch-free, no `#` comments)
This is the contract; do not re-derive. (Ported from the validated `decompose5.py`; `recon`/`note_of` must be
the SAME functions used by both encoder and decoder so residuals are bit-exact.)
```python
import math
import numpy as np
_MAXP = 24
_MINTRI = 6
_FBASE = 16777216.0 / 985248.0
def _lut(ref):
    return np.array([min(65535, int(round(2.0 ** ((n + ref) / 12.0) * _FBASE * 16.0)))
                     for n in range(96)], dtype=np.int64)
def tune_ref(freqs):
    f = np.asarray([x for x in freqs if x > 8], dtype=np.float64)
    if len(f) < 16:
        return 0.0
    frac = (12.0 * np.log2(f)) % 1.0
    ang = 2.0 * math.pi * frac
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2.0 * math.pi)) % 1.0
def recon(note, ref):
    return int(round(2.0 ** ((note + ref) / 12.0) * _FBASE * 16.0))
def note_of(f, ref):
    if f <= 0:
        return 0
    return int(max(0, min(95, round(12.0 * math.log2(f) - ref))))
def gen_hold(s, i, n):
    v = s[i]; j = i
    while j < n and s[j] == v:
        j += 1
    return j - i
def gen_accum(s, i, n):
    if i + 1 >= n:
        return 1, 0
    d = s[i + 1] - s[i]; j = i + 1
    while j + 1 < n and s[j + 1] - s[j] == d:
        j += 1
    return j + 1 - i, d
def gen_table(s, i, n, p):
    if i + p >= n:
        return 0
    j = i + p
    while j < n and s[j] == s[j - p]:
        j += 1
    ln = j - i
    if ln < max(3, p + 1) or len(set(s[i:i + p])) < 2:
        return 0
    return ln
def _tri_seq(start, step, lo, hi, dir0, ln):
    out = [start]; cur = start; d = dir0
    for _ in range(ln - 1):
        nxt = cur + step * d
        if nxt > hi or nxt < lo:
            d = -d; nxt = cur + step * d
        cur = nxt; out.append(cur)
    return out
def gen_tri(s, i, n):
    if i + 1 >= n or s[i + 1] == s[i]:
        return None
    step = abs(s[i + 1] - s[i]); dir0 = 1 if s[i + 1] > s[i] else -1
    cur = s[i]; d = dir0; j = i; hit, lot = [], []
    while j + 1 < n:
        if s[j + 1] == cur + step * d:
            cur += step * d
        elif s[j + 1] == cur - step * d:
            (hit if d > 0 else lot).append(cur); d = -d; cur += step * d
        else:
            break
        j += 1
    if not (hit and lot) or len(set(hit)) != 1 or len(set(lot)) != 1:
        return None
    lo, hi = min(lot), max(hit)
    seq = _tri_seq(s[i], step, lo, hi, dir0, n - i)
    k = 0
    while k < len(seq) and s[i + k] == seq[k]:
        k += 1
    if k < _MINTRI:
        return None
    return k, (step, lo, hi, dir0)
def fit_run(s, i):
    n = len(s)
    best = ("HOLD", gen_hold(s, i, n), None)
    al, d = gen_accum(s, i, n)
    if d != 0 and al > best[1]:
        best = ("ACCUM", al, d)
    tr = gen_tri(s, i, n)
    if tr and tr[0] > best[1]:
        best = ("TRI", tr[0], tr[1])
    for p in range(2, _MAXP + 1):
        tl = gen_table(s, i, n, p)
        if tl > best[1]:
            best = ("TABLE", tl, p)
    return best
def decompose(s):
    n = len(s); out = []; i = 0
    while i < n:
        kind, ln, params = fit_run(s, i)
        out.append((kind, i, max(1, ln), params)); i += max(1, ln)
    return out
```
`channels(S)` (S = `register_state`, shape `(F,25)`): emit the 13 channels of §1.2 as python int lists
(`freq_v = (S[:,b]+256*S[:,b+1]).tolist()`, `pw_lo_v=S[:,b+2].tolist()`, etc.). **Lossless invariant the gate
checks:** for every channel, replaying each atom's decoder over its span reproduces the source list exactly.

---

## 2. Byte-exact wiring into the existing framework

You are **not** inventing a new lossless contract — reuse the arbiter's. Each generator instance is a
`Claim` (`macros/arbiter.py:19`) spanning `[st, st+ln)` on its channel's reg(s), whose `writes` are the
reconstructed per-frame values. `arbitrate(df, claims, validate=True)` (`arbiter.py:104`) drops any claim
that changes the decoded `register_state`. **Because `fit_run` self-verifies, every claim is byte-exact and
the arbiter never drops** — but keep `validate=True` as the standing guard (and run a corpus
`PREFRAMR_ARBITER_STRICT` pass to prove zero drops). `_decoded_state`/`_lossless` (`arbiter.py:41,50`) are the
oracle.

### 2.1 New ops (the EXACT list — nothing else is new)
Reuse `SWEEP_OP=64` for HOLD/ACCUM (keep `SweepDecoder`; repoint `OP_PRODUCER[64]="GeneratorPass"`). Retire
`GLOBAL_OSC_OP=82` (deleted with `GlobalOscPass`). Add exactly these in the free range (never renumber):

| op | id | role / decoder |
|---|---|---|
| `GEN_TRI_OP` | 83 | ATOM; new `GenTriDecoder.expand` replays `_tri_seq(start,step,lo,hi,dir,len)` |
| `GEN_TUNING_OP` | 84 | ATOM; `GenTuningDecoder` stores `ref_q` on the decode state, emits NO register write |
| `GEN_TABLE_DEF` | 85 | CODEBOOK_DEF |
| `GEN_TABLE_STEP` | 86 | CODEBOOK_STEP |
| `GEN_TABLE_END` | 87 | CODEBOOK_END |
| `GEN_TABLE_REF` | 88 | CODEBOOK_REF |

Subreg constants: add `GEN_TRI_SUBREG_{START_HI,START_LO,STEP,LO_HI,LO_LO,HI_HI,HI_LO,DIR,LEN}` and reuse
`SWEEP_SUBREG_*` for ACCUM and the `GLOBAL_OSC_SUBREG_*` naming convention (PERIOD/STATE_BASE/LEN) for the
TABLE DEF rows. For each new op wire the full blast radius: `stfconstants.py` (`*_OP` + subregs), a
`MacroDecoder` subclass + `DECODERS` entry in `decoders.py`, an `OpContract` in `op_contracts.py:_CONTRACT_LIST`
(+ `OP_PRODUCER`/`reference_op_producers()` for the DEF/REF), emit in `GeneratorPass` (§3). The completeness
tests (`op_contracts.missing_contracts`, `test_op_contracts`, `test_codebook_registry`, `test_macro_contracts`)
name anything missed — run them first.

### 2.2 The `GEN_TABLE` codebook family (exact encoding)
Add a 4th `CodebookFamily` (`macros/codebook.py:84`, alongside stamp/wavetable/instrument), `name="generator"`,
`def_op=GEN_TABLE_DEF`, `step_ops=(GEN_TABLE_STEP,)`, `commit_op=GEN_TABLE_END`, `ref=RefSpec(GEN_TABLE_REF,
id_subreg=..., table_less=False)`. Ids are **tune-local ordinals** (memory `codebook-id-snap-corruption`: a
codebook id is a positional pointer, NEVER value-snapped). Reuse `codebook_emit.emit_recurring`
(`codebook_emit.py:12`); `InstrumentProgramPass` is the working template, `StampPass` the structural one.
Concrete entry contents:
- **non-freq channel cycle** `c[0..P-1]` (8-bit): DEF emits `PERIOD`, then `STATE_BASE+m = c[m]` (mirror
  `GlobalOscPass._osc_rows`), then `LEN`; **bank key = `("abs", reg, tuple(c))`**. REF emits `id, LEN`.
- **freq channel cycle** (16-bit, note-relative): DEF emits `PERIOD`, `BASE_NOTE=note_of(c[0],ref)`, then per
  entry `OFFSET=note_of(c[m],ref)-BASE_NOTE` and `RESID_HI/RESID_LO = recon(...)-relative residual`, then
  `LEN`; **bank key = `("note", tuple(offset), tuple(resid))`**. REF emits `id, BASE_NOTE, LEN`. Decode:
  `c[m] = recon(BASE_NOTE+offset[m], ref) + resid[m]`. (Two transposed clean arps share one entry.)
Decode replays `c[k % P]` for `LEN` frames into the channel's reg writes.

---

## 3. PART A — build `GeneratorPass`

A new `macros/generator_pass.py`, `class GeneratorPass(MacroPass)`,
`GATE_FLAGS = frozenset({"generator_pass"})` (default OFF until it gates clean), run **inline on the actual
register channels** (the voice-confusion guardrail — same as InstrumentProgramPass). It is a UNIFICATION of
`SweepPass` (ACCUM) + `GlobalOscPass` (TABLE) over all 13 channels with the §1A longest-wins fitter — reuse
their `_collect_writes` real-frame accounting, `_chunk_starts` `MAX_SPAN` chunking, and `Claim`/`arbitrate`
machinery. Algorithm:
1. `S = register_state(df)`; `ch = channels(S)` (§1A — the 13 channels, no waveform read). Compute
   `ref = tune_ref(all freq frames)`; emit ONE `GEN_TUNING_OP` row with `ref_q` at the head.
2. Per channel: `gens = decompose(series)`; per generator build a `Claim` over its reg(s)/frames with the
   reconstructed per-frame writes — HOLD/ACCUM→`SWEEP_OP` rows (`_sweep_rows`, no period), TRI→`GEN_TRI_OP`
   rows, TABLE→`GEN_TABLE` DEF (first sight of its bank key) or REF (§2.2). A length-1 atom (only possible at
   `i=F−1`) → `SWEEP_OP` `LEN=1`.
3. `arbitrate(df, claims, validate=True)` (self-verifying fitter ⇒ never drops; the guard stays).
- **Test through the real `RegLogParser.parse()`** (memory `test-through-real-parse`), not synthetic dfs.
- **Self-audit:** add `tests/test_generator_residual_zero.py` asserting, on a real-parse corpus sample
  (`reparse=True`, digi-excluded): **raw `SET` == 0 on all 13 channels incl. FREQ**; **every length-1 atom
  starts at frame F−1** (no interior length-1); and `register_state(your parse) == register_state(origin/main
  parse)` per tune. ZERO is the gate; never accept "almost".

### Phasing (each independently green + byte-exact)
- **A1 (scalar channels):** pw_lo/pw_hi (×3), cut_lo, cut_hi, res, modevol through HOLD/ACCUM(`SWEEP_OP`),
  TRI(`GEN_TRI_OP`), TABLE(codebook, absolute key). Gate: those reg classes → raw `SET` 0, byte-exact.
- **A2 (freq):** the three 16-bit freq channels; `GEN_TUNING` + note-relative `GEN_TABLE` keying for freq
  arps; HOLD/ACCUM/TRI on raw 16-bit. Gate: FREQ raw `SET` 0, byte-exact; residual-zero matches the prototype.
- **A3 (predict-window locality):** register `GeneratorPass` in `macros/__init__.py:block_refire_passes` so the
  existing block-refire machinery re-fires it per self-contained block, making the codebook block-local. **Do
  NOT invent a new anchor/block segmentation — reuse the repo's existing blocks.** Lossless gate is unchanged
  (the pass is correct whole-tune); A3 only improves DEF→REF locality. Report gen-tokens/block vs the 2048
  window as a measurement, not a blocker.

---

## 4. PART B — remove the obsolete / conflicting macros

The generator model **subsumes** the entire freq + sweep + codebook stack. Once `generator_pass` gates clean
and is in `REGISTERED_MACROS` (§6), **delete** the subsumed passes (study PR #57 `47af114` — the ctrl-macros
deletion — as the worked blast-radius example). Per pass: remove the file, its `*_OP`(s) from `stfconstants.py`
(**leave op-id holes, never renumber**), its `OpContract`/`OP_PRODUCER` rows, its decoder + `DECODERS` entry +
`__all__`, its entries in `FREQ_BLOCK_PASSES`/`PASSES`/`POST_NORM_PRE_VOICE_PASSES` + imports in
`macros/__init__.py`, its inline instantiation in `reglogparser.py:983`, its `REGISTERED_MACROS` entry, its
`flag_registry` `FLAG_REQUIRES`/`FLAG_CONFLICTS` rows, and its tests.

**DELETE the pass; the "op" column says what happens to its op-ids:**
| pass / file | ops | why subsumed |
|---|---|---|
| `freq_trajectory_pass.py` (FreqTrajectoryPass) | free 45,46 (holes) | freq trajectories = HOLD/ACCUM/TRI/TABLE on the freq channel |
| `skeleton_pass.py` (SkeletonPass) + `wavetable_pass.py` | free 54,55,65–69 (holes) | SKEL/ORN/WAVETABLE = freq generators + the `GEN_TABLE` codebook |
| `sweep_pass.py` (SweepPass) | **RETAIN `SWEEP_OP=64`** (GeneratorPass is its new producer) | delete the PASS only; ACCUM reuses its op + `SweepDecoder` |
| `gradient_pass.py` (GradientPass) | free `GRADIENT_OP=76` (hole) | staged automation = HOLD/ACCUM stages |
| `global_osc_pass.py` (GlobalOscPass) | **RETIRE `GLOBAL_OSC_OP=82`** (hole) | periodic cycle = the `GEN_TABLE` codebook |
| `preset_pass.py` (PresetPass) | free 35,36,40 (holes) | wide-val presets = HOLD/`GEN_TABLE` on pw/cutoff |
| `stamp_pass.py` (StampPass) | free 56–59,63 (holes) | recurring spans = `GEN_TABLE` codebook |
| `per_reg_burst.py` (PerRegBurstPass) | — | single-reg bursts = generators |
| `trajectory_anchor.py` (TrajectoryAnchorPass) | — | annotation for FreqTraj (now gone) |
| `note_off_pass.py`, `init_pass.py` | free 71,75,77 (holes) | gate/init writes are ordinary generator atoms |

**KEEP (orthogonal — not pitch/automation content):** `LoopPass` (`loop_pass`/`loop_transposed`; cross-block
pattern repetition — composes with blocks), `HardRestartPass` (`hard_restart_pass`/`env_multiload`; gate-pair
collapse), `LegatoPerClusterPass`, `TransposePass`, `SubregPass`, `DedupSetPass`, `VoiceBlockOrderPass`
(`voice_canonical_block_order`), `CoarsenPass`, and **`InstrumentProgramPass`** (decision #4: it OWNS
ctrl/AD/SR — already residual-zero — and runs BEFORE `generator_pass`, which never touches those three regs;
this preserves the shipped timbre codebook + its DEF→REF reuse).
**Sequencing the deletion (no transient conflict):** register `FLAG_CONFLICTS` so `generator_pass` excludes
`freq_trajectory_pass`/`skeleton_pass`/`sweep_pass`/`gradient_pass`/`global_osc`/`preset_pass`/`stamp_pass`/
`wavetable_pass`/`per_reg_burst` BEFORE deleting them (so a transient `full_macros` resolves), then drop those
dead conflict rows together with the passes.

**Grep clean** when done (zero dangling refs):
`grep -rn 'FREQ_TRAJ\|TRACK_REF\|SkeletonPass\|SKEL_OP\|ORN_OP\|SweepPass\|SWEEP_OP\|GradientPass\|GLOBAL_OSC\|PresetPass\|PWM_PRESET\|FC_PRESET\|StampPass\|STAMP_\|WavetablePass\|WAVETABLE_\|PerRegBurst\|TrajectoryAnchor\|NoteOffPass\|NOTE_OFF\|InitPass\|INIT_OP' preframr_tokens tests` → empty (modulo KEEP items).

---

## 5. PART C — fix digi detection (confirm exclusion; no SAMPLE op this PR)

**State of the code (verified):** `dump_meta._build_meta_from_raw` (`dump_meta.py:51`) already computes
`vol/ctrl/freq/pw` writes-per-frame **on the raw df** (the correct sub-frame granularity) and
`is_digi = vol_max>=40 or ctrl_max>=20 or pw_max>=40` (`:93`) — so the PWM gap the old notes flag is
**partly closed** (`pw_max>=40`). Your job:
1. **Validate** against the digi taxonomy (`digi_detection_reference.md`): `$D418`/Mahoney (vol), SounDemoN
   (ctrl), **PWM** (pw on a pulse+test+freq≈0 voice). Confirm `pw_max>=40` catches real PWM digis without
   false-positiving on PW *sweeps* (a sweep writes pw ~once/frame ⇒ `pw_max`≈1, far below 40 — safe). On a
   corpus sample, list tunes flipped by each clause; spot-check a PWM tune (e.g. a Hannula/Sledgehammer-class
   tune) is caught.
2. **Refine if needed:** gate the PWM clause on the voice config (`ctrl` pulse+test bit set, freq≈0) to be
   defensible, per the reference's "PWM on a pulse+test+freq≈0 voice"; optionally add the Mahoney all-3-voice
   `ctrl=$49`+`SR=$FF`+filter fingerprint as a distinguisher. Keep row-count OUT of the signal (memory
   `digi-detection`; Baggis is NOT a digi).
3. **Confirm exclusion (NO SAMPLE op this PR):** `is_digi` tunes are a separate sub-frame PCM modality — a
   digi's information lives below the 50 Hz `register_state` resolution, so the per-frame generator model is the
   wrong tool (proven: a register-state density detector fires 0). Confirm the corpus/parse + residual-zero gate
   path **excludes `is_digi`** (it already does via `filter_dump_paths`); add an assertion that a known digi is
   not fed to `generator_pass`. A typed `SAMPLE`/PCM modality is explicitly **out of scope for this PR** —
   record it as a future follow-up; do not build a SAMPLE op.

Add `tests/test_digi_detection.py` (through `_build_meta_from_raw` on real dumps): each digi method's signature
flips `is_digi`; a melodic tune (ctrl_max≤3) and Baggis do not.

---

## 6. PART D — make it the deployed default (the swap)

In `tokenizer_config.py:REGISTERED_MACROS`: **remove** the deleted flags (`freq_trajectory_pass`,
`sweep_pass`/`pw_sweep`/`filter_sweep`, `filter_gradient`, `global_osc`, and `preset_pass`/`stamp_pass`/
`wavetable_pass`/`skeleton_pass` if any were listed) and **add** `"generator_pass"` (+ `instrument_program`
stays). Confirm `named_config("full_macros")` resolves with no `FLAG_CONFLICTS` (you removed the conflicting
flags). `macro_flag_names()` derives dynamically — the new flag auto-registers via `GATE_FLAGS`.

---

## 7. Acceptance gate (all must pass before the PR)

1. **Residual-zero, byte-exact, corpus-wide** (`reparse=True`, digi-excluded, full corpus on fogbank):
   raw `SET` == 0 on **every** register class incl. FREQ; **0 soft-residual** (no escape-hatch op); every
   length-1 atom starts at frame `F−1` (no interior length-1); `register_state` of the new default == decode of
   the new tokens, per tune (the `cb_div`/`PREFRAMR_ARBITER_STRICT` audit shows **zero dropped/diverged claims**).
2. **Cross-driver compatibility** (the user-required proofs — port the xpt scripts as repo tests):
   the SID-Wizard 1.94 example modules and the pydefmon fixtures, rendered through their players to
   register_state, decode byte-exact with 0 bare (`swm_suite.py`/`defmon_test.py` logic, as
   `tests/test_cross_driver_*` — fetch fixtures via the existing `sidwizard-driver`/`pydefmon` caches; **skip
   cleanly if a fixture is unavailable**, never silent-pass).
   - **Waveform-mixing guardrail test (Facemorph):** assert `MUSICIANS/W/Wiklund/Facemorph.1` (a noise-tik
     accent on a non-percussion voice) is byte-exact + 0-bare, and that **no pass reads the waveform bit to
     route pitch** (the encoder must be byte-identical if you permute the waveform nibble of a freq-only
     frame — a property test). Also include a pulse-percussion tune (drum/bass on a pulse voice) to prove the
     converse. These pin the "never classify pitch by waveform" invariant.
   - **Module↔macros round-trip (§7B) — BOTH directions gate, equivalence = SAME OUTPUT:** Tier 1 (forward:
     module→dump→macros→decode, output-equal) and Tier 2 (reverse: macros→emit module→re-render, output-equal
     to the original via `sid_frame_diff`/WAV audition). Any construct that can't be made to render identically
     is a named, logged gap.
3. **No-singleton gate:** extend `tests/test_whole_chip_no_singleton_set.py` to forbid **all** raw `SET`
   (incl. FREQ) and assert the only length-1 atom per channel starts at frame `F−1`.
4. **Repo suite green:** `./build.sh` (black + pylint + pyright + pytest `-n auto --dist worksteal` +
   cov≥85), all Py versions. Focused tests for every new op/pass, **through real `parse()`**, xdist-chunked
   long sweeps; lint forbids non-directive `#` comments + >5-line docstrings.
5. **Learnability triage (direction check, not a hard gate):** run `learnability_triage.py --mode blocks
   --seq_len 8192` (preframr-xpt) on the generator encoding; its **in-block induction-copy must beat the old
   codebook arm's 0.718** (else the note-relative `GEN_TABLE` keys don't recur in-block — apply the §1.4
   residual-keying fix). This is the go/no-go the theory doc flags for any codebook-heavy encoding.
6. **Parse-perf:** the generator pass replaces N passes with 1 — confirm parse wall-time does not regress
   vs `origin/main` on a fixed sample (the suffix-decode/snapshot trick in `design/parse_perf_block_reencode.md`
   if the per-claim validate re-decode dominates; memory `parse-slow-decode-walker`).

---

## 7B. Round-trip fixture tests (REQUIRED) — original tracker module ↔ macros

The strongest correctness evidence is a full loop anchored to **real native tracker modules**, not just
captured dumps: **module → render to dump → parse the dump through your macros → decode the macros back to a
dump → reconstruct an equivalent module.** Add `tests/test_roundtrip_swm.py` and `tests/test_roundtrip_defmon.py`.
These import the tracker reimpls as **test-only deps** (`pysidwizard`, `pydefmon` — both pip-installable; add
to a `[test]`/`dev` extra in `pyproject.toml`). **Skip cleanly** (pytest.importorskip / skip-on-missing-cache)
if a package or its fixture cache is unavailable — never silent-pass. Fixtures: the SID-Wizard 1.94 example
`.swm` modules (via pysidwizard's `tests/_swm_cache.py:swm_path(name)`, which extracts + SHA-verifies from the
cached `SID-Wizard-1.94-with-sources.tar.gz`) and the defMON `.prg` tunes (via pydefmon's
`tools/fetch_fixtures.py` from `defmon-withtunes.d64`).

**Fixture set — use ALL of them, and ASSERT feature coverage (do not cherry-pick):** run **every** SWM module
in the tarball (`SID-Wizard-1.94/**/*.swm`, **91 modules** — the same set xpt `swm_suite.py` validated 91/91)
and **all 9** pydefmon `.prg` tunes. These were chosen because their UNION exercises the entire driver feature
surface — for SWM: `wf_table` arp, `pw_table`, `filter_table`, `vibrato`+`vib_delay`, `hard_restart`,
`gateoff_fx`, `octave_shift`, `chord_table`, `multispeed`, `funktempo`, `tempo_override`, `transpose`, +131
distinct per-row FX codes; for defMON: pitch-mod (vib/slide/arp), `PS` pw-sweep, `ACID` cutoff, `RE` routing,
resonance, filter-mode, test-bit/HR, gate-retrig, sidTAB waveform-walk. **Port `swm_suite.py`'s feature
detector and assert the aggregate coverage set is non-empty for each feature** (so a future fixture change that
drops a feature fails loudly). If any one of the 91+9 is not byte-exact / not 0-bare, the gate fails — no
sampling.

**Helper — module → raw dump (shared by both):** loop `player.play_frame()` (SWM: `SWMPlayer`, defMON:
`DefmonPlayer`); for frame `i`, each returned `(reg, val)` becomes a raw-dump row
`{clock: i*irq, irq: PAL_FRAME_CYCLES, chipno: 0, reg: reg&0x1f, val}` (defMON `reg` is absolute `$D4xx` →
subtract `0xD400`). Write to a temp `.dump.parquet` in the canonical schema `(clock, irq, chipno, reg, val)`
and parse with `RegLogParser(args=named_config("full_macros"|generator config)).parse(tmp, reparse=True)`.

**Equivalence criterion (the definition — use it everywhere below).** Two tunes/modules are equivalent
**iff they produce the same OUTPUT** — NOT binary-identical modules, NOT a construct-by-construct match. "Same
output" = the repo fidelity oracle `sid_frame_diff.diff_dump_vs_pipeline` (CTRL/AD/SR/RES_FILT(23)/MODE_VOL(24)
byte-exact + FREQ within cent tolerance, frame-aligned), with the 12-SID WAV audition as the ultimate arbiter.
So the emitted module may use entirely different instruments/tables/commands and still be equivalent, as long
as it renders to the same registers. A construct only counts as a gap if it changes the output.

### Tier 1 — forward round-trip (ENFORCEABLE, must pass)
`module → play_frame → raw dump D1 → parse(generator_pass) → tokens T → decode(T) → register_state R2`,
assert **`R2` == `register_state(D1)`** under the equivalence oracle (here strict byte-exact, since the macros
are lossless). Proves the macros preserve a *real tracker module's output* end-to-end. (Decode = the existing
`register_state`/`DECODERS` inverse; the byte-exact arbiter contract — anchored to a native module.) Also
assert **0 raw `SET`** and that the only length-1 atom per channel starts at frame `F−1`.

### Tier 2 — reverse round-trip: macros → module → SAME OUTPUT (ENFORCEABLE)
This is the user-required "reverse to an equivalent tune," and equivalence is **same output**, full stop.
**Emit a native module `M2` from the recovered macros**, RE-RENDER it, and assert its output equals the
original: `T → emit M2 → play_frame → dump D3`, assert **`register_state(D3)` ≡ `register_state(D1)`** under the
equivalence oracle. SWM via `pysidwizard.writer` (`SWMFile` → `write_swm`); defMON via `DefmonSong` setters →
`to_file`/`to_bytes`. Because equivalence is output-only, the emitter is **free to choose constructs**: map a
generator to the natural native construct when it inverts cleanly (`GEN_TABLE`→wavetable/arp, `ACCUM`→slide,
`SWEEP`→vibrato/PW), and for anything else **table-encode the recovered per-frame register stream directly**
(both engines' wavetables can drive arbitrary per-frame register sequences) so the output still matches. A
construct that cannot be made to render identically is a **NAMED gap** logged by the test (never a silent
pass) — `xfail` it with the reason, fix or document. Run the WAV audition on a sample of `M2` vs `M1` as the
behavioral backstop.

### (Optional, non-gating) structural diagnostic
For insight only — NOT an equivalence check — report how many recovered generators mapped to the *original*
module's own constructs (e.g. `GEN_TABLE` cycle == that instrument's `wf_table` arp offsets / sidTAB transpose;
`ACCUM`==its portamento; codebook bank size vs the module's distinct-table count). High overlap is evidence the
inversion recovered the tracker's intent; it does not gate (equivalence is output, §criterion).

## 8. PR

`git push -u origin generator-pipeline`; open the PR with `gh pr create`. **Title:** "feat(tokens): unified
generator-MDL pipeline (lossless, residual-zero, driver-agnostic) + remove subsumed freq/codebook macros +
digi-detection hardening". **Body:** the design one-paragraph (§0), the acceptance-gate numbers (corpus
byte-exact %, residual-zero proof, SWM/defMON cross-driver pass, the **module↔macros round-trip** (§7B,
both directions, equivalence = same output) results + any named reverse-emit gaps, removed-pass list with
freed op-ids), and the six locked decisions (§9). End the body with the Claude-Code line. **Do NOT release/tag/PyPI** — the
cross-repo 0.45.0 release (framework floor + xpt rebuild + 12-SID WAV audition) is a separate, later step
(memory `cross-repo-release-ordering`, `tokens-0.45.0-release-pending`).

### Suggested PR split (each independently green + byte-exact; or one big coherent PR)
- **PR1:** PART A1+A2 (`GeneratorPass` over all channels + LUT) behind `generator_pass` default-OFF, with the
  residual-zero + cross-driver tests + the §7B module↔macros round-trip (Tier 1/2). Proves the model without
  touching the default.
- **PR2:** PART C (digi) — independent, small.
- **PR3:** PART A3 (block-refire locality) + PART D (swap into `REGISTERED_MACROS`) + PART B (delete subsumed
  passes). The breaking default change, last, once the model is proven green.

---

## 9. Locked decisions (RESOLVED — implement exactly this; no choices remain)
1. **freq = ONE 16-bit channel** (not note+residual streams). HOLD/ACCUM/TRI on raw 16-bit; only `GEN_TABLE`
   on freq is note-relative-keyed (§1.4) for arp reuse. No per-voice mode flag.
2. **Waveform-agnostic (Facemorph guardrail).** NOTHING reads the waveform bit to route pitch — noise accents
   pitched notes, pulse plays percussion. Pinned by the §7.2 permute-nibble property test.
3. **No `GEN_END`, no anchor invention.** A length-1 atom can occur only at frame `F−1` (encode `SWEEP_OP`
   `LEN=1`); the gate asserts it. Block locality (A3) reuses the existing `block_refire_passes`, no new
   segmentation.
4. **`InstrumentProgramPass` owns ctrl/AD/SR** (already residual-zero); `generator_pass` owns the 13 channels
   of §1.2 (freq/pw_lo/pw_hi/cut_lo/cut_hi/res/modevol) and never touches ctrl/AD/SR. No overlap.
5. **Digi = excluded upstream** by `is_digi`/`filter_dump_paths` (§5 hardens the PWM clause). No SAMPLE op in
   this PR; a digi never reaches the generator pass.
6. **Ops:** reuse `SWEEP_OP=64` (ACCUM, repoint producer); new `GEN_TRI_OP=83`, `GEN_TUNING_OP=84`,
   `GEN_TABLE_DEF/STEP/END/REF=85/86/87/88`; retire `GLOBAL_OSC_OP=82`. Fitter embedded in §1A.

## 10. Guardrails
- **Stay in `preframr-tokens`.** No release/tag/PyPI. No edits to other repos. Do not commit this file.
- **Byte-exact + residual-ZERO is the gate, always** (`validate=True` / `register_state` guard /
  self-verifying fitter). Never accept "almost"; `reparse=True` on every measurement; validate on the corpus,
  not a 50-tune sample; progress markers in every sweep.
- **Leave op-id holes; never renumber.** New ops use `83+`.
- **Lean on the completeness tests** (`op_contracts`, `test_codebook_registry`, `test_macro_contracts`) — they
  catch a missed blast-radius wire.
