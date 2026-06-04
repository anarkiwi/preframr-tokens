# IMPLEMENT: InstrumentProgramPass — collapse the note-associated ctrl/AD/SR macros into one codebook

**Audience:** an engineer/agent working **entirely inside `preframr-tokens`**. Everything you need is in
this repo + the HVSC corpus at `/scratch/preframr/hvsc`. Do **not** require any other repo.

**One-line goal:** add ONE inline codebook pass that interns a voice's per-frame **timbre program**
(`ctrl, AD, SR` walk from a note-onset) as define-on-first DEF + exact REF, so the recurring/non-recurring
ctrl/AD/SR raw-SET residual goes to **zero by construction** — replacing the fragmented cluster
(`ctrl_wavetable`(+nibble), `ctrl_osc`, `ctrl_triple`, `ctrl_bigram`, `onset_def`, and `patch`'s AD/SR
role). **Build behind a new flag, default OFF, alongside the existing passes. Delete nothing in this task.**

---

## 0. Definition of done

1. New flag `instrument_program`, default OFF, NOT in `REGISTERED_MACROS`.
2. With `instrument_program=True` (composed with the current macro set), corpus-wide **raw-SET residual on
   the ctrl/AD/SR registers (4,5,6,11,12,13,18,19,20) == 0** (digi-excluded, `reparse=True`), measured by
   the self-contained script in §6.1.
3. Every emitted claim is **byte-exact**: the pass self-validates via `register_state` (mirror
   `StampPass._stamp_is_lossless`) and falls back to the unclaimed stream on any divergence — so enabling
   the flag can never change the rendered register stream.
4. The full test suite is green under xdist (§6.2), plus new tests (§5) that exercise the pass **through
   the real `RegLogParser.parse()`** (not synthetic dataframes).
5. No existing pass is modified beyond registration/pipeline wiring; nothing is deleted.

**Out of scope (do not attempt here):** freq/pitch residual (that is the pitch-ornament channel —
`skeleton`/`freq_*`/`pre_gate_freq`), PW/filter sweep residual (the `sweep`/`gradient` channel), and
deleting the subsumed passes (a later release once this is the default). Your residual gate is ctrl/AD/SR
**only**.

---

## 1. Why this works (background — self-contained)

C64 music drivers represent a voice's sound as an **instrument = a small per-frame register-write program,
referenced by id and fired at each note-onset** (Hubbard 8-byte record + tables; JCH/SF2
`wave-table|pulse-table|filter-table` by id; defMON sidTAB rows). The wave-table is *walked one row per
frame from note-on* (waveform changes), AD/SR are loaded at onset. Instruments are a **small bank reused
across notes**. Today this single concept is mined by ~6 overlapping passes, each with its own escape
condition (recurrence `MINREP≥2`, `fr_reg_count==1`, an onset floor, an oscillation-period match, a
nibble-lane id space). A ctrl/AD/SR write that falls in the *gap between* those conditions escapes all of
them and becomes raw-SET residual. The driver has no such gaps — every onset-associated write is part of
the program.

**Measured on 861,098 note-spans (corpus sample, `register_state` oracle) — this is why the model is
correct, not a heuristic:**
- AD constant within a gate-held span **97.0%**, SR **96.3%** → AD/SR are onset-anchored; span on the
  gate boundary. (The ~3% that vary = HR multiload / mid-note envelope — carried as extra STEPs.)
- waveform distinct-per-span mean **1.91** (lengths 1–3 dominate) → the program is a short per-frame walk.
- program `(ctrl-walk, AD, SR)` **exact-recurrence within a tune = 98.0%** → a small reused bank; exact
  REF compresses almost everything and **define-on-first covers the 2% unique**. Hence residual==0 by
  construction: every span emits at minimum a DEF.

You can re-run / extend these checks if you want; the harness lives at
`/scratch/tmp/empirical_checks.py` (uses only `preframr_tokens` + corpus).

---

## 2. Design in this repo's terms

A note **span** for a voice = the frames from a gate-on retrigger up to (but excluding) the next gate-on on
that voice's ctrl register — **exactly the span boundary `StampPass._reg_spans` already computes** (gate-on
retrigger; level-change fallback when a voice never gates). For each span build a byte-exact **timbre
signature** = the per-frame tuple of the *forward-filled* `(ctrl, AD, SR)` bytes for that voice. Group spans
by signature (across voices and notes — instruments are shared), and emit with `emit_recurring(minrep=1)`:
the first occurrence of each signature → a DEF (the program), every later occurrence → a single-row REF
carrying the target voice's ctrl reg. A signature seen once is still a DEF. Therefore **every ctrl/AD/SR
write inside any span is consumed by a DEF or REF → no raw SET survives on those regs**.

This is `StampPass` with three changes: (a) the signature/walk fields are `(ctrl, AD, SR)` instead of
`(freq, ctrl)`; (b) `minrep=1` (define-on-first) instead of `STAMP_MINREP`; (c) no transpose-relative
variant (instruments are absolute, not pitch-transposed). Mirror `stamp_pass.py` and the `"stamp"`
`CodebookFamily` closely.

### CRITICAL — pipeline placement (avoids the known voice-confusion bug)
Run this pass in the **inline loop in `reglogparser.parse`** (the same stage as `CtrlWavetablePass` /
`HardRestartPass` / `PreGateFreqPass`), which operates on **actual voice registers** (4/5/6, 11/12/13,
18/19/20 are all present; **no `VOICE_REG` markers, no canonical collapse, no nibble split yet**). Key on
the actual voice regs. Do **not** run it in the post-`voice_canonical_block_order` `PASSES` pipeline — there
all voices' ctrl collapse onto canonical reg4 and keying would conflate voices (this is the exact bug that
caused the `ctrl_wt` rejections; see `design/instrument_state_codebook_design.md` §8). Sanity-check the
stage with this throwaway probe (expect `VOICE_REG=0`, reg7/14 present):
```python
# at the top of your apply(), temporarily:
# print((df["reg"]==VOICE_REG).sum(), (df["reg"]==7).any(), (df["reg"]==14).any())
```

---

## 3. Exact changes, file by file

### 3.1 `preframr_tokens/stfconstants.py`
Append (ops 78–81 are free; highest current op is `INIT_OP = 77`):
```python
INSTR_DEF_OP = 78
INSTR_STEP_OP = 79
INSTR_END_OP = 80
INSTR_REF_OP = 81
```
For the per-frame walk, reuse the voice-relative reg offsets as STEP subregs (mirror StampPass's
`_FREQ_OFFSET=0`/`_CTRL_OFFSET=4` convention): a STEP with `subreg` in `{4,5,6}` sets ctrl/AD/SR
respectively; a STEP with the frame-advance sentinel advances one frame. Define:
```python
INSTR_OFF_CTRL = 4
INSTR_OFF_AD = 5
INSTR_OFF_SR = 6
INSTR_SUBREG_FRAME = STAMP_SUBREG_FRAME  # reuse the existing frame-advance sentinel
```
If `INSTR_*` need to be importable elsewhere, add them to the module's `__all__`/export list exactly as the
`STAMP_*` names are exported.

### 3.2 `preframr_tokens/macros/codebook.py`
Add a family entry to `CODEBOOK_FAMILIES` mirroring `"stamp"` (a multi-STEP DEF committed by an END op,
single-row REF):
```python
"instrument": CodebookFamily(
    name="instrument",
    def_op=INSTR_DEF_OP,
    step_ops=(INSTR_STEP_OP,),
    commit_op=INSTR_END_OP,
    refs=(RefSpec(INSTR_REF_OP),),
),
```
(Match the exact `CodebookFamily` constructor signature used by the `"stamp"` entry — same keyword set.)
Then add an `_InstrumentCodec` mirroring the existing stamp/`_CtrlWtCodec` codec: on each STEP accumulate
the field (`subreg` 4→ctrl, 5→AD, 6→SR) into the pending program; `INSTR_SUBREG_FRAME` advances a frame;
on the END/commit make the program live in the table keyed by id; a REF replays the whole program, writing
each frame's `(ctrl, AD, SR)` to the **REF row's target voice regs** (the REF row's `reg` is the voice ctrl
reg; AD = ctrl_reg+1, SR = ctrl_reg+2 — these are the SID per-voice offsets). Register the codec the same
way the `"stamp"` codec is registered.

### 3.3 `preframr_tokens/macros/decoders.py`
The decode path is the generic `CodebookDecoder` driven by `CODEBOOK_FAMILIES`; once §3.2 is registered,
wire the four `INSTR_*` ops into the `DECODERS` map exactly as the `STAMP_*` ops are wired (find where
`STAMP_DEF_OP`/`STAMP_REF_OP` get their decoder entry near `DECODERS = {…}` at the bottom of the file and
add the parallel `INSTR_*` entries). Verify by grep that all four new ops resolve in `DECODERS`.

### 3.4 NEW FILE `preframr_tokens/macros/instrument_program_pass.py`
Copy `stamp_pass.py` as the scaffold and adapt:
- `GATE_FLAGS = frozenset({"instrument_program"})`; bail unless `args.instrument_program`.
- Reuse `StampPass`-style span construction, but build the signature/walk from the voice's **actual**
  `ctrl (4/11/18)`, `AD (+1)`, `SR (+2)` regs, forward-filled per frame (use the same forward-fill idiom as
  `StampPass._build_span` / `_freq_at`). Span boundaries = gate-on retriggers on the voice ctrl reg (same as
  `StampPass._reg_spans`, but iterate the three ctrl regs, not `FREQ_TRAJ_REGS`).
- Group by **signature alone** (cross-voice/cross-note instrument sharing — this is the 98% bank reuse).
- `emit_recurring(..., minrep=1, ...)`: `emit_first` → DEF rows (header `INSTR_DEF_OP` + per-frame STEP
  atoms emitting only changed fields + `INSTR_SUBREG_FRAME` advance + `INSTR_END_OP`), `emit_ref` → a single
  `INSTR_REF_OP` row on the voice's ctrl reg carrying the program id. Mirror `StampPass._def_rows`.
- Consume **all** ctrl/AD/SR raw-SET row indices inside each span (`rows_of`), so none survive as residual.
- `arbitrate(...)` with one `Claim(..., label="instrument")`, then guard with a `register_state` lossless
  check identical to `StampPass._stamp_is_lossless`; return the un-claimed `df` if it diverges.

### 3.5 register the pass + flag
- The `GATE_FLAGS` on the class is auto-discovered by `flag_registry.macro_flag_names()` (it glob-imports
  `macros`), so the flag name appears automatically. Confirm `"instrument_program"` shows up in
  `macro_flag_names()`.
- Insert an `InstrumentProgramPass()` instance into the **inline pass sequence in
  `reglogparser.parse`** at the same stage as `CtrlWavetablePass`/`HardRestartPass` (grep `parse` for where
  those are instantiated/looped). Order it **before** `SubregPass` and `voice_canonical_block_order`. Place
  it after `HardRestartPass` so HR-shaped onsets are already normalized into the span.
- Do **not** add it to `REGISTERED_MACROS` (`tokenizer_config.py`) yet.
- If `flag_registry.FLAG_CONFLICTS` needs an entry (e.g. it must not run together with the passes it will
  later replace, to avoid double-claiming the same rows during A/B), add a conflict only if your residual
  run in §6.1 shows double-claims; otherwise the arbiter priority resolves overlap and no conflict entry is
  needed. Set the Claim `priority` below (more negative than) the passes it supersedes so it wins the
  ctrl/AD/SR rows (check `_STAMP_PRIORITY=-10`, `_CTRL_WT_PRIORITY`; pick e.g. `-12`).

---

## 4. The DEF/REF byte layout (so the decoder round-trips)

DEF (one per unique program), emitted at the first occurrence's onset position:
```
INSTR_DEF_OP   reg=0  subreg=-1  val=<program_id>
  # frame 0:
  INSTR_STEP_OP reg=0 subreg=INSTR_OFF_CTRL val=<ctrl0>
  INSTR_STEP_OP reg=0 subreg=INSTR_OFF_AD   val=<ad0>     # only if changed vs prev frame
  INSTR_STEP_OP reg=0 subreg=INSTR_OFF_SR   val=<sr0>     # only if changed
  # frame i>0:
  INSTR_STEP_OP reg=0 subreg=INSTR_SUBREG_FRAME val=0     # advance one frame
  INSTR_STEP_OP reg=0 subreg=INSTR_OFF_CTRL val=<ctrl_i>  # only changed fields
  ...
INSTR_END_OP   reg=0  subreg=-1  val=<program_id>
```
REF (one row per later occurrence), `reg` = the target voice's ctrl reg (4/11/18):
```
INSTR_REF_OP   reg=<voice_ctrl_reg>  subreg=-1  val=<program_id>
```
Decoder replays: for frame 0 write the accumulated fields to `(ctrl_reg, ctrl_reg+1, ctrl_reg+2)`; on each
`INSTR_SUBREG_FRAME` advance to the next decode frame and re-emit changed fields. The forward-fill on encode
means a field STEP appears only when the value changes, so the decoder holds the last value (same
convention as StampPass). The `__pos` stamping in `emit_recurring` guarantees the DEF precedes its REFs.

---

## 5. Tests (add to the existing test suite; test THROUGH `parse()`)

Per repo convention, tokenizer passes must be validated through the **full `RegLogParser.parse()`**
(post combine/quantize), never synthetic dataframes (synthetic tests ship false-green). Add to the macro
test module (find where `stamp_pass` / `patch_pass` are tested and follow that file's fixtures):
1. **Round-trip byte-exact:** parse a handful of real corpus tunes with `instrument_program=True` and assert
   `register_state` is byte-identical to parsing the same tunes with `instrument_program=False`
   (the flag must not change the render). Use real `.dump.parquet` fixtures the existing tests already use;
   never skip on missing fixtures.
2. **Residual drained:** for those tunes, assert zero raw `SET_OP` rows remain on regs {4,5,6,11,12,13,18,19,20}
   with the flag on (and assert it was non-zero for at least one tune with the flag off, proving the pass did
   the work).
3. **Define-on-first:** a tune with a once-only instrument still yields an `INSTR_DEF_OP` (no REF) and no
   residual — i.e. `minrep=1` really emits the lone DEF.
4. **Lossless fallback:** confirm the `register_state` guard path returns the unclaimed stream (you can force
   it in a unit test by monkeypatching the codec to mis-decode and asserting `apply` returns the input df).

Repo lint forbids `#` comments (docstrings only) and rejects docstrings >5 lines — keep new docstrings short.

---

## 6. Acceptance gates + exact commands

All commands run from `/scratch/anarkiwi/preframr-tokens`. Use the baked image with the local tree mounted
(fast; no rebuild) — host docker needs `--network host`; the proxpi mirror is reached via that.

### 6.1 Residual gate (self-contained, in-repo) — THE primary gate
Write `/scratch/tmp/instr_resid_gate.py`:
```python
import glob
from collections import Counter
from preframr_tokens import RegLogParser
from preframr_tokens.tokenizer_config import default_tokenizer_args, REGISTERED_MACROS
from preframr_tokens.dump_meta import meta_path_for, read_meta
from preframr_tokens.stfconstants import SET_OP
from preframr_tokens.macros.flag_registry import macro_flag_names

known = set(macro_flag_names())
flags = {f: True for f in REGISTERED_MACROS if f in known}
flags["instrument_program"] = True            # the new pass ON
P = RegLogParser(default_tokenizer_args(seq_len=4096, **flags))
TIMBRE = {4, 5, 6, 11, 12, 13, 18, 19, 20}
sample = sorted(glob.glob("/scratch/preframr/hvsc/**/*.dump.parquet", recursive=True))[::30]
N = len(sample); bad = Counter(); dirty = 0
for i, p in enumerate(sample):
    if i % 25 == 0:
        print(f"[{i}/{N}] dirty={dirty} resid={sum(bad.values())}", flush=True)   # progress marker REQUIRED
    try:
        if getattr(read_meta(meta_path_for(p)), "is_digi", False): continue
    except Exception: pass
    try:
        df = next(P.parse(p, max_perm=1, require_pq=False, reparse=True))
    except StopIteration: continue
    except Exception: continue
    regs = df["reg"].to_numpy(); ops = df["op"].to_numpy()
    n = sum(1 for k in range(len(df)) if int(ops[k]) == SET_OP and int(regs[k]) in TIMBRE)
    if n: dirty += 1; bad[p.rsplit("/",1)[-1]] = n
print(f"DONE timbre_residual={sum(bad.values())} dirty={dirty} top={bad.most_common(20)}")
```
Run it:
```bash
docker run --rm --network host -v $PWD:/tok -v /scratch/preframr:/scratch/preframr \
  -v /scratch/tmp:/stmp:ro -w /tok -e PYTHONPATH=/tok \
  anarkiwi/preframr-xpt:0.2.18 python -u /stmp/instr_resid_gate.py
```
**PASS = `timbre_residual=0 dirty=0`.** Iterate the pass until it does. (Start with `[::200]` for a fast
loop, finish on `[::30]` or denser. A non-zero tail prints the worst tunes — analyze them with the same
forward-fill logic; the usual culprits are spans your boundary missed: never-gated voices, pre-first-onset
preamble writes — decide whether they belong to a span or to `init_preamble`, do not widen the pass to
swallow non-instrument regs.)

### 6.2 Test suite (xdist)
```bash
docker run --rm --network host -v $PWD:/tok -w /tok -e PYTHONPATH=/tok \
  anarkiwi/preframr-xpt:0.2.18 python -m pytest -n auto --dist worksteal -q
```
Coverage auto-combines under xdist. Chunk any long property sweep via `parametrize`. Must be fully green.

### 6.3 Lint
Run the repo's configured `black`/lint (mirror what CI runs; check the repo root for the config). No `#`
comments; docstrings ≤5 lines.

---

## 7. Notes / gotchas (learned the hard way — heed them)
- **Always `reparse=True`** when measuring residual; stale `.pq` caches silently read pre-change tokens and
  will show a false-green/false-red. ZERO is the gate; non-zero means a span you didn't model — fix it,
  don't accept "99%".
- **Progress markers are mandatory** in any corpus sweep/audit script (`[i/N]` flushed) — see §6.1.
- **Exclude digis** (they hammer ctrl and manufacture false bottlenecks): the `is_digi` filter is in §6.1.
- **Validate on the corpus, not a 50-tune sample** — a small sample previously hid an id-collision and a
  cache bug. Use `[::30]` or denser for the final gate.
- The pass is **byte-exact** (register-state), not merely audio-exact — the `register_state` guard is
  non-negotiable; if a claim diverges, drop it, never ship a divergent replay.

## 8. Companion design (context only — NOT required to execute)
The full rationale, the pass-by-pass subsumption table, the contracts, and the §5 measurements live in the
preframr-xpt design `instrument_program_codebook_design.md`. This file is self-sufficient; that one is
background if you want the why in more depth.
