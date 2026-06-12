# WORK ORDER: v3 event-codec hardening (execute mechanically, then delete this file)

**Mission:** action the 2026-06-12 critical analysis of the v3 event model: truth-up the fidelity
contract language, fix the KEYFRAME snapshot gap, add generation-side decode/masking support, guard
the BPE dictionary (stack fix + boundary isolation + merge audit), embed a format version, and
remove verified-dead code. Everything below is pre-researched against this repo at commit `81f3bc6`
plus the sibling repos (`preframr`, `preframr-xpt`) — **no further research is required**; if an
anchor has drifted, re-locate by the named symbol, not the line number.

## Ground rules (read fully before starting)

- Branch `work/v3-hardening` off current `origin/main`. One commit per phase (messages given).
  P8 MAY be split into a second branch/PR if the first PR is otherwise green.
- Gate after every phase: `./run_tests.sh` (black, pylint, pyright, pytest, coverage ≥85%).
- Lint constraints for all NEW code (enforced by `tests/test_lint.py`): docstrings are ONE
  paragraph ≤5 lines; NO narrative `#` comments (only directive ones, e.g. `# pylint: disable`).
- Do NOT change default tokenization output except where a phase explicitly says so (P2 does;
  everything else is opt-in or metadata). Do NOT touch parse-domain modules except those named in
  P7. Do NOT release; bump `fallback_version` in P9 and stop (the operator tags releases —
  a canonical training run pinned to 0.48.0 is in flight in preframr-xpt).
- The fidelity invariant for any codec change: `stream.encode(ow, verify=True)` self-verifies
  against `canonical_writes`; `tests/test_events_roundtrip.py` + `tests/test_events_corpus.py`
  must stay green on the driver fixtures.
- Final step of the LAST phase: `git rm WORK_ORDER_v3_hardening.md` — this file must not survive
  the work.

## P0 — preflight

```bash
git fetch origin && git switch -c work/v3-hardening origin/main
./run_tests.sh   # must be green before any change; if not, STOP and report
```

## P1 — contract truth-up (docs + characterization test)

The PRE primitive was removed (commit `0790372`) but the contract prose still claims transient
preservation / "zero drops". Freq/PW/global values are emitted from SETTLED end-of-frame state;
intra-frame transients on those registers are canonicalized away. Make the words match the code.

1. `preframr_tokens/events/stream.py` module docstring (lines 1–6): replace the whole docstring
   with:

   ```
   """v3 canonical event codec: the oracle is :func:`canonical_writes` -- CTRL/AD/SR change
   activity as ordered typed events at sub-frame resolution (gate-on = NOTE_ON with duration;
   gate-off ALWAYS derived, no NOTE OFF token), with freq/PW/globals emitted from SETTLED
   end-of-frame state (intra-frame transients and same-value rewrites are canonicalized away --
   licensed by raw-vs-canonical renders at the reSID noise floor). Scope: single-speed non-digi."""
   ```

2. `canonical_writes` docstring (symbol at ~line 654): replace `"""The fidelity target: the dump's
   audibly-faithful canonical form -- an exact intra-frame PERMUTATION of the dump's writes (zero
   drops). Per frame: ...` with:

   ```
   """The fidelity target: the dump's audibly-faithful canonical form. Per frame: voices
   ascending, each as [settled-changed freq lo,hi][settled-changed pw lo,hi][cas write sequence
   in driver order, gate-offs derived, onset envelope on its recorded gate-edge side, HR prep on
   the gate=0 side of the off]; then settled-changed globals reg-ascending. CAS order is
   preserved exactly; freq/PW/global intra-frame transients settle to end-of-frame."""
   ```

3. `README.md`, section "Fidelity contract": replace the sentence
   `a byte-exact **intra-frame permutation** of the dump's writes (zero drops)` with
   `byte-exact CTRL/AD/SR change activity in driver order, plus freq/PW/globals from the settled
   end-of-frame state (intra-frame transients and same-value rewrites canonicalize away — licensed
   by reSID noise-floor renders)`.

4. New test in `tests/test_events_stream.py` (follow the existing synthetic-dump helper style in
   that file — it builds dump DataFrames with `clock, irq, chipno, reg, val` rows):

   ```python
   def test_intra_frame_freq_transient_settles_to_end_of_frame(self):
   ```
   Build a dump where one frame writes voice-0 `freq_lo` (reg 0) twice with DIFFERENT values
   (e.g. 0x40 then 0x80) plus a gate-on; assert `canonical_writes(ow)` contains exactly ONE reg-0
   write for that frame with the LAST value, and `stream.encode(ow, verify=True)` succeeds. This
   PINS current behavior; the audibility measurement licensing it is preframr-audio's job (see
   "Out of scope").

Commit: `docs(events): contract prose matches post-PRE behavior; pin transient settling`

## P2 — chunk_keyframe omits NOTE_TABLE deviations (bugfix; changes block bytes)

`chunk_keyframe` (symbol at ~line 1115 of `stream.py`) snapshots TUNING and TICK per voice but not
the recovered note-table deviations, so conditioning state implies wrong absolute freqs for
deviated notes.

1. In `chunk_keyframe`, inside the first `for v in range(3):` loop, directly AFTER the
   `if d.freq_active[v]:` TUNING emission block, add:

   ```python
        if d.freq_active[v] and d.devs[v]:
            out += [voice_tok, NOTE_TABLE]
            _emit_u(out, len(d.devs[v]))
            prev = 0
            for note in sorted(d.devs[v]):
                _emit_s(out, note - prev)
                _emit_s(out, d.devs[v][note])
                prev = note
   ```

2. New test in `tests/test_events_stream.py`:
   `test_chunk_keyframe_carries_note_table_devs` — build a dump whose sustained freq is OFF the
   equal-tempered grid by a constant (so `_freq_layer` recovers a nonzero deviation; e.g. hold
   `freq = pitch_grid.note_freq_at(49, 0.0) + 7` for many frames, gated). Encode whole-tune
   tokens, call `stream.chunk_keyframe(tokens, upto=len(tokens))`, assert `NOTE_TABLE` appears
   between the `KEYFRAME` brackets, and `stream.decode(stream.strip_keyframes(kf + tail)) ==
   stream.decode(tail)` for any valid tail (conditioning stays decode-transparent).

3. This changes training-block bytes: in `preframr_tokens/events/dataset.py` the constant
   `ATOM_CACHE_VERSION` does NOT need bumping (the atom cache is pre-KEYFRAME whole-tune atoms,
   unaffected), but note the version bump in P9 covers downstream dataset-cache invalidation.

Commit: `fix(events): chunk_keyframe snapshots NOTE_TABLE deviations`

## P3 — `decode(extend=)` for open-ended generation

`_Decoder.run` replays exactly the declared frame count `n` (the stream's first varint); generated
continuations beyond `n` are silently dropped.

1. `stream.py`: change `def decode(tokens):` → `def decode(tokens, extend=False):` passing through
   to `_Decoder.run(extend=extend)`; in `run`, `n, last = self.parse()` then
   `return self.replay(max(n, last + 1) if extend else n)`. Docstring (≤5 lines): note default
   truncates at the declared frame count; `extend=True` replays through the last parsed group
   (for model-generated continuations).
2. `preframr_tokens/events/generate.py`: add `extend=False` kwarg to `tokens_to_writes` and
   `tokens_to_dump_df`, passed to `dataset.ids_to_writes` — which needs the same kwarg passed to
   `stream.decode` (`preframr_tokens/events/dataset.py::ids_to_writes`).
3. Test in `tests/test_events_generate.py`: encode a 4-frame synthetic tune; hand-append a valid
   extra group `[*varint(DT=1), VOICE_BASE+0, NI_STEP, *varint(zigzag(+2))]` (build digits with
   `varint.encode_unsigned`/`encode_signed` + `VAR_BASE`); assert default decode output equals the
   un-appended decode, and `decode(toks, extend=True)` contains writes at the new frame.

Commit: `feat(events): decode(extend=) replays past the declared frame count`

## P4 — BPE guards: stack fix + boundary isolation (opt-in)

### P4a — UnigramTrainer stack overflow fix belongs in the library

The known SIGSEGV (recursive `Rc::drop` on 35K+-token sentences) is currently worked around only by
preframr-xpt's runner exporting `RUST_MIN_STACK`. Any other caller crashes.

1. `preframr_tokens/train_worker.py`: at the top of `train_worker(...)` body add
   `os.environ.setdefault("RUST_MIN_STACK", "2000000000")` (+ `import os`). It runs in the spawned
   child before the tokenizers Rust threads start, so it takes effect; an operator override via a
   pre-set env var still wins.
2. New test `tests/test_train_worker_long_sentence.py`: build a `RegTokenizer` over
   `events_alphabet()` with a stub args namespace (`tokenizer="unigram"`, `tkvocab=64`,
   `tkmodel=<tmpfile>`); feed `train_tokenizer([( "<tmp>/x.dump.parquet-ish-name", pd.DataFrame({"n":
   <60_000 random ints in 1..127>}), 0 )])` — use a real temp dir and a filename ending in
   `DUMP_SUFFIX` so `write_uni` derives the `.uni` path. Assert the spawned trainer exits 0 and the
   tkmodel file exists. (Pre-fix this is the SIGSEGV reproducer; keep runtime <60 s — reduce to
   40_000 ids if needed.)

### P4b — event-boundary isolation for unigram merges (default OFF)

Cross-voice merges re-multiplex what the alphabet separates (the old substrate needed
`melody_merge_split` for exactly this). The isolation mechanism already exists
(`train_worker._build_unigram_pre_tokenizer` + `pre_tokenizers.Split(behavior="isolated")`); the
events path just never feeds it chars (its op-based selector matches only macro head-ops).

1. `preframr_tokens/events/dataset.py`: add (near `PAD_ID`):

   ```python
   BOUNDARY_ISOLATION_NS = tuple(
       [stream.VOICE_BASE + 1 + v for v in range(4)] + [stream.KEYFRAME + 1]
   )
   ```
   (n-space = atom_id + 1; requires `from . import stream` which dataset already has via
   `from .pipeline import ...` — if not directly imported, add `from . import stream`.)
2. `preframr_tokens/regtokenizer.py`: in `RegTokenizer.__init__` add `self.isolation_ns = None`.
   Add method (≤5-line docstring: "Unicode chars for the given n-space ids; merged into the
   unigram isolation set."):

   ```python
   def _isolation_chars_for_ns(self, ns):
       if not ns:
           return ""
       return "".join(sorted(set(self.encode_unicode(np.array(ns, dtype=np.int64)))))
   ```
   In `train_tokenizer`, after `isolation_chars = self._isolation_chars_for_ops(...)` (and its
   log), add:

   ```python
        if self.isolation_ns:
            isolation_chars = "".join(
                sorted(set(isolation_chars) | set(self._isolation_chars_for_ns(self.isolation_ns)))
            )
   ```
3. `preframr_tokens/corpus.py`, events path — immediately after the line
   `self.tokenizer.tokens = events_dataset.events_alphabet()` add:

   ```python
        if getattr(self.args, "bpe_isolate_boundaries", False):
            self.tokenizer.isolation_ns = events_dataset.BOUNDARY_ISOLATION_NS
   ```
4. `preframr_tokens/tokenizer_config.py`: add `bpe_isolate_boundaries=False` to the default-args
   construction (grep `default_tokenizer_args` and add the key beside the other tokenizer
   booleans).
5. Test in `tests/test_events_dataset.py`: train a unigram tokenizer (tkvocab≈96) over ~20
   synthetic multi-voice event streams TWICE — isolation off, then on (set
   `tokenizer.isolation_ns = BOUNDARY_ISOLATION_NS` directly). With isolation ON: for every vocab
   piece (`Tokenizer.get_vocab()` keys, skip `<unk>`), `decode_unicode(piece)` → ids; assert every
   piece containing a `BOUNDARY_ISOLATION_NS` id has exactly 1 atom. (No assertion on the
   isolation-OFF model beyond "trains green" — it is the A/B control.)
6. Known residual, record it in the P5 module docstring: DT digits share the varint range with
   value digits, so cross-FRAME welds through a DT cannot be isolated without killing value
   merges — the audit (P5) measures them instead.

Commit: `feat(tokenize): in-library RUST_MIN_STACK + opt-in event-boundary merge isolation`

## P5 — merge-boundary audit tool

New file `preframr_tokens/bpe_audit.py` (root; events-only imports; module docstring ≤5 lines noting
the DT-weld caveat from P4b.6):

```python
def audit_vocab(tokenizer) -> pd.DataFrame: ...
def summarize(frame) -> dict: ...
```

- `audit_vocab(tokenizer)`: `tokenizer` is a loaded `RegTokenizer` over `events_alphabet()` with a
  trained `tkmodel`. For each vocab piece string from `tokenizer.tkmodel.get_vocab()` (skip
  `<unk>`): atoms = `[int(i) - 1 for i in tokenizer.decode_unicode(piece)]`. Columns: `piece_id`,
  `n_atoms`, `crosses_voice` (any atom in `{VOICE_BASE..VOICE_BASE+3, KEYFRAME}` when
  `n_atoms > 1`), `n_kinds` (count of atoms in `{TUNING..G_RAMP}` — import the constants from
  `events.stream`), `all_digits` (all atoms in `[VAR_BASE, VAR_BASE+32)`).
- `summarize(frame)` → `{"n_pieces", "n_multi_atom", "n_crossing_voice", "frac_crossing_voice",
  "n_multi_kind"}` over multi-atom pieces.
- `if __name__ == "__main__":` CLI: `python -m preframr_tokens.bpe_audit <tkmodel.json>` — builds
  the RegTokenizer over `events_alphabet()`, loads the model file, prints `summarize()` then the
  20 `crosses_voice` pieces with the most atoms (piece_id + decoded atom ids).
- Test `tests/test_bpe_audit.py`: reuse P4b.5's isolation-OFF trained model; assert `audit_vocab`
  returns the expected columns and `summarize` keys; with the isolation-ON model assert
  `n_crossing_voice == 0`.
- `README.md`: one line in the Modules list:
  `preframr_tokens.bpe_audit -- merge-table boundary audit (voice/KEYFRAME-crossing + multi-kind merges; run after any unigram train)`.

Commit: `feat(tokenize): bpe_audit -- merge-table boundary audit`

## P6 — embed the event-format version in artifacts

`ATOM_CACHE_VERSION` (dataset.py:35) exists but other artifacts rely on package pinning alone.

1. `stream.py`: add `EVENT_FORMAT_VERSION = 1` next to `VOCAB_SIZE`; append to `__all__`.
2. `dataset.py`: replace `ATOM_CACHE_VERSION = 1` with
   `ATOM_CACHE_VERSION = stream.EVENT_FORMAT_VERSION`.
3. `corpus.py`: `TokenizeMeta` gains field `format_version: int = 0`; the EVENTS-path
   `TokenizeMeta(...)` construction (the one after the scope filter; grep
   `self._tokenize_meta = TokenizeMeta(` — take the occurrence in the events preload, ~line 442)
   passes `format_version=events_stream.EVENT_FORMAT_VERSION`.
4. Sidecar: grep `_write_reg_widths_sidecar` in `corpus.py`; in the JSON it writes add key
   `"_event_format_version": events_stream.EVENT_FORMAT_VERSION`. In the fast-path reader that
   loads that sidecar (same grep, the read side in `iter_block_seqs`' fast path), after load:
   if the key exists and differs from `events_stream.EVENT_FORMAT_VERSION`, `raise ValueError`
   naming both versions; if absent, proceed (pre-versioning artifact).
   The reg-widths reader is shared with the parse-domain path — the check must only fire when the
   key is PRESENT, so parse-domain sidecars (which won't carry it) are unaffected.
5. Test in `tests/test_events_corpus.py`: after a normal preload, rewrite the sidecar JSON with
   `"_event_format_version": 999` and assert the fast path raises `ValueError`.

Commit: `feat(events): EVENT_FORMAT_VERSION embedded in tokenize artifacts`

## P7 — remove verified-dead code

Dependency status was verified 2026-06-12 across `preframr-tokens`, `preframr`, and `preframr-xpt`.
Re-verify each with the given command (expect ONLY the listed hits); if anything NEW appears, skip
that deletion and note it in the PR description instead.

| Delete | Verify command (run from repo root) | Expected remaining hits |
|---|---|---|
| `preframr_tokens/events/factored.py` | `grep -rn "factored" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | docstrings in measure/schema/encoder/decoder/corpus + the two test files below |
| `preframr_tokens/events/tokenize.py` | `grep -rn "events import.*tokenize\|events\.tokenize" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | the two test files below |
| `preframr_tokens/events/encoder.py` + `decoder.py` | `grep -rn "events import.*encoder\|events import.*decoder\|events\.encoder\|events\.decoder" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | the two test files below |
| `preframr_tokens/pipeline_trace.py` + `tests/test_pipeline_trace.py` | `grep -rn "pipeline_trace" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | only its own test |
| `preframr_tokens/tokenizer_profile.py` | `grep -rn "tokenizer_profile" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | only its own module |
| `RegTokenizer.split_cross_boundary_merges` (function) + `tests/test_melody_merge_split.py` | `grep -rn "split_cross_boundary_merges" --include="*.py" . ../preframr/preframr ../preframr-xpt/preframr_experiments` | function def + its test only |

KEEP (verified live — do not touch): `parse_audit.py` (imported by `reglogparser.py`; used by xpt
staging tests), `coarsen_pass.py` (imported by `macros/passes.py`), `constrained_decode.py`
(imported by `preframr/inference/predict.py`), `events/measure.py` (the collapse-measurement
instrument — but fix its docstring: replace the words "the factored event stream" with "event
token streams" since the factored codec is gone), all of `macros/` and `reglogparser.py`.

Test refits required by the deletions:
- `tests/test_events_acceptance.py`: drop `factored`/`tokenize`/`encoder` from the import and
  delete the test functions that use them; KEEP the `gestures`, `oracle`, `varint`, and
  `events.schema` helper tests.
- `tests/test_events_roundtrip.py`: this file carries the 5-driver-fixture roundtrip — DO NOT
  delete the fixture coverage. Rewrite its v0/factored assertions to the stream codec:
  `stream.decode(stream.encode(ow, verify=False)) == stream.canonical_writes(ow)` per fixture.
- `preframr_tokens/events/schema.py`: after the trio is gone, delete `Kind` and `Event` (verify:
  `grep -rn "Kind\|Event(" preframr_tokens/events tests | grep -v schema.py` shows no users);
  KEEP `Shape` (imported by `gestures.py`) and all register-map helpers (imported by `stream.py`).
- `README.md`: no module-list changes needed for the events trio (never listed); remove any
  `pipeline_trace`/`tokenizer_profile` mentions if present (grep README.md).

Commit: `chore: remove dead v0/factored codecs, pipeline_trace, tokenizer_profile, merge-splitter`

## P8 — event-grammar sampling mask (`preframr_tokens/events/constrained.py`) [may be its own PR]

A per-step validity mask over the 127-atom grammar for generation-time logit guarding (the existing
`preframr_tokens/constrained_decode.py` speaks the retired parse-domain space; leave it alone).
Mirror `_Decoder.parse`/`_parse_event` exactly — this is that parser re-expressed as an incremental
token-at-a-time machine. Pure numpy, torch-free.

API:

```python
class EventStreamState:
    def valid_mask(self) -> np.ndarray  # bool, shape (stream.VOCAB_SIZE,)
    def push(self, tok: int) -> None    # raises ValueError on invalid
    @property
    def at_group_boundary(self) -> bool # True when a frame group just completed
```

Implementation contract (field-stack design):
- Internal stack of pending field descriptors; `valid_mask` derives from the stack top, else from
  the group-level state. Field kinds: `("u",)`/`("s",)` = varint digits (`VAR_BASE..+32`; pop when
  a digit without bit 4 arrives; accumulate the decoded value), `("nib", base)` = one token in
  `[base, base+16)`, `("ctrl",)` = push `("nib", NIB_WAVE), ("nib", NIB_ART)`, `("env",)` = two
  `("nib", NIB_ENV)`.
- Group-level states: **HEAD** (stack starts `[("u",)]` for the frame-count header) → **PREAMBLE**:
  allow `{VOICE_*} ∪ digits`; `VOICE_v` → expect one of `{TUNING, NOTE_TABLE, TICK}`; TUNING pushes
  `[u]`, TICK pushes `[u, s]` (record the decoded `(tick, offset)` per voice — NOTE_ON duration
  fields depend on it), NOTE_TABLE pushes `[u]` then, when the count value completes, pushes
  `count × [s, s]`. A digit starts the first DT → **BODY**.
- **BODY**: after a DT varint completes, allow only `{VOICE_*}`; after a `VOICE_v` (enforce
  ascending within the frame group: `v > last_v`), allow event kinds `{NI_STEP..G_RAMP}`; after an
  event body completes, allow `{event kinds} ∪ {VOICE_w : w > v} ∪ digits` (digits start the next
  DT and set `at_group_boundary`).
- Event-kind field pushes (exactly `_parse_event`'s reads): `NI_STEP`/`FD_STEP` → `[s]`;
  `NI_RAMP`/`FD_RAMP` → `[shape, u(len), u(deg), s, deg×s]` (`shape` = one of
  `{SHAPE_POLY, SHAPE_PERIOD}`; push the `deg×s` after the deg value completes); `PW_STEP` →
  `[nib(NIB_ENV), u]`; `PW_RAMP` → `[shape, u, u(deg), nib(NIB_ENV), u, deg×s]`; `G_STEP` →
  `[greg, u]` and `G_RAMP` → `[greg, shape, u, u(deg), u, deg×s]` where `greg` = one of
  `{REG_BASE+21..REG_BASE+24}`; `FLD_CTRL` → `[ctrl]`; `FLD_AD`/`FLD_SR` → `[env]`;
  `FLD_NOTE_ON` → `[ctrl, u(flags)]` then on flags completion push conditionally (`_FLAG_*`
  constants from `stream`): OAD→`[env]`, OSR→`[env]`, (HRAD|HRSR)→`[u(hr_off)]`+HRAD?`[env]`+
  HRSR?`[env]`; then `[u(mode)]` — mask mode to digit tokens `{0,1,2}` (no continuation bit);
  if mode≠0 push `[u, s]` when the voice's tick>1 else `[u]`; if mode==2 push `[ctrl]`.
- `KEYFRAME` is never valid (generation emits pure streams).
- Export `VOICE_BASE`-relative constants by importing from `.stream`, including the `_FLAG_*` and
  `_GO_*` privates (import them explicitly; pylint disable if needed) — do not duplicate values.

Tests (`tests/test_events_constrained.py`):
1. **Conformance**: for ≥5 synthetic dumps (reuse the generators in `test_events_stream.py`),
   `toks = stream.encode(ow, verify=False)`; walk: assert `valid_mask()[tok]` then `push(tok)` for
   every token. Include a dump that exercises TICK (regular durations), NOTE_TABLE (off-grid
   freq), a PW ramp, a global write, and an HR-prepped NOTE_ON.
2. **Rejection**: at 100 sampled positions, assert at least one structurally-wrong token class is
   masked False (e.g. a `VOICE` token mid-varint; a `NIB_WAVE` where a digit is required).
3. **Fuzz**: seeded rng; from a fresh state, sample 2_000 tokens uniformly from the mask; truncate
   to the last position where `at_group_boundary` was True; assert `stream.decode(sampled_prefix)`
   does not raise (the declared frame count may truncate replay — that is fine; parse must accept).

`README.md`: one Modules line:
`preframr_tokens.events.constrained -- per-step grammar-validity mask for sampling over the event alphabet`.

Commit: `feat(events): event-grammar sampling mask (EventStreamState)`

## P9 — wrap up

1. `pyproject.toml`: `fallback_version` `0.48.0` → `0.49.0`. Do not tag/release.
2. `./run_tests.sh` full green.
3. `git rm WORK_ORDER_v3_hardening.md` (this file) — include in the final commit:
   `chore(release): bump fallback_version to 0.49.0; remove executed work order`
4. Push the branch, open a PR to `main` titled `v3 hardening: contract truth-up, BPE guards,
   format version, dead-code removal (+ event sampling mask)`. PR body: one bullet per phase +
   the "Out of scope" list below verbatim.

## Out of scope — hand back in the PR description (do NOT attempt here)

- **preframr-audio**: a pinning test licensing intra-frame different-value freq/PW transient
  settling (P1 pins tokens-side behavior only; the audibility measurement belongs with the chip
  facts).
- **preframr-xpt docs**: drop the `pipeline_trace.py` bullet from `design/README.md` "Elsewhere";
  note in `design/encoding/backlog_tokens_hardening.md` that the real-pipeline events test exists
  (`test_events_corpus.py`) and only the synthetic atom-mix/balance assertions remain open.
- **preframr framework**: adding `--bpe-isolate-boundaries` to `args.py` so specs can A/B it; the
  isolation default flip decision (needs the bpe_audit read + a canonical A/B).
- **Research, not mechanical**: copy-fraction-vs-MDL A/B of the `_SeriesCost` gesture objective;
  the n_frames-header entropy question.
