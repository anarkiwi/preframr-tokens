# AGENTS.md — Codebook macro unification (untracked execution spec)

> **This file is an execution spec, not documentation.** When the operator says
> **"execute agents.md"**, run the [Implementation](#implementation) section top to
> bottom **autonomously, without pausing for per-step approval**. Each step ends in a
> **verify gate** (a shell command). The gate is the contract: proceed only when it
> passes; on failure, apply the step's *repair* guidance and retry; escalate to the
> operator only if a gate still fails after the documented repair.
>
> This spec is written to be runnable **from a clear context** (a fresh agent with no
> memory of the discussion that produced it). All file paths, classes, and methods are
> named explicitly below. Read [Findings](#findings) and [Architecture](#architecture)
> first, then execute.

---

## Findings

`preframr_tokens` encodes SID register streams into compact token "atoms" via a pipeline
of macro passes, and decodes them back via a per-op decoder dispatch
(`preframr_tokens/macros/decoders.py`, table `DECODERS`, driven by
`preframr_tokens/macros/walker.py::FrameWalker` and `…/decode.py::expand_ops`).

Four macro families are **inline codebooks** — a definition is buffered into a live
`id → entry` table, then references replay it:

| Family   | DEF op | STEP op(s) | COMMIT             | REF op(s)              | `DecodeState` table | Payload (entry)                       | Replay target / drain |
|----------|--------|------------|--------------------|------------------------|---------------------|---------------------------------------|-----------------------|
| STAMP    | 56     | 57         | 58 (END)           | 59, **63 REL**         | `stamp_table`       | voice-relative `(off,val)` write-series | target voice freq reg; REL adds base delta @off0; **queued multi-frame** |
| PATCH    | 60     | 61         | STEP @subreg `SR`  | 62 (SET)               | `patch_table`       | `(ad, sr)` tuple                       | ref voice AD/SR regs; **immediate** (def emits on commit) |
| WAVETABLE| 65     | 66         | 67 (END)           | 68, **69 ONESHOT**     | `wavetable_table`   | RLE `(offset,hold)` program + loop     | voice freq reg, note-relative via `SKEL_LUT`; **queued multi-frame** |
| CTRL_WT  | 72     | 73         | STEP @subreg `VAL` | 74 (SET)               | `ctrl_wt_table`     | single ctrl byte                       | ref voice ctrl reg; **immediate** (def emits on commit) |

**The DEF→STEP→COMMIT→REF lifecycle, the id table, the mid-song seed/materialization
(`DecodeState._apply_seed` / `_SEED_TABLE_KEYS`), and the multi-frame drain (via
`state.pending_set_writes`) are identical across all four.** Only the *payload codec*
(how STEP rows serialize into an entry, how a REF replays an entry into register writes)
and the *replay target* differ.

The unifying abstraction **already exists and is already authoritative for two of four
facets**:

- `preframr_tokens/macros/op_contracts.py` defines `CodebookSpec(op_code, table, kind ∈
  {def,commit,ref}, subreg)` and `CODEBOOK_SPECS` (op → spec) and `CODEBOOK_TABLES`.
- `…/validators.py::validate_codebook_refs` / `codebook_live_ids` drive **validation**
  entirely off `CODEBOOK_SPECS`.
- `preframr_tokens/constrained_decode.py` imports `CODEBOOK_SPECS` for the sampling-time
  legality **mask** (a REF is legal iff its id is live).

But the spec does **not** drive decode (4 hand-coded decoders) or encode. Encode is 75%
unified: `…/codebook_emit.py::emit_recurring` (the MINE→GROUP→DEF-once→REF-per-occurrence
skeleton) is used by `stamp_pass.py`, `patch_pass.py`, `ctrl_wavetable_pass.py`;
`wavetable_pass.py` rolls its own (its codec `…/wavetable.py::factorise`/`unroll` is
already isolated and **shared between the encoder and `WavetableDecoder` "so they cannot
disagree"** — the model the rest should follow).

### Consequences (the problems to solve)

1. **Triple maintenance + drift.** Adding/changing a codebook family touches a decoder, a
   `CodebookSpec`, and an encoder, kept consistent only by tests.
2. **A real consistency bug class.** Decode and validation share the *spec* but not the
   *execution*: `validate_codebook_refs` **asserts** on a REF to a non-live id
   (`validators.py:305`), while every decoder **silently returns `None`** on the same
   condition (e.g. `decoders.py::StampDecoder._ref`, `PatchDecoder._ref`,
   `CtrlWtDecoder._ref`, `WavetableDecoder._replay`). Mask, validator, and decoder reason
   about liveness independently.
3. **Untyped state sprawl.** `DecodeState` carries ~13 codebook fields (4 `*_table` dicts +
   ~9 `pending_*`), differently shaped — hostile to the typed-state refactor that future
   numba work needs.

### Irregularities the generic machine MUST absorb (do not "simplify" away)

- **COMMIT trigger differs**: STAMP/WAVETABLE = dedicated END op; PATCH/CTRL_WT = a STEP at
  a terminal subreg. (`CodebookSpec.subreg` already models this.)
- **REF is itself multi-row in some families**: `WAVETABLE_REF` accumulates
  `ID/LEN_HI/LEN_LO/LEAD/LEADOFF…` subreg rows; `STAMP_REL_REF` accumulates `ID/BASE_HI/
  BASE_LO`. REF is an *assembled record*, not a single lookup row.
- **`WAVETABLE_ONESHOT` (69) is a table-less REF** — carries its payload inline, no id.
- **`STAMP_REL_REF` (63)** applies a transpose (signed base delta added at offset 0).
- **id rebind**: a later DEF with the same id overwrites the table entry.
- **def-emits-on-commit**: PATCH and CTRL_WT emit the def's own register writes at commit;
  STAMP and WAVETABLE buffer only.
- **Adjacent non-codebook ops share the sub-pattern but are NOT codebooks**: `SWEEP_OP`,
  `CTRL_OSC_OP`, `FREQ_TRAJ_OP` use the same "accumulate subreg fields → on terminal subreg,
  queue a multi-frame replay" assembler but have no id table. They are **out of scope** for
  this refactor (do not fold them into the codebook table), but the shared sub-primitive
  (`AtomAssembler`, below) is designed so they *can* migrate later.

---

## Architecture (target)

Single source of truth: a **`CodebookFamily` registry** that drives all four facets.

### New module: `preframr_tokens/macros/codebook.py`

```
AtomAssembler                      # Layer 1: subreg-field accumulator
    open(id_val, reg)              #   start a record
    feed(subreg, val) -> bool      #   accumulate; True when terminal subreg reached
    record() -> dict               #   the assembled field map
# Collapses the per-family `pending_*` reassembler state machines into one primitive.
# (FreqTraj/Sweep/CtrlOsc are future clients; not migrated here.)

@dataclass(frozen=True)
class CodebookFamily:
    name: str                      # "stamp" | "patch" | "wavetable" | "ctrl_wt"
    table_index: int               # index into CODEBOOK_TABLES
    def_op: int
    step_ops: frozenset[int]
    commit_op: int | None          # END op, else None
    commit_subreg: int | None      # terminal step subreg, else None
    ref_ops: frozenset[int]        # incl REL / table-less ONESHOT variants
    def_emits: bool                # emit def writes at commit (PATCH/CTRL_WT)
    serialize: Callable            # entry -> list[step-row dicts]   (ENCODE)
    accumulate: Callable           # (assembler.record) -> entry     (DECODE def commit)
    replay: Callable               # (entry, ref_record, state) -> writes  (DECODE ref)
    ref_assembler: Callable | None # builds the ref record (multi-row refs); None = single row

class CodebookDecoder(MacroDecoder):
    # ONE decoder class, instantiated per family. Owns the lifecycle:
    #   def_op       -> cb.open(id)            (table[id] cleared/pending)
    #   step_ops     -> accumulate; if commit-trigger -> commit (table[id]=entry; maybe emit)
    #   commit_op    -> commit
    #   ref_ops      -> assemble ref record; entry=table.get(id);
    #                   on miss: behavior governed by family (see DEAD_REF_POLICY below)
    #                   else family.replay(entry, ref_record, state)

CODEBOOK_FAMILIES: dict[str, CodebookFamily]   # the registry
def codebook_decoders() -> dict[int, MacroDecoder]   # op -> CodebookDecoder, for DECODERS
def codebook_specs() -> dict[int, CodebookSpec]      # derived; must equal the legacy literal
```

The four families' `serialize`/`accumulate`/`replay`/`ref_assembler` are **lifted verbatim**
from the existing decoders/encoders (Stamp `_offsets_in_order`/`_ref`/`_replay_rel`, Patch
`_emit`, CtrlWt `_emit`, Wavetable `factorise`/`unroll`/`_replay`/`_replay_oneshot`). No
behavior change — they move, they don't change.

### `DecodeState` state collapse

Replace the ~13 codebook fields with one structure:

```
self.codebooks = {i: _Codebook() for i in range(len(CODEBOOK_TABLES))}
# _Codebook: { table: dict[int, entry], pending_assembler, pending_ref_assembler }
```

During transition, keep `stamp_table` / `patch_table` / `wavetable_table` / `ctrl_wt_table`
as **read/write properties** backed by `codebooks[i].table` so `_apply_seed`,
`codebook_live_ids`, `resume_decode.py`, and any external caller keep working unchanged.
Remove the shims only after a repo-wide grep shows no remaining direct references.

### Migration strategy: strangler-fig with a differential equivalence gate

The legacy decoders are byte-exact and heavily tested — they are the **oracle**. Never break
them mid-flight. Build the new machine alongside, prove it produces **byte-identical** decode
output over a corpus, *then* switch `DECODERS`, *then* delete the legacy decoders. Every step
keeps the full suite green.

### Non-negotiable invariants (hold at EVERY gate)

- **Byte-exactness**: `expand_ops` output is identical pre/post for all inputs.
- **All pre-existing tests stay green** at every step (they are the oracle).
- **Torch-free** module load; **no `fastmath`**; **numba stays optional** (import-guarded).
- **Token alphabet unchanged** (no op-code or subreg renumbering — that would bump major
  version and invalidate checkpoints).
- One commit per green gate on the working branch (cheap rollback / bisection).

---

## Execution log / resume point

Branch `refactor/codebook-unify` off `main` @ a0a76e6. Done so far (each committed,
full-suite-certified at **930 passed, 1 known pre-existing failure**):

- **Step 0** — baseline recorded (`.codebook-refactor-base`), known-failure pinned.
- **Step 1** — `preframr_tokens/macros/codebook.py` registry + `tests/test_codebook_registry.py`
  (commit `13f6a52`). Derives `CODEBOOK_SPECS` from `CodebookFamily`; literal **not yet
  removed** from `op_contracts.py` (the test pins them equal instead).
- **Step 5 (partial)** — collapsed the four `*_table` dicts on `DecodeState` into
  `self.codebooks` (registry-indexed) with compat properties (commit `b6cc986`). The
  `pending_*` buffers are **not yet** collapsed (do that with the unified decoder).

**Resume here → Step 2** (differential/golden decode harness, additive), then the unified
`CodebookDecoder` (Step 1 decode-hooks + Step 3 switch), then Steps 4/6/7/8/9. Note the lint
gates: **no narrative `#` comments**, **docstrings ≤5 lines, no blank lines** (`tests/test_lint.py`)
— conform or the suite goes red.

## Implementation

> Execute these in order. `RUN` = run it. `GATE` = must pass before proceeding. `REPAIR` =
> what to do if the gate fails. After each GATE passes, `git add -A && git commit -m "<step>"`.

### Step 0 — Preflight & baseline

- RUN: `git checkout -b refactor/codebook-unify` (if it exists, `git checkout` it).
- RUN: `git fetch origin && git log --oneline origin/main -1` — record the remote main SHA in
  a new file `.codebook-refactor-base` (one line: the SHA). Used by the
  [main-advanced](#handling-remote-main-advancing) procedure.
- RUN baseline suite and **record the pass count**:
  `./run_tests.sh 2>&1 | tee /tmp/baseline_tests.txt` (if `run_tests.sh` is absent, use
  `python -m pytest -q -p no:cacheprovider`).
- GATE: baseline suite is green. Record N_pass.
- REPAIR: if baseline is already red, STOP and report — do not refactor on a red baseline.

> **Baseline state recorded at execution start (main @ a0a76e6, this env):**
> `925 passed, 3 skipped, 1 xfailed, 1 FAILED`. The one failure —
> `tests/test_sid_frame_diff.py::TestFrameDiffReleaseGate::test_deployed_and_stack_full_tune`
> — is a full-tune pitch-fidelity test on the all-passes "stack" path (in-flight
> RESID/skeleton fidelity, unrelated to codebook decode dispatch). It is **pre-existing**
> (zero source changes at branch creation) and **invariant** to this refactor (the
> differential gate compares new-vs-legacy decode). **Acceptance gate for every step
> below = no regressions beyond this single known failure** (i.e. ≥ 925 passed and this is
> the only failure). pytest-xdist is absent in this env → run serial:
> `pytest -q -p no:cacheprovider -p no:cov`.

### Step 1 — Family registry + generic machine (additive only)

- Create `preframr_tokens/macros/codebook.py` per [Architecture](#architecture). Lift the
  four families' codec/replay logic **verbatim** from `decoders.py`
  (`StampDecoder`, `PatchDecoder`, `CtrlWtDecoder`, `WavetableDecoder`) and the encoders.
  Register all four in `CODEBOOK_FAMILIES`.
- Do **not** touch `decoders.py::DECODERS`, `op_contracts.py`, or `state.py` yet.
- GATE: `python -c "import preframr_tokens.macros.codebook as c; print(sorted(c.CODEBOOK_FAMILIES))"`
  prints all four; `./run_tests.sh` still green (nothing imports the new module yet).
- REPAIR: import cycle? `codebook.py` may import from `op_contracts`, `state`, `stfconstants`,
  `wavetable`, `passes_base` only — never from `decoders` or the package root.

### Step 2 — Differential equivalence test (the oracle gate)

- Create `tests/test_codebook_machine_equivalence.py`:
  - Build a **corpus** of encoded token DataFrames exercising every family and every REF
    variant (REL, ONESHOT, dead-ref, id-rebind, mid-song seed). Reuse the builders/patterns
    in `tests/test_stamp_pass.py`, `tests/test_patch_pass.py`, `tests/test_wavetable_pass.py`,
    `tests/test_ctrl_wavetable_pass.py`. If `tests/sid_fixtures.py` fixtures are available,
    add HVSC-derived streams too; otherwise skip them (`FixtureUnavailable`).
  - For each encoded df, decode twice: once with the legacy `decoders.DECODERS`, once with a
    map where the codebook ops are overridden by `codebook.codebook_decoders()` (merge +
    monkeypatch `decoders.DECODERS` via `expand_ops`, or parameterize `expand_ops` to accept
    a decoder map — prefer monkeypatch to avoid touching `decode.py` yet).
  - Assert `pandas.testing.assert_frame_equal` on the two `expand_ops` outputs.
- GATE: `python -m pytest tests/test_codebook_machine_equivalence.py -q` is **green**.
- REPAIR: a mismatch means the lifted logic diverged. Diff the failing family's rows; the new
  `replay`/`accumulate` must match the legacy decoder byte-for-byte. Fix `codebook.py`
  **only** (never the legacy decoder). Common causes: `maybe_flush_for` ordering, `__pos`
  ordering, signed-byte sign extension, `pending_set_writes` insertion order.

### Step 3 — Switch decode dispatch

- In `decoders.py`, replace the four per-family decoder registrations in the `DECODERS`
  construction (the `_STAMP_DECODER` / `_PATCH_DECODER` / `_WAVETABLE_DECODER` /
  `_CTRL_WT_DECODER` blocks at the tail of the file) with
  `DECODERS.update(codebook.codebook_decoders())`.
- Keep the legacy decoder *classes* in the file for now (still imported by the equivalence
  test); just stop registering them.
- GATE: full `./run_tests.sh` green, pass count ≥ baseline N_pass.
- REPAIR: if a non-equivalence test fails, a caller imports a legacy decoder class directly —
  grep `grep -rn "StampDecoder\|PatchDecoder\|CtrlWtDecoder\|WavetableDecoder" preframr_tokens tests`
  and repoint to the family machine.

### Step 4 — Make the family registry authoritative for the spec

- Add `codebook.codebook_specs()` that derives the `op → CodebookSpec` map from
  `CODEBOOK_FAMILIES`. In `op_contracts.py`, build `CODEBOOK_SPECS` from it
  (`CODEBOOK_SPECS = codebook_specs()`), keeping the exported name/shape identical so
  `validators.py` and `constrained_decode.py` are untouched.
- Add a test `tests/test_codebook_registry.py::test_specs_match_legacy` asserting the derived
  specs equal the previous hand-written literal (paste the literal into the test as the
  expected value).
- GATE: `python -m pytest tests/test_codebook_registry.py -q` green; full suite green.
- REPAIR: mismatch → fix the family's `def_op`/`commit`/`ref_ops`/`subreg` fields until the
  derived spec matches the literal exactly.

### Step 5 — Collapse `DecodeState` codebook fields

- In `state.py`, introduce `self.codebooks` (keyed `_Codebook` per `CODEBOOK_TABLES` index)
  and replace the ~13 fields. Add compat **properties** `stamp_table`/`patch_table`/
  `wavetable_table`/`ctrl_wt_table` backed by `codebooks[i].table`. Update `_SEED_TABLE_KEYS`
  / `_apply_seed` to seed via `codebooks`.
- GATE: full suite green (esp. `test_resid_zero_integration`, `test_*` touching seed/resume).
- REPAIR: `grep -rn "pending_stamp\|pending_patch\|pending_wavetable\|pending_ctrl_wt\|_table"
  preframr_tokens` — repoint any direct field access to the new machine/`codebooks`.

### Step 6 — Delete legacy codebook decoders

- Remove `StampDecoder`, `PatchDecoder`, `CtrlWtDecoder`, `WavetableDecoder` classes and their
  `_*_DECODER` instances from `decoders.py`. Update `tests/test_codebook_machine_equivalence.py`
  to compare against a frozen golden (snapshot the legacy decode outputs to a fixture file in
  Step 2 so the equivalence test survives legacy deletion), OR convert it to a pure round-trip
  test (`expand_ops(encode(x)) == x`).
- GATE: full suite green; `grep -rn "class StampDecoder\|class PatchDecoder\|class CtrlWtDecoder\|class WavetableDecoder" preframr_tokens` returns nothing.

### Step 7 — Unify the encode codec (share serialize with decode)

- Point `wavetable_pass.py` and the `emit_recurring` callers at each family's
  `CodebookFamily.serialize`/codec so encode and decode share one serialization (the
  `wavetable.py` "cannot disagree" model, applied to all four).
- This step is **gated by the round-trip tests staying green**; if any family's encoder is too
  entangled to migrate cleanly here, leave it and record a TODO in this file's
  [Deferred](#deferred) section — do not risk byte-exactness for encode-side tidiness.
- GATE: full suite green.

### Step 8 — Close the consistency bug class + resilience guards

- Add `tests/test_codebook_consistency.py`:
  - **dead-ref agreement**: for a stream with a REF to a non-live id, assert the three facets
    agree — `validate_codebook_refs` raises **iff** the decoder drops/raises, per a single
    `DEAD_REF_POLICY` constant in `codebook.py` (pick the *current decoder* behavior — silent
    drop — as the policy, and make `validate_codebook_refs` honor it, OR flip both to raise;
    choose silent-drop to preserve existing decode behavior and adjust the validator’s
    contract test accordingly). Document the choice here.
  - **registry completeness** (resilience): iterate every op in `CODEBOOK_SPECS`; assert each
    belongs to a registered `CodebookFamily` with non-None `serialize`/`accumulate`/`replay`.
    Failure message: `"op {op} in CODEBOOK_SPECS has no CodebookFamily — register one in
    codebook.py"`. This is what catches a new family added upstream (see below).
- Promote the contract-tracing PoC: if `trace_contract_poc.py` exists in the repo root, fold
  its tracing-state idea into `tests/test_codebook_consistency.py` as a per-family
  `observed_writes ⊆ macro_contracts.CONTRACTS[pass].writes` check, then delete
  `trace_contract_poc.py`.
- GATE: `python -m pytest tests/test_codebook_consistency.py -q` green; full suite green.

### Step 9 — Final verification

- RUN, all must pass:
  - `./run_tests.sh` (full suite, ≥ baseline N_pass + new tests)
  - `black --check preframr_tokens tests`  (or `black preframr_tokens tests` then re-verify)
  - `pylint` per `pyproject.toml` enabled checks: `pylint preframr_tokens`
  - `pyright` (config `pyrightconfig.json`)
  - `PREFRAMR_NO_NUMBA=1 ./run_tests.sh` if such a guard exists, else skip — confirms no new
    hard numba dependency was introduced.
- GATE: every command above passes. Then `git commit` the final state. **Do not push or open a
  PR** unless the operator asked — leave the branch for review.

---

## New-macro integration guide

To add a fifth codebook family (e.g. a new percussion/instrument codebook):

1. Allocate its op codes in `stfconstants.py` (DEF/STEP/[END]/REF) — append, never renumber.
2. Add its name to `op_contracts.py::CODEBOOK_TABLES`.
3. Register one `CodebookFamily` in `codebook.py::CODEBOOK_FAMILIES` with its `serialize` /
   `accumulate` / `replay` (and `ref_assembler` if its REF is multi-row). The generic machine
   provides def/commit/ref lifecycle, the table, seeding, drain, validation, and the mask —
   **for free**.
4. That's it: decode, `CODEBOOK_SPECS`, validation, and the constrained-decode mask all pick
   it up automatically. `tests/test_codebook_registry.py` and `test_codebook_consistency.py`
   enforce that you supplied a complete family. Add a family-specific round-trip test mirroring
   `test_stamp_pass.py`.

---

## Handling remote `main` advancing

If `origin/main` has moved past `.codebook-refactor-base` (especially if new codebook ops/
families landed upstream), run this **before** Step 9 / when resuming:

1. `git fetch origin && git rebase origin/main` (resolve conflicts favoring upstream behavior;
   the refactor is structural, so prefer upstream's *semantics* and re-apply the structural
   move).
2. `python -m pytest tests/test_codebook_registry.py::test_specs_match_legacy -q` — if a new
   op appeared in the upstream `CODEBOOK_SPECS` literal, this fails. Update the test's expected
   literal to match upstream, then make `codebook_specs()` reproduce it.
3. `python -m pytest tests/test_codebook_consistency.py -k completeness -q` — fails for any
   upstream-added family with no registered `CodebookFamily`. For each, follow the
   [integration guide](#new-macro-integration-guide) (lift the new family's logic from its
   upstream decoder into a `CodebookFamily`). If a new *non-codebook* op appeared, it won't be
   in `CODEBOOK_SPECS` and is correctly ignored.
4. Re-run the differential/round-trip gate (Step 2 corpus, extended with the new family's
   builders) and the full suite. Update `.codebook-refactor-base` to the new SHA.

The registry-completeness test is the safety net: **upstream cannot add a codebook family that
this refactor silently mishandles** — it will fail loudly with the op codes to register.

---

## Deferred (explicitly out of scope for "execute agents.md")

- Migrating `SWEEP_OP` / `CTRL_OSC_OP` / `FREQ_TRAJ_OP` onto `AtomAssembler` (they share the
  assembler sub-pattern but have no id table). The machine is designed to allow it later.
- numba/JIT of the replay drain. Unblocked by the `DecodeState` collapse (Step 5) but not done
  here. Record any per-family `replay` that became a clean int-array kernel as a candidate.
- **Step 7 (encode-codec unification) deferred for STAMP / PATCH / CTRL_WT.** Reason: their encoders
  emit DEF/REF rows through `codebook_emit.emit_recurring`'s per-pass `emit_first` / `emit_ref`
  callbacks, which are bound to each pass's mining/grouping (id assignment, occurrence shape) and have
  no clean standalone "entry" object at emit time to hand a shared `serialize(entry)`. Extracting that
  is a separable refactor with real byte-exactness risk and modest payoff — the encode↔decode
  agreement is already enforced by each pass's byte-exact round-trip test plus the new golden
  (`test_codebook_machine_equivalence`) and contract-trace (`test_codebook_consistency`) gates.
  WAVETABLE already meets the Step 7 "cannot disagree" model: `wavetable.py::factorise`/`unroll` is the
  single codec shared by `WavetablePass` and the wavetable decode codec. Decode is fully unified and
  golden-pinned regardless, which is the refactor's core invariant.

### Decisions recorded during execution

- **DEAD_REF_POLICY = "drop"** (`codebook.py`). A REF to a non-live id is silently dropped at decode
  (preserving the legacy per-family decoders' behavior, now golden-pinned). The offline
  `validators.validate_codebook_refs` stays strict (raises) — a generated stream with a dead ref is
  illegal and must be caught before it is ever decoded. The two are not in conflict: liveness is now
  derived from one source (the registry-driven `CODEBOOK_SPECS`), and the lenient-decode /
  strict-validate split is pinned by `tests/test_codebook_consistency.py`
  (`test_dead_ref_decode_drops_silently` + `test_dead_ref_validator_raises`).
- `trace_contract_poc.py` was promoted into `tests/test_codebook_consistency.py`
  (`test_observed_writes_subset_of_contract`, the per-family `observed_writes ⊆ CONTRACTS[pass].writes`
  trace) and deleted.

## Rollback

Each step is its own commit on `refactor/codebook-unify`. To abandon: `git checkout main`. To
revert one step: `git revert <sha>`. The branch never touches `main` until the operator
reviews.
