# preframr-tokens

The white-box SID decompiler codec. A C64 SID tune is **decompiled** into a
compact op-program — not a dense per-frame register stream — and that program
regenerates the original chip writes **byte-exact** (residual = 0). The whole
tune fits a small context window: every gate tune encodes well under 4096 tokens.
Extracted from the [preframr](https://github.com/anarkiwi/preframr) research
codebase.

Torch-free (numpy + pandas/pyarrow + [py65](https://pypi.org/project/py65/) for
the 6502 emulation, [pygoattracker](https://pypi.org/project/pygoattracker/) for
the GoatTracker backend). The training-side concerns (model, loss, DataLoader,
predict) live in the main `preframr` repo; this package is the encoding layer.

## The idea: `trace = VM(program)`

A SID register dump looks dense — per-frame vibrato, portamento, pulse-width
modulation, arpeggio, envelope ramps — but that density is *generated* by a tiny
fixed op-set running in the player. The frames are *playback*; the composer wrote
*steps*. So the codec recovers the **program**, never the per-frame trace.

### BACC — the bounded accumulator

The special-case effects (vibrato, slide, arp, PWM, ADSR, filter/pitch/pulse
sweeps) are not distinct generators. They are one primitive, the **bounded
accumulator (BACC)**:

> `value += rate every dwell frames`, with `boundary ∈ {wrap-N, reflect, none}`,
> `width ∈ {8, 12}` bits, and an output map `∈ {absolute, base+offset,
> note-table-scaled}` (or a table-walk).

They differ only in their parameters. A voice's whole modulation is one
straight-line PROGRAM of BACC ops. Per-instrument generators are
**pitch-invariant**: vibrato is a depth shift, arp a set of semitone offsets, PWM
a sweep rate — realized through the note table at render time, not stored
per-note. (Free-running modulation falls out naturally: phase is generator
*state*, not reset per note.)

Notes ride a **canonical 12-TET A440 grid**: the note token is the absolute
semitone index on a fixed A440 reference, computed from the onset frequency the
driver actually renders (`pitch.fn_to_grid`). So the same concert pitch is the
**same token across drivers** — a raw onset register write and GoatTracker's
note-table lookup both resolve to one grid index. This is what unifies the token
alphabet across engines.

The gate is **residual = 0, byte-exact**. A lossy codec trivially hits any token
budget, so the budgets only mean anything losslessly — `verify_residual` is the
invariant the codec is held to.

### The virtual machine / recover → render loop

The codec is a set of per-driver **backends**, selected by player fingerprint
(`backends/base.py: select_backend`). Each backend implements three methods:

- `matches(psid)` — does this backend handle the tune's playroutine?
- `recover(psid, nframes, subtune)` — run the playroutine white-box (py65,
  tapping driver RAM) and return a `BaccProgram` (score + pitch-invariant
  instrument generators + initial-state seed).
- `render(program)` — render the program back to a `(nframes, 25)` per-frame SID
  register array.

`recover_program` recovers the program; `render_program` renders it; and
`verify_residual` requires the rendered array to equal the ground-truth dump
**byte-exact** (modulo each backend's small declared don't-care mask, e.g. unused
PW-high bits). The shipped hand backend is **GoatTracker** (gt2reloc), kept as the
worked reference; every other tune is handled by the driver-agnostic **generic**
path (below).

### The flat v2 token alphabet (learnability-first)

The model-facing alphabet is **flat and typed** (`bacc/flat_serialize.py`,
`VOCAB = 576`): a token id's RANGE encodes its kind, so a decoder-only LM never
decodes place value or infers type from grammar position.

- `NOTE_*` — one token per canonical A440 12-TET grid index (pitch proximity →
  embedding proximity), plus `NOTE_REST` / `NOTE_KEYOFF` / `NOTE_KEYON` /
  `NOTE_RAW`. The **same concert pitch is the same `NOTE` id across every driver**.
- `INSTR_REF` / `CMD` / `BYTE` — one token per instrument ordinal / effect / byte
  value; a 16-bit field is a fixed `(lo, hi)` BYTE pair (positional, never a
  varint).
- A widened structural block (`0..63`) carries `BEGIN/END` brackets, the
  `ORDER_*` orderlist ops (incl. `ORDER_CALL`), the `GEN_*` generator enum
  (`GEN_HOLD/RAMP/QUAD/VIBRATO/ARP/TABLEWALK`), and the inline `REF`.

The alphabet has **no LZ / back-offset token and no wide-value escape** (the gate's
C3 / C8 invariants): repetition is content-addressed, not a back-reference.

Note: the *streams* differ between backends (they recover structurally different
decompositions); only the *alphabet* and the *note-pitch tokens* are shared.

### Inline, define-at-first-use layout (prefix-valid)

The GoatTracker flat codec is laid out **inline**: a small front block (header +
the four generator-parameter tables), then the orderlist walked in **play order**
with patterns and instruments **defined at first use** and referenced (`REF` /
`INSTR_REF`) thereafter — backward-reference only, no forward declaration, no
front-loaded section preamble. A def is always to the LEFT of any ref, so **any
prefix cut at an event boundary is itself a valid, decodable, continuable song**.
The pre-v2 base-16 LEB + inline-`REPEAT`/`TRANSPOSE`-LZ alphabet is retired (the
generic path still uses it internally pending its flat port; see the source).

## Two-file input — and the shipped `.sid`-only path

The hand-backend codec is a **two-file** codec: it takes a `.sid` (the player +
song data) and a per-frame register `.dump.parquet` of the ground-truth chip
writes. It recovers the program by emulating the `.sid`, then verifies the render
against the dump. That dump is produced offline by the `anarkiwi/headlessvice`
VICE container.

The **generic** recovery now closes the migration to a **single input file**:

```python
from preframr_tokens.bacc.generic import recover_from_sid

# .sid ALONE -> BaccProgram(driver="generic"); NO .dump.parquet
program, resid, dump = recover_from_sid("Grid_Runner.sid")
assert sum(resid.values()) == 0   # whole-tune, all 25 registers, byte-exact
```

`recover_from_sid` runs the deterministic `preframr-sidtrace` tool ONCE over the
`.sid`, which emits BOTH the per-frame register dump (`.sidwr.bin`) and the bus
trace (`.bus.bin`) in-process. The generic fitter recovers the program from the
bus trace, and the render is verified residual-zero against the SAME-run dump —
the two are internally self-consistent (same emulator, same run). Validated
byte-exact on a range of drivers (e.g. **Grid_Runner**, GoatTracker) from the
`.sid` alone — one driver-agnostic path.

The binary is located via `SIDTRACE_BIN` (env), so the whole-tune residual-zero
test is **skip-if-binary-absent**: the default CI gate runs the committed-dump
oracle in a container with no `preframr-sidtrace` binary and stays self-contained
(no new hard external dependency).

## Install

```bash
pip install preframr-tokens
```

## Quick start

```python
import preframr_tokens as P

sid, dump = "Grid_Runner.sid", "Grid_Runner.1.dump.parquet"

# Recover the BACC program, verify it's byte-exact, serialize to token ids.
assert P.verify_residual(sid, dump, P.CPF)        # residual = 0 (lossless)
program = P.recover_program(sid, dump, P.CPF)     # .sid + .dump -> BaccProgram
ids = P.program_to_ids(program)                   # model-facing token id stream

breakdown, frames = P.measure(program)            # {block: tokens}, frame count
print(breakdown["total"], "tokens", "/", frames, "frames")

state = P.render_program(program)                 # BaccProgram -> (nframes, 25)
prog2 = P.ids_to_program(ids, driver=program.driver)  # round-trips byte-exact
```

## Public API

`preframr_tokens.__all__` is the promised surface; everything under
`preframr_tokens.bacc.*` and `preframr_tokens.codec.*` is internal and may move
between releases — depend on the root re-exports.

- `recover_program(sid, dump, cpf=CPF, subtune=0)` — `(.sid, .dump) → BaccProgram`.
- `render_program(program)` — `BaccProgram → (nframes, 25)` register state.
- `verify_residual(sid, dump, cpf=CPF, subtune=0)` — `True` iff render == dump,
  byte-exact (the gate).
- `program_to_ids(program)` / `ids_to_program(ids, driver=...)` — the model-facing
  token id stream (round-trips byte-exact to the program).
- `measure(program)` — `({block: tokens}, nframes)`; `breakdown["total"]` is the
  pre-BPE token count.
- `VOCAB` / `PAD_ID` — token alphabet size and the reserved padding id.
- `per_frame_state(dump, cpf, maxframes)` / `CPF` / `NTSC_CPF` /
  `cpf_from_meta(prefix)` — the dump reader + PAL/NTSC frame clock (`CPF` = 19656
  PAL cycles/frame, `NTSC_CPF` = 17095).

## The input dump format

A raw tune's register dump is a `.dump.parquet` of register writes captured from a
SID player, with at least the columns `clock` (absolute φ2 cycle), `reg` (0..24),
`val` (byte written), and `chipno` (only chip 0 is read). `per_frame_state`
reconstructs the 25-register state per absolute frame at the tune's frame clock.

Register map (`base = voice * 7`): +0/+1 freq lo/hi, +2/+3 pulse-width lo/hi,
+4 control (bit0 GATE, bit3 TEST, bits4–7 waveform), +5 AD, +6 SR. Globals:
21/22 filter cutoff lo/hi, 23 resonance/routing, 24 mode/volume.

## The gate / tests

The reference budget gate is
`tests/test_goattracker.py::test_grid_runner_context_budget`: it recovers the BACC
program from *Grid_Runner* (Jammer, GoatTracker), asserts `verify_residual` is
`True`, and requires both `< 1 token/frame` and the whole song `< 8192` tokens. Its
fixtures are auto-acquired (the `.sid` is downloaded, the dump rendered). The
generic path is exercised end-to-end (whole-tune residual-zero) in
`tests/test_generic_recovery.py` and `tests/test_sid_only_recovery.py`.

Run the full gate (black, pytest, pylint, pyright, coverage) with:

```bash
./run_tests.sh
```

## Where the design narrative lives

The canonical, end-to-end narrative — how the codec landed, the op-set grounding,
and the cross-driver design — lives in the sibling `preframr-xpt` repo: start at
its `AGENTS.md`, then `design/encoding/` (`sid_player_decompiler.md`,
`sid_opset_inventory.md`, `cross_driver_note_unification.md`).

## Stability

Pre-1.0 releases may break API as the codec evolves. Token-alphabet shape changes
bump the major version since they invalidate downstream checkpoints.

## License

Apache 2.0. See `LICENSE`.
