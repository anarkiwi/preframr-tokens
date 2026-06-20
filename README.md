# preframr-tokens

The white-box SID decompiler codec: a register dump (`.dump.parquet`) becomes an
inline op-program token stream, **residual-zero** (decode reproduces the dump
byte-exact) at **< 1 token / frame**. Extracted from the
[preframr](https://github.com/anarkiwi/preframr) research codebase.

Torch-free (numpy + pandas/pyarrow only). The training-side concerns (model,
loss, DataLoader, predict) live in the main `preframr` repo; this package is the
encoding layer.

## Install

```bash
pip install preframr-tokens
```

## Importing

Import from the package root:

```python
from preframr_tokens import per_frame_state, CPF, measure, verify_residual
```

`preframr_tokens.__all__` is the promised surface. Everything under
`preframr_tokens.codec.*` is internal and may move between releases — depend on
the root re-exports.

## Quick start

```python
import preframr_tokens as P

dump = "Monty_on_the_Run.1.dump.parquet"          # a SID register dump (chip 0)
state = P.per_frame_state(dump, P.CPF, 1_000_000)  # (n_frames, 25) per-frame reg state
assert P.verify_residual(state)                    # lossless: decode == dump, byte-exact
breakdown, frames = P.measure(state)               # token-count breakdown + frame count
print(breakdown["total"] / frames, "tokens/frame")
```

## The input dump format

A raw tune is a `.dump.parquet` of register writes captured from a SID player,
with at least the columns `clock` (absolute φ2 cycle), `reg` (0..24), `val`
(byte written), and `chipno` (SID chip; only chip 0 is read). The 25-register
state is reconstructed per absolute frame at the tune's frame clock
(`CPF` = 19656 PAL cycles/frame, `NTSC_CPF` = 17095; `cpf_from_meta` selects from
a `.meta.txt` sidecar).

Register map (`base = voice * 7`): +0/+1 freq lo/hi, +2/+3 pulse-width lo/hi,
+4 control (bit0 GATE, bit3 TEST, bits4–7 waveform), +5 AD, +6 SR. Globals:
21/22 filter cutoff lo/hi, 23 resonance/routing, 24 mode/volume.

## The codec (inline op-program)

The trace is the output of a tiny deterministic playroutine; the codec recovers
the GENERATOR rather than encoding the per-frame output. Each SID lane (freq ×3,
pulse-width ×3, ctrl/AD/SR ×3, filter globals) is decomposed into a small fixed
op-set and serialized as an inline, time-ordered event stream
`(start_frame, lane, op)` with implicit holds dropped:

- **freq lanes** use an absolute-anchored 12-TET pitch encoder: `NOTE(interval)`
  (a relative semitone step, the small model-facing alphabet) plus parametric
  modulation generators — `VIB` (triangle vibrato), `SLIDE` (linear portamento),
  `ARP` (fixed-interval arpeggio) — and `RAW`/`REST`/`MOD` byte-exact fallbacks.
- **non-freq lanes** use `LOAD(value)` / `RUN`/`WALK` parametric sweeps; sweep
  shapes are shared in a rate-pattern codebook with dwell vectors in a global LZ
  side-stream.

Reuse is **backward-looking only** (inline LZ over events); there is no preamble,
no frozen table, and no forward declaration. Any prefix cut at an event boundary
is itself a valid, decodable, continuable song.

### Residual-zero is the gate

`verify_residual(state)` returns `True` iff the codec's structures reconstruct
the dump byte-exact. This is the invariant the codec is held to — a lossy codec
trivially hits any token budget, so the < 1 token/frame economy only counts when
residual = 0. The permanent gate `tests/test_monty_context_budget.py` asserts
both on the full Rob Hubbard *Monty on the Run* dump and may never be skipped.

## Public API

- `per_frame_state(dump, cpf, maxframes)` — dump (`.parquet`) → `(n_frames, 25)`
  per-frame register state (the codec input).
- `CPF` / `NTSC_CPF` / `cpf_from_meta(prefix)` — the PAL/NTSC frame clock.
- `measure(state)` — `(breakdown, frames)`; `breakdown["total"]` is the pre-BPE
  token count.
- `verify_residual(state)` — `True` iff decode == dump, byte-exact.

## Stability

Pre-1.0 releases may break API as the codec evolves. Token-alphabet shape changes
bump the major version since they invalidate downstream checkpoints.

## License

Apache 2.0. See `LICENSE`.
