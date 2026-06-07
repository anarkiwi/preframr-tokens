# NEXT BUILD — instrument-keyed additive freq model (handoff, UNTRACKED)

Working note for the next context. Branch: `feat/residual-gesture-decompose` on
`/scratch/anarkiwi/gen2-preframr-tokens`. Not committed on purpose — it's a design handoff,
not part of the byte-exact code. Delete once the build lands.

## How we got here (1 paragraph)

We built an encoding-explosion guard (`preframr_tokens/encoding_complexity.py`): per freq voice,
the encoded vocabulary must not exceed the input's structural alphabet, else the generator added
complexity instead of abstracting. It fired on the vibrato-fusion bug — one LFO carried as dozens
of near-singleton residual payloads. We wired the **recovered per-voice note→freq table**
(`pitch_grid.recover_table`) as the residual baseline (committed, byte-exact, full suite green):
that fixed the mis-calibration half, but the guard still fired. We then reverse-engineered the
actual Daglish driver (github.com/anarkiwi/reninja, the WEMUSIC engine) and reconciled it against
Hubbard's Commando driver and Linus's Cauldron-II-remix driver. The driver source is the spec, and
it tells us exactly what's missing.

## The three drivers are one model

`freq_voice_v(t) = table_v[note_v(t)] + modulation_inst(t)`  — additive, note is an INDEX,
modulation is grouped by INSTRUMENT. The only differences are the modulation generator and chorus:

| | note→freq | per-instrument modulation | chorus / detune | evidence |
|---|---|---|---|---|
| Daglish WEMUSIC | `freqtab` 95-note LUT ($B500/$B55F) | add/sub **triangle** delta+phase, per instrument (`freq_sweep_1`, delta at instr offset $0A/$0B) | intra-voice **osc2** 2nd accumulator (`freq_sweep_2`) | reninja SOURCE |
| Hubbard Commando | note LUT | **table** vibrato (sine waveform per-frame; measured 35 smooth step magnitudes, bounded +-49c) | — | register log |
| Cauldron II (Linus) | note LUT, note-index SHARED across chorus voices 1&2 | (vibrato) | **cross-voice** per-voice tuning detune (+12c, std 27c, phasing) | register log + `pitch_grid` chorus guardrail |

Driver record layout (Daglish, confirmed): ~12 instrument records (stride $2F=47B) each bundling
ADSR + PW + waveform + **its own vibrato delta/phase** + arp/wavetable ref; ~15 arp/wavetable
streams ($B006); note/pattern stream = 16-bit words `(note 0-63, instrument 0-~12)`, $0000=loop.
The vibrato is **one delta per instrument**, applied additively to `freqtab[note]`, reset on note-on.

## Where the encoding is NOW (measured, Last_Ninja voice 0)

- recovered table WIRED (commit on branch): byte-exact, residual payloads 68->35, static arps 0%->36% pure.
- guard still fires: encoded 143 > input 41. Breakdown of the 143:
  - **62 held-note/slide freq STARTS** (SWEEP/TRI, raw absolute freq) — should be ~14 note indices.
  - **27 arp/chord SHAPES** — should be ~15 instrument arp tables.
  - **34 residual payloads / 17 distinct VALUES** — should be ~few (per-instrument vibrato deltas).
- Mapping to the driver: I have `freqtab` (the recovered table) but NOT the driver's **instrument
  grouping** of the freq modulation. I factor the summed OUTPUT freq, which entangles
  note+vibrato+arp; the driver stores note-index + one delta per instrument. That entanglement is
  the whole gap.

## THE BUILD (covers all 3 cleanly)

Add the driver's missing abstraction: **instrument-keyed modulation on the freq channel**, mirroring
`InstrumentProgramPass` (which already groups ctrl/AD/SR by instrument). Three parts:

1. **Note-key ALL freq, not just onsets/arps.** Apply the recovered table to the bulk HOLD/ACCUM
   freq stream (the `universal_freq` path) so held notes/slides emit a note INDEX + residual, not a
   raw freq start. Collapses 62 -> ~14. (See "Why universal_freq wasn't wired" below.)
2. **Per-instrument modulation lane (the key new piece).** Group the per-frame freq residual
   (`freq - table[note]`) by the **instrument** the note plays, and fit ONE modulation generator per
   instrument from the existing set: `TRI` (Daglish triangle), `TABLE` (Hubbard sine waveform),
   `ACCUM` (slides). Emit it DEF->REF, reused across every note of that instrument. This collapses
   17 residual values -> ~(# instruments with vibrato). The instrument id should come from
   `InstrumentProgramPass`'s instrument codebook — the vibrato is a property of the instrument,
   exactly like ADSR/waveform in the driver record. Concretely: extend the instrument codebook entry
   to carry a freq-modulation generator alongside the ctrl/AD/SR program.
3. **Key arps by instrument.** The 27 output-fused shapes become ~15 instrument arp tables.

Cauldron's chorus needs no new mechanism: voices 1&2 share the note-index stream (one interval
stream) and differ only by their per-voice `table_v` (the +12c detune lives in the recovered table).
Daglish's osc2 detune is just a second modulation generator on the voice. Both already covered.

Net: the encoded vocabulary becomes the driver's own structural alphabet —
`note-index + per-voice table + ~12 instrument modulators + ~15 instrument arps` — and stops
re-deriving that structure per output segment.

## The guard's input measure must be corrected too

`input_freq_complexity` currently = `distinct notes + binned off-grid levels`. But the driver's own
representation has `notes + ~12 instruments + ~15 arps + per-instrument deltas`. So even the DRIVER's
minimal description exceeds the current input budget (notes+levels). The guard was unfairly strict:
its input side omits the instrument/arp structure the source is literally built from. Fix the input
measure to count `notes + instrument-programs + arp-patterns + modulation-alphabet`, or use the guard
as a RELATIVE regression detector (encoded vocab must not GROW vs the prior encoding — which it
correctly showed: 176 -> 143 -> 90 as we improved). Do not chase an absolute pass on the current
(incomplete) input measure — it's not achievable by any faithful encoding, including the driver's.

## Why universal_freq wasn't wired (the question asked)

`universal_freq` is a default-OFF flag (NOT in `REGISTERED_MACROS`) that re-keys every sounding
HOLD/ACCUM freq atom to a note index, with a documented ~4.5x interval-atom cost. When I wired the
recovered table I scoped it to the paths `universal_pitch` already touches — GEN_TABLE arps and
MELODY_INTERVAL onsets — because I believed the RESIDUAL payloads (68) were the dominant explosion
driver. The held-note STARTS (62) only showed up as the real dominant component in the final
per-component breakdown, after the table was already wired. And critically, `universal_freq` ALONE
only takes voice-0 encoded 143 -> 90 (starts 62 -> 29) — still an explosion — because note-keying
the bulk freq without the per-instrument modulation grouping just moves raw starts to note+residual
without collapsing the vibrato. So it's necessary but not sufficient; it belongs INSIDE the unified
build (part 1), not as a standalone flag flip. That's the honest reason: mis-scoped early, then
subsumed by the correct diagnosis.

## Concrete pointers for the build

- Recovered-table wiring (done) — encode: `generator_pass._melody_context` (computes table),
  `_voice_tuning_claim` (emits it as GEN_TUNING NOTE/FREQ codebook), `_melody_rows`/`_table_rows`
  (baseline = table[note]); decode: `decoders.GenTuningDecoder` (reads table), `MelodyIntervalDecoder`
  + `codebook._replay` (use table). All byte-exact (residual = freq - table[note] both sides).
- Instrument codebook to extend: `preframr_tokens/macros/instrument_program_pass.py` +
  `codebook.py` (`_InstrumentCodec`). The freq modulation should hang off the SAME instrument id.
- The generator fitter: `generator_fit.py` (`decompose`, `gen_tri`, `gen_accum`, `gen_table`) — reuse
  for the per-instrument modulation fit. Note `gen_tri` only matches EXACT constant-step triangles
  (Daglish); Hubbard's sine needs the periodic-TABLE matcher (`gen_table` on the residual), so fit
  the modulation with the FULL set, not just TRI.
- Guard: `encoding_complexity.py` (`input_freq_complexity` needs the structural terms;
  `encoded_freq_complexity` should count residual VALUES not payloads to match input levels).
- Ground-truth driver: github.com/anarkiwi/reninja (`docs/engine_annotated.txt` is the readable spec;
  `src/engine.asm` symbolic; `src/musicdata.asm` raw tables). freqtab=$B500/$B55F, instruments=$B1CE
  (stride $2F), arps=$B006, freq_sweep_1=vibrato.
- Verify: byte-exact via `parse_audit=raise` on a real tune + the full suite (725 passed baseline);
  then re-measure the guard on Last_Ninja (Daglish triangle), a Commando subtune (Hubbard sine), and
  a Cauldron-II-remix subtune (Linus chorus) — the three should all come in at ~the driver's
  structural count once the model is instrument-keyed.
