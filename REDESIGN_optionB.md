# REDESIGN — Option B: the event/tracker token model (escape-free, factored, corpus-global)

Status: DESIGN (no code yet). Supersedes the register-row gesture overlay shipped on
`feat/mdl-optimal-parser`. This is the reference for the build. Every decision below is backed by a probe
in the repo root (see §10); rerun any with `--all` to confirm at full corpus scale.

The current implementation is byte-exact and the flow-test/perf bugs are fixed, but it falls short of the
goals: it produces a **2.4× expansion** (Grid Runner: 328k pre-Unigram rows vs 136k raw writes), a
**per-tune** codebook (polysemous, glitch-token ids), and a **verbatim ctrl/ADSR** stream. Patching the
serialization won't fix the substrate mismatch. This redesign keeps the proven *engine* and replaces the
*representation*.

---

## 0. Code map & integration (read first)

**New module:** `preframr_tokens/events/` — `schema.py` (the Event dataclasses + KIND/field enums, §3.5),
`encoder.py` (§8 recovery → events), `decoder.py` (§7 events → ordered writes), `tokenize.py`
(events ↔ token ids, feeds `regtokenizer` BPE), `grammar.py` (the event-grammar mask, §7.1). The event
stream is a typed structure (a `list[Event]`), **not** a `make_row` DataFrame.

**Ground-truth oracle (must be built — does not exist yet):** the ordered write stream is the **raw dump
rows** `(clock, irq, chipno, reg, val)` filtered to `chipno==0` and sorted by `clock` — that *is* the
ordered `(frame, reg, val)` list. `audit_primitives.register_state` returns **settled `(n,25)` snapshots
only** (verified) — it is the *secondary* check, NOT the oracle. Add `events/oracle.py: ordered_writes(df)`.

**Splice point:** `reglogparser.py:956` runs `for macro_pass in (MdlGesturePass(),):` after
`_combine_regs` (`:940`). For the freq/PW/CTRL/ADSR/filter/vol channels, the event pipeline **replaces**
that: encode from the raw ordered writes (not the combined row df). `frame`/`DELAY` structure and the other
register channels (if any) are handled as today until subsumed.

**Reuse (keep):** `mdl_core` (HOLD/POLY/PERIOD primitives, in the joint DP §8.4 and scalar parse §8.5);
`pitch_grid` (note-table recovery §8.2); `audit_primitives.register_state` (secondary check); the reSID
renderer in `../preframr-audio` (`render_to_samples`). **Retire** for these channels: `mdl_gesture_pass`,
`codebook._GestureCodec`, GESTURE_* ops, `codebook_emit`, the per-tune `bank`, and the arbiter/Claim flow.

**Structural passes (DECIDED — measure-first, BPE-only v1):** retire the register-level structural row
passes for these channels. **v1 relies on BPE over the event stream** for all repetition (loop/pattern/
dedup) and on zig-zag note-intervals for key-invariance (transpose). Do **not** port loop/pattern or
voice-block/coarsen to the event stream in v1. After the collapse/bits measurement (§9.3 / task #2), add
explicit loop/pattern ops **only if** the numbers show BPE leaves long-range repetition (e.g. a chorus
repeated across hundreds of events) uncaptured. This is the one integration decision, now resolved; no
maintainer follow-up needed.

---

## 1. Diagnosis (why redesign, not patch)

Two design choices — not bugs — defeat the goals:

1. **The model token = a register-write row `(op, reg, subreg, val)`.** Gestures are an overlay on a
   register-log substrate, so a gesture instance costs ~9 rows and a plain held note ~13 tokens → net
   expansion. The substrate is a write-list; a Transformer wants an event-list.
2. **The dictionary is per-tune.** Ids are positional, so the vocab token `(GESTURE_REF, reg, ID, cb_id)`
   means a different shape in every tune → polysemous, under-trained "glitch" embeddings + an in-context
   DEF→REF copy each tune (the induction-head failure §7.2 of the original spec warns about).

Keep: the MDL primitives (HOLD/POLY/PERIOD) and `audit_primitives` for reductions. NOTE: the arbiter's
`register_state` "byte-exactness" check is **end-of-frame settled state only** — it is *not* the fidelity
oracle (§2.8); the redesign's oracle is the ordered write stream (§7).

---

## 2. Principles (the invariants the build must honor)

1. **Decouple representations.** The model reads/writes a *musical event stream*; the register-write df
   exists only as the byte-exact ground-truth oracle. A decoder (driver-replay) expands events → exact
   registers. (Today they're the same object — the core mistake.)

2. **Complete encodings, NEVER "common case + escape."** *(Hard invariant.)* An escape hatch is a second,
   rarely-emitted representation for the tail — under-trained, special-cased in constrained-decode, and a
   glitch source exactly when the model strays onto it. Forbidden. Instead every quantity has **one
   representation that exactly covers all values over one bounded alphabet**; rare values cost *more
   tokens of the same alphabet*, never a different path.

3. **Compression comes from BPE/Unigram over a complete encoding — not a dictionary+escape.** Frequent
   digit/field sequences fuse into single learned tokens (the "dictionary" benefit), while rare ones stay
   as digits in the same alphabet (no singletons, no escape). This is the escape-free realization of the
   "corpus-global dictionary" goal.

4. **Factor into orthogonal fields; do not fuse.** Joint (ctrl × envelope) is only weakly coupled
   (§5.3); fusing *tripples* the vocabulary. Keep fields separate and let attention learn the soft
   correlations.

5. **Decoder does all arithmetic; the model selects/copies low-cardinality fields** (note-interval,
   musical length, nibbles). Anchor to recovered *musical units* (tick, note-table) for tempo/key
   invariance and generalization.

6. **No note-off, ever** (§4) — derived at `onset + duration`.

7. **One time encoding for everything.** There is only one kind of time (frame spans), so *every* time
   quantity — note duration, gesture span, PERIOD/LFO cycle, arp step rate, rest/inter-onset gap — uses
   the **same mixed-radix `frame→tick` scheme** (§4) over one digit alphabet. No raw-frame fields, no
   second time encoding anywhere.

8. **Fidelity = the exact ordered register-write stream, NOT end-of-frame settled state.** *(Hard
   invariant. The single most important correction — assuming `register_state` settled state was the
   oracle cost real work.)* The decoder must reproduce the source's **ordered sequence of register writes**
   — frame order, **intra-frame write order across registers**, and same-register repeats within a frame —
   because that order is audibly significant (hard-restart, gate vs ADSR ordering, test/gate pairs).
   `register_state` (the settled `(n,25)`) is a **weak secondary check only**; it is *insufficient* as the
   oracle and must never again be treated as the fidelity target. The reconstruction mechanism is
   **resolved (§13)**: same-frame writes are sequenced by a per-frame order descriptor (a complete,
   BPE-compounded field) — escape-free and exact regardless of order consistency.

---

## 3. The event schema — a NEW representation (the model's I/O)

The model reads/writes an **event stream**, a brand-new representation distinct from the register-write df
(which exists only as the byte-exact ground-truth oracle, §7). **There are no dictionary ids, no DEF/REF,
no per-tune bank anywhere** — every field is its *complete value* over a small fixed alphabet, and
BPE/Unigram fuses the frequent value-sequences into single learned tokens (principle §2.3). That is the
escape-free realization of "corpus-global dictionary," and it removes the per-tune-id polysemy by
construction.

### 3.1 Stream structure

A single time-ordered sequence of **events**. Each event is a small group of field-tokens led by a
**`DT`** (delta-time = frames since the previous event, mixed-radix §4 — this is the *only* timing
mechanism: rests, onsets, sub-note steps are all just `DT`) and a **`VOICE`** tag (0–2, or `GLOBAL` for
the shared filter/volume lane). Event kinds:

**`NOTE_ON`** — one per gate-on edge (no note-off; gate-off derived at `onset + duration`, §6). Fields:

| field | encoding (complete value, BPE-compounded) | evidence |
|---|---|---|
| `note-interval` | zig-zag interval from the voice's previous note | small, centered |
| `attack` | the **ordered, interleaved onset write-sequence** — per write `(Δframe [§4 time scheme], field∈{CTRL,AD,SR}, value)`; values are CTRL waveform bytes and ADSR nibbles (incl. the hard-restart constants **and** the real/sustained AD/SR — the last AD/SR in the attack IS the sustained envelope); BPE compounds recurring attacks. Captures hard-restart AD/SR **and** the CTRL↔ADSR interleaving exactly (§8.3). | 47% of notes have hard-restart; field-order top 605 = 90%, singletons 0.04% of notes; onset ≤6 writes for 51% |
| `duration` | mixed-radix `q·tick + r` (§4) | H(q)=2.44, H(r)=1.58 bits |

The `attack` **subsumes the old "ctrl-program + static envelope" pair** (which could not reproduce the
interleaved hard-restart writes — the §8.3 hole). **There is no separate `envelope` field and no `DEFAULT`
token:** the sustained envelope is the attack's final AD/SR, *held by the decoder* until the next note;
the "factored envelope" benefits (perceptual embedding-tying §5.2, instrument-default reuse) come from BPE
compounding the recurring attack (which carries its envelope), not from a separate field. Within the
attack, CTRL and AD/SR remain **separately-typed tokens** (waveform bytes vs nibbles) — BPE compounds them
softly, so this is *not* the fused (ctrl×env) cross-product id §5.3 forbids; rare combos decompose into the
shared CTRL/nibble sub-tokens.

**`NOTE_STEP`** — a pitch change *under the held gate* (arp / legato / slide; pitch moves ~3.4×/note,
§6). Field: `note-interval` (zig-zag). `DT` says when within the note.

**`MOD_FREQ` / `MOD_PW` / `MOD_CTRL`** — a *per-voice* gesture on the freq-delta / pulse-width / **body
waveform** layer. `MOD_CTRL` homes the post-onset CTRL writes (a looping wavetable past the attack window —
e.g. an alternating waveform held through the note); for a constant waveform there are no body CTRL writes
and no `MOD_CTRL`. Fields: the **complete gesture shape** (kind ∈ HOLD/POLY/PERIOD; POLY only for the
numeric freq/PW channels — for POLY the degree + N-th difference value tokens; for PERIOD the cell value
tokens; CTRL uses HOLD/PERIOD over waveform bytes — *no id*, BPE compounds frequent shapes), an `anchor`
(the gesture's start value — freq-delta base, PW/CTRL initial value), and a `span` that **defaults to the
enclosing note's duration**, emitted (mixed-radix) only when shorter (§4.1). Mid-note AD/SR changes (the
~23% non-static notes, §5.1) are emitted as a short `attack`-form write run at their `DT`, same mechanism.
PW changes are ~78% every-1-frame, ~94% tick-gridded.

**`GLOBAL` events (voice-less) — filter and master volume are shared resources, NOT per-voice.** SID has
ONE filter and ONE master volume for all three voices: reg 21/22 = filter cutoff (11-bit), reg 23 =
resonance (hi nibble) + filter routing (which voices/ext pass through the filter), reg 24 = filter mode
(HP/BP/LP) + master volume (lo nibble). These live on a **single global lane** (`VOICE = GLOBAL`):
- `MOD_CUTOFF` — a cutoff gesture (HOLD/POLY/PERIOD, same complete-shape encoding), span/anchor as above.
- `FILTER_CTL` — resonance + routing-bits + mode-bits (small complete-value fields; change rarely).
- `MOD_VOL` — master-volume gesture (usually constant; HOLD).
The decoder applies these to the shared filter/volume registers. They are time-ordered into the stream
like any event (with `DT` + `VOICE=GLOBAL`).

### 3.2 Headers (decoder state, recovered by the encoder, emitted on first use / change)

Per voice: `TICK` (+ groove offset, §4), `TUNING` + `NOTE_TABLE` (freq base). Plus the per-tune **canonical
per-frame write order** for the order descriptor (§7.0). These are complete-value fields (BPE-compounded),
not ids; they set the anchors the decoder's arithmetic uses and are re-emitted only on change (e.g., a
tempo change re-emits `TICK`). (No instrument-default header — the `attack` carries its own envelope; reuse
is a BPE effect, §3.1.)

### 3.3 What is NOT in the schema

No fused instrument token (§5.3 — fusing tripples the vocabulary). No note-off. No dictionary id / DEF /
REF. No separate envelope field / `DEFAULT` (subsumed by the `attack`, §3.1). No literal/escape op (§2.2).
The "instrument" is the *correlation* the Transformer learns across the co-located `attack` (its CTRL +
ADSR tokens), `note-interval`/pitch, and `duration` (MI ctrl↔env ≈4.4 bits, pitch↔env ≈1 bit, all soft).

### 3.4 Token-level serialization

Each field is one or more tokens drawn from small per-field alphabets (KIND enum, VOICE 0–2/GLOBAL, zig-zag
interval bytes, CTRL waveform values, ADSR nibbles 0–15, mixed-radix digits, order-descriptor reg ids).
The concrete alphabets/op table is §3.5. The Unigram/BPE layer (`regtokenizer`) then compounds frequent
field-sequences — e.g. `NOTE_ON · DT=1tick · interval=0 · attack=[CTRL 41] · dur=1tick` becomes a single
learned token. `constrained_decode` is replaced by an **event-grammar mask** (§7.1): a finite-state grammar
over event kinds + their field sequences, registry-driven like the current contracts, so generation can
only emit decodable events.

### 3.5 Concrete token alphabets (first-cut; bases delegated per §12)

Every event = `[DT][VOICE][KIND][fields…]`. Token families and alphabets:

| token family | alphabet | notes |
|---|---|---|
| `KIND` | NOTE_ON, NOTE_STEP, MOD_FREQ, MOD_PW, MOD_CTRL, MOD_CUTOFF, FILTER_CTL, MOD_VOL, TICK, TUNING, NOTE_TABLE, ORDER | one token; drives the grammar |
| `VOICE` | 0, 1, 2, GLOBAL | one token |
| `DT_Q` / `dur_Q` / `span_Q` | digits 0–15 (base 16), high bit = "continue" | mixed-radix `q` (§4); 1 digit covers 90% |
| `DT_R` / `dur_R` / `span_R` | signed −4..+4 | sub-tick `r` (§4); `{−1,0,+1}` = 99.2% |
| `INTERVAL` | zig-zag byte 0–255 (+continue for rare large) | note-interval |
| `attack`: `ATTACK_LEN` + per-write `(Δf 0–3, FIELD∈{CTRL,AD,SR}, value)` | value: CTRL byte 0–255; AD/SR = 2 nibbles 0–15 | §8.3; BPE compounds whole attacks |
| `SHAPE_KIND` | HOLD, POLY, PERIOD | MOD_* gestures |
| `DEGREE` 0–3 + `DIFF` value tokens | POLY (numeric freq/PW only) | |
| `PERIOD_LEN` + cell value tokens | cell: signed-16 lo/hi (numeric) or waveform byte (CTRL) | |
| `ANCHOR` | signed-16 lo/hi (freq delta) / unsigned (PW/CTRL/cutoff) | gesture start value |
| `ORDER` | a reg-id list (0–24) **or** the per-tune `CANONICAL` token | per-frame steady order (§7.0); `CANONICAL` dominant → BPE≈0 |
| `FILTER_CTL` | res / routing / mode nibbles 0–15 | global filter ctl |
| headers | TICK 2–32, offset signed, tuning const, note-table deviations | small ints (§3.2, §8.1–8.2) |

All alphabets are bounded and complete (§2.2); `grammar.py` enforces field arity, ordering, and `_Q`
continuation. **Soft spots the agent finalizes (no research, see §12):** exact `ORDER` encoding (reg-list
vs permutation index), `_Q` base + endianness, the attack-window close rule (§8.3), whether `ANCHOR` is
omitted when inferable. None affect correctness — only token shape.

---

## 4. Duration — mixed-radix, escape-free

`duration = q·tick + r`, with `q` = whole ticks (≥0), `r` = sub-tick frames, `tick` = per-voice header.

> **Unit note (§13):** measurements used `irq` (video frame) as the base unit. **~9.5% of tunes are
> multi-speed** (player runs N×/irq); for the 90.5% single-speed majority irq = player-call, so the tick/
> duration numbers stand. For the multi-speed minority, re-confirm tick recovery per player-call. The
> mixed-radix *scheme* is unaffected; only the unit may change for that minority.

- **tick** recovered by grid-fit (largest unit in 2–32 frames maximizing on-grid mass); observed ticks
  are 2–8 frames (player speeds).
- **q** = the musical note length, small fixed base; ~1 token for the 90% that are ≤16 ticks, 2 digits of
  the *same* alphabet for longer notes. Tempo-invariant (a quarter note is one token regardless of tempo).
- **r** = sub-tick remainder; alphabet `{-1,0,+1}` covers **99.2%** of notes; a per-voice groove offset
  makes it near-constant (dominant-r = 73.8% within a voice).

This is **exact for every frame count by construction** (verified: 100%). The cases we were going to
"escape" become ordinary low-entropy structure, *not* a separate path:

| former "escape" | now |
|---|---|
| systematic gate-held `N·tick−1` | r = −1 (a per-voice constant; BPE folds it in) |
| swing | r alternates between two values (learnable 2-state) |
| sub-tick percussion blip | q=0, r=blip length — same two fields (15.8% of notes) |
| multi-speed section | r less concentrated — still exact, no new path |
| very long note | q is 2 digits of the same base |

Measured: H(r)=1.58 bits, H(q)=2.44 bits → ~4 bits total, two tiny alphabets, no escape.

### 4.1 This is THE time encoding — gestures use it too

Every time quantity in the system uses this same mixed-radix `q·tick + r` scheme over one digit alphabet
(principle §2.7) — there is no second time encoding and no raw-frame field anywhere:

- **gesture span** (how long a freq-delta / PW / filter / ctrl-program gesture runs) — same scheme.
  Replaces the old raw `LEN_LO/HI` REF subregs entirely.
- **PERIOD / LFO cycle length, arp step rate** — same scheme; these sub-tick rates simply land in the
  frame digit (`q=0`) instead of the tick digit. Exact regardless of tick-alignment (escape-free holds —
  a non-tick-aligned LFO period is just a `(q,r)` with nonzero `r`, never a special path). Replaces the
  old `nbits(p)` period encoding.
- **rest / inter-onset gap** — same scheme.

**Span inheritance:** most freq-delta / PW gestures run the full gate-hold, so a gesture whose span equals
the note's duration emits **no length token at all** — it inherits the note's `duration`. An explicit
mixed-radix span is emitted only for a gesture shorter than the note. This removes the per-gesture length
field for the common case.

Measured (mod-period probe, §10): modulation updates are **78–79% every-1-frame** (wavetable steps →
`(q=0, r=1)`, the frame digit) and **94–95% tick-gridded**. So modulation rates fit the one scheme cleanly
— mostly the smallest unit, the rest on the tick — confirming §2.7. Correctness is unaffected regardless,
since mixed-radix represents any frame count exactly (escape-free).

---

## 5. Envelope, instrument factoring, and pitch

### 5.1 Envelope = 4 nibbles (complete), BPE-compounded
The 4 SID nibbles (Attack, Decay, Sustain, Release) are already a **complete** encoding of all 65,536
envelopes over a 16-value alphabet — not an escape, the whole space. Emit them always; BPE compounds the
~900 frequent envelopes (90% coverage) into single learned tokens; rare envelopes are just less-common
nibble patterns. No dictionary, no escape, no singletons.

The **sustained** envelope is a **static per-note value**, not a shaping gesture: 77% of notes hold one
(AD,SR) the whole gate, and treating it as a trajectory *increases* diversity (19k > 11k). It is the final
AD/SR of the `attack` (§8.3), held by the decoder until the next note. Cross-note stickiness (an instrument
reuses its envelope) is captured by **BPE compounding the recurring attack** (which carries the envelope) —
*not* a separate field, header, or a "use-last" history pointer. The nibble tokens are shared, so a rare
envelope on a common waveform still decomposes into shared sub-tokens (no singleton).

NOTE the distinction the hysteresis probe missed: "static envelope" is about the *sustained* value. The
**hard-restart AD/SR writes** (transient `FF/00` etc., present in **47%** of notes) are *not* the envelope
— they are earlier writes in the ordered `attack` (§8.3), and dropping them was the §7 fidelity hole.

### 5.2 Perceptual generalization (reSID-grounded)
Clustering the full ADSR grid by emulated amplitude envelope: 65,536 → 11.6k (80 ms gate) / 22.8k (400 ms)
perceptual clusters; observed 6,987 → 3,137 / 4,295 (modest 1.6–2.2× — the diversity is mostly *genuine*
intent). **Decisive finding: 100% of (AD,SR) change perceptual cluster between a short and long gate** —
envelope identity is gate-time dependent. Implications:
- **duration must be in the event** next to the envelope (already is) — required for the model to
  interpret/generalize the envelope's effect.
- Use the perceptual clusters to **tie embeddings** of audibly-equivalent envelopes (regularize same
  cluster → near in embedding space), so the model treats them as interchangeable. This is the
  generalization payoff; it does not change the (lossless) nibble encoding.

### 5.3 No fused (ctrl×env) id — separately-typed tokens instead
Gate-on joint over 4.47M note-ons: a **fused (ctrl,env) id would be 37,692 instruments** — *larger* than
the factored sum. MI(ctrl;env)=4.37 of 10.12 bits, 607 envelopes per ctrl-program → envelope is **not**
determined by ctrl-program. So we never mint a single (ctrl,env) id. The `attack` interleaves CTRL and
ADSR for *order*, but keeps them as **separately-typed tokens** (waveform bytes vs nibbles); BPE compounds
frequent attacks softly while rare (ctrl,env) combos decompose into the shared CTRL/nibble sub-tokens — so
no cross-product singleton tail. The two channels have different shapes:
- **CTRL (waveform): low-diversity, frequent-core** — 3,313 distinct programs, top 86 cover 90%, singletons
  0.02% of notes, 95% ≤3 values. BPE → tiny effective vocabulary, no tail risk.
- **ADSR (envelope): genuinely high-diversity** — 6,987 sustained pairs, ~900 cover 90%, a long real tail.
  The 4-nibble complete encoding keeps the tail escape-free; BPE compounds the frequent ~900.

### 5.4 Pitch is a soft, separate predictor
Pitch is orthogonal to the instrument (6.9 distinct pitches per instrument) → its own note field, never
fused. It is weakly predictive: MI(pitch;envelope) ≈ 0.6–1.4 bits, MI(pitch;ctrl-program) ≈ 0.6–1.2 bits
— a soft prior the model gets for free by having pitch co-located in the event. Do **not** hard-condition
the vocabulary on pitch (too weak; would fragment it).

---

## 6. Gate vs pitch separation (hard requirement)

Note-on governs the **gate only**. Pitch changes ~3.4× per gated note (arp), and gate-off is 1:1 with
note-on (0.999), with only 0.1% ties. So:
- `note-on` = a gate (re-trigger) event carrying duration; **note-off is derived** at `onset + duration`.
- the pitch trajectory underneath (arp steps, slides, vibrato) is the **freq note-index / freq-delta
  layer**, independent of the gate. Never equate a note-on with a pitch change.
- the 0.1% tie/drone case = `duration ≥ inter-onset` (gate stays on) — one representation, no escape.

---

## 7. Decoder & fidelity (ordered write stream)

**The fidelity oracle is the exact ordered register-write stream** (§2.8), not `register_state`. The
decoder expands the event stream to an **ordered list of `(frame, reg, value)` writes** and the encoder
asserts it equals the source dump's writes **byte-for-byte in order** — including intra-frame write order
and same-register repeats. `register_state` may be computed as a fast pre-filter, but a pass requires the
ordered-stream match.

### 7.0 How write order is reproduced (the mechanism) — RESOLVED (§13)
Write order is **not assumed/recovered as a fixed canonical order** (that's consistent for only 56–76% of
tunes — fragile). Instead, same-frame register writes are sequenced by a **per-frame order descriptor**: a
complete-value field (the permutation of that frame's changed regs) carried on the steady lane, present on
every frame, BPE-compounded. The dominant canonical order compresses to ≈nothing; deviating frames are
rarer values of the same field; **byte-exact regardless of consistency, escape-free** (§13). The note
attack (§8.3) carries its onset order explicitly. This is the same complete-encoding+BPE principle as
duration/envelope, not an assume-and-escape.

Concrete mechanism — two cases, no recovered canonical permutation, no escape:
- **Onset (the `attack`, §8.3):** the attack is an explicit ordered `(Δframe, field, value)` write
  sequence, so the audibly-critical onset (hard-restart `AD=FF/SR=00`→gate→real-AD/SR→waveform, incl.
  same-reg repeats) is reconstructed write-for-write from the attack itself. No descriptor needed here.
- **Steady frames (between attacks):** each active lane (freq/PW/body-CTRL per voice; global cutoff/vol)
  contributes ≤1 write per frame. Their cross-lane order within a frame is the **order descriptor**: the
  ordered list of reg-ids written that frame (a complete-value field on the steady lane). The decoder pulls
  each listed reg's value from its lane's gesture and writes them in that order. The *per-tune dominant*
  order is one frequent descriptor value → BPE ≈ 0 cost (≥56% of tunes use one order throughout); a frame
  that deviates is a *different value of the same field*, not a special path.
- **Same-reg repeats outside an attack** (rare body cases / the ~9.5% multi-speed extra passes) are emitted
  as a short `attack`-form ordered write run for that span (the same verbatim primitive), so the descriptor
  never needs to list a reg twice. One mechanism, escape-free.

The encoder **asserts** the decoded ordered stream equals the source byte-for-byte; divergence fails loudly
(it is a bug, not an escape). Decode is otherwise exact integer arithmetic: ADSR nibbles, mixed-radix
duration, freq `note_table+delta`, PW/CTRL/filter gestures.

There is **no fall-back-to-literal guard** (original spec §4.5): divergence is a bug, fail loudly. Every
field is a complete encoding, so nothing is unrepresentable — a literal/escape path is neither needed nor
allowed.

### 7.1 Pipeline integration — what changes, what's retired

The event schema is a **separate representation**, not a df transform. The pipeline:

```
raw dump  →  ordered write stream (frame, reg, val)*   [GROUND TRUTH = fidelity oracle, §7]
          →  EVENT ENCODER (§8)  → event stream            [NEW]   (asserts decode == this stream, ordered)
          →  tokenize + BPE (regtokenizer)                 [EXTEND: event tokens, no ids]
          →  (model)
          →  EVENT DECODER (§7)  → ordered write stream     [NEW]
          →  audio (preframr-audio reSID)
```

- The ground truth is the **ordered register-write stream**, not the settled `register_state` (§2.8). The
  encoder consumes it; the decoder is validated against it byte-for-byte in order. The event stream is its
  own typed structure (kind + fields), not a `make_row` df.
- **Retired for the freq/ctrl/ADSR block:** the whole gesture-codebook row machinery
  (`mdl_gesture_pass`, `_GestureCodec`, GESTURE_* ops/subregs, `codebook_emit`, the per-tune `bank`) and
  the arbiter/Claim flow for these channels. The current in-tree wrap fix + gate-on segmentation are an
  end-of-frame baseline to diff against during the rebuild, then removed.
- **KEEP:** `audit_primitives` (settled state as a *secondary* check only), `pitch_grid` table recovery,
  `mdl_core` primitives (reused inside the joint DP and scalar parse), the reSID renderer in
  `preframr-audio`.
- **`constrained_decode` → event-grammar mask:** a finite-state grammar over event kinds and their field
  sequences (DT→VOICE→KIND→fields), registry-driven like today's contracts, with a completeness test
  (every emittable field has a decode + a mask rule). No REF-liveness logic needed (no ids); the only
  constraints are field arity/ordering and mixed-radix digit continuation.
- **BPE/Unigram:** trained on the event-token stream corpus-wide; the shared statistics *are* the
  corpus-global dictionary (§8.6). Keep the existing Unigram machinery; feed it events.

---

## 8. Encoder — the recovery algorithms (concrete)

Input: the **ordered register-write stream** (the fidelity oracle, §7); a per-frame settled value view may
be computed for parsing, but the output is verified against the ordered stream. Output: the event stream +
headers (incl. the canonical per-frame write order, §7.0). All passes deterministic. Order: (1) tick,
(2) note table, (3) gate/attack/envelope segmentation (§8.3, interleaved onset), (4) joint freq parse,
(5) scalar per-voice PW + global filter/volume parse, (6) serialize + Zopfli cost iterate.

### 8.1 Tick + groove offset (per voice)
Collect gate-on durations `d` (gate-on edge → gate-off edge, in frames). `tick` = the largest unit in
`[2,32]` for which ≥90% of `d` are within ±1 of a multiple (prototype: `mod_period_probe._gridfit`, repo
root). `offset` = the per-voice mode of `r = d − round(d/tick)·tick` (the systematic
gate-held-`tick−1`; dominant-r = 73.8%). Emit `TICK`, `offset` headers. Durations then encode as
`q=round(d/tick)`, `r=d−q·tick−offset` (residual now centered at 0, H≈1 bit). Exact for all `d`.

### 8.2 Note table + tuning (per freq voice)
Reuse the existing `pitch_grid.voice_tuning` / `recover_table` / `note_index` / `note_freq_at` (KEEP from
the current system). Emit `TUNING` + `NOTE_TABLE` headers. `note_table[m]` is the exact freq base note `m`
rides; `freq[t] = note_table[note_index[t]] + freq_delta[t]`.

### 8.3 Gate / attack / envelope segmentation (per voice)
Walk the voice's CTRL/AD/SR writes in clock order with frame indices.
- **Notes** = gate-on edges (CTRL bit0 0→1). Each note spans `[onset, onset+duration)`; gate-off derived.
- **attack** = the **ordered, interleaved** CTRL+AD+SR write-sequence of the onset window (prototype:
  `attack_interleave_probe`, window `[onset−3, onset+6]`). **This is the critical fix:** 47% of notes write
  AD/SR more than once in this window (hard-restart `AD=FF/SR=00` then the real envelope) — the previous
  "ctrl-program (CTRL-only) + one static envelope" split dropped those AD/SR writes and the CTRL↔ADSR
  interleaving, breaking the ordered-stream oracle (§7). Emit the attack as the ordered list of writes,
  each `(Δframe, field, value)`:
  - `Δframe` via the §4 time scheme (onset Δframes are tiny → frame digit, mostly 0–1).
  - `field` ∈ {CTRL, AD, SR}; `value` = the CTRL waveform byte or the AD/SR byte (split to nibbles).
  - **No template id, no separate envelope field** — the order is the sequence; BPE compounds the
    recurring attacks (field-order top 605 cover 90%, singletons 0.04% of notes, so the effective vocabulary
    is small). CTRL and ADSR stay separately-typed tokens (no fused id, §3.1).
  The simple no-hard-restart case (53%) is just `gate CTRL + one AD + one SR` — a short attack.
  Attack window = `[onset−3, onset+W]`, extended forward until the voice's CTRL/AD/SR have no further
  same-reg repeat and AD/SR have settled (so the boundary is structural, not a fixed W). Overlapping onsets
  (<window apart, e.g. drums): each write belongs to the **earliest** gate-on whose window it falls in.
- **sustained envelope** = the attack's **final** `(AD,SR)`; the decoder **holds** it until the next note
  (no body AD/SR events for the 77% static case). It is not re-emitted — reuse across notes is a BPE effect
  on the recurring attack, not a header/field.
- **body waveform** = CTRL writes *after* the attack window (a looping wavetable) → a `MOD_CTRL` gesture
  (HOLD/PERIOD over waveform bytes). Constant-waveform notes have none.
- **mid-note AD/SR change** (~23% non-static notes) → a short `attack`-form write run at its `DT`.

### 8.4 Joint 2-D note/modulation parse (the freq layer) — BUILD NOW

Per freq voice, recover the note-index stream and freq-delta gestures **jointly and optimally** by
shortest path (spec §1.2; replaces the greedy `_held_notes` approximation). Operates on
`freq[0..n)` with the recovered `note_table`/`tuning`.

- **DP state:** `(frame i, current note-index m)`. `m` ranges over candidate grid notes near the observed
  freq (see fan-out). Use a dict/sparse table keyed by reachable `(i,m)`.
- **Edges from `(i,m)`:**
  1. **modulation gesture** `(i,m) → (j,m)`: a HOLD/POLY/PERIOD gesture (reuse `mdl_core.mdl_parse`
     primitives) fitted to `freq[i:j] − note_table[m]`, for every `j` the primitive exactly covers.
     Cost = the gesture's **event-token cost** under the current cost model (§8.6). Bound `j−i ≤ MAXLEN`.
  2. **note event** `(i,m) → (i,m')`: a `NOTE_STEP`/`NOTE_ON` to a candidate grid note `m'` whose
     `note_table[m']` is nearest `freq[i]` (fan-out = nearest, ±1, ±2 → ≤5 successors). Cost = the
     `note-interval` token cost for `zig(m'−m)` (§8.6).
- **Objective:** minimise total cost over the path `(0, m0) → (n, ·)`; the shortest path yields the
  note-index segmentation **and** the freq-delta gestures simultaneously. Greedy is forbidden — only the
  global cost keeps a wrapping ramp as one POLY(1) rather than 240 note-events, and lets a recurring
  transient collapse to one shape.
- **Complexity:** `O(n · |m-fanout| · MAXLEN)`; reuse the incremental run precompute from `mdl_core`
  (`_poly_runs`, `_period_edges`, `_hold_runs`) per anchor `m`. Distinct anchors per column are few
  (fan-out ≤5), so this is tractable at corpus scale.
- **Exactness:** every `(i,m)` decomposition is byte-exact because `freq = note_table[m] + gesture(i:j)`
  and the gesture decode is exact; the DP only chooses the cheapest. No residual, no escape.

### 8.5 Scalar channels — per-voice PW, and the global filter/volume lane
1-D `mdl_core.mdl_parse` (wrap-16, the fix already in tree) into HOLD/POLY/PERIOD gestures:
- **per voice:** PW (regs 2/9/16) → `MOD_PW` events.
- **global (once, shared):** cutoff (21/22) → `MOD_CUTOFF`; resonance+routing (23) and mode+master-volume
  (24) → `FILTER_CTL` / `MOD_VOL` on the `VOICE=GLOBAL` lane (§3.1). Not per-voice.
Spans inherit the enclosing note's duration where they coincide (§4.1), else mixed-radix.

### 8.6 Cost model + Zopfli iterate (corpus-global, escape-free)
The DP/parse cost of a field/gesture = its **event-token description length** under a unigram model of the
token alphabet. Bootstrap pass 1 with a static cost (`mdl_core.nbits` for magnitudes + fixed per-token
costs); after a full corpus pass, re-estimate token frequencies and re-run (Zopfli-style, spec §2) until
description length converges. Because there are no ids — only complete-value tokens — "corpus-global"
needs no shared bank: the shared unigram/BPE statistics *are* the global dictionary. 2–3 iterations.

---

## 9. Acceptance / guards

- **No-escape invariant test:** every emittable field decodes for all values; assert the schema has no
  literal/escape op. A round-trip over the corpus must never hit a fallback (there is none).
- **Byte-exact ORDERED WRITE STREAM** (§7, the primary oracle) on the 5 drivers + ≥200-tune corpus
  sample: decoded `(frame, reg, val)` writes equal the source byte-for-byte *in order*. Zero divergences.
  Settled `register_state` is only a secondary pre-filter.
- **Expansion guard (re-add):** output description length ≤ raw input; no gesture/instrument shape encodes
  more than a small bounded primitive (catches the verbatim-explosion class that the deleted
  `test_encoding_complexity.py` missed). (task #3)
- **Collapse / bits measurement on the real encoder** (task #2): target ~10–14× *bits* post-BPE; report
  dictionary/compounded-token size and the rare tail.
- **Event-grammar completeness:** every emittable field has a decode + a mask rule (the §7.1 registry
  test); a generation smoke test under the event-grammar mask emits only decodable event streams (valid
  field arity/ordering, valid mixed-radix continuations) in both atomic and BPE modes.

---

## 10. Evidence (probes, repo root; rerun with `--all`)

- `adsr_diversity_probe.py` — 198 ctrl values; 7,570 sustained envelopes; per-tune ≤8 (66%) / ≤16 (90%);
  hard-restart values overlap envelopes (classify by behavior, not value).
- `adsr_hysteresis_probe.py` — 77% of notes static envelope; H(next|prev)=3.56 vs H(next)=8.68; no compact
  shaping vocabulary (19k trajectories > 11k pairs).
- `gateon_instrument_probe.py` — fused instruments 37,692 > factored 13,160; MI(ctrl;env)=4.37 bits; pitch
  orthogonal (6.9 pitches/instrument).
- `pitch_predicts_probe.py` — MI(pitch;env) 0.6–1.4 bits, MI(pitch;ctrl) 0.6–1.2 bits (soft).
- `adsr_perceptual_cluster.py` (+ `adsr_clusters.npz`) — reSID perceptual clustering; 100% gate-time
  reassignment; 90% coverage by ~700–900 clusters.
- `ctrl_program_probe.py` — ctrl-program **3,313 shapes, top 86 = 90%, singletons 0.02% of notes, 95% ≤3
  values** → low-diversity, frequent-core, BPE-safe (the §5.3 / §8.3 evidence).
- `mod_period_probe.py` — modulation updates **78–79% every-1-frame, 94–95% tick-gridded** → fit the frame
  digit of the one time scheme (§4.1).
- `attack_interleave_probe.py` — **47.3% of notes have hard-restart AD/SR** (what the old static-envelope
  split dropped); onset field-order templates **top 605 cover 90%, singletons 0.04% of notes** → the
  interleaved attack is a frequent-core, BPE-safe vocabulary (the §8.3 fix evidence). Timed templates 45k
  (timing must be factored via the time scheme); onsets ≤6 writes for 51%.
- `duration_probe.py` — gate_off:note_on = 0.999 (note-off redundant), ties 0.1%, legato 343% (pitch is a
  separate layer); durations 94→98.7% on-grid with grid-fit, ticks 2–8; mixed-radix exact 100%, H(r)=1.58,
  H(q)=2.44, dominant-r 73.8%.
- `write_order_probe.py`, `write_order_probe2.py`, `write_order_probe3.py`, `multispeed_probe.py` — §13:
  multi-speed ~9.5%; clean-frame single-canonical-order 56.2% (within-voice 65.5%, cross-voice 76.0%) →
  motivates the per-frame order descriptor.

---

## 11. Build order

0. **Write-order = per-frame order descriptor** (§13, resolved): a complete, BPE-compounded field
   sequencing same-frame writes; escape-free and exact regardless of consistency. Measure its bit cost in
   the collapse/bits measurement (task #2). No longer a blocker.
1. **Event schema + grammar** (§3, §7.1): the event/field types, mixed-radix time codec, event-grammar
   mask. No behavior yet; grammar-completeness test green.
2. **Encoder recovery passes** (§8): tick/offset (8.1), note-table (8.2, reuse), gate/attack/body-CTRL
   segmentation (8.3), **the joint 2-D DP (8.4) — built now, no greedy approximation**, scalar parse
   (8.5). Produces the event stream.
3. **Driver-replay decoder** (§7) → ordered write stream; **byte-exact ORDERED-STREAM match on the 5
   drivers + ≥200-tune corpus, zero divergence, no escape path** (settled state is only a pre-filter).
4. **Tokenize + BPE/Zopfli iterate** (§8.6) over the corpus; complete-value tokens, no ids.
5. **Acceptance** (§9): no-escape invariant test, expansion guard, collapse/bits measurement (target
   ~10–14× bits post-BPE), event-grammar completeness + generation smoke test.
6. Retire the gesture-codebook row machinery for this block (§7.1); delete the prototypes + this-era docs
   once green.

---

## 12. Locked rulings on the remaining decisions

Resolved so the executing agent does not have to choose. (#1 fidelity, #2 global lane are locked in §2.8 /
§7 / §3.1.)

- **#3 Event ordering & same-frame sequencing.** Events are globally time-ordered by absolute frame. Within
  a frame, steady cross-lane write order is the **order descriptor** (§7.0/§13); the `attack` carries its
  own onset order. Event *emission* order in the stream follows the descriptor (so stream order = write
  order). Any residual tie breaks by `(VOICE, KIND)` ascending, deterministically.
- **#4 Note-table / tuning header.** Encode as a **tuning constant + the standard equal-tempered grid**;
  store only per-note *deviations* from the grid (most tunes have none → near-empty header). The tuning
  constant is a small-alphabet complete value (corpus-global via BPE). Reuse `pitch_grid.q_to_tuning`.
- **#5 Sustained envelope (no `DEFAULT` token).** Dropped — the `attack` (§3.1/§8.3) carries the envelope
  as its final AD/SR; the decoder holds it; reuse is a BPE effect on the recurring attack. No
  instrument-default header, no `DEFAULT` token, no keyed lookup.
- **#6 NOTE_STEP vs MOD_FREQ + no-op suppression.** The joint DP's **note-event edges → `NOTE_STEP`**
  (note-index change), its **modulation edges → `MOD_FREQ`**. **No-op suppression:** a zero freq-delta
  (on-grid held note) emits **no** `MOD_FREQ` — the note's pitch + duration already imply it. Likewise a
  constant scalar emits no `MOD_*`. (A held value is the absence of a gesture, never a HOLD-of-0 token.)
- **#7 Tempo / multi-tick.** v1 recovers **one tick per voice**; any drift is absorbed exactly by the
  mixed-radix `r` (escape-free, §4). Tempo-change detection + `TICK` re-emission is a **follow-up
  optimization**, not v1 — it only improves compression, never correctness.
- **#8 Structural passes (loop/pattern/voice-block/coarsen).** v1 = **BPE-only** (retire them); transpose →
  zig-zag intervals. Add explicit loop/pattern ops only if the §9.3 collapse measurement shows BPE misses
  long-range repetition (§0). Decided.

Delegated (safe implementation choices, no research): mixed-radix base for `q` (suggest 16) + digit order;
joint-DP cost constants, `MAXLEN` (suggest ≥ longest expected note span) and equal-cost tie-break (lowest
`j`, then lowest degree); BPE vocab/merge count; perceptual embedding-tying mechanism (§5.2, model-side,
out of encoder scope); multi-SID scope (v1 = chipno 0 only).

---

## 13. RESOLVED (in principle) — write-order as a complete, BPE-compounded field

Write-order consistency was investigated and **the "recover one canonical order" approach was rejected**
as fragile (it can fail → wants an escape, violating §2.2). Measurements (`multispeed_probe.py`,
`write_order_probe{,2,3}.py`):

- **Multi-speed ≈ 9.5%** of tunes (median per-frame max-repeat ≥2), not the earlier buggy "61%". 90.5%
  single-speed; 62.5% ~100% clean single-write frames.
- A single canonical write order is consistent on clean frames for only **56.2%** of tunes; within-voice
  order fixed in **65.5%**, cross-voice in **76.0%** (within-voice is the bigger variance, partly
  attack-vs-steady ordering that §8.3 absorbs). So no recovered fixed order covers enough tunes.

**Resolution (same principle as duration/envelope — complete encoding + BPE, never assume):** the decoder
sequences same-frame register writes by a **per-frame order descriptor** — a complete value (the
permutation of that frame's changed regs) carried on the steady lane. It is *always present*, so:
- the dominant canonical order is the most frequent value → BPE compresses it to ≈nothing (the ≥56%
  fully-canonical frames + 90% single-speed tunes pay ~zero);
- a deviating frame is a *rarer value of the same field*, not a separate path;
- **escape-free and byte-exact regardless of order consistency** — the 56–76% numbers no longer gate
  correctness, they only bound this field's *cost*.

The note **attack** (§8.3) still carries its onset order explicitly (hard-restart). Multi-speed (~9.5%)
tunes: their extra player-pass writes appear as same-frame repeats, sequenced by the same per-frame order
descriptor (which then lists a reg more than once) — still one mechanism, no escape.

**Only open item:** measure the order descriptor's bit cost (expected near-zero after BPE) as part of the
collapse/bits measurement (task #2). Not a blocker; §7.0 fidelity is guaranteed by construction.
