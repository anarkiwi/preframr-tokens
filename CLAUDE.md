# preframr-tokens — agent guidance

The white-box SID decompiler codec. A SID register trace is the byte-exact OUTPUT of a tiny deterministic
playroutine; recover the PROGRAM (notes + pitch-invariant instruments + backward-repeated phrases), never
compress the output. Full design narrative: `/scratch/anarkiwi/preframr/preframr-xpt/AGENTS.md`.

## HARD RULE #0 — nothing is "irreducible" (read before any encoding/recovery work)

Every non-digi SID tune was composed by a human in a tool (DMC / JCH / DefMon / GoatTracker / a hand
player) from patterns + instruments, so a compact source PROVABLY EXISTS — its existence is the proof of
compressibility. "irreducible / entropy wall / incompressible / store the data / minimal recurrence = its
length / near-digi" is ALWAYS a BUG REPORT, never a result: it means structure you have not recovered yet.

Before ANY such conclusion, run the falsification protocol and report it:
- (a) decompose the run's VALUES into pitches — few distinct = it's NOTES (melody/arp), not entropy;
- (b) measure the phrase-repeat % — recurrence = a backward REPEAT/TRANSPOSE the encoder must collapse;
- (c) FIT a generator (accumulator / table-walk / glide / wrapping sweep), don't eyeball it;
- (d) check the frame-count DENOMINATOR — a sparse-writer framed per write-burst fakes a wall.

The floor is the PLAYER (hundreds of bytes), NEVER a general compressor (LZMA cannot represent an
accumulator ramp or a table-walk). The literal floor / register-log reproduction is FAILURE, not a floor.
**tok/frame < 1 (target < 0.5) is CO-EQUAL with residual-0.** When stuck, LIFT ALTITUDE: register → note →
instrument → pattern → orderlist, the level it was authored at.

## Doing codec work

- Dispatch codec/analysis subagents as `subagent_type: sid-codec` — the guardrails above are baked into its
  system prompt (`.claude/agents/sid-codec.md`), so they cannot be forgotten by the dispatcher.
- "Done" is the GATE, not a judgment:
  `SIDTRACE_BIN=/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace PYTHONPATH=. python tools/codec_gate.py <sid>`
  (residual-0 AND < 1 tok/frame). A FAIL on tok/frame = unrecovered structure, never a wall.
- No scratch files in the tree (write throwaways under `/tmp`). A `SubagentStop` hook
  (`.claude/hooks/codec_guard.sh`) rejects a finish that left scratch files or asserted a wall without the
  falsification protocol.
- Full gate: `./run_tests.sh` (black / pylint / pyright / pytest / coverage).
