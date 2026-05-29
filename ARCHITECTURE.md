# preframr-tokens architecture

The **torch-free** parser + tokenizer for SID register-log dumps. Turns a raw
`(clock, irq, chipno, reg, val)` write log into a compact, learnable token stream
(and decodes it back, byte-exact). No torch, no audio — the framework
(`preframr`) and the renderer (`preframr-audio`) depend on this; it depends on
neither. This doc is the canonical reference for the parse→pass→tokenize→decode
pipeline so it does not have to be re-derived from the code each time.

## Topology

- **`preframr-tokens`** (this repo, PyPI) — parse + tokenize + decode + macros +
  `render_play` glue. Public API is the `__all__` in `preframr_tokens/__init__.py`.
- **`preframr-audio`** (PyPI) — SID synthesis / WAV rendering primitives
  (`render_df_to_wav`, fidelity compare).
- **`preframr`** (docker image) — torch train/predict/model; floors a
  `preframr-tokens>=X.Y.Z` version.

## The dump (input)

A parquet with columns `clock, irq, chipno, reg, val` — one row per SID register
write (`DUMP_SUFFIX = ".dump.parquet"`). `clock` = PAL φ2 cycles, `irq` = cycles
of the player IRQ (the frame period, ~19656), `reg` 0–24, `val` 0–255.

### Register map (`reg`)
Per voice `v ∈ {0,1,2}`, base `= v*VOICE_REG_SIZE (=7)`:

| offset | register |
|---|---|
| +0 | **freq lo** |
| +1 | **freq hi** |
| +2 | PW lo |
| +3 | PW hi |
| +4 | **ctrl** (bit0 = gate; bits4-7 = waveform) — `VOICE_CTRL_REG[v]` |
| +5 | attack/decay |
| +6 | sustain/release |

Global: `21 FC_LO_REG` (filter cutoff lo), `22` FC hi, `23 FILTER_REG`
(resonance + routing), `24 MODE_VOL_REG` (filter mode + volume). `MAX_REG = 24`.

The 16-bit **frequency** of voice `v` is `(reg[v*7+1] << 8) | reg[v*7]`. **Never
read it from a single lo or hi write** — a lo write with a stale hi byte is a
garbage pitch. Use the settled value (`combine_reg`, below). This was a real bug
(`design/unified_pitch_encoding.md` "read settled 16-bit freq, not raw bytes").

### Marker registers (negative `reg`, inserted by the parser)
`FRAME_REG = -128` (one frame tick), `DELAY_REG = -127` (N empty frames, val=N),
`VOICE_REG = -126`, `LOOP_OP_REG = -125`, `SUPER_FRAME_REG = -124`,
`VOICE_TRAJ_REG = -123`, `PAD_REG = -1`.

## The atom model

After parsing, every row is an **atom** `(op, reg, subreg, val, diff)`:
- **`op`** — what kind of write/macro this is (table below). Raw writes are
  `SET_OP = 0`.
- **`reg`** — target register (or a marker reg).
- **`subreg`** — sub-field index within a multi-field macro (e.g. FREQ_TRAJ
  splits into V0_HI/V0_LO/COUNT/PERIOD/DELTA); `-1` for plain SETs.
- **`val`** — payload byte (or macro arg).
- **`diff`** — cycles since the previous atom (timing).

### Op codes (`stfconstants.py`)
`SET=0, DIFF=1, FLIP=3, TRANSPOSE=5, FLIP2=7, BACK_REF=15, DO_LOOP=16,
SUBREG_FLUSH=18, PATTERN_REPLAY=22, PATTERN_OVERLAY=23, HARD_RESTART=25,
PWM_PRESET=35, FC_PRESET=36, PWM_PRESET_SHIFTED=40, CTRL_BIGRAM=42,
PWM_SUSTAIN=43, WAVETABLE_SUSTAIN=44, FREQ_TRAJ=45, TRACK_REF=46, FREQ_NUDGE=47,
FREQ_ONSET=48, RELEASE_UPDATE=49, CTRL_TRIPLE=50, CTRL_UPDATE=51, MOTIF=52,
MOTIF_ARG=53`. **Next free op = 54.**

The melodic-pitch atoms are recognised by `is_melody_pitch_atom(op,reg,subreg)`
and `is_freq_onset_atom(...)` (`regtokenizer.py`): op45 V0 (`FT_SUBREG_V0_HI=1`,
`FT_SUBREG_V0_LO=2`) + op48 FREQ_ONSET + op47 FREQ_NUDGE pitch.

### FREQ_TRAJ (op45) subregs
`FT_SUBREG_FLAGS=0, V0_HI=1, V0_LO=2, COUNT_HI=3, COUNT_LO=4, PERIOD=5, DELTA=6`.
With `--freq-v0-interval`, the V0 of each trajectory after the first on a reg is
stored as a **signed semitone-ish interval** from the previous (FLAGS bit
`FT_V0_INTERVAL_BIT`), which is what makes it key-invariant and learnable.

### Slope-able registers
`TRAJ_REGS = (0, 2, 7, 9, 14, 16, 21)` = each voice's freq-lo + PW-lo, plus
filter-cutoff-lo. `FREQ_TRAJ_REGS = (0, 7, 14)` = the three voice freq-lo regs
(the melody channel).

## Parse pipeline — `RegLogParser.parse()` (`reglogparser.py:862`)

`parse(name)` is a generator yielding one parsed `df` per voice-rotation. If a
`*.N.parquet` cache exists (`PARSED_SUFFIX`) it yields that; otherwise it builds
from the dump. Build order (`parse()` body):

1. **`_read_df`** — read parquet, clip `reg ≤ MAX_REG`, `val → VAL_PDTYPE (Int32)`,
   drop multi-chip rows.
2. **`_squeeze_changes`** — drop writes that don't change the held value.
3. **`_combine_regs`** — coalesce each lo+hi pair into one settled 16-bit value
   (freq regs 0/7/14, PW regs 2/9/16, filter 21). See `combine_reg` below.
4. **`_stash_freq_unq`** (if an anchor/trajectory flag is on) — copy the
   full-precision 16-bit freq into a `freq_unq` side column before cent-binning,
   so the trajectory passes see exact freq.
5. **`_quantize_freq_to_cents`** — cent-bin freq (content-tier **lossy by
   design**; raw freq survives in `freq_unq`).
6. **`_simplify_ctrl`, `_simplify_pcm`** — canonicalise control/PCM writes.
7. **`_squeeze_changes`** again.
8. **`_add_frame_reg(diffmax=2048)`** — derive the frame period from the dominant
   `irq` delta, insert `FRAME_REG`/`DELAY_REG` markers, return `irq`.
9. **`_filter`** — reject dumps with too many control writes per frame
   (`>6`), etc.
10. **`_squeeze_frame_regs`**.
11. **Freq + structural passes, in order**: `VoiceTrackPass` →
    `TrajectoryAnchorPass` (annotate, see below) → `FreqTrajectoryPass` (emit
    op45) → `FreqOnsetPass` (emit op48) → `PresetPass` → `PerRegBurstPass` →
    `GateSlopeShiftPass` → `ReleaseUpdatePass`.
12. **`_consolidate_frames`** — fold runs of empty frames into one `DELAY`.
13. **`_cap_delay`** — quantise DELAY vals (≤256).
14. Write per-block `*.N.parquet` + `*.N.parquet.palettes.json`.

### `combine_reg` — the settled 16-bit read (the canonical freq combine)
`combine_reg(df, reg, diffmax=512, bits=0, lobits=8)` (`reglogparser.py`,
module-level; `RegLogParser._combine_reg` delegates) forward-fills the `reg` (lo)
and `reg+1` (hi) bytes and keeps **the last settled value per `clock // diffmax`
bucket**, so a coordinated lo+hi update inside one 512-cycle window is read as one
16-bit value and a half-updated pair is never seen. `bits` masks low bits
(PW/filter quantisation). Public + reused by the frequency audits in
`preframr-xpt` (added 0.30.0) — **do not re-implement the lo/hi combine
anywhere**; call this.

### `TrajectoryAnchorPass` — segmentation (annotation only)
Writes a boolean **`traj_anchor`** column marking note boundaries by **intrinsic
level-change ∪ gate-on** (not gate alone — held-gate/legato drivers move pitch
under one sustained gate). Downstream `FreqTrajectoryPass` consumes `traj_anchor`
as forced segment cuts (`_anchor_chunks`), so no trajectory spans a note boundary.
This is the production source of truth for note segmentation; audits should reuse
it (`detect_anchors`/`pass1_origins`/`pass2_collapse` are exported).

## The pass framework (adding an op)

A new op becomes real only when three layers line up:

1. **Pass** — a `MacroPass` subclass in `preframr_tokens/macros/` with
   `GATE_FLAGS = frozenset({"<flag>"})` and `apply(self, df, args)` that no-ops
   when the flag is off (`freq_onset_pass.py` is the minimal template). Emits rows
   carrying `(op, reg, subreg, val, diff, irq)`.
2. **Decoder** — a `MacroDecoder` with `op_code = <OP>` and `expand(self, row,
   state)` in `macros/decoders.py`, registered in the `DECODERS` dict.
   `expand_ops` (`macros/decode.py`) **asserts `DECODERS[op] is not None`** — an
   unregistered op hard-crashes decode. This is the byte-exact tripwire.
3. **Transform** — a `@register("<name>")` `PassBackedTransform`
   (`macros/transforms_audio_bit_exact.py`) tying `OP_CODES`, `LOSS_TIER`,
   `REQUIRES_ARGS`, `PASS_CLASS`, `DECODER_CLASS` together. `LOSS_TIER="content"`
   makes the op's atoms classify as **content tier** automatically
   (`collect_op_loss_tiers` → `_row_tier` in `vocab_signature.py`). Flag names
   auto-derive from `GATE_FLAGS`/`REQUIRES_ARGS` (`flag_registry.py`).

### Pass ordering (where passes run)
- **`FREQ_BLOCK_PASSES`** (`macros/__init__.py`): TrajectoryAnchor, FreqTrajectory,
  FreqOnset, PerRegBurst, ReleaseUpdate — the freq-encoder passes that turn
  literal SETs into op45/47/48/49. Run **at parse time** (inline in `parse()`)
  **and** re-run per self-contained block via `run_freq_block_passes` (because
  `expand_to_literal_form` decompiles them back to SETs inside the block loop —
  the `freq_passes_re_fire_on_blocks` fix).
- **`PASSES`** (`run_passes`): the pre-norm structural macros (Preset,
  GateSlopeShift, Flip2, Transpose, DedupSet×2, HardRestart, LegatoPerCluster,
  CtrlTriple, CtrlBigram, Subreg).
- **`POST_NORM_PRE_VOICE_PASSES`**: VoiceBlockOrder, Loop, Coarsen.

### Mined-dict passes (codebook artifacts)
`MotifPass`/`MotifDict` (`macros/motif_pass.py`) + `mine_dict_from_dumps`
(`motif_mine.py`) are the pattern for any mined codebook (e.g. an ARP codebook):
a JSON artifact resolved from `args.<dict>` (path or object, cached on args),
mined torch-free, grouped by composer (parent dir of the dump path).

## Decode + fidelity

`expand_ops(df)` (`macros/decode.py`) walks atoms through `DECODERS`, maintaining
`DecodeState`, and reconstructs the literal per-frame register SETs. The
**byte-exact contract**: `forward` then `expand_ops` must reproduce the original
per-frame register state. Enforced by `assert_dfs_render_equivalent`
(`macros/transform.py`) and the per-frame-state oracle
`tests/test_full_pipeline_fidelity.py` (+ `test_motif_fidelity.py`).
`audit_primitives.register_state(xdf)` returns the decoded `(n_frames, 25)`
per-frame register matrix via the same public expand path — the shared reduction
the oracle and profiler use.

## Tokenize: atoms → uids → blocks

- **`RegTokenizer`** (`regtokenizer.py`) — maps atoms ↔ integer uids. With
  `tkvocab > 0` it trains/loads a **Unigram** model (`tkmodel`) that merges
  frequent atom runs into compound uids; with `tkvocab = 0` (**de-merge**) each
  base atom is its own uid. (De-merge isolates the pitch atom for learnability but
  fragments notes for generation — see the melody arc.)
- **`Corpus`** (`corpus.py`) — orchestrates parse → tokenize → cache. Writes one
  **`.N.blocks.npy`** per voiced block next to each dump
  (`df_file.replace(DUMP_SUFFIX, ".{i}.blocks.npy")`), and a **`df-map.csv`**
  (`dump_file, kind, irq, n_rotations`) listing every dump and its `kind`
  (`train` / `val`). `iter_block_seqs` (train+val) and
  `iter_predict_block_seqs` (predict) yield `(kind, blocks_path, seq_meta)`.
- **`parse_corpus`** (`parse_runner.py`) — the batch entry the framework calls.

### Gotcha: df-map paths vs block locations
`df-map.csv` records **canonical staging paths** (e.g.
`/scratch/preframr/{train,eval}/...`) which need NOT exist on disk; the actual
`.blocks.npy` live in the run's work_dir. Tools that forward a checkpoint
(`predict.py`, `audit_checkpoint_per_class`) need the df-map's `dump_file` paths
to resolve to where the blocks actually are — `audit_checkpoint_per_class`
sidesteps this by globbing `work_dir/eval*/*/*.blocks.npy` directly; `predict.py`
needs the df-map rewritten to the work_dir. Treat df-map paths as labels, not
file locations.

## Invariants / gotchas (the things that bite)

- **Settled freq only.** Never derive freq from a raw lo/hi byte; use
  `combine_reg` / `freq_unq`. (16-bit half-updates → octave-high garbage.)
- **Freq passes re-fire per block.** `expand_to_literal_form` decompiles op45/48
  back to SETs; `run_freq_block_passes` must re-run at block start or the freq
  stack silently drops (`freq_passes_re_fire_on_blocks`).
- **Content tier is deliberately lossy** (cents/preset/transpose cent-binned); raw
  freq survives in `freq_unq`. Content-tier-OFF is byte-perfect vs raw.
- **Per-op accuracy needs `vocab_atom`.** A tokens-csv-row-index proxy mis-assigns
  ~58% of ops on a Unigram tokenizer; the per-op read must reconstruct uid→atom
  via the tokenizer (`audit_checkpoint_per_class.build_vocab_atom_map`).
- **Opt-in flags only.** New passes ship gated off (`GATE_FLAGS`); the framework
  declares the matching `--flag` and floors the tokens version. Merging is always
  safe; the deliberate release is the version bump.

## Release

`fallback_version` in `pyproject.toml` tracks the latest `v*` git tag (setuptools-
scm). Tag `vX.Y.Z` → `release.yml` publishes to PyPI (OIDC). CI (`ci.yml`) runs
black / pylint / pyright / pytest+coverage(≥85) on 3.10–3.12. Lint forbids
narrative `#` comments and >5-line docstrings (`tests/test_lint.py`).
