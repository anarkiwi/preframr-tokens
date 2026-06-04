# Parse-path performance review (post codebook-unify refactor)

Branch: `review/parse-perf-postrefactor`
Scope: the encode-side parse pipeline driven by `RegLogParser.parse()`
(`preframr_tokens/reglogparser.py`) and the macro-pass chain, as it stands
after the codebook-unify refactor (PR #51, `a0a76e6..90d8a48`).

No corpus of `*.dump.parquet` is available locally (only raw `.d64`/`.d71`
disk images under `/scratch/preframr/hvsc`), so these findings are from static
analysis of the hot path, not a profile. Each item lists where it fires, how
often, and a rough effort/impact. **Validate against a profile before
investing** — the order below is my best estimate of payoff.

## What the refactor got right (no action)

- The unified codebook machine (`macros/codebook.py`) is clean per-row dispatch
  with `__slots__` codecs and a shared `_Codebook` — no measurable overhead vs.
  the legacy per-family decoders it replaced.
- `FrameWalker` (`macros/walker.py`) already hoists columns to Python lists
  (`.tolist()`), reuses one `_FastRow` buffer, and has a `set_fastpath`. The
  per-row decode loop is already well-tuned.
- `_build_last_diff` (`macros/state.py:289`) is a single groupby, replacing the
  documented per-reg mask scan. Good.
- The parse audit (`parse_audit.py`) is a true no-op when disabled (default), so
  the `audit.after(...)` calls peppered through `parse()` cost nothing in prod.

## Findings, highest payoff first

> **UPDATE (verified against fixtures — do NOT apply as written).** Profiling
> confirms the impact (`_add_voice_reg` is 16–25% of parse, the preview calls
> ~half of that), but the equivalence harness + a direct probe showed the gate
> is **not** invariant to the transform, so this is unsafe as a mechanical swap:
> `_add_voice_reg` collapses voice regs to `reg % VOICE_REG_SIZE`, so on the
> voiced form `_filter`'s ctrl-density count sums across voices (commando: max
> 3/frame) where the raw form keeps regs 4/11/18 separate (max 1/frame). The
> digi-reject threshold `c_max > 6` is tuned on the voiced value, so filtering
> the raw `xdf` makes digi/multispeed rejection ~3× looser. The length check
> also differs (voiced form is ~900 rows longer). All 9 fixtures sit far below
> the threshold (harness shows zero diff), but a dense-ctrl tune could flip.
> Left in place. A safe version would compute the voiced-form ctrl-density and
> length cheaply from the raw arrays without building the full structure — a
> larger change that needs a digi-boundary fixture and maintainer intent.

### 1. Redundant `_add_voice_reg` per rotation, purely to gate `_filter`
`reglogparser.py:1042-1045` then `:1055`

```python
pre_passes_voice_preview = self._add_voice_reg(xdf.copy(), zero_voice_reg=True)
if not self._filter(pre_passes_voice_preview, name):
    break
... # passes run
xdf = self._add_voice_reg(xdf, zero_voice_reg=True)   # built again, for real
```

`_add_voice_reg` is one of the heaviest helpers in the file (multiple
`groupby(...).cumsum()` / `transform("max")`, several `shift`s, a full
`ffill`). It runs **twice per rotation**: once on a throwaway `xdf.copy()` only
to feed `_filter`, once for the real output.

Verified: `_filter` (`:755`) references only `reg`/`op`/FRAME_REG/MODE_VOL_REG
via `norm_df`/`ctrl_match` — it **never touches `VOICE_REG`**. So the voice-reg
transform is unnecessary for the gate. Moreover the gate criteria (song length,
vol-per-frame density, ctrl-changes-per-frame) are **rotation-invariant**, so
the check could be hoisted out of the `_rotate_voice_augment` loop entirely and
run once on the un-voiced `df`.

- Impact: removes ~`max_perm` (≤3) expensive `_add_voice_reg` calls per file,
  i.e. roughly a third of all `_add_voice_reg` work.
- Effort: low-medium. Replace the preview with `self._filter(self._norm_pr_order(xdf), name)`
  (or hoist before the loop). Needs a test run to confirm the gate decision is
  unchanged — the preview currently `break`s the loop; confirm semantics.

### 2. `_consolidate_frames` is a Python per-row dict loop
`reglogparser.py:804-863`

```python
rows = norm_df(orig_df.copy()).to_dict("records")
...
while i < n:               # pure-Python scan, building dict rows
    ...
df = pd.DataFrame(out)
```

`norm_df` already copies + does `frame_reg` (another copy) + `cumsum`/`diff`,
then everything is materialised to a list of dicts and walked in Python,
rebuilding dicts, then re-framed. Runs once per file (pre-rotation), but is
O(rows) in Python with dict-per-row overhead — meaningful on long songs.

The operation is a run-length collapse of marker-only frame runs; it is
expressible as a vectorised numpy grouping (identify maximal `_is_marker` runs
via boundary diff, sum `_units` per run with `np.add.reduceat`). 

- Impact: medium (once/file, but Python-bound on long tunes).
- Effort: medium. The Python version is the reference — keep it behind a test
  and diff outputs.

### 3. `_build_decode_state` rebuilt from scratch by every state-consuming pass
`macros/state.py:301`, `macros/passes_base.py` (`requires_state`)

Each pass that needs decode state re-derives it (`_build_last_diff` groupby +
`DecodeState` alloc of several `np.zeros(MAX_REG+1)` + dict/defaultdict churn),
and `FrameWalker.__init__` re-extracts every column via `.to_numpy()` **and**
`.tolist()` on each construction. Across the pipeline this happens ~30×
(≈18 pre-rotation passes + ~16 per-rotation passes × ≤3 rotations).

Because the df mutates between passes, the state genuinely must be rebuilt —
there's no trivially correct cache. But two cheaper wins exist:
- `requires_state` (`passes_base.py`) does `df.reset_index(drop=True).copy()`
  **unconditionally**, before `_build_decode_state` may return `None`. Order the
  cheap applicability check (op/reg `.any()`) before the copy, as the
  array-based passes (e.g. `TransposePass:51`) already do.
- `DecodeState.__init__` eagerly allocates ~6 per-reg arrays + ~15
  dict/defaultdict/list fields even for passes that touch one of them. Lazy-init
  the rarely-used pending buffers if state construction shows up in a profile.

- Impact: low-medium, diffuse. Confirm with a profile that state-build is hot
  before refactoring `DecodeState`.
- Effort: low for the `requires_state` reorder; medium for lazy state fields.

### 4. Trailing/leading marker strip rebuilds the whole df per row
`reglogparser.py:1023-1028`

```python
while not df.empty and frame_match(df.iloc[-1]):
    df = df.head(len(df) - 1)
while not df.empty and (df.iloc[0]["reg"] == MODE_VOL_REG and df.iloc[0]["val"] == 15):
    df = df.tail(len(df) - 1)
```

Each iteration allocates a fresh DataFrame via `head`/`tail`, so stripping *k*
trailing frames is O(k) copies of an O(n) frame. Find the cut indices in one
vectorised pass and slice once.

- Impact: low (k usually small), but trivially fixable and removes a
  pathological case (many trailing frames).
- Effort: low.

### 5. `combine_val` builds string-named temp columns in a loop
`reglogparser.py:340-356`, called per voice for freq/pcm/filter via
`_combine_regs` (`:796`, 7 `combine_reg` calls/file)

For each byte it creates a `str(i)` column, masks, `ffill().fillna(0)`, casts,
shifts, then sums them back. This is several full-column passes + temp columns
per combine. The same little-endian byte settle is a handful of numpy ops on the
already-extracted arrays (mask → ffill via `np.maximum.accumulate` on a filled
index, shift, add). Runs once/file but over the full raw write stream (the
largest df in the pipeline, pre-squeeze).

- Impact: medium (operates on the widest df, before `_squeeze_changes` shrinks it).
- Effort: medium.

### 6. Vectorisable micro-spots (low effort, low-but-free impact)
- `_freq_to_cent_index` (`:334`): `fv.map(lambda w: fm.fi_map[int(w) & 0xFFFF])`
  is a Python-lambda per element. `fi_map` is a plain dict — do
  `(fv.astype("int64") & 0xFFFF).map(fm.fi_map)` for a C-level map. (Audio-render
  path, not parse-core, but free.)
- `_add_frame_reg` (`:516`): `largest_irqs_sum = sum([k*v for k,v in irq_counts.items()])`
  → `int((irq_counts.index * irq_counts.values).sum())`.
- `_read_df` (`:421`): `chips = df["chipno"].nunique()` is computed for every
  file though the multi-chip branch is rare; `nunique` is fine, but the column
  is dropped right after — confirm it's needed before the `<2` check ordering.
- `FreqMapper.__init__` (`reg_mappers.py`): builds three 65 536-entry dicts
  (`rq_map`, `fi_map`, `if_map`) by Python loop on **every** `RegLogParser`
  construction (once/file). If `RegLogParser` is ever reused across files, cache
  the mapper; otherwise consider numpy arrays instead of dicts for `fi_map`.

## Structural note (not a bug)

The 3× cost of `_rotate_voice_augment` re-running the full pass chain per voice
permutation is inherent to the augmentation (it produces 3 distinct training
rotations). The only safe savings there are hoisting rotation-invariant work out
of the loop — Finding #1 is the clearest instance.

## Suggested order

1, 4, 6 are low-risk quick wins (start here). 2 and 5 are the larger
once-per-file vectorisations. 3 only if a profile shows state-build is hot.
Before any of this: generate a handful of `*.dump.parquet` and run
`cProfile`/`py-spy` on `RegLogParser.parse` to confirm the ranking.
