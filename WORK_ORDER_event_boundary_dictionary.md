# WORK ORDER: event-boundary-respecting dictionary (execute mechanically, then delete this file)

**Mission:** make the unigram BPE-as-dictionary **boundary-respecting**: the trainer must never
form a merge that crosses a grammar-unit boundary of the event stream. Mechanism: segment the
training text into one word per grammar unit using the *real stream parser* as the segmenter
(separator injection in the `.uni` text + a whitespace pre-token split). Vocab purity then
guarantees, by construction, that no encoded stream can ever contain a cross-unit token — the
runtime encode path needs **zero** changes. Everything below is pre-researched against this repo
at commit `a6c4041`; **no further research is required**. If an anchor drifted, re-locate by the
named symbol, not the line number.

**Context (all you need; do not go reading other repos):** the canonical 14M learnability run in
preframr-xpt compared atoms-only (tkvocab=0) vs unconstrained unigram tkvocab=2048. De-confounded
verdict: unconstrained merges cost ~1.2–1.3× bits/canonical-atom at matched training steps while
compressing 2.73×, and the harm mechanism is merges **welding content across event boundaries**.
A boundary-respecting dictionary is therefore expected to keep most of the compression (~1.8–2.5×,
modal event = 3 atoms) at ≈parity quality. The A/B experiment itself happens later, xpt-side — it
is NOT part of this work order. Your deliverable is purely the mechanics in this repo. Bonus
expected from the same change: per-unit words shrink the UnigramTrainer lattice from 35K–150K-char
sentences to ≤~100-char words, retiring the SIGSEGV class behind the `RUST_MIN_STACK` workaround
(keep the workaround anyway).

## Ground rules (read fully before starting)

- Branch `work/event-boundary-dictionary` off current `origin/main`. One commit per phase
  (messages given).
- Gate after every phase: `./run_tests.sh` (black, pylint, pyright, pytest, coverage ≥85%). It
  must be green at P0 before any change; if not, STOP and report.
- Lint constraints for all NEW code (enforced by `tests/test_lint.py`): docstrings ONE paragraph
  ≤5 lines; NO narrative `#` comments (directive only, e.g. `# pylint: disable`).
- **No codec change**: `stream.EVENT_FORMAT_VERSION`, `events/dataset.ATOM_CACHE_VERSION`, the
  atom alphabet, `stream.encode`/`decode`, and `canonical_writes` must be untouched — the atom
  stream and the `.atoms.zst` caches stay byte-identical. The fidelity suites
  (`tests/test_events_roundtrip.py`, `tests/test_events_corpus.py`, `tests/test_events_acceptance.py`)
  must stay green UNMODIFIED.
- Unigram path only. Do NOT touch the `tokenizer == "bpe"` branch in `train_worker.get_tk`
  (legacy), the parse-domain (op,reg,subreg,val) corpus path, or `BOUNDARY_ISOLATION_NS`
  semantics (the new words subsume it for events; the `bpe_isolate_boundaries` flag stays as-is —
  harmless compose, xpt specs may reference it).
- Do NOT release. `pyproject.toml` `fallback_version` is already `0.51.0` (pending); do not bump,
  do not tag (`release.yml` fires on `v*` tags — the operator releases).
- Final step of the LAST phase: `git rm WORK_ORDER_event_boundary_dictionary.md` — this file must
  not survive the work.

## P0 — preflight

```bash
git fetch origin && git switch -c work/event-boundary-dictionary origin/main
./run_tests.sh   # green or STOP
```

## P1 — `stream.unit_starts`: the parser as segmenter

The decoder (`_Decoder`, `preframr_tokens/events/stream.py` ~line 835) already steps the stream
unit-by-unit in `parse()` (~line 993): leading frame-count varint → `[VOICE][HEADER_KIND][payload]`
headers → frame groups of `[DT digits]` then `[VOICE]` markers each followed by
`[KIND][payload]` events (`_EVENT_KINDS` loop). Record those positions:

1. In `_Decoder.__init__`: add `self.unit_starts: list[int] = []`.
2. In `parse()`, append `self.pos` to `self.unit_starts` at exactly five sites: (a) before
   `n = self._u()`; (b) inside the header while-loop, before each `self._parse_header()`; (c) in
   the main loop, before each `dt = self._u()`; (d) before consuming each voice marker
   (`voice = t[self.pos] - VOICE_BASE`); (e) before each `self._parse_event(cur_f, voice)`.
3. Module-level public helper (export in `__all__`):

   ```python
   def unit_starts(tokens) -> list[int]:
       """Grammar-unit start indices of an atom stream (frame-count varint, per-voice headers,
       DT runs, voice markers, events) -- the parser itself is the segmenter, so payload digits
       and DT digits are distinguished exactly. Raises on invalid or KEYFRAME-bearing streams
       (segment whole-tune ``encode`` output only; ``strip_keyframes`` first if needed)."""
       d = _Decoder(tokens)
       d.parse()
       return d.unit_starts
   ```

4. Tests (`tests/test_events_stream.py`): on the existing driver fixtures, assert
   `unit_starts(atoms)[0] == 0`; strictly increasing; every recorded start indexes a digit
   (frame-count/DT), a voice marker, or a `_EVENT_KINDS` atom; and full coverage:
   `decode(atoms)` unchanged, and the spans tile the stream (last span ends at `len(atoms)`).

Commit: `feat(events): unit_starts -- parser-derived grammar-unit segmentation`

## P2 — n-space wrapper in `events/dataset.py`

```python
def unit_starts(n_ids) -> list[int]:
    """Grammar-unit start indices for an n-space stream (``dump_token_ids`` output): shift the
    +1 PAD offset and delegate to :func:`stream.unit_starts` (positions are offset-invariant)."""
    return stream.unit_starts([int(n) - 1 for n in n_ids])
```

Add to `__all__`. Test (`tests/test_events_dataset.py`): equals
`stream.unit_starts` on a fixture's `dump_token_ids` after the −1 shift; starts at 0.

Commit: `feat(events): n-space unit_starts wrapper`

## P3 — separator injection in `RegTokenizer.train_tokenizer`

`preframr_tokens/regtokenizer.py`:

1. In `__init__` (~line 50, next to `self.isolation_ns = None`): add `self.unit_segmenter = None`.
2. In `write_uni` inside `train_tokenizer` (~line 160): after
   `encoded = self.encode_unicode(orig_seq)`, inject a space at each unit start:

   ```python
   if self.unit_segmenter is not None:
       starts = self.unit_segmenter(orig_seq)
       assert starts and starts[0] == 0
       bounds = starts[1:] + [len(encoded)]
       encoded = " ".join(
           encoded[a:b] for a, b in zip(starts, bounds)
       )
   ```

   Safe by construction: event-alphabet chars live at `UNICODE_BASE`+ (0x300+, splitters resync
   to 0 for events) and parse-domain low ids map to `string.punctuation` — U+0020 can never be an
   atom char in either domain.
3. Tests (new `tests/test_dictionary_segmentation.py`): build the events tokenizer
   (`events_dataset.make_tokenizer`), set `unit_segmenter = events_dataset.unit_starts`, render a
   fixture's uni text via the same code path (factor `write_uni`'s body into a small method if
   that is the cleanest way to test it); assert spaces sit exactly at unit starts and
   `decode_unicode(text.replace(" ", ""))` round-trips the ids.

Commit: `feat(tokenizer): unit-segmented .uni emission (separator-injected words)`

## P4 — whitespace split in the unigram pre-tokenizer

`preframr_tokens/train_worker.py`, `_build_unigram_pre_tokenizer` (~line 48): the separator must
bound words and then vanish (never enter any piece):

```python
def _build_unigram_pre_tokenizer(isolation_chars):
    """Compose the unigram pre-tokenizer: whitespace-bounded grammar-unit words first, then
    isolation singletons, then legacy punctuation."""
    parts = [pre_tokenizers.WhitespaceSplit()]
    pattern = _isolation_char_class(isolation_chars)
    if pattern is not None:
        parts.append(
            pre_tokenizers.Split(pattern=Regex(pattern), behavior="isolated", invert=False)
        )
    parts.append(pre_tokenizers.Punctuation())
    return pre_tokenizers.Sequence(parts)
```

Note the saved tkmodel JSON persists this chain, so runtime `encode` applies it automatically;
runtime text contains no spaces, so `WhitespaceSplit` is a no-op there — exactly the intent
(vocab purity does the enforcement). The unigram `decoders.Metaspace` decoder is unaffected
(pieces contain neither `▁` nor spaces). Test: `get_tk(…, tokenizer="unigram")` pre-tokenizes
`"ab cd"`-style synthetic text into two words with the space removed; the `"bpe"` branch is
untouched.

Commit: `feat(tokenizer): whitespace-word unigram pre-tokenizer`

## P5 — wire up the events corpus path (unconditional)

`preframr_tokens/corpus.py`, events `preload` (~line 474): immediately after
`self.tokenizer.tokens = events_dataset.events_alphabet()`, add:

```python
self.tokenizer.unit_segmenter = events_dataset.unit_starts
```

Unconditional — boundary-respecting is THE dictionary behavior for events from now on (matches
the repo's no-macro-flags philosophy; the unconstrained variant stays reproducible via released
0.50.0). Leave the `bpe_isolate_boundaries` branch directly below it untouched. The parse-domain
corpus path never sets `unit_segmenter`, so legacy behavior is preserved there.

Test (`tests/test_corpus.py` or the events corpus suite): after an events `preload` with
`tkvocab>0` on fixtures, the trained tokenizer exists and the assertions of P6 hold.

Commit: `feat(corpus): events dictionary trains on grammar-unit words`

## P6 — the end-to-end weld-free invariant (the point of all this)

New test in `tests/test_dictionary_segmentation.py`: train a tiny unigram dictionary
(`tkvocab` ≈ 256) over the driver fixtures through the real path
(`RegTokenizer.train_tokenizer` with the segmenter set, or events `preload`), then for each
fixture tune assert no emitted token's atom span crosses a unit boundary:

```python
ids = events_dataset.dump_token_ids(df)
starts = set(events_dataset.unit_starts(ids))
seq = tk.encode(np.asarray(ids, dtype=np.int32))
pos = 0
for tid in seq:
    alen = len(tk.decode(np.asarray([tid], dtype=np.uint32)))
    assert not any(p in starts for p in range(pos + 1, pos + alen))
    pos += alen
assert pos == len(ids)
```

Also assert whole-stream decode equality (`tk.decode(seq)` == `ids`) and, informationally, log
`len(ids) / len(seq)` (compression on fixtures; no threshold assert). Add a smoke check that
encoding the longest fixture stream completes in sane wallclock (runtime text is one long word;
the trainer-side SIGSEGV class does not apply to encode, but measure once).

Commit: `test(tokenizer): weld-free vocab invariant + segmentation round-trip`

## P7 — README + cleanup + ship

1. README (the authoritative alphabet/grammar reference): add a short "Dictionary segmentation"
   paragraph under the tokenizer section: unigram BPE-as-dictionary trains over grammar-unit
   words (parser-segmented: headers, DT runs, voice markers, events), so no learned piece can
   cross a unit boundary; runtime encode is unchanged and enforcement is vocab purity.
2. `git rm WORK_ORDER_event_boundary_dictionary.md` (this file must not survive).
3. `./run_tests.sh` green, then:

```bash
git push -u origin work/event-boundary-dictionary
gh pr create --fill --title "feat(tokenizer): event-boundary-respecting dictionary (grammar-unit words)"
```

Commit: `docs(readme): dictionary segmentation; remove executed work order`

PR body must state: no codec change (EVENT_FORMAT_VERSION/ATOM_CACHE_VERSION untouched, atom
streams byte-identical), unigram-only, weld-free invariant tested end-to-end, fixture compression
ratio observed, operator tags 0.51.0 when ready, and the xpt-side A/B (boundary-respecting
tkvocab=2048 vs the atoms-only baseline, gated in bits/canonical-atom) is the follow-up that
consumes this.
