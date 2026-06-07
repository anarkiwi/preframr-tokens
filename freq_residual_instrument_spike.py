"""READ-ONLY hypothesis spike (UNTRACKED, throwaway) for NEXT_BUILD_additive_instrument_model.md.

Load-bearing claim of the build: the per-frame freq RESIDUAL (decompose_voice "mod", in cents),
grouped by the INSTRUMENT the note plays, collapses to ~ONE shared modulation generator per
instrument (Daglish triangle / Hubbard sine-table / slide). If true, the instrument-keyed freq lane
is justified and this script already tells us which generator each instrument needs. If false, the
model is wrong and we learned it in an afternoon, before any byte-exact codebook surgery.

No encoder changes. No Docker/network -- uses already-cached/pre-rendered dumps.
Covers all three driver models the build reconciles:
  trap     = Ben Daglish  (WEMUSIC, triangle vibrato)
  commando = Rob Hubbard  (sine-table vibrato)
  cauldron = Linus        (Cauldron II Remix -- cross-voice chorus/detune)
  grid_runner = Jammer    (extra real-tune sanity)
Run:  python freq_residual_instrument_spike.py trap commando cauldron grid_runner
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_tokens.macros import pitch_grid
from preframr_tokens.macros.generator_fit import decompose
from tests.sid_fixtures import DRIVER_FIXTURES, GRID_RUNNER, ensure_driver_fixture, ensure_dumps

# (freq_lo, freq_hi, ctrl) per voice; AD = ctrl+1, SR = ctrl+2 (SID voice register layout).
_FREQ_VOICE_REGS = {0: (0, 1, 4), 1: (7, 8, 11), 2: (14, 15, 18)}

# Pre-rendered local HVSC dump for the Linus Cauldron II Remix (the chorus driver), not in
# DRIVER_FIXTURES; resolved directly so the spike needs no edit to tracked test infra.
_CAULDRON_DUMP = Path(
    "/scratch/preframr/hvsc/MUSICIANS/L/Linus/Cauldron_II_Remix.1.dump.parquet"
)


def load_reg_df(name):
    """Resolve a tune name to its cached raw register-log DataFrame (clock/irq/chipno/reg/val)."""
    if name in DRIVER_FIXTURES:
        return pd.read_parquet(ensure_driver_fixture(name)), ensure_driver_fixture(name).name
    if name == "grid_runner":
        _head, wide = ensure_dumps(GRID_RUNNER)
        return pd.read_parquet(wide), wide.name
    if name == "cauldron":
        return pd.read_parquet(_CAULDRON_DUMP), _CAULDRON_DUMP.name
    raise KeyError(f"unknown tune {name!r}")


def per_frame_state(reg_df) -> np.ndarray:
    """Reconstruct per-frame settled SID register state[frame, 25] from a raw register log
    (same forward-fill as encoding_complexity.input_freq_complexity)."""
    d = reg_df[reg_df["chipno"] == 0]
    frames = list(dict.fromkeys(d["irq"].tolist()))
    fpos = {f: i for i, f in enumerate(frames)}
    state = np.zeros((len(frames), 25), dtype=np.int64)
    cur = np.zeros(25, dtype=np.int64)
    last = 0
    for f, reg, val in zip(d["irq"].tolist(), d["reg"].tolist(), d["val"].tolist()):
        i = fpos[f]
        if i != last:
            state[last + 1 : i + 1] = cur
            last = i
        if 0 <= int(reg) < 25:
            cur[int(reg)] = int(val)
    state[last:] = cur
    return state


def instrument_per_frame(ctrl, ad, sr):
    """Instrument identity per frame = (waveform bits, AD, SR) latched at each gate rising edge and
    forward-filled across the note span -- the same onset-keyed (ctrl/AD/SR)-program identity
    InstrumentProgramPass interns. Returns (key_per_frame, onset_mask)."""
    n = len(ctrl)
    keys = [None] * n
    onset = np.zeros(n, dtype=bool)
    cur = None
    prev_gate = 0
    for i in range(n):
        gate = int(ctrl[i]) & 1
        if gate and not prev_gate:  # rising edge: new note -> latch instrument
            cur = (int(ctrl[i]) & 0xF0, int(ad[i]), int(sr[i]))
            onset[i] = True
        prev_gate = gate
        keys[i] = cur
    return keys, onset


def gen_signature(mod_seq):
    """The non-static generator runs decompose() fits over a cents-mod sub-sequence: drop HOLD-of-0
    (static, no modulation); return the multiset of (kind, params) for what's left."""
    runs = decompose(list(int(x) for x in mod_seq))
    sig = []
    for kind, _i, _ln, params in runs:
        if kind == "HOLD" and (params is None):
            # HOLD run; static only if the held value is 0
            pass
        sig.append((kind, params))
    return sig


def chorus_check(state):
    """The Cauldron driver claim: two voices SHARE the note-index stream and differ only by a small
    constant per-voice detune (chorus phasing). For each voice pair, on co-voiced frames report the
    fraction at the SAME semitone (|cents offset| < 50) and the median detune in cents."""
    freqs = {}
    for v, (lo, hi, c) in _FREQ_VOICE_REGS.items():
        f = ((state[:, hi] << 8) | state[:, lo]).astype(np.float64)
        gate = (state[:, c] & 1) == 1
        freqs[v] = (f, (f > 8) & gate)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        fa, va = freqs[a]
        fb, vb = freqs[b]
        both = va & vb & (fa > 0) & (fb > 0)
        n = int(both.sum())
        if n < 16:
            print(f"           chorus v{a}~v{b}: co-voiced {n} frames (too few)")
            continue
        cents = 1200.0 * np.log2(fa[both] / fb[both])
        within_note = float(np.mean(np.abs(np.round(cents / 100.0)) == 0))
        print(
            f"           chorus v{a}~v{b}: co-voiced {n}  same-note(<50c) {within_note:.0%}  "
            f"median detune {np.median(cents):+.1f}c  IQR {np.percentile(cents,75)-np.percentile(cents,25):.1f}c"
        )


def analyse(name):
    reg_df, fname = load_reg_df(name)
    state = per_frame_state(reg_df)
    print(f"\n======== {name}  ({fname}, {len(state)} frames) ========")
    chorus_check(state)
    for v, (lo, hi, c) in _FREQ_VOICE_REGS.items():
        freq = (state[:, hi] << 8) | state[:, lo]
        ctrl, ad, sr = state[:, c], state[:, c + 1], state[:, c + 2]
        dec = pitch_grid.decompose_voice(freq)
        mod, voiced = dec["mod"], dec["voiced"]
        if not voiced.any():
            print(f"  voice {v}: silent")
            continue
        keys, onset = instrument_per_frame(ctrl, ad, sr)

        # BEFORE: the doc's "distinct residual VALUES" metric, ungrouped.
        before_vals = {int(m) for m, vo in zip(mod, voiced) if vo and m != 0}

        # Split into note spans (onset..next onset); bucket each span's mod by instrument.
        onset_idx = [i for i in range(len(freq)) if onset[i]] or [0]
        spans_by_instr = defaultdict(list)
        for k, start in enumerate(onset_idx):
            end = onset_idx[k + 1] if k + 1 < len(onset_idx) else len(freq)
            seg = [int(mod[j]) for j in range(start, end) if voiced[j]]
            if seg:
                spans_by_instr[keys[start]].append(seg)

        # AFTER: per instrument, the set of generator signatures across its spans.
        instr_gens = {}
        moving_instruments = 0
        after_total = 0
        for key, spans in spans_by_instr.items():
            sigs = Counter()
            distinct_vals = set()
            for seg in spans:
                for m in seg:
                    if m != 0:
                        distinct_vals.add(m)
                if any(m != 0 for m in seg):
                    # signature of the modulation in this span (kinds only -- params vary by phase)
                    sigs[tuple(sorted({g[0] for g in gen_signature(seg)} - {"HOLD"}))] += 1
            if distinct_vals:
                moving_instruments += 1
                instr_gens[key] = (len(spans), len(distinct_vals), dict(sigs))
                after_total += max(1, len({s for s in sigs}))

        print(
            f"  voice {v}: voiced={int(voiced.sum())}  instruments={len(spans_by_instr)} "
            f"(moving={moving_instruments})"
        )
        print(
            f"           BEFORE distinct mod values (ungrouped) = {len(before_vals)}   "
            f"AFTER sum of per-instrument generator-kinds = {after_total}"
        )
        for key, (nspans, ndv, sigs) in sorted(instr_gens.items()):
            wf, a, s = key
            print(
                f"           instr wf=${wf:02X} AD=${a:02X} SR=${s:02X}: "
                f"{nspans} spans, {ndv} distinct mod-values, gen-kinds={sigs}"
            )


def main():
    names = sys.argv[1:] or ["trap", "commando", "cauldron", "grid_runner"]
    for name in names:
        try:
            analyse(name)
        except Exception as e:  # noqa: BLE001 -- spike: surface any fixture/parse error
            print(f"\n{name}: FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
