"""Absolute-anchored (octave-chain-neutral) universal-pitch encoder / decoder.

The validated per-voice encoders (pitch_universal_encode / _encode_fix) snap each
NOTE onset to a 12-TET grid index and ride MOD/GLIDE excursions as additive float
deltas on a single running Fn THREADED through the WHOLE voice stream. The NOTE op
re-anchors the grid INDEX (exact integer) but decode ALSO threads a float cur_fn
that every MOD/GLIDE mutates per-frame and the NEXT span rides. The GLIDE rescale
leaves cur_fn off-grid, and a leading/post-RAW MOD has NO grid base at all -- so
the float error accumulates geometrically across spans (the 2.60% runaway; worst
cases 15-36 octaves, decoded Fn ~1e14).

THE FIX (this module) -- kill the cross-span FLOAT threading entirely:
  - Every NOTE onset sets the base to the EXACT grid Fn at the onset's absolute
    snapped index, recomputed from scratch -- never carried forward as a float.
    Integer-semitone index accumulation is exact and cannot diverge.
  - MOD/GLIDE excursions CARRY their exact onset base Fn (the raw-stream value) and
    decode rides the deltas from THAT carried base, not from a threaded cur_fn. A
    glide/MOD therefore can never move the next note's base, and a leading/post-RAW
    MOD (no grid anchor) is still exact because it rides its own carried base.
    MOD/GLIDE round-trip is byte-exact on modulation (raw deltas on the raw base).
  - GLIDE re-targets its END to the snapped grid endpoint so it connects two grid
    notes; intermediate frames are interpolated in log-Fn space between the exact
    start base and the exact end grid Fn (no rescale-of-near-zero-span explosion).
  - The micro-detune (chorus) field and the REST (note-off / power-zero) gate are
    preserved. An OUT-OF-GRID RAW escape carries the exact Fn for extreme-Fn
    percussive transients beyond the musical grid (decoded exactly, never snapped).

Model-facing NOTE token stays the SMALL shared relative semitone interval (the
learnability prize); the carried base on MOD is a decode-side reconstruction
field, NOT part of the note-interval alphabet. Glides are NOT a separate token --
they replay as a MOD (raw deltas on the carried base, byte-exact); the only glide
handling left is advancing the grid index to the snapped endpoint so the next NOTE
interval is grid-relative (the next onset re-anchors absolutely, so a glide can
never move the next note's base, and no rescale/retarget distortion is possible).

Token format:
  ("NOTE",  interval_semitones, micro_detune_cents)
  ("RAW",   fn)                                          -- out-of-grid / extreme
  ("REST",  fn)                                          -- Fn -> 0 note-off gate
  ("MOD",   deltas_tuple, n, base_fn)        -- vibrato/arp/glide; carried base_fn
"""

import numpy as np

from preframr_tokens.codec import freq_relative as FR
from preframr_tokens.codec import pitch_universal as P
from preframr_tokens.codec import pitch_universal_encode as E

SEMI = P.SEMI
CENTS_PER_SEMI = E.CENTS_PER_SEMI

GRID_LO_IDX = -48  # ~4 octaves below the grid origin (musical floor)
# The 16-bit Fn register caps at 65535; with phase in [0,SEMI) the grid index of
# any representable Fn (1..65535) is round(log2(Fn)/SEMI) <= 192 (corpus max = 192,
# 100% of onsets <= 192). Set the ceiling AT the hardware ceiling so the whole
# musical range emits real NOTE intervals; only Fn<=0 (-> REST) escapes the grid.
# A lower ceiling forced ~99.6% of onsets to RAW (the regression this fixes).
GRID_HI_IDX = 192  # = Fn 65535 hardware ceiling; no Fn lands above


def _in_grid(idx):
    return GRID_LO_IDX <= idx <= GRID_HI_IDX


def _grid_fn(phase_log2, idx, micro_cents=0.0):
    """EXACT grid Fn at absolute semitone index idx under the voice phase, plus an
    optional per-note micro-detune (cents). Recomputed from scratch every call --
    no float carried forward -> cannot accumulate."""
    return 2.0 ** (phase_log2 + (idx + micro_cents / CENTS_PER_SEMI) * SEMI)


def _glide_end_idx(phase_log2, e, n, base_fn, cur_idx):
    """Advance the grid index across a MOD span if (and only if) it is a glide that
    lands on an in-grid note. IDENTICAL on encode and decode (both have phase, the
    deltas, n, base_fn) -> the relative NOTE chain stays in sync. A glide that ends
    out of grid, or any non-glide MOD, leaves cur_idx unchanged. Only advances once
    the chain is seeded (cur_idx is not None): before the first NOTE the decode side
    has no established phase, so a leading glide must NOT advance the index (the
    next NOTE re-seeds absolutely regardless)."""
    if cur_idx is None:
        return None
    p = len(e)
    end_fn = float(base_fn)
    for k in range(n):
        end_fn += e[k % p]
    if base_fn > 0 and end_fn > 0 and E._is_glide(e, n, base_fn):
        end_idx = int(np.round((E._log2(end_fn) - phase_log2) / SEMI))
        if _in_grid(end_idx):
            return end_idx
    return cur_idx


def encode_voice(f, min_hold=3, detune=True, phase_quant=None):
    """Absolute-anchored encode. Returns (tuning_cents, phase_log2, tokens). The
    decode round-trips with octave-error 0 BY CONSTRUCTION: every onset re-anchors
    to an exact grid index and every excursion carries its exact base -> no
    cross-span float threading exists to diverge.

    phase_quant: optional fn(phase_log2)->phase_log2 that rounds the fitted phase
    to the SAME quantum the serializer will store/decode with. The whole encode
    (grid idx, micro-detune, glide endpoints) then runs under the exact phase the
    decoder reconstructs -> the glide index-advance can never disagree across the
    encode/decode phase-quantization boundary (the residual desync source)."""
    f = [int(x) for x in f]
    fit = E.fit_voice_tuning(f, min_hold)
    if fit is None:
        # too few held base pitches to fit a per-voice phase (continuous
        # vibrato/arp/sweep or sparse voices). Fall back to the canonical grid
        # (phase 0): nearest-semitone snap is bounded +-50c for ANY phase, and the
        # per-note micro-detune captures the residual -> still emits NOTE intervals
        # (the alphabet prize) instead of dumping the whole voice to RAW.
        tuning_cents, phase = 0.0, 0.0
    else:
        tuning_cents, _, _, _, phase = fit
    if phase_quant is not None:
        phase = phase_quant(phase)
        tuning_cents = phase / SEMI * CENTS_PER_SEMI
    toks = FR.encode_freq(f)

    out = []
    cur_fn = 0  # raw running Fn (FR stream) -- encode side only
    last_idx = None  # last ABSOLUTE grid index actually emitted
    for t in toks:
        if t[0] == "NOTE":
            cur_fn += t[1]
            if cur_fn <= 0:
                out.append(("REST", int(cur_fn)))
                continue
            exact = (E._log2(cur_fn) - phase) / SEMI
            idx = int(np.round(exact))
            micro = E._quant_detune((exact - idx) * CENTS_PER_SEMI) if detune else 0.0
            # out-of-grid OR a snapped grid Fn that would overflow the 16-bit Fn
            # register (top-octave edge) -> RAW (exact); never corrupt the register.
            if not _in_grid(idx) or _grid_fn(phase, idx, micro) > 65535.0:
                out.append(("RAW", int(cur_fn)))
                last_idx = None  # a raw note breaks the relative chain seed
                continue
            interval = idx if last_idx is None else idx - last_idx
            out.append(("NOTE", interval, micro))
            last_idx = idx
        else:
            _, e, n = t
            base_fn = cur_fn
            p = len(e)
            end_fn = base_fn
            for k in range(n):
                end_fn += e[k % p]
            out.append(("MOD", tuple(e), n, int(base_fn)))
            last_idx = _glide_end_idx(phase, e, n, base_fn, last_idx)
            cur_fn = end_fn
    return tuning_cents, phase, out


def decode_voice(tuning_cents, phase_log2, toks):
    """Absolute-anchored decode. NO float is threaded across spans. The grid INDEX
    (cur_idx) is the only cross-note state, carried by exact-integer accumulation.

    NOTE  -> jump to the exact grid Fn at the accumulated absolute index.
    RAW   -> exact Fn (out-of-grid / extreme transient); re-seeds the chain.
    REST  -> exact Fn (note-off / power-zero); index unchanged.
    MOD   -> ride raw deltas from the span's CARRIED exact base_fn (byte-exact on
             modulation, vibrato/arp/glide alike; no dependence on the previous
             span's float, no rescale/retarget distortion).
    """
    out = []
    cur_idx = None
    for t in toks:
        if t[0] == "REST":
            out.append(float(t[1]))
        elif t[0] == "RAW":
            cur_idx = None  # raw note re-seeds the chain (matches encode)
            out.append(float(t[1]))
        elif t[0] == "NOTE":
            interval = t[1]
            micro = t[2] if len(t) > 2 else 0.0
            cur_idx = interval if cur_idx is None else cur_idx + interval
            out.append(_grid_fn(phase_log2, cur_idx, micro))
        else:  # MOD (vibrato / arp / glide)
            e, n, base_fn = t[1], t[2], t[3]
            p = len(e)
            acc = float(base_fn)
            for k in range(n):
                acc += e[k % p]
                out.append(acc)
            cur_idx = _glide_end_idx(phase_log2, e, n, base_fn, cur_idx)
    return np.array(out, dtype=np.float64)
