"""Per-voice universal-pitch ENCODER / DECODER prototype.

Builds on the completed investigation (pitch_universal.py / pitch_universal_corpus.py):
a universal 12-TET pitch ALPHABET is viable but the tuning phase must be PER-VOICE,
not per-tune, to preserve chorus DETUNING (e.g. Cauldron II detunes voice 2 vs
voice 1 by ~+12c for a chorus). A single per-tune offset would flatten that.

Design (per voice, independently):
  - Fit the voice's base-pitch set to the universal 12-TET grid and recover ONE
    continuous per-voice tuning scalar (the log-phase offset, in cents). Reuse
    pitch_universal.best_offset (robust weighted-median phase fit) PER VOICE.
  - Re-express NOTE as a canonical SEMITONE INTERVAL (small shared signed int)
    instead of a raw freq delta: the difference between consecutive snapped grid
    note indices. The note INDEX/interval alphabet is universal across all
    tunes+voices (the prize).
  - Keep MOD modulation as relative Fn deltas on top of the snapped base
    (vibrato/arp/glide ride along, proven survivable).
  - GLIDE FIX: a long portamento (a MOD with a constant-sign delta) that ENDS on
    a held note target is followed by a NOTE onset (the encoder already finds the
    endpoint as a fresh note target when the voice holds there). Both the START
    base (NOTE before the glide) and the END base (NOTE after) are snapped to the
    grid, so the glide connects two grid notes and the endpoint does not drift by
    the base-snap distance. We re-target the glide's final Fn to the snapped
    endpoint so it ARRIVES exactly, distributing the deltas proportionally.

Stream (per voice): [TUNE(cents)] then a sequence of
  NOTE(semitone_interval)       -- onset, freq jumps to a snapped grid note
  MOD(deltas, n)                 -- relative Fn deltas for n frames (vibrato/arp)
  GLIDE(deltas, n, end_interval) -- a MOD whose endpoint is a snapped grid note;
                                    decode re-scales so it lands on that note
The TUNE scalar is emitted once per voice (define-on-first-note, backward
thereafter): small, ~1 value/voice, inline-streaming-compatible.

Decode reconstructs Fn from {note index/interval stream + per-voice TUNE scalar +
MOD/GLIDE deltas}. LOSSY vs the original dump on absolute base pitch (bounded by
the snap error) but DETERMINISTIC and faithful on modulation + per-voice detune.
"""

import numpy as np

from preframr_tokens.codec import freq_relative as FR
from preframr_tokens.codec import pitch_universal as P

SEMI = P.SEMI  # one semitone in log2(freq)
CENTS_PER_SEMI = 100.0

# A MOD span is treated as a GLIDE (portamento) when its per-frame deltas are
# (near) monotonic and it spans a large pitch distance -- i.e. it walks from one
# note to another rather than oscillating around one (vibrato/arp).
GLIDE_MIN_CENTS = 60.0  # > a vibrato/arp wobble; a real note->note move

# Per-note micro-detune (chorus) quantum. The per-voice tuning scalar captures a
# voice's STATIC tuning, but some tunes (Cauldron II Remix) apply a TIME-VARYING
# chorus detune WITHIN a voice (notes alternate 0c / +-13c). A single per-voice
# scalar cannot represent that. We attach a small quantized signed micro-detune
# (cents, multiples of DETUNE_Q) to each NOTE = its snap residual vs the voice
# grid. 0 for the un-detuned majority; the universal INTERVAL alphabet is
# unaffected (the grid index is still the snap-rounded integer).
DETUNE_Q = 1.0  # cents quantum for the per-note micro-detune
DETUNE_MAX = 50.0  # clamp (beyond this it is a real grid note)


def _log2(fn):
    return np.log2(float(fn))


def voice_base_pitches(f, min_hold=3):
    """Per-voice held base-pitch set {Fn: hold_frames}. Reuses the investigation's
    held-note extractor (base pitch = freq held >= min_hold frames)."""
    return P.held_base_pitches(f, min_hold)


def fit_voice_tuning(f, min_hold=3):
    """Recover the voice's continuous tuning scalar (cents) via the robust
    weighted-median grid-phase fit, PER VOICE. Returns (tuning_cents, n_base,
    snap_errs_cents, weights, phase_log2). tuning_cents in [0,100)."""
    held = voice_base_pitches(f, min_hold)
    if len(held) < 2:
        return None
    fn = np.array(sorted(held), dtype=np.float64)
    wt = np.array([held[int(x)] for x in fn], dtype=np.float64)
    logf = P.log2fn(fn)
    phase, errs = P.best_offset(logf, robust=True, w=wt)
    return float(phase / SEMI * CENTS_PER_SEMI), len(held), errs, wt, float(phase)


def snap_index(fn, phase_log2):
    """Universal semitone index of an absolute Fn under a voice phase (log2)."""
    return int(np.round((_log2(fn) - phase_log2) / SEMI))


def _is_glide(deltas, n, base_fn):
    """A MOD span is a GLIDE iff its deltas are (near-)monotone in one direction
    and it covers >= GLIDE_MIN_CENTS end-to-end (a note->note portamento)."""
    if base_fn <= 0:
        return False
    cur = float(base_fn)
    p = len(deltas)
    pos = neg = 0
    end = cur
    for k in range(n):
        d = deltas[k % p]
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
        end += d
    if end <= 0:
        return False
    span_c = abs(1200.0 * np.log2(end / base_fn))
    monotone = (pos == 0) or (neg == 0) or (min(pos, neg) <= 0.1 * max(pos, neg))
    return span_c >= GLIDE_MIN_CENTS and monotone


def _quant_detune(cents):
    c = max(-DETUNE_MAX, min(DETUNE_MAX, cents))
    return float(np.round(c / DETUNE_Q) * DETUNE_Q)


def encode_voice(f, min_hold=3, detune=True):
    """Encode one voice freq series -> (tuning_cents, phase_log2, token stream).

    Tokens:
      ("NOTE", interval_semitones, micro_detune_cents)
      ("MOD",  deltas_tuple, n)
      ("GLIDE", deltas_tuple, n, end_interval_semitones, micro_detune_cents)
    interval is relative to the previous note's snapped grid index; the first
    NOTE's "interval" is its absolute grid index (seed). micro_detune_cents is the
    quantized snap residual (chorus); 0 for the un-detuned majority. The universal
    INTERVAL alphabet is built from grid indices only, so detune does not enlarge
    it. Set detune=False to disable (then the chorus collapses to a per-voice
    scalar -- LOSSY on time-varying chorus, kept for ablation).
    GLIDE end_interval is relative to the glide's START snapped index.
    """
    f = [int(x) for x in f]
    fit = fit_voice_tuning(f, min_hold)
    if fit is None:
        return None
    tuning_cents, _, _, _, phase = fit
    toks = FR.encode_freq(f)

    out = []
    cur_fn = 0  # running absolute Fn (raw stream)
    last_idx = None  # last snapped grid index
    for t in toks:
        if t[0] == "NOTE":
            cur_fn += t[1]
            if cur_fn <= 0:
                # sub-audible / power-on zero; keep as a raw note to stay exact-ish
                out.append(("NOTE", 0, 0.0))
                continue
            exact = (_log2(cur_fn) - phase) / SEMI  # fractional grid index
            idx = int(np.round(exact))
            micro = _quant_detune((exact - idx) * CENTS_PER_SEMI) if detune else 0.0
            interval = idx if last_idx is None else idx - last_idx
            out.append(("NOTE", interval, micro))
            last_idx = idx
        else:
            _, e, n = t
            base_fn = cur_fn
            # advance raw running value through the MOD
            p = len(e)
            end_fn = base_fn
            for k in range(n):
                end_fn += e[k % p]
            if base_fn > 0 and _is_glide(e, n, base_fn):
                start_idx = snap_index(base_fn, phase)
                if end_fn > 0:
                    exact_e = (_log2(end_fn) - phase) / SEMI
                    end_idx = int(np.round(exact_e))
                    micro = (
                        _quant_detune((exact_e - end_idx) * CENTS_PER_SEMI)
                        if detune
                        else 0.0
                    )
                else:
                    end_idx, micro = start_idx, 0.0
                out.append(("GLIDE", tuple(e), n, end_idx - start_idx, micro))
                last_idx = end_idx
            else:
                out.append(("MOD", tuple(e), n))
                # MOD does not change the note identity; last_idx unchanged
            cur_fn = end_fn
    return tuning_cents, phase, out


def decode_voice(tuning_cents, phase_log2, toks):
    """Reconstruct the per-frame Fn series from the per-voice stream.

    NOTE -> jump to the snapped grid Fn at the new index.
    MOD  -> replay relative deltas from the current Fn (vibrato/arp).
    GLIDE-> replay deltas but RE-SCALE the per-frame increments so the span lands
            exactly on the snapped END grid Fn (both endpoints on the grid; no
            base-snap drift)."""
    out = []
    cur_fn = 0.0
    cur_idx = None
    for t in toks:
        if t[0] == "NOTE":
            interval = t[1]
            micro = t[2] if len(t) > 2 else 0.0
            if cur_idx is None:
                cur_idx = interval
            else:
                cur_idx += interval
            cur_fn = 2.0 ** (phase_log2 + (cur_idx + micro / CENTS_PER_SEMI) * SEMI)
            out.append(cur_fn)
        elif t[0] == "MOD":
            _, e, n = t
            p = len(e)
            for k in range(n):
                cur_fn += e[k % p]
                out.append(cur_fn)
        else:  # GLIDE
            e, n, end_interval = t[1], t[2], t[3]
            micro = t[4] if len(t) > 4 else 0.0
            p = len(e)
            start_idx = cur_idx if cur_idx is not None else 0
            end_idx = start_idx + end_interval
            start_fn = cur_fn
            target_fn = 2.0 ** (phase_log2 + (end_idx + micro / CENTS_PER_SEMI) * SEMI)
            # nominal end if we replayed deltas raw
            nom_end = start_fn
            for k in range(n):
                nom_end += e[k % p]
            raw_span = nom_end - start_fn
            tgt_span = target_fn - start_fn
            # scale per-frame deltas so the cumulative arrives at target_fn.
            scale = (tgt_span / raw_span) if abs(raw_span) > 1e-9 else 1.0
            acc = start_fn
            for k in range(n):
                acc += e[k % p] * scale
                out.append(acc)
            cur_fn = target_fn
            cur_idx = end_idx
    return np.array(out, dtype=np.float64)
