"""E1 -- byte-exact MDL codec round-trip (UNTRACKED, throwaway).

Proves the MDL gesture representation is a LOSSLESS COVER of real per-frame register state: every one
of the 25 SID register columns encodes to self-contained HOLD/RAMP/PERIOD tokens and decodes back
byte-exact. mdl_parse only ever claims an EXACTLY-fitting run (constant value / constant wrap-delta /
exact period) with a length-1 HOLD literal fallback, so the cover is lossless by construction -- this
validates encoder/decoder consistency, wrap handling, and the per-instance (start,len) + shape split
on real data, the foundation E1 rests on before the codebook-family serialization.

Self-contained token = ("H", v, L) | ("R", v0, step, L) | ("P", v0, cell, L). The reusable SHAPE
(dictionary key) is the ramp step / period cell; (v0, L) are per-instance content.

Run:  python mdl_codec.py            # 5 drivers + GRID_RUNNER
      python mdl_codec.py corpus 40  # + N random corpus tunes
"""

from __future__ import annotations

import glob
import random
import sys

import numpy as np
import pandas as pd

from mdl_parse import mdl_parse
from freq_residual_instrument_spike import per_frame_state
from tests.sid_fixtures import DRIVER_FIXTURES, GRID_RUNNER, ensure_driver_fixture, ensure_dumps

# Every reg 0..24 lives in exactly one channel. 2-reg channels combine lo+hi into one word, parsed in
# its natural domain (freq is wrap-16), then split back to bytes; 1-reg channels parse the byte直接.
_FREQ = [(0, 1), (7, 8), (14, 15)]  # 16-bit, wrap
_PW = [(2, 3), (9, 10), (16, 17)]  # 12-bit word, no wrap
_CUT = [(21, 22)]  # 11-bit cutoff
_SINGLE = [4, 5, 6, 11, 12, 13, 18, 19, 20, 23, 24]  # ctrl/AD/SR x3, res, vol


def _wd(d, wrap):
    d = int(d)
    return ((d + 32768) % 65536) - 32768 if wrap else d


def _difftable(s, i, N, wrap):
    """Initial forward-difference table [v0, Delta1, .., DeltaN] of s at frame i (wrap-aware)."""
    cur = [int(s[i + m]) for m in range(N + 1)]
    dt = [cur[0]]
    for _ in range(N):
        cur = [_wd(cur[m + 1] - cur[m], wrap) for m in range(len(cur) - 1)]
        dt.append(cur[0])
    return dt


def encode(series, wrap=False):
    """Self-contained token list for a value series (lossless)."""
    s = np.asarray(series, dtype=np.int64)
    toks = []
    for kind, i, j, p in mdl_parse(s, wrap):
        if kind == "H":
            toks.append(("H", int(s[i]), j - i))
        elif kind == "D":  # forward-differenced polynomial, degree p[0]
            toks.append(("D", p[0], tuple(_difftable(s, i, p[0], wrap)), j - i))
        else:  # P
            toks.append(("P", int(s[i]), tuple(int(x) for x in p), j - i))
    return toks


def decode(toks, wrap=False):
    """Inverse of encode: regenerate the value series from self-contained tokens."""
    mask = 0xFFFF if wrap else None
    out = []
    for t in toks:
        if t[0] == "H":
            out.extend([t[1]] * t[2])
        elif t[0] == "D":  # forward differencing: emit dt[0], then dt[k]+=dt[k+1]
            N, dt, L = t[1], [int(x) for x in t[2]], t[3]
            for _ in range(L):
                out.append(dt[0])
                for k in range(N):
                    dt[k] += dt[k + 1]
                if mask:
                    dt[0] &= mask
        else:  # P
            v0, cell, L = t[1], t[2], t[3]
            cur = v0
            out.append(cur)
            for k in range(1, L):
                cur = cur + cell[(k - 1) % len(cell)]
                out.append(cur & mask if mask else cur)
    return np.asarray(out, dtype=np.int64)


def roundtrip_state(state):
    """Encode every channel of a (n_frames, 25) state, decode, reassemble (n_frames, 25). Returns
    (recon, n_tokens, n_literal)."""
    n = state.shape[0]
    recon = np.zeros((n, 25), dtype=np.int64)
    ntok = nlit = 0
    for (lo, hi), wrap in [(c, True) for c in _FREQ] + [(c, False) for c in _PW + _CUT]:
        word = (state[:, hi].astype(np.int64) << 8) | state[:, lo]
        toks = encode(word, wrap)
        w = decode(toks, wrap)
        recon[:, lo] = w & 0xFF
        recon[:, hi] = (w >> 8) & 0xFF
        ntok += len(toks)
        nlit += sum(1 for t in toks if t[0] == "H" and t[2] == 1)
    for reg in _SINGLE:
        toks = encode(state[:, reg], False)
        recon[:, reg] = decode(toks, False)
        ntok += len(toks)
        nlit += sum(1 for t in toks if t[0] == "H" and t[2] == 1)
    return recon, ntok, nlit


def check(name, path):
    state = per_frame_state(pd.read_parquet(path))
    recon, ntok, nlit = roundtrip_state(state)
    ok = recon.shape == state.shape and bool((recon == state).all())
    if ok:
        raw = int((state != 0).sum())  # rough naive cost proxy: nonzero per-frame reg writes
        print(
            f"  {name:14} OK  byte-exact ({state.shape[0]} frames)  "
            f"tokens={ntok} (literal={nlit})  vs ~{raw} naive nonzero-cells"
        )
    else:
        diff = recon != state
        fi = int(np.argmax(diff.any(axis=1)))
        r = int(np.where(diff[fi])[0][0])
        print(
            f"  {name:14} FAIL frame {fi} reg {r}: {int(state[fi, r])} -> {int(recon[fi, r])}"
        )
    return ok


_CHANS = [(c, True) for c in _FREQ] + [(c, False) for c in _PW + _CUT]


def tune_tokens(state):
    """All channels of a tune as (channel_key, wrap, tokens)."""
    out = []
    for (lo, hi), wrap in _CHANS:
        word = (state[:, hi].astype(np.int64) << 8) | state[:, lo]
        out.append(((lo, hi), wrap, encode(word, wrap)))
    for reg in _SINGLE:
        out.append((reg, False, encode(state[:, reg], False)))
    return out


def shape_of(tok):
    """The reusable dictionary shape of a token, or None for HOLD (content, not a shape)."""
    if tok[0] == "D":
        return ("D", tok[1], tok[2][-1])  # (degree, N-th difference = the shape)
    if tok[0] == "P":
        return ("P", tok[2])  # period cell
    return None


def serialize_rows(channel_tokens, shape_id):
    """Codebook-family rows for one channel's tokens: HOLD -> (HOLD, v, L); RAMP/PERIOD -> a REF row
    carrying (global shape id, start v0, length L). DEF rows live once in the global preamble, not here
    (corpus-global dictionary = learned vocabulary). Rows are (op, field, val) tuples for the proof."""
    rows = []
    for t in channel_tokens:
        if t[0] == "H":
            rows.append(("HOLD", "VAL", t[1]))
            rows.append(("HOLD", "LEN", t[2]))
        else:
            rows.append(("GREF", "ID", shape_id[shape_of(t)]))
            rows.append(("GREF", "V0", t[1]))
            rows.append(("GREF", "LEN", t[3]))
    return rows


def parse_rows(rows, id_shape):
    """Inverse of serialize_rows: rows + the global id->shape table back to self-contained tokens."""
    toks, i = [], 0
    while i < len(rows):
        op = rows[i][0]
        if op == "HOLD":
            toks.append(("H", rows[i][2], rows[i + 1][2]))
            i += 2
        else:  # GREF: ID, V0, LEN
            kind, shape = id_shape[rows[i][2]]
            v0, L = rows[i + 1][2], rows[i + 2][2]
            toks.append((kind, v0, shape, L) if kind == "R" else ("P", v0, shape, L))
            i += 3
    return toks


def dict_main(n):
    from collections import Counter

    pool = sorted(glob.glob("/scratch/preframr/hvsc/**/*.dump.parquet", recursive=True))
    paths = [ensure_driver_fixture(x) for x in DRIVER_FIXTURES] + [
        p for p in random.Random(1).sample(pool, n)
    ]
    # pass 1: collect shapes corpus-wide with per-tune frequency
    shape_tunes: Counter = Counter()
    shape_uses: Counter = Counter()
    per_tune = []
    for p in paths:
        try:
            state = per_frame_state(pd.read_parquet(p))
        except Exception:  # noqa: BLE001
            continue
        toks = tune_tokens(state)
        per_tune.append((p, state, toks))
        seen = set()
        for _ch, _w, ct in toks:
            for t in ct:
                sh = shape_of(t)
                if sh is not None:
                    shape_uses[sh] += 1
                    seen.add(sh)
        for sh in seen:
            shape_tunes[sh] += 1
    # global dictionary: id per distinct shape
    shapes = sorted(shape_uses, key=lambda s: (-shape_uses[s], str(s)))
    shape_id = {sh: i for i, sh in enumerate(shapes)}
    id_shape = {i: sh for sh, i in shape_id.items()}
    ramp = [s for s in shapes if s[0] == "R"]
    per = [s for s in shapes if s[0] == "P"]
    singles = sum(1 for s in shapes if shape_tunes[s] == 1)
    shared = sum(1 for s in shapes if shape_tunes[s] >= 2)
    uses = sum(shape_uses.values())
    top = sum(shape_uses[s] for s in shapes[: max(1, len(shapes) // 20)])
    print(f"\nCorpus-global gesture dictionary over {len(per_tune)} tunes:")
    print(f"  distinct shapes: {len(shapes)}  (ramp-steps {len(ramp)}, period-cells {len(per)})")
    print(f"  shared (>=2 tunes): {shared}   singletons (1 tune): {singles}  ({100*singles/max(1,len(shapes)):.0f}%)")
    print(f"  top 5% of shapes cover {100*top/max(1,uses):.0f}% of all gesture uses")
    # byte-exact round-trip THROUGH rows + shared dict
    npass = 0
    for p, state, toks in per_tune:
        recon = np.zeros_like(state)
        for ch, wrap, ct in toks:
            rows = serialize_rows(ct, shape_id)
            back = parse_rows(rows, id_shape)
            w = decode(back, wrap)
            if isinstance(ch, tuple):
                recon[:, ch[0]] = w & 0xFF
                recon[:, ch[1]] = (w >> 8) & 0xFF
            else:
                recon[:, ch] = w
        npass += bool((recon == state).all())
    print(f"  byte-exact through rows+shared-dict: {npass}/{len(per_tune)}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "dict":
        return dict_main(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    print("E1 byte-exact MDL codec round-trip:\n")
    tunes = [(n, ensure_driver_fixture(n)) for n in DRIVER_FIXTURES]
    tunes.append(("grid_runner", ensure_dumps(GRID_RUNNER)[1]))
    if len(sys.argv) > 1 and sys.argv[1] == "corpus":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
        pool = sorted(glob.glob("/scratch/preframr/hvsc/**/*.dump.parquet", recursive=True))
        for p in random.Random(0).sample(pool, n):
            tunes.append((p.split("/")[-1][:14], p))
    npass = 0
    for name, path in tunes:
        try:
            npass += check(name, path)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:14} ERROR {type(e).__name__}: {str(e)[:60]}")
    print(f"\n{npass}/{len(tunes)} byte-exact")


if __name__ == "__main__":
    main()
