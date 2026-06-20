"""Lossless lane-grammar miner: decompose every SID lane (per-frame value
sequence) of a byte-exact tune into primitive ops {HOLD, ACCUM, PERIOD, SET},
then pool the op-set corpus-wide across ALL lanes.

Thesis (sid_player_decompiler.md): the grammar = a tiny fixed op-set that
explains all lanes of all tunes; the per-tune params (note SETs, table shapes,
deltas) are the data/music. PERIOD = vibrato/arp/PWM/wavetable; ACCUM = sweep/
portamento; HOLD = sustain; SET = note onset or unmodelled mechanism.

Substrate = the VICE dump (byte-exact, validated). Lossless: ops reconstruct
each lane exactly (asserted).
"""

import numpy as np
import pandas as pd

from preframr_tokens.codec import lsp_validate as V

NREG = 25
PMAX = 64

# lane class -> list of (name) ; multi-byte lanes combined
LANE_CLASS = {
    "freq": "melody/pitch",
    "pw": "pulse-width",
    "ctrl": "gate/waveform",
    "ad": "attack/decay",
    "sr": "sustain/release",
    "fc": "filter-cutoff",
    "res": "filter-res/route",
    "vol": "filter-mode/vol",
}


def per_frame_state(dump, cpf, maxframes=1000):
    df = pd.read_parquet(dump, columns=["clock", "reg", "val", "chipno"])
    df = df[df["chipno"] == 0].sort_values("clock")
    cyc = df["clock"].to_numpy(np.int64)
    reg = df["reg"].to_numpy(int)
    val = df["val"].to_numpy(int)
    if len(cyc) == 0:
        return None
    t0 = V.first_play_cycle(cyc, cpf)
    cur = [0] * 32
    out = {}
    for c, r, vv in zip(cyc, reg, val):
        if r < 32:
            cur[r] = int(vv)
        fi = int(round((c - t0) / cpf))
        if 0 <= fi < maxframes:
            out[fi] = cur[:NREG]
    if not out:
        return None
    nf = max(out) + 1
    seq = np.zeros((nf, NREG), dtype=np.int64)
    last = [0] * NREG
    for f in range(nf):
        if f in out:
            last = out[f]
        seq[f] = last
    return seq


def lanes_from_state(s):
    L = {}
    for v in range(3):
        b = 7 * v
        L[(v, "freq")] = s[:, b + 0] + 256 * s[:, b + 1]
        L[(v, "pw")] = s[:, b + 2] + 256 * (s[:, b + 3] & 0xF)
        L[(v, "ctrl")] = s[:, b + 4]
        L[(v, "ad")] = s[:, b + 5]
        L[(v, "sr")] = s[:, b + 6]
    L[(3, "fc")] = (s[:, 22] << 3) | (s[:, 21] & 7)
    L[(3, "res")] = s[:, 23]
    L[(3, "vol")] = s[:, 24]
    return L


def parse_lane(v):
    """Greedy longest-cover lossless parse into primitive ops.
    op = (type, length, params).  Reconstruct == v (verified by caller)."""
    v = np.asarray(v, dtype=np.int64)
    T = len(v)
    i = 0
    ops = []
    while i < T:
        best = ("set", 1, (int(v[i]),))
        # HOLD
        j = i
        while j + 1 < T and v[j + 1] == v[i]:
            j += 1
        if j - i + 1 > best[1]:
            best = ("hold", j - i + 1, (int(v[i]),))
        # ACCUM (constant non-zero delta, >=3 frames)
        if i + 1 < T:
            d = int(v[i + 1]) - int(v[i])
            if d != 0:
                j = i
                while j + 1 < T and int(v[j + 1]) - int(v[j]) == d:
                    j += 1
                if j - i + 1 >= 3 and j - i + 1 > best[1]:
                    best = ("accum", j - i + 1, (int(v[i]), d))
        # PERIOD (smallest period repeating >=2 full cycles)
        pmax = min(PMAX, (T - i) // 2)
        for p in range(2, pmax + 1):
            k = i + p
            while k < T and v[k] == v[i + ((k - i) % p)]:
                k += 1
            length = k - i
            if length >= 2 * p and length > best[1]:
                best = ("period", length, (p, tuple(int(x) for x in v[i : i + p])))
                break  # smallest period that qualifies; greedy
        ops.append(best)
        i += best[1]
    return ops


def reconstruct(ops):
    out = []
    for typ, length, prm in ops:
        if typ == "hold":
            out += [prm[0]] * length
        elif typ == "set":
            out += [prm[0]]
        elif typ == "accum":
            v0, d = prm
            out += [v0 + d * k for k in range(length)]
        elif typ == "period":
            p, cyc = prm
            out += [cyc[k % p] for k in range(length)]
    return np.asarray(out, dtype=np.int64)


def mine_tune(dump, cpf, maxframes=1000, verify=True):
    s = per_frame_state(dump, cpf, maxframes)
    if s is None or len(s) < 4:
        return None
    lanes = lanes_from_state(s)
    res = []
    for (vi, cls), seq in lanes.items():
        ops = parse_lane(seq)
        if verify:
            rec = reconstruct(ops)
            if len(rec) != len(seq) or not np.array_equal(rec, seq):
                return {"error": f"lossy lane v{vi}_{cls}"}
        for typ, length, prm in ops:
            sig = None
            if typ == "period":
                sig = ("period", prm[0])  # period length = shape class
            elif typ == "accum":
                sig = ("accum", prm[1])  # delta
            res.append((cls, typ, length, sig))
    return {"frames": len(s), "ops": res}
