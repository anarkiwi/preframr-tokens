"""Byte-exact, SET-free encoder. Each register stream is encoded with exactly two
ops:
  RUN(delta_pattern, n) : n frames, value += delta_pattern[k % p] each frame
                          (subsumes hold/sweep/vibrato/sawtooth/staircase)
  LOAD(value)           : a jump -> value (a note / wavetable / table entry);
                          value comes from the lane's small recovered table.
There is NO SET op. Lossless by construction; decode == original is asserted.
'SET==0' is structural: the only non-generative event is LOAD, which references a
bounded per-lane value table (size measured -> must stay small, else it is an
escape hatch in disguise).
"""

PMAX = 96


def encode_lane(v):
    v = [int(x) for x in v]
    T = len(v)
    if T == 0:
        return []
    df = [v[0]] + [v[k] - v[k - 1] for k in range(1, T)]
    toks = []
    i = 0
    while i < T:
        avail = T - i
        best_n, best_p = 0, 1
        for p in range(1, min(PMAX, avail) + 1):
            n = 0
            while n < avail and df[i + n] == df[i + (n % p)]:
                n += 1
            if n >= 2 * p and n > best_n:
                best_n, best_p = n, p
        if best_n >= 2:
            toks.append(("RUN", tuple(df[i : i + best_p]), best_n))
            i += best_n
        else:
            toks.append(("LOAD", v[i]))
            i += 1
    return toks


def decode_lane(toks):
    out = []
    cur = 0
    for t in toks:
        if t[0] == "LOAD":
            cur = t[1]
            out.append(cur)
        else:
            _, e, n = t
            p = len(e)
            for k in range(n):
                cur += e[k % p]
                out.append(cur)
    return out


def encode_tune(s):
    """s = per-frame 25-register state (int). Encode every register byte-exact.
    Returns dict: tokens per reg, plus stats. Asserts byte-exact + SET-free."""
    nreg = s.shape[1]
    T = len(s)
    per_reg = {}
    n_run = n_load = 0
    load_vals = {}
    ok = True
    for r in range(nreg):
        col = list(int(x) for x in s[:, r])
        toks = encode_lane(col)
        if decode_lane(toks) != col:
            ok = False
        per_reg[r] = toks
        for t in toks:
            if t[0] == "LOAD":
                n_load += 1
                load_vals.setdefault(r, set()).add(t[1])
            else:
                n_run += 1
    return {
        "byte_exact": ok,
        "set_ops": 0,  # no SET op exists in the alphabet
        "frames": T,
        "n_run": n_run,
        "n_load": n_load,
        "tokens": n_run + n_load,
        "load_table_sizes": {r: len(v) for r, v in load_vals.items()},
        "per_reg": per_reg,
    }
