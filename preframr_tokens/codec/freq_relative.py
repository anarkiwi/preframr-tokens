"""Relative freq encoder = the note+MOD decomposition, inline & byte-exact.
Two ops on the running freq:
  NOTE(interval) : freq += interval  (a note onset; interval from a small set)
  MOD(deltas, n) : freq += deltas[k % p] each frame for n frames (vibrato/glide)
Everything is relative to the current freq (seeded by the prompt) -> inline,
continuable, transposition-invariant; no frozen table. df[0]=v[0] is the initial
NOTE from power-on 0. Lossless: decode == v (asserted).
"""

PMAX = 96


def encode_freq(v):
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
            toks.append(("MOD", tuple(df[i : i + best_p]), best_n))
            i += best_n
        else:
            toks.append(("NOTE", df[i]))
            i += 1
    return toks


def decode_freq(toks):
    out = []
    cur = 0
    for t in toks:
        if t[0] == "NOTE":
            cur += t[1]
            out.append(cur)
        else:
            _, e, n = t
            p = len(e)
            for k in range(n):
                cur += e[k % p]
                out.append(cur)
    return out
