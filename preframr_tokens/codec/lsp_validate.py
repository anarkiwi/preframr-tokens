"""Fidelity gate: libsidplayfp SID-write trace vs corpus register dump.

Reconstructs the full 25-register SID state per absolute PAL frame for both the
instrumented-libsidplayfp trace and the corpus (VICE) register dump, auto-aligns
the two by the startup-frame lag, and reports byte-exact frame match.

PW-high registers (3, 10, 17) are masked to their 4 SID-significant bits: the
pulse width is 12-bit so bits 4-7 of the PW-high byte are don't-care and ignored
by the chip. libsidplayfp logs the raw CPU-written byte; the VICE corpus dump
logs the SID-significant value. Masking removes this pure logging-convention
difference; it is not an emulation discrepancy. Use --raw to see it.
"""

import sys
import numpy as np
import pandas as pd

SIDWR_DT = np.dtype([("cycle", "<i8"), ("addr", "<u2"), ("reg", "u1"), ("val", "u1")])
NREG = 25
CPF = 19656.0  # PAL cycles/frame (985248 Hz / ~50.12)
NTSC_CPF = 17095.0  # NTSC cycles/frame (1022727 Hz / ~59.83)
PW_HI = (3, 10, 17)


def cpf_from_meta(prefix):
    """Frame-cycle count for the tune's detected clock (NTSC vs PAL).

    The emulator already selects the tune's native clock (model not forced);
    the dump was made at that same clock, so both sides must be framed with it.
    """
    try:
        with open(prefix + ".meta.txt") as fh:
            for ln in fh:
                if ln.startswith("speed=") and "NTSC" in ln:
                    return NTSC_CPF
    except OSError:
        pass
    return CPF


def emu_writes(prefix, base=0xD400):
    a = np.fromfile(prefix + ".sidwr.bin", dtype=SIDWR_DT)
    a = a[(a["addr"] >= base) & (a["addr"] < base + 0x20)]
    return a["cycle"].astype(np.int64), a["reg"].astype(int), a["val"].astype(int)


def dump_writes(dump_path):
    df = pd.read_parquet(dump_path)
    df = df[df["chipno"] == 0].sort_values("clock")
    return (
        df["clock"].to_numpy(np.int64),
        df["reg"].to_numpy(int),
        df["val"].to_numpy(int),
    )


def first_play_cycle(cyc, cpf=CPF):
    if len(cyc) == 0:
        return 0
    bnd = np.nonzero(np.diff(cyc) > 2000)[0]
    starts = [int(cyc[0])] + [int(cyc[b + 1]) for b in bnd]
    for i in range(len(starts) - 3):
        d = [starts[i + k + 1] - starts[i + k] for k in range(3)]
        if all(abs(x - cpf) < 400 for x in d):
            return starts[i]
    return starts[0]


def state_seq(cyc, reg, val, t0, mask, cpf=CPF):
    cur = [0] * 32
    out = {}
    for c, r, v in zip(cyc, reg, val):
        v = int(v)
        r = int(r)
        if mask and r in PW_HI:
            v &= 0x0F
        if r < 32:
            cur[r] = v
        fi = int(round((c - t0) / cpf))
        if fi >= 0:
            out[fi] = tuple(cur[:NREG])
    if not out:
        return []
    nf = max(out) + 1
    seq, last = [], tuple([0] * NREG)
    for f in range(nf):
        if f in out:
            last = out[f]
        seq.append(last)
    return seq


def changed_frames(cyc, reg, val, t0):
    s = set()
    for c in cyc:
        fi = int(round((c - t0) / CPF))
        if fi >= 0:
            s.add(fi)
    return s


def best_lag(e_seq, d_seq, ecf=None, dcf=None, span=40):
    """Find the emu->dump frame lag that maximises full-state agreement.

    Lag may be negative (the corpus host's t0 detection can land a few frames
    earlier or later than the emu's), so scan both directions.
    """
    best = (-1, 0)
    for lag in range(-span, span + 1):
        n = ok = 0
        for i in range(max(0, lag), min(len(e_seq), len(d_seq) + lag)):
            j = i - lag
            if 0 <= j < len(d_seq):
                n += 1
                if e_seq[i] == d_seq[j]:
                    ok += 1
        if n > 20 and ok > best[0]:
            best = (ok, lag)
    return best[1]


def compare(prefix, dump_path, mask=True, verbose=True):
    cpf = cpf_from_meta(prefix)
    ec, er, ev = emu_writes(prefix)
    dc, dr, dv = dump_writes(dump_path)
    et0, dt0 = first_play_cycle(ec, cpf), first_play_cycle(dc, cpf)
    e = state_seq(ec, er, ev, et0, mask, cpf)
    d = state_seq(dc, dr, dv, dt0, mask, cpf)
    lag = best_lag(e, d)
    n = ok = 0
    mm = []
    for i in range(max(0, lag), min(len(e), len(d) + lag)):
        j = i - lag
        if 0 <= j < len(d):
            n += 1
            if e[i] == d[j]:
                ok += 1
            elif len(mm) < 6:
                mm.append(
                    (
                        i,
                        j,
                        [
                            (r, d[j][r], e[i][r])
                            for r in range(NREG)
                            if e[i][r] != d[j][r]
                        ],
                    )
                )
    pct = 100 * ok / max(1, n)
    if verbose:
        print(
            f"{prefix}: {ok}/{n} ({pct:.1f}%) frames byte-exact  [lag={lag} mask_pwhi={mask}]"
        )
        for x in mm[:4]:
            print(f"   mismatch emu_f{x[0]} dump_f{x[1]}: {x[2]}")
    return ok, n, pct, lag


if __name__ == "__main__":
    mask = "--raw" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare(args[0], args[1], mask=mask)
