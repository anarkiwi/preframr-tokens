"""Frame-clock helpers for the register dump.

Reconstructs the full 25-register SID state per absolute PAL frame from a corpus
register dump and locates the tune's first play cycle. The frame clock (CPF /
NTSC_CPF, auto-detected from the .meta.txt) is shared with the BACC renderer so
both sides bin writes onto the same grid.

PW-high registers (3, 10, 17) are masked to their 4 SID-significant bits: the
pulse width is 12-bit so bits 4-7 of the PW-high byte are don't-care and ignored
by the chip; masking removes that pure logging-convention difference.
"""

import numpy as np

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
