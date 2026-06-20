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


BURST_GAP = 2000  # cycles; a write-gap above this starts a new play-call burst


def _burst_starts(cyc, gap=BURST_GAP):
    """Cycle of the first write in each play-call burst.

    A burst boundary is an inter-write gap above ``gap`` cycles. One play-call
    = one burst, so the burst-start cadence is the tune's play period.
    """
    if len(cyc) == 0:
        return np.empty(0, dtype=np.int64)
    bnd = np.nonzero(np.diff(cyc) > gap)[0]
    return np.concatenate(([cyc[0]], cyc[bnd + 1])).astype(np.int64)


def detect_play_period(cyc, cpf_ref=CPF, gap=BURST_GAP):
    """Detect the tune's true play period (cycles between play-calls).

    Driver-agnostic. The play routine fires at a fixed IRQ; each firing emits a
    burst of register writes, so the dominant inter-burst interval *is* the play
    period. Single-speed tunes call play once per raster frame (period == CPF);
    a multispeed tune calls it N times per frame, so its period is a fraction of
    CPF (Galway ~8547 ≈ CPF/2.3; Sanxion ~5900 ≈ CPF/3.3 -- note the ratio need
    not be integer: Galway's IRQ period does not divide the PAL frame evenly).

    Returns ``cpf_ref`` unchanged for single-speed tunes (dominant interval
    within 10% of the raster frame, or too few bursts to tell), so existing
    byte-exact framing is never perturbed. Otherwise returns the detected
    sub-frame period, at which every play-call boundary lands on its own frame
    and no register-value change is dropped.
    """
    starts = _burst_starts(cyc, gap)
    if len(starts) < 4:
        return cpf_ref
    period = float(np.median(np.diff(starts)))
    if period <= 0 or period >= 0.9 * cpf_ref:
        # within ~10% of one raster frame (or longer): single-speed.
        return cpf_ref
    return period


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


def _raw_change_count(reg, val):
    """Register-value changes on the true bus (reg < NREG), in write order."""
    cur = [None] * 32
    n = 0
    for r, v in zip(reg, val):
        r = int(r)
        if r < NREG and cur[r] != int(v):
            n += 1
            cur[r] = int(v)
    return n


def _framed_change_count(cyc, reg, val, cpf):
    """Register-value changes surviving in the per-frame-sampled state."""
    t0 = first_play_cycle(cyc, cpf)
    cur = [0] * 32
    out = {}
    for c, r, v in zip(cyc, reg, val):
        r = int(r)
        if r < 32:
            cur[r] = int(v)
        fi = int(round((c - t0) / cpf))
        if fi >= 0:
            out[fi] = tuple(cur[:NREG])
    if not out:
        return 0
    nf = max(out) + 1
    prev = tuple([0] * NREG)
    last = prev
    n = 0
    for f in range(nf):
        if f in out:
            last = out[f]
        n += sum(1 for a, b in zip(last, prev) if a != b)
        prev = last
    return n


def framing_change_loss(cyc, reg, val, cpf):
    """Fraction of true-bus register-value changes dropped by framing at ``cpf``.

    The losslessness metric for the framing substrate: 0.0 means every register
    change on the raw bus survives the per-frame sampling (lossless); a positive
    value is the fraction silently dropped because >1 play-call collapsed into a
    single frame. Single-CPF framing of a multispeed tune drops a large fraction;
    framing at the detected play period drops ~0.
    """
    raw = _raw_change_count(reg, val)
    if raw == 0:
        return 0.0
    framed = _framed_change_count(cyc, reg, val, cpf)
    return max(0.0, 1.0 - framed / raw)


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
