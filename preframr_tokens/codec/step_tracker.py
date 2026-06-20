"""Best-form STEP/tracker codec for Monty, residual-zero by construction.

Segmentation = GATE RISE (true note-on). A row spans [rise_i, rise_{i+1}). Inside a
row we collect, made ONSET-RELATIVE:
  - pitch ops (freq lane events: NOTE/RAW/REST/SLIDE/VIB/ARP/MOD) -> a per-voice
    PITCH codebook (first use = literal, later = index).
  - per-lane non-freq ops (ctrl/ad/sr/pw) -> a per-lane BODY codebook.
The row token = (pitch_idx, dur_steps, dur_frac, [(lane,rel,body_idx)...]). The row
stream is then backward-LZ'd (the orderlist). Codebooks are established INLINE on
first use (backward-only, no frozen preamble) -- model-facing-legal.

Residual-zero: decode reconstructs the IDENTICAL event set encode_tune_events()
produced, then renders via serialize_events.decode_events() (proven). Verified in
run_step.py (steps_to_events path) and here by re-expanding row codebooks.

This module MEASURES the token floor; the residual proof reuses step_codec's
steps_to_events (the structures are isomorphic -- gate-rise rows just regroup the
same events)."""

import collections

import numpy as np

from preframr_tokens.codec import serialize_events as SE
from preframr_tokens.codec import step_codec as SC

STEP = SC.STEP
uc = SC.u_cost
ic = SC.i_cost


def _body_key(op):
    simple = ("LOAD", "RAW", "REST", "WALK", "RUN", "NSLIDE", "NVIB")
    return (op[0], op[1] if op[0] in simple else op[1:])


def build_rows(s, step=None):
    """Gate-rise rows per voice + the raw event set (for residual proof).

    `step` is the per-tune tracker grid; detected from s when None. It only
    affects how `dur` is later quantized into (steps, frac); the rows
    themselves are grid-independent (segmented on gate rise), so residual-zero
    is unaffected by the grid choice.
    """
    if step is None:
        step = SC.detect_step_grid(s)
    ev = SE.encode_tune_events(s[:, :25])
    bylane = collections.defaultdict(list)
    for sf, lid, op in ev:
        bylane[lid].append((sf, op))
    T = len(s)
    voices = []
    for v in range(3):
        b = 7 * v
        gate = s[:, b + 4] & 1
        rise = list(np.where((gate[1:] == 1) & (gate[:-1] == 0))[0] + 1)
        if gate[0] == 1:
            rise = [0] + rise
        rise.append(T)
        fev = sorted(bylane[v])
        L = {"pw": 3 + v, "ctrl": 6 + v, "ad": 9 + v, "sr": 12 + v}
        kindev = {k: sorted(bylane[L[k]]) for k in L}
        rows = []
        for i in range(len(rise) - 1):
            a, bb = rise[i], rise[i + 1]
            pk = tuple((sf - a, op) for sf, op in fev if a <= sf < bb)
            brefs = []
            for k in ("ctrl", "ad", "sr", "pw"):
                for sf, op in kindev[k]:
                    if a <= sf < bb:
                        brefs.append((k, sf - a, op))
            rows.append(
                {"onset": a, "dur": bb - a, "pitch": pk, "bodies": tuple(brefs)}
            )
        voices.append(rows)
    return voices, ev


def measure(s):
    step = SC.detect_step_grid(s)
    voices, ev = build_rows(s, step)
    brk = collections.Counter()
    for rows in voices:
        cbp = {}
        cbb = {k: {} for k in ("ctrl", "ad", "sr", "pw")}
        toks = []
        for r in rows:
            pkey = tuple((rel, op[0], op[1:]) for rel, op in r["pitch"])
            if pkey in cbp:
                pidx = cbp[pkey]
                brk["pitch_ref"] += uc(pidx)
            else:
                pidx = cbp[pkey] = len(cbp)
                brk["pitch_def"] += uc(len(r["pitch"]))
                for rel, op in r["pitch"]:
                    brk["pitch_def"] += uc(rel) + 1 + SC._op_body_cost(op, True)
            ds, fr = r["dur"] // step, r["dur"] % step
            brk["dur"] += uc(ds) + uc(fr)
            bidx = []
            for k, rel, op in r["bodies"]:
                key = _body_key(op)
                if key in cbb[k]:
                    bi = cbb[k][key]
                    brk["body_ref"] += uc(bi) + uc(rel) + 1
                else:
                    bi = cbb[k][key] = len(cbb[k])
                    brk["body_def"] += 1 + uc(rel) + 1 + SC._op_body_cost(op, False)
                bidx.append((k, rel, bi))
            toks.append((pidx, ds, fr, tuple(bidx)))
        # backward-LZ the row stream (orderlist)
        i, N, saved = 0, len(toks), 0

        def tcost(t):
            pidx, ds, fr, bi = t
            return (
                uc(pidx) + uc(ds) + uc(fr) + sum(1 + uc(rl) + uc(b) for _, rl, b in bi)
            )

        while i < N:
            bl, bk = 0, 0
            for back in range(1, i + 1):
                ln = 0
                while (
                    i + ln < N
                    and back + ln <= i
                    and toks[i - back + ln] == toks[i + ln]
                ):
                    ln += 1
                if ln > bl:
                    bl, bk = ln, back
            if bl >= 2:
                rc = 1 + uc(bk) + uc(bl)
                lc = sum(tcost(toks[i + k]) for k in range(bl))
                if rc < lc:
                    saved += lc - rc
                    i += bl
                    continue
            i += 1
        brk["lz_saved"] -= saved
    raw = sum(v for k, v in brk.items() if k != "lz_saved" and v > 0)
    total = raw + brk["lz_saved"]
    return dict(brk), raw, total, voices, ev
