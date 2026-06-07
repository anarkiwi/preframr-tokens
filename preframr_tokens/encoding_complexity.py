"""Encoding-explosion guard: per freq voice the encoded vocabulary (distinct note-table shapes
+ residual payloads + TRI/SWEEP gesture params) must not exceed the input alphabet (distinct
semitone notes + binned off-grid modulation levels, so one bounded LFO costs O(depth) not
O(distinct sampled freqs)); when it does, the encoder ADDED complexity rather than abstracting
(e.g. one vibrato carried as many per-instance residuals) and check_no_explosion raises.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .stfconstants import (
    GEN_TABLE_REF_OP,
    GEN_TABLE_REF_SUBREG_ID,
    GEN_TABLE_REF_SUBREG_RESID_LO,
    GEN_TABLE_REF_SUBREG_RESID_HI,
    GEN_TABLE_REF_SUBREG_LEN_LO,
    GEN_TRI_OP,
    SWEEP_OP,
)

_ANCHOR = 4455.0
_FREQ_VOICE_REGS = {0: (0, 1, 4), 1: (7, 8, 11), 2: (14, 15, 18)}


class EncodingExplosion(ValueError):
    """The generator encoding minted more distinct freq behaviours than the input held."""


def input_freq_complexity(reg_df) -> dict[int, int]:
    """Per voice: |distinct notes| + |distinct off-grid 5c modulation levels| from the raw
    register log (cols clock/irq/chipno/reg/val). Reconstructs per-frame settled freq+gate.
    """
    d = reg_df[reg_df["chipno"] == 0]
    frames = list(dict.fromkeys(d["irq"].tolist()))
    fpos = {f: i for i, f in enumerate(frames)}
    state = np.zeros((len(frames), 25), dtype=np.int64)
    cur = np.zeros(25, dtype=np.int64)
    last = 0
    for f, reg, val in zip(d["irq"].tolist(), d["reg"].tolist(), d["val"].tolist()):
        i = fpos[f]
        if i != last:
            state[last + 1 : i + 1] = cur
            last = i
        if 0 <= int(reg) < 25:
            cur[int(reg)] = int(val)
    state[last:] = cur
    out: dict[int, int] = {}
    for v, (lo, hi, c) in _FREQ_VOICE_REGS.items():
        fw = (state[:, hi] << 8) | state[:, lo]
        on = fw[(fw > 0) & ((state[:, c] & 1) == 1)].astype(np.float64)
        if not len(on):
            out[v] = 0
            continue
        cents = 1200.0 * np.log2(on / _ANCHOR)
        notes = np.round(cents / 100.0)
        offgrid = cents - notes * 100.0
        n_notes = len(np.unique(notes))
        n_levels = len({int(round(o / 5.0)) for o in offgrid if abs(o) >= 8})
        out[v] = n_notes + n_levels
    return out


def encoded_freq_complexity(token_df) -> dict[int, int]:
    """Per voice (reg//7): distinct note-table shapes + residual payloads + gesture params."""
    voc: dict[int, set] = defaultdict(set)
    ops = token_df["op"].tolist()
    regs = token_df["reg"].tolist()
    subs = token_df["subreg"].tolist()
    vals = token_df["val"].tolist()
    voice = cb = lo = None
    res: list[int] = []
    for op, reg, sb, v in zip(ops, regs, subs, vals):
        op, reg, sb, v = int(op), int(reg), int(sb), int(v)
        gv = reg // 7 if reg in (0, 7, 14) else None
        if op == GEN_TABLE_REF_OP:
            if sb == GEN_TABLE_REF_SUBREG_ID:
                voice, cb, res, lo = gv, v, [], None
            elif sb == GEN_TABLE_REF_SUBREG_RESID_LO:
                lo = v
            elif sb == GEN_TABLE_REF_SUBREG_RESID_HI and lo is not None:
                res.append(((v & 0xFF) << 8) | (lo & 0xFF))
                lo = None
            elif sb == GEN_TABLE_REF_SUBREG_LEN_LO and voice is not None:
                voc[voice].add(("shape", cb))
                if res:
                    voc[voice].add(("resid", tuple(res)))
                voice = cb = None
        elif op in (GEN_TRI_OP, SWEEP_OP) and gv is not None and sb == 0:
            voc[gv].add((op, v))
    return {v: len(s) for v, s in voc.items()}


def check_no_explosion(
    reg_df, token_df, *, tune: str = "", raise_on_explosion: bool = True
):
    """Compare encoded vs input freq complexity per voice; raise (or return report) on explosion."""
    inp = input_freq_complexity(reg_df)
    enc = encoded_freq_complexity(token_df)
    bad = {
        v: (enc[v], inp.get(v, 0)) for v in enc if enc[v] > inp.get(v, 0) and enc[v] > 1
    }
    if bad and raise_on_explosion:
        detail = ", ".join(
            f"voice {v}: encoded {e} > input {i}" for v, (e, i) in bad.items()
        )
        raise EncodingExplosion(
            f"encoding explosion in {tune or '<tune>'} ({detail}) -- the generator pass minted "
            "more distinct freq behaviours than the input contained (likely modulation carried "
            "as per-instance residual rather than a shared gesture)."
        )
    return {"input": inp, "encoded": enc, "explosions": bad}
