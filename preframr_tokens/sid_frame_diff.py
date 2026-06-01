"""Frame-diff fidelity audit: per frame, per register, compare the SID state a raw dump presents to the
chip against the state a parse/decode pipeline reproduces. Discrete registers (CTRL gate/waveform, ADSR,
res/filt, mode/vol) must match exactly; FREQ is compared as a combined word mapped to its cent-index (the
renderer's domain) within a small tolerance, on pitched frames only. Tokens-only: no audio dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preframr_tokens.audit_primitives import register_state
from preframr_tokens.reg_mappers import FreqMapper
from preframr_tokens.stfconstants import FREQ_TRAJ_REGS

FREQ_REGS = tuple(int(r) for r in FREQ_TRAJ_REGS)
CTRL_REGS = (4, 11, 18)
AD_REGS = (5, 12, 19)
SR_REGS = (6, 13, 20)
EXACT_REGS = CTRL_REGS + AD_REGS + SR_REGS + (23, 24)
REG_LABEL = {}
for _i, _r in enumerate(FREQ_REGS):
    REG_LABEL[_r] = f"v{_i}.FREQ"
for _i, _r in enumerate(CTRL_REGS):
    REG_LABEL[_r] = f"v{_i}.CTRL"
for _i, _r in enumerate(AD_REGS):
    REG_LABEL[_r] = f"v{_i}.AD"
for _i, _r in enumerate(SR_REGS):
    REG_LABEL[_r] = f"v{_i}.SR"
REG_LABEL[23] = "RES_FILT"
REG_LABEL[24] = "MODE_VOL"

_SKIP_INIT = 6


def dump_frame_state(dump_path):
    """Raw dump to a ``(n_frames, 25)`` settled byte state: rows sharing an ``irq`` value are one player
    call; the last write per byte-reg settles, forward-filled across frames (the chip holds last value).
    """
    df = pd.read_parquet(dump_path)
    if "chipno" in df.columns:
        df = df[df["chipno"] == 0]
    df = df.sort_values(["irq", "clock"]).reset_index(drop=True)
    cur = np.zeros(25, dtype=np.int64)
    frames = []
    for _irq, grp in df.groupby("irq", sort=True):
        for reg, val in zip(grp["reg"].to_numpy(), grp["val"].to_numpy()):
            r = int(reg)
            if 0 <= r <= 24:
                cur[r] = int(val) & 0xFF
        frames.append(cur.copy())
    return np.asarray(frames)


def _freq_word(state, reg):
    """Combine LO (``reg``) and HI (``reg+1``) of a ``(n,25)`` state into the 16-bit SID freq word; a
    parsed state already holds the combined word in ``reg`` (HI col 0), a raw dump keeps separate bytes.
    """
    lo = state[:, reg].astype(np.int64)
    hi = state[:, reg + 1].astype(np.int64)
    return np.where(hi > 0, (lo & 0xFF) + (hi << 8), lo) & 0xFFFF


def _to_cent_index(words, fm):
    """Map 16-bit freq words to the renderer's cent-index domain; values already at or below the largest
    cent-index key (an already-cent-index stream) pass through unchanged."""
    maxk = max(fm.if_map)
    out = words.copy()
    big = words > maxk
    if big.any():
        lut = fm.fi_map
        out[big] = np.array([lut[int(w) & 0xFFFF] for w in words[big]], dtype=np.int64)
    return out


def best_offset(ref, test, span=24):
    """Frame shift (test delayed by ``k``) minimising EXACT-register mismatch; the dump's irq grid and the
    parser's FRAME_REG grid differ by a few leading init frames."""
    regs = list(EXACT_REGS)
    rr = ref[:, regs].astype(np.int64)
    best_k, best_bad = 0, None
    for k in range(-span, span + 1):
        if k >= 0:
            a, b = rr[: len(rr) - k], test[k:, regs].astype(np.int64)
        else:
            a, b = rr[-k:], test[: len(test) + k, regs].astype(np.int64)
        m = min(len(a), len(b))
        if m < 100:
            continue
        bad = int((a[:m] != b[:m]).any(axis=1).sum())
        if best_bad is None or bad < best_bad:
            best_bad, best_k = bad, k
    return best_k


def diff_states(ref, test, cents=50, freq_tol=1, skip_init=_SKIP_INIT):
    """Align ref/test by frame then diff per register: EXACT_REGS byte-exact; FREQ_REGS combined to a word,
    mapped to cent-index, compared within ``freq_tol`` on every audible frame. Only a TEST-bit frame
    (ctrl bit3) is freq-discardable; a noise frame's freq drives the LFSR rate and IS audible, so it must
    match. Returns a structured result dict."""
    k = best_offset(ref, test)
    if k > 0:
        test = test[k:]
    elif k < 0:
        ref = ref[-k:]
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    fm = FreqMapper(cents=cents)
    res = {
        "offset": k,
        "frames": n,
        "exact": {},
        "freq": {},
        "exact_fail": [],
        "freq_fail": [],
    }
    for r in EXACT_REGS:
        d = (ref[:, r] != test[:, r]).copy()
        d[:skip_init] = False
        c = int(d.sum())
        if c:
            res["exact"][REG_LABEL[r]] = {"frames": c, "first": int(np.argmax(d))}
            res["exact_fail"].append(REG_LABEL[r])
    for i, r in enumerate(FREQ_REGS):
        rc = _to_cent_index(_freq_word(ref, r), fm)
        tc = _to_cent_index(_freq_word(test, r), fm)
        delta = np.abs(rc - tc)
        ctrl = ref[:, CTRL_REGS[i]].astype(np.int64)
        audible = (ctrl & 0x08) == 0
        audible[:skip_init] = False
        over = int(((delta > freq_tol) & audible).sum())
        mx = int(delta[audible].max()) if audible.any() else 0
        res["freq"][REG_LABEL[r]] = {
            "over_tol": over,
            "max_delta": mx,
            "audible": int(audible.sum()),
        }
        if over > 0:
            res["freq_fail"].append(REG_LABEL[r])
    res["ok"] = not res["exact_fail"] and not res["freq_fail"]
    return res


def diff_dump_vs_pipeline(dump_path, xdf, cents=50, freq_tol=1):
    """Diff a raw dump against a parsed/decoded ``xdf`` (a ``RegLogParser.parse`` output)."""
    return diff_states(
        dump_frame_state(dump_path), register_state(xdf), cents=cents, freq_tol=freq_tol
    )


def format_report(res):
    """Render a ``diff_states`` result as human-readable lines."""
    lines = [f"frames={res['frames']} aligned_offset={res['offset']} ok={res['ok']}"]
    for cls, info in res["exact"].items():
        lines.append(
            f"  EXACT FAIL {cls}: {info['frames']} frames (first {info['first']})"
        )
    for cls, info in res["freq"].items():
        flag = "  FREQ FAIL" if info["over_tol"] else "  freq ok "
        lines.append(
            f"{flag} {cls}: over_tol={info['over_tol']} maxdidx={info['max_delta']}"
        )
    return "\n".join(lines)
