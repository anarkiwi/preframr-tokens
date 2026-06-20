"""Pitch-invariant FREQ-INSTRUMENT factoring for the step/tracker codec (Monty).

Proven against Hubbard's playroutine: a freq generator's per-frame Fn deltas are
NOT distinct content -- they are ONE instrument rendered at many pitches. The
driver computes the modulation FROM THE NOTE:

  vibrato (asm L367): diff = (notetab[note+1] - notetab[note]) >> vibrdepth ;
                      freq = base + diff * triangle[counter&7]
                      (triangle = 0,1,2,3,3,2,1,0)
  octarp  (asm L613): freq toggles notetab[note], notetab[note+12] (Fn doubles)
  skydive (asm L590): slide -- dec high byte every 2nd frame

So the INSTRUMENT is one param -- vibrdepth (a shift), or the octave/slide flag --
and the pitch scaling comes from the NOTE-TABLE INTERVAL at render time, replicated
with the driver's exact integer math. The same instrument is ONE definition for
ALL notes.

DECODE RENDERS each generator at the note's pitch by looking up the note table and
replicating (notetab[i+k]-notetab[i]) >> depth, so the per-frame Fn deltas are
reproduced BYTE-EXACT for every note that sits on the table. The residual gate is
the EVENT-CODEC decode (itself <=50c grid-snap-lossy on the base); the handful of
ops whose event-codec base was snapped off the note table carry a sparse exact
correction (never opaque -- the rule + correction reproduce the recorded deltas).

A freq generator = (pitch_anchor, instrument_ref, n[, sparse_correction]):
  pitch_anchor  -- base Fn from the row's preceding NOTE (cur_fn); free when the
                   generator starts at the held pitch, else a base correction int.
  instrument    -- a pitch-INVARIANT shape, dedup'd inline (model-legal):
                     HOLD              lane holds base for n frames
                     OCT               octave arp (notetab[i] <-> notetab[i+12])
                     VIB(depth, tri)   vibrato shift + the triangle unit pattern
                     SLIDE(rate)       linear pitch slide (rate is per-frame Fn add)
                     RATIO(qper)       fallback: per-period log2 ratio of base
  n             -- tile count (cheap uint), so one shape serves every duration.
  correction    -- sparse (frame, exact_Fn) only where the render != original."""

import collections

import numpy as np

from preframr_tokens.codec import pitch_universal_anchor as A
from preframr_tokens.codec import serialize_events as SE
from preframr_tokens.codec import step_codec as SC

SEMI = A.SEMI
uc = SC.u_cost
ic = SC.i_cost
QSTEP = 1.0 / 512.0

# Hubbard Monty note-frequency table (notefreqsl/notefreqsh, asm).
_NT_HEX = """0116 0127 0138 014b 015f 0173 018a 01a1 01ba 01d4 01f0 020e
022d 024e 0271 0296 02bd 02e7 0313 0342 0374 03a9 03e0 041b
045a 049b 04e2 052c 057b 05ce 0627 0685 06e8 0751 07c1 0837
08b4 0937 09c4 0a57 0af5 0b9c 0c4e 0d09 0dd0 0ea3 0f82 106e
1168 126e 1388 14af 15eb 1739 189c 1a13 1ba1 1d46 1f04 20dc
22d0 24dc 2710 295e 2bd6 2e72 3138 3426 3742 3a8c 3e08 41b8
45a0 49b8 4e20 52bc 57ac 5ce4 6270 684c 6e84 7518 7c10 8370
8b40 9370 9c40 a578 af58 b9c8 c4e0 d098 dd08 ea30 f820 fd2e"""
NOTETAB = [int(x, 16) for x in _NT_HEX.split()]
_NT_INDEX = {v: i for i, v in enumerate(NOTETAB)}


def _note_idx(base):
    """Note-table index for a base Fn (exact if on table, else nearest)."""
    i = _NT_INDEX.get(base)
    if i is not None:
        return i
    return int(np.argmin([abs(v - base) for v in NOTETAB]))


def _voice_phase(s, v):
    fn = s[:, 2 * v].astype(np.int64) | (s[:, 2 * v + 1].astype(np.int64) << 8)
    return A.encode_voice(fn)[1]


def _gen(op):
    nm = op[0]
    if nm == "MOD":
        return list(op[1]), op[2], int(op[3])
    if nm == "SLIDE":
        return [int(op[1])], op[2], int(op[3])
    if nm == "ARP":
        return SE._arp_period_deltas(op[1]), op[2], int(op[3])
    if nm == "VIB":
        _, s_up, u, s_dn, d, rot, n, base = op
        canon = [s_up] * u + [s_dn] * d
        p = len(canon)
        return [canon[(rot + k) % p] for k in range(p)], n, int(base)
    return None


def _levels(e, n, base):
    out, acc, p = [], float(base), len(e)
    for k in range(n):
        acc += e[k % p]
        out.append(int(round(acc)))
    return out


def _vib_depth(d, base):
    """Driver-derived vibrato depth: the shift s.t. (notetab[i+1]-notetab[i])>>s == d.
    Returns the shift (so d is FREE -- derived from the note table at decode), or
    None if no shift reproduces d (then d is carried per-occurrence)."""
    i = _note_idx(base)
    if i + 1 >= len(NOTETAB):
        return None
    gd = NOTETAB[i + 1] - NOTETAB[i]
    for s in range(8):
        if (gd >> s) == d:
            return s
    return None


def _fit_vibrato(op):
    """If the op is a triangle vibrato, return (tri_unit, dwells, d): the per-period
    level offsets == d * tri_unit. The (tri_unit, dwells) is the pitch-INVARIANT
    instrument shape; d is the vibrato amplitude (driver: a note-table interval
    shifted by vibrdepth)."""
    if op[0] != "ARP":
        return None
    steps = op[1]
    lds = [ld for ld, _ in steps]
    dwells = tuple(int(w) for _, w in steps)
    nz = [abs(x) for x in lds if x != 0]
    if not nz:
        return None
    d = int(np.gcd.reduce(nz))
    tri = tuple(int(x) // d for x in lds)
    if max(abs(t) for t in tri) > 8:
        return None
    return tri, dwells, d


def _shape_of(op, base):
    """Pitch-invariant instrument shape + (per-occurrence amplitude d, derive-shift).
    Returns (shape, d, shift). shape is the dedup'd codebook key; d/shift are
    per-occurrence (shift!=None -> d is note-table-derived and FREE)."""
    e, _, _ = _gen(op)
    if all(d == 0 for d in e):
        return ("HOLD",), 0, None
    if op[0] == "VIB" and op[1] == op[7] and op[3] == -op[7]:
        return ("OCT",), 0, None
    if op[0] == "SLIDE":
        return ("SLIDE", int(op[1])), 0, None
    vf = _fit_vibrato(op)
    if vf is not None:
        tri, dwells, d = vf
        shift = _vib_depth(d, base)
        return ("VIB", tri, dwells), d, shift
    # MOD / residual generators: the per-period delta pattern e is PITCH-INVARIANT
    # (it does not depend on base -- it is the driver's per-cycle table data, the
    # music). Store it literally so it dedups across notes and renders byte-exact.
    return ("DELTA", tuple(int(x) for x in e)), 0, None


def _render(shape, n, base, d=0, shift=None):
    tag = shape[0]
    if tag == "HOLD":
        return [int(round(base))] * n
    if tag == "OCT":
        return [(2 * base if k % 2 == 0 else base) for k in range(n)]
    if tag == "SLIDE":
        rate = shape[1]
        out, acc = [], float(base)
        for _ in range(n):
            acc += rate
            out.append(int(round(acc)))
        return out
    if tag == "VIB":
        tri, dwells = shape[1], shape[2]
        if shift is not None:  # driver-derived amplitude (free)
            i = _note_idx(base)
            gd = (NOTETAB[i + 1] - NOTETAB[i]) if i + 1 < len(NOTETAB) else 0
            d = gd >> shift
        period = []
        for t, w in zip(tri, dwells):
            period += [base + d * t] * w
        return [period[k % len(period)] for k in range(n)]
    if tag == "DELTA":
        e = shape[1]
        return _levels(e, n, base)
    qper = shape[1]
    p = len(qper)
    return [
        (
            int(round(base * 2.0 ** (qper[k % p] * QSTEP * SEMI)))
            if qper[k % p] is not None
            else 0
        )
        for k in range(n)
    ]


def measure_freq(s):
    ev = SE.encode_tune_events(s[:, :25])
    bylane = collections.defaultdict(list)
    for sf, lid, op in ev:
        if lid < 3:
            bylane[lid].append((sf, op))
    brk = collections.Counter()
    n_instr = 0
    residual_ok = True
    real_instr = set()
    for v in range(3):
        ops = sorted(bylane[v])
        cb = {}
        cur_fn, cur_idx, last_ti = 0, None, None
        for sf, op in ops:
            nm = op[0]
            if nm == "NOTE":
                interval, micro = op[1], op[2]
                cur_idx = interval if cur_idx is None else cur_idx + interval
                cur_fn = min(
                    65535,
                    max(
                        0, int(round(A._grid_fn(SE._phase_unq(op[3]), cur_idx, micro)))
                    ),
                )
                last_ti = _note_idx(cur_fn)
                brk["note"] += ic(op[1]) + ic(op[2])
                continue
            if nm in ("RAW", "REST"):
                cur_fn = int(op[1]) if nm == "RAW" else 0
                last_ti = _note_idx(cur_fn) if nm == "RAW" else None
                brk["note"] += uc(op[1])
                continue
            g = _gen(op)
            if g is None:
                brk["literal"] += SC._op_body_cost(op, True)
                continue
            e, n, base = g
            shape, d, shift = _shape_of(op, base)
            real_instr.add(shape if shape[0] != "RATIO" else ("RATIO",))
            # anchor base as a note-table-index delta from the row's last note
            # (0 when the generator plays at the held pitch). Off-table -> raw Fn.
            if base in _NT_INDEX:
                bi = _NT_INDEX[base]
                brk["base_corr"] += ic(bi - last_ti) if last_ti is not None else uc(bi)
                last_ti = bi
            else:
                brk["base_corr"] += ic(base)
            if shape in cb:
                brk["instr_ref"] += uc(cb[shape])
            else:
                cb[shape] = len(cb)
                brk["instr_def"] += 1
                if shape[0] == "VIB":
                    tri, dwells = shape[1], shape[2]
                    brk["instr_def"] += uc(len(tri))
                    for t in tri:
                        brk["instr_def"] += ic(t)
                    for w in dwells:
                        brk["instr_def"] += uc(w)
                elif shape[0] == "SLIDE":
                    brk["instr_def"] += ic(shape[1])
                elif shape[0] == "DELTA":
                    e_ = shape[1]
                    nz = [(k, x) for k, x in enumerate(e_) if x != 0]
                    brk["instr_def"] += uc(len(e_)) + uc(len(nz))
                    prev = -1
                    for k, x in nz:
                        brk["instr_def"] += uc(k - prev - 1) + ic(x)
                        prev = k
            if shape[0] == "VIB":
                # 1 flag bit: derived (free amplitude) vs carried d
                brk["vib_amp"] += 1 + (0 if shift is not None else uc(d))
            brk["n_field"] += uc(n)
            target = _levels(e, n, base)
            render = _render(shape, n, base, d, shift)
            if render != target:
                diffs = [
                    (k, t) for k, (t, r) in enumerate(zip(target, render)) if t != r
                ]
                brk["corr"] += uc(len(diffs))
                prev = -1
                for k, t in diffs:
                    brk["corr"] += uc(k - prev - 1) + ic(t)
                    prev = k
                fixed = list(render)
                for k, t in diffs:
                    fixed[k] = t
                if fixed != target:
                    residual_ok = False
            cur_fn = target[-1]
            if cur_fn in _NT_INDEX:  # slide may land on a new table note
                last_ti = _NT_INDEX[cur_fn]
        n_instr += len(cb)
        brk[f"instr_v{v}"] = len(cb)
    brk["freq_total"] = (
        brk["note"]
        + brk["base_corr"]
        + brk["instr_def"]
        + brk["instr_ref"]
        + brk["n_field"]
        + brk["vib_amp"]
        + brk["corr"]
        + brk["literal"]
    )
    brk["n_instr"] = n_instr
    brk["n_real_instr"] = len(real_instr)
    brk["residual_ok"] = int(residual_ok)
    return dict(brk)


if __name__ == "__main__":
    from preframr_tokens.codec import lane_grammar as G
    from preframr_tokens.codec import lsp_validate as V

    DUMP = (
        "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/"
        "Monty_on_the_Run.1.dump.parquet"
    )
    s = G.per_frame_state(DUMP, V.CPF, 10**9)[:, :25]
    b = measure_freq(s)
    BASE = 12997 + 6053
    print("=== pitch-invariant freq-instrument factoring (note-table render) ===")
    for k in (
        "note",
        "base_corr",
        "instr_def",
        "instr_ref",
        "n_field",
        "vib_amp",
        "corr",
        "literal",
        "freq_total",
        "n_instr",
        "n_real_instr",
        "instr_v0",
        "instr_v1",
        "instr_v2",
        "residual_ok",
    ):
        print("  %-13s %8d" % (k, b.get(k, 0)))
    print("\nbaseline freq lane (pitch_def+pitch_ref) =", BASE)
    print(
        "factored freq lane total =",
        b["freq_total"],
        "(delta %+d)" % (b["freq_total"] - BASE),
    )
