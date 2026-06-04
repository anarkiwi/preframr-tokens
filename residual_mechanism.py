#!/usr/bin/env python3
"""Mechanism-complete residual-SET analysis. Every raw SET is unmodeled driver behaviour; classify
each against register_state ground truth into a NAMED mechanism + the existing-pass precondition it
failed. UNEXPLAINED is a work queue (dumped in full), never an acceptable result."""
import sys
from collections import Counter, defaultdict
import numpy as np
from preframr_tokens import RegLogParser
from preframr_tokens.audit_primitives import register_state
from preframr_tokens.tokenizer_config import default_tokenizer_args
from preframr_tokens.stfconstants import SET_OP, FRAME_REG, DELAY_REG, VOICE_REG_SIZE, MODE_VOL_REG

_BASE = ("preset_pass","hard_restart_pass","legato_pass_c2","legato_pass_c4",
         "voice_canonical_block_order","ctrl_bigram_pass","loop_pass","loop_transposed")
_CODEBOOK = ("skeleton_pass","held_arp","zero_plain","slide_wide","slide_landing","stamp_pass",
             "sweep_pass","sweep_loop","pw_sweep","filter_sweep","wavetable_pass","wt_short",
             "wt_oneshot","patch_pass","ctrl_osc","modevol_gradient","env_gradient","filter_gradient",
             "ctrl_gradient","init_preamble","note_off","note_on","ctrl_wavetable","env_wavetable",
             "filter_wavetable","modevol_wavetable","freq_wavetable","pw_wavetable","onset_instrument")
RN = {0:"FREQ_LO",1:"FREQ_HI",2:"PW_LO",3:"PW_HI",4:"CTRL",5:"AD",6:"SR"}
MIDI_LO_F, MIDI_HI_F = 268, 49096   # ~C1..B7 SID freq-word band (rough MIDI range)


def role(reg):
    reg = int(reg)
    if 0 <= reg < 21: return RN[reg % VOICE_REG_SIZE]
    return {21:"FC_LO",22:"FC_HI",23:"RES_FILT",24:"MODE_VOL"}.get(reg, f"R{reg}")


def decoded_frame_index(df):
    """Cumulative decoded frame per row (FRAME=+1, DELAY=+val) -- matches register_state expansion."""
    regs = df["reg"].to_numpy(); vals = df["val"].to_numpy()
    out = np.empty(len(df), dtype=np.int64); f = -1
    for i in range(len(df)):
        r = int(regs[i])
        if r == FRAME_REG: f += 1
        elif r == DELAY_REG: f += max(1, int(vals[i]))
        out[i] = f
    return out


def classify_seq(tl, f, vreg_active):
    """Classify the decoded timeline tl[:] of one reg around frame f. Returns a mechanism tag for the
    SHAPE of the value's generator."""
    n = len(tl)
    if f < 0 or f >= n: return "oob"
    lo, hi = max(0, f-12), min(n, f+13)
    w = tl[lo:hi]
    cur = int(tl[f])
    # neighbourhood deltas (consecutive distinct levels)
    seg = tl[max(0,f-6):min(n,f+7)].astype(np.int64)
    d = np.diff(seg)
    nz = d[d != 0]
    # periodic? find smallest period P<=8 with w repeating
    period = None
    for P in range(2, 9):
        if hi-lo >= 2*P and all(w[i] == w[i-P] for i in range(P, hi-lo)):
            period = P; break
    if period: return f"periodic_table(P={period})"
    if len(nz) >= 3:
        signs = np.sign(nz)
        flips = int((signs[:-1] != signs[1:]).sum())
        if flips >= 2 and np.all(np.abs(nz) <= 256):
            return "oscillation"
        if np.all(nz > 0) or np.all(nz < 0):
            return "ramp"
    # step to a new sustained level (held >=4 frames after)
    after = tl[f:min(n,f+6)]
    if len(after) >= 3 and np.all(after == cur):
        return "step_hold"
    if len(nz) == 0:
        return "constant"
    return "irregular"


def main():
    args = default_tokenizer_args(seq_len=4096, **{f: True for f in _BASE + _CODEBOOK})
    parser = RegLogParser(args)
    grand = Counter(); grand_role = Counter()
    unexplained = []
    for path in sys.argv[1:]:
        name = path.split("/")[-1]
        try:
            df = next(parser.parse(path, max_perm=1, require_pq=False, reparse=True))
        except StopIteration:
            continue
        try:
            state = register_state(df)
        except Exception as e:
            print(f"[{name}] register_state failed: {e}"); continue
        F = state.shape[0]
        dfi = decoded_frame_index(df)
        regs = df["reg"].to_numpy(); ops = df["op"].to_numpy()
        subs = df["subreg"].to_numpy() if "subreg" in df.columns else np.full(len(df),-1)
        vals = df["val"].to_numpy()
        # bundle: regs written per decoded-frame (encoded df rows)
        frame_regs = defaultdict(set)
        for i in range(len(df)):
            r = int(regs[i])
            if 0 <= r < 25: frame_regs[int(dfi[i])].add(role(r) if r<21 else role(r))
        mech = Counter()
        for i in range(len(df)):
            if int(ops[i]) != SET_OP: continue
            r = int(regs[i])
            if not (0 <= r < 25): continue
            ro = role(r); f = int(dfi[i]); v = r // VOICE_REG_SIZE if r < 21 else -1
            grand_role[ro] += 1
            # ground-truth timelines
            tl = state[:, r] if r < 25 else None
            shape = classify_seq(tl, f, None) if tl is not None else "?"
            # gate at this frame (voice ctrl)
            gate_now = gate_prev = None
            if 0 <= v < 3 and f < F and f-1 >= 0:
                creg = v*VOICE_REG_SIZE+4
                gate_now = int(state[f, creg]) & 1
                gate_prev = int(state[f-1, creg]) & 1
            trigger = (gate_now == 1 and gate_prev == 0)
            release = (gate_now == 0 and gate_prev == 1)
            startup = f < max(8, int(0.03*F))
            in_midi = (ro == "FREQ_LO") and (MIDI_LO_F <= int(vals[i]) <= MIDI_HI_F)
            # recurrence of this exact decoded value on this reg
            rec = int((state[:, r] == int(state[f, r])).sum()) if (r < 25 and f < F) else 0
            # ---- mechanism assignment (priority maps to a fix) ----
            m = None
            if startup: m = f"STARTUP/{ro}/{shape}"
            elif ro in ("AD","SR"):
                # patch precondition checks
                same = [j for j in range(len(df)) if int(dfi[j])==f and int(regs[j])//VOICE_REG_SIZE==v and int(regs[j])%VOICE_REG_SIZE==(r%VOICE_REG_SIZE) and int(ops[j])==SET_OP]
                if len(same) > 1: m = "ENVELOPE/hard_restart_multiload"
                elif trigger or release: m = f"ENVELOPE/{'release' if release else 'trigger'}_not_bundled"
                else: m = f"ENVELOPE/{shape}"
            elif ro in ("FREQ_LO","FREQ_HI"):
                if shape.startswith("periodic"): m = f"FREQ/{shape}"
                elif shape in ("ramp","oscillation"): m = f"FREQ/{shape}"
                elif shape == "step_hold":
                    m = "FREQ/note_in_range" if in_midi else "FREQ/note_out_of_range"
                else: m = f"FREQ/{shape}"
            elif ro in ("PW_LO","PW_HI"): m = f"PW/{shape}"
            elif ro == "CTRL":
                if release or (int(vals[i]) & 1)==0 and (gate_prev==1): m = "CTRL/gate_off_release"
                elif trigger: m = "CTRL/gate_on_trigger"
                else: m = f"CTRL/{shape}"
            elif ro in ("FC_LO","FC_HI","RES_FILT"): m = f"FILTER/{shape}"
            elif ro == "MODE_VOL": m = f"MODEVOL/{shape}"
            else: m = f"OTHER/{ro}/{shape}"
            # refine the crude "irregular"/OTHER shapes into NAMED mechanisms by recurrence + reg:
            # nothing is "irregular" -- it is either a one-time init write, a recurring table-driven
            # state (codebook-able), or a genuinely rare state that is the real work queue.
            if "irregular" in m or "OTHER" in m:
                if ro in ("MODE_VOL", "RES_FILT", "FC_LO", "FC_HI") and rec <= 3 and startup:
                    m = f"INIT/one_time_setup/{ro}"
                elif rec >= 8:
                    m = f"STATE_CODEBOOK/{ro}/recurring"   # table-driven recurring state
                else:
                    m = f"RARE/{ro}(rec={rec})"
                    unexplained.append((name, f, ro, v, int(subs[i]), int(vals[i]), shape, rec, trigger, release))
            mech[m] += 1
        total = sum(1 for i in range(len(df)) if int(ops[i])==SET_OP and 0<=int(regs[i])<25)
        print(f"\n[{name}] {F} frames; {total} residual SETs")
        for m, c in mech.most_common():
            print(f"   {c:5d}  {m}")
        grand += mech
    print(f"\n\n===== AGGREGATE mechanism census =====")
    tot = sum(grand.values())
    for m, c in grand.most_common():
        print(f"   {c:6d} ({100*c/tot:4.1f}%)  {m}")
    print(f"\n===== UNEXPLAINED / irregular ({len(unexplained)}) — full dump (work queue) =====")
    for rec in unexplained[:60]:
        print(f"   {rec}")


if __name__ == "__main__":
    main()
