import sys, numpy as np
from collections import Counter
from preframr_tokens import RegLogParser
from preframr_tokens.tokenizer_config import default_tokenizer_args
from preframr_tokens.stfconstants import SET_OP, FRAME_REG, DELAY_REG, VOICE_REG_SIZE
flags=("preset_pass","hard_restart_pass","legato_pass_c2","legato_pass_c4","voice_canonical_block_order","ctrl_bigram_pass","loop_pass","loop_transposed","skeleton_pass","held_arp","zero_plain","slide_wide","slide_landing","stamp_pass","sweep_pass","sweep_loop","pw_sweep","filter_sweep","wavetable_pass","wt_short","wt_oneshot","patch_pass","ctrl_osc","modevol_gradient","env_gradient","filter_gradient","ctrl_gradient","init_preamble","note_off","note_on","ctrl_wavetable","env_wavetable","filter_wavetable","modevol_wavetable","freq_wavetable","pw_wavetable","onset_instrument")
args=default_tokenizer_args(seq_len=4096, **{f:True for f in flags})
parser=RegLogParser(args)
def fidx(df):
    regs=df["reg"].to_numpy(); vals=df["val"].to_numpy()
    out=np.empty(len(df),dtype=np.int64); f=-1
    for i in range(len(df)):
        r=int(regs[i])
        if r==FRAME_REG: f+=1
        elif r==DELAY_REG: f+=max(1,int(vals[i]))
        out[i]=f
    return out
def rolename(r):
    if 0<=r<21: return {0:"FREQ",1:"FREQ",2:"PW",3:"PW",4:"CTRL",5:"AD",6:"SR"}[r%7]
    return {21:"FC",22:"FC",23:"RESFILT",24:"MODEVOL"}.get(r,"?")
bucket=Counter()
total=0
for path in sys.argv[1:]:
    try: df=next(parser.parse(path, max_perm=1, require_pq=False, reparse=True))
    except StopIteration: continue
    fi=fidx(df); regs=df["reg"].to_numpy(); ops=df["op"].to_numpy(); vals=df["val"].to_numpy()
    F=int(fi.max())+1 if len(fi) else 0
    # per-reg list of (frame, val) for residual SETs (any subreg)
    byreg={}
    for i in range(len(df)):
        if int(ops[i])==SET_OP and 0<=int(regs[i])<25:
            byreg.setdefault(int(regs[i]),[]).append((int(fi[i]),int(vals[i])))
    for r,seq in byreg.items():
        seq.sort()
        frames=[f for f,_ in seq]; valseq=[v for _,v in seq]
        ro=rolename(r)
        for k,(f,v) in enumerate(seq):
            total+=1
            startup = f < max(8, int(0.03*F))
            gprev = frames[k]-frames[k-1] if k>0 else 999
            gnext = frames[k+1]-frames[k] if k+1<len(frames) else 999
            gap=min(gprev,gnext)
            nrun=sum(1 for ff,_ in seq if ff==f)  # multi-write same frame
            valcount=valseq.count(v)
            if nrun>1: b="multiwrite_sameframe"          # hard-restart multiload etc
            elif startup and valcount<=2: b="init_startup"
            elif gap<=1: b="per_frame_dense"             # oscillation/sweep territory
            elif valcount>=3: b="recurring_value"        # codebook territory
            else: b="held_step_automation"               # sparse held curve
            bucket[(ro,b)]+=1
print(f"TOTAL residual SETs: {total}")
agg=Counter()
for (ro,b),c in bucket.items(): agg[b]+=c
print("=== by shape bucket ===")
for b,c in agg.most_common(): print(f"  {c:5d} ({100*c/total:4.1f}%)  {b}")
print("=== by role x bucket ===")
for (ro,b),c in sorted(bucket.items(),key=lambda x:-x[1])[:25]: print(f"  {c:5d}  {ro:8s} {b}")
