import glob, os
from preframr_tokens.dump_meta import meta_path_for, read_meta
corpus=os.environ.get("PREFRAMR_RESID_CORPUS","/scratch/preframr/hvsc")
allf=sorted(glob.glob(os.path.join(corpus,"**","*.dump.parquet"),recursive=True))
sample=allf[::1500]
keep=[]
for p in sample:
    try:
        m=read_meta(meta_path_for(p))
        if getattr(m,"is_digi",False): continue
    except Exception: pass
    keep.append(p)
open("/tok/.resid_sample.txt","w").write("\n".join(keep))
print(len(keep),"kept of",len(sample))
