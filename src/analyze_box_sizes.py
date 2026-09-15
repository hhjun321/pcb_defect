import os, glob, collections
import numpy as np
from PIL import Image

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

ROOT = PCB_ROOT + "/pcb-defect-dataset"
NAMES = {0:"mouse_bite",1:"spur",2:"missing_hole",3:"short",4:"open_circuit",5:"spurious_copper"}

# 1) image size survey (sample + full uniqueness check on a subset)
sizes = collections.Counter()
imgs = sorted(glob.glob(os.path.join(ROOT,"train","images","*")))
for p in imgs[::50]:
    sizes[Image.open(p).size] += 1
print("== image size sample (train, every 50th, n=%d)" % len(imgs[::50]))
for s,c in sizes.most_common():
    print("  ", s, c)

# 2) box stats per split
def stats(split):
    rows=[]
    for lp in glob.glob(os.path.join(ROOT,split,"labels","*.txt")):
        ip = lp.replace(os.sep+"labels"+os.sep, os.sep+"images"+os.sep).replace(".txt",".jpg")
        W,H = Image.open(ip).size if os.path.exists(ip) else (600,600)
        for line in open(lp):
            f=line.split()
            if len(f)!=5: continue
            c=int(f[0]); cx,cy,w,h=map(float,f[1:])
            rows.append((c, cx*W, cy*H, w*W, h*H, W, H, os.path.basename(lp)))
    return rows

allrows={}
for sp in ["train","val","test"]:
    allrows[sp]=stats(sp)
    print("== %s: %d boxes" % (sp,len(allrows[sp])))

def desc(a,label):
    a=np.asarray(a,dtype=float)
    q=np.percentile(a,[0,1,5,25,50,75,95,99,100])
    print("  %-8s n=%5d mean=%6.2f std=%5.2f | min=%.1f p1=%.1f p5=%.1f p25=%.1f p50=%.1f p75=%.1f p95=%.1f p99=%.1f max=%.1f"
          % (label,len(a),a.mean(),a.std(),*q))

print("\n===== missing_hole (class 2) box size in pixels =====")
for sp in ["train","val","test"]:
    r=[x for x in allrows[sp] if x[0]==2]
    print("-- %s (n=%d)" % (sp,len(r)))
    desc([x[3] for x in r],"w")
    desc([x[4] for x in r],"h")
    desc([max(x[3],x[4]) for x in r],"maxwh")
    desc([np.sqrt(x[3]*x[4]) for x in r],"sqrt(wh)")
    desc([x[3]/x[4] for x in r],"aspect")

print("\n===== all classes, train, sqrt(w*h) =====")
for c in sorted(NAMES):
    r=[x for x in allrows["train"] if x[0]==c]
    desc([np.sqrt(x[3]*x[4]) for x in r], NAMES[c])

# 3) missing_hole: how many per image, and images containing ONLY class2
print("\n===== per-image composition (train) =====")
byimg=collections.defaultdict(list)
for x in allrows["train"]: byimg[x[7]].append(x[0])
n_any2=sum(1 for v in byimg.values() if 2 in v)
n_only2=sum(1 for v in byimg.values() if set(v)=={2})
n_no2=sum(1 for v in byimg.values() if 2 not in v)
print("  images total=%d, contain missing_hole=%d, ONLY missing_hole=%d, no missing_hole=%d" % (len(byimg),n_any2,n_only2,n_no2))
cnt2=collections.Counter(sum(1 for c in v if c==2) for v in byimg.values())
print("  missing_hole count per image:", dict(sorted(cnt2.items())))
cntall=collections.Counter(len(v) for v in byimg.values())
print("  total boxes per image:", dict(sorted(cntall.items())))

# 4) candidate normalization sizes -> what fraction of boxes fit
print("\n===== fit rate for fixed patch size (train missing_hole, maxwh <= S) =====")
mw=np.array([max(x[3],x[4]) for x in allrows["train"] if x[0]==2])
for S in [16,24,32,40,48,56,64,80]:
    print("  S=%3d -> %.1f%% fit" % (S, 100.0*(mw<=S).mean()))
