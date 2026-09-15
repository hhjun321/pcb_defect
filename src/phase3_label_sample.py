"""
Phase 3b - sample non-GT candidates for manual validity labelling.

Why: AUC(GT vs non-GT) is the wrong yardstick for the context stage.  The
non-GT set mixes two classes with OPPOSITE requirements:

    normal pad        -> a valid Copy-Paste insertion site, must be KEPT
    silkscreen / gap  -> not a pad at all, must be REJECTED

Averaged together they cancel, which is why context scored 0.853 vs 0.848.
So we label a stratified sample by hand and score the context stage on the
question it actually answers: pad vs non-pad.

Run this, label the montages, then run phase3_label_eval.py.
"""
import os, sys, glob, json
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candgen as CG
import phase3_context as P3

DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase3"
P2 = PCB_ROOT + "/outputs/phase2"
os.makedirs(OUT, exist_ok=True)

N_SAMPLE = 120
PER_SHEET = 30
TILE = 150
RNG = np.random.default_rng(7)

single = np.load(os.path.join(OUT, "C_single.npy"))
cen = np.load(os.path.join(OUT, "C_kmeans.npy"))
Dz = np.load(os.path.join(P2, "Dz_mean_boardtrain.npy"))
TM = CG.make_templates(Dz)

print("collecting stage-A candidates on mh_board val ...")
rec = []
for stem, ip, lp in P3.load_split("val"):
    col = cv2.imread(ip)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    gts = P3.gts_of(lp, W, H)
    for sc, x, y, s, ft in CG.detect(col, TM, topn=10, use_ring_gate=True):
        is_gt = any(abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 and 0.5 <= s / max(w, h) <= 2.0
                    for cx, cy, w, h in gts)
        if is_gt:
            continue
        v = P3.context_patch(gray, x, y, s)
        if v is None:
            continue
        rec.append(dict(stem=stem, x=int(x), y=int(y), s=int(s), ncc=float(sc),
                        ctx_single=float(v @ single), ctx_kmeans=float((cen @ v).max()),
                        spec=float(ft["spec"]), core_sat=float(ft["core_sat"]),
                        ring_sat=float(ft["ring_sat"]), ring_val=float(ft["ring_val"])))
print("non-GT candidates available: %d" % len(rec))

# stratify by brightness score so the sample spans the whole ranking
order = sorted(range(len(rec)), key=lambda i: -rec[i]["ncc"])
strata = np.array_split(np.array(order), 4)
pick = []
per = N_SAMPLE // len(strata)
for st in strata:
    pick.extend(RNG.choice(st, size=min(per, len(st)), replace=False).tolist())
pick = pick[:N_SAMPLE]
sample = [rec[i] for i in pick]
for k, r in enumerate(sample):
    r["id"] = k

print("sampled %d (stratified into %d brightness-score bands)" % (len(sample), len(strata)))
json.dump(sample, open(os.path.join(OUT, "label_sample.json"), "w"), indent=1)

# ---- render montages
for sheet in range((len(sample) + PER_SHEET - 1) // PER_SHEET):
    chunk = sample[sheet * PER_SHEET:(sheet + 1) * PER_SHEET]
    tiles = []
    for r in chunk:
        col = cv2.imread(os.path.join(DS, "val", "images", r["stem"] + ".jpg"))
        H, W = col.shape[:2]
        R = int(round(1.6 * r["s"]))
        x0, y0 = max(0, r["x"] - R), max(0, r["y"] - R)
        x1, y1 = min(W, r["x"] + R), min(H, r["y"] + R)
        t = col[y0:y1, x0:x1].copy()
        if t.size == 0:
            t = np.zeros((TILE, TILE, 3), np.uint8)
        t = cv2.resize(t, (TILE, TILE))
        k = TILE / float(x1 - x0)
        hs = r["s"] / 2.0
        cv2.rectangle(t,
                      (int((r["x"] - hs - x0) * k), int((r["y"] - hs - y0) * k)),
                      (int((r["x"] + hs - x0) * k), int((r["y"] + hs - y0) * k)),
                      (255, 0, 0), 2)
        cv2.putText(t, str(r["id"]), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(t, str(r["id"]), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.full((TILE, TILE, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    fn = os.path.join(OUT, "label_sheet_%d.png" % sheet)
    cv2.imwrite(fn, np.vstack(rows))
    print("saved", fn)
