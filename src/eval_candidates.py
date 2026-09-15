"""Evaluate candidate generator v2: recall (stage A) + visual overlay (stage B)."""
import os, sys, glob, json, collections
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candgen as CG

DS = PCB_ROOT + "/datasets/mh_board"
FIX = PCB_ROOT + "/pcb-defect-dataset-fixed/train"
OUT = PCB_ROOT + "/outputs/phase2"
Dz = np.load(os.path.join(OUT, "Dz_mean_boardtrain.npy"))
TM = CG.make_templates(Dz)
TOPK = [1, 3, 5, 10, 20, 50, 100]


def hit(x, y, s, gts, used):
    for gi, (cx, cy, w, h) in enumerate(gts):
        if gi in used:
            continue
        if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 and 0.5 <= s / max(w, h) <= 2.0:
            return gi
    return None


def run(tag, use_ring_gate, need_spec=False, grid_cell=0):
    hits = collections.Counter()
    ngt = 0
    ranks = []
    kept_total = 0
    for ip in sorted(glob.glob(os.path.join(DS, "val", "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        col = cv2.imread(ip)
        H, W = col.shape[:2]
        gts = CG.read_gts(os.path.join(DS, "val", "labels", stem + ".txt"), W, H)
        cands = CG.detect(col, TM, topn=200, use_ring_gate=use_ring_gate,
                          need_spec=need_spec, grid_cell=grid_cell)
        kept_total += len(cands)
        used = set()
        for rank, (sc, x, y, s, ft) in enumerate(cands, 1):
            gi = hit(x, y, s, gts, used)
            if gi is not None:
                used.add(gi)
                ranks.append(rank)
        for K in TOPK:
            u2 = set()
            for sc, x, y, s, ft in cands[:K]:
                gi = hit(x, y, s, gts, u2)
                if gi is not None:
                    u2.add(gi)
            hits[K] += len(u2)
        ngt += len(gts)
    r = np.array(ranks) if ranks else np.array([999])
    print("[%s]  GT=%d  avg candidates/img=%.1f" % (tag, ngt, kept_total / 292))
    print("   recall@K : " + "  ".join("K=%d:%.1f%%" % (K, 100 * hits[K] / ngt) for K in TOPK))
    print("   GT rank  : median=%.0f p75=%.0f p90=%.0f  found=%d/%d"
          % (np.median(r), np.percentile(r, 75), np.percentile(r, 90), len(ranks), ngt))
    return {str(K): hits[K] / ngt for K in TOPK}


print("=" * 74)
print("STAGE A - detection recall on mh_board val (292 imgs)")
print("=" * 74)
res = {}
res["v1_no_gate"] = run("v1: no gate", False)
res["v3_ring"] = run("v3: ring gate", True)
res["v3_ring_spec"] = run("v3: ring gate + spec gate", True, need_spec=True)
res["v3_ring_spec_grid"] = run("v3: ring + spec + grid-cap(100px,2)", True, need_spec=True, grid_cell=100)

print("")
print("=" * 74)
print("STAGE B - insertion sites on the 3 demo images")
print("=" * 74)
STEMS = ["light_01_missing_hole_01_1_600",
         "light_04_missing_hole_03_2_600",
         "light_08_missing_hole_07_1_600"]
tiles = []
for stem in STEMS:
    col = cv2.imread(os.path.join(FIX, "images", stem + ".jpg"))
    H, W = col.shape[:2]
    gts = [g for g in CG.read_gts(os.path.join(FIX, "labels", stem + ".txt"), W, H)]
    # labels here are 6-class; keep only class 2
    gts = []
    for line in open(os.path.join(FIX, "labels", stem + ".txt")):
        f = line.split()
        if len(f) == 5 and int(f[0]) == 2:
            cx, cy, w, h = [float(v) for v in f[1:]]
            gts.append((cx * W, cy * H, w * W, h * H))
    sel = CG.sites(col, TM, gts, topn=20, grid_cell=100, grid_cap=2)
    vis = col.copy()
    for rank, (sc, x, y, s, ft) in enumerate(sel, 1):
        h2 = s // 2
        cv2.rectangle(vis, (x - h2, y - h2), (x + h2, y + h2), (255, 0, 0), 2)
        cv2.putText(vis, str(rank), (x - h2, y - h2 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 0, 0), 1, cv2.LINE_AA)
    for cx, cy, w, h in gts:
        cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)),
                      (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 3)
    cv2.putText(vis, stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(vis)
    xs = [c[1] for c in sel]; ys = [c[2] for c in sel]
    spread = float(np.std(xs) + np.std(ys)) if sel else 0.0
    print("  %s  sites=%d  score %.3f..%.3f  spatial spread(std x+y)=%.0f px"
          % (stem, len(sel), sel[-1][0] if sel else 0, sel[0][0] if sel else 0, spread))
Hm = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, Hm - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
cv2.imwrite(os.path.join(OUT, "candidate_overlay_v3.png"), np.hstack(tiles))
print("  saved candidate_overlay_v3.png")

# zoom montage of v2 sites
z = []
for stem in STEMS:
    col = cv2.imread(os.path.join(FIX, "images", stem + ".jpg"))
    H, W = col.shape[:2]
    gts = []
    for line in open(os.path.join(FIX, "labels", stem + ".txt")):
        f = line.split()
        if len(f) == 5 and int(f[0]) == 2:
            cx, cy, w, h = [float(v) for v in f[1:]]
            gts.append((cx * W, cy * H, w * W, h * H))
    for sc, x, y, s, ft in CG.sites(col, TM, gts, topn=6, grid_cell=100, grid_cap=2):
        R = 55
        x0, y0 = max(0, x - R), max(0, y - R)
        t = col[y0:min(H, y + R), x0:min(W, x + R)].copy()
        if t.size == 0:
            continue
        t = cv2.resize(t, (150, 150))
        k = 150 / (2 * R)
        cv2.rectangle(t, (int((x - s / 2 - x0) * k), int((y - s / 2 - y0) * k)),
                      (int((x + s / 2 - x0) * k), int((y + s / 2 - y0) * k)), (255, 0, 0), 2)
        z.append(t)
if z:
    rows = [np.hstack(z[i:i + 6]) for i in range(0, len(z) - len(z) % 6, 6)]
    if rows:
        cv2.imwrite(os.path.join(OUT, "candidate_zoom_v3.png"), np.vstack(rows))
        print("  saved candidate_zoom_v3.png")

json.dump(res, open(os.path.join(OUT, "recall_v3.json"), "w"), indent=2)
