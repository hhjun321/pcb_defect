"""
Overlay on the same 3 images:
  RED  = GT missing_hole (ground truth defect)
  BLUE = brightness-search candidate insertion sites (top-K peaks NOT overlapping GT)
"""
import os, glob
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

SRC = PCB_ROOT + "/pcb-defect-dataset-fixed/train"
DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase2"
TMPL_NPY = os.path.join(OUT, "Dz_mean_boardtrain.npy")
S = 32
SCALES = [20, 24, 28, 32, 36, 40, 48]
TOPK = 20

STEMS = ["light_01_missing_hole_01_1_600",
         "light_04_missing_hole_03_2_600",
         "light_08_missing_hole_07_1_600"]

Dz = np.load(TMPL_NPY).astype(np.float32)
TM = {s: cv2.resize(Dz, (s, s), interpolation=cv2.INTER_LINEAR) for s in SCALES}


def search(img, topn=400):
    H, W = img.shape
    best = np.full((H, W), -2.0, np.float32)
    bs = np.zeros((H, W), np.int32)
    for s in SCALES:
        r = cv2.matchTemplate(img, TM[s], cv2.TM_CCOEFF_NORMED)
        full = np.full((H, W), -2.0, np.float32)
        o = s // 2
        full[o:o + r.shape[0], o:o + r.shape[1]] = r
        m = full > best
        best[m] = full[m]
        bs[m] = s
    flat = best.ravel()
    idx = np.argpartition(flat, -topn * 40)[-topn * 40:]
    idx = idx[np.argsort(-flat[idx])]
    taken = np.zeros((H, W), bool)
    out = []
    for i in idx:
        y, x = divmod(int(i), W)
        if taken[y, x]:
            continue
        s = int(bs[y, x])
        rad = max(4, s // 2)
        taken[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = True
        out.append((float(best[y, x]), x, y, s))
        if len(out) >= topn:
            break
    return out


tiles = []
for stem in STEMS:
    ip = os.path.join(SRC, "images", stem + ".jpg")
    lp = os.path.join(SRC, "labels", stem + ".txt")
    col = cv2.imread(ip)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    gts = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5 and int(f[0]) == 2:
            cx, cy, w, h = [float(v) for v in f[1:]]
            gts.append((cx * W, cy * H, w * W, h * H))

    cands = search(gray, topn=400)
    keep = []
    for sc, x, y, s in cands:
        if any(abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 for cx, cy, w, h in gts):
            continue
        keep.append((sc, x, y, s))
        if len(keep) >= TOPK:
            break

    vis = col.copy()
    for rank, (sc, x, y, s) in enumerate(keep, 1):
        h2 = s // 2
        cv2.rectangle(vis, (x - h2, y - h2), (x + h2, y + h2), (255, 0, 0), 2)   # BLUE
        cv2.putText(vis, str(rank), (x - h2, y - h2 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 0, 0), 1, cv2.LINE_AA)
    for cx, cy, w, h in gts:
        cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                      (0, 0, 255), 3)                                            # RED
    cv2.putText(vis, stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, stem, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(vis)
    print("%s  GT=%d  candidates drawn=%d  score range %.3f..%.3f"
          % (stem, len(gts), len(keep), keep[-1][0], keep[0][0]))

Hm = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, Hm - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
canvas = np.hstack(tiles)
fn = os.path.join(OUT, "candidate_overlay.png")
cv2.imwrite(fn, canvas)
print("saved", fn, canvas.shape)

# also a zoomed montage of the top candidates with surroundings
z = []
for stem in STEMS:
    ip = os.path.join(SRC, "images", stem + ".jpg")
    col = cv2.imread(ip)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    lp = os.path.join(SRC, "labels", stem + ".txt")
    gts = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5 and int(f[0]) == 2:
            cx, cy, w, h = [float(v) for v in f[1:]]
            gts.append((cx * W, cy * H, w * W, h * H))
    cands = [c for c in search(gray, topn=400)
             if not any(abs(c[1] - cx) <= w / 2 and abs(c[2] - cy) <= h / 2 for cx, cy, w, h in gts)][:6]
    for sc, x, y, s in cands:
        R = 55
        x0, y0 = max(0, x - R), max(0, y - R)
        t = col[y0:min(H, y + R), x0:min(W, x + R)].copy()
        if t.size == 0:
            continue
        t = cv2.resize(t, (150, 150))
        cv2.rectangle(t, (int((x - s / 2 - x0) * 150 / (2 * R)), int((y - s / 2 - y0) * 150 / (2 * R))),
                      (int((x + s / 2 - x0) * 150 / (2 * R)), int((y + s / 2 - y0) * 150 / (2 * R))),
                      (255, 0, 0), 2)
        z.append(t)
if z:
    rows = [np.hstack(z[i:i + 6]) for i in range(0, len(z) - len(z) % 6, 6)]
    if rows:
        cv2.imwrite(os.path.join(OUT, "candidate_zoom.png"), np.vstack(rows))
        print("saved candidate_zoom.png")
