"""
Calibrate a 'valid pad site' filter for brightness-search candidates.

Rationale
---------
A usable insertion site must have the SAME metal annulus as a real missing_hole
site.  Only the CENTER differs (real defect = green substrate, normal pad =
drilled hole / metal).  So all features are computed on the RING only, plus one
core-vs-ring contrast term.  Ring features are therefore shared by GT defects
and normal pads, while silkscreen text / substrate blobs fail them.

Features on a square patch of side s around the candidate centre:
  ring mask : 0.24s <= r <= 0.34s      core mask : r <= 0.14s
  (bands taken from the measured radial profile: ring peak at r/s~0.28)
  f_ring_z  : mean z-brightness of ring         (bright metal annulus)
  f_ring_min: min over 16 angular sectors of    (ANGULAR COMPLETENESS ->
              mean z in that sector               silkscreen strokes fail)
  f_core_z  : mean z-brightness of core         (dark centre)
  f_contrast: f_ring_z - f_core_z
  f_ring_sat: mean HSV saturation of ring       (metal = low, green PCB = high)
  f_ring_val: mean HSV value of ring
"""
import os, glob, json
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase2"
S = 32


def masks(s):
    yy, xx = np.mgrid[0:s, 0:s]
    cx = cy = (s - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / s
    ang = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)  # 0..1
    ring = (r >= 0.24) & (r <= 0.34)
    core = r <= 0.14
    sect = np.clip((ang * 16).astype(int), 0, 15)
    return ring, core, sect


_MC = {}


def get_masks(s):
    if s not in _MC:
        _MC[s] = masks(s)
    return _MC[s]


def features(col, x, y, s):
    """col: BGR uint8 image; (x,y) centre; s: box side"""
    H, W = col.shape[:2]
    h2 = s // 2
    x0, y0, x1, y1 = x - h2, y - h2, x - h2 + s, y - h2 + s
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    bgr = col[y0:y1, x0:x1]
    if bgr.shape[0] != s or bgr.shape[1] != s:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    z = (g - g.mean()) / (g.std() + 1e-6)
    ring, core, sect = get_masks(s)
    ring_z = float(z[ring].mean())
    core_z = float(z[core].mean())
    sm = []
    for k in range(16):
        m = ring & (sect == k)
        if m.sum() > 0:
            sm.append(float(z[m].mean()))
    ring_min = float(np.min(sm)) if sm else -9.0
    return dict(ring_z=ring_z, core_z=core_z, contrast=ring_z - core_z,
                ring_min=ring_min,
                ring_sat=float(hsv[..., 1][ring].mean()),
                ring_val=float(hsv[..., 2][ring].mean()),
                core_sat=float(hsv[..., 1][core].mean()))


if __name__ == "__main__":
    print("=" * 74)
    print("GT missing_hole RING FEATURE DISTRIBUTION (mh_board train)")
    print("=" * 74)
    rows = []
    for ip in sorted(glob.glob(os.path.join(DS, "train", "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(DS, "train", "labels", stem + ".txt")
        col = cv2.imread(ip)
        H, W = col.shape[:2]
        for line in open(lp):
            f = line.split()
            if len(f) != 5:
                continue
            cx, cy, w, h = [float(v) for v in f[1:]]
            cx *= W; cy *= H; w *= W; h *= H
            s = int(round(max(w, h)))
            ft = features(col, int(round(cx)), int(round(cy)), s)
            if ft:
                rows.append(ft)
    keys = ["ring_z", "ring_min", "core_z", "contrast", "ring_sat", "ring_val", "core_sat"]
    A = {k: np.array([r[k] for r in rows]) for k in keys}
    print("n = %d GT sites" % len(rows))
    print("%-10s %8s %8s %8s %8s %8s %8s" % ("feature", "p1", "p5", "p25", "p50", "p95", "p99"))
    for k in keys:
        a = A[k]
        print("%-10s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (k, *[np.percentile(a, q) for q in (1, 5, 25, 50, 95, 99)]))
    json.dump({k: dict(p1=float(np.percentile(A[k], 1)), p5=float(np.percentile(A[k], 5)),
                       p50=float(np.percentile(A[k], 50)), p95=float(np.percentile(A[k], 95)),
                       p99=float(np.percentile(A[k], 99))) for k in keys},
              open(os.path.join(OUT, "gt_ring_features.json"), "w"), indent=2)
    print("")
    print("saved gt_ring_features.json")
