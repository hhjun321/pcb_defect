"""
Calibrate the two synthesis constants from real missing_hole sites.

QC of the first core-transplant build showed two systematic mismatches:

  1. the transplanted core was too bright and too saturated.  The colour was
     sampled from the solder resist OUTSIDE the pad, but the real core sits in
     the pad opening and is a deeper, darker green.  Fixed by measuring the
     real core-to-surrounding-resist colour transfer and applying it.

  2. the synthesised boxes were tighter than real GT boxes: the box side came
     straight from the discrete template scale {20,24,...,48} rather than from
     the pad.  Fixed by measuring where the metal ring ends in a real GT box,
     which gives the box side as a multiple of the pad's outer radius.

Writes outputs/phase4/core_calibration.json
"""
import os, glob, json
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase4"
os.makedirs(OUT, exist_ok=True)

CORE_SAMPLE_R = 0.10     # solid green core
RESIST_LO, RESIST_HI = 0.40, 0.48   # annulus of a 1.9*s window = outside the pad


def radial(s):
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2.0
    return np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / s


ratios, offsets = [], []
pad_edges = []
n = 0
for ip in sorted(glob.glob(os.path.join(DS, "train", "images", "*.jpg")))[::2]:
    stem = os.path.splitext(os.path.basename(ip))[0]
    col = cv2.imread(ip)
    H, W = col.shape[:2]
    for line in open(os.path.join(DS, "train", "labels", stem + ".txt")):
        f = line.split()
        if len(f) != 5:
            continue
        cx, cy, w, h = [float(v) for v in f[1:]]
        cx *= W; cy *= H; w *= W; h *= H
        s = int(round(max(w, h)))
        h2 = s // 2
        x0, y0 = int(round(cx)) - h2, int(round(cy)) - h2
        big = int(round(1.9 * s))
        bx, by = int(round(cx)) - big // 2, int(round(cy)) - big // 2
        if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
            continue
        if bx < 0 or by < 0 or bx + big > W or by + big > H:
            continue
        patch = col[y0:y0 + s, x0:x0 + s].astype(np.float32)
        around = col[by:by + big, bx:bx + big].astype(np.float32)
        rs, rb = radial(s), radial(big)
        core = patch[rs <= CORE_SAMPLE_R].reshape(-1, 3)
        resist = around[(rb >= RESIST_LO) & (rb <= RESIST_HI)].reshape(-1, 3)
        if len(core) < 5 or len(resist) < 20:
            continue
        cmu = np.median(core, 0)
        rmu = np.median(resist, 0)
        ratios.append(cmu / np.maximum(rmu, 1.0))
        offsets.append(cmu - rmu)

        # where does the metal ring end?  saturation rises back towards resist
        hsv = cv2.cvtColor(col[by:by + big, bx:bx + big], cv2.COLOR_BGR2HSV).astype(np.float32)
        S = hsv[..., 1]
        bins = np.arange(0.05, 0.50, 0.02)
        prof = np.array([S[(rb >= b) & (rb < b + 0.02)].mean() for b in bins])
        i_min = int(np.argmin(prof))
        lo, hi = prof[i_min], prof[-1]
        thr = (lo + hi) / 2.0
        j = i_min
        while j < len(prof) - 1 and prof[j] < thr:
            j += 1
        # radius in units of the GT box side s  (window is 1.9*s wide)
        pad_edges.append(float(bins[j] * 1.9))
        n += 1
    if n >= 900:
        break

ratios = np.array(ratios)
offsets = np.array(offsets)
pe = np.array(pad_edges)
R = np.median(ratios, 0)
O = np.median(offsets, 0)
print("=" * 70)
print("CORE COLOUR TRANSFER  (real core vs resist just outside the pad, n=%d)" % len(ratios))
print("=" * 70)
print("   per-channel ratio  core/resist  B=%.3f G=%.3f R=%.3f" % tuple(R))
print("   per-channel offset core-resist  B=%+.1f G=%+.1f R=%+.1f" % tuple(O))
print("")
print("=" * 70)
print("PAD OUTER EDGE inside a real GT box (n=%d)" % len(pe))
print("=" * 70)
print("   r_edge / box_side : p25=%.3f  median=%.3f  p75=%.3f" %
      (np.percentile(pe, 25), np.median(pe), np.percentile(pe, 75)))
print("   => box_side = pad_outer_radius / %.3f" % np.median(pe))

json.dump(dict(core_ratio=R.tolist(), core_offset=O.tolist(),
               pad_edge_frac=float(np.median(pe)), n=len(ratios)),
          open(os.path.join(OUT, "core_calibration.json"), "w"), indent=2)
print("")
print("saved core_calibration.json")
