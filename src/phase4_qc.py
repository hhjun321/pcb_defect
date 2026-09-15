"""
Phase 4 QC - does a synthesised missing_hole look like a real one?

Visual inspection is not enough: the failure modes that matter (core too small,
core the wrong colour, halo at the feather edge) all show up as a mismatch in
the RADIAL PROFILE of saturation and value around the box centre.  Real
missing_holes have a very stable profile:

    sat  231 at the centre (green substrate)  ->  48 at r/s 0.24-0.32 (metal)
    val  116 at the centre                    -> 155 at the ring

So the profile of the inserted boxes is compared against the profile of the
real ones, on the same images, and summarised as a mean absolute deviation.
Prints a table and writes qc_profile_<dataset>.png.

usage: python src/phase4_qc.py mh_board_brightspeccp [mh_board]
"""
import os, sys, glob
import numpy as np
import cv2
import matplotlib

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DSDIR = PCB_ROOT + "/datasets"
OUT = PCB_ROOT + "/outputs/phase4"
BINS = np.arange(0, 0.52, 0.04)


def profile(col, cx, cy, s):
    H, W = col.shape[:2]
    h2 = s // 2
    x0, y0 = int(round(cx)) - h2, int(round(cy)) - h2
    if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H or s < 10:
        return None
    hsv = cv2.cvtColor(col[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2HSV).astype(np.float32)
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / s
    S, V = hsv[..., 1], hsv[..., 2]
    ps, pv = [], []
    for i in range(len(BINS) - 1):
        m = (r >= BINS[i]) & (r < BINS[i + 1])
        ps.append(S[m].mean() if m.any() else np.nan)
        pv.append(V[m].mean() if m.any() else np.nan)
    return ps, pv


def boxes_of(lp, W, H):
    out = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5:
            cx, cy, w, h = [float(v) for v in f[1:]]
            out.append((cx * W, cy * H, w * W, h * H))
    return out


def collect(synth, base, limit=600):
    """returns (real_profiles, synth_profiles) - synth = boxes not present in base"""
    R, Sy = ([], []), ([], [])
    n = 0
    for ip in sorted(glob.glob(os.path.join(DSDIR, synth, "train", "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp_s = os.path.join(DSDIR, synth, "train", "labels", stem + ".txt")
        lp_b = os.path.join(DSDIR, base, "train", "labels", stem + ".txt")
        if not os.path.exists(lp_b):
            continue
        col = cv2.imread(ip)
        colb = cv2.imread(os.path.join(DSDIR, base, "train", "images", stem + ".jpg"))
        H, W = col.shape[:2]
        bb = boxes_of(lp_b, W, H)
        ss = boxes_of(lp_s, W, H)
        key = lambda b: (round(b[0]), round(b[1]))
        base_keys = {key(b) for b in bb}
        for b in bb:
            p = profile(colb, b[0], b[1], int(round(max(b[2], b[3]))))
            if p:
                R[0].append(p[0]); R[1].append(p[1])
        for b in ss:
            if key(b) in base_keys:
                continue
            p = profile(col, b[0], b[1], int(round(max(b[2], b[3]))))
            if p:
                Sy[0].append(p[0]); Sy[1].append(p[1]); n += 1
        if n >= limit:
            break
    return R, Sy


def main(synth, base):
    R, Sy = collect(synth, base)
    rs = np.nanmedian(np.array(R[0], dtype=float), 0)
    rv = np.nanmedian(np.array(R[1], dtype=float), 0)
    ss = np.nanmedian(np.array(Sy[0], dtype=float), 0)
    sv = np.nanmedian(np.array(Sy[1], dtype=float), 0)
    print("=" * 70)
    print("RADIAL PROFILE  real (n=%d)  vs  synthetic (n=%d)   [%s]"
          % (len(R[0]), len(Sy[0]), synth))
    print("=" * 70)
    print("%-11s %17s %17s" % ("", "saturation", "value"))
    print("%-11s %8s %8s %8s %8s" % ("r/s", "real", "synth", "real", "synth"))
    for i in range(len(rs)):
        print("%.2f-%.2f %8.1f %8.1f %8.1f %8.1f" % (BINS[i], BINS[i + 1], rs[i], ss[i], rv[i], sv[i]))
    mad_s = np.nanmean(np.abs(rs - ss))
    mad_v = np.nanmean(np.abs(rv - sv))
    print("")
    print("mean |real - synth| :  saturation %.1f   value %.1f   (0-255 scale)" % (mad_s, mad_v))
    core = slice(0, 4)   # r/s < 0.16
    print("core only (r/s<0.16):  saturation %.1f   value %.1f"
          % (np.nanmean(np.abs(rs[core] - ss[core])), np.nanmean(np.abs(rv[core] - sv[core]))))

    x = (BINS[:-1] + BINS[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, rs, "o-", label="real"); ax[0].plot(x, ss, "s--", label="synthetic")
    ax[0].set_xlabel("r / box side"); ax[0].set_ylabel("saturation"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(x, rv, "o-", label="real"); ax[1].plot(x, sv, "s--", label="synthetic")
    ax[1].set_xlabel("r / box side"); ax[1].set_ylabel("value"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle("%s  |  MAD sat %.1f, val %.1f" % (synth, mad_s, mad_v))
    plt.tight_layout()
    fn = os.path.join(OUT, "qc_profile_%s.png" % synth)
    plt.savefig(fn, dpi=130)
    plt.close()
    print("saved", fn)
    return mad_s, mad_v


if __name__ == "__main__":
    synth = sys.argv[1] if len(sys.argv) > 1 else "mh_board_brightspeccp"
    base = sys.argv[2] if len(sys.argv) > 2 else "mh_board"
    main(synth, base)
