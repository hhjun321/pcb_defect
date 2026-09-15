"""
Final integrity check of every released dataset before hand-off to the cloud.

Checks, per dataset:
  - every image has a label file and vice versa
  - every box is inside [0,1] and has positive size
  - class id is always 0
  - val/test are byte-identical to the baseline split (only train may differ)
  - train box count == baseline box count + boxes recorded in synthesis_meta.json
"""
import os, glob, json, hashlib
import numpy as np

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

DSDIR = PCB_ROOT + "/datasets"
BASES = ["mh_board", "mh_orig"]
ARMS = ["randomcp", "brightnesscp", "contextcp", "brightspeccp"]


def scan(ds, sp):
    idir = os.path.join(DSDIR, ds, sp, "images")
    ldir = os.path.join(DSDIR, ds, sp, "labels")
    I = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(idir, "*.jpg"))}
    L = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ldir, "*.txt"))}
    nbox = 0
    bad_range = bad_cls = 0
    for stem in L:
        for line in open(os.path.join(ldir, stem + ".txt")):
            f = line.split()
            if len(f) != 5:
                continue
            nbox += 1
            c = int(f[0])
            cx, cy, w, h = [float(v) for v in f[1:]]
            if c != 0:
                bad_cls += 1
            if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx - w / 2 and cx + w / 2 <= 1
                    and 0 <= cy - h / 2 and cy + h / 2 <= 1):
                bad_range += 1
    return dict(imgs=len(I), lbls=len(L), unpaired=len(I ^ L), boxes=nbox,
                bad_range=bad_range, bad_cls=bad_cls, stems=L)


def dirhash(ds, sp):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(DSDIR, ds, sp, "labels", "*.txt"))):
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


print("=" * 96)
print("%-26s %7s %7s %9s %8s %8s %8s" % ("dataset / split", "images", "labels", "unpaired", "boxes", "bad_rng", "bad_cls"))
print("=" * 96)
ok = True
summary = []
for base in BASES:
    base_counts = {sp: scan(base, sp) for sp in ["train", "val", "test"]}
    base_hash = {sp: dirhash(base, sp) for sp in ["val", "test"]}
    for ds in [base] + ["%s_%s" % (base, a) for a in ARMS]:
        meta = None
        mp = os.path.join(DSDIR, ds, "synthesis_meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp))
        for sp in ["train", "val", "test"]:
            r = scan(ds, sp)
            flag = ""
            if r["unpaired"] or r["bad_range"] or r["bad_cls"]:
                flag = "  <-- PROBLEM"
                ok = False
            print("%-26s %7d %7d %9d %8d %8d %8d%s"
                  % (ds + "/" + sp, r["imgs"], r["lbls"], r["unpaired"], r["boxes"],
                     r["bad_range"], r["bad_cls"], flag))
            if sp in ("val", "test"):
                h = dirhash(ds, sp)
                if h != base_hash[sp]:
                    print("     ^ val/test labels DIFFER from %s (%s vs %s)" % (base, h, base_hash[sp]))
                    ok = False
            if sp == "train" and meta:
                expect = base_counts["train"]["boxes"] + meta["boxes_added"]
                if r["boxes"] != expect:
                    print("     ^ train box count %d != baseline %d + added %d"
                          % (r["boxes"], base_counts["train"]["boxes"], meta["boxes_added"]))
                    ok = False
                summary.append((ds, base_counts["train"]["boxes"], meta["boxes_added"], r["boxes"],
                                meta["mean_size_added"]))
        print("-" * 96)

print("")
print("=" * 96)
print("TRAIN BOX BUDGET")
print("=" * 96)
print("%-26s %10s %10s %10s %12s" % ("dataset", "real", "synthetic", "total", "mean size px"))
for ds, b, a, t, ms in summary:
    print("%-26s %10d %10d %10d %12.1f" % (ds, b, a, t, ms))
print("")
print("ALL CHECKS PASSED" if ok else "*** PROBLEMS FOUND ***")
