"""
Build corrected / re-split PCB datasets.

1) pcb-defect-dataset-fixed   : 6-class, original split, label filename bug fixed
                                (labels *_256.txt -> *_600.txt), images hardlinked.
2) datasets/mh_orig           : missing_hole-only, ORIGINAL split assignment, class remapped to 0
3) datasets/mh_board          : missing_hole-only, BOARD-DISJOINT split, class remapped to 0

Originals are never modified.
"""
import os, re, glob, json, shutil, collections
import numpy as np

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

SRC = PCB_ROOT + "/pcb-defect-dataset"
FIX = PCB_ROOT + "/pcb-defect-dataset-fixed"
DS = PCB_ROOT + "/datasets"
MH = 2
SPLITS = ["train", "val", "test"]
NAMES6 = ["mouse_bite", "spur", "missing_hole", "short", "open_circuit", "spurious_copper"]

NAME_RE = re.compile(r"^(rotation_(?P<deg>\d+)_)?(?P<flip>l_)?light_(?P<light>\d+)_missing_hole_(?P<board>\d+)_(?P<tile>\d+)_600$")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def ensure(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------- 1) fixed
print("=" * 70)
print("1) BUILD pcb-defect-dataset-fixed  (label filename bug fix)")
print("=" * 70)
report = {}
for sp in SPLITS:
    di, dl = os.path.join(FIX, sp, "images"), os.path.join(FIX, sp, "labels")
    ensure(di, dl)
    imgs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(SRC, sp, "images", "*.jpg"))}
    lbls = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(SRC, sp, "labels", "*.txt"))}
    direct = renamed = missing = 0
    for stem, ip in imgs.items():
        link(ip, os.path.join(di, stem + ".jpg"))
        if stem in lbls:
            lp = lbls[stem]; direct += 1
        else:
            alt = stem.replace("_600", "_256")
            if alt in lbls:
                lp = lbls[alt]; renamed += 1
            else:
                open(os.path.join(dl, stem + ".txt"), "w").close()
                missing += 1
                continue
        link(lp, os.path.join(dl, stem + ".txt"))
    report[sp] = dict(images=len(imgs), direct=direct, renamed=renamed, still_missing=missing)
    print("  %-5s images=%4d  label_direct=%4d  label_renamed(_256->_600)=%4d  still_missing=%d"
          % (sp, len(imgs), direct, renamed, missing))

with open(os.path.join(FIX, "data.yaml"), "w") as f:
    f.write("path: %s\ntrain: train/images\nval: val/images\ntest: test/images\n\nnc: 6\nnames:\n" % FIX)
    for i, n in enumerate(NAMES6):
        f.write("  %d: %s\n" % (i, n))

# verify
print("  verify:")
for sp in SPLITS:
    I = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(FIX, sp, "images", "*.jpg"))}
    L = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(FIX, sp, "labels", "*.txt"))}
    n2 = 0
    for p in glob.glob(os.path.join(FIX, sp, "labels", "*.txt")):
        for line in open(p):
            f_ = line.split()
            if len(f_) == 5 and int(f_[0]) == MH:
                n2 += 1
    print("    %-5s img=%d lbl=%d unpaired=%d  missing_hole_boxes=%d" % (sp, len(I), len(L), len(I ^ L), n2))

# ---------------------------------------------------------------- gather mh
print("")
print("=" * 70)
print("2) MISSING_HOLE-ONLY SUBSET")
print("=" * 70)
recs = []  # (orig_split, stem, board, light, tile, rot, flip, nbox)
for sp in SPLITS:
    for p in sorted(glob.glob(os.path.join(FIX, sp, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if "missing_hole" not in stem:
            continue
        m = NAME_RE.match(stem)
        assert m, stem
        lp = os.path.join(FIX, sp, "labels", stem + ".txt")
        lines = [l.split() for l in open(lp)]
        boxes = [l for l in lines if len(l) == 5 and int(l[0]) == MH]
        other = [l for l in lines if len(l) == 5 and int(l[0]) != MH]
        recs.append(dict(split=sp, stem=stem, board=int(m.group("board")),
                         light=int(m.group("light")), tile=int(m.group("tile")),
                         rot=int(m.group("deg")) if m.group("deg") else 0,
                         flip=bool(m.group("flip")), nbox=len(boxes), nother=len(other)))
print("  missing_hole images=%d  boxes=%d  non-mh boxes inside them=%d"
      % (len(recs), sum(r["nbox"] for r in recs), sum(r["nother"] for r in recs)))
print("  per original split:", {sp: sum(1 for r in recs if r["split"] == sp) for sp in SPLITS})
print("  boxes per original split:", {sp: sum(r["nbox"] for r in recs if r["split"] == sp) for sp in SPLITS})

bybrd = collections.Counter()
bybrd_box = collections.Counter()
for r in recs:
    bybrd[r["board"]] += 1
    bybrd_box[r["board"]] += r["nbox"]
print("  per-board images:", dict(sorted(bybrd.items())))
print("  per-board boxes :", dict(sorted(bybrd_box.items())))


def write_subset(name, assign):
    """assign: stem -> split"""
    root = os.path.join(DS, name)
    for sp in SPLITS:
        ensure(os.path.join(root, sp, "images"), os.path.join(root, sp, "labels"))
    cnt = collections.Counter()
    box = collections.Counter()
    for r in recs:
        tgt = assign[r["stem"]]
        src_i = os.path.join(FIX, r["split"], "images", r["stem"] + ".jpg")
        src_l = os.path.join(FIX, r["split"], "labels", r["stem"] + ".txt")
        link(src_i, os.path.join(root, tgt, "images", r["stem"] + ".jpg"))
        out = []
        for line in open(src_l):
            f_ = line.split()
            if len(f_) == 5 and int(f_[0]) == MH:
                out.append("0 " + " ".join(f_[1:]))
        with open(os.path.join(root, tgt, "labels", r["stem"] + ".txt"), "w") as fo:
            fo.write("\n".join(out) + ("\n" if out else ""))
        cnt[tgt] += 1
        box[tgt] += len(out)
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: train/images\nval: val/images\ntest: test/images\n\nnc: 1\nnames:\n  0: missing_hole\n"
                % root.replace("\\", "/"))
    print("  [%s] images=%s  boxes=%s" % (name, dict(cnt), dict(box)))
    return cnt, box


# --- mh_orig : keep original split
write_subset("mh_orig", {r["stem"]: r["split"] for r in recs})

# --- mh_board : board-disjoint, target 70/15/15 by box count
boards = sorted(bybrd_box)
tot = sum(bybrd_box.values())
target = {"train": .70 * tot, "val": .15 * tot, "test": .15 * tot}
order = sorted(boards, key=lambda b: -bybrd_box[b])
acc = {"train": 0, "val": 0, "test": 0}
bsplit = {}
for b in order:
    # assign to the split furthest below its target (proportionally)
    sp = max(SPLITS, key=lambda s: (target[s] - acc[s]) / target[s])
    bsplit[b] = sp
    acc[sp] += bybrd_box[b]
print("")
print("  board -> split:", {b: bsplit[b] for b in boards})
print("  boards per split:", {sp: sorted(b for b in boards if bsplit[b] == sp) for sp in SPLITS})
write_subset("mh_board", {r["stem"]: bsplit[r["board"]] for r in recs})

json.dump({"board_split": {str(k): v for k, v in bsplit.items()},
           "fix_report": report},
          open(os.path.join(DS, "split_info.json"), "w"), indent=2)

# --- leakage re-audit on both
print("")
print("  LEAKAGE RE-AUDIT (test keys also present in train):")
for name, assign in [("mh_orig", {r["stem"]: r["split"] for r in recs}),
                     ("mh_board", {r["stem"]: bsplit[r["board"]] for r in recs})]:
    for fields in [("board",), ("board", "light", "tile")]:
        tr = set(tuple(r[f] for f in fields) for r in recs if assign[r["stem"]] == "train")
        te = set(tuple(r[f] for f in fields) for r in recs if assign[r["stem"]] == "test")
        print("    %-9s key=%-22s test_seen_in_train=%d/%d" % (name, "+".join(fields), len(te & tr), len(te)))
print("")
print("done. roots:")
print("  " + FIX)
print("  " + os.path.join(DS, "mh_orig"))
print("  " + os.path.join(DS, "mh_board"))
