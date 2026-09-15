"""
Phase 2 - Brightness-based candidate search + RQ2 evaluation.

Template  : D_mean built from mh_board TRAIN crops only (no peeking at val/test).
Search    : multi-scale normalized cross-correlation (TM_CCOEFF_NORMED == per-window
            z-normalized matching, exactly the lighting-invariant MSE we argued for).
Eval      : on mh_board VAL images -> recall@K of true missing_hole among top-K peaks,
            plus a HARD-NEGATIVE check (top peaks that are not GT = normal pads?).
"""
import os, glob, json, collections
import numpy as np
import cv2
import matplotlib

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase2"
os.makedirs(OUT, exist_ok=True)
S = 32
SCALES = [20, 24, 28, 32, 36, 40, 48]
TOPK_REPORT = [1, 3, 5, 10, 20, 50, 100]
rng = np.random.default_rng(0)


def load_split(sp):
    out = []
    for ip in sorted(glob.glob(os.path.join(DS, sp, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(DS, sp, "labels", stem + ".txt")
        boxes = []
        for line in open(lp):
            f = line.split()
            if len(f) == 5:
                boxes.append([float(v) for v in f[1:]])
        out.append((stem, ip, boxes))
    return out


# ---------------- template from TRAIN ----------------
print("=" * 70)
print("TEMPLATE (mh_board train)")
print("=" * 70)
train = load_split("train")
crops = []
for stem, ip, boxes in train:
    A = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    H, W = A.shape
    for cx, cy, w, h in boxes:
        cx *= W; cy *= H; w *= W; h *= H
        x0, y0 = max(0, int(round(cx - w / 2))), max(0, int(round(cy - h / 2)))
        x1, y1 = min(W, int(round(cx + w / 2))), min(H, int(round(cy + h / 2)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crops.append(cv2.resize(A[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_LINEAR))
C = np.stack(crops)
Cz = (C - C.mean((1, 2), keepdims=True)) / (C.std((1, 2), keepdims=True) + 1e-6)
Dz = Cz.mean(0).astype(np.float32)
np.save(os.path.join(OUT, "Dz_mean_boardtrain.npy"), Dz)
print("train images=%d  crops=%d" % (len(train), len(C)))
print("Dz center=%.2f  ring(r=9)=%.2f  min=%.2f max=%.2f" % (Dz[S // 2, S // 2], Dz[S // 2, S // 2 + 9], Dz.min(), Dz.max()))

TMPL = {s: cv2.resize(Dz, (s, s), interpolation=cv2.INTER_LINEAR) for s in SCALES}


# ---------------- search ----------------
def search(img, topn=200):
    """multi-scale NCC -> NMS peaks. returns list of (score, cx, cy, size)."""
    H, W = img.shape
    best = np.full((H, W), -2.0, np.float32)
    bests = np.zeros((H, W), np.int32)
    for s in SCALES:
        r = cv2.matchTemplate(img, TMPL[s], cv2.TM_CCOEFF_NORMED)  # (H-s+1, W-s+1)
        full = np.full((H, W), -2.0, np.float32)
        off = s // 2
        full[off:off + r.shape[0], off:off + r.shape[1]] = r
        m = full > best
        best[m] = full[m]
        bests[m] = s
    # NMS via greedy on sorted scores with suppression radius
    flat = best.ravel()
    idx = np.argpartition(flat, -topn * 40)[-topn * 40:]
    idx = idx[np.argsort(-flat[idx])]
    taken = np.zeros((H, W), bool)
    res = []
    for i in idx:
        y, x = divmod(int(i), W)
        if taken[y, x]:
            continue
        s = int(bests[y, x])
        rad = max(4, s // 2)
        taken[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = True
        res.append((float(best[y, x]), x, y, s))
        if len(res) >= topn:
            break
    return res


def gt_boxes_px(boxes, W, H):
    out = []
    for cx, cy, w, h in boxes:
        out.append((cx * W, cy * H, w * W, h * H))
    return out


def hit(cand, gts):
    """candidate center inside GT box AND scale within 2x"""
    _, x, y, s = cand
    for (cx, cy, w, h) in gts:
        if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2:
            if 0.5 <= s / max(w, h) <= 2.0:
                return True
    return False


print("")
print("=" * 70)
print("SEARCH on mh_board VAL")
print("=" * 70)
val = load_split("val")
n_img = len(val)
recall_hits = collections.Counter()
n_gt_total = 0
rank_of_gt = []
pos_scores, neg_scores = [], []
neg_patches, pos_patches = [], []
per_img = []
for k, (stem, ip, boxes) in enumerate(val):
    A = cv2.imread(ip, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    H, W = A.shape
    gts = gt_boxes_px(boxes, W, H)
    cands = search(A, topn=200)
    matched = set()
    for rank, c in enumerate(cands):
        _, x, y, s = c
        for gi, (cx, cy, w, h) in enumerate(gts):
            if gi in matched:
                continue
            if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 and 0.5 <= s / max(w, h) <= 2.0:
                matched.add(gi)
                rank_of_gt.append(rank + 1)
                pos_scores.append(c[0])
                if len(pos_patches) < 64:
                    h2 = s // 2
                    p = A[max(0, y - h2):y + h2, max(0, x - h2):x + h2]
                    if p.size:
                        pos_patches.append(cv2.resize(p, (S, S)))
                break
    for K in TOPK_REPORT:
        cnt = 0
        m2 = set()
        for rank, c in enumerate(cands[:K]):
            _, x, y, s = c
            for gi, (cx, cy, w, h) in enumerate(gts):
                if gi in m2:
                    continue
                if abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 and 0.5 <= s / max(w, h) <= 2.0:
                    m2.add(gi); cnt += 1
                    break
        recall_hits[K] += cnt
    n_gt_total += len(gts)
    # hard negatives = top peaks that are NOT gt
    nneg = 0
    for c in cands[:20]:
        if not hit(c, gts):
            neg_scores.append(c[0])
            nneg += 1
            if len(neg_patches) < 64 and rng.random() < 0.25:
                _, x, y, s = c
                h2 = s // 2
                p = A[max(0, y - h2):y + h2, max(0, x - h2):x + h2]
                if p.size:
                    neg_patches.append(cv2.resize(p, (S, S)))
    per_img.append(dict(stem=stem, n_gt=len(gts), n_found=len(matched)))
    if (k + 1) % 25 == 0:
        print("  %d/%d images..." % (k + 1, n_img))

print("")
print("val images=%d  GT missing_hole=%d" % (n_img, n_gt_total))
print("")
print("RECALL@K  (GT found among top-K brightness candidates per image)")
for K in TOPK_REPORT:
    print("   K=%3d -> %5.1f%%  (%d/%d)" % (K, 100 * recall_hits[K] / n_gt_total, recall_hits[K], n_gt_total))
rk = np.array(rank_of_gt)
print("")
print("rank of GT among candidates: median=%.0f  p25=%.0f p75=%.0f p90=%.0f  (found %d/%d)"
      % (np.median(rk), np.percentile(rk, 25), np.percentile(rk, 75), np.percentile(rk, 90), len(rk), n_gt_total))

ps, ns = np.array(pos_scores), np.array(neg_scores)
print("")
print("=" * 70)
print("HARD NEGATIVE CHECK  (top-20 peaks that are NOT missing_hole)")
print("=" * 70)
print("NCC score  GT missing_hole : mean=%.3f median=%.3f p10=%.3f" % (ps.mean(), np.median(ps), np.percentile(ps, 10)))
print("NCC score  hard negatives  : mean=%.3f median=%.3f p90=%.3f  (n=%d)" % (ns.mean(), np.median(ns), np.percentile(ns, 90), len(ns)))
allv = np.concatenate([ps, ns]); lab = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
auc = (ranks[lab == 1].sum() - len(ps) * (len(ps) + 1) / 2) / (len(ps) * len(ns))
print("AUC (higher NCC => missing_hole) = %.3f    [0.5 = brightness alone CANNOT separate]" % auc)
print("-> low AUC here is EXPECTED and is the justification for the Context stage (Step 6).")

# figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(Dz, cmap="RdBu_r"); ax[0].set_title("Dz_mean (board-train)")
ax[1].hist(ps, bins=50, alpha=.6, density=True, label="GT missing_hole")
ax[1].hist(ns, bins=50, alpha=.6, density=True, label="hard negative (top peak, not GT)")
ax[1].set_xlabel("NCC score"); ax[1].legend(); ax[1].set_title("AUC=%.3f" % auc)
ks = TOPK_REPORT
ax[2].plot(ks, [100 * recall_hits[K] / n_gt_total for K in ks], "o-")
ax[2].set_xscale("log"); ax[2].set_xlabel("K"); ax[2].set_ylabel("recall %")
ax[2].set_title("recall@K of GT among brightness candidates"); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "search_summary.png"), dpi=130); plt.close()


def montage(patches, title, fn):
    if not patches:
        return
    n = min(64, len(patches))
    g = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(g, g, figsize=(8, 8))
    for a in np.ravel(axes):
        a.axis("off")
    for a, p in zip(np.ravel(axes), patches[:n]):
        a.imshow(p, cmap="gray"); a.axis("off")
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fn), dpi=130); plt.close()


montage(pos_patches, "TRUE missing_hole found by brightness search", "pos_patches.png")
montage(neg_patches, "HARD NEGATIVES: top brightness peaks that are NOT missing_hole", "neg_patches.png")

json.dump(dict(recall={str(K): recall_hits[K] / n_gt_total for K in TOPK_REPORT},
               n_gt=n_gt_total, n_img=n_img, auc_hardneg=float(auc)),
          open(os.path.join(OUT, "search_metrics.json"), "w"), indent=2)
print("")
print("figures -> %s" % OUT)
