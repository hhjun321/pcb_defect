"""
Phase 3 - Context Template Matching (research flow Step 6 / Step 7).

The brightness stage ranks a location by the defect's OWN appearance.  It
cannot tell a real pad from a silkscreen glyph or an inter-pad gap, because
those mimic the same dark-core / bright-ring signature.  Measured ceiling:
AUC(GT vs non-GT brightness peak) = 0.703.

This stage compares the SURROUNDING PCB STRUCTURE instead.  The defect itself
is masked out, exactly as the research flow specifies, so the two stages use
disjoint information:

    brightness stage : r < R_IN   (the defect)      -> candidate search
    context stage    : R_IN <= r <= R_OUT           -> candidate verification

Two context models are built and compared:

  single : one mean context template (the literal reading of the flow doc)
  kmeans : K cluster centres, score = max similarity over centres.  PCB context
           is genuinely multi-modal (pad array / trace field / blank resist), so
           a single mean blurs to nothing; this is the proposed improvement.

Scoring is zero-mean unit-norm correlation over the annulus mask only, so it is
invariant to lighting exactly like the brightness stage.

Outputs (outputs/phase3/):
  C_single.npy, C_kmeans.npy, context_metrics.json, context_summary.png,
  context_templates.png, rejected_by_context.png
"""
import os, sys, glob, json, collections
import numpy as np
import cv2
import matplotlib

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candgen as CG

DS = PCB_ROOT + "/datasets/mh_board"
OUT = PCB_ROOT + "/outputs/phase3"
P2 = PCB_ROOT + "/outputs/phase2"
os.makedirs(OUT, exist_ok=True)

CTX = 96          # context patch is resampled to CTX x CTX
CTX_SPAN = 3.0    # patch covers CTX_SPAN * s around the centre
R_IN = 0.18       # inner radius (fraction of CTX) - everything inside is the
                  # defect + its own ring, excluded from context
R_OUT = 0.50
K_CLUSTERS = 12
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- masks
def _ctx_mask():
    yy, xx = np.mgrid[0:CTX, 0:CTX]
    c = (CTX - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / CTX
    return (r >= R_IN) & (r <= R_OUT)


MASK = _ctx_mask()
NMASK = int(MASK.sum())


def context_patch(gray, x, y, s):
    """Annulus-normalised context descriptor at (x,y) for box side s.
    Returns a flat unit-norm zero-mean vector over the annulus, or None."""
    half = int(round(CTX_SPAN * s / 2))
    H, W = gray.shape
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    p = cv2.resize(gray[y0:y1, x0:x1], (CTX, CTX), interpolation=cv2.INTER_AREA)
    v = p[MASK].astype(np.float32)
    v -= v.mean()
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


def to_image(vec):
    """put a flat annulus vector back into a CTX x CTX image for plotting"""
    img = np.full((CTX, CTX), np.nan, np.float32)
    img[MASK] = vec
    return img


# ---------------------------------------------------------------- data
def load_split(sp):
    out = []
    for ip in sorted(glob.glob(os.path.join(DS, sp, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        out.append((stem, ip, os.path.join(DS, sp, "labels", stem + ".txt")))
    return out


def gts_of(lp, W, H):
    g = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5:
            cx, cy, w, h = [float(v) for v in f[1:]]
            g.append((cx * W, cy * H, w * W, h * H))
    return g


# ---------------------------------------------------------------- build
def build_templates():
    print("=" * 74)
    print("BUILD CONTEXT TEMPLATES from mh_board train GT surroundings")
    print("=" * 74)
    V = []
    for stem, ip, lp in load_split("train"):
        col = cv2.imread(ip)
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape
        for cx, cy, w, h in gts_of(lp, W, H):
            v = context_patch(gray, int(round(cx)), int(round(cy)), int(round(max(w, h))))
            if v is not None:
                V.append(v)
    V = np.stack(V)
    print("context descriptors: %s  (annulus pixels per descriptor: %d)" % (V.shape, NMASK))

    single = V.mean(0)
    single /= np.linalg.norm(single) + 1e-9

    # k-means (cosine == euclidean on unit vectors)
    Z = np.float32(V)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, lab, cen = cv2.kmeans(Z, K_CLUSTERS, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    lab = lab.ravel()
    sizes = collections.Counter(lab.tolist())

    # how well does each model describe a held-out GT context?
    sim_single = V @ single
    sim_kmeans = (V @ cen.T).max(1)
    print("self-similarity of GT contexts:")
    print("   single mean : median=%.3f  p25=%.3f p75=%.3f" %
          (np.median(sim_single), np.percentile(sim_single, 25), np.percentile(sim_single, 75)))
    print("   kmeans K=%-2d : median=%.3f  p25=%.3f p75=%.3f" %
          (K_CLUSTERS, np.median(sim_kmeans), np.percentile(sim_kmeans, 25), np.percentile(sim_kmeans, 75)))
    print("   cluster sizes:", dict(sorted(sizes.items())))
    np.save(os.path.join(OUT, "C_single.npy"), single)
    np.save(os.path.join(OUT, "C_kmeans.npy"), cen)
    return single, cen


# ---------------------------------------------------------------- evaluate
def auc(pos, neg):
    a = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(a)
    rk = np.empty(len(a))
    rk[o] = np.arange(1, len(a) + 1)
    return float((rk[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(single, cen):
    print("")
    print("=" * 74)
    print("EVALUATE on mh_board val - does context separate GT from FP peaks?")
    print("=" * 74)
    Dz = np.load(os.path.join(P2, "Dz_mean_boardtrain.npy"))
    TM = CG.make_templates(Dz)
    rec = []          # (is_gt, ncc, ctx_single, ctx_kmeans, stem, x, y, s)
    for stem, ip, lp in load_split("val"):
        col = cv2.imread(ip)
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape
        gts = gts_of(lp, W, H)
        cands = CG.detect(col, TM, topn=20, use_ring_gate=True)   # stage A only
        for sc, x, y, s, ft in cands:
            v = context_patch(gray, x, y, s)
            if v is None:
                continue
            is_gt = any(abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 and 0.5 <= s / max(w, h) <= 2.0
                        for cx, cy, w, h in gts)
            rec.append((is_gt, sc, float(v @ single), float((cen @ v).max()), stem, x, y, s))
    isgt = np.array([r[0] for r in rec])
    ncc = np.array([r[1] for r in rec])
    cs = np.array([r[2] for r in rec])
    ck = np.array([r[3] for r in rec])
    print("candidates scored: %d  (GT %d / non-GT %d)" % (len(rec), isgt.sum(), (~isgt).sum()))

    res = {}
    res["brightness_only"] = auc(ncc[isgt], ncc[~isgt])
    res["context_single"] = auc(cs[isgt], cs[~isgt])
    res["context_kmeans"] = auc(ck[isgt], ck[~isgt])
    # combined: z-score each then sum with weight
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)
    best_w, best_auc = None, -1
    for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        comb = z(ncc) + w * z(ck)
        a = auc(comb[isgt], comb[~isgt])
        if a > best_auc:
            best_auc, best_w = a, w
    res["combined_best"] = best_auc
    res["combined_best_lambda"] = best_w
    comb = z(ncc) + best_w * z(ck)

    print("")
    print("AUC (GT vs non-GT brightness peak):")
    print("   brightness NCC only              : %.3f   <- Phase 2 baseline" % res["brightness_only"])
    print("   context, single mean template    : %.3f   <- flow-doc version" % res["context_single"])
    print("   context, k-means K=%-2d           : %.3f   <- proposed" % (K_CLUSTERS, res["context_kmeans"]))
    print("   brightness + context (lambda=%.2f): %.3f" % (best_w, best_auc))

    # precision@K per image using each ranking
    def prec_at_k(score, K=5):
        by = collections.defaultdict(list)
        for i, r in enumerate(rec):
            by[r[4]].append(i)
        tot = hit = 0
        for stem, idxs in by.items():
            idxs = sorted(idxs, key=lambda i: -score[i])[:K]
            hit += sum(1 for i in idxs if rec[i][0])
            tot += len(idxs)
        return hit / max(tot, 1)

    print("")
    print("precision@5 per image (fraction of top-5 that are real missing_hole):")
    for name, sco in [("brightness", ncc), ("context_kmeans", ck), ("combined", comb)]:
        print("   %-16s %.3f" % (name, prec_at_k(sco, 5)))
    res["precision_at5"] = {n: prec_at_k(s, 5) for n, s in
                            [("brightness", ncc), ("context_kmeans", ck), ("combined", comb)]}
    json.dump(res, open(os.path.join(OUT, "context_metrics.json"), "w"), indent=2)

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (vals, name, key) in zip(ax, [(ncc, "brightness NCC", "brightness_only"),
                                         (ck, "context (kmeans)", "context_kmeans"),
                                         (comb, "combined", "combined_best")]):
        a.hist(vals[isgt], bins=50, alpha=.6, density=True, label="GT missing_hole")
        a.hist(vals[~isgt], bins=50, alpha=.6, density=True, label="non-GT peak")
        a.set_title("%s  AUC=%.3f" % (name, res[key]))
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "context_summary.png"), dpi=130)
    plt.close()

    g = int(np.ceil(np.sqrt(K_CLUSTERS + 1)))
    fig, axes = plt.subplots(g, g, figsize=(9, 9))
    for a in np.ravel(axes):
        a.axis("off")
    axes = np.ravel(axes)
    axes[0].imshow(to_image(single), cmap="gray")
    axes[0].set_title("single mean", fontsize=8)
    axes[0].axis("off")
    for i in range(K_CLUSTERS):
        axes[i + 1].imshow(to_image(cen[i]), cmap="gray")
        axes[i + 1].set_title("k%d" % i, fontsize=8)
        axes[i + 1].axis("off")
    plt.suptitle("context templates (defect masked out, annulus only)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "context_templates.png"), dpi=130)
    plt.close()

    # ---- what does context reject that brightness liked?
    order_b = np.argsort(-ncc)
    demoted = []
    rank_c = {i: r for r, i in enumerate(np.argsort(-comb))}
    for i in order_b[:400]:
        if not rec[i][0] and rank_c[i] > len(rec) * 0.5:
            demoted.append(i)
        if len(demoted) >= 24:
            break
    if demoted:
        tiles = []
        for i in demoted:
            _, sc, _, _, stem, x, y, s = rec[i]
            col = cv2.imread(os.path.join(DS, "val", "images", stem + ".jpg"))
            H, W = col.shape[:2]
            R = int(1.5 * s)
            t = col[max(0, y - R):min(H, y + R), max(0, x - R):min(W, x + R)].copy()
            if t.size == 0:
                continue
            t = cv2.resize(t, (120, 120))
            cv2.rectangle(t, (60 - int(s * 30 / R), 60 - int(s * 30 / R)),
                          (60 + int(s * 30 / R), 60 + int(s * 30 / R)), (255, 0, 0), 2)
            tiles.append(t)
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
        if rows:
            cv2.imwrite(os.path.join(OUT, "rejected_by_context.png"), np.vstack(rows))
            print("")
            print("saved rejected_by_context.png (%d FPs demoted by the context stage)" % len(tiles))
    print("")
    print("figures -> %s" % OUT)
    return res


if __name__ == "__main__":
    single, cen = build_templates()
    evaluate(single, cen)
