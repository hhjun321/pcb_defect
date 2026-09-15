"""
Phase 4 - Copy-Paste synthesis (research flow Step 7 / Step 8).

Builds one augmented TRAIN set per placement strategy.  val/test are copied
through untouched (hardlinks), so every arm is evaluated on identical held-out
data and only the training distribution differs.

Placement strategies
--------------------
  random      uniform position, size drawn from the GT size distribution.
              Rejects overlap only.  (flow doc: "Random CP")
  brightness  stage-A multi-scale NCC on D_mean, ring gate, grid cap.
  context     brightness top-K, then re-ranked by context-template similarity
              and the best N taken.  This is the literal Step 6/7 procedure.
              Kept as an ablation arm: Phase 3 measured its contribution at
              AUC 0.524 (chance) with optimal weight lambda = 0.
  brightspec  brightness score combined with the specular-highlight feature,
              z(ncc) + 1.5 * z(spec).  Phase 3: AUC 0.918, 100% valid
              through-hole pads in the top 20.  <- proposed method

Pasting
-------
Two modes; `core` is the default and the one used for the released sets.

  core  CORE TRANSPLANT.  A missing hole physically means "the pad is there,
        the drilled hole is not", so the target pad's own metal annulus is
        KEPT and only the green substrate core is transplanted into its
        centre.  The core colour is taken from the solder resist immediately
        around the target pad and only the donor's core TEXTURE is carried
        over.  Illumination, pad shape and pad size therefore match by
        construction - nothing is transferred across lighting conditions.

  full  the naive variant: resize the whole donor crop over the site and
        colour-harmonise it with ring-band statistics.  Kept for comparison;
        QC showed washed-out pink haloes wherever the donor/target ring
        standard deviations differed, and circular donors pasted onto oval
        pads.

The new annotation is the site box itself, so labels stay exact by construction.
"""
import os, sys, glob, json, argparse, collections
import numpy as np
import cv2

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candgen as CG
import phase3_context as P3

ROOT = PCB_ROOT
DSDIR = os.path.join(ROOT, "datasets")
P2 = os.path.join(ROOT, "outputs", "phase2")
P3DIR = os.path.join(ROOT, "outputs", "phase3")
OUTFIG = os.path.join(ROOT, "outputs", "phase4")
os.makedirs(OUTFIG, exist_ok=True)

METHODS = ["random", "brightness", "context", "brightspec"]
LAMBDA_SPEC = 1.5
TOPK_POOL = 20      # brightness shortlist handed to the context re-ranker


# ------------------------------------------------------------------ helpers
def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def read_boxes(lp, W, H):
    out = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5:
            cx, cy, w, h = [float(v) for v in f[1:]]
            out.append((cx * W, cy * H, w * W, h * H))
    return out


def overlaps(x, y, s, boxes, pad=2.0):
    for cx, cy, w, h in boxes:
        if abs(x - cx) * 2 < (s + w) * pad / 2 and abs(y - cy) * 2 < (s + h) * pad / 2:
            return True
    return False


def feather(s, r_in=0.50, width=0.08):
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / s
    a = np.clip((r_in - r) / width, 0.0, 1.0)
    return a.astype(np.float32)[..., None]


_RAD = {}


def _radial(s):
    r = _RAD.get(s)
    if r is None:
        yy, xx = np.mgrid[0:s, 0:s]
        c = (s - 1) / 2.0
        r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / s
        _RAD[s] = r
    return r


_BINS = np.arange(0.05, 0.50, 0.02)
_BINIDX = {}


def _bin_index(big):
    """per-window map from pixel -> saturation-profile bin, cached by size"""
    bi = _BINIDX.get(big)
    if bi is None:
        rb = _radial(big)
        bi = np.digitize(rb, _BINS) - 1
        bi[(rb < _BINS[0]) | (rb >= _BINS[-1] + 0.02)] = -1
        _BINIDX[big] = bi
    return bi


def ring_stats(bgr):
    s = bgr.shape[0]
    ring, core, sect = CG._masks(s)
    v = bgr[ring].reshape(-1, 3).astype(np.float32)
    return v.mean(0), v.std(0) + 1e-3


CORE_R = 0.20        # transplanted core radius (fraction of the site box)
CORE_FEATHER = 0.11  # Chosen by measurement, not by fitting an alpha model.
                     # Inverting the real saturation profile as if it were a
                     # linear alpha blend suggests a much wider ramp
                     # (r_in 0.245, width 0.225) - that was built and measured
                     # and came out WORSE: profile MAD rose from sat 21.6 to
                     # 28.0 and the core-region MAD from 22.9 to 44.9.  HSV
                     # saturation is (max-min)/max, so blending green core with
                     # bright metal is strongly non-linear and the inversion is
                     # invalid.  0.20/0.11 reproduces the real core almost
                     # exactly: 253/243/181 vs real 239/223/187 at r/s
                     # 0.02/0.06/0.10.
TEXTURE_GAIN = 0.6   # how much of the donor core texture to carry over

_CAL = json.load(open(os.path.join(OUTFIG, "core_calibration.json")))
CORE_RATIO = np.array(_CAL["core_ratio"], np.float32)     # core = resist * ratio
PAD_EDGE_FRAC = float(_CAL["pad_edge_frac"])              # box = r_out / frac


def estimate_pad_box(col, x, y, s0):
    """Re-estimate the box side from the pad itself.

    The template scale comes from a discrete set {20,24,...,48} and was measured
    to produce boxes tighter than real GT boxes (synthetic saturation at
    r/s 0.44-0.48 was 95 vs 153 for real).  Here the metal ring's outer edge is
    found from the radial saturation profile and converted with the calibrated
    PAD_EDGE_FRAC, so synthetic boxes carry the same geometry as real ones.
    """
    H, W = col.shape[:2]
    s = s0
    for _ in range(2):
        big = int(round(1.9 * s))
        bx, by = x - big // 2, y - big // 2
        if bx < 0 or by < 0 or bx + big > W or by + big > H:
            return None
        hsv = cv2.cvtColor(col[by:by + big, bx:bx + big], cv2.COLOR_BGR2HSV)
        S = hsv[..., 1].astype(np.float32)
        bi = _bin_index(big)
        nb = len(_BINS)
        cnt = np.bincount(bi[bi >= 0], minlength=nb).astype(np.float32)
        tot = np.bincount(bi[bi >= 0], weights=S[bi >= 0], minlength=nb)
        prof = tot / np.maximum(cnt, 1)
        i_min = int(np.argmin(prof))
        thr = (prof[i_min] + prof[-1]) / 2.0
        j = i_min
        while j < nb - 1 and prof[j] < thr:
            j += 1
        r_out = _BINS[j] * 1.9 * s          # pad outer radius in pixels
        s_new = int(round(r_out / PAD_EDGE_FRAC))
        s_new = int(np.clip(s_new, 16, 70))
        if abs(s_new - s) <= 1:
            s = s_new
            break
        s = s_new
    return s


def paste_full(dst, donor, x, y, s):
    """naive variant: whole donor crop, ring-statistics colour transfer."""
    H, W = dst.shape[:2]
    h2 = s // 2
    x0, y0 = x - h2, y - h2
    if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
        return False
    tgt = dst[y0:y0 + s, x0:x0 + s].astype(np.float32)
    src = cv2.resize(donor, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    ms, ss = ring_stats(src.astype(np.uint8))
    mt, st = ring_stats(tgt.astype(np.uint8))
    gain = np.clip(st / ss, 0.7, 1.4)          # unclamped gain caused pink haloes
    src = np.clip((src - ms) * gain + mt, 0, 255)
    a = feather(s)
    dst[y0:y0 + s, x0:x0 + s] = (a * src + (1 - a) * tgt).astype(np.uint8)
    return True


def paste_core(dst, donor, x, y, s):
    """core transplant: keep the target pad's metal, replace its centre with
    solder-resist-coloured substrate carrying the donor core's texture."""
    H, W = dst.shape[:2]
    h2 = s // 2
    x0, y0 = x - h2, y - h2
    pad = int(round(0.9 * s))                  # need resist samples around the pad
    if x0 - pad < 0 or y0 - pad < 0 or x0 + s + pad > W or y0 + s + pad > H:
        return False
    tgt = dst[y0:y0 + s, x0:x0 + s].astype(np.float32)

    # local solder-resist colour: an annulus just OUTSIDE the pad
    big = int(round(1.9 * s))
    bx, by = x - big // 2, y - big // 2
    around = dst[by:by + big, bx:bx + big].astype(np.float32)
    rb = _radial(big)
    resist = around[(rb >= 0.40) & (rb <= 0.48)].reshape(-1, 3)
    if len(resist) < 20:
        return False
    resist_mu = np.median(resist, axis=0) * CORE_RATIO   # calibrated: the core
    # sits in the pad opening and is a deeper, darker green than open resist

    # donor core texture, mean-removed
    ds = donor.shape[0]
    rd = _radial(ds)
    dcore_m = rd <= CORE_R
    if dcore_m.sum() < 9:
        return False
    dc = cv2.resize(donor, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rs = _radial(s)
    core_m = rs <= CORE_R
    tex = dc - dc[core_m].mean(0)

    new = resist_mu + TEXTURE_GAIN * tex
    new = np.clip(new, 0, 255)
    a = feather(s, r_in=CORE_R, width=CORE_FEATHER)
    dst[y0:y0 + s, x0:x0 + s] = (a * new + (1 - a) * tgt).astype(np.uint8)
    return True


def paste(dst, donor, x, y, s, mode="core"):
    return paste_core(dst, donor, x, y, s) if mode == "core" else paste_full(dst, donor, x, y, s)


# ------------------------------------------------------------------ donors
def build_donor_pool(split):
    """real missing_hole crops from the TRAIN images of `split`"""
    pool = []
    base = os.path.join(DSDIR, split, "train")
    for ip in sorted(glob.glob(os.path.join(base, "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        col = cv2.imread(ip)
        H, W = col.shape[:2]
        for cx, cy, w, h in read_boxes(os.path.join(base, "labels", stem + ".txt"), W, H):
            s = int(round(max(w, h)))
            x0, y0 = int(round(cx)) - s // 2, int(round(cy)) - s // 2
            if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H or s < 12:
                continue
            pool.append((col[y0:y0 + s, x0:x0 + s].copy(), s, stem))
    return pool


# ------------------------------------------------------------------ sites
def pick_random(col, gts, n, rng):
    """uniform placement, size drawn from this image's GT size distribution"""
    H, W = col.shape[:2]
    sizes = [int(round(max(w, h))) for _, _, w, h in gts] or [28]
    out, placed = [], list(gts)
    for _ in range(200):
        if len(out) >= n:
            break
        s = int(rng.choice(sizes)) if rng.random() < 0.5 else int(rng.integers(20, 46))
        x = int(rng.integers(s, W - s))
        y = int(rng.integers(s, H - s))
        if overlaps(x, y, s, placed):
            continue
        out.append((x, y, s, 0.0))
        placed.append((x, y, s, s))
    return out


def pick_sites(method, col, TM, gts, n, rng, single, cen):
    H, W = col.shape[:2]
    if method == "random":
        return pick_random(col, gts, n, rng)

    cands = CG.detect(col, TM, topn=TOPK_POOL * 3, use_ring_gate=True,
                      need_spec=(method == "brightspec"),
                      grid_cell=100, grid_cap=2)
    cands = [c for c in cands if not overlaps(c[1], c[2], c[3], gts, pad=1.0)]
    if not cands:
        return []

    if method == "brightness":
        ranked = cands
    elif method == "context":
        short = cands[:TOPK_POOL]
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
        scored = []
        for sc, x, y, s, ft in short:
            v = P3.context_patch(gray, x, y, s)
            cscore = float((cen @ v).max()) if v is not None else -1.0
            scored.append((cscore, x, y, s, ft, sc))
        scored.sort(key=lambda t: -t[0])
        ranked = [(t[5], t[1], t[2], t[3], t[4]) for t in scored]
    elif method == "brightspec":
        ncc = np.array([c[0] for c in cands])
        spec = np.array([c[4]["spec"] for c in cands])
        zz = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        sco = zz(ncc) + LAMBDA_SPEC * zz(spec)
        ranked = [cands[i] for i in np.argsort(-sco)]
    else:
        raise ValueError(method)

    out, placed = [], list(gts)
    for sc, x, y, s, ft in ranked:
        if len(out) >= n:
            break
        s2 = estimate_pad_box(col, x, y, s)
        if s2 is None:
            continue
        if overlaps(x, y, s2, placed, pad=1.0):
            continue
        out.append((x, y, s2, float(sc)))
        placed.append((x, y, s2, s2))
    return out


# ------------------------------------------------------------------ build
def _rank(method, cands, col, cen):
    """rank a shared candidate pool according to one placement strategy"""
    if method == "brightness":
        return cands
    if method == "brightspec":
        pool = [c for c in cands if c[4]["spec"] >= CG.G_SPEC]
        if not pool:
            return []
        ncc = np.array([c[0] for c in pool])
        spec = np.array([c[4]["spec"] for c in pool])
        zz = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        sco = zz(ncc) + LAMBDA_SPEC * zz(spec)
        return [pool[i] for i in np.argsort(-sco)]
    if method == "context":
        short = cands[:TOPK_POOL]
        gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
        scored = []
        for sc, x, y, s, ft in short:
            v = P3.context_patch(gray, x, y, s)
            scored.append((float((cen @ v).max()) if v is not None else -1.0, sc, x, y, s, ft))
        scored.sort(key=lambda t: -t[0])
        return [(t[1], t[2], t[3], t[4], t[5]) for t in scored]
    raise ValueError(method)


def build_all(split, methods, n_insert, seed, paste_mode="core"):
    """Build every placement arm in ONE pass over the images.

    Reading the image, the 7-scale template match and the ring features are
    identical for every arm, so they are computed once and the arms differ only
    in how the shared candidate pool is ranked.  That also makes the ablation
    cleaner: every arm chooses from exactly the same pool.
    """
    rng = {m: np.random.default_rng(seed + i) for i, m in enumerate(methods)}
    src = os.path.join(DSDIR, split)
    dsts = {m: os.path.join(DSDIR, "%s_%scp" % (split, m)) for m in methods}
    for m, dst in dsts.items():
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(dst, sp, "images"), exist_ok=True)
            os.makedirs(os.path.join(dst, sp, "labels"), exist_ok=True)
        for sp in ["val", "test"]:
            for p in glob.glob(os.path.join(src, sp, "images", "*.jpg")):
                link(p, os.path.join(dst, sp, "images", os.path.basename(p)))
            for p in glob.glob(os.path.join(src, sp, "labels", "*.txt")):
                link(p, os.path.join(dst, sp, "labels", os.path.basename(p)))
        with open(os.path.join(dst, "data.yaml"), "w") as f:
            f.write("path: %s\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
                    "nc: 1\nnames:\n  0: missing_hole\n" % dst.replace("\\", "/"))

    Dz = np.load(os.path.join(P2, "Dz_mean_boardtrain.npy"))
    TM = CG.make_templates(Dz)
    cen = np.load(os.path.join(P3DIR, "C_kmeans.npy"))
    donors = build_donor_pool(split)
    print("  donor pool: %d real missing_hole crops" % len(donors), flush=True)

    stat = {m: dict(added=0, sizes=[]) for m in methods}
    qc = {m: [] for m in methods}
    n_img = 0
    for ip in sorted(glob.glob(os.path.join(src, "train", "images", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(src, "train", "labels", stem + ".txt")
        col = cv2.imread(ip)
        H, W = col.shape[:2]
        gts = read_boxes(lp, W, H)
        base_lines = [l.rstrip("\n") for l in open(lp) if l.strip()]
        shared = CG.detect(col, TM, topn=TOPK_POOL * 3, use_ring_gate=True,
                           need_spec=False, grid_cell=100, grid_cap=2)
        shared = [c for c in shared if not overlaps(c[1], c[2], c[3], gts, pad=1.0)]

        for m in methods:
            r = rng[m]
            if m == "random":
                picks = pick_random(col, gts, n_insert, r)
            else:
                ranked = _rank(m, shared, col, cen)
                picks, placed = [], list(gts)
                for sc, x, y, s, ft in ranked:
                    if len(picks) >= n_insert:
                        break
                    s2 = estimate_pad_box(col, x, y, s)
                    if s2 is None or overlaps(x, y, s2, placed, pad=1.0):
                        continue
                    picks.append((x, y, s2, float(sc)))
                    placed.append((x, y, s2, s2))
            out = col.copy()
            lines = list(base_lines)
            for (x, y, s, sc) in picks:
                cand = [d for d in donors if d[2] != stem and 0.6 <= d[1] / s <= 1.7]
                if not cand:
                    cand = [d for d in donors if d[2] != stem]
                d = cand[int(r.integers(len(cand)))]
                if not paste(out, d[0], x, y, s, paste_mode):
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f" % (x / W, y / H, s / W, s / H))
                stat[m]["added"] += 1
                stat[m]["sizes"].append(s)
                if len(qc[m]) < 24 and r.random() < 0.05:
                    R = int(1.8 * s)
                    t = out[max(0, y - R):min(H, y + R), max(0, x - R):min(W, x + R)].copy()
                    if t.size:
                        qc[m].append(cv2.resize(t, (150, 150)))
            cv2.imwrite(os.path.join(dsts[m], "train", "images", stem + ".jpg"), out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(os.path.join(dsts[m], "train", "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        n_img += 1
        if n_img % 200 == 0:
            print("    %d images ..." % n_img, flush=True)

    metas = []
    for m in methods:
        sz = stat[m]["sizes"]
        meta = dict(base=split, method=m, n_insert=n_insert, seed=seed, paste_mode=paste_mode,
                    train_images=n_img, boxes_added=stat[m]["added"],
                    mean_size_added=float(np.mean(sz)) if sz else 0.0,
                    lambda_spec=LAMBDA_SPEC, topk_pool=TOPK_POOL)
        json.dump(meta, open(os.path.join(dsts[m], "synthesis_meta.json"), "w"), indent=2)
        if qc[m]:
            rows = [np.hstack(qc[m][i:i + 6]) for i in range(0, len(qc[m]) - len(qc[m]) % 6, 6)]
            if rows:
                cv2.imwrite(os.path.join(OUTFIG, "qc_%s.png" % os.path.basename(dsts[m])),
                            np.vstack(rows))
        print("  [%s] imgs=%d  boxes added=%d (%.2f/img)  mean size=%.1fpx"
              % (os.path.basename(dsts[m]), n_img, stat[m]["added"],
                 stat[m]["added"] / max(n_img, 1), meta["mean_size_added"]), flush=True)
        metas.append(meta)
    return metas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["mh_board", "mh_orig"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--n-insert", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paste", default="core", choices=["core", "full"])
    a = ap.parse_args()
    allmeta = []
    for sp in a.splits:
        print("=" * 74)
        print("BASE SPLIT: %s" % sp)
        print("=" * 74)
        allmeta.extend(build_all(sp, a.methods, a.n_insert, a.seed, a.paste))
    json.dump(allmeta, open(os.path.join(OUTFIG, "synthesis_summary.json"), "w"), indent=2)
    print("")
    print("done -> %s" % DSDIR)
