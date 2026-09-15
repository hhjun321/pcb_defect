import os, re, glob, json, collections
import numpy as np
from PIL import Image
import matplotlib

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = PCB_ROOT + "/pcb-defect-dataset"
OUT  = PCB_ROOT + "/outputs/phase1"
os.makedirs(OUT, exist_ok=True)
MH = 2
S  = 32
rng = np.random.default_rng(0)

NAME_RE = re.compile(r"^(rotation_(?P<deg>\d+)_)?(?P<flip>l_)?light_(?P<light>\d+)_missing_hole_(?P<board>\d+)_(?P<tile>\d+)_600$")


def parse(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return dict(rot=int(d["deg"]) if d["deg"] else 0,
                flip=bool(d["flip"]),
                light=int(d["light"]),
                board=int(d["board"]),
                tile=int(d["tile"]))


def mh_stems(split):
    out = []
    for p in glob.glob(os.path.join(ROOT, split, "images", "*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if "missing_hole" in stem:
            out.append(stem)
    return sorted(out)


# ---------- A. leakage audit ----------
print("=" * 70)
print("A. SPLIT LEAKAGE AUDIT (missing_hole images only)")
print("=" * 70)
meta = {sp: {s: parse(s) for s in mh_stems(sp)} for sp in ["train", "val", "test"]}
for sp in meta:
    bad = [k for k, v in meta[sp].items() if v is None]
    print("%s: %d imgs, unparsed=%d" % (sp, len(meta[sp]), len(bad)))


def keyset(sp, fields):
    return set(tuple(v[f] for f in fields) for v in meta[sp].values() if v)


for fields in [("board",), ("board", "light"), ("board", "light", "tile"),
               ("board", "light", "tile", "rot"), ("board", "light", "tile", "rot", "flip")]:
    tr, va, te = keyset("train", fields), keyset("val", fields), keyset("test", fields)
    print("  key=%-30s |train|=%4d |test|=%4d  test_seen_in_train=%d/%d (%.0f%%)  val_seen=%d/%d"
          % ("+".join(fields), len(tr), len(te), len(te & tr), len(te),
             100 * len(te & tr) / max(len(te), 1), len(va & tr), len(va)))
print("")
print("  -> identical physical scene (board+light+tile) shared across splits means")
print("     train/test differ only by rotation/flip augmentation = LEAKAGE.")

# ---------- B. crop extraction (TRAIN ONLY) ----------
print("")
print("=" * 70)
print("B. MISSING_HOLE CROP EXTRACTION (train only)")
print("=" * 70)
crops = []
info = []
for stem in mh_stems("train"):
    ip = os.path.join(ROOT, "train", "images", stem + ".jpg")
    lp = os.path.join(ROOT, "train", "labels", stem + ".txt")
    if not os.path.exists(lp):
        continue
    im = Image.open(ip).convert("L")
    W, H = im.size
    A = np.asarray(im, dtype=np.float32)
    for line in open(lp):
        f = line.split()
        if len(f) != 5 or int(f[0]) != MH:
            continue
        cx, cy, w, h = [float(v) for v in f[1:]]
        cx *= W; cy *= H; w *= W; h *= H
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x1, y1 = int(round(cx + w / 2)), int(round(cy + h / 2))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patch = Image.fromarray(A[y0:y1, x0:x1]).resize((S, S), Image.BILINEAR)
        crops.append(np.asarray(patch, dtype=np.float32))
        m = parse(stem)
        m.update(dict(stem=stem, x0=x0, y0=y0, x1=x1, y1=y1, w=x1 - x0, h=y1 - y0, imgW=W, imgH=H))
        info.append(m)
C = np.stack(crops)
print("crops extracted: %s" % (C.shape,))
np.save(os.path.join(OUT, "crops_train_32.npy"), C)
json.dump(info, open(os.path.join(OUT, "crops_train_meta.json"), "w"))

# ---------- C. brightness template ----------
print("")
print("=" * 70)
print("C. BRIGHTNESS TEMPLATE  D_mean / D_std")
print("=" * 70)
D_mean = C.mean(0)
D_std = C.std(0)
Cz = (C - C.mean((1, 2), keepdims=True)) / (C.std((1, 2), keepdims=True) + 1e-6)
Dz_mean = Cz.mean(0)
Dz_std = Cz.std(0)
np.save(os.path.join(OUT, "D_mean.npy"), D_mean)
np.save(os.path.join(OUT, "D_std.npy"), D_std)
np.save(os.path.join(OUT, "Dz_mean.npy"), Dz_mean)
np.save(os.path.join(OUT, "Dz_std.npy"), Dz_std)

print("raw crop mean brightness : grand mean=%.1f, std ACROSS crops=%.1f (lighting/board spread)"
      % (C.mean(), C.mean((1, 2)).std()))
print("D_mean  : min=%.1f max=%.1f center=%.1f corner=%.1f"
      % (D_mean.min(), D_mean.max(), D_mean[S // 2, S // 2], D_mean[:4, :4].mean()))
print("D_std   : mean=%.1f max=%.1f" % (D_std.mean(), D_std.max()))
print("Dz_mean : min=%.2f max=%.2f center=%.2f corner=%.2f"
      % (Dz_mean.min(), Dz_mean.max(), Dz_mean[S // 2, S // 2], Dz_mean[:4, :4].mean()))
print("Dz_std  : mean=%.2f  <- residual spread after per-crop lighting normalization" % Dz_std.mean())


def r2(X, T):
    res = ((X - T) ** 2).sum((1, 2))
    tot = ((X - X.mean((1, 2), keepdims=True)) ** 2).sum((1, 2))
    return 1 - res / tot


r2_raw = r2(C, D_mean)
r2_z = r2(Cz, Dz_mean)
print("")
print("R^2 of template per sample: raw    median=%.3f  p25=%.3f p75=%.3f"
      % (np.median(r2_raw), np.percentile(r2_raw, 25), np.percentile(r2_raw, 75)))
print("R^2 of template per sample: z-norm median=%.3f  p25=%.3f p75=%.3f"
      % (np.median(r2_z), np.percentile(r2_z, 25), np.percentile(r2_z, 75)))
print("  (higher = single mean template genuinely represents samples -> RQ1)")

yy, xx = np.mgrid[0:S, 0:S]
r = np.sqrt((xx - (S - 1) / 2) ** 2 + (yy - (S - 1) / 2) ** 2)
bins = np.arange(0, 17, 1.0)
idx = np.digitize(r.ravel(), bins) - 1
prof = np.array([Dz_mean.ravel()[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
profs = np.array([Dz_std.ravel()[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)])
print("")
print("radial profile of Dz_mean:")
for b, (v, s) in enumerate(zip(prof, profs)):
    if not np.isnan(v):
        print("   r=%2d-%2d px   z=%+.2f  (std %.2f)" % (b, b + 1, v, s))

# ---------- D. defect vs random background ----------
print("")
print("=" * 70)
print("D. DEFECT vs RANDOM BACKGROUND PATCH (same train images)")
print("=" * 70)
bg = []
by_stem = collections.defaultdict(list)
for m in info:
    by_stem[m["stem"]].append(m)
for stem, ms in by_stem.items():
    A = np.asarray(Image.open(os.path.join(ROOT, "train", "images", stem + ".jpg")).convert("L"), dtype=np.float32)
    H, W = A.shape
    boxes = [(m["x0"], m["y0"], m["x1"], m["y1"]) for m in ms]
    got = 0
    tries = 0
    while got < 3 and tries < 60:
        tries += 1
        sz = int(rng.integers(20, 40))
        x0 = int(rng.integers(0, W - sz))
        y0 = int(rng.integers(0, H - sz))
        if any(not (x0 + sz < bx0 or x0 > bx1 or y0 + sz < by0 or y0 > by1) for bx0, by0, bx1, by1 in boxes):
            continue
        p = Image.fromarray(A[y0:y0 + sz, x0:x0 + sz]).resize((S, S), Image.BILINEAR)
        bg.append(np.asarray(p, dtype=np.float32))
        got += 1
B = np.stack(bg)
Bz = (B - B.mean((1, 2), keepdims=True)) / (B.std((1, 2), keepdims=True) + 1e-6)
print("background patches: %s" % (B.shape,))
mse_def = ((Cz - Dz_mean) ** 2).mean((1, 2))
mse_bg = ((Bz - Dz_mean) ** 2).mean((1, 2))
print("MSE(sample, Dz_mean)  defect : median=%.3f  p90=%.3f" % (np.median(mse_def), np.percentile(mse_def, 90)))
print("MSE(sample, Dz_mean)  bgrnd  : median=%.3f  p10=%.3f" % (np.median(mse_bg), np.percentile(mse_bg, 10)))
allv = np.concatenate([mse_def, mse_bg])
lab = np.concatenate([np.ones(len(mse_def)), np.zeros(len(mse_bg))])
order = np.argsort(allv)
ranks = np.empty(len(allv))
ranks[order] = np.arange(1, len(allv) + 1)
auc = (ranks[lab == 0].sum() - len(mse_bg) * (len(mse_bg) + 1) / 2) / (len(mse_bg) * len(mse_def))
print("AUC (lower MSE => defect) = %.3f   [0.5=useless, 1.0=perfect]" % auc)
print("raw mean brightness  defect=%.1f  background=%.1f" % (C.mean((1, 2)).mean(), B.mean((1, 2)).mean()))

# ---------- E. figures ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
im0 = ax[0, 0].imshow(D_mean, cmap="gray"); ax[0, 0].set_title("D_mean (raw gray)"); plt.colorbar(im0, ax=ax[0, 0])
im1 = ax[0, 1].imshow(D_std, cmap="magma"); ax[0, 1].set_title("D_std (raw)"); plt.colorbar(im1, ax=ax[0, 1])
im2 = ax[0, 2].imshow(Dz_mean, cmap="RdBu_r"); ax[0, 2].set_title("Dz_mean (per-crop z-norm)"); plt.colorbar(im2, ax=ax[0, 2])
im3 = ax[1, 0].imshow(Dz_std, cmap="magma"); ax[1, 0].set_title("Dz_std"); plt.colorbar(im3, ax=ax[1, 0])
xs = np.arange(len(prof)) + 0.5
ax[1, 1].plot(xs, prof, "o-")
ax[1, 1].fill_between(xs, prof - profs, prof + profs, alpha=.25)
ax[1, 1].axhline(0, color="k", lw=.5)
ax[1, 1].set_xlabel("radius (px in 32x32 crop)"); ax[1, 1].set_ylabel("z brightness")
ax[1, 1].set_title("radial profile of Dz_mean")
ax[1, 2].hist(mse_def, bins=60, alpha=.6, density=True, label="defect")
ax[1, 2].hist(mse_bg, bins=60, alpha=.6, density=True, label="random bg")
ax[1, 2].set_xlabel("MSE to Dz_mean"); ax[1, 2].legend()
ax[1, 2].set_title("separability AUC=%.3f" % auc)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "template_summary.png"), dpi=130)
plt.close()

k = 64
sel = rng.choice(len(C), k, replace=False)
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for a, i in zip(axes.ravel(), sel):
    a.imshow(C[i], cmap="gray")
    a.axis("off")
plt.suptitle("random 64 missing_hole crops (32x32 gray, raw)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "crop_grid.png"), dpi=130)
plt.close()

print("")
print("figures -> %s" % OUT)
