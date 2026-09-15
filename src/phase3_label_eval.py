"""
Phase 3c - score the context stage on the question it actually answers.

Phase 3a measured AUC(GT vs non-GT) and found context adds nothing (0.848 ->
0.853).  That metric is ill-posed: the non-GT pool mixes normal pads (which the
context stage SHOULD keep - they are the Copy-Paste targets) with silkscreen /
gaps / mounting holes (which it SHOULD reject).  The two requirements cancel.

Here the 120 hand-labelled non-GT candidates are used instead:

    th   through-hole pad   -> VALID insertion site  (a missing_hole needs a hole)
    smd  surface-mount pad  -> invalid (metal, but no hole)
    no   not a pad          -> invalid

Reported:
  AUC(valid vs invalid) for each scoring function
  valid-pad rate @ top-K  (the quantity that actually governs Phase 4 quality)
"""
import os, sys, json
import numpy as np
import matplotlib

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = PCB_ROOT + "/outputs/phase3"

sample = json.load(open(os.path.join(OUT, "label_sample.json")))
man = json.load(open(os.path.join(OUT, "labels_manual.json")))["labels"]

for r in sample:
    r["cls"] = man[str(r["id"])]

cls = np.array([r["cls"] for r in sample])
valid = (cls == "th")
print("=" * 74)
print("HAND-LABELLED NON-GT CANDIDATES (mh_board val, stage-A top-10, n=%d)" % len(sample))
print("=" * 74)
for c in ["th", "smd", "no"]:
    print("   %-4s %3d  (%4.1f%%)" % (c, (cls == c).sum(), 100 * (cls == c).mean()))
print("   valid insertion sites (th) = %.1f%% of what the brightness stage proposes" % (100 * valid.mean()))

# score by brightness-score band to show where the failures live
ncc = np.array([r["ncc"] for r in sample])
band = np.argsort(np.argsort(-ncc)) // (len(sample) // 4)
print("")
print("   by brightness-score quartile (Q1 = highest NCC):")
for q in range(4):
    m = band == q
    print("     Q%d  ncc %.3f..%.3f   th %4.1f%%  smd %4.1f%%  no %4.1f%%"
          % (q + 1, ncc[m].min(), ncc[m].max(),
             100 * (cls[m] == "th").mean(), 100 * (cls[m] == "smd").mean(), 100 * (cls[m] == "no").mean()))

cs = np.array([r["ctx_single"] for r in sample])
ck = np.array([r["ctx_kmeans"] for r in sample])
spec = np.array([r["spec"] for r in sample])
core_sat = np.array([r["core_sat"] for r in sample])
ring_sat = np.array([r["ring_sat"] for r in sample])


def auc(score, pos):
    p, n = score[pos], score[~pos]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    a = np.concatenate([p, n])
    lab = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    o = np.argsort(a)
    rk = np.empty(len(a))
    rk[o] = np.arange(1, len(a) + 1)
    return float((rk[lab == 1].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def z(a):
    return (a - a.mean()) / (a.std() + 1e-9)


print("")
print("=" * 74)
print("AUC( valid through-hole pad  vs  invalid )")
print("=" * 74)
rows = [("brightness NCC", ncc),
        ("context single", cs),
        ("context kmeans", ck),
        ("spec (specular)", spec),
        ("-core_sat", -core_sat),
        ("-ring_sat", -ring_sat)]
for name, s in rows:
    print("   %-18s %.3f" % (name, auc(s, valid)))

best = (None, -1)
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    a = auc(z(ncc) + w * z(ck), valid)
    if a > best[1]:
        best = (w, a)
print("   %-18s %.3f   (lambda=%.2f)" % ("brightness+context", best[1], best[0]))
comb = z(ncc) + best[0] * z(ck)

# a fuller combination using the cheap ring features too
best3 = (None, None, -1)
for w in [0.0, 0.5, 1.0, 1.5]:
    for u in [0.0, 0.5, 1.0, 1.5]:
        a = auc(z(ncc) + w * z(ck) + u * z(spec), valid)
        if a > best3[2]:
            best3 = (w, u, a)
print("   %-18s %.3f   (lambda_ctx=%.2f, lambda_spec=%.2f)" % ("bright+ctx+spec", best3[2], best3[0], best3[1]))
comb3 = z(ncc) + best3[0] * z(ck) + best3[1] * z(spec)

print("")
print("=" * 74)
print("VALID-PAD RATE @ top-K   (what Phase 4 insertion quality depends on)")
print("=" * 74)
print("   %-18s %s" % ("ranking", "  ".join("K=%-3d" % K for K in (10, 20, 40, 60))))
for name, s in [("brightness NCC", ncc), ("context kmeans", ck),
                ("brightness+context", comb), ("bright+ctx+spec", comb3)]:
    o = np.argsort(-s)
    cells = []
    for K in (10, 20, 40, 60):
        cells.append("%5.1f%%" % (100 * valid[o[:K]].mean()))
    print("   %-18s %s" % (name, "  ".join(cells)))
print("   %-18s %s" % ("(random baseline)", "  ".join("%5.1f%%" % (100 * valid.mean()) for _ in range(4))))

res = dict(n=len(sample),
           class_counts={c: int((cls == c).sum()) for c in ["th", "smd", "no"]},
           auc={name: auc(s, valid) for name, s in rows},
           auc_combined=best[1], lambda_ctx=best[0],
           auc_combined3=best3[2], lambda3=[best3[0], best3[1]],
           valid_rate_at_K={name: {str(K): float(valid[np.argsort(-s)[:K]].mean()) for K in (10, 20, 40, 60)}
                            for name, s in [("brightness", ncc), ("context", ck),
                                            ("combined", comb), ("combined3", comb3)]})
json.dump(res, open(os.path.join(OUT, "validity_metrics.json"), "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a, (s, name) in zip(ax, [(ncc, "brightness NCC"), (ck, "context kmeans"), (comb3, "bright+ctx+spec")]):
    a.hist(s[valid], bins=18, alpha=.6, density=True, label="valid through-hole pad")
    a.hist(s[~valid], bins=18, alpha=.6, density=True, label="invalid")
    a.set_title("%s   AUC=%.3f" % (name, auc(s, valid)))
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "validity_summary.png"), dpi=130)
plt.close()
print("")
print("saved validity_metrics.json, validity_summary.png")
