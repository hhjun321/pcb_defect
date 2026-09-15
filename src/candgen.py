"""
Brightness candidate generator (v3).

v1 was plain multi-scale NCC on D_mean.  Three failure modes were measured on
the demo images and fixed here:

  FIX 1  spatial clustering    -> GRID CAP (<= grid_cap peaks per grid_cell px
                                  cell).  A global minimum-distance rule was
                                  tried first and rejected: it also deletes true
                                  sites that legitimately sit close together
                                  (recall 92.6% -> 84.1% @K=50).
  FIX 2  silkscreen-text FP    -> ring-validity gate calibrated on the GT ring
                                  feature distribution, plus a specular gate
                                  (metal pads carry a highlight, matte white
                                  paint does not).  Partial: removes ~55% of
                                  silkscreen FPs.  Rotational symmetry and ring
                                  saturation were both tried and DO NOT
                                  separate (white paint is desaturated too;
                                  glyphs are more symmetric than tight GT
                                  boxes).  Residual FPs are left for the
                                  Context stage - that is what it is for.
  FIX 3  substrate-blob FP     -> chroma gate, ring must be metal (low sat).

Two stages, deliberately separated:

  stage A  detect()   brightness NCC peaks + ring gate.  Used for the RQ2
                      recall metric and as the Context-stage ablation baseline,
                      so the spec gate is OFF by default here: the brightness
                      stage must not pre-remove the FPs whose removal is the
                      Context stage's measured contribution.
  stage B  sites()    stage A + spec gate + grid cap, minus anything
                      overlapping a GT box, minus anything whose core is green
                      substrate -> normal pads = Copy-Paste insertion sites.

Gate thresholds come from outputs/phase2/gt_ring_features.json
(src/calibrate_ring_features.py regenerates it).
"""
import os
import collections
import numpy as np
import cv2

S = 32
SCALES = [20, 24, 28, 32, 36, 40, 48]

# gates calibrated from GT ring feature distribution (outputs/phase2/gt_ring_features.json)
G_RING_Z = 0.31      # GT p5
G_CONTRAST = 0.38    # GT p5
G_RING_MIN = -1.11   # GT p1   (angular completeness)
G_RING_SAT = 100.0   # GT p95  (ring must be metal, not green substrate)
G_RING_VAL = 195.0   # GT p99  (reject blown-out white silkscreen paint)
G_CORE_SAT = 110.0   # insertion site only: core must NOT be green -> normal pad
G_SPEC = 30.0        # GT p5 of (V_p98 - V_p50): metal pads carry a specular
                     # highlight, matte silkscreen paint does not

_MC = {}


def _masks(s):
    if s in _MC:
        return _MC[s]
    yy, xx = np.mgrid[0:s, 0:s]
    c = (s - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / s
    ang = (np.arctan2(yy - c, xx - c) + np.pi) / (2 * np.pi)
    ring = (r >= 0.24) & (r <= 0.34)
    core = r <= 0.14
    sect = np.clip((ang * 16).astype(int), 0, 15)
    _MC[s] = (ring, core, sect)
    return _MC[s]


def features(col, x, y, s):
    H, W = col.shape[:2]
    h2 = s // 2
    x0, y0 = x - h2, y - h2
    if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
        return None
    bgr = col[y0:y0 + s, x0:x0 + s]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    z = (g - g.mean()) / (g.std() + 1e-6)
    ring, core, sect = _masks(s)
    V = hsv[..., 2]
    spec = float(np.percentile(V, 98) - np.percentile(V, 50))
    rz, cz = float(z[ring].mean()), float(z[core].mean())
    sm = [float(z[ring & (sect == k)].mean()) for k in range(16) if (ring & (sect == k)).sum()]
    return dict(ring_z=rz, core_z=cz, contrast=rz - cz,
                ring_min=float(np.min(sm)) if sm else -9.0,
                ring_sat=float(hsv[..., 1][ring].mean()),
                ring_val=float(hsv[..., 2][ring].mean()),
                core_sat=float(hsv[..., 1][core].mean()),
                spec=spec)


def ring_ok(ft):
    return (ft is not None
            and ft["ring_z"] >= G_RING_Z
            and ft["contrast"] >= G_CONTRAST
            and ft["ring_min"] >= G_RING_MIN
            and ft["ring_sat"] <= G_RING_SAT
            and ft["ring_val"] <= G_RING_VAL)


def make_templates(Dz):
    return {s: cv2.resize(Dz.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
            for s in SCALES}


def _peaks(gray, TM, pool):
    H, W = gray.shape
    best = np.full((H, W), -2.0, np.float32)
    bs = np.zeros((H, W), np.int32)
    for s in SCALES:
        r = cv2.matchTemplate(gray, TM[s], cv2.TM_CCOEFF_NORMED)
        full = np.full((H, W), -2.0, np.float32)
        o = s // 2
        full[o:o + r.shape[0], o:o + r.shape[1]] = r
        m = full > best
        best[m] = full[m]
        bs[m] = s
    flat = best.ravel()
    n = min(len(flat) - 1, pool)
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(-flat[idx])]
    taken = np.zeros((H, W), bool)
    out = []
    for i in idx:
        y, x = divmod(int(i), W)
        if taken[y, x]:
            continue
        s = int(bs[y, x])
        rad = max(4, s // 2)
        taken[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = True
        out.append((float(best[y, x]), x, y, s))
    return out


def detect(col, TM, topn=200, use_ring_gate=True, need_spec=False,
           grid_cell=0, grid_cap=2, pool=200000):
    """stage A: brightness peaks (+ ring gate).

    Spatial diversity uses a GRID CAP (at most `grid_cap` candidates per
    `grid_cell` px cell) rather than a global minimum distance: a min-distance
    rule also deletes true sites that legitimately sit close together, which
    cost 14% recall in v2.
    Returns [(score, x, y, s, feat)].
    """
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kept = []
    cell = collections.Counter()
    for sc, x, y, s in _peaks(gray, TM, pool):
        ft = features(col, x, y, s)
        if use_ring_gate and not ring_ok(ft):
            continue
        if need_spec and (ft is None or ft["spec"] < G_SPEC):
            continue
        if grid_cell > 0:
            key = (x // grid_cell, y // grid_cell)
            if cell[key] >= grid_cap:
                continue
            cell[key] += 1
        kept.append((sc, x, y, s, ft))
        if len(kept) >= topn:
            break
    return kept


def sites(col, TM, gts, topn=20, grid_cell=100, grid_cap=2):
    """stage B: insertion sites = normal pads (not an existing defect, core not green)"""
    out = []
    for sc, x, y, s, ft in detect(col, TM, topn=topn * 12, need_spec=True,
                                  grid_cell=grid_cell, grid_cap=grid_cap):
        if any(abs(x - cx) <= w / 2 and abs(y - cy) <= h / 2 for cx, cy, w, h in gts):
            continue
        if ft["core_sat"] > G_CORE_SAT:
            continue
        out.append((sc, x, y, s, ft))
        if len(out) >= topn:
            break
    return out


def read_gts(lp, W, H):
    gts = []
    for line in open(lp):
        f = line.split()
        if len(f) == 5:
            cx, cy, w, h = [float(v) for v in f[1:]]
            gts.append((cx * W, cy * H, w * W, h * H))
    return gts
