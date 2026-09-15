# Brightness-Guided Copy-Paste Augmentation for PCB Missing-Hole Detection

Local work ends at dataset synthesis. YOLO training and evaluation run in the cloud.

---

## 1. What to train on

Eight training sets plus two baselines. `val`/`test` are byte-identical across every
arm of a given split, so only the training distribution differs.

| dataset | placement strategy |
|---|---|
| `datasets/mh_board` | baseline, no augmentation |
| `datasets/mh_board_randomcp` | uniform random position |
| `datasets/mh_board_brightnesscp` | brightness template NCC |
| `datasets/mh_board_contextcp` | brightness top-20, re-ranked by context template |
| `datasets/mh_board_brightspeccp` | brightness + specular feature ← **proposed** |
| `datasets/mh_orig*` | same four arms on the original split |

Each carries its own `data.yaml` (single class, `0: missing_hole`) and
`synthesis_meta.json` (seed, insertion count, parameters).

### Which split to report

`mh_orig` keeps the split that ships with the dataset. It leaks: every test
`board+light+tile` combination also appears in train, differing only by a
rotation or flip, so test scores are optimistic.

`mh_board` is board-disjoint (train boards 1,3,4,6,9–20 / val 5,8 / test 2,7) and
has zero leakage, but only two test boards.

Report both. `mh_board` is the conservative estimate.

### Train box budget

Every arm attempts 2 insertions per image; guided arms realise fewer because they
refuse sites that fail their validity gate.

| dataset | real | synthetic | total |
|---|---|---|---|
| mh_board_randomcp | 2328 | 2176 | 4504 |
| mh_board_brightnesscp | 2328 | 1884 | 4212 |
| mh_board_contextcp | 2328 | 2215 | 4543 |
| mh_board_brightspeccp | 2328 | 1997 | 4325 |
| mh_orig_randomcp | 2902 | 2682 | 5584 |
| mh_orig_brightnesscp | 2902 | 2275 | 5177 |
| mh_orig_contextcp | 2902 | 2691 | 5593 |
| mh_orig_brightspeccp | 2902 | 2417 | 5319 |

Random ends up with ~15% MORE training boxes than the proposed method. If the
proposed method still wins, the gain cannot be attributed to data volume.

### Metrics to collect

mAP@0.5, mAP@0.5:0.95, precision, recall, F1. The class of interest is the only
class, so per-class AP is the overall AP. Recall matters most: the point of the
augmentation is to stop missing holes being missed.

---

## 2. Two dataset defects found and fixed

**Label filenames.** 2164 of 8534 train images (and 264 val / 239 test) had their
label file saved as `*_256.txt` while the image is `*_600.jpg`. YOLO pairs by
stem, so those images trained as empty background. For `missing_hole` this
discarded 705 of 2902 boxes — 24%. Coordinates were fine; only the name was
wrong, verified by rendering the `_256` labels onto the `_600` images. Fixed
non-destructively in `pcb-defect-dataset-fixed/` (labels renamed, images
hardlinked; the original tree is untouched).

**Split leakage.** There are only 20 physical boards, photographed under 10
lighting conditions and augmented with rotation/flip/tiling. The shipped split
randomises over those variants:

| key | test keys also in train |
|---|---|
| board | 20/20 |
| board+light | 93/93 |
| board+light+tile | 163/163 |

`mh_board` re-splits by board to remove this.

---

## 3. Method, and what the measurements changed

### The defect

A `missing_hole` is a pad whose hole was never drilled: the metal annulus is
present, the centre shows bare green substrate. Measured on 2194 real sites —
ring saturation 48, core saturation 178.

### Stage 1 — brightness search (works)

`D_mean`, a 32×32 mean of z-normalised defect crops, is a clean bullseye: centre
z −1.00, ring peak +0.84 at r 8–10 px. Multi-scale NCC over scales 20–48 finds
real defects with recall 90.9% @ K=50, median GT rank 2.

Per-crop z-normalisation is required, not optional: a single template explains
R² 0.269 of a raw crop but 0.398 of a z-normalised one, because crop mean
brightness varies by ±14.6 grey levels across lighting conditions.

A ring-validity gate calibrated on the GT ring statistics cuts candidates from
200 to 25 per image and lifts recall@10 from 67.5% to 82.1%.

### Stage 2 — context matching (does not work; kept as an ablation)

The research plan verifies candidates by comparing surrounding PCB structure.
Built both the single mean template of the plan and a 12-cluster k-means version.
The k-means model is clearly the better description of context — GT self-similarity
0.566 vs 0.283, and the clusters are interpretable (neighbours left/right,
neighbours above/below, isolated pad). It still does not help:

| scoring | AUC, valid pad vs invalid |
|---|---|
| brightness NCC | 0.752 |
| context, single mean | 0.576 |
| context, k-means | 0.524 |
| brightness + context | 0.752 (optimal λ = 0) |

Missing holes occur **on pads**, so a real defect site and a normal pad have the
same surroundings by construction. Worse, the false positives — silkscreen
glyphs, inter-pad gaps, mounting holes — sit inside pad arrays too. Structure
cannot separate them.

Note this only became visible after fixing the evaluation. Scored as
AUC(GT vs non-GT) the answer looks like a wash (0.848 → 0.853), because the
non-GT pool mixes normal pads (which must be KEPT — they are the insertion
targets) with non-pads (which must be REJECTED), and the two requirements
cancel. 120 hand-labelled candidates separate them.

### Stage 2 replaced — specular verification (works)

What distinguishes a pad from silkscreen paint is not its surroundings but its
material: metal produces a specular highlight, matte paint does not.
`spec = V_p98 − V_p50`:

| scoring | AUC | valid-pad rate @ top-20 |
|---|---|---|
| random baseline | — | 62.5% |
| brightness NCC | 0.752 | 85.0% |
| brightness + context | 0.752 | 85.0% |
| **brightness + 1.5·spec** | **0.918** | **100.0%** |

Rotational symmetry and ring saturation were also tried and rejected: white
silkscreen is desaturated like metal, and glyphs are more rotationally symmetric
than a tight GT box (GT median 0.34 vs FP 0.4–0.78).

### Stage 3 — pasting

Not "resize a donor crop over the site". A missing hole means the pad stays and
the hole is absent, so the target pad's own metal annulus is kept and only the
green core is transplanted. Core colour comes from the solder resist around the
target pad, scaled by a calibrated factor (B 0.740, G 0.803, R 0.889) because the
core sits in the pad opening and is darker than open resist. Box side is
re-estimated from the pad itself (`pad_outer_radius / 0.399`) rather than taken
from the discrete template scale.

---

## 4. Synthesis QC

Realism is measured, not eyeballed: the radial saturation/value profile of every
inserted defect is compared against real ones on the same images
(`src/phase4_qc.py`).

| arm | sat MAD | val MAD |
|---|---|---|
| contextcp | 19.0 | 11.0 |
| brightnesscp | 21.0 | 12.4 |
| brightspeccp | 21.6 | 11.4 |
| **randomcp** | **108.3** | 8.6 |

The three guided arms land within 2.6 of each other, so defect rendering quality
is not a confound between them — only placement differs.

Random's profile is flat at saturation 224–251 across every radius: it pastes a
green disc onto bare board with no metal ring at all. Those objects are not
missing holes, and they are labelled as if they were.

Pasting parameters were chosen by measurement. Full history:

| configuration | sat MAD | val MAD | core sat MAD |
|---|---|---|---|
| donor crop + ring-statistics transfer | 32.4 | 22.8 | 21.2 |
| core transplant, calibrated (0.20 / 0.11) | **21.6** | **11.4** | 22.9 |
| core transplant, wider feather (0.245 / 0.225) | 28.0 | 10.7 | 44.9 |

The wider feather came from inverting the real saturation profile as a linear
alpha blend. It measured worse. HSV saturation is (max−min)/max, so blending a
green core with bright metal is strongly non-linear and the inversion is invalid.

---

## 5. Limitations

- Validity labels: n = 120, one val split, single annotator.
- `mh_board` test is only two boards.
- Residual profile mismatch at r/s 0.16–0.28: the synthetic ring is more purely
  metallic than the real one (saturation 29 vs 53). Ring position matches
  exactly, so this is not a geometry error.
- The specular feature is calibrated on this dataset's illumination. Another
  imaging setup needs recalibration via `src/phase4_calibrate_core.py` and
  `src/calibrate_ring_features.py`.

---

## 6. Files

```
src/
  build_datasets.py            label-filename fix + mh_orig / mh_board splits
  analyze_box_sizes.py         box size distribution
  phase1_template.py           D_mean brightness template, RQ1
  calibrate_ring_features.py   GT ring statistics -> gate thresholds
  candgen.py                   candidate generator (shared module)
  phase2_search.py             multi-scale NCC search, RQ2
  eval_candidates.py           gate ablation
  overlay_candidates.py        red GT / blue candidate overlays
  phase3_context.py            context templates, single vs k-means
  phase3_label_sample.py       stratified sample for manual labelling
  phase3_label_eval.py         validity AUC, RQ3
  phase4_calibrate_core.py     core colour + pad size calibration
  phase4_copypaste.py          synthesis, all arms in one pass
  phase4_qc.py                 radial profile realism check
  verify_datasets.py           final integrity check

outputs/phase1..4/             templates (.npy), metrics (.json), figures (.png)
datasets/                      the ten training sets
pcb-defect-dataset-fixed/      6-class corrected dataset (originals untouched)
pcb-defect-dataset/            original, never modified
```

Reproduce in order: `build_datasets` → `phase1_template` →
`calibrate_ring_features` → `phase2_search` → `phase3_context` →
`phase3_label_sample` → (label) → `phase3_label_eval` →
`phase4_calibrate_core` → `phase4_copypaste` → `phase4_qc` → `verify_datasets`.
