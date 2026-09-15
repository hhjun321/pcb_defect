# step1 — `phase1_template.py` + `calibrate_ring_features.py` (밝기 템플릿 · ring 임계, pcb-mh 정본)

> **목적**: 실제 `missing_hole` 결함에서 공통 밝기 템플릿 `D_mean`을 만들고(RQ1),
> 후보 검증에 쓸 **ring gate 임계값**을 GT 분포에서 유도한다.
> 입력 `datasets/mh_board` → 출력 `outputs/phase1`(템플릿) + `outputs/phase2/gt_ring_features.json`(임계).
> **실행 환경**: **CPU**.
> **전제**: phase0 완료 — `datasets/mh_board/train`에 1212장 / 2328 box 존재.
> ⚠️ **crop별 z 정규화는 옵션이 아니다**: 조명 조건이 10종이라 crop 평균 밝기의 crop 간 표준편차가
> **14.6 grey level**이다. raw로 두면 조명차가 패턴차를 압도한다(단일 템플릿 R² raw 0.269 vs z 0.398).
> ⚠️ **ring/core 마스크 반경을 임의로 바꾸지 말 것**: `ring 0.24~0.34 / core ≤0.14`는 실측 radial profile에서
> 유도한 값이다. 초기에 `ring 0.32~0.47 / core ≤0.22`로 잡았다가 `contrast` median이 1.51 → 0.12로 무너져
> bullseye와 모순되는 결과가 나왔다.
> ⚠️ **재실행 시 step2~4 전부 재실행**: 템플릿과 임계가 바뀌면 후보·합성 결과가 모두 달라진다.
> **실행 순서 체인**: phase0 → **step1** → step2 → step3 → step4 → exp_yolo.

---

## STEP 0 — 공통 환경 셀 (pcb-mh 전 문서 동일 — 그대로 복사)

```python
# ===== 0-A. Drive 마운트 + 코드 클론 (전 문서 동일 — 환경 셀보다 먼저) =====
from google.colab import drive
import os
drive.mount('/content/drive')

REPO = '/content/pcb_defect'
if not os.path.isdir(REPO + '/.git'):
    !git clone https://github.com/hhjun321/pcb_defect.git {REPO}
else:
    !git -C {REPO} pull --ff-only
!ls {REPO}/src | head
```

> **코드 = 리포(`/content/pcb_defect`) · 데이터/산출물 = Drive(`/content/drive/MyDrive/pcb_defect`)**.
> 세션 재시작마다 클론 셀을 다시 돌린다(세션 로컬 디스크는 휘발). 코드를 Drive에 복사하지 않는다 —
> 버전이 갈라진다.

```python
import os, json, glob

# ===== 공통 환경 (pcb-mh 전 문서 동일 — 수정 금지) =====
os.environ['REPO']     = '/content/pcb_defect'                 # 코드(형상) — git clone 대상, 세션 로컬
os.environ['DRIVE']    = '/content/drive/MyDrive/pcb_defect'   # 데이터·산출물 루트 (업로드 경로에 맞게 여기만 수정)
os.environ['PCB_ROOT'] = os.environ['DRIVE']                   # src/*.py가 읽는 유일한 데이터 루트
os.environ['PCB_SRC']  = f"{os.environ['REPO']}/src"           # 코드는 REPO에서 읽는다 (Drive 아님)
os.environ['PCB_RAW']  = f"{os.environ['PCB_ROOT']}/pcb-defect-dataset"
os.environ['PCB_FIX']  = f"{os.environ['PCB_ROOT']}/pcb-defect-dataset-fixed"
os.environ['PCB_DS']   = f"{os.environ['PCB_ROOT']}/datasets"
os.environ['PCB_OUT']  = f"{os.environ['PCB_ROOT']}/outputs"
os.environ['PCB_RUNS'] = f"{os.environ['PCB_ROOT']}/runs"

def O(phase, *rest):
    p = f"{os.environ['PCB_OUT']}/{phase}"
    return os.path.join(p, *rest) if rest else p
def D(name):
    return f"{os.environ['PCB_DS']}/{name}"

SPLITS = ["mh_board", "mh_orig"]
ARMS   = ["random", "brightness", "context", "brightspec"]
def arms_of(split):
    return [split] + [f"{split}_{a}cp" for a in ARMS]

os.chdir(os.environ['PCB_ROOT'])
print("PCB_ROOT =", os.environ['PCB_ROOT'])
print("PCB_SRC  =", os.environ['PCB_SRC'])
```

전제 확인:

```python
from pathlib import Path
assert Path(D('mh_board'), 'train', 'images').exists(), "phase0 미완료"
print("train images =", len(glob.glob(f"{D('mh_board')}/train/images/*.jpg")))   # 1212
```

---

## STEP 1 — box 크기 분포 실측 (선택, 정규화 크기 근거)

```python
!python $PCB_SRC/analyze_box_sizes.py
```

기대 출력 (train, n=2902):

```
w        p5=19.0 p25=23.0 p50=27.0 p75=34.0 p95=45.0  max=69.0
h        p5=19.0 p25=23.0 p50=28.0 p75=34.0 p95=44.0  max=70.0
sqrt(wh) p5=19.8 p25=23.3 p50=27.4 p75=33.6 p95=43.4  max=64.8
aspect   mean=1.01 std=0.19
fit rate for fixed patch size  S= 32 -> 64.0% fit
```

해석:
- `aspect ≈ 1.01` → **정사각 정규화가 타당**하다.
- native median 27.4px를 32×32로 정규화하면 확대율 1.17배로 원본 스케일에 가깝다.
- 단, `maxwh ≤ 32`를 만족하는 box는 64%뿐이다 → **step2의 sliding window는 반드시 multi-scale**이어야 한다.

---

## STEP 2 — 밝기 템플릿 `D_mean` 생성 (RQ1)

```python
!python $PCB_SRC/phase1_template.py
```

수행 내용:
1. split leakage 재감사 출력
2. train GT crop 추출 → 32×32 grayscale
3. `D_mean`(raw) / `Dz_mean`(crop별 z 정규화) + 각 std map 저장
4. radial profile 산출
5. 결함 crop vs 무작위 배경 패치 분리도(AUC)

소요: 약 3–5분

### 2-1. 로그 대조 (실측 확정값)

```
raw crop mean brightness : grand mean=114.3, std ACROSS crops=14.6 (lighting/board spread)
Dz_mean : min=-1.02 max=0.95 center=-1.02 corner=-0.86

R^2 of template per sample: raw    median=0.269  p25=-0.002 p75=0.431
R^2 of template per sample: z-norm median=0.398  p25=0.220  p75=0.508

radial profile of Dz_mean:
   r= 0- 1 px   z=-1.00
   r= 4- 5 px   z=-0.31
   r= 6- 7 px   z=+0.38
   r= 8- 9 px   z=+0.82
   r= 9-10 px   z=+0.84      <- ring peak
   r=13-14 px   z=+0.03
   r=15-16 px   z=-0.41

AUC (lower MSE => defect) = 0.981
```

**RQ1 판정 기준**
- 중심 z가 **−1.0 부근**, r 8–10px에서 **+0.8 이상**의 링 피크 → bullseye 성립
- `R^2 z-norm > raw` (0.398 vs 0.269) → z 정규화의 필요성 확인

> ⚠️ `AUC = 0.981`은 **무작위 배경** 대비 값이라 쉬운 지표다.
> 진짜 경쟁자인 정상 pad를 negative로 쓰면 step2에서 **0.703**으로 떨어진다. 이 수치를 최종 성능으로 인용하지 말 것.

### 2-2. 그림 확인

```python
from IPython.display import Image, display
display(Image(O('phase1', 'template_summary.png')))
display(Image(O('phase1', 'crop_grid.png')))
```

- `template_summary.png` — `D_mean` / `D_std` / `Dz_mean` / `Dz_std` / radial profile / 분리도 히스토그램
- `crop_grid.png` — 무작위 64개 crop. **어두운 원판 + 밝은 링**이 균일하게 반복되어야 한다

---

## STEP 3 — ring gate 임계 캘리브레이션

```python
!python $PCB_SRC/calibrate_ring_features.py
```

GT 2194개의 ring/core 특징 분포를 측정해 게이트 임계를 유도한다.
마스크는 `ring 0.24 ≤ r/s ≤ 0.34`, `core r/s ≤ 0.14` (STEP 2의 radial profile에서 유도).

기대 출력 (실측 확정값):

```
n = 2194 GT sites
feature          p1       p5      p25      p50      p95      p99
ring_z         0.06     0.31     0.67     0.85     1.29     1.40
ring_min      -1.11    -0.96    -0.52    -0.03     0.82     1.00
core_z        -1.64    -1.37    -0.90    -0.60     0.18     0.61
contrast      -0.26     0.38     1.14     1.51     2.18     2.40
ring_sat      19.04    27.26    38.05    48.36   100.27   120.68
ring_val     118.65   128.39   143.80   153.82   180.16   192.10
core_sat      76.59   107.84   149.46   178.38   234.13   249.09
```

**결함 정의가 여기서 수치로 확정된다**: `ring_sat 48`(은색 금속) vs `core_sat 178`(초록 기판 노출).

`src/candgen.py`가 소비하는 게이트 임계:

| 상수 | 값 | 근거 |
|---|---|---|
| `G_RING_Z` | 0.31 | `ring_z` p5 |
| `G_CONTRAST` | 0.38 | `contrast` p5 |
| `G_RING_MIN` | −1.11 | `ring_min` p1 (16섹터 각도 완전성) |
| `G_RING_SAT` | 100.0 | `ring_sat` p95 (링이 금속일 것) |
| `G_RING_VAL` | 195.0 | `ring_val` p99 |
| `G_SPEC` | 30.0 | `spec` p5 (step3에서 사용) |

> `contrast` median이 **1.51 부근**으로 나와야 정상이다. 0.1대가 나오면 마스크 반경이 잘못된 것이다.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/phase1/D_mean.npy`, `D_std.npy` | raw grayscale 평균/표준편차 템플릿 (32×32) |
| `outputs/phase1/Dz_mean.npy`, `Dz_std.npy` | crop별 z 정규화 템플릿 — **매칭에 쓰는 정본** |
| `outputs/phase1/crops_train_32.npy`, `crops_train_meta.json` | 추출된 GT crop과 메타 |
| `outputs/phase1/template_summary.png`, `crop_grid.png` | 검증용 그림 |
| `outputs/phase2/gt_ring_features.json` | ring gate 임계 (step2 소비) |

---

## 판정 / 다음 단계

- [ ] `Dz_mean` radial profile — 중심 z ≈ −1.00, r 8–10px 피크 ≈ +0.84
- [ ] `R^2 z-norm(0.398) > raw(0.269)`
- [ ] `crop_grid.png`에서 bullseye가 육안으로 균일
- [ ] `gt_ring_features.json` 생성, `contrast` p50 ≈ 1.51, `ring_sat` p50 ≈ 48, `core_sat` p50 ≈ 178
- [ ] `aspect ≈ 1.01`, `sqrt(wh)` p50 ≈ 27.4

통과 시 → **step2**(`phase2_search.py`, 입력 `outputs/phase1` + `gt_ring_features.json` → 출력 `outputs/phase2`).

---

## 무결성 / 정직 (_SPEC §5)

- 템플릿은 **`mh_board` train만**으로 만든다. val/test를 열람하면 이후 recall·AUC가 전부 낙관 편향된다.
- 게이트 임계는 **GT 분위수 그대로**다. 결과를 보고 손으로 조정하지 않는다. 바꾸려면 본 스크립트를 재실행한다.
- 마스크 반경(`ring 0.24~0.34`, `core ≤0.14`)은 실측 유도값이다. 눈대중 변경 금지.
- `AUC 0.981`(무작위 배경 대비)을 최종 성능으로 인용하지 않는다. 정상 pad 대비 값은 step2에서 0.703이다.
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
