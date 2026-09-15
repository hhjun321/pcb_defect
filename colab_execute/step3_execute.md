# step3 — `phase3_context.py` (후보 검증: Context 기각 · Specular 채택, pcb-mh 정본)

> **목적**: 후보 위치의 **타당성 검증** 방법을 결정한다. 연구 흐름 원안의 Context(주변 PCB 구조) 검증을
> 구현·측정하고, 유효 pad 판별력이 없음을 확인한 뒤 **specular(금속 정반사)** 로 교체한다.
> 입력 `outputs/phase2/Dz_mean_boardtrain.npy` → 출력 `outputs/phase3`.
> **실행 환경**: **CPU**.
> **전제**: step2 완료 — `outputs/phase2/Dz_mean_boardtrain.npy` 존재.
> ⚠️ **`AUC(GT vs non-GT)`는 잘못된 지표다**: non-GT 안에 요구가 **정반대**인 두 부류가 섞여 상쇄된다.
> ```
> non-GT peak
>  ├ 정상 pad          Context가 통과시켜야 함 (삽입 후보다)
>  └ 실크스크린/틈     Context가 걸러야 함
> ```
> 이 지표로 보면 Context 기여가 0.848 → 0.853으로 "미미한 개선"처럼 보인다. 실제로는 두 요구가 상쇄된 것이다.
> 반드시 **`AUC(유효 through-hole pad vs 무효)`** 로 평가한다.
> ⚠️ **수동 라벨은 step2 후보 생성 규칙에 종속된다**: step1/step2를 재실행해 후보가 달라지면
> `labels_manual.json`은 무효다. 재라벨이 필요하다.
> **실행 순서 체인**: phase0 → step1 → step2 → **step3** → step4 → exp_yolo.

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

> **루트 3개**: `REPO`(코드) / `WORK`(작업 데이터, 세션 로컬) / `DRIVE`(zip 원본 + outputs 영구 보관).
> 세션 재시작마다 클론 셀을 다시 돌린다(세션 로컬 디스크는 휘발). 코드를 Drive에 복사하지 않는다 —
> 버전이 갈라진다.

```python
import os, json, glob

# ===== 공통 환경 (pcb-mh 전 문서 동일 — 수정 금지) =====
os.environ['REPO']     = '/content/pcb_defect'                          # 코드(형상) — git clone, 세션 로컬
os.environ['DRIVE']    = '/content/drive/MyDrive/pcb_defect'            # 영구 보관(outputs·백업)
os.environ['PCB_ZIP']  = '/content/drive/MyDrive/data/PCB/archive.zip'  # 원본 데이터 zip (Drive)
os.environ['WORK']     = '/content/pcb_work'                            # 작업 루트 — 세션 로컬 디스크
os.environ['PCB_ROOT'] = os.environ['WORK']                             # src/*.py가 읽는 유일한 데이터 루트
os.environ['PCB_SRC']  = f"{os.environ['REPO']}/src"                    # 코드는 REPO에서 (Drive 아님)
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

os.makedirs(os.environ['PCB_ROOT'], exist_ok=True)   # WORK는 세션마다 새로 만든다
os.chdir(os.environ['PCB_ROOT'])
print("PCB_ROOT =", os.environ['PCB_ROOT'])
print("PCB_SRC  =", os.environ['PCB_SRC'])
```

```python
# ===== 0-C. 작업 루트 준비 + 원본 압축해제 (전 문서 동일) =====
import os, glob, time, subprocess

W, DR, Z = os.environ['WORK'], os.environ['DRIVE'], os.environ['PCB_ZIP']
os.makedirs(W, exist_ok=True)
os.makedirs(f"{DR}/outputs", exist_ok=True)

# outputs는 Drive에 영구 보관 — WORK/outputs를 Drive로 심볼릭 링크
lnk = f"{W}/outputs"
if os.path.islink(lnk):
    pass
elif os.path.isdir(lnk):
    raise SystemExit(f"{lnk}가 실제 디렉터리다. 내용을 {DR}/outputs로 옮기고 지운 뒤 다시 실행하라.")
else:
    os.symlink(f"{DR}/outputs", lnk)

# 원본 데이터: 세션 로컬에 없으면 zip에서 복원 (-n = 기존 파일 보존)
assert os.path.exists(Z), f"zip 없음: {Z}"
if not os.path.isdir(f"{os.environ['PCB_RAW']}/train/images"):
    t = time.time()
    subprocess.run(['unzip', '-q', '-n', Z, '-d', W], check=True)
    print("unzip %.1f분" % ((time.time() - t) / 60))

print("raw images :", len(glob.glob(f"{os.environ['PCB_RAW']}/*/images/*.jpg")))   # 10668
print("outputs    ->", os.path.realpath(lnk))
print("datasets   :", sorted(os.listdir(os.environ['PCB_DS'])) if os.path.isdir(os.environ['PCB_DS']) else "(없음)")
!df -h /content | tail -1
```

> `unzip -n`이라 이미 풀린 세션에서는 즉시 통과한다. 첫 실행은 21,337파일 / 1.19GB — **2~4분**.
> 작업 루트를 로컬(`/content`)에 두는 이유: Drive(FUSE)는 **하드링크를 지원하지 않아**
> `build_datasets.py`·`phase4_copypaste.py`가 전부 실복사로 떨어진다(데이터셋 10종 ≈ 2.2GB → Drive 압박).
> 로컬은 하드링크가 동작해 `val`/`test`가 추가 용량 없이 공유된다.

```python
# ===== 0-D. 세션 재시작 복원 — datasets 없으면 재생성 =====
# split 결정은 결정적(정렬 + board별 box 수 greedy)이라 재실행해도 동일 구성이 나온다.
if not os.path.isdir(D('mh_board') + '/train/images'):
    print("datasets 없음 → build_datasets.py 재실행")
    !python $PCB_SRC/build_datasets.py | tail -25
else:
    print("datasets OK:", sorted(os.listdir(os.environ['PCB_DS'])))
```

전제 확인:

```python
from pathlib import Path
assert Path(O('phase2', 'Dz_mean_boardtrain.npy')).exists(), "step2 미완료"
```

---

## Context 설계 (정본)

정보 분리를 엄격히 지킨다. 두 단계가 **겹치지 않는 픽셀**을 본다.

```
brightness stage : r < 0.18·CTX        결함 자체      → 후보 탐색
context stage    : 0.18 ≤ r ≤ 0.50     주변 PCB 구조  → 후보 검증
```

| 항목 | 값 |
|---|---|
| context patch | 중심 기준 `3.0 × s` 영역 → 96×96 리샘플 |
| 마스크 | 중앙 결함부(`r < 0.18`) 제외, annulus만 |
| 정규화 | annulus 내 zero-mean unit-norm 상관 (밝기 단계와 동일한 조명 불변성) |
| 템플릿 A | **single mean** — 연구 흐름 원안의 직역 |
| 템플릿 B | **k-means K=12** — 개선안 |

---

## STEP 1 — Context 템플릿 생성 + 1차 평가

```python
!python $PCB_SRC/phase3_context.py
```

소요: 약 10분

기대 출력 (실측 확정값):

```
context descriptors: (1818, 6296)
self-similarity of GT contexts:
   single mean : median=0.283  p25=0.168 p75=0.382
   kmeans K=12 : median=0.566  p25=0.402 p75=0.678

candidates scored: 3113  (GT 508 / non-GT 2605)

AUC (GT vs non-GT brightness peak):
   brightness NCC only              : 0.848
   context, single mean template    : 0.650
   context, k-means K=12            : 0.625
   brightness + context (lambda=0.25): 0.853

precision@5 per image:
   brightness       0.325
   context_kmeans   0.259
   combined         0.326
```

### 1-1. Context 모델 자체는 잘 학습됐다

`single mean 0.283` vs `k-means 0.566` — **주변 구조가 다봉분포**라는 가설이 검증됐다.
원안의 단일 평균 템플릿은 뭉개져 정보가 소실된다. **k-means가 옳은 모델링**이다(ablation 1개 확보).

```python
from IPython.display import Image, display
display(Image(O('phase3', 'context_templates.png')))
```

클러스터가 해석 가능해야 한다: `k1` 좌우 이웃 pad, `k2`/`k8` 상하 이웃, `k0`/`k7` 고립 pad, `k3`/`k6` 좌우 밀집.

### 1-2. 그런데 개선폭은 +0.005다

`0.848 → 0.853`. 이 시점에서 "Context는 미미하게 기여한다"고 결론지으면 **틀린다.**
지표가 잘못됐기 때문이다. STEP 2에서 재설계한다.

---

## STEP 2 — 평가 재설계: 유효성 수동 라벨

### 2-1. 층화 표본 생성

```python
!python $PCB_SRC/phase3_label_sample.py
```

기대 출력:

```
non-GT candidates available: 1501
sampled 120 (stratified into 4 brightness-score bands)
saved outputs/phase3/label_sheet_0.png ... label_sheet_3.png
```

stage-A top-10의 non-GT 후보를 밝기 점수 4분위로 층화해 120개를 뽑는다.

### 2-2. 라벨 정의

| 클래스 | 정의 | 판정 |
|---|---|---|
| `th` | **관통홀 pad** — 금속 환형 + 구멍. missing_hole이 물리적으로 발생 가능 | **유효** |
| `smd` | 표면실장 pad — 금속이지만 구멍이 없다 | 무효 |
| `no` | pad 아님 — 실크스크린 글자/해칭, 맨 기판, pad 사이 틈, 대형 마운팅홀 | 무효 |

### 2-3. 라벨 확인

`labels_manual.json`은 **리포(`$REPO/outputs/phase3/`)에 포함**돼 있다(n=120).
데이터 루트는 Drive이므로 **Drive outputs로 한 번 복사**해야 `phase3_label_eval.py`가 읽는다.

```python
import shutil, os
src = f"{os.environ['REPO']}/outputs/phase3/labels_manual.json"
dst = O('phase3', 'labels_manual.json')
os.makedirs(O('phase3'), exist_ok=True)
if not os.path.exists(dst):
    shutil.copy2(src, dst)          # 이미 있으면 덮지 않는다 (재라벨 보호)
print(dst, os.path.exists(dst))
```

재라벨이 필요한 경우에만 몽타주를 열어 작성한다.

```python
from IPython.display import Image, display
for i in range(4):
    display(Image(O('phase3', f'label_sheet_{i}.png')))
```

```python
man = json.load(open(O('phase3', 'labels_manual.json')))['labels']
from collections import Counter
print(Counter(man.values()), len(man))     # {'th': 75, 'smd': 12, 'no': 33}  120
```

---

## STEP 3 — 재평가 (RQ3)

```python
!python $PCB_SRC/phase3_label_eval.py
```

기대 출력 (실측 확정값):

```
th    75  (62.5%)
smd   12  (10.0%)
no    33  (27.5%)
   valid insertion sites (th) = 62.5% of what the brightness stage proposes

   by brightness-score quartile (Q1 = highest NCC):
     Q1  ncc .732..0.879   th 90.0%  smd  0.0%  no 10.0%
     Q2  ncc .672..0.730   th 76.7%  smd  3.3%  no 20.0%
     Q3  ncc .575..0.670   th 40.0%  smd 26.7%  no 33.3%
     Q4  ncc .219..0.563   th 43.3%  smd 10.0%  no 46.7%

AUC( valid through-hole pad  vs  invalid )
   brightness NCC     0.752
   context single     0.576
   context kmeans     0.524
   spec (specular)    0.874
   -core_sat          0.423
   -ring_sat          0.566
   brightness+context 0.752   (lambda=0.00)
   bright+ctx+spec    0.918   (lambda_ctx=0.00, lambda_spec=1.50)

VALID-PAD RATE @ top-K
   ranking            K=10   K=20   K=40   K=60
   brightness NCC      90.0%  85.0%  85.0%  83.3%
   context kmeans      80.0%  70.0%  70.0%  63.3%
   brightness+context  90.0%  85.0%  85.0%  83.3%
   bright+ctx+spec    100.0% 100.0%  97.5%  93.3%
   (random baseline)   62.5%  62.5%  62.5%  62.5%
```

### 3-1. 판정 — Context 기각

- `context kmeans AUC = 0.524` → **무작위 수준**
- `brightness+context`의 최적 가중치 **λ = 0.00** → 어떤 비율로 섞어도 개선이 없다

**원인**: `missing_hole`은 **pad 위에서만** 발생한다. 따라서 실제 결함 주변 구조와 정상 pad 주변 구조가
동일하다. 더구나 오검출 대상(실크스크린·틈·마운팅홀)도 pad array 한가운데에 있다.
**구조로는 구분이 불가능하다.**

Context는 **negative result**로 보고하고, `contextcp` arm으로만 유지한다.
가설을 세우고 반증한 것이므로 약점이 아니라 기여다.

### 3-2. 판정 — Specular 채택

`spec = V_p98 − V_p50` 단독 AUC **0.874**, brightness와 결합 시 **0.918**,
상위 20개 유효 pad 비율 **100%**(무작위 기준선 62.5%).

금속은 정반사 하이라이트를 만들고, 무광 실크스크린 페인트와 기판은 만들지 않는다.
**타당성을 결정하는 것은 주변 구조가 아니라 후보 자신의 재질이다.**

```
원안: Brightness (탐색) → Context 구조 (검증)
정본: Brightness (탐색) → Specular 재질 증거 (검증)
```

```python
from IPython.display import Image, display
display(Image(O('phase3', 'validity_summary.png')))
```

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/phase3/C_single.npy` | 단일 평균 context 템플릿 |
| `outputs/phase3/C_kmeans.npy` | k-means K=12 중심 — **step4 `contextcp` arm이 소비** |
| `outputs/phase3/context_metrics.json` | 1차 평가 (GT vs non-GT) |
| `outputs/phase3/label_sample.json` | 층화 표본 120개 + 특징값 |
| `outputs/phase3/labels_manual.json` | 수동 유효성 라벨 (th/smd/no) |
| `outputs/phase3/validity_metrics.json` | **RQ3 정본 지표** |
| `outputs/phase3/context_templates.png`, `context_summary.png`, `validity_summary.png` | 그림 |
| `outputs/phase3/label_sheet_0..3.png` | 라벨링 몽타주 |

---

## 판정 / 다음 단계

- [ ] `k-means self-sim 0.566 > single mean 0.283` — 다봉분포 가설 검증
- [ ] `context_templates.png` 클러스터가 해석 가능 (좌우/상하 이웃, 고립)
- [ ] `labels_manual.json` — th 75 / smd 12 / no 33
- [ ] **`lambda_ctx = 0.00`** — Context 기여 없음
- [ ] `spec AUC ≈ 0.874`, `bright+ctx+spec AUC ≈ 0.918`
- [ ] `bright+ctx+spec` 유효 pad 비율 top-20 = **100%**

통과 시 → **step4**(`phase4_copypaste.py`, 입력 `outputs/phase2` + `outputs/phase3` → 출력 `datasets/{split}_{arm}cp`).

---

## 무결성 / 정직 (_SPEC §5)

- **`AUC(GT vs non-GT)`를 RQ3의 답으로 인용하지 않는다.** 정본 지표는 `AUC(유효 pad vs 무효)`다.
- `λ_spec = 1.5`는 본 단계의 탐색 결과 확정값이다. downstream 결과를 보고 재조정하지 않는다.
- 수동 라벨은 **n=120, val split 1개, 단일 라벨러**다. 논문에 한계로 명시한다.
- Context는 기각됐지만 **arm으로 유지**한다. 삭제하면 negative result의 근거가 사라진다.
- 후보 생성 규칙이 바뀌면 `labels_manual.json`은 무효다. 재사용 금지.
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
