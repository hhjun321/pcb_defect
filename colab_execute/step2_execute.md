# step2 — `phase2_search.py` (multi-scale 후보 탐색 · 게이트 ablation, pcb-mh 정본)

> **목적**: `Dz_mean`으로 정상 PCB에서 결함 후보 위치를 탐색하고(RQ2), ring gate의 기여를 ablation으로 분리한다.
> 입력 `outputs/phase1` + `outputs/phase2/gt_ring_features.json` → 출력 `outputs/phase2`.
> **실행 환경**: **CPU**. (`cv2.matchTemplate` 기반)
> **전제**: step1 완료 — `outputs/phase2/gt_ring_features.json` 존재.
> ⚠️ **stage A에 spec 게이트 금지**: 탐지·ablation용 stage A는 **ring gate만** 쓴다.
> `spec`을 여기서 쓰면 step3에서 측정할 기여도를 밝기 단계가 미리 갉아먹어 ablation이 오염된다.
> `spec`은 step4의 삽입 후보 선정(stage B)에서만 쓴다.
> ⚠️ **다양성 제약은 grid-cap만**: 전역 min-distance는 **금지**다. 가까이 붙은 GT까지 삭제해
> recall@50이 92.6% → 84.1%로 떨어지는 것이 실측됐다.
> ⚠️ **단일 스케일 금지**: `maxwh ≤ 32`를 만족하는 GT는 64%뿐이다(step1 STEP 1). 7스케일 필수.
> **실행 순서 체인**: phase0 → step1 → **step2** → step3 → step4 → exp_yolo.

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
os.environ['DRIVE']    = '/content/drive/MyDrive/data/pcb_defect'       # 영구 보관(outputs·백업)
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
assert Path(O('phase2', 'gt_ring_features.json')).exists(), "step1 미완료"
print(json.load(open(O('phase2', 'gt_ring_features.json')))['contrast'])   # p50 ≈ 1.51
```

---

## 탐색 규격 (정본)

| 항목 | 값 | 근거 |
|---|---|---|
| 템플릿 | `Dz_mean` (**mh_board train 전용**) | val/test 미열람 |
| 스케일 | `{20, 24, 28, 32, 36, 40, 48}` | GT maxwh p1=18 ~ max=70 커버 |
| 매칭 | `cv2.matchTemplate(TM_CCOEFF_NORMED)` | window별 zero-mean unit-norm 상관 = z 정규화 MSE와 동치 |
| 스케일 합성 | 픽셀별 최대값 + argmax 스케일 | |
| NMS | 반경 `max(4, s/2)` | |
| 다양성 | grid-cap (100px 셀당 2개) | min-distance는 GT 삭제로 기각 |
| 평가 | `mh_board` val 292장 / GT 636개 | |
| GT 매칭 기준 | 중심이 GT box 내부 **and** `0.5 ≤ s/maxwh ≤ 2.0` | |

---

## STEP 1 — 후보 탐색 + hard negative 측정 (RQ2)

```python
!python $PCB_SRC/phase2_search.py
```

소요: 약 10–15분 (val 292장 × 7스케일)

기대 출력 (실측 확정값):

```
train images=1212  crops=2328
Dz center=-1.02  ring(r=9)=0.83

val images=292  GT missing_hole=636

RECALL@K  (GT found among top-K brightness candidates per image)
   K=  1 ->  31.1%
   K=  3 ->  57.1%
   K=  5 ->  62.7%
   K= 10 ->  67.5%
   K= 20 ->  84.0%
   K= 50 ->  92.6%
   K=100 ->  95.8%

rank of GT among candidates: median=3  p25=1 p75=15 p90=27  (found 622/636)

HARD NEGATIVE CHECK  (top-20 peaks that are NOT missing_hole)
NCC score  GT missing_hole : mean=0.750 median=0.774
NCC score  hard negatives  : mean=0.690 median=0.688  (n=5306)
AUC (higher NCC => missing_hole) = 0.703
```

**RQ2 판정 기준**
- `recall@50 ≥ 90%`, `GT rank median ≤ 3`
- `top-200` 내 발견 ≥ 97%

**hard negative AUC 0.703의 의미**
step1에서 무작위 배경 대비 AUC는 0.981이었다. negative를 **정상 pad**로 바꾸자 0.703으로 떨어진다.
밝기만으로는 결함과 정상 pad를 가릴 수 없다 → **후보 검증 단계(step3)가 필요하다는 정량적 근거**다.

그림 확인:

```python
from IPython.display import Image, display
display(Image(O('phase2', 'search_summary.png')))   # 템플릿 / 분리도 / recall@K
display(Image(O('phase2', 'pos_patches.png')))      # 탐지된 진짜 결함
display(Image(O('phase2', 'neg_patches.png')))      # hard negative = 정상 pad
```

`neg_patches.png`가 **정상 pad로 가득**하면 정상이다. 이것이 "밝기 템플릿은 pad-like 위치를 찾는다"는 증거다.

---

## STEP 2 — 게이트 ablation

```python
!python $PCB_SRC/eval_candidates.py
```

소요: 약 15–20분 (4개 구성 × val 292장)

기대 출력 (실측 확정값):

```
[v1: no gate]                        GT=636  avg candidates/img=200.0
   recall@K : K=1:31.1%  K=10:67.5%  K=20:83.8%  K=50:92.6%  K=100:95.8%
   GT rank  : median=3 p75=15 p90=27  found=622/636
[v3: ring gate]                      GT=636  avg candidates/img=25.1
   recall@K : K=1:33.8%  K=10:82.1%  K=20:87.3%  K=50:90.9%  K=100:91.0%
   GT rank  : median=2 p75=4 p90=10  found=579/636
[v3: ring gate + spec gate]          GT=636  avg candidates/img=22.8
   recall@K : K=1:33.5%  K=10:76.7%  K=20:82.1%  K=50:85.2%  K=100:85.4%
[v3: ring + spec + grid-cap(100px,2)] GT=636  avg candidates/img=16.2
   recall@K : K=1:33.5%  K=10:77.0%  K=20:81.4%  K=50:84.0%  K=100:84.0%
```

### 판정 — 무엇을 채택하는가

| 구성 | cand/img | recall@10 | recall@50 | GT rank median | 채택 |
|---|---|---|---|---|---|
| 게이트 없음 | 200 | 67.5% | 92.6% | 3 | — |
| **ring gate** | **25** | **82.1%** | **90.9%** | **2** | **stage A** |
| ring + spec | 23 | 76.7% | 85.2% | 2 | — |
| ring + spec + grid-cap | 16 | 77.0% | 84.0% | 2 | **stage B**(step4) |

- **stage A(탐지·ablation)** = ring gate만. 후보가 1/8로 줄면서 recall@10이 **+14.6%p**.
- **stage B(삽입 후보)** = ring + spec + grid-cap. step4에서만 쓴다.
- `spec`을 stage A에 넣으면 recall@50이 90.9 → 85.2%로 떨어지고, 무엇보다 step3의 측정 대상을 미리 지워버린다.

---

## STEP 3 — 후보 시각화 (오검출 유형 확인)

```python
!python $PCB_SRC/overlay_candidates.py
```

```python
from IPython.display import Image, display
display(Image(O('phase2', 'candidate_overlay.png')))   # red=GT, blue=후보 top-20
display(Image(O('phase2', 'candidate_zoom.png')))      # 후보 확대
```

여기서 확인되는 **오검출 3종**(step3/step4가 처리할 대상):

| 유형 | 원인 |
|---|---|
| 공간 쏠림 | NMS가 국소 억제만 해 고득점 pad 열이 top-K를 독식 → **grid-cap으로 해결** |
| 실크스크린 글자 | `5V`, `Support`, `GND` 등. `o`·`e`·`P`는 **흰 획이 링, 속공간이 초록 코어** 역할 |
| pad 사이 틈 / 빈 기판 | 주변 pad들이 링, 틈이 코어 역할 |

> 시도했으나 **기각된 판별자**(재시도 금지):
> - **회전대칭 s90/s180** — GT median 0.34인데 FP가 0.4~0.78로 오히려 더 대칭적이다.
>   GT box는 pad에 타이트해 타원/비대칭 요소가 들어가는 반면 글자 글리프는 해당 스케일에서 대칭적으로 보인다.
> - **`ring_sat` 단독으로 실크스크린 판별** — 흰 페인트도 저채도라 금속과 구분되지 않는다.

---

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/phase2/Dz_mean_boardtrain.npy` | **step3·step4가 소비하는 정본 템플릿** |
| `outputs/phase2/search_metrics.json` | recall@K, hard negative AUC |
| `outputs/phase2/recall_v3.json` | 게이트 ablation 결과 |
| `outputs/phase2/search_summary.png` | 템플릿 / 분리도 / recall@K |
| `outputs/phase2/pos_patches.png`, `neg_patches.png` | 탐지 결함 / hard negative |
| `outputs/phase2/candidate_overlay.png`, `candidate_zoom.png` | 후보 시각화 |

---

## 판정 / 다음 단계

- [ ] `recall@50 ≈ 92.6%` (게이트 없음), `GT rank median = 3`
- [ ] ring gate 적용 시 `cand/img 200 → 25`, `recall@10 67.5 → 82.1%`, `rank median 3 → 2`
- [ ] hard negative `AUC ≈ 0.703` — 0.9대가 나오면 negative를 잘못 잡은 것이다
- [ ] `neg_patches.png`가 정상 pad로 채워져 있음
- [ ] `Dz_mean_boardtrain.npy` 생성

통과 시 → **step3**(`phase3_context.py`, 입력 `outputs/phase2/Dz_mean_boardtrain.npy` → 출력 `outputs/phase3`).

---

## 무결성 / 정직 (_SPEC §5)

- 템플릿은 `mh_board` **train 전용**이다. 평가는 val에서만 한다.
- **stage A에 spec 게이트 금지** — step3 기여도 측정이 오염된다.
- 다양성 제약은 grid-cap만 쓴다. min-distance 재도입 금지(recall −8.5%p 실측).
- 게이트 임계는 step1의 GT 분위수 확정값이다. 결과를 보고 조정하지 않는다.
- `AUC 0.981`(무작위 배경)과 `AUC 0.703`(정상 pad)을 혼용해 인용하지 않는다. 후자가 유효 수치다.
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
