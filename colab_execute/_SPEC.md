# pcb_defect 파이프라인 정본 스펙 (SPEC) — 전 문서 공유

> 이 파일은 `colab_execute/`의 모든 실행 가이드(phase0 / step1~4 / exp_yolo)가 **공유하는 정본**이다.
> 각 문서는 아래 §1 공통 환경 셀을 **그대로 복사**하고, 자기 스테이지에 해당하는 §3 명령만 사용한다.
> **env·output 루트 규약을 문서마다 재발명 금지.**

---

## 0. 최종 규격 요약 (pcb-mh)

연구 흐름 원안(`missing_hole_research_flow.md`)에서 **두 지점이 측정 결과로 교체**됐다. 정본은 아래다.

- **결함 정의**: `missing_hole` = 구멍이 가공되지 않은 pad. 금속 환형 존재 + 중심 substrate 노출.
  실측(GT 2194): `ring_sat 48` / `core_sat 178`.
- **후보 탐색(step2)**: `D_mean`(32×32, grayscale, **crop별 z 정규화**) multi-scale NCC.
  스케일 `{20,24,28,32,36,40,48}` 7단계. **z 정규화 미적용 금지** — 조명 10종으로 crop 평균 밝기 std가 14.6이라
  raw MSE는 조명차가 패턴차를 압도한다(R² 0.269 vs 0.398).
- **후보 1차 검증(step2)**: **ring gate**. 임계는 GT ring 분포에서 유도
  (`ring_z≥0.31`, `contrast≥0.38`, `ring_min≥−1.11`, `ring_sat≤100`, `ring_val≤195`).
  후보 200→25개/이미지, recall@10 67.5→82.1%.
- **후보 2차 검증(step3)**: **specular**. `spec = V_p98 − V_p50`.
  원안의 Context(주변 구조) 검증은 **기각**됐다 — 유효 pad 판별 AUC 0.524(무작위 수준),
  brightness 결합 시 최적 가중치 **λ=0.00**. `missing_hole`이 정상 pad 위에서 발생하므로
  실제 결함 위치와 정상 pad의 주변 구조가 동일하기 때문이다.
  Context는 **ablation arm으로만 유지**한다(negative result 보고용).
- **합성(step4)**: **core transplant**. donor를 통째로 덮지 않는다.
  대상 pad의 금속 환형을 보존하고 중심 코어만 치환. 코어 색은 주변 resist × 캘리브레이션 계수
  `(B 0.740, G 0.803, R 0.889)`, box는 `pad 외곽반경 / 0.399`로 이미지에서 직접 추정.
  → 조명·형태 정합이 **구조적으로 보장**된다.
- **데이터 분할**: 물리 보드 20장이 조명 10종 × 회전/반전 × 타일로 증강된 구조라
  공개 split은 `board+light+tile` 기준 test 163/163이 train에 존재한다(**leakage**).
  `mh_orig`(공개 split)와 `mh_board`(board 분리, leakage 0)를 **둘 다 만들어 병기**한다.
- **실행 순서 체인 = phase0 → step1 → step2 → step3 → step4 → exp_yolo**
- **재수행 범위**: step1(템플릿) 재실행 시 step2~4 전부 재실행.
  step3의 수동 라벨(`labels_manual.json`)은 step2 후보 생성 규칙이 바뀌면 무효 — 재라벨 필요.

---

## 1. 공통 환경 셀 (모든 문서 STEP 0에 그대로)

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

def O(phase, *rest):                      # outputs/{phase}[/...]
    p = f"{os.environ['PCB_OUT']}/{phase}"
    return os.path.join(p, *rest) if rest else p
def D(name):                              # datasets/{name}
    return f"{os.environ['PCB_DS']}/{name}"

SPLITS = ["mh_board", "mh_orig"]          # mh_board = leakage 0 (주 결과), mh_orig = 공개 split
ARMS   = ["random", "brightness", "context", "brightspec"]   # brightspec = 제안 방법
def arms_of(split):                       # baseline 포함 5종
    return [split] + [f"{split}_{a}cp" for a in ARMS]

os.chdir(os.environ['PCB_ROOT'])
print("PCB_ROOT =", os.environ['PCB_ROOT'])
print("PCB_SRC  =", os.environ['PCB_SRC'])
print("datasets  =", sorted(os.listdir(os.environ['PCB_DS'])) if os.path.isdir(os.environ['PCB_DS']) else "(없음)")
```

> `src/*.py`는 전부 `PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")`로 루트를 읽는다.
> **환경변수만 설정하면 스크립트 수정이 불필요하다.** 소스 문자열을 sed로 치환하지 말 것.
>
> **루트 2분할 규약 (Colab 정본)**
> - `REPO = /content/pcb_defect` — 코드·형상. `git clone https://github.com/hhjun321/pcb_defect.git`.
>   세션 로컬이라 재시작마다 다시 클론/`git pull`. **Drive에 복사 금지**(버전 분기).
> - `DRIVE = PCB_ROOT = /content/drive/MyDrive/pcb_defect` — 데이터(`pcb-defect-dataset*`),
>   `datasets/`, `outputs/`, `runs/`. 실행 산출물은 전부 여기 남는다.
> - `PCB_SRC = $REPO/src`. 스크립트는 리포에서 실행되고 데이터는 Drive에서 읽/쓴다.
> - 리포에 동봉된 소형 산출물(`outputs/phase3/labels_manual.json`)은 step3에서 Drive `outputs/phase3/`로
>   **1회 복사**한다. 기존 파일이 있으면 덮지 않는다.

---

## 2. output 루트 규약 (phase-first — 반드시 준수)

전 산출물은 `outputs/{phase}` 아래. 데이터셋은 `datasets/{name}`.

| phase | 경로 `O(phase)` | 생성 문서 | 소비 문서 |
|-------|----------------|-----------|-----------|
| `phase1` | `outputs/phase1` | step1 | step2 |
| `phase2` | `outputs/phase2` | step1(ring 임계) / step2(탐색) | step2/3/4 |
| `phase3` | `outputs/phase3` | step3 | step4 |
| `phase4` | `outputs/phase4` | step4 | step4 |
| `datasets/` | `datasets/{name}` | phase0(baseline) / step4(증강) | exp_yolo |
| `runs/` | `runs/{ds}_s{seed}` | exp_yolo | 분석 |

주요 산출 파일:

| 파일 | 생성 | 소비 |
|------|------|------|
| `pcb-defect-dataset-fixed/` | phase0 | phase0(split 생성) |
| `datasets/mh_board`, `mh_orig`, `split_info.json` | phase0 | step1~4, exp_yolo |
| `outputs/phase1/D_mean.npy`, `Dz_mean.npy` | step1 | (참고) |
| `outputs/phase2/gt_ring_features.json` | step1 | step2(게이트 임계) |
| `outputs/phase2/Dz_mean_boardtrain.npy` | step2 | step3, step4 |
| `outputs/phase3/C_single.npy`, `C_kmeans.npy` | step3 | step4(context arm) |
| `outputs/phase3/labels_manual.json` | step3(수동) | step3 평가 |
| `outputs/phase4/core_calibration.json` | step4 | step4(합성) |
| `datasets/{split}_{arm}cp/` | step4 | exp_yolo |

---

## 3. 스테이지별 정본 명령 (핵심 = 변경된 부분)

### phase0 — 데이터셋 복구 + split 생성
```python
!python $PCB_SRC/build_datasets.py
```
- 라벨 파일명 버그(`*_256.txt` ↔ `*_600.jpg`) 수정. 원본 무손상, 하드링크.
- `mh_orig`(공개 split) + `mh_board`(board 분리) 동시 생성.
- 검증: `still_missing=0`, `unpaired=0`, `mh_board key=board test_seen_in_train=0/2`.

### step1 — 밝기 템플릿 + ring 임계
```python
!python $PCB_SRC/analyze_box_sizes.py         # box 크기 분포 (선택)
!python $PCB_SRC/phase1_template.py           # D_mean, RQ1
!python $PCB_SRC/calibrate_ring_features.py   # ring gate 임계
```
- `phase1_template.py`는 **crop별 z 정규화**를 전제로 한다. 끄지 말 것.
- `calibrate_ring_features.py`의 마스크 반경은 `ring 0.24~0.34 / core ≤0.14`.
  실측 radial profile에서 유도한 값이며, 임의로 바꾸면 `contrast` median이 무너진다(1.51 → 0.12 실측).

### step2 — 후보 탐색 + 게이트 ablation
```python
!python $PCB_SRC/phase2_search.py       # multi-scale NCC, RQ2
!python $PCB_SRC/eval_candidates.py     # 게이트 ablation (v1 vs v3)
!python $PCB_SRC/overlay_candidates.py  # red GT / blue 후보 시각화
```
- 템플릿은 **`mh_board` train만**으로 만든다(val/test 미열람).
- **stage A(탐지·ablation용)는 ring gate만** 적용한다. `spec`을 여기서 쓰면 step3의 기여도를
  미리 갉아먹어 ablation이 오염된다.
- 다양성 제약은 **grid-cap(100px 셀당 2개)**. min-distance는 금지 — 가까이 붙은 GT까지 삭제한다
  (recall@50 92.6→84.1% 실측).

### step3 — Context 평가 + 유효성 라벨
```python
!python $PCB_SRC/phase3_context.py        # 템플릿 생성(single/k-means) + 1차 평가
!python $PCB_SRC/phase3_label_sample.py   # 수동 라벨용 층화 표본 120개 + 몽타주 4장
# (라벨링: outputs/phase3/labels_manual.json — 리포지토리에 이미 포함)
!python $PCB_SRC/phase3_label_eval.py     # 유효성 AUC, RQ3
```
- `AUC(GT vs non-GT)`는 **잘못된 지표**다. non-GT 안에 요구가 정반대인 두 부류
  (정상 pad=통과 대상 / 실크스크린·틈=제거 대상)가 섞여 상쇄된다.
  반드시 `AUC(유효 through-hole pad vs 무효)`로 본다.
- 통과 조건: `lambda_ctx=0.00`, `bright+ctx+spec AUC≈0.918`.

### step4 — 합성 캘리브레이션 + Copy-Paste + QC
```python
!python $PCB_SRC/phase4_calibrate_core.py   # 코어 색 + pad 크기 캘리브레이션
!python $PCB_SRC/phase4_copypaste.py --splits mh_board mh_orig \
        --methods random brightness context brightspec --n-insert 2
!python $PCB_SRC/phase4_qc.py mh_board_brightspeccp mh_board
!python $PCB_SRC/verify_datasets.py
```
- **전 arm이 동일 후보 pool을 공유**한다(이미지당 템플릿 매칭 1회). 속도 ~4배이자,
  arm 간 차이가 순위 전략에서만 발생하도록 보장한다.
- 코어 크기는 `r_in 0.20 / feather 0.11`. **선형 alpha 역산으로 넓히지 말 것** —
  HSV 채도가 `(max−min)/max`라 비선형이며, 실제로 넓혀 재보니 악화됐다(sat MAD 21.6→28.0).
- `val`/`test`는 하드링크로 원본 통과. **train만 변형**한다.

### exp_yolo — downstream detection
```python
# 전 arm 동일 조건: 동일 모델·에폭·imgsz·batch·seed
model.train(data=f"{D(ds)}/data.yaml", epochs=100, imgsz=640, batch=16, seed=SEED, ...)
metrics = model.val(split='test', data=f"{D(ds)}/data.yaml")
```
- `data.yaml`의 `path`는 Windows 절대경로다. **exp_yolo STEP 1에서 반드시 재작성**한다.
- 주 결과는 `mh_board`(leakage 0). `mh_orig`는 참고로 병기.
- 주요 지표는 **Recall** — 증강 목적이 미검출 감소다.

---

## 4. 데이터셋 규약

| dataset | 구성 | train | val | test | 비고 |
|---|---|---|---|---|---|
| `mh_board` | board 분리 (train {1,3,4,6,9–20} / val {5,8} / test {2,7}) | 1212 / 2328 | 292 / 636 | 328 / 648 | **leakage 0, 주 결과** |
| `mh_orig` | 공개 split | 1477 / 2902 | 165 / 331 | 190 / 379 | board+light+tile 163/163 누수 |

증강 arm (각 split × 4):

| arm | 위치 결정 | train box (mh_board) | train box (mh_orig) |
|---|---|---|---|
| `_randomcp` | 균등 무작위 | 2328 + 2176 = 4504 | 2902 + 2682 = 5584 |
| `_brightnesscp` | NCC + ring gate | 2328 + 1884 = 4212 | 2902 + 2275 = 5177 |
| `_contextcp` | brightness top-20 → context 재정렬 | 2328 + 2215 = 4543 | 2902 + 2691 = 5593 |
| `_brightspeccp` | `z(ncc) + 1.5·z(spec)` **제안** | 2328 + 1997 = 4325 | 2902 + 2417 = 5319 |

- 전 arm **삽입 시도는 이미지당 2회로 동일**. 유도 방식은 게이트 탈락분을 거부해 실현 수가 적다.
- `random`이 제안 방법보다 box가 약 15% 많다 → 제안 방법이 이기면 데이터 양 효과가 아님이 증명된다.
- 단일 클래스 `nc: 1, names: {0: missing_hole}`.

---

## 5. 공통 무결성 / 정직 (전 문서 말미)

- **사후 튜닝 금지**: 게이트 임계(`ring_*`, `G_SPEC`), `λ_spec=1.5`, `n_insert=2`, seed, epochs를
  결과 보고 후 변경하지 않는다.
- **캘리브레이션 값은 실측 확정값**이다. `core_calibration.json`(ratio, `pad_edge_frac=0.399`),
  `gt_ring_features.json`(게이트 임계)을 눈대중으로 고치지 않는다. 바꾸려면 재캘리브레이션 스크립트를 돌린다.
- **val/test 불변**: 전 arm에서 동일해야 한다. `verify_datasets.py`가 라벨 SHA256으로 검사한다.
- **템플릿은 train만으로 만든다**. `Dz_mean_boardtrain.npy`는 `mh_board` train 전용이다.
- **stage A에 spec 게이트 금지** — step3 기여도 측정이 오염된다.
- **`mh_board` test는 보드 2장**이다. 절대값 단독 해석 금지, Δ와 시드 분산을 함께 본다.
- 수동 라벨은 n=120·단일 라벨러다. 논문에 한계로 명시한다.
- 테스트 코드 신규 작성·pytest 금지. 검증은 각 문서의 Colab 셀 실행으로 한다.
