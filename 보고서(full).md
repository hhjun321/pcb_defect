# pcb-mh 전체 진행 보고서 (full)

> **목적**: 학습(exp_yolo) 직전까지의 모든 진행 내용을 기록한다. 최초 설계에서 출발해
> **무엇이 틀렸고 · 어떻게 측정했고 · 무엇으로 교체했는지**를 순서대로 남긴다. 각 단계를 다시 공부할 때 참고한다.
> **대상 독자**: 파이프라인을 재현하거나 파라미터를 바꾸려는 사람.
> **최종 결과만 필요하면** → `보고서(simple).md` (논문 작성용, 시행착오 제외).
> **실행 절차는** → `colab_execute/_SPEC.md` + 단계별 `*_execute.md`.
> **실행 순서 체인**: phase0 → step1 → step2 → step3 → step4 → exp_yolo.
> **범위**: 로컬 작업은 step4(합성 데이터셋 생성)에서 끝난다. exp_yolo는 클라우드에서 수행한다.
> ⚠️ 본 문서의 모든 수치는 **실측 확정값**이다. 추정치·예상치는 명시적으로 구분해 표기한다.

---

## 0. 최초 설계와 정본 설계

### 0.1 최초 설계 (`missing_hole_research_flow.md`)

```text
결함 자체의 밝기 특성 추출
   → 정상 PCB 후보 탐색 (Brightness Search)
   → 주변 PCB 구조 검증 (Context Matching)
   → 실제 결함 Copy-Paste
   → YOLO 성능 평가
```

### 0.2 정본 설계

```text
결함 자체의 밝기 특성 추출                          step1
   → 정상 PCB 후보 탐색 (Brightness Search + Ring Gate)   step2
   → 후보 재질 검증 (Specular Verification)         step3   ← Context에서 교체
   → 실제 결함 Core Transplant                      step4   ← 통째 Copy-Paste에서 교체
   → YOLO 성능 평가                                 exp_yolo
```

### 0.3 교체 2건 요약

| 지점 | 원안 | 정본 | 교체 근거 (실측) |
|---|---|---|---|
| 후보 검증 | 주변 PCB 구조 (Context) | 후보 재질 (Specular) | Context AUC 0.524(무작위 수준), 최적 λ=0.00 / Specular AUC 0.874, 결합 0.918 |
| 결함 합성 | donor crop 통째 붙여넣기 | 대상 pad 금속 보존 + 코어만 이식 | 통째 방식 sat MAD 32.4 → core 이식 21.6 |

### 0.4 단계 ↔ 실행 문서 대응

| 단계 | 내용 | 실행 문서 | 본 보고서 절 |
|---|---|---|---|
| phase0 | 데이터셋 복구 + split 생성 | `colab_execute/phase0_execute.md` | §1, §2 |
| step1 | 밝기 템플릿 `D_mean` + ring 임계 | `step1_execute.md` | §3 |
| step2 | multi-scale 탐색 + 게이트 ablation | `step2_execute.md` | §4 |
| step3 | Context 기각 + Specular 채택 | `step3_execute.md` | §5 |
| step4 | Core Transplant 합성 + QC | `step4_execute.md` | §6, §7 |
| exp_yolo | YOLO 학습·평가 | `exp_yolo_execute.md` | §9 (RQ4, 미수행) |

---

## 1. phase0 — 데이터 확인

### 1.1 데이터셋 구성

```text
pcb-defect-dataset/
  train/  images 8534  labels 8534
  val/    images 1066  labels 1066
  test/   images 1068  labels 1068
```

6 클래스: `0 mouse_bite, 1 spur, 2 missing_hole, 3 short, 4 open_circuit, 5 spurious_copper`
클래스별 train box 수는 2731~2980으로 균형이 잡혀 있다.

### 1.2 이미지 1장 = 1클래스

```text
train 8534장 중
  missing_hole 포함 : 1477장  (전부 missing_hole 단독)
  missing_hole 없음 : 7057장
missing_hole box 개수/이미지 : 1개 595 / 2개 475 / 3개 287 / 4개 104 / 5개 16
```

한 이미지에 여러 클래스가 섞이지 않는다. 따라서 missing_hole 연구는
1832장(train 1477 / val 165 / test 190), box 3612개의 **단일 클래스 문제**로 다룬다.

### 1.3 이미지 크기 혼재

train 샘플 171장 중 129장이 600×600, 42장이 601×601이다.
정규화 좌표를 픽셀로 바꿀 때 **이미지마다 W, H를 읽어야 한다.** 하드코딩하면 어긋난다.

### 1.4 box 크기 실측 (train, n=2902)

| 지표 | p5 | p25 | **p50** | p75 | p95 | max |
|---|---|---|---|---|---|---|
| w | 19 | 23 | **27** | 34 | 45 | 69 |
| h | 19 | 23 | **28** | 34 | 44 | 70 |
| sqrt(wh) | 19.8 | 23.3 | **27.4** | 33.6 | 43.4 | 64.8 |

- **aspect = 1.01 (std 0.19)** → 거의 정사각. 정사각 정규화가 타당하다.
- val/test 분포도 동일(median 26~28)해 split 편향은 없다.
- 6클래스 중 중간 크기. open_circuit 최소(21), short 최대(36).

### 1.5 32×32 정규화 판단

native median 27.4px를 32×32로 리사이즈하면 확대율 1.17배로 거의 원본 스케일이다. 원안의 32×32는 적절하다.

다만 `maxwh ≤ 32`를 만족하는 box는 **64%뿐**(p95=46)이다.
→ **sliding window는 multi-scale이어야 한다.** 단일 스케일이면 큰 hole을 놓친다.
정본은 scale `{20, 24, 28, 32, 36, 40, 48}` 7단계다.

---

## 2. phase0 — 데이터셋 결함 2건: 발견과 수정

두 건 모두 최초 설계에 없던 문제다. 수정하지 않으면 이후 모든 실험이 무의미하다.

### 2.1 라벨 파일명 불일치 (box 24% 유실)

**증상**: 이미지 `..._600.jpg`에 대응하는 라벨이 `..._256.txt`로 저장돼 있었다.
YOLO는 파일 stem으로 짝을 맺으므로, 이 이미지들은 **라벨 없는 배경**으로 학습된다.

| split | 라벨 없는 이미지 | 그중 missing_hole |
|---|---|---|
| train | 2164 / 8534 (25%) | 366 |
| val | 264 / 1066 | 47 |
| test | 239 / 1068 | 45 |

missing_hole box 2902개 중 **705개(24%)가 버려지고 있었다.**

**검증 3단계**

1. `_600 → _256` 치환으로 짝이 맞는가 → train 2164/2164, val 264/264, test 239/239 **전부 해소**
2. box 기하가 같은가 → `_256` 라벨 w_med 27.0 / p5 19.0 / p95 44.6, `_600`은 27.0 / 19.0 / 44.9. 좌표 범위 이탈 0건
3. 실제로 결함 위에 놓이는가 → `_256` 라벨을 `_600` 이미지에 렌더링 → 정확히 missing hole 위에 안착
   (`outputs/phase1/label_rename_check.png`)

**조치**: 원본을 건드리지 않고 `pcb-defect-dataset-fixed/`를 만든다.
라벨은 이름을 고쳐 복사, 이미지는 하드링크. 결과 `unpaired=0`,
missing_hole box **train 2902 / val 331 / test 379 전량 복구**.

### 2.2 Split Leakage

**증상**: 파일명을 파싱하면 구조가 드러난다.

```text
[rotation_{90|270}_][l_]light_{01..12}_missing_hole_{01..20}_{tile}_600
```

물리 보드는 **20장뿐**이다. 10가지 조명 × {원본, 좌우반전(`l_`), 90° 회전, 270° 회전} × 타일 1~5로 증강돼 있다.
그런데 split은 이 증강본 단위로 무작위 분할됐다.

| 키 | test 중 train에도 존재 |
|---|---|
| board | 20 / 20 (100%) |
| board + light | 93 / 93 (100%) |
| **board + light + tile** | **163 / 163 (100%)** |
| + rotation | 76 / 187 (41%) |
| + rotation + flip | 0 / 190 (0%) |

즉 train과 test가 **동일 장면의 회전/반전 쌍**이다. test 성능이 부풀려진다.

**조치**: split을 둘 다 만들어 병기한다.

| 데이터셋 | 구성 | leakage |
|---|---|---|
| `mh_orig` | 공개 split 유지 | board+light+tile 163/163 |
| `mh_board` | board 단위 분리 | **0 / 82** |

`mh_board` 보드 배정 (box 수 70/15/15 목표 greedy, **고정값**):

```text
train : 1, 3, 4, 6, 9, 10, 11~20   → 1212장 / 2328 box
val   : 5, 8                        →  292장 /  636 box
test  : 2, 7                        →  328장 /  648 box
```

보드별 규모 편차가 크다는 점은 기록해 둔다.

```text
board 01~10 : 이미지 140~172장, box 288~348
board 11~20 : 이미지  20~ 32장, box  32~ 52
```

**한계**: `mh_board`는 test 보드가 2장뿐이라 통계적 변동이 크다.
논문에서는 두 split 결과를 병기하고 `mh_board`를 보수적 추정으로 서술한다.

---

## 3. step1 — 밝기 템플릿과 ring 임계

### 3.1 결함의 물리적 정체 확정

초반에는 "missing_hole은 어둡다" 수준의 이해만 있었다.
GT box 주변을 확대(`outputs/phase2/gt_context_zoom.png`)해 정체를 확정했다.

| | 외형 |
|---|---|
| 정상 pad | 은색 금속 전체가 균일하게 밝음, 정반사 하이라이트 |
| **missing_hole** | 은색 링 + **중앙이 초록색(기판 노출)** = 구멍 미가공 |

수치로도 확정된다 (GT 2194개, box 한 변 기준 반경):

| feature | p5 | **p50** | p95 |
|---|---|---|---|
| ring_z | 0.31 | **0.85** | 1.29 |
| core_z | −1.37 | **−0.60** | 0.18 |
| contrast | 0.38 | **1.51** | 2.18 |
| ring_sat | 27 | **48** | 100 |
| core_sat | 108 | **178** | 234 |
| ring_val | 128 | **154** | 180 |

`ring_sat 48`(금속) vs `core_sat 178`(초록 기판). **결함 정의가 수치로 확정됐다.**

> **시행착오 — 마스크 반경**
> ring/core 마스크를 처음에 `ring 0.32~0.47`, `core ≤0.22`로 잡았다.
> 그 결과 `contrast` median이 **0.12**로 나와 bullseye와 모순됐다.
> 실측 radial profile(§3.2)과 대조해 `ring 0.24~0.34`, `core ≤0.14`로 고치자 **1.51**로 정상화됐다.
> **마스크는 실측 프로파일에서 잡아야 한다.** `step1_execute.md` STEP 3에 임계표로 고정.

### 3.2 `D_mean` — 공통 밝기 템플릿 (RQ1)

각 GT crop을 32×32로 리사이즈, grayscale 변환, **crop별 z 정규화** 후 평균.

```text
radial profile of Dz_mean
 r=0-2px   z = -1.00   어두운 중심
 r=4px     z = -0.31
 r=6px     z = +0.38
 r=8-10px  z = +0.84   ← 밝은 링 (peak)
 r=13px    z = +0.03
 r=15px    z = -0.41
```

완전한 **bullseye(어두운 원판 + 밝은 환형)**. 64개 무작위 샘플 육안 확인에서도 매우 균일하다
(`outputs/phase1/crop_grid.png`).

**RQ1 답: 성립.**

### 3.3 조명 정규화가 필수인 이유

crop 평균 밝기의 crop 간 표준편차가 **14.6 grey level**이다. 조명 조건이 10가지이기 때문이다.

단일 템플릿의 샘플별 설명력 R²:

| 방식 | median | p25 | p75 |
|---|---|---|---|
| raw | 0.269 | −0.002 | 0.431 |
| **z 정규화** | **0.398** | 0.220 | 0.508 |

raw로 MSE를 계산하면 조명 차이가 패턴 차이를 압도한다.
따라서 매칭은 **window별 z 정규화 후** 수행한다.
구현상 `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`가 정확히 이 연산이다.

---

## 4. step2 — 후보 탐색

### 4.1 Multi-scale NCC 탐색 (RQ2)

- 템플릿: `mh_board` **train만**으로 만든 `Dz_mean` (1212장 / 2328 crop)
- 스케일: `{20, 24, 28, 32, 36, 40, 48}` — 각 스케일로 템플릿을 리사이즈해 매칭
- 스케일별 응답을 픽셀 최대값으로 합성, NMS로 피크 추출
- 평가: `mh_board` val 292장 / GT 636개

```text
recall@K   K=1  31.1%   K=3  57.1%   K=5  62.7%   K=10 67.5%
           K=20 83.8%   K=50 92.6%   K=100 95.8%
GT 순위    median 3, p75 15, p90 27
top-200 내 발견 622/636 (97.8%)
```

**RQ2 답: 성립.** 다만 원안의 "Top 10"은 recall 67.5%로 부족하다. K=20~50이 적정.

### 4.2 첫 hard negative 측정 — 0.981의 함정

step1에서 "결함 crop vs 무작위 배경 패치" AUC가 **0.981**로 나왔다.
그러나 이건 쉬운 negative다. 진짜 경쟁자는 **정상 pad**다.

탐색 결과의 top-20 피크 중 GT가 아닌 것(n=5306)을 hard negative로 두고 다시 재면:

```text
NCC  GT missing_hole : mean 0.750  median 0.774
NCC  hard negative   : mean 0.690  median 0.688
AUC = 0.703          ← 0.981에서 폭락
```

**교훈**: negative를 무엇으로 잡느냐가 지표를 좌우한다. 무작위 배경은 아무것도 증명하지 못한다.
`step2_execute.md` 무결성 절에 "두 수치를 혼용 인용 금지"로 고정.

### 4.3 후보 시각화로 드러난 오검출 3종

red=GT / blue=후보 top-20 오버레이(`outputs/phase2/candidate_overlay.png`)와
확대 몽타주에서 세 가지 문제가 보였다.

1. **공간 쏠림** — top-20이 한쪽 pad 열 하나에 몰림. NMS가 국소 억제만 하기 때문
2. **실크스크린 글자 오검출** — `5V`, `Support`, `GND` 등 흰 글자.
   `o`, `e`, `P` 같은 글자는 **흰 획이 링, 속공간이 초록 코어** 역할을 해 bullseye를 흉내낸다
3. **빈 기판 / pad 사이 틈** — 주변 pad들이 링, 틈이 코어 역할

### 4.4 Ring Gate — 시도한 판별자 전부 기록

GT ring 분포(§3.1)를 임계값으로 삼아 게이트를 만들었다.

| 판별자 | 결과 | 채택 |
|---|---|---|
| `ring_z ≥ 0.31`, `contrast ≥ 0.38`, `ring_min ≥ −1.11` (16섹터 각도 완전성) | 후보 200→25개, recall@10 67.5→82.1% | 채택 |
| `ring_sat ≤ 100` (링=금속) | 초록 기판 얼룩 제거 | 채택 |
| `ring_val ≤ 195` | 과노출 백색 억제 | 채택 |
| **회전대칭 s90 / s180** | **실패.** GT median 0.34인데 FP가 0.4~0.78로 더 대칭적 | 기각 |
| **`ring_sat` 단독으로 실크스크린 판별** | **실패.** 흰 페인트도 저채도 | 기각 |
| `spec = V_p98 − V_p50` (금속 정반사) | 실크스크린 FP의 55% 제거, GT 손실 5% | stage B 한정 |

**회전대칭이 실패한 이유**: GT box는 pad에 타이트해 타원/비대칭 요소가 들어가는 반면,
글자 글리프는 해당 스케일에서 오히려 대칭적으로 보인다.

기각된 두 판별자는 `step2_execute.md` STEP 3에 **재시도 금지**로 명시했다.

### 4.5 게이트 ablation과 2단계 분리

| 구성 | cand/img | recall@10 | recall@50 | GT rank median |
|---|---|---|---|---|
| 게이트 없음 | 200 | 67.5% | 92.6% | 3 |
| **ring gate** | **25** | **82.1%** | **90.9%** | **2** |
| ring + spec | 23 | 76.7% | 85.2% | 2 |
| ring + spec + grid-cap | 16 | 77.0% | 84.0% | 2 |

**채택 구조 — stage A / stage B 분리**

- **stage A (탐지·ablation용)**: ring gate만. 밝기 단계의 순수 성능을 보존한다.
  `spec`을 여기서 쓰면 step3의 기여도를 미리 갉아먹어 **ablation이 오염된다.**
- **stage B (삽입 후보)**: ring + spec + grid-cap. step4에서만 쓴다.

### 4.6 다양성 제약 — min-distance 기각

처음엔 후보 간 **최소거리** `max(40px, 2.0×s)`를 걸었다.
공간 쏠림은 해결됐으나 **가까이 붙은 GT까지 삭제**했다.

```text
recall@50  92.6% → 84.1%
발견 GT    622/636 → 535/636
```

**교체**: grid-cap (100px 셀당 최대 2개). 전역 삭제 없이 분산만 강제한다.
`step2_execute.md` 무결성 절에 "min-distance 재도입 금지(recall −8.5%p 실측)"로 고정.

---

## 5. step3 — Context 기각, Specular 채택

### 5.1 설계

정보 분리를 엄격히 지켰다. 두 단계가 **겹치지 않는 픽셀**을 본다.

```text
brightness stage : r < 0.18·CTX        결함 자체      → 후보 탐색
context stage    : 0.18 ≤ r ≤ 0.50     주변 PCB 구조  → 후보 검증
```

- context patch = 중심 기준 `3.0 × s` 영역을 96×96으로 리샘플, 중앙 결함부 마스킹
- annulus 내부만 zero-mean unit-norm 상관 → 밝기 단계와 동일한 조명 불변성
- 템플릿 2종: **single mean**(원안) vs **k-means K=12**(개선안)

### 5.2 Context 모델 자체는 잘 학습됐다

GT context descriptor 1818개 (차원 6296).

| 모델 | GT self-similarity median |
|---|---|
| single mean | 0.283 |
| **k-means K=12** | **0.566** |

클러스터가 해석 가능하다 (`outputs/phase3/context_templates.png`):
`k1` 좌우 이웃 pad, `k2`/`k8` 상하 이웃, `k0`/`k7` 고립 pad, `k3`/`k6` 좌우 밀집.

**주변 구조가 다봉분포**라는 가설이 검증됐다. 원안의 단일 평균 템플릿은 뭉개져 정보가 소실된다.
→ **k-means가 옳은 모델링.** (ablation 1개 확보)

### 5.3 1차 평가 — 개선 없음

`AUC(GT vs non-GT brightness peak)`, `mh_board` val, 후보 3113개(GT 508 / non-GT 2605):

| scoring | AUC |
|---|---|
| brightness NCC only | 0.848 |
| context, single mean | 0.650 |
| context, k-means | 0.625 |
| brightness + context | **0.853** (λ=0.25) |

precision@5: brightness 0.325 / context 0.259 / combined 0.326.

개선폭 **+0.005**. 사실상 0이다.

### 5.4 원인 — 지표가 틀렸다

missing_hole은 **pad 위에서만** 발생한다. 따라서

```text
GT 결함 주변 구조  ≈  정상 pad 주변 구조     (둘 다 pad array)
```

Context는 이 둘을 구분할 수 없고, **구분해서도 안 된다.** 정상 pad는 우리가 원하는 삽입 후보다.

`GT vs non-GT`는 non-GT 안에 **요구가 정반대인 두 부류**를 섞어놓은 지표다.

```text
non-GT peak
 ├ 정상 pad          Context가 통과시켜야 함 (삽입 후보)
 └ 실크스크린/틈     Context가 걸러야 함
```

두 요구가 같은 AUC 안에서 상쇄된다.
`step3_execute.md` 헤더에 "`AUC(GT vs non-GT)`는 잘못된 지표"로 경고 고정.

### 5.5 평가 재설계 — 수동 라벨

stage-A top-10의 non-GT 후보 1501개에서 밝기 점수 4분위로 층화 추출한 120개를 직접 라벨링했다.

| 클래스 | 정의 | 개수 |
|---|---|---|
| `th` | 관통홀 pad — 금속 환형 + 구멍. missing_hole 발생 가능 → **유효** | 75 (62.5%) |
| `smd` | 표면실장 pad — 금속이지만 구멍 없음 → 무효 | 12 (10.0%) |
| `no` | pad 아님 — 실크스크린/기판/마운팅홀/틈 → 무효 | 33 (27.5%) |

밝기 점수 사분위별 구성:

| | ncc 범위 | th | smd | no |
|---|---|---|---|---|
| Q1 | .732~.879 | 90.0% | 0.0% | 10.0% |
| Q2 | .672~.730 | 76.7% | 3.3% | 20.0% |
| Q3 | .575~.670 | 40.0% | 26.7% | 33.3% |
| Q4 | .219~.563 | 43.3% | 10.0% | 46.7% |

### 5.6 재평가 — Context 기각, Specular 채택 (RQ3)

`AUC(유효 관통홀 pad vs 무효)`:

| 판별자 | AUC |
|---|---|
| brightness NCC | 0.752 |
| context, single mean | 0.576 |
| **context, k-means** | **0.524** (무작위 수준) |
| **spec (금속 정반사)** | **0.874** |
| −core_sat | 0.423 |
| −ring_sat | 0.566 |
| brightness + context | 0.752 (**λ_ctx = 0.00**) |
| **brightness + 1.5·spec** | **0.918** |

유효 pad 비율 @ top-K:

| 랭킹 | K=10 | K=20 | K=40 | K=60 |
|---|---|---|---|---|
| 무작위 기준선 | 62.5% | 62.5% | 62.5% | 62.5% |
| brightness | 90.0% | 85.0% | 85.0% | 83.3% |
| context k-means | 80.0% | 70.0% | 70.0% | 63.3% |
| brightness + context | 90.0% | 85.0% | 85.0% | 83.3% |
| **brightness + spec** | **100.0%** | **100.0%** | **97.5%** | **93.3%** |

**결론**

삽입 위치의 타당성을 결정하는 것은 주변 구조가 아니라 **후보 자신의 재질**이다.
금속은 정반사 하이라이트를 만들고, 무광 실크스크린 페인트와 기판은 만들지 않는다.

```text
원안: Brightness (탐색) → Context 구조 (검증)
정본: Brightness (탐색) → Specular 재질 증거 (검증)
```

Context는 **negative result로 보고**하고 `contextcp` arm으로 유지한다.
"결함은 정상 pad와 동일한 구조적 문맥에서 발생하므로, 문맥은 삽입 위치 타당성을 판별하지 못한다."
가설을 세우고 반증한 것이므로 약점이 아니라 기여다. **삭제하면 근거가 사라진다.**

**한계**: n=120, val split 1개, 단일 라벨러.

---

## 6. step4 — Copy-Paste 합성

### 6.1 붙여넣기 1차 시도와 실패

**1차 방식**: donor crop(금속 링 + 초록 코어) 전체를 target에 리사이즈해 덮고,
ring 밴드 통계로 채널별 mean/std를 전이.

**QC 결과 실패**:

- ring std 비율 스케일링이 폭주 → 분홍/백색 후광, 채도 과다
- feather 경계가 밝은 테두리로 보임
- 원형 donor를 타원형 pad에 붙여 형태 불일치

radial profile MAD: saturation **32.4**, value **22.8**.

### 6.2 근본 재설계 — Core Transplant

missing_hole의 물리적 정의는 *pad는 그대로, 구멍만 없음*이다.
그런데 1차 방식은 donor의 금속 링까지 통째로 옮겨 조명·형태 불일치를 자초했다.

```text
1차: donor patch 전체(금속링+초록코어) → target에 덮어쓰기 + 통계 전이
정본: target pad의 금속링 그대로 보존
      donor의 초록 코어만 중심부에 이식
      코어 색상은 target 이미지의 로컬 기판색에서 샘플링
```

조명·형태 정합이 **구조적으로 보장된다.** 타겟 자신의 픽셀을 쓰기 때문이다.
통계 전이가 불필요해 폭주도 없다. 타원 pad에도 자연스럽게 맞는다.

### 6.3 캘리브레이션 — 추측 대신 실측

2차 방식 QC에서 두 가지 계통 오차가 남았다.

**① 코어 색상 과밝음** (val 124 vs 실제 105)
pad **바깥** resist 색을 그대로 썼기 때문이다. 실제 코어는 pad 개구부 안쪽이라 더 진하고 어둡다.

GT 900개에서 `core / resist` 비를 실측:

```text
ratio  B 0.740   G 0.803   R 0.889
offset B −14.0   G −29.0   R −2.0
```

→ `core = resist_median × ratio` 적용.

**② box 크기 불일치**
합성 `r/s 0.24~0.36`의 채도가 27 (실제 50), `0.44~0.48`이 95 (실제 153).
템플릿의 이산 스케일 `{20,24,...,48}`을 그대로 box로 썼기 때문에 pad 대비 너무 타이트했다.

GT 900개에서 금속 링 외곽이 끝나는 반경을 채도 프로파일로 측정:

```text
r_edge / box_side : median 0.399
→ box_side = pad_외곽반경 / 0.399
```

합성 시 pad에서 직접 반경을 추정(2회 반복 수렴)하도록 교체.

### 6.4 Feather 폭 — 선형 alpha 역산은 무효

GT radial profile에서 alpha를 역산하면 더 넓은 램프가 나온다.

```text
r/s    real_sat  → alpha = (sat − 49) / (245 − 49)
0.02     239        0.97
0.10     187        0.70
0.14     143        0.48
0.18     102        0.27
0.22      68        0.10
→ 선형 피팅 r_in = 0.245, width = 0.225
```

실제로 만들어 재보니 **악화**됐다.

| 설정 | sat MAD | val MAD | core sat MAD |
|---|---|---|---|
| donor 통째 + ring 통계전이 | 32.4 | 22.8 | 21.2 |
| **core 이식 (r_in 0.20, w 0.11)** | **21.6** | **11.4** | 22.9 |
| core 이식 (r_in 0.245, w 0.225) | 28.0 | 10.7 | 44.9 |

**원인**: HSV 채도는 `(max − min) / max`다.
초록 코어를 밝은 금속과 섞으면 V가 올라가며 S가 비선형으로 급락한다.
**alpha 선형 가정이 성립하지 않는다.**

0.20 / 0.11이 코어 내부를 거의 정확히 재현한다 (253/243/181 vs 실제 239/223/187).
→ 0.20 / 0.11 유지. `step4_execute.md` 헤더에 "선형 alpha 역산 재시도 금지"로 고정.

### 6.5 최종 합성 파라미터 (정본)

```text
donor      = 같은 split TRAIN 풀의 실제 missing_hole crop
             (같은 이미지 제외, 크기비 0.6~1.7 매칭)
코어 색    = 주변 resist median × (B 0.740, G 0.803, R 0.889)
코어 텍스처 = donor core 편차의 60% 반영
코어 크기  = r_in 0.20, feather 0.11
box 크기   = pad 외곽반경 / 0.399  (이미지에서 직접 추정, 2회 반복 수렴)
삽입 개수  = 2 / 이미지 (시도 기준, 전 arm 동일)
seed       = arm별 0~3
```

annotation은 삽입 site box 자체이므로 **라벨 오차가 0**이다.

### 6.6 단일 패스 재구조화

초기 구현은 method마다 donor pool을 재생성하고 템플릿 매칭을 반복해 8종 생성에 과도한 시간이 걸렸다.

**개선**: 이미지당 템플릿 매칭을 1회만 수행하고, 4종이 **동일 후보 pool**을 서로 다른 방식으로 정렬만 한다.

- 속도 약 4배
- **ablation이 더 깨끗해진다**: 모든 arm이 같은 pool에서 선택하므로 차이가 순위 전략에서만 발생한다

부수 최적화: radial map, bin index map을 크기별 캐싱.

### 6.7 실험군 4종

| arm | 위치 결정 | spec 게이트 |
|---|---|---|
| `random` | 균등 무작위, 크기는 GT 분포에서 추출, 겹침만 회피 | — |
| `brightness` | stage-A NCC + ring gate + grid cap | 없음 |
| `context` | brightness top-20 → context 유사도 재정렬 (원안 Step 6/7 직역, ablation) | 없음 |
| `brightspec` | `z(ncc) + 1.5·z(spec)` ← **제안 방법** | **적용** |

---

## 7. step4 — 합성 품질 검증

육안이 아니라 **radial profile로 측정**한다.
코어가 작다 / 색이 다르다 / 후광이 있다 같은 실패는 전부 프로파일 불일치로 나타난다.

### 7.1 실제 GT 기준 프로파일

| r/s | sat | val | 영역 |
|---|---|---|---|
| 0.00–0.04 | 231.6 | 116.2 | 초록 코어 중심 |
| 0.12–0.16 | 127.9 | 128.0 | 반값 지점 |
| 0.24–0.32 | 47.6 | 155.2 | 금속 링 |
| 0.44–0.48 | 153.8 | 118.7 | 바깥 resist |

### 7.2 최종 결과

`mh_board`:

| arm | sat MAD | val MAD | core sat MAD | core val MAD |
|---|---|---|---|---|
| context | **19.0** | 11.0 | 23.9 | 8.0 |
| brightness | 21.0 | 12.4 | 23.1 | 4.9 |
| brightspec | 21.6 | 11.4 | 22.9 | **4.8** |
| **random** | **108.3** | 8.6 | 44.2 | 3.1 |

`mh_orig`: context 19.0 / brightspec 20.9 / brightness 21.0 / **random 108.7** — 동일 경향.

### 7.3 Random CP의 실패가 수치로 드러남

```text
r/s        real   random
0.00-0.04  239.0   251.5
0.12-0.16  143.7   227.3
0.24-0.28   52.9   228.7   ← 금속 링이 있어야 할 자리
0.44-0.48  153.2   224.0
```

random은 **전 반경 채도 224~251로 평평**하다.
맨 기판에 초록 원판을 붙인 것에 불과하며 **금속 링이 아예 없다.**
물리적으로 missing_hole이 아닌 물체를 결함으로 라벨링하고 있다.
sat MAD 108.3은 유도 방식(19~22)의 **5배**다.

### 7.4 유도 3종의 정합과 잔차

일치 구간(`brightspec`):

```text
             real   synth
r/s 0.00-0.04  239.0  253.2   코어 중심
r/s 0.08-0.12  186.7  181.0
r/s 0.32-0.36   56.8   59.2   pad 바깥
r/s 0.44-0.48  152.2  156.4   resist
```

잔차 구간 `r/s 0.16~0.28`: 합성 링이 실제보다 순수 금속에 가깝다 (sat 29 vs 53).
링 **위치는 정확히 일치**(최저점 둘 다 r/s 0.24–0.28)하므로 기하 오차가 아니다.
실제 결함 pad의 링이 초록을 더 머금고 있는 것으로 보인다.
feather 확대로 보정을 시도했으나 악화가 확인(§6.4)돼 현 설정을 유지한다.

전체 MAD 19~22 / 255 ≈ **8%**.
유도 3종이 서로 2.6 이내로 근접하므로 **삽입 위치만 다르고 결함 렌더링 품질은 동일**하다.
실험군 간 비교에서 렌더링 품질이 **교란변수로 작용하지 않음이 보장**된다.

---

## 8. 최종 산출물과 검증

### 8.1 데이터셋 10종

```text
datasets/mh_board                    baseline (증강 없음)
datasets/mh_board_randomcp
datasets/mh_board_brightnesscp
datasets/mh_board_contextcp
datasets/mh_board_brightspeccp       ← 제안 방법
datasets/mh_orig                     baseline
datasets/mh_orig_randomcp
datasets/mh_orig_brightnesscp
datasets/mh_orig_contextcp
datasets/mh_orig_brightspeccp
```

각각 `data.yaml`(단일 클래스 `0: missing_hole`)과
`synthesis_meta.json`(seed, 삽입 수, 파라미터)을 포함한다.

### 8.2 무결성 검증 (`src/verify_datasets.py`)

```text
ALL CHECKS PASSED
```

- 10개 데이터셋 × 3 split: `unpaired = 0`, 좌표 범위 이탈 0, 클래스 오류 0
- **val/test 라벨 SHA256이 전 arm 동일** — 학습 분포만 다름을 해시로 보증
- train box 수 = baseline + `synthesis_meta.json` 기록값, 전부 일치

### 8.3 Train box 예산

| dataset | real | synthetic | total | 평균 크기 px |
|---|---|---|---|---|
| mh_board_randomcp | 2328 | **2176** | 4504 | 32.0 |
| mh_board_brightnesscp | 2328 | 1884 | 4212 | 34.9 |
| mh_board_contextcp | 2328 | 2215 | 4543 | 32.8 |
| mh_board_brightspeccp | 2328 | **1997** | 4325 | 35.3 |
| mh_orig_randomcp | 2902 | **2682** | 5584 | 31.7 |
| mh_orig_brightnesscp | 2902 | 2275 | 5177 | 34.9 |
| mh_orig_contextcp | 2902 | 2691 | 5593 | 32.9 |
| mh_orig_brightspeccp | 2902 | **2417** | 5319 | 35.5 |

삽입 **시도**는 전 arm 2회/이미지로 동일하다.
유도 방식은 게이트를 통과하지 못한 자리를 거부해 실현 수가 적다.

→ **random이 제안 방법보다 box가 약 15% 많다.**
그럼에도 제안 방법이 이기면 "데이터 양 덕분"이라는 반론이 원천 봉쇄된다.
**결과 보고 시 반드시 함께 명시한다.**

### 8.4 용량

```text
mh_board*   204M + 153M × 4
mh_orig*    3.1M + 184M × 4
합계        약 1.4 GB (baseline은 하드링크 공유)
```

---

## 9. 핵심 연구 질문 답변

### RQ1 — missing_hole에 공통 밝기 패턴이 존재하는가?

**성립.** 중심 z −1.00, r 8–10px에서 링 피크 +0.84의 bullseye.
단, **crop별 z 정규화가 전제**다 (R² 0.269 → 0.398).
→ §3.2, `step1_execute.md`

### RQ2 — 그 패턴으로 정상 PCB에서 후보를 찾을 수 있는가?

**성립.** multi-scale NCC로 recall@50 = 90.9%, GT 순위 median 2.
ring gate 적용 시 후보가 이미지당 200 → 25개로 줄면서 recall@10이 67.5 → 82.1%로 오른다.
→ §4.1, §4.5, `step2_execute.md`

### RQ3 — 주변 PCB 구조를 추가로 고려하면 더 적절한 위치를 고를 수 있는가?

**기각.** Context의 유효 pad 판별 AUC는 0.524(무작위 수준),
brightness와 결합해도 최적 가중치가 λ = 0이다.
결함이 정상 pad와 동일한 구조적 문맥에서 발생하기 때문이다.

**대체 답**: 위치 타당성은 주변이 아니라 후보 자신의 **재질**로 판별된다.
정반사 특징 `spec` 단독 AUC 0.874, brightness와 결합 시 **0.918**,
상위 20개 유효 pad 비율 **100%**.
→ §5.6, `step3_execute.md`

### RQ4 — 위치 인식 Copy-Paste가 Random보다 YOLO 성능을 높이는가?

**미수행 — `exp_yolo_execute.md`에서 검증한다.**
학습 전 단계에서 확보한 근거:

- Random은 sat MAD 108.3으로 금속 링이 없는 비결함을 결함으로 라벨링한다 (§7.3)
- 유도 3종은 sat MAD 19~22로 실제와 정합하며 서로 2.6 이내 (§7.2)
- Random이 제안 방법보다 학습 box를 15% 더 갖는다 (§8.3)

해석 규칙은 `exp_yolo_execute.md` §해석 규칙에 **사전 고정**돼 있다. 결과를 보고 바꾸지 않는다.

---

## 10. 한계

- 유효성 라벨: n = 120, val split 1개, 단일 라벨러
- `mh_board` test는 보드 2장뿐
- `r/s 0.16~0.28` 잔차: 합성 링이 실제보다 순수 금속에 가깝다 (sat 29 vs 53). 링 위치는 정확히 일치하므로 기하 오차는 아니다
- `spec` 특징은 이 데이터셋의 조명 조건에 캘리브레이션돼 있다. 다른 촬영 환경에서는 `phase4_calibrate_core.py`, `calibrate_ring_features.py`로 재캘리브레이션이 필요하다
- 물리 보드가 20장뿐이라 board 단위 분할의 자유도가 근본적으로 제한된다

---

## 11. 스크립트 구성과 재현

```text
src/
  build_datasets.py            phase0  라벨 파일명 수정 + split 2종 생성
  analyze_box_sizes.py         step1   box 크기 분포 실측
  phase1_template.py           step1   D_mean 밝기 템플릿, RQ1
  calibrate_ring_features.py   step1   GT ring 통계 → 게이트 임계값
  candgen.py                   step2~4 후보 생성기 (공용 모듈)
  phase2_search.py             step2   multi-scale NCC 탐색, RQ2
  eval_candidates.py           step2   게이트 ablation
  overlay_candidates.py        step2   red GT / blue 후보 오버레이
  phase3_context.py            step3   context 템플릿, single vs k-means
  phase3_label_sample.py       step3   수동 라벨용 층화 표본 생성
  phase3_label_eval.py         step3   유효성 AUC, RQ3
  phase4_calibrate_core.py     step4   코어 색상 + pad 크기 캘리브레이션
  phase4_copypaste.py          step4   합성, 전 arm 단일 패스
  phase4_qc.py                 step4   radial profile 현실성 검증
  verify_datasets.py           step4   최종 무결성 검사
```

전 스크립트는 `PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")`로 루트를 읽는다.
**환경변수만 설정하면 소스 수정 없이 Colab에서 실행된다.**

실행 순서:

```text
phase0 → step1 → step2 → step3 → (수동 라벨링) → step4 → exp_yolo
```

**재수행 범위**
- step1 재실행 → step2~4 전부 재실행
- step2 후보 생성 규칙 변경 → `labels_manual.json` 무효, 재라벨 필요
- step4 파라미터 변경 → exp_yolo 재실행

상세 절차는 `colab_execute/_SPEC.md` 및 단계별 `*_execute.md`를 본다.

---

## 12. 무결성 / 정직 (`_SPEC` §5)

- **사후 튜닝 금지**: 게이트 임계(`ring_*`, `G_SPEC`), `λ_spec = 1.5`, `n_insert = 2`, seed, epochs를
  결과 보고 후 변경하지 않는다.
- **캘리브레이션 값은 실측 확정값**이다. `core_calibration.json`(ratio, `pad_edge_frac = 0.399`),
  `gt_ring_features.json`(게이트 임계)을 눈대중으로 고치지 않는다. 바꾸려면 재캘리브레이션 스크립트를 돌린다.
- **val/test 불변**: 전 arm에서 동일해야 한다. `verify_datasets.py`가 라벨 SHA256으로 검사한다.
- **템플릿은 train만으로 만든다**. `Dz_mean_boardtrain.npy`는 `mh_board` train 전용이다.
- **stage A에 spec 게이트 금지** — step3 기여도 측정이 오염된다.
- **기각된 판별자 재시도 금지**: 회전대칭(GT 0.34 < FP 0.4~0.78), `ring_sat` 단독 실크스크린 판별,
  min-distance 다양성 제약(recall −8.5%p), feather 선형 alpha 역산(MAD +6.4).
- **Context arm 삭제 금지**: 기각됐지만 negative result의 근거다.
- **`mh_board` test는 보드 2장**이다. 절대값 단독 해석 금지, Δ와 시드 분산을 함께 본다.
- **두 AUC를 혼용 인용 금지**: 0.981(무작위 배경 대비)은 쉬운 지표다. 유효 수치는 0.703(정상 pad 대비)이다.
- 수동 라벨은 n=120·단일 라벨러다. 논문에 한계로 명시한다.
- 테스트 코드 신규 작성·pytest 금지. 검증은 각 execute 문서의 Colab 셀 실행으로 한다.
