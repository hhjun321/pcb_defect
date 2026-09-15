# Brightness- and Specular-Aware Copy-Paste Augmentation for PCB Missing Hole Detection

> **목적**: 논문 작성 참고용. 확정된 방법과 수치만 담는다. 시행착오·교체 이력은 제외한다.
> **시행착오 포함 버전** → `보고서(full).md`
> **실행 절차** → `colab_execute/_SPEC.md` + 단계별 `*_execute.md`
> **실행 순서 체인**: phase0 → step1 → step2 → step3 → step4 → exp_yolo
> **현재 진척**: step4(합성 데이터셋 생성)까지 완료. exp_yolo는 미수행.
> ⚠️ 본 문서의 모든 수치는 **실측 확정값**이다.

---

## 요약

PCB `missing_hole` 결함의 합성 데이터 증강에서, 결함을 삽입할 위치를 결정하는 방법을 제안한다.
실제 결함의 밝기 특성으로 후보 위치를 탐색하고, 후보의 금속 정반사 특성으로 타당성을 검증한 뒤,
대상 pad의 금속 환형을 보존한 채 결함 코어만 이식한다.
학습 전 단계에서 제안 방법은 상위 20개 후보의 유효 pad 비율 **100%**,
합성 결함의 radial profile 오차 **8%**를 달성했다.

---

## 1. 대상과 데이터

### 1.1 결함 정의

`missing_hole`은 구멍이 가공되지 않은 pad다. 금속 환형(annular ring)은 존재하고,
중심에는 기판 substrate가 노출된다.

실제 결함 2194개 측정:

| 영역 | 채도 (HSV S) | 명도 (HSV V) |
|---|---|---|
| 금속 링 (r/s 0.24–0.34) | 48 | 154 |
| 초록 코어 (r/s ≤ 0.14) | 178 | 116 |

### 1.2 데이터셋

PCB 결함 데이터셋, 6클래스. 이미지 1장에 단일 클래스만 존재한다.
`missing_hole` 포함 이미지는 1832장, box 3612개다.

box 크기 (train, n = 2902):

| 지표 | p5 | p50 | p95 |
|---|---|---|---|
| w (px) | 19 | 27 | 45 |
| h (px) | 19 | 28 | 44 |
| aspect | 0.8 | **1.01** | 1.3 |

aspect가 1에 수렴하므로 정사각 정규화가 타당하다.
median 27.4px를 32×32로 정규화하면 확대율 1.17배로 원본 스케일에 가깝다.

### 1.3 데이터 분할

물리 보드 20장이 10가지 조명 × 회전/반전 × 타일로 증강된 구조다.
공개 split은 증강본 단위로 무작위 분할돼 있어, 동일 장면(`board+light+tile`)이
train과 test에 함께 존재한다(163/163).

따라서 두 가지 split을 구성해 병기한다.

| 데이터셋 | 구성 | train | val | test | leakage |
|---|---|---|---|---|---|
| `mh_orig` | 공개 split | 1477장 / 2902 box | 165 / 331 | 190 / 379 | 163/163 |
| `mh_board` | board 단위 분리 | 1212장 / 2328 box | 292 / 636 | 328 / 648 | **0/82** |

`mh_board` 보드 배정: train {1,3,4,6,9–20} / val {5,8} / test {2,7}.

`mh_board`를 주 결과로, `mh_orig`를 참고로 서술한다.

---

## 2. 제안 방법

```text
Missing Hole 결함 수집
   → 크기 정규화 32×32 · grayscale · crop별 z 정규화        step1
   → 공통 밝기 템플릿 D_mean 생성                            step1
   → 정상 PCB multi-scale NCC 탐색  (후보 생성)              step2
   → Ring Gate                       (구조 검증)             step2
   → Specular Verification           (재질 검증)             step3
   → Core Transplant                 (결함 합성)             step4
   → Synthetic Dataset → YOLO 학습 → 성능 평가              exp_yolo
```

### 2.1 밝기 템플릿 `D_mean` (step1)

각 결함 영역을 32×32로 정규화하고 grayscale 변환 후, **crop별 z 정규화**를 거쳐 평균한다.

$$D_{mean}(x,y) = \frac{1}{N}\sum_{i=1}^{N} \frac{D_i(x,y) - \mu_i}{\sigma_i}$$

crop별 z 정규화는 필수다. 조명 조건이 10가지여서 crop 평균 밝기의 표준편차가 14.6 grey level에 달한다.
단일 템플릿의 샘플별 설명력 R²는 raw 0.269, z 정규화 0.398이다.

### 2.2 후보 탐색 (step2)

정상 영역에 대해 multi-scale 정규화 상호상관(NCC)을 수행한다.
스케일은 `{20, 24, 28, 32, 36, 40, 48}` 7단계이며, 각 스케일별 응답을 픽셀 최대값으로 합성한 뒤
NMS로 피크를 추출한다.

$$\text{score}(p) = \max_{s \in S} \; \text{NCC}\big(W_p^{(s)},\; D_{mean}^{(s)}\big)$$

`TM_CCOEFF_NORMED`는 window별 zero-mean unit-norm 상관이므로,
z 정규화된 MSE와 동치이며 조명 불변성을 갖는다.

공간 다양성은 grid-cap(100px 셀당 최대 2개)으로 확보한다.

### 2.3 Ring Gate (step2)

실제 결함의 ring 통계를 임계값으로 삼아 후보를 1차 검증한다.

| 조건 | 임계값 | 근거 |
|---|---|---|
| `ring_z` | ≥ 0.31 | GT p5 |
| `contrast` = ring_z − core_z | ≥ 0.38 | GT p5 |
| `ring_min` (16섹터 최소) | ≥ −1.11 | GT p1, 각도 완전성 |
| `ring_sat` | ≤ 100 | GT p95, 링이 금속일 것 |
| `ring_val` | ≤ 195 | GT p99 |

### 2.4 Specular Verification (step3)

금속 pad는 정반사 하이라이트를 형성하나, 무광 실크스크린 페인트와 기판은 형성하지 않는다.

$$\text{spec} = V_{p98} - V_{p50}$$

최종 순위는 밝기 점수와 정반사 특징의 z 결합이다.

$$\text{score} = z(\text{NCC}) + 1.5 \cdot z(\text{spec})$$

### 2.5 Core Transplant (step4)

결함의 물리적 정의가 "pad는 존재하고 구멍만 없음"이므로,
대상 pad의 금속 환형을 보존하고 중심 코어만 치환한다.

| 항목 | 값 | 근거 |
|---|---|---|
| donor | 같은 split **train 풀**의 실제 결함 crop (같은 이미지 제외, 크기비 0.6–1.7) | 누수 방지 |
| 코어 색상 | 주변 solder resist median × (B 0.740, G 0.803, R 0.889) | GT 900개 캘리브레이션 |
| 코어 텍스처 | donor core 편차의 60% | |
| 코어 크기 | `r_in = 0.20`, feather `0.11` | |
| box 크기 | pad 외곽 반경을 이미지에서 직접 추정, `box_side = r_out / 0.399` | GT 900개 캘리브레이션 |

코어 색상 계수는 코어가 pad 개구부 내부라 개방 resist보다 어둡다는 사실을 반영한다.
annotation은 삽입 위치의 box 자체이므로 라벨 오차가 없다.

---

## 3. 실험 설계

| 실험군 | 데이터셋 | 삽입 위치 결정 |
|---|---|---|
| Baseline | `{split}` | 증강 없음 |
| Random CP | `{split}_randomcp` | 균등 무작위, 크기는 GT 분포에서 추출 |
| Brightness CP | `{split}_brightnesscp` | NCC + Ring Gate |
| Context CP | `{split}_contextcp` | Brightness top-20을 주변 구조 유사도로 재정렬 |
| **Brightness+Spec CP** | `{split}_brightspeccp` | `z(NCC) + 1.5·z(spec)` ← **제안 방법** |

두 split(`mh_orig`, `mh_board`) 각각에 대해 구성하며, 총 10개 데이터셋이다.
모든 실험군에서 삽입 시도는 이미지당 2회로 동일하고, `val`/`test`는 동일하다.

모든 arm은 **동일한 후보 pool**에서 선택한다. 이미지당 템플릿 매칭은 1회만 수행하며
arm은 순위 전략만 달리한다. 따라서 arm 간 차이는 삽입 위치 결정에서만 발생한다.

---

## 4. 학습 전 단계 결과

### 4.1 RQ1 — 공통 밝기 패턴의 존재 (step1)

`D_mean`의 radial profile은 명확한 bullseye다.

| r (32×32 기준) | z |
|---|---|
| 0–2 px | −1.00 |
| 4 px | −0.31 |
| 6 px | +0.38 |
| **8–10 px** | **+0.84** |
| 13 px | +0.03 |
| 15 px | −0.41 |

중심이 어둡고 r 8–10px에서 밝기가 최대인 패턴이 여러 샘플에서 반복된다.

### 4.2 RQ2 — 밝기 기반 후보 탐색 (step2)

`mh_board` val 292장 / GT 636개 기준.

| 구성 | 후보/이미지 | recall@10 | recall@50 | GT 순위 median |
|---|---|---|---|---|
| NCC only | 200 | 67.5% | 92.6% | 3 |
| **NCC + Ring Gate** | **25** | **82.1%** | **90.9%** | **2** |

Ring Gate는 후보를 1/8로 줄이면서 recall@10을 14.6%p 높인다.

### 4.3 RQ3 — 후보 검증 방법 (step3)

비-GT 후보 120개를 밝기 점수 4분위로 층화 추출해 유효성을 수동 라벨링했다.

| 클래스 | 정의 | 비율 |
|---|---|---|
| `th` | 관통홀 pad — 결함 발생 가능, **유효** | 62.5% |
| `smd` | 표면실장 pad — 구멍 없음, 무효 | 10.0% |
| `no` | pad 아님 (실크스크린·기판·마운팅홀), 무효 | 27.5% |

`AUC(유효 pad vs 무효)`:

| 판별자 | AUC |
|---|---|
| Brightness NCC | 0.752 |
| Context (single mean) | 0.576 |
| Context (k-means K=12) | 0.524 |
| **Specular** | **0.874** |
| Brightness + Context | 0.752 (λ = 0.00) |
| **Brightness + Specular** | **0.918** |

유효 pad 비율 @ top-K:

| 랭킹 | K=10 | K=20 | K=40 | K=60 |
|---|---|---|---|---|
| 무작위 기준선 | 62.5% | 62.5% | 62.5% | 62.5% |
| Brightness | 90.0% | 85.0% | 85.0% | 83.3% |
| Brightness + Context | 90.0% | 85.0% | 85.0% | 83.3% |
| **Brightness + Specular** | **100.0%** | **100.0%** | **97.5%** | **93.3%** |

주변 구조(Context)는 삽입 위치 타당성을 판별하지 못한다.
`missing_hole`은 정상 pad 위에서 발생하므로 실제 결함 위치와 정상 pad의 주변 구조가 동일하고,
오검출 대상인 실크스크린·pad 간 틈 역시 pad array 내부에 위치하기 때문이다.
타당성을 결정하는 것은 주변 구조가 아니라 후보 자신의 재질이다.

참고로 Context 모델링 자체는 k-means가 우월하다
(GT context self-similarity: single mean 0.283 vs k-means K=12 0.566).
주변 구조가 다봉분포임에도 판별력이 없다는 점이 위 결론을 뒷받침한다.

### 4.4 합성 품질 검증 (step4)

삽입된 결함과 실제 결함의 radial profile을 동일 이미지에서 비교한다.

실제 결함 기준 프로파일:

| r/s | sat | val |
|---|---|---|
| 0.00–0.04 | 231.6 | 116.2 |
| 0.12–0.16 | 127.9 | 128.0 |
| 0.24–0.32 | 47.6 | 155.2 |
| 0.44–0.48 | 153.8 | 118.7 |

평균 절대 편차 (0–255 스케일, `mh_board`):

| 실험군 | sat MAD | val MAD |
|---|---|---|
| Context CP | 19.0 | 11.0 |
| Brightness CP | 21.0 | 12.4 |
| Brightness+Spec CP | 21.6 | 11.4 |
| **Random CP** | **108.3** | 8.6 |

`mh_orig`에서도 동일하다 (Context 19.0 / Brightness+Spec 20.9 / Brightness 21.0 / Random 108.7).

Random CP는 전 반경에서 채도가 224–251로 평평하다.
맨 기판 위에 코어만 붙은 형태로 **금속 환형이 존재하지 않는다.**
물리적으로 결함이 아닌 대상에 결함 라벨이 부여된다.

유도 방식 3종은 MAD 19–22로 실제와 정합하며 서로 2.6 이내다.
따라서 실험군 간 비교에서 결함 렌더링 품질은 교란변수가 아니며,
차이는 삽입 위치에서만 발생한다.

### 4.5 학습 데이터 구성

| dataset | 실제 box | 합성 box | 합계 |
|---|---|---|---|
| mh_board_randomcp | 2328 | 2176 | 4504 |
| mh_board_brightnesscp | 2328 | 1884 | 4212 |
| mh_board_contextcp | 2328 | 2215 | 4543 |
| mh_board_brightspeccp | 2328 | 1997 | 4325 |
| mh_orig_randomcp | 2902 | 2682 | 5584 |
| mh_orig_brightnesscp | 2902 | 2275 | 5177 |
| mh_orig_contextcp | 2902 | 2691 | 5593 |
| mh_orig_brightspeccp | 2902 | 2417 | 5319 |

삽입 시도는 전 실험군 동일(2회/이미지)이며,
유도 방식은 검증을 통과하지 못한 위치를 거부하므로 실현 개수가 적다.
**Random CP가 제안 방법보다 학습 box를 약 15% 더 보유한다.**

### 4.6 데이터 무결성

- 10개 데이터셋 × 3 split에서 이미지↔라벨 짝 불일치 0, 좌표 범위 이탈 0, 클래스 오류 0
- `val`/`test` 라벨이 전 실험군에서 SHA256 동일 — 학습 분포만 다름을 보증
- train box 수 = baseline + 합성 기록값, 전부 일치

---

## 5. 평가 계획 (RQ4)

동일 YOLO 모델을 각 데이터셋으로 학습하고 `test`에서 평가한다.
`val`/`test`는 모든 실험군에서 동일하므로 학습 분포만 비교 대상이다.

| 항목 | 값 |
|---|---|
| 모델 | yolov8s |
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| seed | 0 (권장: 1, 2 추가해 평균±표준편차) |

지표: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.
단일 클래스 문제이므로 전체 지표가 곧 `missing_hole` 지표다.
증강의 목적이 미검출 감소이므로 **Recall을 주요 지표**로 본다.

`mh_orig`과 `mh_board` 결과를 병기하며, leakage가 없는 `mh_board`를 보수적 추정으로 서술한다.

### 해석 규칙 (사전 고정)

| 관측 | 해석 |
|---|---|
| Brightness+Spec > Random | 제안 방법 성립. Random이 box를 15% 더 가진 상태에서의 우세이므로 데이터 양 효과가 아니다 |
| Context CP ≈ Brightness CP | 예측된 결과. §4.3에서 유효 pad 비율이 양쪽 모두 85%로 동일했다 |
| Random CP ≤ Baseline | 가능하다. §4.4에서 Random은 금속 링이 없는 비결함을 결함으로 라벨링함이 확인됐다 |
| arm 간 차이가 시드 표준편차 이내 | 유의하지 않다고 보고한다. `mh_board` test가 보드 2장임을 함께 명시한다 |

---

## 6. 한계

- 유효성 라벨은 120개, val split 1개, 단일 라벨러 기준이다
- `mh_board`의 test는 보드 2장으로 통계적 변동이 크다
- `r/s 0.16–0.28` 구간에서 합성 링이 실제보다 순수 금속에 가깝다(채도 29 vs 53).
  링 위치는 정확히 일치하므로 기하 오차는 아니다
- `spec` 특징은 본 데이터셋의 조명 조건에 캘리브레이션돼 있어, 다른 촬영 환경에서는 재캘리브레이션이 필요하다
- 물리 보드가 20장으로 제한돼 board 단위 분할의 자유도가 근본적으로 낮다

---

## 7. 제목 후보

- **Brightness- and Specular-Aware Copy-Paste Augmentation for PCB Missing Hole Detection**
- Material-Guided Defect Placement for Synthetic PCB Defect Augmentation
- Where to Paste: Physically Plausible Defect Placement for PCB Missing Hole Detection

---

## 8. 재현

| 단계 | 스크립트 | 실행 문서 |
|---|---|---|
| phase0 | `build_datasets.py` | `colab_execute/phase0_execute.md` |
| step1 | `phase1_template.py`, `calibrate_ring_features.py` | `step1_execute.md` |
| step2 | `phase2_search.py`, `eval_candidates.py` | `step2_execute.md` |
| step3 | `phase3_context.py`, `phase3_label_eval.py` | `step3_execute.md` |
| step4 | `phase4_calibrate_core.py`, `phase4_copypaste.py`, `phase4_qc.py`, `verify_datasets.py` | `step4_execute.md` |
| exp_yolo | (ultralytics) | `exp_yolo_execute.md` |

정본 규격은 `colab_execute/_SPEC.md`에 있다.
