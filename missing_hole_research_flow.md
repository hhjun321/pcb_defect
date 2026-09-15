# Missing Hole Brightness- and Context-Aware Copy-Paste 연구 흐름

## 1. 연구 개요

### 연구 핵심 아이디어

실제 `missing_hole` 결함 자체에서 공통적으로 나타나는 **밝기 특성**을 추출하여 정상 PCB 이미지에서 유사한 밝기 특성을 가진 후보 위치를 먼저 탐색한다. 이후 후보 위치의 **주변 PCB 구조(context)**를 비교하여 최종 삽입 위치를 결정하고, 실제 `missing_hole` 결함을 Copy-Paste하여 합성 데이터를 생성한다.

### 핵심 흐름

> **결함 자체의 밝기 특성 추출 → 정상 PCB 후보 탐색 → 주변 PCB 구조 검증 → 실제 결함 Copy-Paste → YOLO 성능 평가**

---

## 2. 연구 목적

PCB 결함 데이터에서 `missing_hole`의 실제 결함 위치를 고려한 Copy-Paste 증강 방법을 제안한다.

단순히 결함을 임의의 위치에 삽입하는 Random Copy-Paste와 달리,

1. 실제 `missing_hole` 결함의 공통적인 밝기 특성을 분석하고,
2. 정상 PCB에서 유사한 밝기 특성을 가진 위치를 후보로 선정한 후,
3. 해당 위치의 주변 PCB 구조가 실제 결함이 발생한 위치와 유사한지를 확인하여,
4. 보다 적절한 위치에 결함을 삽입하는 것을 목표로 한다.

---

# 3. 전체 연구 흐름

```text
                    Original PCB Dataset
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Missing Hole                  Normal PCB
             │                           │
             ▼                           │
     결함 영역 추출                      │
             │                           │
             ▼                           │
      크기 정규화 / Gray                 │
             │                           │
             ▼                           │
   Missing Hole 공통 밝기 특성           │
             │                           │
             └──── Brightness Search ────┘
                         │
                         ▼
                후보 위치 여러 개 선정
                         │
                         ▼
              Context Template Matching
                         │
                         ▼
                  최종 삽입 위치
                         │
                         ▼
               실제 Missing Hole
                  Copy-Paste
                         │
                         ▼
                 Synthetic Dataset
                         │
                         ▼
                    YOLO 학습
                         │
                         ▼
                    성능 평가
```

---

# 4. Step 1. Missing Hole 결함 데이터 수집

전체 데이터에서 YOLO annotation을 이용하여 `missing_hole`에 해당하는 bounding box를 수집한다.

```text
Original Dataset
      │
      ▼
YOLO Annotation
      │
      ▼
missing_hole 선택
      │
      ▼
결함 영역 추출
```

### 주의사항

YOLO bounding box는 반드시 실제 결함의 정확한 segmentation mask를 의미하지 않는다.

따라서 연구에서 사용하는 **결함 영역(defect region)**과 **주변 context 영역**을 명확하게 구분해야 한다.

본 연구의 핵심 밝기 특성은 **주변 PCB가 아니라 missing_hole 결함 자체의 밝기 특성**을 대상으로 한다.

---

# 5. Step 2. Missing Hole 크기 정규화

각 `missing_hole`의 크기가 서로 다르기 때문에 동일한 크기로 변환한다.

예:

```text
Missing Hole 1 → 32 × 32
Missing Hole 2 → 32 × 32
Missing Hole 3 → 32 × 32
...
Missing Hole N → 32 × 32
```

그리고 grayscale로 변환하여 밝기값을 사용한다.

각 결함을 다음과 같이 표현할 수 있다.

\[
D_i(x,y)
\]

여기서,

- `i`: 개별 missing_hole 샘플
- `x,y`: 결함 이미지의 위치
- `D_i(x,y)`: 해당 위치의 밝기값

---

# 6. Step 3. Missing Hole의 공통 밝기 특성 추출

## 6.1 평균 밝기 패턴

모든 `missing_hole` 샘플의 동일한 위치에서 평균 밝기를 계산한다.

\[
D_{mean}(x,y)
=
\frac{1}{N}
\sum_{i=1}^{N}D_i(x,y)
\]

이를 통해 하나의 **대표 Missing Hole Brightness Template**를 생성한다.

```text
D1     D2     D3     ...     DN
 │      │      │              │
 └──────┴──────┴──────────────┘
              │
              ▼
       평균 밝기 계산
              │
              ▼
      D_mean(x,y)
```

## 6.2 연구에서 확인할 내용

단순히 "`missing_hole`은 어둡다"라는 특성을 확인하는 것이 아니라,

> **missing_hole 내부의 밝기가 공간적으로 어떤 형태로 분포하는가?**

를 확인한다.

예를 들어 중앙이 상대적으로 어둡고 주변으로 갈수록 밝아지는 일정한 패턴이 여러 샘플에서 반복되는지를 확인한다.

---

# 7. Step 4. 정상 PCB에서 밝기 기반 후보 탐색

정상 PCB 이미지를 대상으로 sliding window를 수행한다.

예를 들어 32×32 크기의 window를 일정 간격으로 이동시키면서 각 영역의 밝기 패턴을 추출한다.

```text
Normal PCB

┌─────────────────────────────┐
│                             │
│     ┌──────────┐            │
│     │ 32 × 32  │            │
│     │ Candidate │            │
│     └──────────┘            │
│                             │
│            ↓ 이동           │
│                             │
│                ┌──────────┐ │
│                │ 32 × 32  │ │
│                │ Candidate │ │
│                └──────────┘ │
│                             │
└─────────────────────────────┘
```

각 후보 영역을

\[
W_p(x,y)
\]

라고 한다.

---

# 8. Step 5. Brightness Similarity 계산

대표적인 missing_hole 밝기 패턴 `D_mean`과 각 후보 영역 `W_p`를 비교한다.

학부 수준의 첫 번째 구현에서는 MSE(Mean Squared Error)를 사용할 수 있다.

\[
MSE(p)
=
\frac{1}{K}
\sum_{x,y}
\left(
D_{mean}(x,y)-W_p(x,y)
\right)^2
\]

여기서 `K`는 비교하는 전체 픽셀 수이다.

### 해석

- MSE가 작음 → 실제 missing_hole과 밝기 패턴이 유사
- MSE가 큼 → 실제 missing_hole과 밝기 패턴이 다름

따라서 MSE가 작은 위치를 후보로 선정한다.

```text
Normal PCB
     │
     ▼
Sliding Window
     │
     ▼
Brightness Similarity
     │
     ▼
Score 계산
     │
     ▼
Top-K 후보 선정
```

예:

```text
전체 후보 위치
      ↓
Brightness Similarity
      ↓
Top 10 후보
```

---

# 9. Step 6. 주변 PCB Context를 이용한 후보 검증

밝기 특성만으로 최종 위치를 결정할 경우 정상 PCB의 다른 어두운 영역이 잘못 선택될 가능성이 있다.

따라서 밝기 기반으로 선정된 후보에 대해서만 **주변 PCB 구조(context)**를 추가적으로 비교한다.

## 9.1 Context Template 생성

실제 `missing_hole` 주변 영역을 포함하여 Template를 생성한다.

```text
┌───────────────────────┐
│                       │
│      PCB Context      │
│                       │
│        [Hole]         │
│                       │
│      PCB Context      │
│                       │
└───────────────────────┘
```

Template의 중심에 있는 실제 결함 영역은 context 비교에서 제외하고, 주변 PCB 구조를 비교 대상으로 사용한다.

```text
┌───────────────────────┐
│    비교 영역          │
│                       │
│       [제외]          │ ← Missing Hole
│                       │
│    비교 영역          │
└───────────────────────┘
```

### 핵심

Brightness 단계와 Context 단계의 역할을 분리한다.

| 단계 | 사용하는 정보 | 목적 |
|---|---|---|
| Brightness Search | **결함 자체의 밝기 특성** | 후보 위치 탐색 |
| Context Matching | **결함 주변 PCB 구조** | 후보 위치 검증 |
| Copy-Paste | **실제 missing_hole 결함** | 데이터 합성 |

---

# 10. Step 7. 최종 삽입 위치 선정

Brightness Search에서 얻은 Top-K 후보에 대해 Context Matching을 수행한다.

```text
Brightness Search
       │
       ▼
Candidate 1
Candidate 2
Candidate 3
...
Candidate 10
       │
       ▼
Context Matching
       │
       ▼
Context Similarity 비교
       │
       ▼
Best Location
```

최종적으로 주변 PCB 구조가 가장 유사한 위치를 선택한다.

\[
p^*
=
\arg\max_p S_{context}(p)
\]

여기서 `S_context`는 실제 missing_hole 주변 context와 후보 위치의 구조적 유사도이다.

---

# 11. Step 8. 실제 Missing Hole Copy-Paste

최종 위치가 결정되면 대표 밝기 template을 삽입하는 것이 아니라, **실제 missing_hole 결함 이미지**를 사용하여 Copy-Paste한다.

```text
실제 Missing Hole
        +
최종 선정 위치
        │
        ▼
    Copy-Paste
        │
        ▼
Synthetic Missing Hole
```

### 역할 구분

**대표 밝기 패턴**

→ 결함을 어디에 넣을지 찾기 위한 기준

**실제 Missing Hole**

→ 최종 데이터 증강에 사용하는 결함

이 구분을 연구 방법에서 명확하게 유지한다.

---

# 12. Step 9. 비교 실험 설계

제안 방법의 효과를 확인하기 위해 최소 3개 방법을 비교한다.

| 실험군 | 방법 | 삽입 위치 |
|---|---|---|
| Baseline | Original Dataset | 증강 없음 |
| Random CP | Random Copy-Paste | 정상 PCB의 임의 위치 |
| Proposed | Brightness + Context Copy-Paste | 제안 방법으로 선정 |

필요하면 추가 실험으로 다음을 구성할 수 있다.

| 실험군 | 방법 | 목적 |
|---|---|---|
| Brightness CP | Brightness 기반만 사용 | Brightness 단계의 효과 확인 |
| Context CP | Context 기반만 사용 | Context 단계의 효과 확인 |
| Brightness + Context CP | 두 단계 사용 | 제안 방법의 최종 성능 확인 |

---

# 13. Step 10. YOLO 성능 평가

각 증강 데이터셋으로 동일한 YOLO 모델을 학습하고 성능을 비교한다.

주요 평가 지표:

- mAP@0.5
- Precision
- Recall
- F1-score
- `missing_hole` class AP

특히 연구의 목적이 `missing_hole` 탐지 성능 향상이므로 전체 mAP뿐만 아니라 **missing_hole class의 AP 및 Recall**을 함께 확인한다.

---

# 14. 핵심 연구 질문

### RQ1
`missing_hole`에는 여러 샘플에서 공통적으로 나타나는 밝기 패턴이 존재하는가?

↓

### RQ2
이러한 밝기 패턴을 이용하여 정상 PCB에서 `missing_hole`이 삽입되기 적합한 후보 영역을 찾을 수 있는가?

↓

### RQ3
밝기 기반 후보 탐색 후 주변 PCB 구조를 추가적으로 고려하면 더 적절한 삽입 위치를 선택할 수 있는가?

↓

### RQ4
이러한 위치 인식 Copy-Paste augmentation이 Random Copy-Paste보다 YOLO의 `missing_hole` 탐지 성능을 향상시키는가?

---

# 15. 최종 연구 개념도

```text
┌──────────────────────────────────────────────┐
│              Missing Hole Dataset            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              Defect Region Extraction
                       │
                       ▼
                Size Normalization
                       │
                       ▼
                 Grayscale Images
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Missing Hole Brightness      │
        │ Characteristic Extraction    │
        │                              │
        │ D_mean(x,y)                  │
        └──────────────┬───────────────┘
                       │
                       │ Brightness Matching
                       ▼
┌──────────────────────────────────────────────┐
│                 Normal PCB                   │
│                                              │
│          Sliding Window Search               │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
                 Top-K Candidates
                       │
                       │ Context Matching
                       ▼
        ┌──────────────────────────────┐
        │ 주변 PCB 구조 유사도 비교    │
        └──────────────┬───────────────┘
                       │
                       ▼
                Final Location
                       │
                       ▼
            Actual Missing Hole
                  Copy-Paste
                       │
                       ▼
              Synthetic Dataset
                       │
                       ▼
                  YOLO Training
                       │
                       ▼
             Performance Evaluation
```

---

# 16. 연구의 핵심 차별점

본 연구에서 가장 강조할 부분은 **밝기 특성과 Context의 역할을 구분하는 것**이다.

```text
                Proposed Method
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  Defect Brightness          PCB Context
   Characteristic             Structure
          │                         │
          ▼                         ▼
   후보 위치 탐색              후보 위치 검증
          │                         │
          └────────────┬────────────┘
                       ▼
               최종 삽입 위치 결정
                       │
                       ▼
                Copy-Paste
```

따라서 연구의 핵심은 단순한 Copy-Paste가 아니라,

> **"실제 결함 자체의 밝기 특성을 이용하여 결함 후보 위치를 찾고, 해당 위치의 PCB 주변 구조를 추가로 확인하여 합성 위치를 결정한다."**

로 정리할 수 있다.

---

# 17. 구현 순서

복잡도를 낮추기 위해 다음 순서로 구현한다.

### Phase 1 — 데이터 확인

- `missing_hole` annotation 추출
- 결함 이미지 추출
- 결함 크기 및 밝기 분포 확인
- 정상 PCB 이미지 분리

### Phase 2 — Brightness 방법

- 결함 크기 정규화
- grayscale 변환
- 평균 밝기 패턴 생성
- 정상 PCB sliding window
- MSE 기반 유사 위치 탐색

### Phase 3 — Context 방법

- 실제 결함 주변 Template 생성
- 결함 영역을 제외한 주변 구조 비교
- Brightness Top-K 후보에 Context Matching 적용
- 최종 위치 선정

### Phase 4 — Copy-Paste

- 실제 missing_hole 결함 선택
- 최종 위치에 삽입
- annotation 생성
- Synthetic Dataset 생성

### Phase 5 — 성능 평가

- Baseline
- Random Copy-Paste
- Brightness Copy-Paste
- Brightness + Context Copy-Paste
- YOLO 성능 비교

---

# 18. 연구 제목 후보

### 추천

**Brightness- and Context-Aware Copy-Paste Augmentation for Missing Hole Detection**

### 간결한 제목

**Brightness-Guided Copy-Paste Augmentation for PCB Missing Hole Detection**

### 방법을 강조하는 제목

**Defect Brightness-Guided and Context-Aware Synthetic Augmentation for PCB Missing Hole Detection**

현재 단계에서는 첫 번째 제목이 **Brightness와 Context라는 두 핵심 요소를 모두 표현하면서 연구 방법도 명확하게 보여주므로 가장 적합하다.**
