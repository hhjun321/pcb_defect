# step4 — `phase4_copypaste.py` (Core Transplant 합성 · QC, pcb-mh 정본)

> **목적**: 실험군 4종(`random` / `brightness` / `context` / `brightspec`)의 증강 학습셋을 생성하고,
> 합성 결함의 현실성을 **radial profile로 정량 검증**한다.
> 입력 `datasets/{split}` + `outputs/phase2` + `outputs/phase3` → 출력 `datasets/{split}_{arm}cp`.
> **실행 환경**: **CPU**.
> **전제**: step2·step3 완료 — `outputs/phase2/Dz_mean_boardtrain.npy`, `outputs/phase3/C_kmeans.npy` 존재.
> ⚠️ **donor를 통째로 붙이지 않는다**: missing_hole의 물리적 정의는 *pad는 그대로, 구멍만 없음*이다.
> donor의 금속 링까지 옮기면 조명·형태 불일치를 자초한다(1차 시도 실측: 분홍 후광, sat MAD 32.4).
> 정본은 **core transplant** — 대상 pad의 금속 환형을 보존하고 중심 코어만 치환한다.
> ⚠️ **캘리브레이션 값을 눈대중으로 바꾸지 말 것**: 코어 색 비율과 `pad_edge_frac`은 GT 900개 실측값이다.
> ⚠️ **feather 폭을 선형 alpha 역산으로 넓히지 말 것**: HSV 채도가 `(max−min)/max`라 비선형이다.
> 역산대로 넓혀 만들어 재보니 악화됐다(sat MAD 21.6 → 28.0, core 22.9 → 44.9).
> ⚠️ **`val`/`test`는 변형 금지**: 하드링크로 원본을 통과시킨다. train만 달라져야 공정 비교가 된다.
> **실행 순서 체인**: phase0 → step1 → step2 → step3 → **step4** → exp_yolo.

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
for p in [O('phase2', 'Dz_mean_boardtrain.npy'), O('phase3', 'C_kmeans.npy')]:
    assert Path(p).exists(), f"선행 단계 미완료: {p}"
print("OK")
```

---

## 합성 규격 (정본)

### 위치 결정 — 4 arm

| arm | 순위 결정 | spec 게이트 |
|---|---|---|
| `random` | 균등 무작위, 크기는 해당 이미지 GT 분포에서 추출. 겹침만 회피 | — |
| `brightness` | stage-A NCC (ring gate + grid-cap) | 없음 |
| `context` | brightness top-20 → context 유사도 재정렬 (원안 Step 6/7 직역, ablation) | 없음 |
| `brightspec` | `z(ncc) + 1.5·z(spec)` | **적용** |

> **전 arm이 동일 후보 pool을 공유한다.** 이미지당 템플릿 매칭은 1회만 수행하고 순위만 달리 매긴다.
> 속도가 ~4배 빨라지는 동시에, arm 간 차이가 **순위 전략에서만** 발생하도록 보장된다.

### 붙여넣기 — core transplant

| 항목 | 값 | 근거 |
|---|---|---|
| donor | 같은 split **TRAIN 풀**의 실제 결함 crop (같은 이미지 제외, 크기비 0.6~1.7) | 누수 방지 |
| 코어 색 | 주변 resist median × `(B 0.740, G 0.803, R 0.889)` | GT 900개 캘리브레이션 |
| 코어 텍스처 | donor core 편차의 60% | |
| 코어 크기 | `r_in 0.20`, feather `0.11` | 3개 설정 실측 비교 |
| box 크기 | `pad 외곽반경 / 0.399` (이미지에서 직접 추정, 2회 반복 수렴) | GT 900개 캘리브레이션 |
| 삽입 수 | 2 / 이미지 (**시도** 기준, 전 arm 동일) | |
| seed | arm별 `0..3` | |

annotation은 삽입 site box 자체이므로 **라벨 오차가 0**이다.

---

## STEP 1 — 코어 색상 · pad 크기 캘리브레이션

```python
!python $PCB_SRC/phase4_calibrate_core.py
```

기대 출력 (실측 확정값):

```
CORE COLOUR TRANSFER  (real core vs resist just outside the pad, n=900)
   per-channel ratio  core/resist  B=0.740 G=0.803 R=0.889
   per-channel offset core-resist  B=-14.0 G=-29.0 R=-2.0

PAD OUTER EDGE inside a real GT box (n=900)
   r_edge / box_side : p25=0.399  median=0.399  p75=0.475
   => box_side = pad_outer_radius / 0.399
```

의미:
- 코어는 pad **개구부 안쪽**이라 개방 resist보다 어둡고 진하다. 비율 없이 바깥 색을 그대로 쓰면
  합성 코어가 과밝아진다(1차 실측: val 124 vs 실제 105).
- box는 템플릿의 이산 스케일 `{20,24,...,48}`이 아니라 **pad에서 직접** 추정해야 실제 GT와 기하가 맞는다.

---

## STEP 2 — 합성 (8종 생성)

```python
!python $PCB_SRC/phase4_copypaste.py \
    --splits mh_board mh_orig \
    --methods random brightness context brightspec \
    --n-insert 2
```

소요: `mh_board`(1212장) 약 10분 + `mh_orig`(1477장) 약 12분 = **약 20–25분**

기대 출력 (실측 확정값):

```
BASE SPLIT: mh_board
  donor pool: 2194 real missing_hole crops
  [mh_board_randomcp]      imgs=1212  boxes added=2176 (1.80/img)  mean size=32.0px
  [mh_board_brightnesscp]  imgs=1212  boxes added=1884 (1.55/img)  mean size=34.9px
  [mh_board_contextcp]     imgs=1212  boxes added=2215 (1.83/img)  mean size=32.8px
  [mh_board_brightspeccp]  imgs=1212  boxes added=1997 (1.65/img)  mean size=35.3px

BASE SPLIT: mh_orig
  donor pool: 2730 real missing_hole crops
  [mh_orig_randomcp]       imgs=1477  boxes added=2682 (1.82/img)  mean size=31.7px
  [mh_orig_brightnesscp]   imgs=1477  boxes added=2275 (1.54/img)  mean size=34.9px
  [mh_orig_contextcp]      imgs=1477  boxes added=2691 (1.82/img)  mean size=32.9px
  [mh_orig_brightspeccp]   imgs=1477  boxes added=2417 (1.64/img)  mean size=35.5px
```

> 삽입 **시도**는 전 arm 2회/이미지로 동일하다. 유도 방식은 게이트를 통과하지 못한 자리를 거부하므로
> 실현 수가 적다. 결과적으로 **`random`이 제안 방법보다 box를 약 15% 더 갖는다.**
> 그럼에도 제안 방법이 이기면 "데이터 양 덕분"이라는 반론이 원천 봉쇄된다.

육안 확인:

```python
from IPython.display import Image, display
for a in ARMS:
    print(f"===== {a} =====")
    display(Image(O('phase4', f'qc_mh_board_{a}cp.png')))
```

`brightspec`은 은색 pad 중앙에 초록 코어가 자연스럽게 들어가 있어야 한다.
`random`은 맨 기판 위에 초록 원판이 떠 있는 형태로 보인다 — **이것이 정상**이며, STEP 3에서 수치로 확인된다.

---

## STEP 3 — 합성 품질 QC (radial profile)

육안이 아니라 **실제 결함의 radial profile과 대조**한다.
코어가 작다/색이 다르다/후광이 있다 같은 실패는 전부 프로파일 불일치로 나타난다.

```python
for a in ARMS:
    print(f"\n########## {a} ##########")
    !python $PCB_SRC/phase4_qc.py mh_board_{a}cp mh_board
```

### 3-1. 실제 GT 기준 프로파일

| r/s | sat | val | 영역 |
|---|---|---|---|
| 0.00–0.04 | 231.6 | 116.2 | 초록 코어 중심 |
| 0.12–0.16 | 127.9 | 128.0 | 반값 지점 |
| 0.24–0.32 | 47.6 | 155.2 | 금속 링 |
| 0.44–0.48 | 153.8 | 118.7 | 바깥 resist |

### 3-2. 기대 결과 (실측 확정값)

`mh_board`:

| arm | sat MAD | val MAD | core sat MAD | core val MAD |
|---|---|---|---|---|
| context | **19.0** | 11.0 | 23.9 | 8.0 |
| brightness | 21.0 | 12.4 | 23.1 | 4.9 |
| brightspec | 21.6 | 11.4 | 22.9 | **4.8** |
| **random** | **108.3** | 8.6 | 44.2 | 3.1 |

`mh_orig`: context 19.0 / brightspec 20.9 / brightness 21.0 / **random 108.7** — 동일 경향.

### 3-3. Random CP의 실패가 수치로 드러난다

```
r/s        real   random
0.00-0.04  239.0   251.5
0.12-0.16  143.7   227.3
0.24-0.28   52.9   228.7   <- 금속 링이 있어야 할 자리
0.44-0.48  153.2   224.0
```

random은 **전 반경 채도 224–251로 평평**하다. 맨 기판에 초록 원판을 붙인 것에 불과하며
**금속 링이 아예 없다.** 물리적으로 missing_hole이 아닌 대상에 결함 라벨을 부여하고 있다.
sat MAD 108.3은 유도 방식(19–22)의 **5배**다.

### 3-4. 유도 3종의 정합과 잔차

일치 구간(`brightspec`):

```
             real   synth
r/s 0.00-0.04  239.0  253.2   코어 중심
r/s 0.08-0.12  186.7  181.0
r/s 0.32-0.36   56.8   59.2   pad 바깥
r/s 0.44-0.48  152.2  156.4   resist
```

잔차 구간 `r/s 0.16–0.28`: 합성 링이 실제보다 순수 금속에 가깝다(sat 29 vs 53).
링 **위치는 정확히 일치**(최저점 둘 다 r/s 0.24–0.28)하므로 기하 오차가 아니다.
feather 확대로 보정을 시도했으나 악화가 확인돼 현 설정을 유지한다.

**전체 MAD 19–22 / 255 ≈ 8%. 유도 3종이 서로 2.6 이내로 근접한다**
→ 삽입 위치만 다르고 결함 렌더링 품질은 동일하므로, 실험군 비교에서 렌더링 품질이 **교란변수가 아니다**.

---

## STEP 4 — 무결성 검증

```python
!python $PCB_SRC/verify_datasets.py
```

검사 항목:
- 이미지↔라벨 짝, 좌표 범위 `[0,1]`, 클래스 id는 항상 0
- **`val`/`test` 라벨이 baseline과 byte-identical**(SHA256) — train만 달라야 한다
- `train box 수 == baseline + synthesis_meta.json의 boxes_added`

기대 출력 말미:

```
TRAIN BOX BUDGET
dataset                          real  synthetic      total mean size px
mh_board_randomcp                2328       2176       4504         32.0
mh_board_brightnesscp            2328       1884       4212         34.9
mh_board_contextcp               2328       2215       4543         32.8
mh_board_brightspeccp            2328       1997       4325         35.3
mh_orig_randomcp                 2902       2682       5584         31.7
mh_orig_brightnesscp             2902       2275       5177         34.9
mh_orig_contextcp                2902       2691       5593         32.9
mh_orig_brightspeccp             2902       2417       5319         35.5

ALL CHECKS PASSED
```

`ALL CHECKS PASSED`가 아니면 **exp_yolo로 진행하지 않는다.**

---

## 출력 파일

| 경로 | 내용 |
|------|------|
| `outputs/phase4/core_calibration.json` | 코어 색 비율 + `pad_edge_frac` |
| `datasets/{split}_{arm}cp/` | 증강 학습셋 8종 (train만 변형, val/test 하드링크) |
| `datasets/{split}_{arm}cp/data.yaml` | `nc: 1`, `names: {0: missing_hole}` |
| `datasets/{split}_{arm}cp/synthesis_meta.json` | seed, 삽입 수, 파라미터, boxes_added |
| `outputs/phase4/qc_{ds}.png` | 합성 결과 육안 QC |
| `outputs/phase4/qc_profile_{ds}.png` | radial profile 대조 그림 |
| `outputs/phase4/synthesis_summary.json` | 전 arm 메타 집계 |

---

## 백업 — `datasets/`를 Drive로 (세션 보존)

작업 루트는 세션 로컬이라 런타임이 끊기면 `datasets/`가 사라진다. step4 재실행은 수십 분,
tar 복사는 수 분이다. **exp_yolo를 다른 세션에서 돌릴 계획이면 여기서 백업한다.**

```python
import os, subprocess, time
BK = f"{os.environ['DRIVE']}/backup"
os.makedirs(BK, exist_ok=True)
tar = f"{BK}/datasets.tar"

t = time.time()
subprocess.run(['tar', '-C', os.environ['PCB_ROOT'], '-cf', tar, 'datasets'], check=True)
print("saved %s  %.2f GB  %.1f분" % (tar, os.path.getsize(tar) / 1e9, (time.time() - t) / 60))
```

- `datasets/` 전체를 **tar 1개**로 묶는다. 데이터셋 10종이 `val`/`test`를 하드링크로 공유하므로
  tar가 링크 항목으로 저장해 중복이 제거된다(개별 tar 10개 ≈ 2.2GB → 통합 tar ≈ 1.4GB).
- 압축하지 않는다. 내용이 JPEG라 gzip 이득이 없고 시간만 든다.
- 복원은 exp_yolo STEP 0-E에서 자동으로 한다.

---

## 판정 / 다음 단계

- [ ] 캘리브레이션 `ratio B=0.740 G=0.803 R=0.889`, `pad_edge_frac=0.399`
- [ ] 8종 생성 완료, `boxes added`가 위 표와 일치
- [ ] QC: 유도 3종 **sat MAD 19–22**, 서로 2.6 이내
- [ ] QC: `random` **sat MAD ≈ 108** (금속 링 부재의 정량 증거)
- [ ] `qc_mh_board_brightspeccp.png` 육안 — pad 중앙에 초록 코어, 후광 없음
- [ ] `verify_datasets.py` → **`ALL CHECKS PASSED`**

통과 시 → **exp_yolo**(YOLO 학습·평가, 입력 `datasets/*` → 출력 `runs/`).

---

## 무결성 / 정직 (_SPEC §5)

- **사후 튜닝 금지**: `n_insert=2`, `λ_spec=1.5`, `CORE_R=0.20`, `CORE_FEATHER=0.11`, seed는
  downstream 결과를 보고 변경하지 않는다.
- 캘리브레이션 값은 실측 확정값이다. 바꾸려면 `phase4_calibrate_core.py`를 재실행해 근거를 갱신한다.
- `val`/`test`는 전 arm 동일해야 한다. `verify_datasets.py`의 SHA256 검사가 이를 보증한다.
- donor는 **같은 split의 train 풀**에서만 뽑는다. val/test 이미지를 donor로 쓰면 누수다.
- `random` arm의 box가 더 많다는 사실을 **결과 보고 시 반드시 명시**한다. 제안 방법에 유리한 방향의 편향이 아니다.
- feather 폭 선형 alpha 역산 재시도 금지(악화 실측).
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
