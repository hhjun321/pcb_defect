# phase0 — `build_datasets.py` (데이터셋 복구 + split 생성, pcb-mh 정본)

> **목적**: 원본 `pcb-defect-dataset/`의 **라벨 파일명 버그를 복구**하고, missing_hole 전용 단일 클래스
> 데이터셋 2종(`mh_orig` 공개 split / `mh_board` board 분리)을 생성한다.
> 입력 `pcb-defect-dataset/` → 출력 `pcb-defect-dataset-fixed/` + `datasets/mh_orig`, `datasets/mh_board`.
> **실행 환경**: **CPU**. GPU 불필요.
> **전제**: 원본 zip이 Drive `/content/drive/MyDrive/data/PCB/archive.zip`에 있을 것
> (21,337파일 / 1.19GB, 최상위 폴더 `pcb-defect-dataset/` = train/val/test × images/labels + data.yaml).
> STEP 0-C가 이를 작업 루트 `/content/pcb_work`로 압축해제한다.
> ⚠️ **이 단계를 건너뛰면 이후 전부 무효**: 원본은 train 이미지 2164장(25%)의 라벨이
> `*_256.txt`로 저장돼 있어 `*_600.jpg`와 짝이 맺어지지 않는다. YOLO는 그 이미지들을 **라벨 없는 배경**으로
> 학습하며, `missing_hole` box 2902개 중 **705개(24%)가 통째로 버려진다**.
> ⚠️ **재실행 시 step1~4 전부 재실행**: split 구성이 바뀌면 템플릿·게이트 임계·합성 결과가 모두 어긋난다.
> **실행 순서 체인**: **phase0** → step1 → step2 → step3 → step4 → exp_yolo.
> **산출 데이터셋**: `mh_orig`(공개 split) · `mh_board`(board 분리, leakage 0 — 주 결과).

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
os.environ['PCB_ZIP']  = '/content/drive/MyDrive/data/pcb_defect/archive.zip'  # 원본 데이터 zip (Drive)
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

의존성:

```python
!pip -q install opencv-python-headless numpy matplotlib pillow
```

---

## STEP 1 — 원본 존재 확인 (실행 전 필수)

```python
from pathlib import Path
raw = Path(os.environ['PCB_RAW'])
assert raw.exists(), f"원본 없음: {raw}"
for sp in ['train', 'val', 'test']:
    ni = len(list((raw/sp/'images').glob('*.jpg')))
    nl = len(list((raw/sp/'labels').glob('*.txt')))
    print(f"{sp:6s} images={ni:6d}  labels={nl:6d}")
```

기대 출력:

```
train  images=  8534  labels=  8534
val    images=  1066  labels=  1066
test   images=  1068  labels=  1068
```

### 1-1. 버그 실증 (선택 — 근거를 눈으로 확인할 때)

```python
for sp in ['train', 'val', 'test']:
    I = {p.stem for p in (raw/sp/'images').glob('*')}
    L = {p.stem for p in (raw/sp/'labels').glob('*')}
    io, lo = I - L, L - I
    fixed = sum(1 for s in io if s.replace('_600', '_256') in lo)
    print(f"{sp:6s} 라벨없는이미지={len(io):5d}  '_600→_256' 치환으로 해소={fixed:5d}")
```

기대 출력 — **치환으로 100% 해소되는 것이 이 버그의 증거**다:

```
train  라벨없는이미지= 2164  '_600→_256' 치환으로 해소= 2164
val    라벨없는이미지=  264  '_600→_256' 치환으로 해소=  264
test   라벨없는이미지=  239  '_600→_256' 치환으로 해소=  239
```

---

## STEP 2 — 실행

```python
!python $PCB_SRC/build_datasets.py
```

수행 내용:
1. `pcb-defect-dataset-fixed/` 생성 — 라벨은 이름을 고쳐 복사, 이미지는 **하드링크**. 원본 무손상.
2. missing_hole 전용 서브셋 추출(클래스 `2` → `0`으로 remap, `nc: 1`).
3. `mh_orig`(공개 split 유지) + `mh_board`(board 단위 재분할) 생성.
4. `datasets/split_info.json`에 board→split 매핑 기록.

소요: 약 3–5분 (파일 수가 많아 I/O 바운드. Drive면 더 걸릴 수 있다)

---

## STEP 3 — 결과 확인

### 3-1. 로그 대조 (실측 확정값)

```
train images=8534  label_direct=6370  label_renamed(_256->_600)=2164  still_missing=0
val   images=1066  label_direct= 802  label_renamed(_256->_600)= 264  still_missing=0
test  images=1068  label_direct= 829  label_renamed(_256->_600)= 239  still_missing=0
  verify:
    train img=8534 lbl=8534 unpaired=0  missing_hole_boxes=2902
    val   img=1066 lbl=1066 unpaired=0  missing_hole_boxes=331
    test  img=1068 lbl=1068 unpaired=0  missing_hole_boxes=379

  missing_hole images=1832  boxes=3612  non-mh boxes inside them=0
  [mh_orig]  images={'train': 1477, 'val': 165, 'test': 190}  boxes={'train': 2902, 'val': 331, 'test': 379}
  boards per split: {'train': [1,3,4,6,9,10,11,...,20], 'val': [5,8], 'test': [2,7]}
  [mh_board] images={'train': 1212, 'test': 328, 'val': 292}  boxes={'train': 2328, 'test': 648, 'val': 636}

  LEAKAGE RE-AUDIT (test keys also present in train):
    mh_orig   key=board                  test_seen_in_train=20/20
    mh_orig   key=board+light+tile       test_seen_in_train=163/163
    mh_board  key=board                  test_seen_in_train=0/2
    mh_board  key=board+light+tile       test_seen_in_train=0/82
```

반드시 확인할 값: `still_missing=0`, `unpaired=0`, `mh_board ... test_seen_in_train=0/2`.

### 3-2. 데이터셋 집계

```python
print(f"{'dataset':<12}{'split':<7}{'images':>8}{'labels':>8}{'boxes':>8}")
for ds in SPLITS:
    for sp in ['train', 'val', 'test']:
        ni = len(glob.glob(f"{D(ds)}/{sp}/images/*.jpg"))
        nl = len(glob.glob(f"{D(ds)}/{sp}/labels/*.txt"))
        nb = sum(len([l for l in open(p) if l.strip()])
                 for p in glob.glob(f"{D(ds)}/{sp}/labels/*.txt"))
        print(f"{ds:<12}{sp:<7}{ni:>8}{nl:>8}{nb:>8}")
```

| dataset | split | images | labels | boxes |
|---|---|---|---|---|
| mh_board | train | 1212 | 1212 | 2328 |
| mh_board | val | 292 | 292 | 636 |
| mh_board | test | 328 | 328 | 648 |
| mh_orig | train | 1477 | 1477 | 2902 |
| mh_orig | val | 165 | 165 | 331 |
| mh_orig | test | 190 | 190 | 379 |

### 3-3. 복구된 라벨이 실제 결함 위에 놓이는지 시각 확인 (권장)

```python
import cv2, random
import matplotlib.pyplot as plt

files = sorted(glob.glob(f"{D('mh_board')}/train/images/*.jpg"))
random.seed(0)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, ip in zip(axes, random.sample(files, 3)):
    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for line in open(ip.replace('/images/', '/labels/').replace('.jpg', '.txt')):
        f = line.split()
        if len(f) != 5: continue
        cx, cy, w, h = [float(v) for v in f[1:]]
        cv2.rectangle(img, (int((cx-w/2)*W), int((cy-h/2)*H)),
                           (int((cx+w/2)*W), int((cy+h/2)*H)), (255, 0, 0), 2)
    ax.imshow(img); ax.axis('off')
plt.tight_layout(); plt.show()
```

box 안이 **은색 링 + 초록 중심**이면 정상이다. 은색으로 꽉 찬 pad면 라벨 오류다.

---

## 출력 파일

| 경로 | 내용 |
|------|------|
| `pcb-defect-dataset-fixed/` | 6클래스 전체, 라벨 파일명 복구본. 이미지는 원본 하드링크 |
| `datasets/mh_orig/` | missing_hole 단일 클래스, 공개 split |
| `datasets/mh_board/` | missing_hole 단일 클래스, board 분리 (leakage 0) |
| `datasets/*/data.yaml` | `nc: 1`, `names: {0: missing_hole}` |
| `datasets/split_info.json` | board→split 매핑 + 라벨 복구 리포트 |

---

## 판정 / 다음 단계

- [ ] `still_missing=0` (3종 split 전부)
- [ ] `unpaired=0`, missing_hole box **train 2902 / val 331 / test 379**
- [ ] `mh_board`: `test_seen_in_train=0/2` (board), `0/82` (board+light+tile)
- [ ] `mh_board` train 1212장/2328 box, `mh_orig` train 1477장/2902 box
- [ ] STEP 3-3 시각 확인 — box 안이 은색 링 + 초록 중심

통과 시 → **step1**(`phase1_template.py`, 입력 `datasets/mh_board` → 출력 `outputs/phase1`, `outputs/phase2/gt_ring_features.json`).

---

## 무결성 / 정직 (_SPEC §5)

- 원본 `pcb-defect-dataset/`은 **절대 수정하지 않는다**. 복구본은 별도 트리 + 하드링크다.
- `mh_board`의 board 배정은 box 수 70/15/15 목표 greedy 결과이며 **고정값**이다. 결과를 보고 재배정하지 않는다.
- `mh_board` test는 보드 2장이다. 절대값 단독 해석 금지 — Δ와 시드 분산을 함께 본다.
- `mh_orig`는 leakage가 있는 split임을 논문에 명시하고, `mh_board`를 보수적 추정으로 서술한다.
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
