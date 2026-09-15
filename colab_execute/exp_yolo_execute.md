# exp_yolo — downstream detection (YOLO 학습·평가, pcb-mh 정본)

> **목적**: 실험군 5종(Baseline / Random CP / Brightness CP / Context CP / **Brightness+Spec CP**)을
> 동일 조건으로 학습하고 `test`에서 평가해 **RQ4**에 답한다.
> 입력 `datasets/*` → 출력 `runs/{ds}_s{seed}`.
> **실행 환경**: **GPU** (T4 이상). 런타임 → 런타임 유형 변경 → GPU.
> **전제**: step4 완료 — `verify_datasets.py`가 `ALL CHECKS PASSED`를 출력했을 것.
> ⚠️ **STEP 1(`data.yaml` 재작성)을 반드시 먼저 실행**한다. `path`에 Windows 절대경로가 들어 있어
> 건너뛰면 `Dataset not found`로 실패한다.
> ⚠️ **전 arm 동일 조건**: 모델·에폭·imgsz·batch·seed를 고정한다. 한 arm만 바꾸면 비교가 무효다.
> ⚠️ **주 결과는 `mh_board`**(leakage 0). `mh_orig`는 참고로 병기한다.
> ⚠️ **주요 지표는 Recall**: 증강의 목적이 미검출 감소다. mAP만 보고하지 않는다.
> **실행 순서 체인**: phase0 → step1 → step2 → step3 → step4 → **exp_yolo**.

---

## 실험군 정의

| arm | 데이터셋 | 삽입 위치 결정 |
|---|---|---|
| Baseline | `{split}` | 증강 없음 |
| Random CP | `{split}_randomcp` | 균등 무작위 |
| Brightness CP | `{split}_brightnesscp` | NCC + ring gate |
| Context CP | `{split}_contextcp` | brightness top-20 → context 재정렬 (ablation) |
| **Brightness+Spec CP** | `{split}_brightspeccp` | `z(ncc)+1.5·z(spec)` ← **제안** |

`val`/`test`는 전 arm 동일하다. 학습 분포만 다르다.

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
os.makedirs(os.environ['PCB_RUNS'], exist_ok=True)
print("PCB_ROOT =", os.environ['PCB_ROOT'])
print("PCB_SRC  =", os.environ['PCB_SRC'])
```

```python
!nvidia-smi
!pip -q install ultralytics
import ultralytics; ultralytics.checks()
```

---

## STEP 1 — `data.yaml` 경로 재작성 (**반드시 실행**)

```python
n = 0
for d in sorted(os.listdir(os.environ['PCB_DS'])):
    p = os.path.join(D(d), 'data.yaml')
    if not os.path.exists(p):
        continue
    with open(p, 'w') as f:
        f.write(f"path: {D(d)}\n")
        f.write("train: train/images\nval: val/images\ntest: test/images\n\n")
        f.write("nc: 1\nnames:\n  0: missing_hole\n")
    n += 1
print(f"rewritten {n} data.yaml")     # 10
```

---

## STEP 2 — 데이터셋 점검

```python
print(f"{'dataset':<26}{'train':>8}{'val':>7}{'test':>7}{'tr.box':>9}")
for split in SPLITS:
    for d in arms_of(split):
        cnt = {sp: len(glob.glob(f"{D(d)}/{sp}/images/*.jpg")) for sp in ['train','val','test']}
        box = sum(len([l for l in open(p) if l.strip()])
                  for p in glob.glob(f"{D(d)}/train/labels/*.txt"))
        print(f"{d:<26}{cnt['train']:>8}{cnt['val']:>7}{cnt['test']:>7}{box:>9}")
```

기대 출력 (실측 확정값):

```
dataset                      train    val   test   tr.box
mh_board                      1212    292    328     2328
mh_board_randomcp             1212    292    328     4504
mh_board_brightnesscp         1212    292    328     4212
mh_board_contextcp            1212    292    328     4543
mh_board_brightspeccp         1212    292    328     4325
mh_orig                       1477    165    190     2902
mh_orig_randomcp              1477    165    190     5584
mh_orig_brightnesscp          1477    165    190     5177
mh_orig_contextcp             1477    165    190     5593
mh_orig_brightspeccp          1477    165    190     5319
```

한 값이라도 어긋나면 step4로 돌아간다.

### 2-1. (권장) Drive I/O 회피 — 로컬 복사

Drive에서 직접 학습하면 I/O 병목이 크다. 로컬 디스크로 복사한 뒤 학습하고 결과만 Drive에 남긴다.

```python
LOCAL = '/content/pcb_ds'
!mkdir -p $LOCAL
for split in SPLITS:
    for d in arms_of(split):
        if not os.path.exists(f"{LOCAL}/{d}"):
            !cp -r "{D(d)}" "{LOCAL}/"
            # 복사본의 data.yaml path 갱신
            with open(f"{LOCAL}/{d}/data.yaml", 'w') as f:
                f.write(f"path: {LOCAL}/{d}\ntrain: train/images\nval: val/images\ntest: test/images\n\n")
                f.write("nc: 1\nnames:\n  0: missing_hole\n")
print(sorted(os.listdir(LOCAL)))
DSBASE = LOCAL          # 아래 학습에서 사용. Drive 직접 학습이면 DSBASE = os.environ['PCB_DS']
```

---

## STEP 3 — 학습 설정 (전 arm 고정)

```python
MODEL    = 'yolov8s.pt'
EPOCHS   = 100
IMGSZ    = 640
BATCH    = 16
PATIENCE = 30
RUNDIR   = os.environ['PCB_RUNS']      # Drive — 세션이 끊겨도 결과 유지
```

| 항목 | 값 | 비고 |
|---|---|---|
| 모델 | `yolov8s` | 전 arm 동일 |
| epochs | 100 | 전 arm 동일 |
| imgsz | 640 | 원본 600×600 기준 |
| batch | 16 | OOM 시 8 또는 4로 낮추되 **전 arm에 동일 적용** |
| patience | 30 | |
| seed | 0 (필요 시 1, 2 추가) | |
| 기본 증강 | Ultralytics 기본값 유지 | 전 arm 동일하므로 비교에 영향 없음 |

---

## STEP 4 — 학습 함수

```python
from ultralytics import YOLO
import gc, torch

def train_one(ds, seed=0):
    name = f'{ds}_s{seed}'
    yaml = f'{DSBASE}/{ds}/data.yaml'
    model = YOLO(MODEL)
    model.train(data=yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                seed=seed, patience=PATIENCE,
                project=RUNDIR, name=name, exist_ok=True,
                val=True, plots=True, verbose=False)
    m = model.val(split='test', data=yaml,
                  project=RUNDIR, name=f'{name}_test', exist_ok=True)
    res = dict(dataset=ds, seed=seed,
               mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
               precision=float(m.box.mp), recall=float(m.box.mr))
    res['f1'] = 2*res['precision']*res['recall'] / max(res['precision']+res['recall'], 1e-9)
    json.dump(res, open(f'{RUNDIR}/{name}_test.json', 'w'), indent=2)
    print(res)
    del model; gc.collect(); torch.cuda.empty_cache()
    return res
```

---

## STEP 5 — 실행: `mh_board` (주 결과, leakage 0)

```python
results = []
for ds in arms_of('mh_board'):
    if os.path.exists(f"{RUNDIR}/{ds}_s0_test.json"):
        print(f"↷ skip {ds} — 이미 완료"); continue
    print(f"\n===== {ds} =====")
    results.append(train_one(ds, seed=0))
```

**소요 시간 목안** (T4, 100 epoch, imgsz 640, batch 16)

| split | arm 1개 | 5개 합 |
|---|---|---|
| `mh_board` (1212장) | 약 40–60분 | 약 4–5시간 |
| `mh_orig` (1477장) | 약 50–70분 | 약 5–6시간 |

Colab 세션 제한을 고려해 **한 번에 2–3개씩 나눠 실행**한다.
`RUNDIR`이 Drive이고 완료분은 `↷ skip`으로 건너뛰므로, 세션이 끊겨도 이어서 돌리면 된다.

---

## STEP 6 — 실행: `mh_orig` (참고)

```python
for ds in arms_of('mh_orig'):
    if os.path.exists(f"{RUNDIR}/{ds}_s0_test.json"):
        print(f"↷ skip {ds}"); continue
    print(f"\n===== {ds} =====")
    results.append(train_one(ds, seed=0))
```

---

## STEP 7 — 시드 반복 (권장)

`mh_board`의 test는 보드 2장뿐이라 변동이 크다. 시드 3개로 평균±표준편차를 낸다.

```python
for seed in [1, 2]:
    for ds in arms_of('mh_board'):
        if os.path.exists(f"{RUNDIR}/{ds}_s{seed}_test.json"):
            continue
        results.append(train_one(ds, seed=seed))
```

---

## STEP 8 — 결과 집계

```python
import pandas as pd

rows = [json.load(open(p)) for p in sorted(glob.glob(f'{RUNDIR}/*_test.json'))]
df = pd.DataFrame(rows)

ARM_NAME = {'': 'Baseline', 'randomcp': 'Random CP', 'brightnesscp': 'Brightness CP',
            'contextcp': 'Context CP', 'brightspeccp': 'Brightness+Spec CP'}
order = list(ARM_NAME.values())
df['split'] = df['dataset'].map(lambda d: 'mh_board' if d.startswith('mh_board') else 'mh_orig')
df['arm']   = df['dataset'].map(
    lambda d: ARM_NAME[d.replace('mh_board', '').replace('mh_orig', '').lstrip('_')])

agg = (df.groupby(['split', 'arm'])[['mAP50', 'mAP50_95', 'precision', 'recall', 'f1']]
         .agg(['mean', 'std']).round(4).reindex(order, level='arm'))
display(agg)
agg.to_csv(f'{RUNDIR}/summary.csv')
print('saved', f'{RUNDIR}/summary.csv')
```

### 8-1. Baseline 대비 개선폭

```python
for sp in SPLITS:
    sub = df[df.split == sp].groupby('arm')[['mAP50', 'recall']].mean()
    if 'Baseline' not in sub.index:
        continue
    base = sub.loc['Baseline']
    print(f'\n=== {sp} (Baseline 대비 %p) ===')
    for arm in order[1:]:
        if arm in sub.index:
            d = (sub.loc[arm] - base) * 100
            print(f'  {arm:<22} mAP50 {d.mAP50:+6.2f}   recall {d.recall:+6.2f}')
```

### 8-2. 보고 형식

```
                      mh_board (누수 0, 주 결과)     mh_orig (누수 있음, 참고)
arm                   mAP50    Recall                mAP50    Recall
Baseline               ...      ...                   ...      ...
Random CP              ...      ...                   ...      ...
Brightness CP          ...      ...                   ...      ...
Context CP             ...      ...                   ...      ...
Brightness+Spec CP     ...      ...                   ...      ...
```

---

## 해석 규칙 (사전 고정 — 결과를 보고 바꾸지 않는다)

| 관측 | 해석 |
|---|---|
| Brightness+Spec > Random | 제안 방법 성립. **Random이 box를 15% 더 가진 상태**에서의 우세이므로 데이터 양 효과가 아니다 |
| Context CP ≈ Brightness CP | **예측된 결과**. step3에서 유효 pad 비율이 양쪽 모두 85%로 동일했다 |
| Random CP ≤ Baseline | 가능하다. random은 sat MAD 108로 금속 링이 없는 비결함을 결함으로 라벨링한다 |
| arm 간 차이가 시드 표준편차 이내 | 유의하지 않다고 보고한다. `mh_board` test가 보드 2장임을 함께 명시한다 |

---

## 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| `Dataset ... not found` | STEP 1 미실행. `data.yaml`의 `path`가 Windows 경로로 남아 있다 |
| 라벨이 전부 무시되고 background로 학습 | 이미지/라벨 stem 불일치. step4 STEP 4(`verify_datasets.py`)로 `unpaired=0` 확인 |
| `CUDA out of memory` | `BATCH`를 8 또는 4로. **전 arm에 동일 적용**해야 비교가 유지된다 |
| 학습이 비정상적으로 느림 | Drive I/O 병목. STEP 2-1의 로컬 복사를 수행한다 |
| 세션 끊김 | `RUNDIR`이 Drive다. 동일 셀 재실행 시 완료분은 `↷ skip`된다 |
| arm 간 결과 차이가 거의 없음 | STEP 7로 시드를 3개로 늘린다. 그래도 없으면 없는 대로 보고한다 |

---

## 판정 / 다음 단계

- [ ] STEP 1 `data.yaml` 10개 재작성
- [ ] STEP 2 데이터셋 점검 — box 수가 문서 값과 일치
- [ ] `mh_board` 5개 arm 학습 완료 (`runs/*_test.json` 5개)
- [ ] `mh_orig` 5개 arm 학습 완료
- [ ] 시드 3개 반복 (권장)
- [ ] `runs/summary.csv` 생성, Baseline 대비 개선폭 산출

완료 시 → `보고서(simple).md` §5 평가 계획의 결과란을 채우고, RQ4를 확정한다.

---

## 무결성 / 정직 (_SPEC §5)

- **사후 튜닝 금지**: epochs·imgsz·batch·seed·모델을 결과 보고 후 변경하지 않는다.
  한 arm만 조건을 바꾸는 것은 비교 무효다.
- `mh_board` test는 **보드 2장**이다. 절대값 단독 해석 금지 — Δ와 시드 분산을 함께 보고한다.
- `mh_orig`는 `board+light+tile` 기준 test 163/163이 train에 존재하는 **누수 split**이다.
  절대값을 주 결과로 인용하지 않는다.
- **Random CP가 제안 방법보다 학습 box를 약 15% 더 갖는다**는 사실을 결과 표에 함께 명시한다.
- Context CP가 개선되지 않는 것은 **예측된 결과**다. 사후에 Context를 재튜닝해 살리지 않는다.
- 주요 지표는 Recall이다. mAP만 유리하게 골라 보고하지 않는다.
- 테스트 코드 신규 작성·pytest 금지 — 검증은 본 문서 셀 실행으로 한다.
