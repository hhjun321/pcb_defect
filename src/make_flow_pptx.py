"""전체 연구 흐름(정본) 다이어그램을 1장짜리 PPTX로 생성한다.

근거 문서:
  missing_hole_research_flow.md §3 (원안 흐름)
  보고서(full).md §0.2 정본 설계, §0.3 교체 2건

출력: {PCB_ROOT}/outputs/slides/pcb_mh_flow.pptx
"""
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

PCB_ROOT = os.environ.get("PCB_ROOT", r"D:/project/pcb_defect")
OUT_DIR = os.path.join(PCB_ROOT, "outputs", "slides")
OUT_PATH = os.path.join(OUT_DIR, "pcb_mh_flow.pptx")

FONT = "맑은 고딕"

INK = RGBColor(0x1A, 0x20, 0x2C)
MUTED = RGBColor(0x5A, 0x67, 0x80)
SWAP_INK = RGBColor(0x9C, 0x4A, 0x0E)
ARROW = RGBColor(0x94, 0xA3, 0xB8)

# kind -> (fill, line, text)
STYLE = {
    "data":   (RGBColor(0xE9, 0xED, 0xF2), RGBColor(0x4A, 0x55, 0x68), INK),
    "proc":   (RGBColor(0xDC, 0xE9, 0xF7), RGBColor(0x2B, 0x6C, 0xB0), INK),
    "swap":   (RGBColor(0xFD, 0xEB, 0xD8), RGBColor(0xDD, 0x6B, 0x20), INK),
    "final":  (RGBColor(0xE2, 0xF2, 0xE6), RGBColor(0x2F, 0x85, 0x5A), INK),
    "banner": (RGBColor(0x2D, 0x3A, 0x4A), RGBColor(0x2D, 0x3A, 0x4A), RGBColor(0xFF, 0xFF, 0xFF)),
}


def box(slide, x, y, w, h, main, sub=None, kind="proc", size=12, sub_size=9.5):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.12
    fill, line, color = STYLE[kind]
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.25)
    shape.shadow.inherit = False

    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Emu(36000)
    tf.margin_top = tf.margin_bottom = Emu(18000)

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = main
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = color

    for line in (sub.split("\n") if sub else []):
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = line
        r2.font.size = Pt(sub_size)
        r2.font.name = FONT
        r2.font.color.rgb = color if kind == "banner" else MUTED
    return shape


def arrow(slide, x, y, w, h, direction="down"):
    shp = MSO_SHAPE.DOWN_ARROW if direction == "down" else MSO_SHAPE.RIGHT_ARROW
    shape = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ARROW
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def text(slide, x, y, w, h, lines, size=10, bold_first=False, color=MUTED, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(2)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.name = FONT
        r.font.bold = bold_first and i == 0
        r.font.color.rgb = color
    return tb


def stage(slide, x, y, label):
    """단계 배지 (step1 ... exp_yolo)."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(0.82), Inches(0.26))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x2D, 0x3A, 0x4A)
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(9)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return shape


def table(slide, x, y, w, col_w, rows, row_h=0.27, size=9, hi_row=None):
    """단순 수치표. hi_row 지정 시 해당 행을 강조색으로 칠한다."""
    n_r, n_c = len(rows), len(rows[0])
    gt = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    tbl = gt.table
    tbl.first_row = False
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Emu(36000)
            cell.margin_top = cell.margin_bottom = 0
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RGBColor(0x2D, 0x3A, 0x4A)
                fg = RGBColor(0xFF, 0xFF, 0xFF)
            elif i == hi_row:
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xEB, 0xD8)
                fg = SWAP_INK
            else:
                cell.fill.fore_color.rgb = RGBColor(0xF7, 0xF9, 0xFC)
                fg = INK
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.bold = (i == 0) or (i == hi_row)
            r.font.color.rgb = fg
    return gt


def card(slide, x, y, w, h, title, lines, accent=RGBColor(0x2B, 0x6C, 0xB0),
         fill=RGBColor(0xF7, 0xF9, 0xFC), size=9.5):
    """제목 + 본문 여러 줄을 담는 설명 카드."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = accent
    shape.line.width = Pt(1.0)
    shape.shadow.inherit = False

    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Emu(90000)
    tf.margin_top = Emu(63000)
    tf.margin_bottom = Emu(45000)

    p = tf.paragraphs[0]
    p.space_after = Pt(4)
    r = p.add_run()
    r.text = title
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = accent

    for line in lines:
        pi = tf.add_paragraph()
        pi.space_after = Pt(2)
        ri = pi.add_run()
        ri.text = line
        ri.font.size = Pt(size)
        ri.font.name = FONT
        ri.font.color.rgb = INK if not line.startswith("✗") else SWAP_INK
    return shape


def slide_flow(prs):
    """1쪽 — 전체 연구 흐름 (정본)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # 제목
    text(slide, 0.55, 0.20, 9.0, 0.42, ["전체 연구 흐름"], size=26, bold_first=True, color=INK)
    text(slide, 0.58, 0.68, 11.5, 0.3,
         ["밝기 기반 후보 탐색 → 정반사 재질 검증 → 코어 이식 합성 (정본 설계)"], size=12)

    # 데이터 소스 배너
    box(slide, 0.55, 1.03, 12.2, 0.52, "원본 PCB 데이터셋",
        sub="   missing_hole 1832장 · box 3612개   |   split 2종: mh_board(보드 분리) / mh_orig(공개 split)",
        kind="banner", size=12.5, sub_size=10)

    LX, LW = 0.55, 2.9          # 왼쪽: 결함 밝기 템플릿 열
    CX, CW = 4.85, 3.6          # 가운데: 본 파이프라인 열
    lcx = LX + LW / 2
    ccx = CX + CW / 2

    arrow(slide, lcx - 0.15, 1.59, 0.3, 0.22)
    arrow(slide, ccx - 0.15, 1.59, 0.3, 0.22)

    # ── 왼쪽: 결함 밝기 템플릿 (step1)
    stage(slide, LX, 1.58, "step1")
    box(slide, LX, 1.85, LW, 0.42, "결함 이미지 (missing_hole)", kind="data", size=11.5)
    arrow(slide, lcx - 0.15, 2.32, 0.3, 0.22)
    box(slide, LX, 2.58, LW, 0.62, "결함 영역 추출 · 크기 정규화",
        sub="32×32 리사이즈 · crop별 z 정규화", size=11.5)
    arrow(slide, lcx - 0.15, 3.25, 0.3, 0.22)
    box(slide, LX, 3.51, LW, 0.66, "공통 밝기 패턴 D_mean",
        sub="중심 z −1.00 / 링 피크 +0.84 (과녁형)", size=11.5)

    # ── 가운데: 정상 PCB → 합성
    box(slide, CX, 1.85, CW, 0.42, "정상 PCB 이미지", kind="data", size=11.5)
    arrow(slide, ccx - 0.15, 2.32, 0.3, 1.12)

    stage(slide, CX, 3.19, "step2")
    box(slide, CX, 3.51, CW, 0.66, "밝기 기반 후보 탐색 + 링 게이트",
        sub="다중 스케일 NCC {20…48} · recall@50 90.9%", size=12)
    arrow(slide, 3.55, 3.72, 1.22, 0.24, direction="right")   # D_mean → 탐색

    arrow(slide, ccx - 0.15, 4.22, 0.3, 0.22)
    box(slide, CX, 4.48, CW, 0.52, "후보 위치 다수",
        sub="이미지당 200 → 25개 (게이트 통과분)", size=12)

    arrow(slide, ccx - 0.15, 5.05, 0.3, 0.22)
    stage(slide, CX - 0.95, 5.44, "step3")
    box(slide, CX, 5.31, CW, 0.52, "정반사 검증 (후보 재질 판별)",
        sub="유효 관통홀 pad만 통과 · spec 특징", kind="swap", size=12)

    arrow(slide, ccx - 0.15, 5.88, 0.3, 0.22)
    stage(slide, CX - 0.95, 6.27, "step4")
    box(slide, CX, 6.14, CW, 0.52, "코어 이식 합성 (Core Transplant)",
        sub="대상 pad 금속 링 보존 · 초록 코어만 이식", kind="swap", size=12)

    # ── 아래: 합성 데이터셋 → 학습 → 평가
    arrow(slide, 6.05, 6.71, 0.3, 0.22)
    box(slide, 4.85, 6.97, 2.4, 0.45, "합성 데이터셋", kind="data", size=11.5)
    arrow(slide, 7.32, 7.08, 0.33, 0.24, direction="right")
    stage(slide, 3.92, 7.06, "exp_yolo")
    box(slide, 7.70, 6.97, 2.2, 0.45, "YOLO 학습", size=11.5)
    arrow(slide, 9.97, 7.08, 0.33, 0.24, direction="right")
    box(slide, 10.35, 6.97, 2.4, 0.45, "성능 평가 (mAP50, Δ)", kind="final", size=11.5)

    # ── 오른쪽 주석: 원안 대비 교체 2건
    text(slide, 8.70, 5.28, 4.3, 0.7,
         ["◀ 교체 1 — 원안 '주변 구조(Context) 검증'",
          "   Context AUC 0.524 (무작위 수준), 최적 λ = 0.00",
          "   Specular 0.874 → 밝기 결합 0.918, top-20 유효 pad 100%"],
         size=9.5, bold_first=True, color=SWAP_INK)
    text(slide, 8.70, 6.11, 4.3, 0.7,
         ["◀ 교체 2 — 원안 'donor crop 통째 Copy-Paste'",
          "   통째 방식 채도 MAD 32.4 → 코어 이식 21.6",
          "   대상 픽셀 사용 → 조명·형태 정합이 구조적으로 보장"],
         size=9.5, bold_first=True, color=SWAP_INK)

    # ── 왼쪽 하단: 비교 실험군 + QC
    text(slide, 0.55, 4.45, 3.15, 0.3, ["비교 실험군 4종 (삽입 위치만 다름)"],
         size=10.5, bold_first=True, color=INK)
    text(slide, 0.62, 4.78, 3.15, 1.1,
         ["random         균등 무작위 (대조군)",
          "brightness    밝기 탐색 + 링 게이트",
          "context        밝기 → 주변 구조 재정렬 (기각안)",
          "brightspec    z(ncc) + 1.5·z(spec) ← 제안 방법"],
         size=9.5)
    text(slide, 0.62, 6.02, 3.15, 0.6,
         ["합성 QC: 유도 3종 채도 MAD 19~22 (서로 2.6 이내)",
          "random 108.3 — 금속 링 없음 = 물리적으로 결함 아님"],
         size=9.5, color=SWAP_INK)


def plain(slide, shp_type, x, y, w, h, fill=None, line=None, dash=False, lw=1.0):
    """도식용 기본 도형 (텍스트 없음)."""
    shape = slide.shapes.add_shape(shp_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.shadow.inherit = False
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(lw)
        if dash:
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return shape


def caption(slide, x, y, w, lines, size=8.5, color=MUTED, bold=False, align=PP_ALIGN.CENTER):
    return text(slide, x, y, w, 0.2 * len(lines), lines, size=size, color=color,
                bold_first=bold, align=align)


def slide_window(prs):
    """2쪽 — sliding window 도식 (원안 §7 그림의 도식화)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    text(slide, 0.55, 0.20, 11.0, 0.42,
         ["step2 — 정상 PCB에서 밝기 기반 후보 탐색 (sliding window)"],
         size=24, bold_first=True, color=INK)
    text(slide, 0.58, 0.66, 12.2, 0.3,
         ["정상 PCB 위를 window로 훑으며 각 위치의 밝기 패턴을 D_mean과 비교한다"], size=11)

    PAD_FILL = RGBColor(0xC2, 0xC8, 0xD2)
    PAD_LINE = RGBColor(0x8A, 0x93, 0xA1)
    BOARD_FILL = RGBColor(0xE4, 0xEE, 0xE1)
    BOARD_LINE = RGBColor(0x6E, 0x86, 0x66)
    WIN = RGBColor(0x2B, 0x6C, 0xB0)
    HIT = RGBColor(0xDD, 0x6B, 0x20)

    # ── 보드 캔버스
    caption(slide, 0.55, 1.12, 3.2, ["Normal PCB (정상 PCB 이미지)"], size=11, bold=True,
            color=INK, align=PP_ALIGN.LEFT)
    plain(slide, MSO_SHAPE.RECTANGLE, 0.55, 1.42, 6.95, 4.72, fill=BOARD_FILL, line=BOARD_LINE, lw=1.5)

    # pad 배열 (장식) — 5열 × 4행
    cxs = [1.35 + i * 1.30 for i in range(5)]
    cys = [2.15 + j * 1.05 for j in range(4)]
    for cy in cys:
        for cx in cxs:
            plain(slide, MSO_SHAPE.OVAL, cx - 0.26, cy - 0.26, 0.52, 0.52,
                  fill=PAD_FILL, line=PAD_LINE, lw=0.75)
            plain(slide, MSO_SHAPE.OVAL, cx - 0.11, cy - 0.11, 0.22, 0.22,
                  fill=BOARD_FILL, line=PAD_LINE, lw=0.5)

    def window(cx, cy, side, color, dash=True, lw=1.5):
        plain(slide, MSO_SHAPE.RECTANGLE, cx - side / 2, cy - side / 2, side, side,
              fill=None, line=color, dash=dash, lw=lw)

    W = 1.00   # 화면상 window 한 변 (= 32 px 상당)

    window(cxs[0], cys[0], W, WIN)
    window(cxs[1], cys[0], W, WIN)
    window(cxs[2], cys[1], W, WIN)
    window(cxs[4], cys[2], W, HIT, dash=False, lw=2.25)

    caption(slide, 0.68, 2.72, 1.10, ["32×32 window"], size=8.5, color=WIN, bold=True)
    caption(slide, 5.93, 4.82, 1.30, ["NCC 최대 → 후보 피크"], size=8.5, color=HIT, bold=True)

    # 이동 표시 — pad 사이 빈 공간에만 배치
    arrow(slide, 1.87, 2.03, 0.26, 0.24, direction="right")   # 열 이동
    caption(slide, 1.62, 3.30, 0.86, ["행 이동"], size=8.5)
    arrow(slide, 1.88, 2.78, 0.24, 0.48)                      # 행 이동
    arrow(slide, 3.17, 2.03, 0.26, 0.24, direction="right")
    arrow(slide, 4.47, 3.08, 0.26, 0.24, direction="right")

    caption(slide, 1.60, 5.72, 5.00,
            ["모든 위치 p × 7 스케일에 대해 NCC 계산"], size=9.5, color=INK, bold=True)
    caption(slide, 1.60, 5.96, 5.00,
            ["점선 = 각 위치의 window W_p     실선(주황) = 응답 최대 위치"], size=8.5)

    # ── 오른쪽: 스케일 / 점수 계산
    RX = 7.75
    caption(slide, RX, 1.12, 5.0, ["① 다중 스케일 — 원안 32×32 단일 → 정본 7 스케일"],
            size=11, bold=True, color=INK, align=PP_ALIGN.LEFT)
    plain(slide, MSO_SHAPE.ROUNDED_RECTANGLE, RX, 1.42, 5.03, 1.42,
          fill=RGBColor(0xF7, 0xF9, 0xFC), line=RGBColor(0x2B, 0x6C, 0xB0), lw=1.0)
    for k, (side, lab) in enumerate([(0.44, "20"), (0.62, "28"), (0.80, "36"), (1.02, "48")]):
        cx = RX + 0.85 + k * 1.12
        window(cx, 2.05, side, WIN, dash=True, lw=1.25)
        caption(slide, cx - 0.45, 2.60, 0.9, [f"{lab}×{lab}"], size=8.5)
    caption(slide, RX + 0.10, 1.50, 2.2, ["스케일 {20, 24, 28, 32, 36, 40, 48}"], size=8.5,
            align=PP_ALIGN.LEFT)

    caption(slide, RX, 3.00, 5.0, ["② 점수 계산과 후보 추출"],
            size=11, bold=True, color=INK, align=PP_ALIGN.LEFT)
    chain = [
        ("window별 z 정규화 후 상관", "NCC(p) = corr(D_mean, W_p) — cv2.matchTemplate TM_CCOEFF_NORMED"),
        ("스케일별 score map 합성", "픽셀별 최대 응답 선택 → 크기 무관 단일 응답맵"),
        ("NMS 피크 추출", "국소 최대만 남김 → 상위 200개 후보"),
        ("링 게이트 통과분만 유지", "이미지당 25개 · recall@10 67.5% → 82.1%"),
    ]
    for i, (head, sub) in enumerate(chain):
        y = 3.30 + i * 0.72
        kind = "swap" if i == len(chain) - 1 else "proc"
        box(slide, RX, y, 5.03, 0.58, head, sub=sub, kind=kind, size=10.5, sub_size=8.5)
        if i < len(chain) - 1:
            arrow(slide, RX + 2.40, y + 0.60, 0.24, 0.11)

    # ── 각주
    text(slide, 0.58, 6.34, 12.2, 0.6,
         ["원안은 32×32 단일 스케일 + MSE였다. box 실측 median 27.4 px이지만 maxwh ≤ 32를 만족하는 box는 64%뿐이라 "
          "단일 스케일은 큰 hole을 놓친다 → 7 스케일로 교체.",
          "MSE는 조명 차이(조명 10종, crop 평균 밝기 편차 14.6 grey level)가 패턴 차이를 압도한다. "
          "window별 z 정규화 상관(NCC)은 이 문제를 구조적으로 제거한다 (R² 0.269 → 0.398)."],
         size=9.5)


def slide_search(prs):
    """3쪽 — step2 탐색 성능·게이트 상세."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    text(slide, 0.55, 0.20, 10.5, 0.42,
         ["step2 — 탐색 성능과 링 게이트"], size=24, bold_first=True, color=INK)
    text(slide, 0.58, 0.66, 12.0, 0.3,
         ["다중 스케일 sliding window NCC → 스케일 합성·NMS → 링 게이트  "
          "(원안: 32×32 단일 스케일 + MSE → 정본: 7 스케일 + 정규화 상호상관)"], size=11)

    # ── 파이프라인 5단계 (가로)
    PW, PITCH, PY, PH = 2.2, 2.62, 1.12, 1.02
    steps = [
        ("① 입력: 정상 PCB 이미지",
         "600×600 / 601×601 혼재\n이미지마다 W·H 읽어 좌표 환산", "data"),
        ("② 다중 스케일 템플릿",
         "D_mean을 7 스케일로 리사이즈\n{20, 24, 28, 32, 36, 40, 48}", "proc"),
        ("③ sliding window NCC",
         "cv2.matchTemplate\nTM_CCOEFF_NORMED\n= window별 z 정규화", "proc"),
        ("④ 스케일 합성 + NMS",
         "픽셀별 최대 응답 합성\n피크 추출 → 상위 200개", "proc"),
        ("⑤ 링 게이트",
         "GT 실측 임계로 필터\n200개 → 이미지당 25개", "swap"),
    ]
    for i, (title, sub, kind) in enumerate(steps):
        x = 0.55 + i * PITCH
        box(slide, x, PY, PW, PH, title, sub=sub, kind=kind, size=11, sub_size=8.5)
        if i < len(steps) - 1:
            arrow(slide, x + PW + 0.04, PY + PH / 2 - 0.12, 0.34, 0.24, direction="right")

    # ── 링 게이트 조건
    text(slide, 0.55, 2.42, 4.0, 0.28, ["링 게이트 조건 (GT 2194개 실측 분포에서 고정)"],
         size=10.5, bold_first=True, color=INK)
    card(slide, 0.55, 2.74, 3.95, 1.82, "통과 조건", [
        "ring_z ≥ 0.31            링이 주변보다 밝은가",
        "contrast ≥ 0.38        링 − 코어 대비",
        "ring_min ≥ −1.11      16섹터 각도 완전성",
        "ring_sat ≤ 100         링 = 금속 (초록 기판 배제)",
        "ring_val ≤ 195         과노출 백색 억제",
    ], size=9.5)

    # ── 게이트 ablation
    text(slide, 4.80, 2.42, 4.6, 0.28, ["게이트 ablation (mh_board val 292장 / GT 636)"],
         size=10.5, bold_first=True, color=INK)
    table(slide, 4.80, 2.74, 4.55, [1.75, 0.72, 0.78, 0.78, 0.52],
          [["구성", "cand/img", "recall@10", "recall@50", "rank"],
           ["게이트 없음", "200", "67.5%", "92.6%", "3"],
           ["ring gate (채택)", "25", "82.1%", "90.9%", "2"],
           ["ring + spec", "23", "76.7%", "85.2%", "2"],
           ["ring + spec + grid-cap", "16", "77.0%", "84.0%", "2"]],
          row_h=0.29, size=9, hi_row=2)

    # ── recall@K
    text(slide, 9.70, 2.42, 3.2, 0.28, ["게이트 없는 원 탐색 recall@K"],
         size=10.5, bold_first=True, color=INK)
    table(slide, 9.70, 2.74, 3.08, [1.0, 2.08],
          [["K", "recall"],
           ["1", "31.1%"],
           ["5", "62.7%"],
           ["10", "67.5%  ← 원안 Top-10"],
           ["20", "83.8%"],
           ["50", "92.6%  ← 채택 구간"],
           ["100", "95.8%"]],
          row_h=0.26, size=9, hi_row=5)

    # ── 하단 카드 3종
    card(slide, 0.55, 4.72, 3.95, 1.92, "stage A / stage B 분리", [
        "stage A  링 게이트만 — 탐지 성능·ablation 용",
        "stage B  링 + spec + grid-cap — step4 삽입 후보 용",
        "",
        "✗ stage A에 spec 게이트 금지",
        "   step3(정반사)의 기여도 측정이 오염된다",
    ])
    card(slide, 4.80, 4.72, 4.55, 1.92, "기각된 판별자 (재시도 금지)", [
        "✗ 회전대칭 s90 / s180 — GT median 0.34 < FP 0.4~0.78",
        "   글자 글리프가 이 스케일에선 오히려 더 대칭적",
        "✗ ring_sat 단독 실크스크린 판별 — 흰 페인트도 저채도",
        "✗ min-distance max(40px, 2.0·s) — 붙은 GT까지 삭제",
        "   recall@50 92.6% → 84.1% → grid-cap(100px 셀당 2개)로 교체",
    ], accent=RGBColor(0xC0, 0x53, 0x21))
    card(slide, 9.70, 4.72, 3.08, 1.92, "지표·오검출 주의", [
        "AUC 0.981 = 무작위 배경 대비 (무의미)",
        "AUC 0.703 = 정상 pad hard negative 대비",
        "→ 두 수치 혼용 인용 금지",
        "",
        "오검출 3종: 실크스크린 글자(o, e, P),",
        "pad 사이 틈, 국소 공간 쏠림",
    ], accent=RGBColor(0xC0, 0x53, 0x21))

    # ── 각주
    text(slide, 0.58, 6.78, 12.2, 0.5,
         ["원안 MSE → 정규화 상호상관(NCC) 교체 근거: 조명 10종으로 crop 평균 밝기 편차 14.6 grey level. "
          "raw MSE는 조명 차이가 패턴 차이를 압도한다 (설명력 R² raw 0.269 → crop별 z 정규화 0.398).",
          "템플릿 D_mean은 mh_board train(1212장 / 2328 crop)만으로 생성한다. val·test 누출 금지."],
         size=9.5)


def slide_experiment(prs):
    """4쪽 — 비교 실험 설계 (원안 §12 3군 → 정본 5군 × split 2종)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    text(slide, 0.55, 0.20, 11.5, 0.42, ["비교 실험 설계"], size=24, bold_first=True, color=INK)
    text(slide, 0.58, 0.66, 12.2, 0.3,
         ["원안 3군(Baseline / Random CP / Proposed) → 정본 5군 × split 2종 = 데이터셋 10종. "
          "삽입 위치만 다르고 결함 렌더링·삽입 시도 수는 전 arm 동일하다."], size=11)

    # ── 실험군 정의
    caption(slide, 0.55, 1.08, 5.0, ["실험군 정의"], size=11, bold=True, color=INK,
            align=PP_ALIGN.LEFT)
    table(slide, 0.55, 1.38, 12.23, [1.45, 4.55, 1.05, 1.28, 3.90],
          [["실험군", "삽입 위치 결정", "spec 게이트", "합성 box(board)", "원안 대응"],
           ["baseline", "증강 없음 — 실제 결함만", "—", "0", "Baseline (Original Dataset)"],
           ["random", "균등 무작위, 크기는 GT 분포에서 추출, 겹침만 회피", "—", "2176",
            "Random CP (대조군)"],
           ["brightness", "stage-A NCC + 링 게이트 + grid cap", "없음", "1884",
            "원안 Step 4~5 (밝기까지만)"],
           ["context", "brightness top-20 → context 유사도 재정렬", "없음", "2215",
            "원안 Proposed → 기각, negative result로 유지"],
           ["brightspec", "z(ncc) + 1.5·z(spec) — 정반사 재질 증거 결합", "적용", "1997",
            "정본 제안 방법 (원안 Context를 대체)"]],
          row_h=0.32, size=9.5, hi_row=5)

    # ── 데이터셋 10종
    caption(slide, 0.55, 3.52, 4.0, ["데이터셋 10종 (split 2종 × 5군)"], size=11, bold=True,
            color=INK, align=PP_ALIGN.LEFT)
    card(slide, 0.55, 3.82, 3.95, 2.06, "split 2종을 항상 병기한다", [
        "mh_board   보드 단위 분리 — leakage 0 / 82",
        "                 보수적 추정. test 보드 2장뿐",
        "mh_orig     공개 split 유지 — board+light+tile",
        "                 163/163 누출. 비교용으로만 인용",
        "",
        "각 split × {baseline, random, brightness,",
        "context, brightspec} = 10개 데이터셋",
        "data.yaml 단일 클래스 0: missing_hole",
    ], size=9.5)

    # ── train box 예산
    caption(slide, 4.80, 3.52, 4.6, ["train box 예산 (mh_board, 실측)"], size=11, bold=True,
            color=INK, align=PP_ALIGN.LEFT)
    table(slide, 4.80, 3.82, 4.55, [1.90, 0.85, 0.95, 0.85],
          [["데이터셋", "실제", "합성", "합계"],
           ["mh_board (baseline)", "2328", "0", "2328"],
           ["  + randomcp", "2328", "2176", "4504"],
           ["  + brightnesscp", "2328", "1884", "4212"],
           ["  + contextcp", "2328", "2215", "4543"],
           ["  + brightspeccp", "2328", "1997", "4325"]],
          row_h=0.30, size=9.5, hi_row=5)
    caption(slide, 4.80, 5.72, 4.55,
            ["삽입 시도는 전 arm 2회/이미지로 동일. 유도 방식은 게이트 탈락분을 거부해 실현 수가 적다."],
            size=8.5, align=PP_ALIGN.LEFT)

    # ── 통제 조건
    caption(slide, 9.70, 3.52, 3.2, ["통제 조건 (교란변수 차단)"], size=11, bold=True,
            color=INK, align=PP_ALIGN.LEFT)
    card(slide, 9.70, 3.82, 3.08, 2.06, "전 arm 공통", [
        "삽입 시도 2회 / 이미지",
        "seed 0 (synthesis_meta.json 기록값)",
        "동일 후보 pool을 정렬만 달리함",
        "  → 차이는 순위 전략에서만 발생",
        "val / test 라벨 SHA256 전 arm 동일",
        "합성 QC: 유도 3종 채도 MAD 19~22",
        "  (서로 2.6 이내 = 렌더링 품질 동일)",
    ], size=9.5)

    # ── 해석 규칙
    card(slide, 0.55, 6.08, 12.23, 1.10, "결과 해석 규칙 (사전 고정 — exp_yolo_execute.md)", [
        "지표는 mAP50과 baseline 대비 Δ. random은 brightspec보다 합성 box가 board +9.0%(2176 vs 1997), "
        "orig +11.0%(2682 vs 2417), brightness 대비로는 +15.5% / +17.9% 많다 → 데이터 양 반론을 원천 봉쇄한다.",
        "mh_board test는 보드 2장뿐이므로 절대값 단독 해석 금지, Δ와 시드 분산을 함께 본다. "
        "게이트 임계·λ_spec 1.5·n_insert 2·seed·epochs는 결과를 본 뒤 변경하지 않는다.",
    ], accent=RGBColor(0xC0, 0x53, 0x21), size=9.5)


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_flow(prs)
    slide_window(prs)
    slide_search(prs)
    slide_experiment(prs)

    os.makedirs(OUT_DIR, exist_ok=True)
    prs.save(OUT_PATH)
    print("saved:", OUT_PATH, f"({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
