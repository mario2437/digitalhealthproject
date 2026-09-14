"""KSC2026 심사용 양식(DocForm_1)에 맞춰 Markdown 초안을 .docx로 변환한다.

사용법:
    python md2docx.py 초안.md 초안.docx

지원하는 Markdown (초안 작성 규칙):
    ---                         앞머리(front matter)
    title_ko: 국문 제목
    title_en: English Title
    ---
    ## 요약                     → 1단, 가운데 정렬 소제목
    (요약 문단)
    ## 1. 서론                  → 여기서부터 2단
    ### 1.1 소절
    일반 문단 (**굵게** 지원)
    - 글머리표 목록
    표 1. 캡션                  → 표 제목은 표 위
    | a | b |                   파이프 표
    ![그림 1. 캡션](figures/fig1.png){width=8}   → 그림 제목은 그림 아래, width는 cm
    ## 참고문헌
    [1] 저자, 제목, ...
    <!-- 주석 -->               → 무시
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

TEMPLATE = Path(__file__).with_name("ksc_review_template.docx")
FONT = "함초롬돋움"
BODY_PT = 9.5  # 학회 규정: 9pt 이상


def set_font(run, size, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)


def add_inline(par, text, size, bold=False):
    # **굵게** 와 `코드` 만 처리
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            set_font(par.add_run(part[2:-2]), size, True)
        elif part.startswith("`"):
            set_font(par.add_run(part[1:-1]), size, bold)
        else:
            set_font(par.add_run(part), size, bold)


def para(doc, text="", size=BODY_PT, bold=False, align=None, before=0, after=2, indent=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.15
    if indent:
        pf.first_line_indent = Pt(size)
    if align:
        p.alignment = align
    if text:
        add_inline(p, text, size, bold)
    return p


def set_cols(sect_pr, num):
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), "425")


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    ncol = max(len(r) for r in cells)
    t = doc.add_table(rows=len(cells), cols=ncol)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), "000000")
        borders.append(b)
    t._tbl.tblPr.append(borders)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(cells):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            add_inline(cell.paragraphs[0], r[j] if j < len(r) else "", BODY_PT - 1.5, bold=(i == 0))
    para(doc, after=2)


def convert(md_path, out_path):
    src = Path(md_path)
    text = re.sub(r"<!--.*?-->", "", src.read_text(encoding="utf-8"), flags=re.S)
    meta = {}
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]

    doc = Document(TEMPLATE)
    body = doc.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)

    para(doc, meta.get("title_ko", ""), 16, True, WD_ALIGN_PARAGRAPH.CENTER, after=4)
    para(doc, meta.get("title_en", ""), 12, True, WD_ALIGN_PARAGRAPH.CENTER, after=10)

    two_col_started = False
    lines = text.splitlines()
    i = 0
    buf = []

    def flush():
        if buf:
            para(doc, " ".join(s.strip() for s in buf), indent=two_col_started)
            buf.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("## "):
            flush()
            title = s[3:].strip()
            if title.replace(" ", "") == "요약":
                para(doc, "요   약", 10.5, True, WD_ALIGN_PARAGRAPH.CENTER, before=2, after=3)
            else:
                if not two_col_started:
                    # 제목·요약(1단)과 본문(2단)을 연속 구역으로 나눈다
                    doc.add_section(WD_SECTION.CONTINUOUS)
                    set_cols(doc.sections[-2]._sectPr, 1)
                    set_cols(doc.sections[-1]._sectPr, 2)
                    two_col_started = True
                para(doc, title, 10.5, True, before=6, after=2)
        elif s.startswith("### "):
            flush()
            para(doc, s[4:].strip(), BODY_PT + 0.5, True, before=3, after=1)
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
            continue
        elif re.match(r"^표\s*\d+\.", s):
            flush()
            para(doc, s, BODY_PT - 0.5, True, WD_ALIGN_PARAGRAPH.CENTER, before=3, after=2)
        elif s.startswith("!["):
            flush()
            mm = re.match(r"!\[(.*?)\]\((.*?)\)(\{width=([\d.]+)\})?", s)
            cap, path, width = mm.group(1), mm.group(2), float(mm.group(4) or 8)
            img = (src.parent / path).resolve()
            p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=3, after=0)
            if img.exists():
                p.add_run().add_picture(str(img), width=Cm(width))
            else:
                set_font(p.add_run(f"[그림 파일 없음: {path}]"), BODY_PT, True)
            para(doc, cap, BODY_PT - 0.5, True, WD_ALIGN_PARAGRAPH.CENTER, after=4)
        elif re.match(r"^[-*] ", s):
            flush()
            p = para(doc, "· " + s[2:], after=1)
            p.paragraph_format.left_indent = Pt(BODY_PT)
        elif re.match(r"^\[\d+\]", s):
            flush()
            p = para(doc, s, BODY_PT - 1, after=1)
            p.paragraph_format.left_indent = Pt(14)
            p.paragraph_format.first_line_indent = Pt(-14)
        else:
            buf.append(line)
        i += 1
    flush()
    doc.core_properties.author = ""  # 심사용: 문서 속성에 작성자를 남기지 않는다
    doc.core_properties.last_modified_by = ""
    doc.save(out_path)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    print("saved", sys.argv[2])
