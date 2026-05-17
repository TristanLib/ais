#!/usr/bin/env python3
"""Export a repository Markdown manuscript to a readable PDF draft."""

from __future__ import annotations

import argparse
import html
import re
from urllib.parse import unquote
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, letter, landscape, portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Image as ReportLabImage,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    return escaped


def cjk_font_name() -> str:
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/53fe5be564086fefc7523ccd0a31200acf92e0e5.asset/AssetData/STHEITI.ttf",
        "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/10e7a462a671950b802274fad767b566ff8457d1.asset/AssetData/STXIHEI.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("CJKBody", path))
            return "CJKBody"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def make_styles(cjk_font: str | None = None, serif: bool = False) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body_font = cjk_font or ("Times-Roman" if serif else "Helvetica")
    heading_font = cjk_font or ("Times-Bold" if serif else "Helvetica-Bold")
    table_header_font = cjk_font or ("Times-Bold" if serif else "Helvetica-Bold")
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName=heading_font,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName=heading_font,
            fontSize=14,
            leading=17,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName=heading_font,
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "QuoteCustom",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=9,
            leading=12,
            leftIndent=18,
            rightIndent=18,
            textColor=colors.HexColor("#444444"),
            spaceBefore=4,
            spaceAfter=7,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f5f5f5"),
            borderColor=colors.HexColor("#dddddd"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=9.3,
            leading=12.5,
            leftIndent=16,
            firstLineIndent=-8,
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "TableCellCustom",
            parent=base["BodyText"],
            fontName=body_font,
            fontSize=6.6,
            leading=8,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCustom",
            parent=base["BodyText"],
            fontName=table_header_font,
            fontSize=6.6,
            leading=8,
            wordWrap="CJK",
            textColor=colors.white,
        ),
    }


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def collect_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        if not is_table_separator(lines[i]):
            rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
        i += 1
    return rows, i


def table_widths(column_count: int, available_width: float) -> list[float]:
    if column_count <= 0:
        return []
    if column_count >= 10:
        weights = [1.25, 1.4, 0.7, 0.82, 0.82, 0.82, 0.82, 0.78, 0.72, 0.72]
        if column_count != len(weights):
            weights = [1.0] * column_count
    elif column_count == 7:
        weights = [1.5, 0.9, 0.8, 0.8, 0.95, 1.05, 1.1]
    else:
        weights = [1.0] * column_count
    total = sum(weights)
    return [available_width * weight / total for weight in weights]


def build_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], available_width: float) -> Table:
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(data, colWidths=table_widths(max_cols, available_width), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def resolve_image_path(raw_target: str, base_dir: Path) -> Path:
    target = raw_target.strip().strip("<>").split()[0]
    target = unquote(target)
    path = Path(target)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def build_image(image_path: Path, alt_text: str, styles: dict[str, ParagraphStyle], available_width: float) -> list:
    if not image_path.exists():
        return [Paragraph(inline_markup(f"[Missing figure: {image_path}]"), styles["quote"])]
    image = ReportLabImage(str(image_path))
    max_height = 4.8 * inch
    scale = min(available_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    flowables = [image, Spacer(1, 0.06 * inch)]
    if alt_text:
        flowables.append(Paragraph(inline_markup(alt_text), styles["quote"]))
    flowables.append(Spacer(1, 0.08 * inch))
    return flowables


def markdown_to_story(markdown: str, styles: dict[str, ParagraphStyle], available_width: float, base_dir: Path) -> list:
    lines = markdown.splitlines()
    story: list = []
    in_code = False
    code_lines: list[str] = []
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 0.06 * inch))
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            rows, next_i = collect_table(lines, i)
            story.append(build_table(rows, styles, available_width))
            story.append(Spacer(1, 0.12 * inch))
            i = next_i
            continue

        if stripped == "\\pagebreak":
            story.append(PageBreak())
            i += 1
            continue

        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            alt_text, target = image_match.groups()
            story.extend(build_image(resolve_image_path(target, base_dir), alt_text, styles, available_width))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles["title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline_markup(stripped[3:].strip()), styles["h1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:].strip()), styles["h2"]))
        elif stripped.startswith(">"):
            story.append(Paragraph(inline_markup(stripped.lstrip("> ").strip()), styles["quote"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(inline_markup(stripped[2:].strip()), styles["bullet"], bulletText="-"))
        else:
            story.append(Paragraph(inline_markup(stripped), styles["body"]))
        i += 1

    if code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["code"]))
    return story


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(document.pagesize[0] - document.rightMargin, 0.35 * inch, f"Page {document.page}")
    canvas.restoreState()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="paper/submission_manuscript.md")
    parser.add_argument("--output", default="paper/submission_manuscript.pdf")
    parser.add_argument("--cjk", action="store_true", help="Use built-in CJK font support for Chinese/Japanese/Korean text.")
    parser.add_argument("--page-size", choices=["letter", "a4"], default="letter")
    parser.add_argument("--orientation", choices=["portrait", "landscape"], default="landscape")
    parser.add_argument("--serif", action="store_true", help="Use built-in Times fonts for an English manuscript draft.")
    parser.add_argument("--margin-inch", type=float, default=0.45, help="Uniform page margin in inches.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_name = cjk_font_name() if args.cjk else None
    styles = make_styles(cjk_font=font_name, serif=args.serif)
    base_page_size = A4 if args.page_size == "a4" else letter
    page_size = landscape(base_page_size) if args.orientation == "landscape" else portrait(base_page_size)
    margin = args.margin_inch * inch
    margins = {
        "leftMargin": margin,
        "rightMargin": margin,
        "topMargin": margin,
        "bottomMargin": max(margin, 0.55 * inch),
    }
    available_width = page_size[0] - margins["leftMargin"] - margins["rightMargin"]
    document = SimpleDocTemplate(str(output_path), pagesize=page_size, **margins)
    story = markdown_to_story(input_path.read_text(encoding="utf-8"), styles, available_width, input_path.parent)
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
