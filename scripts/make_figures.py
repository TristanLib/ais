#!/usr/bin/env python3
"""Create lightweight PNG figures from final experiment artifacts.

This intentionally uses only the Python standard library so the conservative
pipeline still creates auditable figures before the plotting environment is
fully repaired. The source CSV files remain the authority for publication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


class Canvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx : idx + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        left = max(0, x)
        right = min(self.width, x + w)
        if right <= left:
            return
        for yy in range(max(0, y), min(self.height, y + h)):
            start = (yy * self.width + left) * 3
            end = (yy * self.width + right) * 3
            self.pixels[start:end] = bytes(color) * (right - left)

    def line(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x, y = x1, y1
        while True:
            self.set_pixel(x, y, color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for row_idx, row in enumerate(glyph):
                for col_idx, value in enumerate(row):
                    if value == "1":
                        self.rect(cursor + col_idx * scale, y + row_idx * scale, scale, scale, color)
            cursor += 6 * scale

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        row_width = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * row_width : (y + 1) * row_width])

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def short_name(model: str) -> str:
    names = {
        "constant_velocity": "CV",
        "linear_lstsq": "LINEAR",
        "lstm_baseline": "LSTM",
        "transformer_baseline": "TRANSFORMER",
    }
    return names.get(model, model.upper()[:12])


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier * base
        if value <= candidate:
            return candidate
    return 10 * base


def make_bar_chart(metrics: list[dict[str, str]], path: Path) -> dict[str, Any]:
    ok_rows = [row for row in metrics if row.get("status") == "ok" and as_float(row.get("ade_meters", "")) is not None]
    canvas = Canvas(1200, 720)
    black = (30, 34, 40)
    gray = (210, 214, 220)
    colors = [(46, 94, 170), (214, 95, 61), (68, 145, 98), (130, 84, 180)]
    plot_x, plot_y, plot_w, plot_h = 110, 95, 1010, 500
    canvas.text(110, 35, "MODEL ADE COMPARISON (M)", black, 3)
    canvas.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, black)
    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_h, black)

    max_ade = nice_max(max([as_float(row["ade_meters"]) or 0.0 for row in ok_rows], default=1.0))
    for tick in range(0, 6):
        value = max_ade * tick / 5
        y = int(plot_y + plot_h - (value / max_ade) * plot_h)
        canvas.line(plot_x, y, plot_x + plot_w, y, gray)
        canvas.text(15, y - 10, f"{value:.0f}", black, 2)

    if ok_rows:
        gap = 80
        bar_w = max(70, int((plot_w - gap * (len(ok_rows) + 1)) / len(ok_rows)))
        for idx, row in enumerate(ok_rows):
            ade = as_float(row["ade_meters"]) or 0.0
            bar_h = int((ade / max_ade) * plot_h)
            x = plot_x + gap + idx * (bar_w + gap)
            y = plot_y + plot_h - bar_h
            canvas.rect(x, y, bar_w, bar_h, colors[idx % len(colors)])
            canvas.text(x, y - 30, f"{ade:.1f}", black, 2)
            canvas.text(x, plot_y + plot_h + 20, short_name(row["model"]), black, 2)
    else:
        canvas.text(320, 330, "NO OK MODEL METRICS", black, 3)

    canvas.text(110, 655, "SOURCE: OUTPUTS/FINAL/MODEL_METRICS.CSV", (90, 96, 106), 2)
    canvas.save_png(path)
    return {"path": str(path), "type": "bar_chart", "models": [row.get("model") for row in ok_rows]}


def make_distribution(errors: list[dict[str, str]], path: Path) -> dict[str, Any]:
    by_model: dict[str, list[float]] = {}
    for row in errors:
        value = as_float(row.get("ade_meters", ""))
        if value is not None:
            by_model.setdefault(row.get("model", ""), []).append(value)

    canvas = Canvas(1200, 720)
    black = (30, 34, 40)
    gray = (210, 214, 220)
    colors = [(46, 94, 170), (214, 95, 61), (68, 145, 98), (130, 84, 180)]
    plot_x, plot_y, plot_w, plot_h = 110, 95, 1010, 500
    canvas.text(110, 35, "ADE ERROR DISTRIBUTION (M)", black, 3)
    canvas.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, black)
    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_h, black)

    values = [value for vals in by_model.values() for value in vals]
    if not values:
        canvas.text(340, 330, "NO PER-SAMPLE ERRORS", black, 3)
        canvas.save_png(path)
        return {"path": str(path), "type": "histogram", "models": []}

    x_max = nice_max(sorted(values)[int(0.95 * (len(values) - 1))])
    bins = 20
    selected_models = list(by_model)[:4]
    histograms: dict[str, list[int]] = {}
    max_count = 1
    for model in selected_models:
        counts = [0] * bins
        for value in by_model[model]:
            idx = min(bins - 1, int(max(0.0, min(value, x_max)) / x_max * bins))
            counts[idx] += 1
        histograms[model] = counts
        max_count = max(max_count, max(counts))

    for tick in range(0, 6):
        y = int(plot_y + plot_h - (tick / 5) * plot_h)
        canvas.line(plot_x, y, plot_x + plot_w, y, gray)
        canvas.text(35, y - 10, f"{max_count * tick / 5:.0f}", black, 2)

    bin_w = plot_w / bins
    sub_w = max(2, int(bin_w / max(1, len(selected_models))))
    for model_idx, model in enumerate(selected_models):
        color = colors[model_idx % len(colors)]
        for bin_idx, count in enumerate(histograms[model]):
            h = int((count / max_count) * plot_h)
            x = int(plot_x + bin_idx * bin_w + model_idx * sub_w)
            y = plot_y + plot_h - h
            canvas.rect(x, y, sub_w, h, color)
        legend_x = 810
        legend_y = 105 + model_idx * 34
        canvas.rect(legend_x, legend_y, 22, 14, color)
        canvas.text(840, legend_y - 2, short_name(model), black, 2)

    canvas.text(110, 610, f"X AXIS CAPPED AT APPROX. 95TH PERCENTILE: {x_max:.0f} M", (90, 96, 106), 2)
    canvas.text(110, 655, "SOURCE: OUTPUTS/FINAL/PER_SAMPLE_ERRORS.CSV", (90, 96, 106), 2)
    canvas.save_png(path)
    return {"path": str(path), "type": "histogram", "models": selected_models, "x_axis_cap_meters": x_max}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/final")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_csv(output_dir / "model_metrics.csv")
    errors = read_csv(output_dir / "per_sample_errors.csv")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "figures": [
            make_bar_chart(metrics, figures_dir / "model_ade_bar.png"),
            make_distribution(errors, figures_dir / "error_distributions.png"),
        ],
        "notes": [
            "Figures are generated from final CSV artifacts.",
            "The standard-library renderer is intended for reproducibility; visual styling can be upgraded without changing source data.",
        ],
    }
    (figures_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Figures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
