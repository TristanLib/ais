#!/usr/bin/env python3
"""Generate manuscript-ready tables and summary text from final artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_number(value: str) -> str:
    if value in {"", None}:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != 0 and abs(number) < 0.001:
        return f"{number:.2e}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.3f}"


def markdown_metrics_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Model | Status | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | n test | Notes |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {model} | {status} | {ade} | {fde} | {rmse} | {mae} | {n_test} | {notes} |".format(
                model=row.get("model", ""),
                status=row.get("status", ""),
                ade=fmt_number(row.get("ade_meters", "")),
                fde=fmt_number(row.get("fde_meters", "")),
                rmse=fmt_number(row.get("rmse_meters", "")),
                mae=fmt_number(row.get("mae_meters", "")),
                n_test=row.get("n_test", ""),
                notes=(row.get("notes", "") or "").replace("|", "/"),
            )
        )
    return header + "\n".join(body) + "\n"


def latex_metrics_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "Model & ADE (m) & FDE (m) & RMSE (m) & MAE (m) & $n$ \\\\",
        "\\hline",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        model = row.get("model", "").replace("_", "\\_")
        lines.append(
            f"{model} & {fmt_number(row.get('ade_meters', ''))} & "
            f"{fmt_number(row.get('fde_meters', ''))} & "
            f"{fmt_number(row.get('rmse_meters', ''))} & "
            f"{fmt_number(row.get('mae_meters', ''))} & {row.get('n_test', '')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def stats_summary(stats: dict[str, Any]) -> str:
    if not stats:
        return "Statistical summary is not available yet.\n"

    lines = ["## Statistical Summary", ""]
    for model, record in stats.get("models", {}).items():
        ci = record.get("ade_ci_percentile", ["NA", "NA"])
        lines.append(
            "- {model}: ADE mean {mean} m, median {median} m, 95% percentile interval [{lo}, {hi}] m.".format(
                model=model,
                mean=fmt_number(str(record.get("ade_mean", ""))),
                median=fmt_number(str(record.get("ade_median", ""))),
                lo=fmt_number(str(ci[0])),
                hi=fmt_number(str(ci[1])),
            )
        )
    for model, record in stats.get("pairwise_vs_reference", {}).items():
        paired_t = record.get("paired_t") or {}
        p_value = paired_t.get("p_value")
        if p_value == 0.0:
            p_text = "<1e-300"
        else:
            p_text = fmt_number(str(p_value)) if p_value is not None else "NA"
        lines.append(
            f"- Versus {stats.get('reference')}, {model} mean ADE difference "
            f"{fmt_number(str(record.get('mean_difference_b_minus_a', '')))} m; paired t-test p={p_text}."
        )
    lines.append("")
    return "\n".join(lines)


def artifact_checklist(paths: list[Path]) -> str:
    lines = [
        "| Artifact | Exists |",
        "|---|---:|",
    ]
    for path in paths:
        lines.append(f"| `{path}` | {'yes' if path.exists() else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--paper-dir", default="paper")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    paper_dir = Path(args.paper_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "model_metrics.csv"
    stats_path = output_dir / "statistical_tests.json"
    run_manifest_path = output_dir / "run_manifest.json"
    data_manifest_path = audit_dir / "data_manifest.json"

    metrics_rows = read_csv(metrics_path)
    stats = read_json(stats_path)
    run_manifest = read_json(run_manifest_path)
    data_manifest = read_json(data_manifest_path)

    metrics_md = markdown_metrics_table(metrics_rows)
    (tables_dir / "model_metrics.md").write_text(metrics_md, encoding="utf-8")
    (tables_dir / "model_metrics.tex").write_text(latex_metrics_table(metrics_rows), encoding="utf-8")

    required_paths = [
        metrics_path,
        output_dir / "per_sample_errors.csv",
        output_dir / "error_summary_by_horizon.csv",
        output_dir / "error_summary_by_group.csv",
        stats_path,
        output_dir / "reproducibility_check.json",
        output_dir / "data_quality_report.json",
        output_dir / "figures" / "model_ade_bar.png",
        output_dir / "figures" / "error_distributions.png",
        output_dir / "figures" / "figure_manifest.json",
        data_manifest_path,
        audit_dir / "feature_schema.json",
        audit_dir / "split_manifest.csv",
        audit_dir / "environment.json",
        audit_dir / "git_status.txt",
        audit_dir / "run_command.txt",
    ]
    checklist = artifact_checklist(required_paths)
    (tables_dir / "artifact_checklist.md").write_text(checklist, encoding="utf-8")

    warning_lines = []
    if run_manifest.get("is_debug_run"):
        warning_lines.append(
            "This artifact pack was generated with debug sample caps and must not be submitted as final paper evidence."
        )
    warning_lines.extend(data_manifest.get("warnings", []))

    summary = [
        "# Generated Results Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Result Table",
        "",
        metrics_md,
        stats_summary(stats),
        "## Artifact Checklist",
        "",
        checklist,
    ]
    if warning_lines:
        summary.extend(["## Warnings", ""])
        summary.extend([f"- {warning}" for warning in warning_lines])
        summary.append("")

    summary.extend(
        [
            "## Manuscript Use Rule",
            "",
            "Only cite numbers from the table above when the run manifest shows `is_debug_run=false` and all required artifacts exist.",
            "",
        ]
    )
    (paper_dir / "generated_results_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Paper assets written to {tables_dir} and {paper_dir / 'generated_results_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
