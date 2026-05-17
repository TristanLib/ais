#!/usr/bin/env python3
"""Audit whether the conservative publication evidence pack is internally complete."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, evidence: str, severity: str = "error") -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence,
        }
    )


def row_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_conservative.yaml")
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--output", default="outputs/final/publication_readiness_report.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any error check fails.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    paper_dir = Path(args.paper_dir)

    config = read_json(config_path)
    environment = read_json(audit_dir / "environment.json")
    data_manifest = read_json(audit_dir / "data_manifest.json")
    run_manifest = read_json(output_dir / "run_manifest.json")
    stats = read_json(output_dir / "statistical_tests.json")
    reproducibility = read_json(output_dir / "reproducibility_check.json")
    model_rows = read_csv(output_dir / "model_metrics.csv")
    error_rows = read_csv(output_dir / "per_sample_errors.csv")
    horizon_rows = read_csv(output_dir / "error_summary_by_horizon.csv")
    group_rows = read_csv(output_dir / "error_summary_by_group.csv")

    checks: list[dict[str, Any]] = []
    required_files = [
        config_path,
        Path("requirements.txt"),
        audit_dir / "environment.json",
        audit_dir / "data_manifest.json",
        audit_dir / "feature_schema.json",
        audit_dir / "split_manifest.csv",
        audit_dir / "git_status.txt",
        audit_dir / "run_command.txt",
        output_dir / "run_manifest.json",
        output_dir / "model_metrics.csv",
        output_dir / "per_sample_errors.csv",
        output_dir / "error_summary_by_horizon.csv",
        output_dir / "error_summary_by_group.csv",
        output_dir / "statistical_tests.json",
        output_dir / "reproducibility_check.json",
        output_dir / "data_quality_report.json",
        output_dir / "figures" / "model_ade_bar.png",
        output_dir / "figures" / "error_distributions.png",
        output_dir / "figures" / "figure_manifest.json",
        output_dir / "tables" / "model_metrics.md",
        output_dir / "tables" / "model_metrics.tex",
        output_dir / "tables" / "artifact_checklist.md",
        paper_dir / "conservative_scope.md",
        paper_dir / "conservative_draft.md",
        paper_dir / "generated_results_summary.md",
        paper_dir / "conservative_manuscript.md",
    ]
    for path in required_files:
        add_check(checks, f"file exists: {path}", path.exists(), str(path))

    add_check(
        checks,
        "environment check passes",
        bool(environment.get("ok")),
        str(environment.get("missing_required_modules", [])),
    )

    processed_file = data_manifest.get("files", {}).get("processed_npz", {})
    raw_files = data_manifest.get("files", {}).get("raw_files", [])
    add_check(
        checks,
        "processed data checksum recorded",
        bool(processed_file.get("sha256")),
        str(processed_file.get("sha256")),
    )
    add_check(
        checks,
        "raw data checksum recorded",
        bool(raw_files) and all(file_record.get("sha256") for file_record in raw_files),
        str([file_record.get("sha256") for file_record in raw_files]),
    )
    add_check(
        checks,
        "raw data record and vessel counts are profiled",
        bool(raw_files)
        and all(
            file_record.get("profile", {}).get("record_count")
            and file_record.get("profile", {}).get("unique_mmsi_count")
            for file_record in raw_files
        ),
        str(
            [
                {
                    "records": file_record.get("profile", {}).get("record_count"),
                    "unique_mmsi": file_record.get("profile", {}).get("unique_mmsi_count"),
                }
                for file_record in raw_files
            ]
        ),
    )
    inferred_units = data_manifest.get("data_policy", {}).get("inferred_position_units", {}).get("inferred")
    configured_units = data_manifest.get("data_policy", {}).get("coordinate_units")
    add_check(
        checks,
        "coordinate units are inferred and configured consistently",
        inferred_units == configured_units == "degrees_latlon_wgs84",
        f"inferred={inferred_units}, configured={configured_units}",
    )
    add_check(
        checks,
        "data manifest warnings are empty",
        not data_manifest.get("warnings"),
        str(data_manifest.get("warnings")),
    )

    add_check(
        checks,
        "run is not a debug sample-cap run",
        run_manifest.get("is_debug_run") is False,
        f"is_debug_run={run_manifest.get('is_debug_run')}",
    )
    add_check(
        checks,
        "metric unit is verified Haversine meters",
        run_manifest.get("unit_status", {}).get("unit_verified") is True
        and run_manifest.get("unit_status", {}).get("metric_distance") == "haversine_meters",
        str(run_manifest.get("unit_status", {})),
    )
    if run_manifest.get("is_debug_run") is False:
        add_check(
            checks,
            "same-seed metric stability check passes",
            reproducibility.get("status") == "pass",
            str(reproducibility.get("status")),
        )

    final_models = config.get("models", {}).get("final_main", [])
    rows_by_model = {row.get("model"): row for row in model_rows}
    add_check(
        checks,
        "all configured final models appear in model_metrics.csv",
        set(final_models).issubset(rows_by_model),
        f"configured={final_models}, present={sorted(rows_by_model)}",
    )
    for model_name in final_models:
        row = rows_by_model.get(model_name, {})
        add_check(
            checks,
            f"{model_name} completed successfully",
            row.get("status") == "ok",
            str(row),
        )
        metric_values = [row_float(row, key) for key in ("ade_meters", "fde_meters", "rmse_meters", "mae_meters")]
        add_check(
            checks,
            f"{model_name} has positive finite metrics",
            all(value is not None and value >= 0 for value in metric_values),
            str(metric_values),
        )
        checkpoint = output_dir / "checkpoints" / (
            "linear_lstsq.npz" if model_name == "linear_lstsq" else f"{model_name}.pt"
        )
        if model_name == "constant_velocity":
            add_check(
                checks,
                f"{model_name} has training log",
                (output_dir / "training_logs" / f"{model_name}.json").exists(),
                str(output_dir / "training_logs" / f"{model_name}.json"),
            )
        else:
            add_check(checks, f"{model_name} checkpoint exists", checkpoint.exists(), str(checkpoint))

    expected_test_count = None
    if model_rows:
        test_counts = {row.get("model"): int(float(row.get("n_test", "0") or 0)) for row in model_rows}
        expected_test_count = max(test_counts.values()) if test_counts else None
    error_counts: dict[str, int] = {}
    for row in error_rows:
        error_counts[row.get("model", "")] = error_counts.get(row.get("model", ""), 0) + 1
    if expected_test_count is not None:
        for model_name in final_models:
            add_check(
                checks,
                f"{model_name} has per-sample errors for every test sample",
                error_counts.get(model_name) == expected_test_count,
                f"count={error_counts.get(model_name)}, expected={expected_test_count}",
            )

    horizon_count = int(config.get("experiment", {}).get("forecast_steps", 0))
    horizon_counts: dict[str, int] = {}
    for row in horizon_rows:
        horizon_counts[row.get("model", "")] = horizon_counts.get(row.get("model", ""), 0) + 1
    for model_name in final_models:
        add_check(
            checks,
            f"{model_name} has horizon metrics for every forecast step",
            horizon_counts.get(model_name) == horizon_count,
            f"count={horizon_counts.get(model_name)}, expected={horizon_count}",
        )

    group_models = {row.get("model") for row in group_rows}
    add_check(
        checks,
        "group summary exists for all final models",
        set(final_models).issubset(group_models),
        f"present={sorted(group_models)}",
    )
    add_check(
        checks,
        "statistical summary covers all final models",
        set(final_models).issubset(stats.get("models", {})),
        f"present={sorted(stats.get('models', {}))}",
    )
    expected_pairwise = [model for model in final_models if model != stats.get("reference")]
    add_check(
        checks,
        "paired tests cover non-reference final models",
        set(expected_pairwise).issubset(stats.get("pairwise_vs_reference", {})),
        f"expected={expected_pairwise}, present={sorted(stats.get('pairwise_vs_reference', {}))}",
    )

    generated_summary = paper_dir / "generated_results_summary.md"
    summary_text = generated_summary.read_text(encoding="utf-8") if generated_summary.exists() else ""
    add_check(
        checks,
        "generated paper summary has no debug warning",
        "debug sample caps" not in summary_text,
        "paper/generated_results_summary.md",
    )
    add_check(
        checks,
        "unsupported headline claims are not in generated summary",
        "9.4 m" not in summary_text and "23,000" not in summary_text,
        "paper/generated_results_summary.md",
    )
    manuscript = paper_dir / "conservative_manuscript.md"
    manuscript_text = manuscript.read_text(encoding="utf-8") if manuscript.exists() else ""
    draft = paper_dir / "conservative_draft.md"
    draft_text = draft.read_text(encoding="utf-8") if draft.exists() else ""
    add_check(
        checks,
        "generated manuscript has no TODO placeholders",
        "TODO" not in manuscript_text,
        "paper/conservative_manuscript.md",
    )
    add_check(
        checks,
        "unsupported headline claims are not in generated manuscript",
        "9.4 m" not in manuscript_text and "23,000" not in manuscript_text,
        "paper/conservative_manuscript.md",
    )
    add_check(
        checks,
        "conservative draft entry has no TODO placeholders",
        "TODO" not in draft_text,
        "paper/conservative_draft.md",
    )

    error_failures = [check for check in checks if not check["passed"] and check["severity"] == "error"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not error_failures else "fail",
        "error_count": len(error_failures),
        "checks": checks,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Publication readiness audit written to {output_path}: {report['status']}")
    if args.strict and error_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
