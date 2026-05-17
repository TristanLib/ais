#!/usr/bin/env python3
"""Audit high-quality-journal roadmap artifacts and remaining gaps."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception:
            return {}


def file_status(path: str) -> dict[str, Any]:
    p = Path(path)
    return {"path": path, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}


def phase_record(name: str, deliverables: list[str], checks: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [path for path in deliverables if not Path(path).exists()]
    failed_checks = [check for check in checks if not check["passed"]]
    if missing or failed_checks:
        status = "partial" if len(missing) < len(deliverables) else "missing"
    else:
        status = "pass"
    return {
        "phase": name,
        "status": status,
        "deliverables": [file_status(path) for path in deliverables],
        "checks": checks,
        "missing": missing,
        "failed_checks": failed_checks,
    }


def main() -> int:
    output_dir = Path("outputs/final_submission")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = read_json(Path("outputs/audit/multiday_data_manifest.json"))
    run_manifest = read_json(Path("outputs/final_multiday/run_manifest.json"))
    risk_metrics = read_json(Path("outputs/final_risk/risk_metrics.json"))
    latest_manifest = read_json(Path("outputs/latest_predictions/prediction_manifest.json"))
    submission_manifest = read_json(Path("outputs/final_submission/submission_manifest.json"))
    config = read_config(Path("configs/experiment_multiday.yaml"))
    model_rows = read_csv_rows(Path("outputs/final_multiday/model_metrics.csv"))
    tuning_rows = read_csv_rows(Path("outputs/final_multiday/neural_tuning_results.csv"))
    tuning_protocol = read_json(Path("outputs/final_multiday/neural_tuning_protocol.json"))
    split_policies = sorted({row.get("split_policy", "") for row in model_rows if row.get("split_policy")})
    model_status = {f"{row.get('split_policy')}::{row.get('model')}": row.get("status") for row in model_rows}
    expected_models = set(config.get("models", {}).get("final_main", []))
    expected_split_policies = {"temporal_test", "vessel_disjoint_test"}
    observed_ok = {
        (row.get("split_policy"), row.get("model"))
        for row in model_rows
        if row.get("status") == "ok"
    }
    missing_model_runs = sorted(
        f"{split}::{model}"
        for split in expected_split_policies
        for model in expected_models
        if (split, model) not in observed_ok
    )
    expected_neural_models = {
        model for model in expected_models if model in {"lstm_baseline", "gru_baseline", "transformer_baseline", "tcn_baseline"}
    }
    tuned_ok_models = {row.get("model") for row in tuning_rows if row.get("status") == "ok"}
    missing_tuning_models = sorted(model for model in expected_neural_models if model not in tuned_ok_models)
    dataset_summary = data_manifest.get("dataset_summary", {})
    data_command_options = data_manifest.get("command_options", {})
    raw_profiles = data_manifest.get("raw_files", [])
    source_dates = dataset_summary.get("source_dates", [])
    sampling_strategy = data_command_options.get("sampling_strategy", "none")
    sample_rows_per_file = data_command_options.get("sample_rows_per_file")
    sample_time_blocks = data_command_options.get("sample_time_blocks")
    raw_checksums_computed = bool(raw_profiles) and all(row.get("sha256") for row in raw_profiles)
    final_grade_sampling = (
        data_command_options.get("max_rows_per_file") is None
        and (
            sample_rows_per_file is None
            or sampling_strategy in {"systematic_raw_row", "stratified_time_blocks"}
        )
        and (
            not sample_time_blocks
            or sampling_strategy == "stratified_time_blocks"
        )
    )
    phase_a = phase_record(
        "Phase A: data protocol",
        [
            "configs/experiment_multiday.yaml",
            "scripts/build_multiday_dataset.py",
            "outputs/audit/multiday_data_manifest.json",
            "outputs/audit/split_policy.json",
            "data/processed/multiday_high_quality_processed.npz",
        ],
        [
            {"name": "metadata-rich sample count > 0", "passed": dataset_summary.get("sample_count", 0) > 0, "value": dataset_summary.get("sample_count")},
            {"name": "temporal split exists", "passed": "temporal_split_counts" in dataset_summary, "value": dataset_summary.get("temporal_split_counts")},
            {"name": "vessel-disjoint split exists", "passed": "vessel_split_counts" in dataset_summary, "value": dataset_summary.get("vessel_split_counts")},
            {"name": "at least four source dates for high-quality target", "passed": len(source_dates) >= 4, "value": source_dates},
            {
                "name": "dataset uses uncapped or documented systematic sampling protocol",
                "passed": final_grade_sampling,
                "value": {
                    "max_rows_per_file": data_command_options.get("max_rows_per_file"),
                    "sample_rows_per_file": sample_rows_per_file,
                    "sample_time_blocks": sample_time_blocks,
                    "sampling_strategy": sampling_strategy,
                },
            },
            {
                "name": "raw checksums computed",
                "passed": raw_checksums_computed,
                "value": [row.get("checksum_status") for row in raw_profiles],
            },
        ],
    )
    phase_b = phase_record(
        "Phase B: expanded benchmark",
        [
            "scripts/final_train_eval_multiday.py",
            "src/models/gru.py",
            "src/models/tcn.py",
            "outputs/final_multiday/model_metrics.csv",
            "outputs/final_multiday/per_sample_errors.csv",
            "outputs/final_multiday/neural_tuning_protocol.json",
            "outputs/final_multiday/neural_tuning_results.csv",
        ],
        [
            {"name": "all configured smoke-run models completed", "passed": bool(model_rows) and all(row.get("status") == "ok" for row in model_rows), "value": model_status},
            {"name": "all configured models present on both final split policies", "passed": not missing_model_runs, "value": missing_model_runs},
            {"name": "debug cap is not final paper run", "passed": not run_manifest.get("is_debug_run", True), "value": run_manifest.get("is_debug_run")},
            {
                "name": "neural models have documented validation-set tuning protocol",
                "passed": bool(tuning_protocol) and not missing_tuning_models,
                "value": {
                    "protocol": tuning_protocol.get("protocol"),
                    "selection_metric": tuning_protocol.get("selection_metric"),
                    "missing_models": missing_tuning_models,
                },
            },
        ],
    )
    phase_c = phase_record(
        "Phase C: generalization and error analysis",
        [
            "outputs/final_multiday/generalization_metrics.csv",
            "outputs/final_multiday/error_summary_by_horizon.csv",
            "outputs/final_multiday/error_summary_by_group.csv",
            "outputs/final_multiday/statistical_tests.json",
        ],
        [
            {"name": "temporal and vessel-disjoint metrics exist", "passed": {"temporal_test", "vessel_disjoint_test"}.issubset(set(split_policies)), "value": split_policies},
        ],
    )
    phase_d = phase_record(
        "Phase D: risk-warning demonstration",
        [
            "scripts/evaluate_risk_warning.py",
            "outputs/final_risk/risk_scenarios.csv",
            "outputs/final_risk/risk_metrics.json",
            "outputs/final_risk/figures/risk_case_studies.png",
            "scripts/predict_latest_ais.py",
            "outputs/latest_predictions/prediction_manifest.json",
        ],
        [
            {"name": "AIS-derived risk scenarios exist", "passed": risk_metrics.get("scenario_generation", {}).get("scenario_count", 0) > 0, "value": risk_metrics.get("scenario_generation")},
            {"name": "latest/offline prediction export ran", "passed": latest_manifest.get("sample_count", 0) > 0, "value": latest_manifest.get("sample_count")},
        ],
    )
    phase_e = phase_record(
        "Phase E: submission package",
        [
            "paper/submission_manuscript.md",
            "paper/references.bib",
            "paper/tables/high_quality_model_metrics.md",
            "paper/tables/risk_warning_metrics.md",
            "outputs/final_submission/submission_manifest.json",
        ],
        [
            {"name": "venue-specific submission manuscript exists", "passed": Path("paper/submission_manuscript.md").exists(), "value": None},
            {"name": "bibliography exists", "passed": Path("paper/references.bib").exists(), "value": None},
            {
                "name": "submission manifest points to current manuscript",
                "passed": submission_manifest.get("manuscript") == "paper/submission_manuscript.md",
                "value": submission_manifest.get("manuscript"),
            },
            {
                "name": "submission pack generated from current debug/final state",
                "passed": "is_debug_run" in submission_manifest,
                "value": submission_manifest.get("is_debug_run"),
            },
        ],
    )
    phases = [phase_a, phase_b, phase_c, phase_d, phase_e]
    blocking_gaps = []
    if len(source_dates) < 4:
        blocking_gaps.append("Dataset is still not a true 4-day/4-week high-quality-journal dataset.")
    if not final_grade_sampling:
        blocking_gaps.append("Dataset was built with max_rows_per_file; final submission needs an uncapped or defensible documented sampling protocol.")
    if not raw_checksums_computed:
        blocking_gaps.append("Raw-file checksums were skipped; final submission needs computed raw checksums.")
    if run_manifest.get("is_debug_run", True):
        blocking_gaps.append("Current multiday benchmark metrics are a capped smoke run, not final paper metrics.")
    if missing_model_runs:
        blocking_gaps.append("Not all configured benchmark models have completed final metrics on both split policies.")
    if missing_tuning_models:
        blocking_gaps.append("Neural baseline tuning protocol is missing completed validation runs for: " + ", ".join(missing_tuning_models))
    if not Path("paper/submission_manuscript.md").exists():
        blocking_gaps.append("Submission manuscript has not been generated from the high-quality artifacts.")
    if not Path("paper/references.bib").exists():
        blocking_gaps.append("Verified bibliography is missing.")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "not_submission_ready" if blocking_gaps else "submission_ready_candidate",
        "phases": phases,
        "blocking_gaps": blocking_gaps,
        "current_claim_boundary": (
            "The repository now supports an artifact-complete high-quality-journal candidate package when this report has "
            "no blocking gaps. This is not an acceptance guarantee: broader all-day traffic claims, architecture-superiority "
            "claims, or autonomous collision-avoidance claims still require additional evidence beyond the current artifacts."
        ),
    }
    (output_dir / "readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"High-quality readiness report written to {output_dir / 'readiness_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
