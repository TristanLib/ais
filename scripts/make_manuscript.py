#!/usr/bin/env python3
"""Render a conservative manuscript draft from audited repository artifacts."""

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


def metric(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def find_row(rows: list[dict[str, str]], model_name: str) -> dict[str, str]:
    for row in rows:
        if row.get("model") == model_name:
            return row
    return {}


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Model | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Test samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        lines.append(
            "| {model} | {ade} | {fde} | {rmse} | {mae} | {n_test} |".format(
                model=row.get("model"),
                ade=fmt(metric(row, "ade_meters")),
                fde=fmt(metric(row, "fde_meters")),
                rmse=fmt(metric(row, "rmse_meters")),
                mae=fmt(metric(row, "mae_meters")),
                n_test=row.get("n_test"),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--output", default="paper/conservative_manuscript.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    paper_dir = Path(args.paper_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_rows = read_csv(output_dir / "model_metrics.csv")
    stats = read_json(output_dir / "statistical_tests.json")
    data_manifest = read_json(audit_dir / "data_manifest.json")
    env = read_json(audit_dir / "environment.json")
    run_manifest = read_json(output_dir / "run_manifest.json")

    cv = find_row(metrics_rows, "constant_velocity")
    linear = find_row(metrics_rows, "linear_lstsq")
    lstm = find_row(metrics_rows, "lstm_baseline")
    transformer = find_row(metrics_rows, "transformer_baseline")

    split_counts = data_manifest.get("split_counts", {})
    data_policy = data_manifest.get("data_policy", {})
    processed = data_manifest.get("files", {}).get("processed_npz", {})
    raw_files = data_manifest.get("files", {}).get("raw_files", [])
    raw_file = raw_files[0] if raw_files else {}
    raw_profile = raw_file.get("profile", {})
    python_info = env.get("python", {})
    modules = {record.get("module"): record.get("version") for record in env.get("modules", [])}

    lines = [
        "# A Reproducible Evaluation of Simple and Deep Learning Baselines for Short-Term Vessel Trajectory Prediction from AIS Data",
        "",
        f"Generated from repository artifacts at {datetime.now(timezone.utc).isoformat()}.",
        "",
        "## Abstract",
        "",
        (
            "Short-term vessel trajectory prediction is often presented as a natural application for deep sequence "
            "models, yet operational maritime datasets can make simple kinematic baselines difficult to beat. "
            f"This study evaluates four baselines on an audited AIS trajectory dataset with {split_counts.get('train')} "
            f"training samples, {split_counts.get('val')} validation samples, and {split_counts.get('test')} test samples. "
            f"Using a {run_manifest.get('config', {}).get('experiment', {}).get('history_steps')}-minute input window and "
            f"a {run_manifest.get('config', {}).get('experiment', {}).get('forecast_steps')}-minute forecast horizon, "
            f"the constant-velocity baseline achieved {fmt(metric(cv, 'ade_meters'))} m ADE and "
            f"{fmt(metric(cv, 'fde_meters'))} m FDE. Linear least squares reached "
            f"{fmt(metric(linear, 'ade_meters'))} m ADE, while the deliberately conservative LSTM and Transformer "
            f"baselines produced {fmt(metric(lstm, 'ade_meters'))} m and "
            f"{fmt(metric(transformer, 'ade_meters'))} m ADE, respectively. "
            "The results support a conservative conclusion: before claiming architectural superiority, AIS prediction "
            "studies should archive preprocessing, metric definitions, model failures, and per-sample error evidence."
        ),
        "",
        "## 1. Introduction",
        "",
        (
            "AIS data provide frequent position, speed, and heading observations for vessels, making them attractive for "
            "short-horizon forecasting and downstream navigational decision support. The practical challenge is not only "
            "to train a model, but to prove that the model comparison is reproducible and that reported meter-level "
            "errors are computed from consistent geographic units. This manuscript therefore narrows the project to a "
            "benchmark-style claim: under the audited current protocol, simple kinematic models are strong baselines, "
            "and naive deep learning runs can fail badly enough that they should be reported as evidence rather than hidden."
        ),
        "",
        "## 2. Data and Reproducible Pipeline",
        "",
        (
            f"The processed data file is `{processed.get('path')}` with SHA256 `{processed.get('sha256')}`. "
            f"The raw AIS file is `{raw_file.get('path')}` with SHA256 `{raw_file.get('sha256')}`. "
            f"The raw profile contains {raw_profile.get('record_count')} AIS records from "
            f"{raw_profile.get('unique_mmsi_count')} MMSI values, spanning "
            f"{raw_profile.get('timestamp_min')} to {raw_profile.get('timestamp_max')}. "
            f"The saved split protocol is `{data_policy.get('split_protocol')}` with "
            f"{split_counts.get('train')} training, {split_counts.get('val')} validation, and "
            f"{split_counts.get('test')} test samples. The processed NPZ does not contain MMSI or timestamp metadata, "
            "so this paper does not claim vessel-disjoint splits, temporal cross-validation, regional generalization, "
            "or group-specific performance beyond the all-sample summary."
        ),
        "",
        (
            f"The first two position dimensions were audited as `{data_policy.get('coordinate_units')}`. "
            f"The inferred coordinate range is latitude "
            f"{data_policy.get('inferred_position_units', {}).get('lat_min')} to "
            f"{data_policy.get('inferred_position_units', {}).get('lat_max')} and longitude "
            f"{data_policy.get('inferred_position_units', {}).get('lon_min')} to "
            f"{data_policy.get('inferred_position_units', {}).get('lon_max')}. "
            f"The final metric distance is `{data_policy.get('metric_distance')}`."
        ),
        "",
        "## 3. Models",
        "",
        (
            "The main comparison includes constant velocity, linear least squares, an LSTM baseline, and a Transformer "
            "baseline. The LSTM and Transformer are intentionally treated as conservative baselines rather than optimized "
            "architectures. Their role is to test whether the current repository can regenerate and archive deep-learning "
            "behavior, including poor performance, under a single auditable command."
        ),
        "",
        "## 4. Metrics and Statistics",
        "",
        (
            "ADE and FDE are Haversine displacement errors in meters between predicted and target WGS84 latitude/longitude "
            "positions. RMSE and MAE are computed from local north/east component errors in meters. The statistical report "
            "uses aligned per-sample errors and paired tests against the constant-velocity reference, with Bonferroni "
            "correction metadata stored in `outputs/final/statistical_tests.json`."
        ),
        "",
        "## 5. Results",
        "",
        markdown_table(metrics_rows),
        "",
        (
            f"The constant-velocity model is the strongest model by ADE in the current final evidence pack. "
            f"Linear least squares is worse by {fmt(stats.get('pairwise_vs_reference', {}).get('linear_lstsq', {}).get('mean_difference_b_minus_a', 0.0))} m mean ADE. "
            f"The LSTM and Transformer baselines are much worse than the kinematic baselines, with mean ADE differences "
            f"of {fmt(stats.get('pairwise_vs_reference', {}).get('lstm_baseline', {}).get('mean_difference_b_minus_a', 0.0))} m and "
            f"{fmt(stats.get('pairwise_vs_reference', {}).get('transformer_baseline', {}).get('mean_difference_b_minus_a', 0.0))} m, respectively."
        ),
        "",
        "Generated figures:",
        "",
        "- `outputs/final/figures/model_ade_bar.png`",
        "- `outputs/final/figures/error_distributions.png`",
        "",
        "## 6. Discussion",
        "",
        (
            "The current evidence supports a reproducibility-centered interpretation. Constant velocity remains highly "
            "competitive for short-horizon AIS forecasting, while the neural baselines demonstrate that model complexity "
            "alone is not evidence of better navigational prediction. Because failures are archived with logs and checkpoints, "
            "the repository now supports a publishable negative or cautionary result instead of an unsupported architecture-win story."
        ),
        "",
        "## 7. Limitations",
        "",
        "- The processed NPZ lacks MMSI, timestamps, vessel type, and region fields.",
        "- The paper does not claim temporal cross-validation, vessel-disjoint generalization, or regional robustness.",
        "- GNN, STT, PINN, and recovery experiments are excluded from the main ranking until full-data evidence is regenerated.",
        "- CPA/TCPA and collision-avoidance modules are treated as future downstream work, not as proven operational avoidance.",
        "- The LSTM and Transformer settings are conservative baselines, not tuned state-of-the-art architectures.",
        "",
        "## 8. Reproducibility Statement",
        "",
        "The final evidence pack was generated with:",
        "",
        "```bash",
        "PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh",
        "```",
        "",
        f"The run is marked `is_debug_run={run_manifest.get('is_debug_run')}`. "
        f"The Python executable was `{python_info.get('executable')}` running Python {python_info.get('version')}. "
        f"Key package versions include numpy {modules.get('numpy')}, scipy {modules.get('scipy')}, "
        f"pandas {modules.get('pandas')}, scikit-learn {modules.get('sklearn')}, matplotlib {modules.get('matplotlib')}, "
        f"PyYAML {modules.get('yaml')}, and torch {modules.get('torch')}.",
        "",
        "The authoritative artifact map is `outputs/final/publication_readiness_report.json`.",
        "",
        "## 9. Conclusion",
        "",
        (
            "The conservative evidence pack shows that the current project can support a reproducible AIS trajectory "
            "prediction benchmark paper. The strongest supported claim is not that deep learning wins, but that auditable "
            "preprocessing, metric definitions, and failure reporting are essential before maritime trajectory-prediction "
            "models are used to support stronger navigational or collision-avoidance claims."
        ),
        "",
        "## References",
        "",
        "1. NOAA Office for Coastal Management and BOEM, MarineCadastre.gov AIS data: https://marinecadastre.gov/ais/",
        "2. NOAA Digital Coast Marine Cadastre overview: https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html",
        "3. PyTorch project: https://pytorch.org/projects/pytorch/",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Manuscript written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
