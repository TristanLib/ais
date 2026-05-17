#!/usr/bin/env python3
"""Evaluate AIS-derived risk-warning behavior from trajectory predictions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_SPEC = importlib.util.spec_from_file_location("final_train_eval_base", PROJECT_ROOT / "scripts" / "final_train_eval.py")
MULTI_SPEC = importlib.util.spec_from_file_location("final_train_eval_multiday", PROJECT_ROOT / "scripts" / "final_train_eval_multiday.py")
if FINAL_SPEC is None or FINAL_SPEC.loader is None or MULTI_SPEC is None or MULTI_SPEC.loader is None:
    raise RuntimeError("Could not load benchmark helpers")
FINAL = importlib.util.module_from_spec(FINAL_SPEC)
FINAL_SPEC.loader.exec_module(FINAL)
MULTI = importlib.util.module_from_spec(MULTI_SPEC)
MULTI_SPEC.loader.exec_module(MULTI)

EARTH_RADIUS_M = 6_371_008.8
NMI_M = 1852.0


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise RuntimeError(f"Could not parse {path}") from exc


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def split_key_from_policy(policy: str) -> str:
    if policy == "temporal_test":
        return "temporal_split"
    if policy == "vessel_disjoint_test":
        return "vessel_split"
    raise ValueError(f"Unknown risk split policy: {policy}")


def subset(data: dict[str, np.ndarray], split_key: str, split_value: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.where(data[split_key] == split_value)[0]
    if limit:
        idx = idx[:limit]
    return data["X"][idx], data["y"][idx], idx


def haversine_nmi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    meters = FINAL.haversine_meters(a, b)
    return meters / NMI_M


def first_warning_step(distances_nmi: np.ndarray, threshold_nmi: float) -> int | None:
    hits = np.where(distances_nmi <= threshold_nmi)[0]
    if hits.size == 0:
        return None
    return int(hits[0] + 1)


def origin_time(start_time: str, history_steps: int, dt_minutes: int) -> str:
    started = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    return (started + timedelta(minutes=(history_steps - 1) * dt_minutes)).isoformat()


def pair_distance_summary(path_a: np.ndarray, path_b: np.ndarray, threshold_nmi: float, step_minutes: int) -> dict[str, Any]:
    distances = haversine_nmi(path_a, path_b)
    min_pos = int(np.argmin(distances))
    first_step = first_warning_step(distances, threshold_nmi)
    return {
        "min_cpa_nmi": float(distances[min_pos]),
        "tcpa_minutes": float((min_pos + 1) * step_minutes),
        "warning": bool(first_step is not None),
        "first_warning_minutes": None if first_step is None else float(first_step * step_minutes),
    }


def build_scenario_pairs(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_idx: np.ndarray,
    data: dict[str, np.ndarray],
    history_steps: int,
    dt_minutes: int,
    search_radius_nmi: float,
    warning_threshold_nmi: float,
    step_minutes: int,
    max_scenarios: int,
) -> list[dict[str, Any]]:
    local_by_start: dict[str, list[int]] = {}
    for local_i, global_i in enumerate(sample_idx):
        local_by_start.setdefault(str(data["start_time"][global_i]), []).append(local_i)

    scenarios: list[dict[str, Any]] = []
    for start_time, members in sorted(local_by_start.items()):
        if len(members) < 2:
            continue
        for offset, local_a in enumerate(members[:-1]):
            origin_a = X_eval[local_a, -1, :2]
            for local_b in members[offset + 1 :]:
                origin_b = X_eval[local_b, -1, :2]
                origin_distance = float(haversine_nmi(origin_a.reshape(1, 2), origin_b.reshape(1, 2))[0])
                if origin_distance > search_radius_nmi:
                    continue
                truth_summary = pair_distance_summary(y_eval[local_a], y_eval[local_b], warning_threshold_nmi, step_minutes)
                global_a = int(sample_idx[local_a])
                global_b = int(sample_idx[local_b])
                scenarios.append(
                    {
                        "scenario_id": len(scenarios),
                        "local_a": local_a,
                        "local_b": local_b,
                        "sample_index_a": global_a,
                        "sample_index_b": global_b,
                        "mmsi_a": int(data["mmsi"][global_a]),
                        "mmsi_b": int(data["mmsi"][global_b]),
                        "start_time": start_time,
                        "origin_time": origin_time(start_time, history_steps, dt_minutes),
                        "region_a": str(data["region"][global_a]),
                        "region_b": str(data["region"][global_b]),
                        "origin_distance_nmi": origin_distance,
                        "truth_min_cpa_nmi": truth_summary["min_cpa_nmi"],
                        "truth_tcpa_minutes": truth_summary["tcpa_minutes"],
                        "truth_warning": truth_summary["warning"],
                        "truth_first_warning_minutes": truth_summary["first_warning_minutes"],
                    }
                )
                if len(scenarios) >= max_scenarios:
                    return scenarios
    return scenarios


def predict_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_shape: tuple[int, int],
) -> tuple[np.ndarray, float, float]:
    started = time.time()
    if model_name == "constant_velocity":
        return FINAL.constant_velocity_predict(X_eval, y_shape[0]), 0.0, time.time() - started
    if model_name == "constant_acceleration":
        return MULTI.constant_acceleration_predict(X_eval, y_shape[0]), 0.0, time.time() - started
    if model_name == "kalman_filter_cv":
        return MULTI.kalman_filter_cv_predict(X_eval, y_shape[0]), 0.0, time.time() - started
    if model_name == "linear_lstsq":
        train_start = time.time()
        model = FINAL.fit_linear_lstsq(X_train, y_train)
        train_seconds = time.time() - train_start
        pred = FINAL.predict_linear_lstsq(model, X_eval, y_shape)
        return pred, train_seconds, time.time() - started - train_seconds
    raise ValueError(f"Risk-warning runner does not support model: {model_name}")


def classify_warning(truth_warning: bool, pred_warning: bool) -> str:
    if truth_warning and pred_warning:
        return "true_positive"
    if truth_warning and not pred_warning:
        return "false_negative"
    if not truth_warning and pred_warning:
        return "false_positive"
    return "true_negative"


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    summary: dict[str, Any] = {}
    for model, model_rows in sorted(by_model.items()):
        counts = {name: sum(1 for row in model_rows if row["classification"] == name) for name in ["true_positive", "false_positive", "false_negative", "true_negative"]}
        tp = counts["true_positive"]
        fp = counts["false_positive"]
        fn = counts["false_negative"]
        tn = counts["true_negative"]
        lead_errors = [abs(float(row["lead_time_error_minutes"])) for row in model_rows if row["lead_time_error_minutes"] != ""]
        cpa_errors = [abs(float(row["pred_min_cpa_nmi"]) - float(row["truth_min_cpa_nmi"])) for row in model_rows]
        summary[model] = {
            "n_scenarios": len(model_rows),
            **counts,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn),
            "false_alarm_rate": None if fp + tn == 0 else fp / (fp + tn),
            "missed_warning_rate": None if tp + fn == 0 else fn / (tp + fn),
            "mean_abs_lead_time_error_minutes": None if not lead_errors else float(np.mean(lead_errors)),
            "mean_abs_cpa_error_nmi": None if not cpa_errors else float(np.mean(cpa_errors)),
        }
    return summary


def make_case_study_figure(
    output_path: Path,
    scenarios: list[dict[str, Any]],
    predictions: dict[str, np.ndarray],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    if not scenarios:
        ax.text(0.5, 0.5, "No AIS-derived encounter scenarios found", ha="center", va="center")
        ax.set_axis_off()
    else:
        scenario = next((row for row in scenarios if row["truth_warning"]), scenarios[0])
        local_a = int(scenario["local_a"])
        local_b = int(scenario["local_b"])
        ax.plot(X_eval[local_a, :, 1], X_eval[local_a, :, 0], color="#2b6cb0", alpha=0.35, label="A history")
        ax.plot(X_eval[local_b, :, 1], X_eval[local_b, :, 0], color="#c05621", alpha=0.35, label="B history")
        ax.plot(y_eval[local_a, :, 1], y_eval[local_a, :, 0], color="#2b6cb0", linewidth=2, label="A true future")
        ax.plot(y_eval[local_b, :, 1], y_eval[local_b, :, 0], color="#c05621", linewidth=2, label="B true future")
        for model, pred in list(predictions.items())[:2]:
            ax.plot(pred[local_a, :, 1], pred[local_a, :, 0], linestyle="--", linewidth=1.5, label=f"A {model}")
            ax.plot(pred[local_b, :, 1], pred[local_b, :, 0], linestyle=":", linewidth=1.5, label=f"B {model}")
        ax.set_title(f"Risk-warning case study {scenario['scenario_id']}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_multiday.yaml")
    parser.add_argument("--split-policy")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-scenarios", type=int)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    risk_config = config.get("risk_warning", {})
    output_dir = Path(risk_config.get("output_dir", "outputs/final_risk"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_npz(Path(config["data"]["processed_npz"]))
    split_policy = args.split_policy or risk_config.get("split_policy", "temporal_test")
    split_key = split_key_from_policy(split_policy)
    max_eval_samples = args.max_eval_samples or int(risk_config.get("max_eval_samples", 5000))
    max_scenarios = args.max_scenarios or int(risk_config.get("max_scenarios", 2000))
    models = args.models or risk_config.get("models", ["constant_velocity", "kalman_filter_cv", "linear_lstsq"])
    X_train, y_train, _ = subset(data, split_key, "train", None)
    X_eval, y_eval, eval_idx = subset(data, split_key, "test", max_eval_samples)

    history_steps = int(config["experiment"]["history_steps"])
    dt_minutes = int(config["experiment"].get("dt_minutes", 1))
    step_minutes = int(risk_config.get("forecast_step_minutes", dt_minutes))
    warning_threshold_nmi = float(risk_config.get("warning_threshold_nmi", 0.5))
    search_radius_nmi = float(risk_config.get("search_radius_nmi", 3.0))

    scenarios = build_scenario_pairs(
        X_eval,
        y_eval,
        eval_idx,
        data,
        history_steps,
        dt_minutes,
        search_radius_nmi,
        warning_threshold_nmi,
        step_minutes,
        max_scenarios,
    )
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    model_runtime: dict[str, Any] = {}
    for model in models:
        pred, train_seconds, inference_seconds = predict_model(model, X_train, y_train, X_eval, y_eval.shape[1:])
        predictions[model] = pred
        model_runtime[model] = {"train_seconds": train_seconds, "inference_seconds": inference_seconds}
        for scenario in scenarios:
            local_a = int(scenario["local_a"])
            local_b = int(scenario["local_b"])
            pred_summary = pair_distance_summary(pred[local_a], pred[local_b], warning_threshold_nmi, step_minutes)
            lead_time_error = ""
            if scenario["truth_first_warning_minutes"] is not None and pred_summary["first_warning_minutes"] is not None:
                lead_time_error = float(pred_summary["first_warning_minutes"] - scenario["truth_first_warning_minutes"])
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "model": model,
                    "split_policy": split_policy,
                    "sample_index_a": scenario["sample_index_a"],
                    "sample_index_b": scenario["sample_index_b"],
                    "mmsi_a": scenario["mmsi_a"],
                    "mmsi_b": scenario["mmsi_b"],
                    "origin_time": scenario["origin_time"],
                    "region_a": scenario["region_a"],
                    "region_b": scenario["region_b"],
                    "origin_distance_nmi": scenario["origin_distance_nmi"],
                    "truth_min_cpa_nmi": scenario["truth_min_cpa_nmi"],
                    "truth_tcpa_minutes": scenario["truth_tcpa_minutes"],
                    "truth_warning": scenario["truth_warning"],
                    "pred_min_cpa_nmi": pred_summary["min_cpa_nmi"],
                    "pred_tcpa_minutes": pred_summary["tcpa_minutes"],
                    "pred_warning": pred_summary["warning"],
                    "lead_time_error_minutes": lead_time_error,
                    "classification": classify_warning(bool(scenario["truth_warning"]), bool(pred_summary["warning"])),
                }
            )

    fieldnames = [
        "scenario_id",
        "model",
        "split_policy",
        "sample_index_a",
        "sample_index_b",
        "mmsi_a",
        "mmsi_b",
        "origin_time",
        "region_a",
        "region_b",
        "origin_distance_nmi",
        "truth_min_cpa_nmi",
        "truth_tcpa_minutes",
        "truth_warning",
        "pred_min_cpa_nmi",
        "pred_tcpa_minutes",
        "pred_warning",
        "lead_time_error_minutes",
        "classification",
    ]
    write_csv(output_dir / "risk_scenarios.csv", rows, fieldnames)
    make_case_study_figure(output_dir / "figures" / "risk_case_studies.png", scenarios, predictions, X_eval, y_eval)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_policy": split_policy,
        "models": models,
        "scenario_generation": {
            "max_eval_samples": max_eval_samples,
            "evaluated_samples": int(X_eval.shape[0]),
            "scenario_count": len(scenarios),
            "search_radius_nmi": search_radius_nmi,
            "warning_threshold_nmi": warning_threshold_nmi,
            "truth_warning_count": sum(1 for scenario in scenarios if scenario["truth_warning"]),
        },
        "model_runtime": model_runtime,
        "metrics_by_model": summarize_metrics(rows),
        "notes": [
            "Risk labels are AIS-derived from observed future trajectory separation over the forecast horizon.",
            "This is a risk-warning decision-support evaluation, not a validated autonomous collision-avoidance simulation.",
        ],
        "outputs": {
            "risk_scenarios": str(output_dir / "risk_scenarios.csv"),
            "risk_case_studies": str(output_dir / "figures" / "risk_case_studies.png"),
        },
    }
    (output_dir / "risk_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Risk-warning metrics written to {output_dir / 'risk_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
