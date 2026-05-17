#!/usr/bin/env python3
"""Run offline trajectory prediction and risk-warning export for a new AIS CSV."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BUILD_SPEC = importlib.util.spec_from_file_location("build_multiday_dataset", PROJECT_ROOT / "scripts" / "build_multiday_dataset.py")
FINAL_SPEC = importlib.util.spec_from_file_location("final_train_eval_base", PROJECT_ROOT / "scripts" / "final_train_eval.py")
MULTI_SPEC = importlib.util.spec_from_file_location("final_train_eval_multiday", PROJECT_ROOT / "scripts" / "final_train_eval_multiday.py")
if (
    BUILD_SPEC is None
    or BUILD_SPEC.loader is None
    or FINAL_SPEC is None
    or FINAL_SPEC.loader is None
    or MULTI_SPEC is None
    or MULTI_SPEC.loader is None
):
    raise RuntimeError("Could not load pipeline helpers")
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD)
FINAL = importlib.util.module_from_spec(FINAL_SPEC)
FINAL_SPEC.loader.exec_module(FINAL)
MULTI = importlib.util.module_from_spec(MULTI_SPEC)
MULTI_SPEC.loader.exec_module(MULTI)

NMI_M = 1852.0


def load_config(path: Path) -> dict[str, Any]:
    return BUILD.load_config(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def origin_time(start_time: str, history_steps: int, dt_minutes: int) -> str:
    started = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    return (started + timedelta(minutes=(history_steps - 1) * dt_minutes)).isoformat()


def haversine_nmi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return FINAL.haversine_meters(a, b) / NMI_M


def sample_arrays(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([sample["X"] for sample in samples]), np.stack([sample["y"] for sample in samples])


def load_linear_model(config: dict[str, Any], checkpoint: Path | None) -> dict[str, np.ndarray] | None:
    if checkpoint and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    processed = Path(config["data"]["processed_npz"])
    if not processed.exists():
        return None
    with np.load(processed, allow_pickle=True) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]
    return FINAL.fit_linear_lstsq(X_train, y_train)


def predict_models(
    models: list[str],
    X: np.ndarray,
    forecast_steps: int,
    y_shape: tuple[int, int],
    linear_model: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    predictions: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for model in models:
        if model == "constant_velocity":
            predictions[model] = FINAL.constant_velocity_predict(X, forecast_steps)
        elif model == "constant_acceleration":
            predictions[model] = MULTI.constant_acceleration_predict(X, forecast_steps)
        elif model == "kalman_filter_cv":
            predictions[model] = MULTI.kalman_filter_cv_predict(X, forecast_steps)
        elif model == "linear_lstsq":
            if linear_model is None:
                skipped[model] = "No linear checkpoint or processed training data is available."
            else:
                predictions[model] = FINAL.predict_linear_lstsq(linear_model, X, y_shape)
        else:
            skipped[model] = f"Model is not supported by predict_latest_ais.py: {model}"
    return predictions, skipped


def prediction_rows(
    predictions: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    history_steps: int,
    dt_minutes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, pred in predictions.items():
        for sample_i, sample in enumerate(samples):
            for step_i, point in enumerate(pred[sample_i], start=1):
                rows.append(
                    {
                        "model": model,
                        "sample_id": sample_i,
                        "mmsi": sample["mmsi"],
                        "origin_time": origin_time(sample["start_time"], history_steps, dt_minutes),
                        "forecast_step": step_i,
                        "forecast_minutes": step_i * dt_minutes,
                        "pred_lat": float(point[0]),
                        "pred_lon": float(point[1]),
                        "source_start_time": sample["start_time"],
                        "source_end_time": sample["end_time"],
                        "region": sample["region"],
                    }
                )
    return rows


def risk_warning_rows(
    predictions: dict[str, np.ndarray],
    samples: list[dict[str, Any]],
    X: np.ndarray,
    warning_threshold_nmi: float,
    search_radius_nmi: float,
    history_steps: int,
    dt_minutes: int,
    max_pairs: int,
) -> list[dict[str, Any]]:
    by_start: dict[str, list[int]] = {}
    for sample_i, sample in enumerate(samples):
        by_start.setdefault(sample["start_time"], []).append(sample_i)

    rows: list[dict[str, Any]] = []
    for model, pred in predictions.items():
        pair_count = 0
        for start_time, members in sorted(by_start.items()):
            if len(members) < 2:
                continue
            for offset, a in enumerate(members[:-1]):
                origin_a = X[a, -1, :2]
                for b in members[offset + 1 :]:
                    origin_b = X[b, -1, :2]
                    origin_distance = float(haversine_nmi(origin_a.reshape(1, 2), origin_b.reshape(1, 2))[0])
                    if origin_distance > search_radius_nmi:
                        continue
                    distances = haversine_nmi(pred[a], pred[b])
                    min_pos = int(np.argmin(distances))
                    min_distance = float(distances[min_pos])
                    if min_distance > warning_threshold_nmi:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "sample_id_a": a,
                            "sample_id_b": b,
                            "mmsi_a": samples[a]["mmsi"],
                            "mmsi_b": samples[b]["mmsi"],
                            "origin_time": origin_time(start_time, history_steps, dt_minutes),
                            "region_a": samples[a]["region"],
                            "region_b": samples[b]["region"],
                            "origin_distance_nmi": origin_distance,
                            "pred_min_cpa_nmi": min_distance,
                            "pred_tcpa_minutes": float((min_pos + 1) * dt_minutes),
                            "warning_threshold_nmi": warning_threshold_nmi,
                        }
                    )
                    pair_count += 1
                    if pair_count >= max_pairs:
                        break
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_multiday.yaml")
    parser.add_argument("--input", help="New AIS CSV to process. Defaults to the first configured raw file.")
    parser.add_argument("--output-dir", default="outputs/latest_predictions")
    parser.add_argument("--models", nargs="*", default=["constant_velocity", "kalman_filter_cv", "linear_lstsq"])
    parser.add_argument("--linear-checkpoint", default="outputs/final_multiday/temporal_test/checkpoints/linear_lstsq.npz")
    parser.add_argument("--max-rows", type=int, default=250000)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-risk-pairs", type=int, default=500)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    input_path = Path(args.input or config["data"]["raw_files"][0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, raw_profiles = BUILD.load_raw_files([input_path], config, args.max_rows, args.chunksize)
    warnings: list[str] = []
    if df.empty:
        warnings.append("No valid AIS records were loaded from the input CSV.")
        samples: list[dict[str, Any]] = []
    else:
        samples = BUILD.create_samples(df, config)
        samples = sorted(samples, key=lambda sample: sample["start_time"])[-args.max_samples :]
    if not samples:
        warnings.append("No trajectory windows were created from the input CSV.")
        (output_dir / "prediction_manifest.json").write_text(
            json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "warnings": warnings, "raw_files": raw_profiles}, indent=2),
            encoding="utf-8",
        )
        print(f"No latest predictions written; see {output_dir / 'prediction_manifest.json'}")
        return 1

    X, y = sample_arrays(samples)
    forecast_steps = y.shape[1]
    history_steps = int(config["experiment"]["history_steps"])
    dt_minutes = int(config["experiment"].get("dt_minutes", 1))
    linear_model = load_linear_model(config, Path(args.linear_checkpoint) if args.linear_checkpoint else None)
    predictions, skipped = predict_models(args.models, X, forecast_steps, y.shape[1:], linear_model)
    pred_rows = prediction_rows(predictions, samples, history_steps, dt_minutes)
    risk_config = config.get("risk_warning", {})
    risk_rows = risk_warning_rows(
        predictions,
        samples,
        X,
        float(risk_config.get("warning_threshold_nmi", 0.5)),
        float(risk_config.get("search_radius_nmi", 3.0)),
        history_steps,
        dt_minutes,
        args.max_risk_pairs,
    )
    write_csv(
        output_dir / "trajectory_predictions.csv",
        pred_rows,
        ["model", "sample_id", "mmsi", "origin_time", "forecast_step", "forecast_minutes", "pred_lat", "pred_lon", "source_start_time", "source_end_time", "region"],
    )
    write_csv(
        output_dir / "risk_warnings.csv",
        risk_rows,
        ["model", "sample_id_a", "sample_id_b", "mmsi_a", "mmsi_b", "origin_time", "region_a", "region_b", "origin_distance_nmi", "pred_min_cpa_nmi", "pred_tcpa_minutes", "warning_threshold_nmi"],
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "input": str(input_path),
        "raw_files": raw_profiles,
        "sample_count": len(samples),
        "models_requested": args.models,
        "models_written": sorted(predictions),
        "skipped_models": skipped,
        "warning_count": len(risk_rows),
        "outputs": {
            "trajectory_predictions": str(output_dir / "trajectory_predictions.csv"),
            "risk_warnings": str(output_dir / "risk_warnings.csv"),
            "prediction_manifest": str(output_dir / "prediction_manifest.json"),
        },
        "notes": [
            "This is an offline prediction export for recently supplied AIS files, not a live AIS stream.",
            "Risk warnings are based on predicted pairwise separation within the forecast horizon.",
        ],
    }
    (output_dir / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Latest AIS predictions written to {output_dir / 'trajectory_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
