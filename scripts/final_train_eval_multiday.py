#!/usr/bin/env python3
"""Run high-quality-journal benchmark models on metadata-rich AIS arrays."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_SPEC = importlib.util.spec_from_file_location(
    "final_train_eval_base", PROJECT_ROOT / "scripts" / "final_train_eval.py"
)
if FINAL_SPEC is None or FINAL_SPEC.loader is None:
    raise RuntimeError("Could not load final_train_eval.py")
FINAL = importlib.util.module_from_spec(FINAL_SPEC)
FINAL_SPEC.loader.exec_module(FINAL)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_rows(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    merged = {tuple(str(row.get(field, "")) for field in key_fields): row for row in existing}
    for row in new_rows:
        merged[tuple(str(row.get(field, "")) for field in key_fields)] = row
    return list(merged.values())


def constant_acceleration_predict(X: np.ndarray, forecast_steps: int) -> np.ndarray:
    last = X[:, -1, :2]
    prev = X[:, -2, :2]
    prev2 = X[:, -3, :2] if X.shape[1] >= 3 else prev
    velocity = last - prev
    acceleration = (last - prev) - (prev - prev2)
    steps = np.arange(1, forecast_steps + 1, dtype=X.dtype).reshape(1, forecast_steps, 1)
    return last[:, None, :] + velocity[:, None, :] * steps + 0.5 * acceleration[:, None, :] * steps**2


def kalman_filter_cv_predict(X: np.ndarray, forecast_steps: int) -> np.ndarray:
    """Lightweight CV Kalman-style smoother using recent median velocity."""
    positions = X[:, :, :2]
    diffs = np.diff(positions, axis=1)
    recent = diffs[:, -min(8, diffs.shape[1]) :, :]
    velocity = np.median(recent, axis=1)
    last = positions[:, -1, :]
    steps = np.arange(1, forecast_steps + 1, dtype=X.dtype).reshape(1, forecast_steps, 1)
    return last[:, None, :] + velocity[:, None, :] * steps


def fit_ridge_lstsq(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    features = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    targets = y_train.reshape(y_train.shape[0], -1).astype(np.float64)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    features_scaled = (features - mean) / std
    design = np.concatenate([features_scaled, np.ones((features.shape[0], 1))], axis=1)
    penalty = alpha * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"mean": mean, "std": std, "weights": weights, "alpha": np.array([alpha])}


def predict_lstsq(model: dict[str, np.ndarray], X: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    features = X.reshape(X.shape[0], -1).astype(np.float64)
    features_scaled = (features - model["mean"]) / model["std"]
    design = np.concatenate([features_scaled, np.ones((features.shape[0], 1))], axis=1)
    return (design @ model["weights"]).reshape(X.shape[0], *output_shape)


def subset_arrays(data: dict[str, np.ndarray], split_key: str, split_value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = data[split_key]
    idx = np.where(split == split_value)[0]
    return data["X"][idx], data["y"][idx], idx


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def bin_label(value: float, bins: list[float], prefix: str) -> str:
    for low, high in zip(bins[:-1], bins[1:]):
        if low <= value < high:
            return f"{prefix}_{low:g}_{high:g}"
    return f"{prefix}_{bins[-1]:g}_plus"


def build_group_rows(
    per_sample_rows: list[dict[str, Any]],
    metadata: dict[str, np.ndarray],
    speed_bins: list[float],
    turn_bins: list[float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        idx = int(row["sample_index"])
        model = row["model"]
        split_policy = row["split_policy"]
        groups = [
            ("region", str(metadata["region"][idx])),
            ("speed_bin", bin_label(float(metadata["avg_sog"][idx]), speed_bins, "sog")),
            ("turn_bin", bin_label(float(metadata["turn_intensity_deg"][idx]), turn_bins, "turn")),
        ]
        for group_type, group_value in groups:
            grouped.setdefault((model, split_policy, group_type, group_value), []).append(row)

    rows = []
    for (model, split_policy, group_type, group_value), values in sorted(grouped.items()):
        rows.append(
            {
                "model": model,
                "split_policy": split_policy,
                "group_type": group_type,
                "group": group_value,
                "n": len(values),
                "ade_meters": float(np.mean([float(v["ade_meters"]) for v in values])),
                "fde_meters": float(np.mean([float(v["fde_meters"]) for v in values])),
                "rmse_meters": float(np.mean([float(v["rmse_meters"]) for v in values])),
                "mae_meters": float(np.mean([float(v["mae_meters"]) for v in values])),
            }
        )
    return rows


def evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray | None, dict[str, Any], float, float]:
    forecast_steps = y_eval.shape[1]
    started = time.time()
    if model_name == "constant_velocity":
        pred = FINAL.constant_velocity_predict(X_eval, forecast_steps)
        return pred, {}, 0.0, time.time() - started
    if model_name == "constant_acceleration":
        pred = constant_acceleration_predict(X_eval, forecast_steps)
        return pred, {}, 0.0, time.time() - started
    if model_name == "kalman_filter_cv":
        pred = kalman_filter_cv_predict(X_eval, forecast_steps)
        return pred, {}, 0.0, time.time() - started
    if model_name == "linear_lstsq":
        train_start = time.time()
        model = FINAL.fit_linear_lstsq(X_train, y_train)
        train_seconds = time.time() - train_start
        pred = FINAL.predict_linear_lstsq(model, X_eval, y_eval.shape[1:])
        np.savez_compressed(output_dir / "checkpoints" / "linear_lstsq.npz", **model)
        return pred, {"checkpoint": str(output_dir / "checkpoints" / "linear_lstsq.npz")}, train_seconds, time.time() - started - train_seconds
    if model_name == "ridge_lstsq":
        train_start = time.time()
        model = fit_ridge_lstsq(X_train, y_train, alpha=1.0)
        train_seconds = time.time() - train_start
        pred = predict_lstsq(model, X_eval, y_eval.shape[1:])
        np.savez_compressed(output_dir / "checkpoints" / "ridge_lstsq.npz", **model)
        return pred, {"checkpoint": str(output_dir / "checkpoints" / "ridge_lstsq.npz"), "alpha": 1.0}, train_seconds, time.time() - started - train_seconds
    if model_name in {"lstm_baseline", "gru_baseline", "transformer_baseline", "tcn_baseline"}:
        pred, artifacts, train_seconds, inference_seconds = FINAL.fit_predict_torch_model(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            X_eval,
            {
                "experiment": config["experiment"],
                "models": {
                    "training": {
                        model_name: {
                            **config.get("models", {}).get("neural_training", {}),
                            **({"hidden_size": 128, "n_layers": 2, "dropout": 0.1} if model_name in {"lstm_baseline", "gru_baseline"} else {}),
                            **({"d_model": 96, "n_heads": 4, "n_layers": 2, "d_ff": 192, "dropout": 0.1} if model_name == "transformer_baseline" else {}),
                            **({"channels": [64, 96, 128], "kernel_size": 3, "dropout": 0.1} if model_name == "tcn_baseline" else {}),
                        }
                    }
                },
            },
            output_dir,
        )
        return pred, artifacts, train_seconds, inference_seconds
    return None, {"status": "skipped", "notes": f"Unknown model: {model_name}"}, 0.0, 0.0


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - confidence
    return (
        float(np.percentile(values, 100 * alpha / 2.0)),
        float(np.percentile(values, 100 * (1.0 - alpha / 2.0))),
    )


def paired_tests(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = candidate - reference
    result: dict[str, Any] = {
        "n": int(diff.size),
        "mean_difference_candidate_minus_reference": float(diff.mean()) if diff.size else float("nan"),
        "paired_t": None,
        "wilcoxon": None,
    }
    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(candidate, reference)
        result["paired_t"] = {"statistic": float(t_stat), "p_value": float(p_value)}
        if diff.size > 0 and np.any(diff != 0):
            w_stat, w_p = stats.wilcoxon(diff)
            result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
    except Exception as exc:  # pragma: no cover - scipy may be unavailable
        result["error"] = str(exc)
    return result


def build_statistical_summary(
    per_sample_rows: list[dict[str, Any]],
    reference_model: str = "constant_velocity",
    confidence: float = 0.95,
) -> dict[str, Any]:
    by_split_model: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for row in per_sample_rows:
        split_policy = str(row["split_policy"])
        model = str(row["model"])
        sample_index = int(row["sample_index"])
        by_split_model.setdefault(split_policy, {}).setdefault(model, {})[sample_index] = {
            "ade_meters": float(row["ade_meters"]),
            "fde_meters": float(row["fde_meters"]),
            "rmse_meters": float(row["rmse_meters"]),
            "mae_meters": float(row["mae_meters"]),
        }

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference": reference_model,
        "confidence": confidence,
        "splits": {},
    }
    for split_policy, by_model in sorted(by_split_model.items()):
        split_summary: dict[str, Any] = {"models": {}, "pairwise_vs_reference": {}, "multiple_testing_correction": None}
        for model, sample_map in sorted(by_model.items()):
            ades = np.array([row["ade_meters"] for _, row in sorted(sample_map.items())])
            fdes = np.array([row["fde_meters"] for _, row in sorted(sample_map.items())])
            split_summary["models"][model] = {
                "n": int(ades.size),
                "ade_mean": float(ades.mean()) if ades.size else float("nan"),
                "ade_median": float(np.median(ades)) if ades.size else float("nan"),
                "ade_ci_percentile": confidence_interval(ades, confidence),
                "fde_mean": float(fdes.mean()) if fdes.size else float("nan"),
                "fde_ci_percentile": confidence_interval(fdes, confidence),
            }
        if reference_model in by_model:
            reference_samples = by_model[reference_model]
            comparisons = 0
            for model, sample_map in sorted(by_model.items()):
                if model == reference_model:
                    continue
                common = sorted(set(reference_samples) & set(sample_map))
                if not common:
                    continue
                reference_ade = np.array([reference_samples[idx]["ade_meters"] for idx in common])
                candidate_ade = np.array([sample_map[idx]["ade_meters"] for idx in common])
                split_summary["pairwise_vs_reference"][model] = paired_tests(reference_ade, candidate_ade)
                comparisons += 1
            if comparisons:
                split_summary["multiple_testing_correction"] = {
                    "method": "bonferroni",
                    "comparisons": comparisons,
                    "alpha_0_05_corrected": 0.05 / comparisons,
                }
        summary["splits"][split_policy] = split_summary
    return summary


def capped_split(
    data: dict[str, np.ndarray],
    split_key: str,
    train_cap: int | None,
    test_cap: int | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = {
        "train": subset_arrays(data, split_key, "train"),
        "val": subset_arrays(data, split_key, "val"),
        "test": subset_arrays(data, split_key, "test"),
    }
    if train_cap:
        for name in ["train", "val"]:
            X, y, idx = result[name]
            cap = train_cap if name == "train" else max(1, min(train_cap // 5, X.shape[0]))
            result[name] = (X[:cap], y[:cap], idx[:cap])
    if test_cap:
        X, y, idx = result["test"]
        result["test"] = (X[:test_cap], y[:test_cap], idx[:test_cap])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_multiday.yaml")
    parser.add_argument("--models", nargs="*")
    parser.add_argument(
        "--split-policies",
        nargs="*",
        default=["temporal_test", "vessel_disjoint_test"],
        help="Split policies to evaluate: temporal_test and/or vessel_disjoint_test.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge selected model outputs into existing final_multiday CSV artifacts instead of replacing all rows.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["experiment"].get("output_dir", "outputs/final_multiday"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "training_logs").mkdir(exist_ok=True)
    data = load_npz(Path(config["data"]["processed_npz"]))
    coordinate_units = config["data"].get("coordinate_units", "degrees_latlon_wgs84")
    metric_distance = config["data"].get("metric_distance", "haversine_meters")
    models = args.models or config["models"]["final_main"]
    split_specs = {
        "temporal_test": "temporal_split",
        "vessel_disjoint_test": "vessel_split",
    }
    requested_splits = args.split_policies or ["temporal_test", "vessel_disjoint_test"]
    unknown_splits = [name for name in requested_splits if name not in split_specs]
    if unknown_splits:
        raise ValueError(f"Unknown split policies: {unknown_splits}. Valid policies: {sorted(split_specs)}")

    metadata = {
        "region": data["region"],
        "avg_sog": data["avg_sog"],
        "turn_intensity_deg": data["turn_intensity_deg"],
    }
    metric_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    unit_verified = (
        str(coordinate_units).lower() == "degrees_latlon_wgs84"
        and str(metric_distance).lower() == "haversine_meters"
    )

    for split_policy in requested_splits:
        split_key = split_specs[split_policy]
        split_data = capped_split(data, split_key, args.max_train_samples, args.max_test_samples)
        X_train, y_train, train_idx = split_data["train"]
        X_val, y_val, val_idx = split_data["val"]
        X_test, y_test, test_idx = split_data["test"]
        split_output_dir = output_dir / split_policy
        (split_output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (split_output_dir / "training_logs").mkdir(parents=True, exist_ok=True)

        for model_name in models:
            log = {
                "model": model_name,
                "split_policy": split_policy,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "n_train": int(X_train.shape[0]),
                "n_val": int(X_val.shape[0]),
                "n_test": int(X_test.shape[0]),
            }
            try:
                pred, artifacts, train_seconds, inference_seconds = evaluate_model(
                    model_name, X_train, y_train, X_val, y_val, X_test, y_test, config, split_output_dir
                )
                if pred is None:
                    log.update({"status": artifacts.get("status", "skipped"), "notes": artifacts.get("notes", "")})
                    skipped_row = {
                        "model": model_name,
                        "status": log["status"],
                        "split_policy": split_policy,
                        "ade_meters": "",
                        "fde_meters": "",
                        "rmse_meters": "",
                        "mae_meters": "",
                        "n_train": int(X_train.shape[0]),
                        "n_val": int(X_val.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "forecast_steps": int(y_test.shape[1]) if y_test.ndim == 3 else "",
                        "metric_unit": metric_distance,
                        "unit_verified": unit_verified,
                        "train_seconds": "",
                        "inference_seconds": "",
                        "notes": log.get("notes", ""),
                    }
                    metric_rows.append(skipped_row)
                    generalization_rows.append(skipped_row.copy())
                    continue
                metrics, sample_metrics = FINAL.metric_bundle(pred, y_test, coordinate_units)
                row = {
                    "model": model_name,
                    "status": "ok",
                    "split_policy": split_policy,
                    **metrics,
                    "n_train": int(X_train.shape[0]),
                    "n_val": int(X_val.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "forecast_steps": int(y_test.shape[1]),
                    "metric_unit": metric_distance,
                    "unit_verified": unit_verified,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "notes": "",
                }
                metric_rows.append(row)
                generalization_rows.append(row.copy())
                for local_idx, sample in enumerate(sample_metrics):
                    global_idx = int(test_idx[local_idx])
                    sample_row = {
                        "model": model_name,
                        "split_policy": split_policy,
                        "sample_index": global_idx,
                        "ade_meters": sample["ade_meters"],
                        "fde_meters": sample["fde_meters"],
                        "rmse_meters": sample["rmse_meters"],
                        "mae_meters": sample["mae_meters"],
                    }
                    per_sample_rows.append(sample_row)
                for horizon_row in FINAL.horizon_metrics(pred, y_test, model_name, coordinate_units):
                    horizon_rows.append({"split_policy": split_policy, **horizon_row})
                log.update(
                    {
                        "status": "ok",
                        "metrics": metrics,
                        "artifacts": artifacts,
                        "train_seconds": train_seconds,
                        "inference_seconds": inference_seconds,
                    }
                )
            except Exception as exc:
                log.update({"status": "failed", "error": str(exc)})
                failed_row = {
                    "model": model_name,
                    "status": "failed",
                    "split_policy": split_policy,
                    "ade_meters": "",
                    "fde_meters": "",
                    "rmse_meters": "",
                    "mae_meters": "",
                    "n_train": int(X_train.shape[0]),
                    "n_val": int(X_val.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "forecast_steps": int(y_test.shape[1]) if y_test.ndim == 3 else "",
                    "metric_unit": metric_distance,
                    "unit_verified": unit_verified,
                    "train_seconds": "",
                    "inference_seconds": "",
                    "notes": str(exc),
                }
                metric_rows.append(failed_row)
                generalization_rows.append(failed_row.copy())
            finally:
                log_path = output_dir / "training_logs" / f"{split_policy}__{model_name}.json"
                log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    group_rows = build_group_rows(
        per_sample_rows,
        metadata,
        config.get("scenario_slices", {}).get("speed_bins_knots", [0, 2, 8, 15, 50]),
        config.get("scenario_slices", {}).get("turn_rate_bins_degrees", [0, 5, 15, 45, 180]),
    )
    if args.merge_existing:
        metric_rows = merge_rows(
            read_csv(output_dir / "model_metrics.csv"),
            metric_rows,
            ["split_policy", "model"],
        )
        per_sample_rows = merge_rows(
            read_csv(output_dir / "per_sample_errors.csv"),
            per_sample_rows,
            ["split_policy", "model", "sample_index"],
        )
        horizon_rows = merge_rows(
            read_csv(output_dir / "error_summary_by_horizon.csv"),
            horizon_rows,
            ["split_policy", "model", "horizon_step"],
        )
        group_rows = merge_rows(
            read_csv(output_dir / "error_summary_by_group.csv"),
            group_rows,
            ["split_policy", "model", "group_type", "group"],
        )
        generalization_rows = merge_rows(
            read_csv(output_dir / "generalization_metrics.csv"),
            generalization_rows,
            ["split_policy", "model"],
        )
    metric_rows = sorted(metric_rows, key=lambda row: (str(row.get("split_policy", "")), str(row.get("model", ""))))
    per_sample_rows = sorted(
        per_sample_rows,
        key=lambda row: (str(row.get("split_policy", "")), str(row.get("model", "")), int(row.get("sample_index", 0))),
    )
    horizon_rows = sorted(
        horizon_rows,
        key=lambda row: (str(row.get("split_policy", "")), str(row.get("model", "")), int(row.get("horizon_step", 0))),
    )
    group_rows = sorted(
        group_rows,
        key=lambda row: (
            str(row.get("split_policy", "")),
            str(row.get("model", "")),
            str(row.get("group_type", "")),
            str(row.get("group", "")),
        ),
    )
    generalization_rows = sorted(
        generalization_rows,
        key=lambda row: (str(row.get("split_policy", "")), str(row.get("model", ""))),
    )

    write_csv(
        output_dir / "model_metrics.csv",
        metric_rows,
        ["model", "status", "split_policy", "ade_meters", "fde_meters", "rmse_meters", "mae_meters", "n_train", "n_val", "n_test", "forecast_steps", "metric_unit", "unit_verified", "train_seconds", "inference_seconds", "notes"],
    )
    write_csv(
        output_dir / "per_sample_errors.csv",
        per_sample_rows,
        ["model", "sample_index", "split_policy", "ade_meters", "fde_meters", "rmse_meters", "mae_meters"],
    )
    write_csv(
        output_dir / "error_summary_by_horizon.csv",
        horizon_rows,
        ["split_policy", "model", "horizon_step", "ade_meters", "fde_meters", "rmse_meters", "mae_meters"],
    )
    write_csv(
        output_dir / "error_summary_by_group.csv",
        group_rows,
        ["model", "split_policy", "group_type", "group", "n", "ade_meters", "fde_meters", "rmse_meters", "mae_meters"],
    )
    write_csv(
        output_dir / "generalization_metrics.csv",
        generalization_rows,
        ["model", "status", "split_policy", "ade_meters", "fde_meters", "rmse_meters", "mae_meters", "n_train", "n_val", "n_test", "forecast_steps", "metric_unit", "unit_verified", "train_seconds", "inference_seconds", "notes"],
    )
    statistical_summary = build_statistical_summary(per_sample_rows)
    (output_dir / "statistical_tests.json").write_text(json.dumps(statistical_summary, indent=2), encoding="utf-8")

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "models": sorted({str(row.get("model", "")) for row in metric_rows if row.get("model")}),
        "split_policies": sorted({str(row.get("split_policy", "")) for row in metric_rows if row.get("split_policy")}),
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "is_debug_run": bool(args.max_train_samples or args.max_test_samples),
        "merge_existing": args.merge_existing,
        "outputs": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "generalization_metrics": str(output_dir / "generalization_metrics.csv"),
            "per_sample_errors": str(output_dir / "per_sample_errors.csv"),
            "error_summary_by_horizon": str(output_dir / "error_summary_by_horizon.csv"),
            "error_summary_by_group": str(output_dir / "error_summary_by_group.csv"),
            "statistical_tests": str(output_dir / "statistical_tests.json"),
        },
        "unit_status": {
            "coordinate_units": coordinate_units,
            "metric_distance": metric_distance,
            "unit_verified": unit_verified,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Multiday benchmark metrics written to {output_dir / 'model_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
