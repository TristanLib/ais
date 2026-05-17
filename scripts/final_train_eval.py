#!/usr/bin/env python3
"""Run final conservative trajectory-prediction experiments.

This script intentionally avoids relying on the older training scripts. It writes
auditable artifacts under outputs/final and never extends 15-step targets to 30
steps for main reported results.
"""

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

EARTH_RADIUS_M = 6_371_008.8


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse {path}. Install PyYAML or keep the config JSON-compatible."
            ) from exc


def constant_velocity_predict(X: np.ndarray, forecast_steps: int) -> np.ndarray:
    last = X[:, -1, :2]
    prev = X[:, -2, :2] if X.shape[1] >= 2 else X[:, -1, :2]
    velocity = last - prev
    steps = np.arange(1, forecast_steps + 1, dtype=X.dtype).reshape(1, forecast_steps, 1)
    return last[:, None, :] + velocity[:, None, :] * steps


def fit_linear_lstsq(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    features = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    targets = y_train.reshape(y_train.shape[0], -1).astype(np.float64)

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    features_scaled = (features - mean) / std
    design = np.concatenate([features_scaled, np.ones((features.shape[0], 1))], axis=1)
    weights, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return {"mean": mean, "std": std, "weights": weights}


def predict_linear_lstsq(model: dict[str, np.ndarray], X: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    features = X.reshape(X.shape[0], -1).astype(np.float64)
    features_scaled = (features - model["mean"]) / model["std"]
    design = np.concatenate([features_scaled, np.ones((features.shape[0], 1))], axis=1)
    flat_pred = design @ model["weights"]
    return flat_pred.reshape(X.shape[0], *output_shape)


def haversine_meters(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance for arrays whose last dimension is [lat, lon] degrees."""
    lat1 = np.deg2rad(a[..., 0])
    lon1 = np.deg2rad(a[..., 1])
    lat2 = np.deg2rad(b[..., 0])
    lon2 = np.deg2rad(b[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def component_errors_meters(pred: np.ndarray, target: np.ndarray, coordinate_units: str) -> np.ndarray:
    """Return component-level errors in meters for RMSE/MAE-style summaries."""
    coordinate_units = coordinate_units.lower()
    if coordinate_units == "degrees_latlon_wgs84":
        mean_lat = np.deg2rad((pred[..., 0] + target[..., 0]) / 2.0)
        north = np.deg2rad(pred[..., 0] - target[..., 0]) * EARTH_RADIUS_M
        east = np.deg2rad(pred[..., 1] - target[..., 1]) * EARTH_RADIUS_M * np.cos(mean_lat)
        return np.stack([north, east], axis=-1)
    return pred - target


def displacement_errors(pred: np.ndarray, target: np.ndarray, coordinate_units: str) -> np.ndarray:
    coordinate_units = coordinate_units.lower()
    if coordinate_units == "degrees_latlon_wgs84":
        return haversine_meters(pred, target)
    return np.linalg.norm(pred - target, axis=-1)


def metric_bundle(
    pred: np.ndarray, target: np.ndarray, coordinate_units: str
) -> tuple[dict[str, float], list[dict[str, float]]]:
    component_diff = component_errors_meters(pred, target, coordinate_units)
    displacement = displacement_errors(pred, target, coordinate_units)

    per_sample = []
    for idx in range(pred.shape[0]):
        sample_diff = component_diff[idx]
        sample_disp = displacement[idx]
        per_sample.append(
            {
                "sample_index": idx,
                "ade_meters": float(sample_disp.mean()),
                "fde_meters": float(sample_disp[-1]),
                "rmse_meters": float(np.sqrt(np.mean(sample_diff**2))),
                "mae_meters": float(np.mean(np.abs(sample_diff))),
            }
        )

    metrics = {
        "ade_meters": float(displacement.mean()),
        "fde_meters": float(displacement[:, -1].mean()),
        "rmse_meters": float(np.sqrt(np.mean(component_diff**2))),
        "mae_meters": float(np.mean(np.abs(component_diff))),
    }
    return metrics, per_sample


def horizon_metrics(
    pred: np.ndarray, target: np.ndarray, model_name: str, coordinate_units: str
) -> list[dict[str, Any]]:
    rows = []
    for horizon in range(1, target.shape[1] + 1):
        component_diff = component_errors_meters(
            pred[:, :horizon, :], target[:, :horizon, :], coordinate_units
        )
        displacement = displacement_errors(pred[:, :horizon, :], target[:, :horizon, :], coordinate_units)
        rows.append(
            {
                "model": model_name,
                "horizon_step": horizon,
                "ade_meters": float(displacement.mean()),
                "fde_meters": float(displacement[:, -1].mean()),
                "rmse_meters": float(np.sqrt(np.mean(component_diff**2))),
                "mae_meters": float(np.mean(np.abs(component_diff))),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def torch_device(torch_module: Any, preference: str) -> str:
    if preference == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preference == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if preference == "mps" and (
        not hasattr(torch_module.backends, "mps") or not torch_module.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but not available.")
    return preference


def torch_model_config(
    model_name: str, train_config: dict[str, Any], input_dim: int, forecast_steps: int, output_dim: int
) -> dict[str, Any]:
    if model_name == "lstm_baseline":
        return {
            "model": {
                "architecture": {
                    "d_input": input_dim,
                    "hidden_size": int(train_config.get("hidden_size", 128)),
                    "n_layers": int(train_config.get("n_layers", 2)),
                    "dropout": float(train_config.get("dropout", 0.1)),
                    "bidirectional": False,
                    "forecast_steps": forecast_steps,
                    "output_dim": output_dim,
                    "head_hidden": int(train_config.get("head_hidden", 128)),
                    "head_dropout": float(train_config.get("dropout", 0.1)),
                },
                "teacher_forcing_ratio": 0.0,
            }
        }
    if model_name == "gru_baseline":
        return {
            "model": {
                "architecture": {
                    "d_input": input_dim,
                    "hidden_size": int(train_config.get("hidden_size", 128)),
                    "n_layers": int(train_config.get("n_layers", 2)),
                    "dropout": float(train_config.get("dropout", 0.1)),
                    "forecast_steps": forecast_steps,
                    "output_dim": output_dim,
                    "head_hidden": int(train_config.get("head_hidden", 128)),
                },
                "teacher_forcing_ratio": 0.0,
            }
        }
    if model_name == "transformer_baseline":
        return {
            "model": {
                "architecture": {
                    "d_input": input_dim,
                    "d_model": int(train_config.get("d_model", 96)),
                    "n_heads": int(train_config.get("n_heads", 4)),
                    "n_layers": int(train_config.get("n_layers", 2)),
                    "d_ff": int(train_config.get("d_ff", 192)),
                    "dropout": float(train_config.get("dropout", 0.1)),
                    "pos_encoding_type": "sinusoidal",
                    "max_seq_len": 512,
                    "forecast_steps": forecast_steps,
                    "output_dim": output_dim,
                    "use_neighbor_attention": False,
                    "neighbor_d_model": 64,
                }
            }
        }
    if model_name == "tcn_baseline":
        return {
            "model": {
                "architecture": {
                    "d_input": input_dim,
                    "channels": train_config.get("channels", [64, 96, 128]),
                    "kernel_size": int(train_config.get("kernel_size", 3)),
                    "dropout": float(train_config.get("dropout", 0.1)),
                    "forecast_steps": forecast_steps,
                    "output_dim": output_dim,
                    "head_hidden": int(train_config.get("head_hidden", 128)),
                }
            }
        }
    raise ValueError(f"Unsupported torch model: {model_name}")


def create_torch_model(model_name: str, model_config: dict[str, Any]) -> Any:
    if model_name == "lstm_baseline":
        module_path = PROJECT_ROOT / "src" / "models" / "lstm.py"
        spec = importlib.util.spec_from_file_location("conservative_lstm_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load LSTM module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        create_lstm_model = module.create_lstm_model
        return create_lstm_model(model_config, "basic")
    if model_name == "gru_baseline":
        module_path = PROJECT_ROOT / "src" / "models" / "gru.py"
        spec = importlib.util.spec_from_file_location("conservative_gru_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load GRU module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        create_gru_model = module.create_gru_model
        return create_gru_model(model_config, "basic")
    if model_name == "transformer_baseline":
        module_path = PROJECT_ROOT / "src" / "models" / "transformer.py"
        spec = importlib.util.spec_from_file_location("conservative_transformer_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Transformer module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        create_transformer_model = module.create_transformer_model
        return create_transformer_model(model_config, "encoder")
    if model_name == "tcn_baseline":
        module_path = PROJECT_ROOT / "src" / "models" / "tcn.py"
        spec = importlib.util.spec_from_file_location("conservative_tcn_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load TCN module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        create_tcn_model = module.create_tcn_model
        return create_tcn_model(model_config, "basic")
    raise ValueError(f"Unsupported torch model: {model_name}")


def fit_predict_torch_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    X_test: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], float, float]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for neural final runs. Install the project requirements first."
        ) from exc

    seed = int(config["experiment"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_config = config["models"].get("training", {}).get(model_name, {})
    device = torch_device(torch, str(train_config.get("device", "auto")))
    epochs = int(train_config.get("epochs", 8))
    batch_size = int(train_config.get("batch_size", 128))
    learning_rate = float(train_config.get("learning_rate", 1e-3))
    early_stopping_patience = train_config.get("early_stopping_patience")
    early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)

    x_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0, keepdims=True).reshape(1, 1, -1)
    x_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0, keepdims=True).reshape(1, 1, -1)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y_train.reshape(-1, y_train.shape[-1]).mean(axis=0, keepdims=True).reshape(1, 1, -1)
    y_std = y_train.reshape(-1, y_train.shape[-1]).std(axis=0, keepdims=True).reshape(1, 1, -1)
    y_std[y_std < 1e-8] = 1.0

    X_train_scaled = ((X_train - x_mean) / x_std).astype(np.float32)
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    X_test_scaled = ((X_test - x_mean) / x_std).astype(np.float32)
    X_val_scaled = ((X_val - x_mean) / x_std).astype(np.float32) if X_val is not None else None
    y_val_scaled = ((y_val - y_mean) / y_std).astype(np.float32) if y_val is not None else None

    model_config = torch_model_config(
        model_name,
        train_config,
        input_dim=X_train.shape[-1],
        forecast_steps=y_train.shape[1],
        output_dim=y_train.shape[-1],
    )
    model = create_torch_model(model_name, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_scaled))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    val_tensors = None
    if X_val_scaled is not None and y_val_scaled is not None:
        val_tensors = (
            torch.from_numpy(X_val_scaled).to(device),
            torch.from_numpy(y_val_scaled).to(device),
        )

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
        }
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_tensors[0])
                val_loss = criterion(val_pred, val_tensors[1])
            epoch_record["val_loss"] = float(val_loss.detach().cpu().item())
            if early_stopping_patience is not None:
                current_val = epoch_record["val_loss"]
                if current_val < best_val_loss - 1e-8:
                    best_val_loss = current_val
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        history.append(epoch_record)
        if (
            val_tensors is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            break
    train_seconds = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    inference_start = time.time()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X_test_scaled.shape[0], batch_size):
            batch = torch.from_numpy(X_test_scaled[start : start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    inference_seconds = time.time() - inference_start

    pred_scaled = np.concatenate(preds, axis=0)
    pred = pred_scaled * y_std + y_mean

    checkpoint_path = output_dir / "checkpoints" / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": train_config,
            "normalization": {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            },
        },
        checkpoint_path,
    )

    artifact_log = {
        "model_config": model_config,
        "training_config": train_config,
        "device": device,
        "epochs": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch if best_epoch else None,
        "best_val_loss": best_val_loss if best_state is not None else None,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    return pred.astype(np.float64), artifact_log, train_seconds, inference_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_conservative.yaml")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to run. Defaults to config models.final_main.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for debugging. Do not use for final paper metrics.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for debugging. Do not use for final paper metrics.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["experiment"].get("output_dir", "outputs/final"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    keys = config["data"]["array_keys"]
    data_path = Path(config["data"]["processed_npz"])
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as data:
        X_train = data[keys["X_train"]]
        y_train = data[keys["y_train"]]
        X_val = data[keys["X_val"]]
        y_val = data[keys["y_val"]]
        X_test = data[keys["X_test"]]
        y_test = data[keys["y_test"]]

    if args.max_train_samples:
        X_train = X_train[: args.max_train_samples]
        y_train = y_train[: args.max_train_samples]
        X_val = X_val[: max(1, min(args.max_train_samples // 5, X_val.shape[0]))]
        y_val = y_val[: max(1, min(args.max_train_samples // 5, y_val.shape[0]))]
    if args.max_test_samples:
        X_test = X_test[: args.max_test_samples]
        y_test = y_test[: args.max_test_samples]

    forecast_steps = y_test.shape[1]
    configured_steps = config["experiment"]["forecast_steps"]
    if forecast_steps != configured_steps:
        raise ValueError(
            f"Configured forecast_steps={configured_steps}, but y_test has {forecast_steps}."
        )

    models = args.models or config["models"]["final_main"]
    coordinate_units = str(config["data"].get("coordinate_units", "unknown"))
    metric_distance = str(config["data"].get("metric_distance", "raw_euclidean"))
    unit_verified = (
        coordinate_units.lower() == "degrees_latlon_wgs84"
        and metric_distance.lower() == "haversine_meters"
    ) or ("meter" in coordinate_units.lower() and "unknown" not in coordinate_units.lower())
    unit_note = (
        ""
        if unit_verified
        else "Coordinate units or distance metric are not verified; do not cite these metrics as meters."
    )
    metric_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for model_name in models:
        started = time.time()
        log: dict[str, Any] = {
            "model": model_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "forecast_steps": int(forecast_steps),
        }
        try:
            if model_name == "constant_velocity":
                train_seconds = 0.0
                inference_start = time.time()
                pred = constant_velocity_predict(X_test, forecast_steps)
                inference_seconds = time.time() - inference_start
            elif model_name == "linear_lstsq":
                train_start = time.time()
                linear_model = fit_linear_lstsq(X_train, y_train)
                train_seconds = time.time() - train_start
                inference_start = time.time()
                pred = predict_linear_lstsq(linear_model, X_test, y_test.shape[1:])
                inference_seconds = time.time() - inference_start
                np.savez_compressed(output_dir / "checkpoints" / "linear_lstsq.npz", **linear_model)
            elif model_name in {"lstm_baseline", "transformer_baseline"}:
                pred, torch_artifacts, train_seconds, inference_seconds = fit_predict_torch_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    config,
                    output_dir,
                )
                log.update(torch_artifacts)
            else:
                log["status"] = "skipped"
                log["notes"] = "Model is not implemented in the conservative final runner yet."
                metric_rows.append(
                    {
                        "model": model_name,
                        "status": "skipped",
                        "ade_meters": "",
                        "fde_meters": "",
                        "rmse_meters": "",
                        "mae_meters": "",
                        "n_train": int(X_train.shape[0]),
                        "n_test": int(X_test.shape[0]),
                        "forecast_steps": int(forecast_steps),
                        "metric_unit": metric_distance,
                        "unit_verified": unit_verified,
                        "train_seconds": "",
                        "inference_seconds": "",
                        "notes": log["notes"],
                    }
                )
                continue

            metrics, sample_metrics = metric_bundle(pred, y_test, coordinate_units)
            elapsed = time.time() - started

            metric_rows.append(
                {
                    "model": model_name,
                    "status": "ok",
                    **metrics,
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "forecast_steps": int(forecast_steps),
                    "metric_unit": metric_distance,
                    "unit_verified": unit_verified,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "notes": unit_note,
                }
            )
            for sample in sample_metrics:
                per_sample_rows.append({"model": model_name, **sample})
            horizon_rows.extend(horizon_metrics(pred, y_test, model_name, coordinate_units))
            group_rows.append(
                {
                    "model": model_name,
                    "group": "all_samples",
                    **metrics,
                    "n_test": int(X_test.shape[0]),
                    "notes": "No vessel, timestamp, region, or ship-type metadata is present in the NPZ.",
                }
            )

            if args.save_predictions:
                np.savez_compressed(output_dir / f"{model_name}_predictions.npz", pred=pred, target=y_test)

            log.update(
                {
                    "status": "ok",
                    "metrics": metrics,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "elapsed_seconds": elapsed,
                }
            )
        except Exception as exc:  # pragma: no cover - archives failures for audit
            log.update({"status": "failed", "error": str(exc)})
            metric_rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "ade_meters": "",
                    "fde_meters": "",
                    "rmse_meters": "",
                    "mae_meters": "",
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "forecast_steps": int(forecast_steps),
                    "metric_unit": metric_distance,
                    "unit_verified": unit_verified,
                    "train_seconds": "",
                    "inference_seconds": "",
                    "notes": str(exc),
                }
            )
        finally:
            log_path = output_dir / "training_logs" / f"{model_name}.json"
            log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    write_csv(
        output_dir / "model_metrics.csv",
        metric_rows,
        [
            "model",
            "status",
            "ade_meters",
            "fde_meters",
            "rmse_meters",
            "mae_meters",
            "n_train",
            "n_test",
            "forecast_steps",
            "metric_unit",
            "unit_verified",
            "train_seconds",
            "inference_seconds",
            "notes",
        ],
    )
    write_csv(
        output_dir / "per_sample_errors.csv",
        per_sample_rows,
        ["model", "sample_index", "ade_meters", "fde_meters", "rmse_meters", "mae_meters"],
    )
    write_csv(
        output_dir / "error_summary_by_horizon.csv",
        horizon_rows,
        ["model", "horizon_step", "ade_meters", "fde_meters", "rmse_meters", "mae_meters"],
    )
    write_csv(
        output_dir / "error_summary_by_group.csv",
        group_rows,
        ["model", "group", "ade_meters", "fde_meters", "rmse_meters", "mae_meters", "n_test", "notes"],
    )

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "models": models,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
        "is_debug_run": bool(args.max_train_samples or args.max_test_samples),
        "outputs": {
            "model_metrics": str(output_dir / "model_metrics.csv"),
            "per_sample_errors": str(output_dir / "per_sample_errors.csv"),
            "error_summary_by_horizon": str(output_dir / "error_summary_by_horizon.csv"),
            "error_summary_by_group": str(output_dir / "error_summary_by_group.csv"),
        },
        "unit_status": {
            "coordinate_units": coordinate_units,
            "metric_distance": metric_distance,
            "unit_verified": unit_verified,
            "notes": unit_note,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Final metrics written to {output_dir / 'model_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
