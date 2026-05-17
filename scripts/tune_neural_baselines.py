#!/usr/bin/env python3
"""Run a documented validation-set proxy search for neural AIS baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def subset(data: dict[str, np.ndarray], split_key: str, split_value: str, cap: int | None) -> tuple[np.ndarray, np.ndarray]:
    idx = np.where(data[split_key] == split_value)[0]
    if cap:
        idx = idx[:cap]
    return data["X"][idx], data["y"][idx]


def expand_grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_multiday.yaml")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--split-policy", default="temporal_split")
    parser.add_argument("--proxy-train-samples", type=int)
    parser.add_argument("--proxy-val-samples", type=int)
    parser.add_argument("--max-configs-per-model", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/final_multiday")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    tuning_config = config.get("models", {}).get("neural_tuning", {})
    search_spaces = tuning_config.get("search_space", {})
    models = args.models or sorted(search_spaces)
    train_cap = args.proxy_train_samples or int(tuning_config.get("proxy_train_samples", 8192))
    val_cap = args.proxy_val_samples or int(tuning_config.get("proxy_val_samples", 2048))
    output_dir = Path(args.output_dir)
    tuning_dir = output_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(Path(config["data"]["processed_npz"]))
    X_train, y_train = subset(data, args.split_policy, "train", train_cap)
    X_val, y_val = subset(data, args.split_policy, "val", val_cap)
    if X_train.size == 0 or X_val.size == 0:
        raise RuntimeError("Tuning requires non-empty train and validation arrays.")

    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for model_name in models:
        grid = expand_grid(search_spaces.get(model_name, {}))
        if args.max_configs_per_model:
            grid = grid[: args.max_configs_per_model]
        model_rows: list[dict[str, Any]] = []
        for config_index, candidate in enumerate(grid, start=1):
            started = time.time()
            run_dir = tuning_dir / model_name / f"config_{config_index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            train_config = {
                **config.get("models", {}).get("neural_training", {}),
                **candidate,
            }
            run_config = {
                "experiment": config["experiment"],
                "models": {"training": {model_name: train_config}},
            }
            row: dict[str, Any] = {
                "model": model_name,
                "config_index": config_index,
                "status": "running",
                "proxy_train_samples": int(X_train.shape[0]),
                "proxy_val_samples": int(X_val.shape[0]),
                "config_json": json.dumps(candidate, sort_keys=True),
            }
            try:
                pred, artifacts, train_seconds, inference_seconds = FINAL.fit_predict_torch_model(
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_val,
                    run_config,
                    run_dir,
                )
                metrics, _ = FINAL.metric_bundle(pred, y_val, config["data"].get("coordinate_units", "degrees_latlon_wgs84"))
                row.update(
                    {
                        "status": "ok",
                        "validation_ade_meters": metrics["ade_meters"],
                        "validation_fde_meters": metrics["fde_meters"],
                        "validation_rmse_meters": metrics["rmse_meters"],
                        "validation_mae_meters": metrics["mae_meters"],
                        "train_seconds": train_seconds,
                        "inference_seconds": inference_seconds,
                        "wall_seconds": time.time() - started,
                        "best_epoch": artifacts.get("best_epoch"),
                        "stopped_early": artifacts.get("stopped_early"),
                        "checkpoint": artifacts.get("checkpoint"),
                        "notes": "",
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "validation_ade_meters": "",
                        "validation_fde_meters": "",
                        "validation_rmse_meters": "",
                        "validation_mae_meters": "",
                        "train_seconds": "",
                        "inference_seconds": "",
                        "wall_seconds": time.time() - started,
                        "best_epoch": "",
                        "stopped_early": "",
                        "checkpoint": "",
                        "notes": str(exc),
                    }
                )
            rows.append(row)
            model_rows.append(row)
        ok_rows = [row for row in model_rows if row.get("status") == "ok"]
        if ok_rows:
            best = min(ok_rows, key=lambda row: float(row["validation_ade_meters"]))
            selected[model_name] = {
                "config_index": best["config_index"],
                "validation_ade_meters": best["validation_ade_meters"],
                "config": json.loads(str(best["config_json"])),
            }

    fieldnames = [
        "model",
        "config_index",
        "status",
        "validation_ade_meters",
        "validation_fde_meters",
        "validation_rmse_meters",
        "validation_mae_meters",
        "proxy_train_samples",
        "proxy_val_samples",
        "train_seconds",
        "inference_seconds",
        "wall_seconds",
        "best_epoch",
        "stopped_early",
        "checkpoint",
        "config_json",
        "notes",
    ]
    write_csv(output_dir / "neural_tuning_results.csv", rows, fieldnames)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "split_policy": args.split_policy,
        "selection_metric": tuning_config.get("selection_metric", "validation_ade_meters"),
        "protocol": tuning_config.get("protocol", "validation_set_proxy_search"),
        "models": models,
        "max_configs_per_model": args.max_configs_per_model,
        "proxy_train_samples": int(X_train.shape[0]),
        "proxy_val_samples": int(X_val.shape[0]),
        "search_space": search_spaces,
        "selected_configs": selected,
        "results_csv": str(output_dir / "neural_tuning_results.csv"),
        "scope_note": (
            "This artifact documents validation-set neural baseline sensitivity and early stopping. "
            "It is a proxy tuning run for reviewer transparency; final test claims remain tied to "
            "outputs/final_multiday/model_metrics.csv."
        ),
    }
    (output_dir / "neural_tuning_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Neural tuning protocol written to {output_dir / 'neural_tuning_protocol.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
