#!/usr/bin/env python3
"""Run final_train_eval twice and compare metric stability under the same seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["model"]: row for row in csv.DictReader(handle)}


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def relative_difference(a: float, b: float) -> float:
    denominator = max(1.0, abs(a), abs(b))
    return abs(a - b) / denominator


def run_once(config: dict[str, Any], run_dir: Path, python_bin: str) -> dict[str, dict[str, str]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = json.loads(json.dumps(config))
    config_copy["experiment"]["output_dir"] = str(run_dir)
    temp_config = run_dir / "config.json"
    temp_config.write_text(json.dumps(config_copy, indent=2), encoding="utf-8")
    subprocess.run(
        [python_bin, "scripts/final_train_eval.py", "--config", str(temp_config)],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return read_metrics(run_dir / "model_metrics.csv")


def compare_metrics(
    first: dict[str, dict[str, str]], second: dict[str, dict[str, str]], tolerance: float
) -> tuple[bool, list[dict[str, Any]]]:
    metric_keys = ["ade_meters", "fde_meters", "rmse_meters", "mae_meters"]
    comparisons: list[dict[str, Any]] = []
    ok = True
    for model_name in sorted(set(first) | set(second)):
        first_row = first.get(model_name, {})
        second_row = second.get(model_name, {})
        status_match = first_row.get("status") == second_row.get("status")
        model_ok = status_match
        metric_records = []
        for key in metric_keys:
            a = as_float(first_row.get(key, ""))
            b = as_float(second_row.get(key, ""))
            if a is None or b is None:
                same = a is None and b is None
                diff = None
            else:
                diff = relative_difference(a, b)
                same = math.isfinite(diff) and diff <= tolerance
            model_ok = model_ok and same
            metric_records.append({"metric": key, "first": a, "second": b, "relative_difference": diff, "passed": same})
        comparisons.append(
            {
                "model": model_name,
                "status_first": first_row.get("status"),
                "status_second": second_row.get("status"),
                "status_match": status_match,
                "metrics": metric_records,
                "passed": model_ok,
            }
        )
        ok = ok and model_ok
    return ok, comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_conservative.yaml")
    parser.add_argument("--output", default="outputs/final/reproducibility_check.json")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = read_json(Path(args.config))
    with tempfile.TemporaryDirectory(prefix="metric_stability_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        first = run_once(config, temp_dir / "run_a", args.python_bin)
        second = run_once(config, temp_dir / "run_b", args.python_bin)

    passed, comparisons = compare_metrics(first, second, args.tolerance)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "tolerance": args.tolerance,
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Metric stability report written to {output_path}: {report['status']}")
    if args.strict and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
