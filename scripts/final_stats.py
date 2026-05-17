#!/usr/bin/env python3
"""Generate statistical summaries from final per-sample errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def read_errors(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    by_model: dict[str, dict[int, dict[str, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            idx = int(row["sample_index"])
            by_model.setdefault(model, {})[idx] = {
                "ade_meters": float(row["ade_meters"]),
                "fde_meters": float(row["fde_meters"]),
                "rmse_meters": float(row["rmse_meters"]),
                "mae_meters": float(row["mae_meters"]),
            }
    return by_model


def confidence_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - confidence
    return (
        float(np.percentile(values, 100 * alpha / 2)),
        float(np.percentile(values, 100 * (1 - alpha / 2))),
    )


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    diff = b - a
    result: dict[str, Any] = {
        "mean_difference_b_minus_a": float(diff.mean()),
        "n": int(diff.size),
        "paired_t": None,
        "wilcoxon": None,
    }
    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(b, a)
        result["paired_t"] = {
            "statistic": float(t_stat),
            "p_value": float(p_value),
        }
        if diff.size > 0 and np.any(diff != 0):
            w_stat, w_p = stats.wilcoxon(diff)
            result["wilcoxon"] = {
                "statistic": float(w_stat),
                "p_value": float(w_p),
            }
    except Exception as exc:  # pragma: no cover - archives optional scipy failures
        result["error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", default="outputs/final/per_sample_errors.csv")
    parser.add_argument("--output", default="outputs/final/statistical_tests.json")
    parser.add_argument("--reference", default="constant_velocity")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    errors_path = Path(args.errors)
    if not errors_path.exists():
        raise FileNotFoundError(f"Per-sample error file missing: {errors_path}")

    by_model = read_errors(errors_path)
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(errors_path),
        "reference": args.reference,
        "confidence": args.confidence,
        "models": {},
        "pairwise_vs_reference": {},
        "multiple_testing_correction": None,
    }

    for model, sample_map in by_model.items():
        ades = np.array([row["ade_meters"] for _, row in sorted(sample_map.items())])
        fdes = np.array([row["fde_meters"] for _, row in sorted(sample_map.items())])
        ade_ci = confidence_interval(ades, args.confidence)
        fde_ci = confidence_interval(fdes, args.confidence)
        summary["models"][model] = {
            "n": int(ades.size),
            "ade_mean": float(ades.mean()) if ades.size else math.nan,
            "ade_median": float(np.median(ades)) if ades.size else math.nan,
            "ade_std": float(ades.std(ddof=1)) if ades.size > 1 else 0.0,
            "ade_ci_percentile": ade_ci,
            "fde_mean": float(fdes.mean()) if fdes.size else math.nan,
            "fde_ci_percentile": fde_ci,
        }

    if args.reference in by_model:
        reference_samples = by_model[args.reference]
        comparisons = 0
        for model, sample_map in by_model.items():
            if model == args.reference:
                continue
            common = sorted(set(reference_samples) & set(sample_map))
            if not common:
                continue
            ref_ade = np.array([reference_samples[idx]["ade_meters"] for idx in common])
            model_ade = np.array([sample_map[idx]["ade_meters"] for idx in common])
            summary["pairwise_vs_reference"][model] = paired_tests(ref_ade, model_ade)
            comparisons += 1
        if comparisons:
            summary["multiple_testing_correction"] = {
                "method": "bonferroni",
                "comparisons": comparisons,
                "alpha_0_05_corrected": 0.05 / comparisons,
            }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Statistical summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
