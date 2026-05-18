#!/usr/bin/env python3
"""Generate The Journal of Navigation submission-candidate artifacts.

The script intentionally keeps every numerical claim tied to the current
repository evidence pack. It does not run new experiments; it converts audited
outputs into a JON-style manuscript, figures, supplementary notes, cover letter,
submission checklist, and a machine-readable manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


AUTHOR_NAME = "Li Bo"
AUTHOR_AFFILIATION = "China Maritime Service Center, China"
AUTHOR_EMAIL = "li.bo@cmaritime.com.cn"
SHORT_TITLE = "AIS Benchmark for Risk Warning"
PUBLIC_REPO_URL = "https://github.com/TristanLib/ais"
PUBLIC_REPO_TAG = "jon-submission-v1.3"


MODEL_ORDER = [
    "kalman_filter_cv",
    "constant_velocity",
    "ridge_lstsq",
    "linear_lstsq",
    "gru_baseline",
    "lstm_baseline",
    "tcn_baseline",
    "transformer_baseline",
    "constant_acceleration",
]

PLOT_MODELS = [
    "kalman_filter_cv",
    "constant_velocity",
    "ridge_lstsq",
    "linear_lstsq",
    "gru_baseline",
    "lstm_baseline",
    "tcn_baseline",
    "transformer_baseline",
]

MODEL_LABELS = {
    "constant_acceleration": "CA",
    "constant_velocity": "CV",
    "gru_baseline": "GRU",
    "kalman_filter_cv": "Kalman-CV",
    "linear_lstsq": "OLS",
    "lstm_baseline": "LSTM",
    "ridge_lstsq": "Ridge",
    "tcn_baseline": "TCN",
    "transformer_baseline": "Transformer",
}

SPLIT_LABELS = {
    "temporal_test": "Temporal holdout",
    "vessel_disjoint_test": "Vessel-disjoint holdout",
}

COLORS = {
    "kalman_filter_cv": "#1b9e77",
    "constant_velocity": "#377eb8",
    "ridge_lstsq": "#984ea3",
    "linear_lstsq": "#ff7f00",
    "gru_baseline": "#a6cee3",
    "lstm_baseline": "#e41a1c",
    "tcn_baseline": "#f781bf",
    "transformer_baseline": "#999999",
    "constant_acceleration": "#d95f02",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 1) -> str:
    number = as_float(value)
    if number != number:
        return "NA"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    return f"{number:.{digits}f}"


def fmt3(value: Any) -> str:
    number = as_float(value)
    if number != number:
        return "NA"
    return f"{number:.3f}"


def lookup_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["split_policy"], row["model"]): row for row in rows}


def best_by_split(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        split = row.get("split_policy", "")
        if split not in best or as_float(row.get("ade_meters")) < as_float(best[split].get("ade_meters")):
            best[split] = row
    return best


def stats_for(stats: dict[str, Any], split: str, model: str) -> dict[str, Any]:
    return stats.get("splits", {}).get(split, {}).get("models", {}).get(model, {})


def model_sort_key(row: dict[str, str]) -> tuple[int, int]:
    split_index = 0 if row.get("split_policy") == "temporal_test" else 1
    try:
        model_index = MODEL_ORDER.index(row.get("model", ""))
    except ValueError:
        model_index = 999
    return split_index, model_index


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9'-]*\b", text))


def make_pipeline_figure(path: Path) -> None:
    labels = [
        ("NOAA historical\nAIS", "Raw files and counts\nMMSI, timestamps\nchecksums"),
        ("Cleaning and\nresampling", "One-minute grid\nWGS84 latitude\nand longitude"),
        ("Windowing", "30 min history\n15 min forecast"),
        ("Split\nprotocols", "Temporal holdout\nvessel-disjoint\nholdout"),
        ("Model\nfamilies", "Kinematic, Kalman\nlinear and neural\nbaselines"),
        ("Evaluation", "ADE/FDE/RMSE/MAE\nscenario slices\npaired tests"),
        ("Risk\nwarning", "CPA/TCPA metrics\nprecision, recall\nfalse alarms, misses"),
    ]
    fig, ax = plt.subplots(figsize=(15.8, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_positions = [0.085, 0.223, 0.361, 0.499, 0.637, 0.775, 0.913]
    y = 0.53
    box_w = 0.112
    box_h = 0.60
    for i, ((title, body), x) in enumerate(zip(labels, x_positions)):
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.028",
            linewidth=1.25,
            edgecolor="#253746",
            facecolor="#f7fbff" if i % 2 == 0 else "#edf7f2",
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y + 0.150,
            title,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#1f2d3d",
            linespacing=1.05,
        )
        ax.text(
            x,
            y - 0.120,
            body,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#394b59",
            linespacing=1.12,
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - box_w / 2 - 0.012, y),
                xytext=(x + box_w / 2 + 0.012, y),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#506070"),
            )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.06)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_model_performance_figure(rows: list[dict[str, str]], path: Path) -> None:
    table = lookup_rows(rows)
    splits = ["temporal_test", "vessel_disjoint_test"]
    models = PLOT_MODELS
    x = list(range(len(models)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, split in zip([-width / 2, width / 2], splits):
        values = [as_float(table[(split, model)]["ade_meters"]) for model in models]
        ax.bar([i + offset for i in x], values, width=width, label=SPLIT_LABELS[split], alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("ADE (m, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=25, ha="right")
    ax.set_title("Trajectory prediction error by split protocol")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_horizon_figure(rows: list[dict[str, str]], path: Path) -> None:
    models = ["kalman_filter_cv", "constant_velocity", "ridge_lstsq", "linear_lstsq", "gru_baseline"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["split_policy"], row["model"])
        grouped.setdefault(key, []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, split in zip(axes, ["temporal_test", "vessel_disjoint_test"]):
        for model in models:
            series = sorted(grouped.get((split, model), []), key=lambda r: int(r["horizon_step"]))
            if not series:
                continue
            ax.plot(
                [int(r["horizon_step"]) for r in series],
                [as_float(r["ade_meters"]) for r in series],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=MODEL_LABELS[model],
                color=COLORS[model],
            )
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("Forecast horizon step (minutes)")
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.set_yscale("log")
    axes[0].set_ylabel("ADE (m, log scale)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Error growth over the 15-minute prediction horizon", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_risk_figure(risk_metrics: dict[str, Any], path: Path) -> None:
    metrics = risk_metrics["metrics_by_model"]
    models = ["kalman_filter_cv", "constant_velocity", "linear_lstsq"]
    labels = [MODEL_LABELS[m] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [1.8, 1.0]})
    bar_metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("false_alarm_rate", "False alarm rate"),
        ("missed_warning_rate", "Missed warning rate"),
    ]
    width = 0.2
    x = list(range(len(models)))
    for i, (key, label) in enumerate(bar_metrics):
        axes[0].bar([j + (i - 1.5) * width for j in x], [metrics[m][key] for m in models], width, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("CPA/TCPA warning classification")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(labels, [metrics[m]["mean_abs_cpa_error_nmi"] for m in models], color=[COLORS[m] for m in models])
    axes[1].set_ylabel("Mean absolute CPA error (nmi)")
    axes[1].set_title("CPA error")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_slice_figure(rows: list[dict[str, str]], path: Path) -> None:
    models = ["kalman_filter_cv", "constant_velocity", "ridge_lstsq", "linear_lstsq"]
    split = "temporal_test"
    speed_groups = ["sog_0_2", "sog_2_8", "sog_8_15", "sog_15_50"]
    region_groups = ["east_gulf_coast", "west_coast", "hawaii_pacific", "other"]
    by_key = {(r["model"], r["split_policy"], r["group_type"], r["group"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, group_type, groups, title in [
        (axes[0], "speed_bin", speed_groups, "Speed slices"),
        (axes[1], "region", region_groups, "Regional slices"),
    ]:
        x = list(range(len(groups)))
        width = 0.18
        for i, model in enumerate(models):
            vals = [
                as_float(by_key.get((model, split, group_type, group), {}).get("ade_meters"))
                for group in groups
            ]
            ax.bar([j + (i - 1.5) * width for j in x], vals, width, label=MODEL_LABELS[model], color=COLORS[model])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    axes[0].set_ylabel("ADE (m, log scale)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Scenario-slice sensitivity on the temporal holdout", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    figures_dir: Path,
    model_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
    risk_metrics: dict[str, Any],
    risk_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pipeline_protocol": figures_dir / "jon_pipeline_protocol.png",
        "model_performance": figures_dir / "jon_model_performance.png",
        "horizon_degradation": figures_dir / "jon_horizon_degradation.png",
        "risk_warning_metrics": figures_dir / "jon_risk_warning_metrics.png",
        "scenario_slice_errors": figures_dir / "jon_scenario_slice_errors.png",
    }
    make_pipeline_figure(paths["pipeline_protocol"])
    make_model_performance_figure(model_rows, paths["model_performance"])
    make_horizon_figure(horizon_rows, paths["horizon_degradation"])
    make_risk_figure(risk_metrics, paths["risk_warning_metrics"])
    make_slice_figure(group_rows, paths["scenario_slice_errors"])
    source_case = risk_dir / "figures" / "risk_case_studies.png"
    if source_case.exists():
        case_path = figures_dir / "jon_risk_case_studies.png"
        shutil.copy2(source_case, case_path)
        paths["risk_case_studies"] = case_path
    return {key: str(value) for key, value in paths.items()}


def trajectory_table(rows: list[dict[str, str]], stats: dict[str, Any]) -> str:
    lines = [
        "| Split | Model | Mean ADE (m) | Median ADE (m) | 95% ADE interval (m) | FDE (m) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=model_sort_key):
        split = row["split_policy"]
        model = row["model"]
        model_stats = stats_for(stats, split, model)
        ci = model_stats.get("ade_ci_percentile", ["NA", "NA"])
        interval = f"{fmt(ci[0])} to {fmt(ci[1])}" if len(ci) == 2 else "NA"
        lines.append(
            "| {split} | {model} | {ade} | {median} | {interval} | {fde} |".format(
                split=SPLIT_LABELS.get(split, split),
                model=MODEL_LABELS.get(model, model),
                ade=fmt(row.get("ade_meters")),
                median=fmt(model_stats.get("ade_median")),
                interval=interval,
                fde=fmt(row.get("fde_meters")),
            )
        )
    return "\n".join(lines)


def risk_table(risk_metrics: dict[str, Any]) -> str:
    metrics = risk_metrics["metrics_by_model"]
    lines = [
        "| Model | TP | FP | FN | TN | Precision | Recall | False alarm | Missed warning | Lead-time error (min) | CPA error (nmi) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["kalman_filter_cv", "constant_velocity", "linear_lstsq"]:
        row = metrics[model]
        lines.append(
            "| {model} | {tp} | {fp} | {fn} | {tn} | {precision} | {recall} | {far} | {miss} | {lead} | {cpa} |".format(
                model=MODEL_LABELS[model],
                tp=row["true_positive"],
                fp=row["false_positive"],
                fn=row["false_negative"],
                tn=row["true_negative"],
                precision=fmt3(row["precision"]),
                recall=fmt3(row["recall"]),
                far=fmt3(row["false_alarm_rate"]),
                miss=fmt3(row["missed_warning_rate"]),
                lead=fmt3(row["mean_abs_lead_time_error_minutes"]),
                cpa=fmt3(row["mean_abs_cpa_error_nmi"]),
            )
        )
    return "\n".join(lines)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def trimmed_mean(values: list[float], trim_fraction: float = 0.01) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    cut = int(len(ordered) * trim_fraction)
    trimmed = ordered[cut : len(ordered) - cut] if cut else ordered
    return sum(trimmed) / len(trimmed)


def tail_diagnostics_table(per_sample_rows: list[dict[str, str]]) -> str:
    selected = [
        ("temporal_test", "kalman_filter_cv"),
        ("temporal_test", "constant_velocity"),
        ("temporal_test", "ridge_lstsq"),
        ("temporal_test", "linear_lstsq"),
        ("vessel_disjoint_test", "kalman_filter_cv"),
        ("vessel_disjoint_test", "constant_velocity"),
        ("vessel_disjoint_test", "ridge_lstsq"),
        ("vessel_disjoint_test", "linear_lstsq"),
    ]
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in per_sample_rows:
        key = (row["split_policy"], row["model"])
        if key in selected:
            grouped.setdefault(key, []).append(as_float(row["ade_meters"]))
    lines = [
        "| Split | Model | P50 ADE | P90 ADE | P95 ADE | P99 ADE | Mean ADE | 1% trimmed mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, model in selected:
        values = grouped.get((split, model), [])
        lines.append(
            "| {split} | {model} | {p50} | {p90} | {p95} | {p99} | {mean} | {trimmed} |".format(
                split=SPLIT_LABELS.get(split, split),
                model=MODEL_LABELS.get(model, model),
                p50=fmt(percentile(values, 0.50)),
                p90=fmt(percentile(values, 0.90)),
                p95=fmt(percentile(values, 0.95)),
                p99=fmt(percentile(values, 0.99)),
                mean=fmt(sum(values) / len(values) if values else float("nan")),
                trimmed=fmt(trimmed_mean(values)),
            )
        )
    return "\n".join(lines)


def compact_group_summary(group_rows: list[dict[str, str]], group_type: str) -> str:
    rows = [
        row
        for row in group_rows
        if row["model"] == "constant_velocity"
        and row["split_policy"] == "temporal_test"
        and row["group_type"] == group_type
    ]
    if group_type == "speed_bin":
        order = {"sog_0_2": 0, "sog_2_8": 1, "sog_8_15": 2, "sog_15_50": 3}
        label = {
            "sog_0_2": "0-2 kn",
            "sog_2_8": "2-8 kn",
            "sog_8_15": "8-15 kn",
            "sog_15_50": "15-50 kn",
        }
        rows = sorted(rows, key=lambda row: order.get(row["group"], 99))
    else:
        label = {}
        rows = sorted(rows, key=lambda row: row["group"])
    total = sum(int(row["n"]) for row in rows)
    return "; ".join(
        f"{label.get(row['group'], row['group'])} {int(row['n']):,} ({100 * int(row['n']) / total:.1f}%)"
        for row in rows
    )


def risk_scenario_summary(risk_scenario_rows: list[dict[str, str]]) -> str:
    rows = [row for row in risk_scenario_rows if row["model"] == "constant_velocity"]
    if not rows:
        return "not available"
    warning_count = sum(1 for row in rows if row["truth_warning"] == "True")
    regions: dict[str, int] = {}
    for row in rows:
        key = row["region_a"] if row["region_a"] == row["region_b"] else "mixed"
        regions[key] = regions.get(key, 0) + 1
    region_text = ", ".join(f"{key} {value}" for key, value in sorted(regions.items(), key=lambda item: (-item[1], item[0])))
    cpa = [as_float(row["truth_min_cpa_nmi"]) for row in rows]
    tcpa = [as_float(row["truth_tcpa_minutes"]) for row in rows]
    return (
        f"{len(rows):,} scenarios; {warning_count:,} truth warnings; "
        f"median observed CPA {percentile(cpa, 0.50):.3f} nmi; "
        f"median observed TCPA {percentile(tcpa, 0.50):.1f} min; regions: {region_text}"
    )


def data_quality_table(
    dataset: dict[str, Any],
    group_rows: list[dict[str, str]],
    risk_scenario_rows: list[dict[str, str]],
) -> str:
    lines = [
        "| Item | Value | Interpretation |",
        "|---|---:|---|",
        f"| Source dates | {len(dataset['source_dates'])} | Multi-date historical AIS sample; not a seasonal or global coverage claim. |",
        f"| Trajectory windows | {dataset['sample_count']:,} | Adequate scale for a benchmark-style evaluation. |",
        f"| Unique MMSI values | {dataset['unique_mmsi_count']:,} | Broad vessel-identity coverage for vessel-disjoint testing. |",
        f"| Mean SOG | {fmt(dataset['avg_sog_knots']['mean'], 2)} kn; max {fmt(dataset['avg_sog_knots']['max'], 2)} kn | Low-speed traffic is prominent, so aggregate metrics are interpreted with slice results. |",
        f"| Temporal-test speed bins | {compact_group_summary(group_rows, 'speed_bin')} | The temporal test set is low-speed dominated. |",
        f"| Temporal-test regions | {compact_group_summary(group_rows, 'region')} | Region composition is not balanced; regional robustness is not claimed. |",
        f"| Interpolation ratio | mean {fmt(dataset['interpolation_ratio']['mean'], 3)}; max {fmt(dataset['interpolation_ratio']['max'], 3)} | Interpolation is moderate and remains a possible contributor to tail errors. |",
        f"| Turn intensity | mean {fmt(dataset['turn_intensity_deg']['mean'], 2)} deg; max {fmt(dataset['turn_intensity_deg']['max'], 2)} deg | Manoeuvring heterogeneity supports horizon and slice analysis. |",
        f"| Risk-warning scenarios | {risk_scenario_summary(risk_scenario_rows)} | Decision-support slice, not a full encounter-population model. |",
    ]
    return "\n".join(lines)


def neural_config_table(neural_protocol: dict[str, Any]) -> str:
    selected = neural_protocol.get("selected_configs", {})
    lines = [
        "| Model | Selected architecture/configuration | Optimizer | Learning rate | Batch | Epochs | Early stopping | Proxy validation ADE (m) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in ["gru_baseline", "lstm_baseline", "tcn_baseline", "transformer_baseline"]:
        entry = selected.get(model, {})
        config = entry.get("config", {})
        architecture_keys = [
            key
            for key in config
            if key not in {"learning_rate", "batch_size", "epochs", "early_stopping_patience"}
        ]
        architecture = ", ".join(f"{key}={config[key]}" for key in sorted(architecture_keys))
        lines.append(
            "| {model} | {architecture} | Adam | {lr} | {batch} | {epochs} | {patience} | {val} |".format(
                model=MODEL_LABELS.get(model, model),
                architecture=architecture,
                lr=config.get("learning_rate", "NA"),
                batch=config.get("batch_size", "NA"),
                epochs=config.get("epochs", "NA"),
                patience=config.get("early_stopping_patience", "NA"),
                val=fmt(entry.get("validation_ade_meters")),
            )
        )
    return "\n".join(lines)


def make_references() -> str:
    return """Bai, S., Kolter, J. Z. and Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv:1803.01271. https://arxiv.org/abs/1803.01271.

Bergstra, J. and Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 1724-1734. doi:10.3115/v1/D14-1179.

Dalsnes, B. R., Hexeberg, S., Flaten, A. L., Eriksen, B. O. H. and Brekke, E. F. (2018). The Neighbor Course Distribution Method with Gaussian Mixture Models for AIS-Based Vessel Trajectory Prediction. Proceedings of the 21st International Conference on Information Fusion. doi:10.23919/ICIF.2018.8455607.

Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.

Endsley, M. R. (1995). Toward a Theory of Situation Awareness in Dynamic Systems. Human Factors, 37, 32-64. doi:10.1518/001872095779049543.

Fujii, Y. and Tanaka, K. (1971). Traffic Capacity. The Journal of Navigation, 24, 543-552. doi:10.1017/S0373463300022384.

Goodwin, E. M. (1975). A Statistical Study of Ship Domains. The Journal of Navigation, 28, 328-344. doi:10.1017/S0373463300041230.

Hansen, M. G., Jensen, T. K., Lehn-Schioler, T., Melchild, K., Rasmussen, F. M. and Ennemark, F. (2013). Empirical Ship Domain based on AIS Data. The Journal of Navigation, 66, 931-940. doi:10.1017/S0373463313000489.

Hexeberg, S., Flaten, A. L., Eriksen, B. O. H. and Brekke, E. F. (2017). AIS-Based Vessel Trajectory Prediction. Proceedings of the 20th International Conference on Information Fusion. doi:10.23919/ICIF.2017.8009762.

Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9, 1735-1780. doi:10.1162/neco.1997.9.8.1735.

International Maritime Organization. (1972). Convention on the International Regulations for Preventing Collisions at Sea, 1972 (COLREGs). https://www.imo.org/en/About/Conventions/Pages/COLREG.aspx. Accessed 16 May 2026.

Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering, 82, 35-45. doi:10.1115/1.3662552.

Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. Proceedings of the International Conference on Learning Representations. https://arxiv.org/abs/1412.6980.

Leek, J. T. and Peng, R. D. (2015). Reproducible Research Can Still Be Wrong: Adopting a Prevention Approach. Proceedings of the National Academy of Sciences, 112, 1645-1646. doi:10.1073/pnas.1421412111.

MarineCadastre.gov. (2024). AIS Data. https://marinecadastre.gov/ais/. Accessed 16 May 2026.

Millefiori, L. M., Braca, P., Bryan, K. and Willett, P. (2016). Modeling Vessel Kinematics using a Stochastic Mean-Reverting Process for Long-Term Prediction. IEEE Transactions on Aerospace and Electronic Systems, 52, 2313-2330. doi:10.1109/TAES.2016.150596.

Mou, J. M., van der Tak, C. and Ligteringen, H. (2010). Study on Collision Avoidance in Busy Waterways by using AIS Data. Ocean Engineering, 37, 483-490. doi:10.1016/j.oceaneng.2010.01.012.

NOAA Office for Coastal Management. (2026). Marine Cadastre. https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html. Accessed 16 May 2026.

Pallotta, G., Vespe, M. and Bryan, K. (2013). Vessel Pattern Knowledge Discovery from AIS Data: A Framework for Anomaly Detection and Route Prediction. Entropy, 15, 2218-2245. doi:10.3390/e15062218.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and others. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and others. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://www.jmlr.org/papers/v12/pedregosa11a.html.

Pineau, J., Vincent-Lamarre, P., Sinha, K., Lariviere, V., Beygelzimer, A., d'Alche-Buc, F., Fox, E. and Larochelle, H. (2021). Improving Reproducibility in Machine Learning Research. Journal of Machine Learning Research, 22, 1-20. https://www.jmlr.org/papers/v22/20-303.html.

Ristic, B., La Scala, B., Morelande, M. and Gordon, N. (2008). Statistical Analysis of Motion Patterns in AIS Data: Anomaly Detection and Motion Prediction. Proceedings of the 11th International Conference on Information Fusion. doi:10.1109/ICIF.2008.4632190.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F. and Dennison, D. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems, 28, 2503-2511.

Statheros, T., Howells, G. and Maier, K. M. (2008). Autonomous Ship Collision Avoidance Navigation Concepts, Technologies and Techniques. The Journal of Navigation, 61, 129-142. doi:10.1017/S037346330700447X.

Szlapczynski, R. and Szlapczynska, J. (2017). Review of Ship Safety Domains: Models and Applications. Ocean Engineering, 145, 277-289. doi:10.1016/j.oceaneng.2017.09.020.

Tu, E., Zhang, G., Rachmawati, L., Rajabally, E. and Huang, G. B. (2018). Exploiting AIS Data for Intelligent Maritime Navigation: A Comprehensive Survey From Data to Methodology. IEEE Transactions on Intelligent Transportation Systems, 19, 1559-1582. doi:10.1109/TITS.2017.2724551.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. and Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30, 5998-6008. https://arxiv.org/abs/1706.03762.

Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L. and Teal, T. K. (2017). Good Enough Practices in Scientific Computing. PLOS Computational Biology, 13, e1005510. doi:10.1371/journal.pcbi.1005510."""


def make_manuscript(
    generated_at: str,
    data_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    readiness: dict[str, Any],
    model_rows: list[dict[str, str]],
    stats: dict[str, Any],
    risk_metrics: dict[str, Any],
    group_rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, str]],
    risk_scenario_rows: list[dict[str, str]],
) -> str:
    dataset = data_manifest["dataset_summary"]
    config = run_manifest.get("config", {}).get("experiment", {})
    best = best_by_split(model_rows)
    temporal_best = best.get("temporal_test", {})
    vessel_best = best.get("vessel_disjoint_test", {})
    risk_generation = risk_metrics["scenario_generation"]
    kalman_risk = risk_metrics["metrics_by_model"]["kalman_filter_cv"]
    cv_risk = risk_metrics["metrics_by_model"]["constant_velocity"]
    split_counts = dataset["temporal_split_counts"]
    vessel_counts = dataset["vessel_split_counts"]
    regions = ", ".join(dataset["regions"])
    dates = ", ".join(dataset["source_dates"])
    model_lookup = lookup_rows(model_rows)
    cv_temporal = model_lookup[("temporal_test", "constant_velocity")]
    cv_vessel = model_lookup[("vessel_disjoint_test", "constant_velocity")]
    ridge_vessel = model_lookup[("vessel_disjoint_test", "ridge_lstsq")]
    neural_temporal = model_lookup[("temporal_test", "transformer_baseline")]
    trajectory_md = trajectory_table(model_rows, stats)
    data_quality_md = data_quality_table(dataset, group_rows, risk_scenario_rows)
    tail_md = tail_diagnostics_table(per_sample_rows)
    risk_md = risk_table(risk_metrics)

    return f"""# A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support

Article type: Research Article

Short title: {SHORT_TITLE}

Author: {AUTHOR_NAME}

Affiliation: {AUTHOR_AFFILIATION}

Corresponding author: {AUTHOR_NAME}, {AUTHOR_EMAIL}

## ABSTRACT

Short-horizon AIS trajectory prediction is often framed as a machine-learning problem, but navigation use also requires strong baselines, reproducible splits and downstream warning evidence. This study reports an auditable benchmark built from NOAA historical AIS data on four 2024 dates, containing {dataset['sample_count']:,} trajectory windows from {dataset['unique_mmsi_count']:,} MMSI values. Each sample uses {config.get('history_steps', 30)} one-minute history points to forecast {config.get('forecast_steps', 15)} one-minute positions. Kinematic, Kalman-style, linear and neural baselines are evaluated with Haversine ADE/FDE and local-component RMSE/MAE, followed by CPA/TCPA risk-warning metrics. Kalman-CV gives the best mean ADE in the reported benchmark run: {fmt(temporal_best.get('ade_meters'))} m on the temporal holdout and {fmt(vessel_best.get('ade_meters'))} m on the vessel-disjoint holdout. Under the documented baseline configurations, the neural sequence baselines do not outperform the strongest motion baselines. In {risk_generation['scenario_count']:,} AIS-derived encounter scenarios, Kalman-CV reaches precision {fmt3(kalman_risk['precision'])} and recall {fmt3(kalman_risk['recall'])}. The contribution is a reproducible navigation benchmark, not autonomous collision-avoidance validation.

## 1. INTRODUCTION

AIS trajectory prediction has become a familiar component in maritime traffic analysis, port monitoring, route inference and collision-risk assessment. The availability of large historical AIS archives has also encouraged increasingly complex forecasting models. The difficulty is that better model architecture alone does not automatically produce better navigation evidence. A forecast that appears accurate under one split can fail when evaluated on vessels not seen during training, and an average position error can obscure whether a risk-warning system produces more missed warnings or more false alarms. Navigation research therefore needs protocols that connect data provenance, baseline strength, forecast metrics and downstream warning behaviour.

The core problem addressed in this paper is methodological rather than purely architectural. Many AIS prediction studies compare proposed models with weak or inconsistently implemented baselines, use split definitions that are difficult to reproduce, or stop the evaluation at ADE and FDE. Those metrics are useful, but they are not the end of the navigation question. A shipboard or shore-based decision-support layer must also consider the closest point of approach, the time to closest approach, warning thresholds, false alarms and missed warnings. A trajectory model that reduces an average error by a small margin may not improve a CPA/TCPA warning, while a model with a modest mean ADE can still be valuable if its warning behaviour is stable and interpretable.

For navigation readers, AIS prediction is best understood as a support layer for maritime situational awareness, not as a stand-alone deep-learning leaderboard. This framing leads to three research questions. RQ1 asks how simple kinematic, statistical and neural baselines compare under an audited short-horizon AIS protocol. RQ2 asks whether model rankings remain stable under temporal and vessel-disjoint holdouts. RQ3 asks how trajectory-prediction differences affect CPA/TCPA warning precision, recall, false alarms and missed warnings. These questions are deliberately conservative because they map to the evidence available in the audited data and result files.

The navigation context also changes the burden of proof. In a generic time-series benchmark, it may be sufficient to show that one model has a lower average error than another. In a maritime setting, the user of a prediction layer may be a watch officer, a vessel-traffic-service operator, a shore-based monitoring analyst or an automated advisory module. These users do not only need a point forecast. They need to know when a forecast can be trusted, when it is likely to produce nuisance alarms, and when it might miss a close encounter. This is why the paper treats the data protocol, split design and risk-warning evaluation as part of the same contribution.

The first contribution is a reproducible AIS benchmark protocol. It records raw-file checksums, a processed manifest, temporal split labels and vessel-disjoint split labels for a short-horizon forecasting task. The processed artefact keeps both temporal and vessel-disjoint labels, so the same data build supports two complementary forms of generalisation testing. This is important because temporal holdouts test later windows from the same broad traffic distribution, while vessel-disjoint holdouts ask whether the model transfers to MMSI values not used for training.

The second contribution is a baseline-centred empirical finding. In the reported benchmark run, the Kalman-style constant-velocity baseline is the best mean-ADE model in both holdouts, and the ordinary constant-velocity baseline remains a strong reference. Ridge regression is close to Kalman-CV on the vessel-disjoint holdout ({fmt(ridge_vessel.get('ade_meters'))} m ADE), while the documented neural sequence baselines remain behind the kinematic and linear baselines. This result should not be read as a universal claim that neural models are unsuitable for maritime prediction. It is a cautionary, reproducible result: architecture-superiority claims require stronger tuning, preprocessing, split discipline and external validation than a single reported run.

The third contribution is a navigation-facing warning evaluation. It translates selected trajectory pairs into CPA/TCPA warning classifications and reports precision, recall, false alarms, missed warnings, lead-time error and CPA error. The labels are derived from observed future separation inside the forecast horizon, so the analysis remains a historical decision-support evaluation rather than a closed-loop collision-avoidance simulation.

This conservative framing is also intended to make the work useful even if a reader disagrees with the specific baseline set. The evidence chain is reusable: new models can be added, alternative warning thresholds can be tested, and later AIS source dates can be processed without rewriting the manuscript logic. The workflow therefore contributes a reproducible evaluation pattern as well as a numerical result. The result reported here is one state of that workflow, not a claim that the model list is final.

The remainder of the paper is organised as follows. Section 2 reviews related work on AIS prediction, simple baselines, neural sequence models, CPA/TCPA risk assessment and reproducible evaluation. Section 3 describes the data and the reproducible protocol. Section 4 defines the model families and metrics. Section 5 explains the risk-warning evaluation design. Section 6 reports trajectory and warning results. Section 7 discusses navigation implications. Sections 8 and 9 state limitations and conclusions.

## 2. RELATED WORK

AIS data have been used for route discovery, anomaly detection, traffic characterisation and short-term vessel prediction. A recurring theme in this literature is that maritime trajectories are highly structured by geography, navigation rules, traffic lanes, port approaches and operational constraints, but they are also noisy because AIS reception, reporting intervals and vessel manoeuvres are irregular. Pallotta et al. (2013) framed AIS pattern discovery as a way to support anomaly detection and route prediction. Tu et al. (2018) reviewed AIS analytics across data handling and methodology, highlighting the breadth of tasks that depend on reliable AIS preprocessing. Ristic et al. (2008), Hexeberg et al. (2017), Dalsnes et al. (2018) and Millefiori et al. (2016) illustrate the range of probabilistic and statistical approaches used before recent deep-learning enthusiasm.

The same literature also shows why AIS prediction cannot be separated from data engineering. AIS messages can be sparse, duplicated, delayed or spatially inconsistent, and the same physical vessel can produce many overlapping windows. A benchmark that hides these choices behind a preprocessed array is difficult to interpret. For that reason, the present work records the sampling choices and preserves metadata needed for temporal and vessel-disjoint evaluation. This is not only a reproducibility convenience; it is a defence against overoptimistic generalisation claims.

For short horizons, simple motion assumptions remain difficult to dismiss. Constant velocity, constant acceleration and Kalman-style filtering are not merely straw men; they encode the fact that many vessels maintain course and speed for short intervals, especially when the forecast horizon is measured in minutes. Kalman (1960) remains foundational for recursive filtering, and navigation applications often benefit from the interpretability and computational economy of such filters. The present benchmark therefore treats kinematic baselines as first-class models. A proposed model that does not outperform them under well-defined splits should not be described as a navigation advance without a specific explanation of where its value lies.

This baseline stance follows a broader principle in safety-related prediction. A complex model should earn its complexity by providing a measurable advantage under a relevant operational criterion. That advantage might be lower FDE, better stability during manoeuvres, fewer missed warnings, calibrated uncertainty or robustness to vessel-disjoint testing. If the advantage is absent, the simpler model may still be preferable because it is faster, easier to audit and easier to explain. In this study, the strongest reported result comes from such a simple model family.

Neural sequence models provide flexible function approximation and can learn nonlinear temporal patterns, but they are sensitive to representation, normalisation, training regime and data split. LSTM networks (Hochreiter and Schmidhuber, 1997), GRU networks (Cho et al., 2014), temporal convolutional networks (Bai et al., 2018) and Transformers (Vaswani et al., 2017) are all reasonable baseline families for sequence forecasting. Their inclusion in this benchmark does not imply that the particular shallow configurations used here exhaust the potential of neural AIS prediction. Instead, they serve as documented baselines that expose a practical issue: neural models require careful protocol, scaling and training-budget choices before they can support navigation-facing superiority claims.

Collision-risk analysis adds another layer. Closest point of approach and time to closest approach are standard quantities for encounter assessment, and ship-domain studies provide additional geometric context. Goodwin (1975), Fujii and Tanaka (1971), Hansen et al. (2013), Mou et al. (2010), Statheros et al. (2008), and Szlapczynski and Szlapczynska (2017) show that navigation risk is shaped by encounter geometry, traffic density, domain assumptions and decision rules. The present study does not validate a complete collision-avoidance controller. It uses CPA/TCPA warning metrics to ask whether differences in trajectory forecasts matter for a simple decision-support warning layer.

The distinction between warning support and avoidance control is essential. A warning system can be evaluated from historical AIS by asking whether the predicted encounter geometry would have raised an alert. A collision-avoidance controller requires a much richer experimental setting, including own-ship dynamics, target-ship response, COLREGs interpretation, human supervision and failure handling. The present paper stays on the former side of that boundary. This makes the experiment narrower, but also makes the claims more defensible.

Reproducibility is central because AIS studies can be surprisingly difficult to compare. Small changes in cleaning rules, interpolation, sample slicing, vessel filtering, split construction or distance metrics can alter results. Scientific-computing recommendations such as Wilson et al. (2017) and reproducibility discussions such as Leek and Peng (2015) and Pineau et al. (2021) motivate the artefact-first design used here. Statistical-comparison guidance from Demsar (2006) also supports paired evaluation and transparent reporting rather than a single aggregate number. The benchmark therefore stores generated evidence as CSV and JSON artefacts and regenerates manuscript tables from those artefacts.

The related-work gap addressed by this paper is therefore not the absence of AIS prediction models. It is the absence, in many model-centred papers, of a compact evidence chain that a navigation reviewer can audit from raw data manifest to risk-warning metric. By combining a conservative model comparison with downstream warning analysis, this work tries to make model evaluation legible to both machine-learning and navigation-safety readers.

## 3. DATA AND REPRODUCIBLE PROTOCOL

The data source is the NOAA MarineCadastre.gov AIS archive (MarineCadastre.gov, 2024; NOAA Office for Coastal Management, 2026). The processed artefact covers four source dates: {dates}. The data manifest records {dataset['sample_count']:,} trajectory windows, {dataset['unique_mmsi_count']:,} unique MMSI values, and a time range from {dataset['time_range']['start']} to {dataset['time_range']['end']}. The region labels represented in the processed build are {regions}. The mean speed over ground recorded in the processed sample metadata is {fmt(dataset['avg_sog_knots']['mean'], 2)} knots, with a maximum of {fmt(dataset['avg_sog_knots']['max'], 2)} knots. The mean turn-intensity metadata value is {fmt(dataset['turn_intensity_deg']['mean'], 2)} degrees, and the mean interpolation ratio is {fmt(dataset['interpolation_ratio']['mean'], 3)}.

Each example consists of a fixed-length historical sequence and a fixed-length future sequence. The protocol uses {config.get('history_steps', 30)} input steps and {config.get('forecast_steps', 15)} forecast steps, with one-minute spacing after resampling. The processed coordinates remain WGS84 latitude and longitude. ADE and FDE are computed as Haversine distances in metres. RMSE and MAE are computed on local north/east component errors, also in metres. This avoids confusing angular degrees with metric displacement, a common source of inflated or misleading claims in geospatial prediction.

The one-minute grid is a compromise between AIS reporting irregularity and navigational interpretability. It is short enough to support 15-minute risk-warning analysis and long enough to reduce the influence of individual message jitter. The window design also keeps the task deliberately local: the model is asked to forecast near-future motion, not to infer an entire voyage plan. This is why simple baselines are expected to be strong and why failure to beat them is informative rather than surprising.

The temporal split uses {split_counts['train']:,} training samples, {split_counts['val']:,} validation samples and {split_counts['test']:,} test samples. The vessel-disjoint split uses {vessel_counts['train']:,} training samples, {vessel_counts['val']:,} validation samples and {vessel_counts['test']:,} test samples. The temporal split evaluates future time blocks, while the vessel-disjoint split holds out MMSI values from training. Neither split should be interpreted as all-day seasonal validation or live AIS deployment. The split design is stronger than a single random split, but it is still a historical time-block protocol.

The vessel-disjoint split is particularly important because AIS windows from the same vessel are not independent in a behavioural sense. If a model sees many windows from a vessel during training, it may partly learn that vessel's typical operating area or motion regime. Holding out MMSI values reduces this leakage and creates a more demanding test. It is still not perfect, because vessels can share routes and regions, but it moves the benchmark closer to the way a deployed system would encounter previously unseen targets.

### Data quality and representativeness

The dataset is suitable for a reproducible benchmark, but it should not be read as a balanced survey of all maritime operating conditions. Table 1 summarises the main representativeness checks available from the processed metadata and the risk-warning scenario file. The low mean speed and the temporal-test speed-bin composition indicate that many windows are slow-moving, near-shore or port-influenced. This does not invalidate a short-horizon benchmark, but it does mean that aggregate ADE must be interpreted together with speed, region and horizon summaries.

Table 1. Data representativeness and quality summary.

{data_quality_md}

Figure 1 summarises the evidence chain from raw AIS to risk-warning metrics.

![Figure 1. Reproducible evidence chain used in this study.](figures/jon_pipeline_protocol.png)

The analysis is designed so that manuscript tables and figures remain tied to machine-readable evidence rather than hidden spreadsheet calculations. The main tables and figures are regenerated from the public release's data manifest, model-metric tables, horizon and group summaries, statistical summaries and risk-warning metrics. The supplementary file lists the machine-readable evidence files used for this manuscript, while the main text reports only the evidence needed to understand and evaluate the navigation claim.

A practical benefit of this design is that future revisions can be regenerated after new data or models are added. If a reviewer asks for another time block, a different risk threshold or an additional baseline, the project can update the evidence files and regenerate the manuscript tables. This reduces the chance that a paper, a figure and a code output drift apart during revision.

## 4. TRAJECTORY PREDICTION MODELS AND METRICS

The benchmark contains nine model entries. Constant velocity extrapolates future positions from the latest short-term motion. Constant acceleration extends that assumption by estimating acceleration from the recent sequence. The Kalman-style constant-velocity model applies a filtering/smoothing view of the same physical assumption. Ordinary least squares and ridge least squares map the flattened history window to future displacements. LSTM, GRU, Transformer and TCN entries provide neural sequence baselines implemented with PyTorch (Paszke et al., 2019). Linear tooling uses the scientific Python stack, including scikit-learn where appropriate (Pedregosa et al., 2011).

The model set is intentionally mixed. It is not a catalogue of every possible maritime predictor, and it is not designed to make a universal neural-model claim. It is designed to answer whether common baseline families survive a reproducible, metadata-rich, short-horizon AIS protocol. This is why the main comparison includes simple baselines even when they are not novel. In safety-related navigation research, a non-novel baseline can still be a novel contribution if it prevents unsupported claims about more complex systems.

The primary trajectory metrics are average displacement error, final displacement error, RMSE and MAE. ADE is the mean Haversine distance between predicted and observed positions over the forecast horizon. FDE is the Haversine distance at the final forecast step. RMSE and MAE provide component-level local displacement summaries. The model ranking in the text is based on mean ADE, while the tables also report median ADE and empirical 95 percent ADE intervals to show the skew caused by difficult or anomalous windows.

Mean ADE is useful for ranking, but it is not sufficient for navigation interpretation. A model can have a low median error and a high mean error if most windows are easy and a small number have very large errors. Such upper-tail cases matter because they may correspond to manoeuvres, low-speed drift, port approaches, sparse reporting or unusual regional geometry. The manuscript therefore reports mean, median and empirical intervals together. This is a deliberate guard against presenting a single attractive number as the full story.

Paired comparisons are computed against constant velocity because CV is the main operationally meaningful reference. The statistical artefact includes paired t-tests, Wilcoxon tests and Bonferroni-corrected comparison counts. The paper does not overstate these tests. Because AIS errors are highly skewed, the mean and the median tell different stories. In particular, Kalman-CV has very low median ADE in both holdouts but still has large upper-tail errors, which explains why mean ADE remains in kilometres even though the median error is measured in metres.

The modelling protocol also records neural tuning separately from final test claims. The neural proxy search provides reviewer transparency about the limited validation configurations explored, but final conclusions remain tied to the full-split test metrics. This separation matters because using test performance to tune neural models would undermine the benchmark. The present neural results are therefore best read as baseline outcomes under documented settings, not as a final word on neural maritime forecasting.

## 5. GENERALISATION AND RISK-WARNING EVALUATION

Generalisation is evaluated using both temporal and vessel-disjoint holdouts. Temporal holdout is useful because a deployed predictor normally faces later windows than those used during training. Vessel-disjoint holdout is useful because it asks whether a model depends on learning idiosyncratic behaviour from vessels seen during training. The two tests are complementary; a model that looks acceptable under temporal evaluation may not transfer to unseen MMSI values.

For downstream risk-warning evaluation, the benchmark selects AIS-derived encounter scenarios from the temporal test setting. The scenario builder scans the first {risk_generation['evaluated_samples']:,} temporal-test samples, groups windows that share the same start time, and pairs vessels whose origin positions are within {risk_generation['search_radius_nmi']} nautical miles. The reported scenarios are paired-vessel cases rather than single-trajectory examples. For each pair, the observed future trajectories define the truth warning label: a warning is positive if the discrete minimum separation over the 15 one-minute future positions is no greater than {risk_generation['warning_threshold_nmi']} nautical miles. Predicted trajectories are processed with the same discrete CPA/TCPA calculation to obtain model-level warning labels.

The discrete CPA/TCPA calculation deliberately matches the sampled forecast horizon rather than a continuous relative-motion formula. For each scenario and model, the comparison yields true positives, false positives, false negatives and true negatives. Lead-time error is computed when both observed and predicted futures cross the warning threshold, and CPA error is the absolute difference between predicted and observed minimum separation in nautical miles. The warning metrics are precision, recall, false-alarm rate, missed-warning rate, mean absolute lead-time error and mean absolute CPA error.

This design allows the same trajectory outputs to be judged in a way that resembles operational decision support. Precision reflects the credibility of raised warnings, while recall reflects the ability to capture true close-approach cases. False-alarm rate is important because excessive alarms can reduce user trust, and missed-warning rate is important because missed close approaches are safety critical. CPA error adds a continuous measure that is easier to interpret in nautical miles.

The risk-warning evaluation uses {risk_generation['scenario_count']:,} scenarios from {risk_generation['evaluated_samples']:,} evaluated samples. The search radius is {risk_generation['search_radius_nmi']} nautical miles, the warning threshold is {risk_generation['warning_threshold_nmi']} nautical miles and the truth warning count is {risk_generation['truth_warning_count']:,}. These choices produce an interpretable operational slice, but they are not a certification criterion. They should be read as a reproducible decision-support experiment.

The warning threshold is intentionally fixed rather than optimised per model. Optimising a different threshold for every model could improve individual scores, but it would make the comparison less transparent. A fixed threshold gives reviewers a stable basis for interpreting the confusion matrices. Alternative thresholds are a reasonable future sensitivity study and can be added without changing the rest of the evidence pipeline.

## 6. RESULTS

Figure 2 reports the main trajectory-performance comparison. Kalman-CV is the best mean-ADE model in both split protocols. It achieves {fmt(temporal_best.get('ade_meters'))} m ADE and {fmt(temporal_best.get('fde_meters'))} m FDE on the temporal holdout, and {fmt(vessel_best.get('ade_meters'))} m ADE and {fmt(vessel_best.get('fde_meters'))} m FDE on the vessel-disjoint holdout. Constant velocity reaches {fmt(cv_temporal.get('ade_meters'))} m temporal ADE and {fmt(cv_vessel.get('ade_meters'))} m vessel-disjoint ADE. The gap between CV and Kalman-CV is larger under vessel-disjoint evaluation, suggesting that filtered motion estimates provide robustness when specific vessel identities are not seen during training.

![Figure 2. ADE comparison across temporal and vessel-disjoint holdouts. The log scale is used because neural and acceleration baselines have much larger errors than the strongest kinematic models.](figures/jon_model_performance.png)

Table 2 gives the numerical trajectory results. The distinction between mean and median is important. For example, Kalman-CV has median ADE {fmt(stats_for(stats, 'temporal_test', 'kalman_filter_cv').get('ade_median'))} m on the temporal holdout and {fmt(stats_for(stats, 'vessel_disjoint_test', 'kalman_filter_cv').get('ade_median'))} m on the vessel-disjoint holdout, while its mean ADE remains much larger. This pattern is consistent with many short windows being easy and a smaller number of windows being difficult due to manoeuvres, sparse/interpolated reporting, regional geometry or other traffic effects.

Table 2. Trajectory-prediction metrics generated from the reported benchmark evidence package.

{trajectory_md}

Table 3 expands this point for the strongest kinematic and linear baselines. The P99 values are much larger than the medians, so the kilometre-scale means are driven by a small but important tail of difficult windows. The 1 percent trimmed means are far lower than the untrimmed means, which supports interpreting the aggregate results as a mixture of many easy short-horizon cases and a small set of manoeuvre, reporting, interpolation or regional edge cases.

Table 3. Tail-error diagnostics for selected baselines.

{tail_md}

The neural baselines should be interpreted carefully. The Transformer baseline records {fmt(neural_temporal.get('ade_meters'))} m temporal ADE in the reported run, and the other neural baselines are also far above the strongest kinematic and linear baselines. This is not evidence that neural AIS prediction is impossible. Under the documented baseline configurations and preprocessing protocol, the neural sequence baselines did not outperform the strongest kinematic baselines. The useful publication claim is therefore methodological: strong baselines, documented tuning, split discipline and downstream warning evaluation are necessary before asserting architecture superiority. The selected neural configurations and validation proxy-search summary are reported in Supplementary File S1.

The constant-acceleration result is also instructive. Although acceleration may appear to be a richer physical assumption than constant velocity, it performs very poorly in the reported aggregate metrics. This likely reflects the sensitivity of acceleration estimates to noisy or interpolated short-window position changes. For navigation applications, a physically plausible model family still needs to be numerically stable under the reporting characteristics of AIS. More parameters are not automatically better when the observations are irregular and manoeuvres are sparse.

Figure 3 shows error growth across the 15 forecast steps. The short-horizon character of the task is visible: errors generally increase with horizon, and the more stable baselines retain better behaviour over time. This figure is useful for navigation readers because a 15-minute average can hide whether an error appears immediately or accumulates near the end of the horizon.

![Figure 3. Horizon-wise ADE degradation for selected model families.](figures/jon_horizon_degradation.png)

Figure 4 reports the CPA/TCPA warning metrics. Kalman-CV and constant velocity are close in precision and recall. Kalman-CV has precision {fmt3(kalman_risk['precision'])} and recall {fmt3(kalman_risk['recall'])}; CV has precision {fmt3(cv_risk['precision'])} and recall {fmt3(cv_risk['recall'])}. Kalman-CV has a slightly lower false-alarm rate and missed-warning rate, while CV has a slightly lower mean absolute CPA error in nautical miles. The linear least-squares baseline produces a much higher false-alarm rate and larger CPA error, showing that position-prediction quality does not translate uniformly into warning quality.

![Figure 4. CPA/TCPA warning classification and CPA-error metrics for selected trajectory models.](figures/jon_risk_warning_metrics.png)

Table 4. AIS-derived risk-warning metrics.

{risk_md}

Figure 5 reports scenario-slice sensitivity for speed and region groups on the temporal holdout. The purpose is not to claim complete regional robustness. Instead, the figure makes visible that aggregate results are shaped by traffic composition. Low-speed and high-speed slices can have very different error magnitudes, and regions with fewer samples should be interpreted with greater caution. This supports the paper's conservative boundary: the reported evidence is stronger than a single aggregate benchmark but is not yet a full seasonal or global generalisation study.

The slice analysis also provides a practical debugging tool. If a future model improves the aggregate ADE but worsens high-speed or regional slices, a navigation reviewer may reasonably question whether the model is safer or merely better aligned with the dominant traffic class. Conversely, a model that is slightly worse on the aggregate but much more stable in a safety-critical slice could be worth further study. This is another reason to keep group summaries in the supplementary evidence package.

![Figure 5. Scenario-slice sensitivity by speed bin and region on the temporal holdout.](figures/jon_scenario_slice_errors.png)

Figure 6 provides AIS-derived encounter examples from the existing risk-warning output. Such figures are essential because navigation readers need to inspect whether the warning problem looks operationally plausible, not merely whether a metric table is favourable.

![Figure 6. AIS-derived encounter case studies from the risk-warning evaluation.](figures/jon_risk_case_studies.png)

## 7. DISCUSSION

The main result is conservative but practically useful. A simple, transparent motion model remains difficult to beat for short-horizon AIS forecasting, and a Kalman-style version of that model is the strongest reported baseline by ADE under both temporal and vessel-disjoint holdouts. This finding has immediate implications for navigation research. A new AIS prediction model should be compared against well-implemented kinematic baselines, not only against other neural models or weakened classical references. If it fails to improve on those baselines, the paper can still contribute if it explains where the failure occurs and what it reveals about the evaluation protocol.

For navigation research, the value of this result is an explicit evidence standard. The comparison links data provenance, split definitions, trajectory metrics and warning outputs before drawing a practical conclusion. That stance is well suited to safety-related navigation work, where a carefully bounded benchmark can be more useful than a premature performance claim.

The risk-warning results also temper the interpretation of ADE. Kalman-CV has the best trajectory ADE, but CV has a slightly lower mean absolute CPA error in the reported risk slice. The difference is small, and both models produce strong warning precision and recall, but the point is important. Downstream navigation metrics can reorder or nuance model preferences. An operational warning layer may care about false alarms, missed warnings and lead-time stability as much as mean position error. Future AIS prediction papers should therefore include at least one downstream navigation-safety metric when making decision-support claims.

This result also suggests a direction for future model design. Instead of training only to minimise pointwise displacement error, a navigation-specific predictor might include losses or calibration objectives related to encounter geometry, CPA uncertainty or warning-threshold stability. Such a model would need careful validation, but it would align the optimisation target more closely with the decision-support task. The reported benchmark provides a baseline against which such extensions can be tested.

The vessel-disjoint evaluation is one of the more important additions relative to a simpler evidence report. Vessel identity can leak behaviour into random splits, especially when multiple windows from the same MMSI appear in train and test sets. Holding out MMSI values does not solve every generalisation issue, but it reduces one obvious source of overoptimism. The reported results show that CV degrades substantially more than Kalman-CV under vessel-disjoint testing, while ridge regression becomes competitive with Kalman-CV. This suggests that model robustness depends on both the physical assumption and the split boundary.

The neural results should remain claim-bounded. These neural baselines are deliberately modest and documented as baselines. The tuning protocol records a validation-set proxy search, but it does not represent a comprehensive neural architecture search. A future neural study could improve scaling, loss design, coordinate parameterisation, sequence length, map context, vessel type metadata or encounter-aware objectives. Such improvements may well change the neural results. The contribution here is that those future claims should be made against this kind of evidence chain: auditable data, strong baselines, split definitions, statistical summaries and downstream warning analysis.

For practitioners, the workflow has a direct use. Running the pipeline on updated historical AIS source dates can regenerate the same tables, figures and warning outputs. The repository also includes an offline prediction-export utility that produces trajectory predictions and risk-warning records from prepared AIS data. This is not a live operational service, but it demonstrates how the benchmark can be extended into a monitoring workflow, subject to the same limitations about historical data quality, warning thresholds and absence of closed-loop validation.

The practical route from this benchmark to an operational tool would require several additional layers. First, the data ingestion layer would need live AIS handling, latency monitoring and missing-message logic. Second, the prediction layer would need calibrated uncertainty rather than only point trajectories. Third, the warning layer would need human-factor evaluation so that alert frequency, phrasing and timing support rather than distract operators. Fourth, any avoidance recommendation would need explicit COLREGs interpretation and closed-loop testing. The present work does not complete those steps, but it gives them a reproducible starting point.

## 8. LIMITATIONS

The evidence is historical. It does not ingest a live AIS stream and does not validate real-time deployment. The four source dates provide a stronger protocol than a single-day experiment, but they do not prove all-day, seasonal or global generalisation. The processed artefact includes useful metadata, yet it does not include every contextual variable that could matter, such as vessel class reliability, weather, traffic-control context, chart constraints or planned routes.

The risk-warning task is an AIS-derived decision-support evaluation. It is not a simulator of bridge-team behaviour, not a COLREGs compliance proof and not an autonomous collision-avoidance validation. The International Regulations for Preventing Collisions at Sea remain a legal and operational framework far richer than the thresholded CPA/TCPA labels used here (International Maritime Organization, 1972). Any operational use would require human-factors assessment, false-alarm tolerance analysis, traffic-service integration, reliability engineering and domain-specific validation.

## 9. CONCLUSIONS

This paper reports a reproducible AIS trajectory-prediction benchmark for navigation risk-warning support. The evidence chain links raw historical AIS data, checksummed preprocessing, temporal and vessel-disjoint splits, kinematic/statistical/neural baselines, paired statistical summaries, scenario-slice analysis and CPA/TCPA warning metrics. The best mean-ADE model is Kalman-CV on both temporal and vessel-disjoint holdouts. Constant velocity remains a strong short-horizon baseline, and the documented neural sequence baselines do not outperform the simple baselines under this controlled protocol. The practical lesson is not that deep learning has no role in maritime prediction, but that navigation papers need strong baselines, reproducible splits and downstream warning metrics before making decision-support claims.

## DATA AND CODE AVAILABILITY

The code, configuration files, generated figures, compact evidence artefacts and manuscript-generation workflow are available at {PUBLIC_REPO_URL}, archived under tag `{PUBLIC_REPO_TAG}`. No separate archival DOI is available for this release at the time of submission. The source data are derived from public NOAA MarineCadastre.gov historical AIS files subject to NOAA data access terms; the repository does not redistribute raw NOAA AIS files or processed NumPy arrays.

The full reproduction command and evidence-file inventory are provided in Supplementary File S1 and in the repository README.

The supplementary package accompanying this manuscript includes manifest summaries, full model metrics, horizon and group summaries, statistical summaries, neural tuning records and risk-warning metrics, but excludes the large per-sample error file to satisfy supplementary file-size guidance.

## COMPETING INTERESTS

The author declares no competing interests.

## AUTHOR CONTRIBUTIONS

{AUTHOR_NAME} is the sole author and was responsible for study conception, methodology, software workflow, validation, formal analysis, interpretation, manuscript drafting and final approval.

## FUNDING

This research received no specific grant from any funding agency, commercial or not-for-profit sectors.

## AI-USE DECLARATION

Generative AI tools were used in May 2026 for assistance with code drafting, language editing, manuscript structuring and consistency checks against repository evidence files. The author verified all numerical results and is responsible for the final content.

## ACKNOWLEDGEMENTS

The author thanks his family for their patience, encouragement and support during the preparation of this work.

## SUPPLEMENTARY MATERIAL

Supplementary File S1 contains the reproducibility manifest, full benchmark tables, neural baseline configuration summary, scenario summaries and risk-warning evidence used to generate this manuscript. The individual supplementary archive is kept below the 10 MB per-file guideline.

## REFERENCES

{make_references()}
"""


def make_cover_letter(generated_at: str) -> str:
    return f"""# Cover Letter for The Journal of Navigation

Date: 18 May 2026

To the Editor,

The Journal of Navigation

Dear Editor,

I am pleased to submit the manuscript entitled "A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support" for consideration as a Research Article in The Journal of Navigation.

The manuscript is intended for the journal's navigation-science readership because it links short-horizon AIS trajectory prediction with CPA/TCPA risk-warning behaviour under an auditable real-data protocol. Rather than presenting a new deep-learning architecture, the paper addresses a practical evaluation problem: navigation decision-support claims require strong simple baselines, transparent split definitions, reproducible evidence and downstream warning metrics.

The main contributions are:

- A metadata-rich real-AIS benchmark built from NOAA historical AIS data, with checksums, temporal holdout labels, vessel-disjoint holdout labels and generated evidence artefacts.
- A conservative comparison of kinematic, Kalman-style, linear and neural baselines showing that Kalman-CV is the strongest current ADE baseline under both temporal and vessel-disjoint holdouts.
- A CPA/TCPA warning evaluation that reports precision, recall, false alarms, missed warnings and CPA error without claiming autonomous collision-avoidance validation.

The manuscript is original and is not under consideration elsewhere. The author approves its submission and declares no competing interests. The research received no specific grant from any funding agency, commercial or not-for-profit sectors. The public code release is available at {PUBLIC_REPO_URL}, tag `{PUBLIC_REPO_TAG}`.

Suggested reviewers, if requested: to be entered only after checking conflicts of interest.

Sincerely,

{AUTHOR_NAME}
{AUTHOR_AFFILIATION}
{AUTHOR_EMAIL}
"""


def make_checklist(generated_at: str, word_count_value: int, zip_size_mb: float, figure_paths: dict[str, str]) -> str:
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths.values())
    return f"""# JON Submission Checklist

Generated: {generated_at}

## Manuscript Files

- Main Markdown draft: `paper/jon_manuscript.md`
- Word upload candidate: `paper/jon_manuscript.docx`
- PDF review draft: `paper/jon_manuscript.pdf`
- Cover letter draft: `paper/jon_cover_letter.md`
- Supplementary notes: `paper/jon_supplementary_materials.md`
- Supplementary archive: `paper/jon_supplementary_materials.zip` ({zip_size_mb:.2f} MB)
- ScholarOne copy-paste metadata: `paper/jon_scholarone_metadata.md`
- Reference and DOI audit: `paper/jon_reference_audit.md`
- Chinese submission guide: `paper/jon_submission_guide_zh.md`

## Current Automated Status

- Article type: Research Article.
- Main text word count estimate: {word_count_value:,} words including declarations and references.
- Author metadata inserted for {AUTHOR_NAME}, {AUTHOR_AFFILIATION}, {AUTHOR_EMAIL}.
- Short title inserted: `{SHORT_TITLE}`.
- Public code citation inserted: {PUBLIC_REPO_URL}, tag `{PUBLIC_REPO_TAG}`.
- Keywords are not typed into the manuscript body; choose them in ScholarOne instead.
- Suggested ScholarOne keywords: AIS, maritime navigation, trajectory prediction, risk warning.
- Figures are embedded in the Markdown and DOCX/PDF drafts and are also available as separate PNG files:
{figure_lines}
- Supplementary archive is below 10 MB.
- Numerical claims are generated from repository evidence files.
- Reference DOI/style audit, British-English wording pass, ScholarOne copy-paste metadata, cover-letter cleanup, and Chinese submission guide have been prepared.

## Required Human Items Before Upload

- Add ORCID in ScholarOne if available.
- Re-check that the no-specific-funding and no-competing-interests statements are accurate.
- Create and insert a Zenodo or equivalent DOI if an archival DOI is obtained before submission.
- Review `paper/jon_reference_audit.md` and resolve any editorial DOI query if raised.
- Add suggested reviewers only after checking conflicts of interest.
- Confirm the manuscript is not under review elsewhere and all authors approve submission.
- Review the ScholarOne-generated PDF before final submission.

## Official JON Constraints Tracked

- Use ScholarOne via Cambridge for submission.
- Research Article is a supported article type.
- Single-blind peer review is used.
- Average length target is 6,000-8,000 words and up to about 20 pages including figures/tables.
- Abstract should be about 150 words.
- Short title should be no more than 40 characters.
- Initial Word or LaTeX submission is accepted.
- Figures and tables should be visible in context for review.
- Do not type keywords into the manuscript; select them online.
- References should use Harvard/JON author-date style and not be numbered.
- Include competing-interest and AI-use declarations where applicable.
- Keep each supplementary file no larger than 10 MB.
"""


def make_authorial_polish_workflow(generated_at: str) -> str:
    return f"""# Authorial Polish and De-Template Workflow

Generated: {generated_at}

Purpose: add a mandatory manuscript-polishing step that removes generated-report flavour, repetitive phrasing, and repository-facing wording while preserving research integrity and the required AI-use declaration.

This workflow is not a way to hide material assistance or bypass disclosure. The AI-use declaration remains in the manuscript where required. The goal is to make the text read like an authored navigation-science paper: specific, concise, claim-bounded, and consistent with the evidence.

## Step Added to the Full Workflow

Run this after automated manuscript generation and before final language editing:

1. Evidence lock: confirm every number in the abstract, results, tables and conclusion maps to `outputs/audit/`, `outputs/final_multiday/`, `outputs/final_risk/`, or `outputs/final_submission/`.
2. Authorial voice pass: replace generic phrases such as "this paper presents" when overused, vary paragraph openings, remove redundant cautionary sentences, and make transitions reflect the actual argument.
3. Journal-reader pass: replace repository-facing wording with navigation-facing wording. Keep file paths in data/code availability or supplementary notes rather than in the main argument.
4. Claim-boundary pass: keep limitations precise, but avoid defensive repetition. The paper should be conservative without sounding apologetic.
5. Figure/table pass: make sure every figure is introduced by a substantive sentence and followed by an interpretation, not merely a caption restatement.
6. Human metadata pass: replace placeholders for authors, affiliations, funding, acknowledgements and contributions.
7. AI-use integrity pass: keep the AI-use declaration accurate and transparent. Do not remove it just to make the manuscript sound less generated.
8. Final read-aloud pass: read each section aloud and shorten sentences that sound like templated prose.

## Typical Edits to Make Manually

- Replace broad claims with exact claims tied to the current evidence pack.
- Convert list-like paragraphs into a narrative argument.
- Remove repeated phrases such as "under this controlled protocol" if they appear too often in nearby paragraphs.
- Replace "the project provides" with the actual research contribution where possible.
- Check that British English spelling is consistent for the JON version.
- Keep technical terms such as AIS, CPA, TCPA, ADE and FDE stable.

## What Not To Do

- Do not remove limitations that protect the claim boundary.
- Do not add live-AIS, autonomous collision-avoidance, all-day, seasonal, or architecture-superiority claims unless new evidence is generated.
- Do not remove the AI-use declaration if AI tools contributed to drafting, code generation, analysis or figure preparation.
"""


def make_supplementary(
    generated_at: str,
    data_manifest: dict[str, Any],
    model_rows: list[dict[str, str]],
    stats: dict[str, Any],
    risk_metrics: dict[str, Any],
    neural_protocol: dict[str, Any],
) -> str:
    dataset = data_manifest["dataset_summary"]
    return f"""# Supplementary Material S1: Reproducibility and Evidence Summary

Generated: {generated_at}

This supplementary note documents the artifacts used to generate the manuscript. It is designed as a concise companion to the main manuscript. The large per-sample error CSV is intentionally excluded from the zipped supplementary package because it is about 55 MB, which exceeds the per-file supplementary guideline.

## Dataset Summary

- Processed sample count: {dataset['sample_count']:,}
- Unique MMSI values: {dataset['unique_mmsi_count']:,}
- Source dates: {', '.join(dataset['source_dates'])}
- Regions: {', '.join(dataset['regions'])}
- Temporal split counts: {dataset['temporal_split_counts']}
- Vessel-disjoint split counts: {dataset['vessel_split_counts']}

## Included Evidence Files

- `multiday_data_manifest.json`
- `model_metrics.csv`
- `generalization_metrics.csv`
- `error_summary_by_horizon.csv`
- `error_summary_by_group.csv`
- `statistical_tests.json`
- `neural_tuning_protocol.json`
- `neural_tuning_results.csv`
- `risk_metrics.json`
- `risk_scenarios.csv`
- `readiness_report.json`

## Reproducibility Command

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_high_quality_pipeline.sh
```

## Full Model Metrics

{trajectory_table(model_rows, stats)}

## Neural Baseline Configuration Summary

The neural sequence baselines use the same 30-step input and 15-step output task as the other models. The proxy search used validation ADE for configuration selection, with final claims tied to the full temporal and vessel-disjoint test runs.

{neural_config_table(neural_protocol)}

## Risk-Warning Metrics

{risk_table(risk_metrics)}
"""


def split_name_zh(split: str) -> str:
    return {
        "temporal_test": "时间保持测试",
        "vessel_disjoint_test": "船舶不相交测试",
    }.get(split, split)


def model_label_zh(model: str) -> str:
    return {
        "constant_acceleration": "常加速度",
        "constant_velocity": "常速度",
        "gru_baseline": "GRU",
        "kalman_filter_cv": "Kalman-CV",
        "linear_lstsq": "最小二乘",
        "lstm_baseline": "LSTM",
        "ridge_lstsq": "岭回归",
        "tcn_baseline": "TCN",
        "transformer_baseline": "Transformer",
    }.get(model, model)


def trajectory_table_zh(rows: list[dict[str, str]], stats: dict[str, Any]) -> str:
    lines = [
        "| 划分策略 | 模型 | 平均ADE/m | 中位ADE/m | 95% ADE区间/m | FDE/m |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=model_sort_key):
        split = row["split_policy"]
        model = row["model"]
        model_stats = stats_for(stats, split, model)
        ci = model_stats.get("ade_ci_percentile", ["NA", "NA"])
        interval = f"{fmt(ci[0])}至{fmt(ci[1])}" if len(ci) == 2 else "NA"
        lines.append(
            "| {split} | {model} | {ade} | {median} | {interval} | {fde} |".format(
                split=split_name_zh(split),
                model=model_label_zh(model),
                ade=fmt(row.get("ade_meters")),
                median=fmt(model_stats.get("ade_median")),
                interval=interval,
                fde=fmt(row.get("fde_meters")),
            )
        )
    return "\n".join(lines)


def risk_table_zh(risk_metrics: dict[str, Any]) -> str:
    metrics = risk_metrics["metrics_by_model"]
    lines = [
        "| 模型 | TP | FP | FN | TN | 精确率 | 召回率 | 误报率 | 漏报率 | CPA误差/n mile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["kalman_filter_cv", "constant_velocity", "linear_lstsq"]:
        row = metrics[model]
        lines.append(
            "| {model} | {tp} | {fp} | {fn} | {tn} | {precision} | {recall} | {far} | {miss} | {cpa} |".format(
                model=model_label_zh(model),
                tp=row["true_positive"],
                fp=row["false_positive"],
                fn=row["false_negative"],
                tn=row["true_negative"],
                precision=fmt3(row["precision"]),
                recall=fmt3(row["recall"]),
                far=fmt3(row["false_alarm_rate"]),
                miss=fmt3(row["missed_warning_rate"]),
                cpa=fmt3(row["mean_abs_cpa_error_nmi"]),
            )
        )
    return "\n".join(lines)


def make_chinese_manuscript(
    generated_at: str,
    data_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    readiness: dict[str, Any],
    model_rows: list[dict[str, str]],
    stats: dict[str, Any],
    risk_metrics: dict[str, Any],
) -> str:
    dataset = data_manifest["dataset_summary"]
    config = run_manifest.get("config", {}).get("experiment", {})
    best = best_by_split(model_rows)
    temporal_best = best.get("temporal_test", {})
    vessel_best = best.get("vessel_disjoint_test", {})
    risk_generation = risk_metrics["scenario_generation"]
    kalman_risk = risk_metrics["metrics_by_model"]["kalman_filter_cv"]
    cv_risk = risk_metrics["metrics_by_model"]["constant_velocity"]
    model_lookup = lookup_rows(model_rows)
    cv_temporal = model_lookup[("temporal_test", "constant_velocity")]
    cv_vessel = model_lookup[("vessel_disjoint_test", "constant_velocity")]
    ridge_vessel = model_lookup[("vessel_disjoint_test", "ridge_lstsq")]
    transformer_temporal = model_lookup[("temporal_test", "transformer_baseline")]
    dates = "、".join(dataset["source_dates"])
    regions = "、".join(dataset["regions"])
    trajectory_md = trajectory_table_zh(model_rows, stats)
    risk_md = risk_table_zh(risk_metrics)
    readiness_status = readiness.get("overall_status", "not_audited")

    return f"""# 面向航行风险预警的AIS船舶轨迹预测可复现基准研究

文章类型：研究论文中文工作稿

作者：李波

单位：中国海事服务中心

通信作者：李波，li.bo@cmaritime.com.cn

## 摘要

船舶短时轨迹预测通常被表述为机器学习问题，但面向航行安全的应用还需要透明的基线模型、可复核的数据划分、泛化评估以及与风险预警指标之间的明确联系。本文构建了一个面向AIS历史数据的可复现轨迹预测与CPA/TCPA风险预警评估基准。当前证据包来自NOAA历史AIS数据，覆盖4个数据源日期（{dates}），包含{dataset['sample_count']:,}个轨迹窗口和{dataset['unique_mmsi_count']:,}个MMSI。每个样本使用{config.get('history_steps', 30)}个一分钟历史位置预测未来{config.get('forecast_steps', 15)}个一分钟位置。协议记录原始文件校验和、处理后文件校验和、船舶标识、时间、区域、航速、转向强度、时间保持划分和船舶不相交划分。实验比较运动学、Kalman风格、线性统计和神经网络基线，并进一步评估预测轨迹对AIS衍生CPA/TCPA预警精确率、召回率、误报率和漏报率的影响。当前非调试运行中，Kalman风格常速度模型在时间保持测试中取得{fmt(temporal_best.get('ade_meters'))} m ADE，在船舶不相交测试中取得{fmt(vessel_best.get('ade_meters'))} m ADE，均为最优。常速度模型在时间保持测试中仍是强基线（{fmt(cv_temporal.get('ade_meters'))} m ADE），但在船舶不相交测试中误差增大（{fmt(cv_vessel.get('ade_meters'))} m ADE）。在{risk_generation['scenario_count']:,}个AIS衍生会遇场景中，Kalman风格常速度模型的预警精确率为{fmt3(kalman_risk['precision'])}，召回率为{fmt3(kalman_risk['recall'])}，误报率为{fmt3(kalman_risk['false_alarm_rate'])}，漏报率为{fmt3(kalman_risk['missed_warning_rate'])}。本文不宣称完成自主避碰验证；贡献在于提供一条可审计的航行预测与风险预警证据链，说明简单基线和下游预警指标应成为AIS预测论文的必要组成部分。

关键词：AIS；船舶轨迹预测；CPA/TCPA；风险预警；可复现基准；运动学基线

## 1 引言

船舶自动识别系统（AIS）为海上交通监测、港口态势感知、航线推断、异常检测和碰撞风险评估提供了重要数据基础。随着历史AIS数据规模扩大，越来越多研究尝试使用LSTM、GRU、Transformer等深度序列模型进行船舶轨迹预测。然而，在航行安全场景中，模型结构复杂并不自动等于证据更强。预测结果可能在某一种随机划分下表现较好，却在未见船舶、不同时间块或特定会遇场景中明显退化。仅报告平均位置误差也无法说明模型是否会减少漏报、误报或CPA/TCPA预警偏差。

本文关注的是方法论问题，而不是提出一个新的深度学习结构。许多AIS轨迹预测研究的可比性受限于数据清洗、插值、轨迹窗口构造、坐标度量、训练测试划分和基线模型实现。对于航行安全研究而言，预测模型只有在强基线、可复核划分和下游风险指标下仍然成立，才适合被描述为决策支持能力的提升。因此，本文提出三个研究问题：第一，简单运动学、线性统计和神经网络基线在统一短时AIS协议下表现如何；第二，模型排序在时间保持和船舶不相交测试中是否稳定；第三，轨迹预测误差如何影响CPA/TCPA风险预警精确率、召回率、误报率和漏报率。

本文的贡献包括三点。第一，建立保留MMSI、时间、数据源日期、区域、航速和转向强度等元数据的AIS基准流程，使同一处理后数据同时支持时间保持和船舶不相交测试。第二，在统一协议下比较运动学、Kalman风格、线性统计和神经网络模型，强调常速度及其平滑变体不是弱基线，而是短时航行预测中必须严肃对照的强基线。第三，将轨迹预测结果映射到CPA/TCPA风险预警指标，避免只凭ADE或FDE讨论航行安全意义。

## 2 相关工作

AIS数据已广泛用于船舶行为建模、航线发现、异常检测和短时预测。相关研究表明，船舶轨迹既受到航道、港口、交通组织和操作习惯约束，也受到AIS上报间隔、接收质量和机动行为影响。因此，AIS预测模型的性能很大程度上取决于预处理和评估协议。若只给出处理后的数组而不说明清洗、插值和划分方式，外部读者很难判断模型结果是否可复核。

短时预测任务中，简单运动学模型具有天然优势。多数船舶在几分钟到十几分钟尺度内保持相对稳定的航向和航速，常速度、常加速度和Kalman滤波模型能够以较低计算代价给出可解释预测。复杂模型若不能在同等协议下明显优于这些基线，其工程价值就需要重新审视。本文因此将常速度、常加速度和Kalman风格常速度模型作为核心基线，而不是把它们作为形式化对照。

神经序列模型具有较强非线性拟合能力，但也更依赖坐标表示、归一化、损失函数、训练预算和调参策略。本文纳入LSTM、GRU、Transformer和TCN基线，是为了记录这些常见模型在当前协议下的可复核表现，而不是否定神经网络在所有AIS预测任务中的潜力。更强的神经模型仍可能通过地图约束、船型信息、航路上下文、概率预测或会遇感知损失取得改进，但这些改进需要在强基线和严格划分下重新验证。

碰撞风险评估进一步要求把轨迹误差转化为航行相关指标。CPA和TCPA是会遇风险分析中的常见量，但它们并不等同于完整避碰决策。本文采用CPA/TCPA预警精确率、召回率、误报率和漏报率作为决策支持指标，保持在离线历史AIS评估范围内，不把结果表述为COLREGs合规或闭环自主避碰能力。

## 3 数据与可复现协议

本文使用NOAA MarineCadastre.gov公开AIS历史数据。当前处理后证据包覆盖{dates}，时间范围为{dataset['time_range']['start']}至{dataset['time_range']['end']}，区域包括{regions}。数据清单记录原始文件校验和、处理后文件校验和、样本数量、MMSI数量、区域标签、平均航速、转向强度和插值比例。当前样本平均对地航速为{fmt(dataset['avg_sog_knots']['mean'], 2)} kn，最大对地航速为{fmt(dataset['avg_sog_knots']['max'], 2)} kn，平均转向强度为{fmt(dataset['turn_intensity_deg']['mean'], 2)}°，平均插值比例为{fmt(dataset['interpolation_ratio']['mean'], 3)}。

每个轨迹窗口使用一分钟间隔。模型输入为{config.get('history_steps', 30)}个历史步，输出为{config.get('forecast_steps', 15)}个未来步。处理后的坐标保持为WGS84经纬度，ADE和FDE采用Haversine距离，以米为单位；RMSE和MAE采用局部北东分量误差，同样以米为单位。这样可以避免把经纬度角度直接当作平面距离导致的指标误读。

时间保持划分包含{dataset['temporal_split_counts']['train']:,}个训练样本、{dataset['temporal_split_counts']['val']:,}个验证样本和{dataset['temporal_split_counts']['test']:,}个测试样本。船舶不相交划分包含{dataset['vessel_split_counts']['train']:,}个训练样本、{dataset['vessel_split_counts']['val']:,}个验证样本和{dataset['vessel_split_counts']['test']:,}个测试样本。前者考察后续时间块泛化，后者考察未见MMSI船舶泛化。两者均为历史数据协议，不能直接解释为实时AIS系统或全天候季节性验证。

图1给出从原始AIS到风险预警指标的证据链。

![图1 AIS轨迹预测与风险预警可复现证据链。](figures/jon_pipeline_protocol.png)

当前高质量审计报告状态为`{readiness_status}`，生成时无阻塞缺口。本文所有主要数值均由仓库中的`outputs/audit/`、`outputs/final_multiday/`、`outputs/final_risk/`和`outputs/final_submission/`文件生成。

## 4 模型与评价指标

基准包含九类模型：常速度、常加速度、Kalman风格常速度、普通最小二乘、岭回归、LSTM、GRU、Transformer和TCN。常速度模型由近期运动外推未来位置；常加速度模型进一步估计加速度；Kalman风格模型使用近期速度平滑降低噪声影响；线性模型将历史窗口映射到未来位移；神经网络模型作为常见序列学习基线。

主要轨迹指标为ADE、FDE、RMSE和MAE。ADE表示整个预测时域内预测位置与真实位置的平均Haversine距离，FDE表示最后一个预测步的Haversine距离。由于AIS误差分布具有明显偏态，本文除平均ADE外还报告中位ADE和经验95%区间。模型排序以平均ADE为主，同时结合风险预警指标讨论实际意义。

## 5 风险预警评价设计

风险预警实验从时间保持测试集中构造AIS衍生会遇场景。真实未来轨迹用于生成真实预警标签：若预测时域内真实最小CPA低于给定阈值，则该场景为正例。模型预测轨迹用于计算预测CPA/TCPA并产生预测预警标签，进而得到TP、FP、FN和TN。本文报告精确率、召回率、误报率、漏报率、平均提前时间误差和平均CPA绝对误差。

当前风险实验使用{risk_generation['scenario_count']:,}个场景，来自{risk_generation['evaluated_samples']:,}个评估样本；搜索半径为{risk_generation['search_radius_nmi']} n mile，预警阈值为{risk_generation['warning_threshold_nmi']} n mile，真实预警数量为{risk_generation['truth_warning_count']:,}。该设计用于评价决策支持指标，而非验证自主避碰系统。

## 6 实验结果

图2展示主要模型在两种划分下的ADE。Kalman风格常速度模型在时间保持和船舶不相交测试中均为平均ADE最低的模型。常速度模型在时间保持测试中达到{fmt(cv_temporal.get('ade_meters'))} m ADE，但在船舶不相交测试中增至{fmt(cv_vessel.get('ade_meters'))} m ADE。岭回归在船舶不相交测试中达到{fmt(ridge_vessel.get('ade_meters'))} m ADE，说明线性统计模型在未见船舶测试中仍具有竞争力。

![图2 两种划分协议下的ADE对比。](figures/jon_model_performance.png)

表1给出完整轨迹预测结果。Kalman-CV在时间保持测试中的中位ADE为{fmt(stats_for(stats, 'temporal_test', 'kalman_filter_cv').get('ade_median'))} m，在船舶不相交测试中的中位ADE为{fmt(stats_for(stats, 'vessel_disjoint_test', 'kalman_filter_cv').get('ade_median'))} m，但平均ADE仍达到公里级。这说明多数短窗口较容易预测，少量困难窗口显著拉高均值。

表1 轨迹预测结果

{trajectory_md}

当前神经网络基线未支持结构优越性结论。例如Transformer在时间保持测试中的ADE为{fmt(transformer_temporal.get('ade_meters'))} m，明显高于Kalman-CV和常速度模型。这一结果不能推广为“神经网络不适合AIS预测”，但可以支持一个更稳妥的结论：若缺少严格预处理、调参和划分协议，朴素神经模型可能在短时AIS预测中明显失败。

图3展示预测时域内误差随步长增长的情况。该图有助于判断误差是早期即出现，还是主要在预测末端累积。

![图3 15分钟预测时域内的ADE变化。](figures/jon_horizon_degradation.png)

图4和表2展示CPA/TCPA风险预警结果。Kalman-CV的精确率为{fmt3(kalman_risk['precision'])}、召回率为{fmt3(kalman_risk['recall'])}；常速度模型的精确率为{fmt3(cv_risk['precision'])}、召回率为{fmt3(cv_risk['recall'])}。两者预警表现接近，但Kalman-CV的误报率和漏报率略低。最小二乘模型的误报率和CPA误差更高，说明位置预测误差与风险预警质量并非完全等价。

![图4 CPA/TCPA风险预警分类指标与CPA误差。](figures/jon_risk_warning_metrics.png)

表2 AIS衍生风险预警指标

{risk_md}

图5给出速度和区域场景切片结果。切片分析不用于宣称完整区域泛化，而是提醒读者：总体误差可能受低速样本、特定航区或少量困难样本影响。未来模型若只改善总体均值，却恶化关键场景切片，其航行安全价值仍需重新评估。

![图5 时间保持测试中的速度与区域切片误差。](figures/jon_scenario_slice_errors.png)

图6展示风险预警实验中的AIS衍生会遇案例。案例图可以帮助读者判断预警任务是否具有航行解释性，而不是只依赖指标表。

![图6 风险预警评价中的AIS衍生会遇案例。](figures/jon_risk_case_studies.png)

## 7 讨论

本文最重要的结论是保守但具有工程意义：在15分钟短时AIS预测任务中，简单、可解释的运动学基线仍然很难被轻易超越。Kalman风格常速度模型通过对近期速度进行平滑，在两种测试划分中均取得最低ADE。对于航行研究而言，这意味着新模型必须与强基线比较，而不能只与其他神经模型比较。

风险预警结果进一步说明，ADE并不是唯一的航行安全指标。Kalman-CV的ADE最低，但常速度模型在当前风险场景中的平均CPA误差略低。两者差异不大，却提醒我们：下游预警指标可能改变或细化对模型优劣的理解。面向航行决策支持的轨迹预测研究应至少包含一种与CPA/TCPA、会遇几何或预警行为相关的指标。

项目也具有现实应用意义。只要获取更新的历史AIS数据并按相同流程处理，就可以重新生成当前周期的轨迹预测、模型对比和风险预警候选结果。项目中的`predict_latest_ais.py`已经给出离线最新数据预测与风险预警导出流程。不过，这仍不是实时系统；若要进入实际应用，还需要实时AIS接入、延迟监控、不确定性估计、告警人因评估以及更完整的COLREGs和闭环仿真验证。

## 8 局限性

当前证据来自历史AIS数据，并不包含实时AIS流处理。四个数据源日期比单日实验更强，但不能证明全天候、季节性或全球泛化。处理后数据包含有用元数据，但尚未纳入天气、交通管制、精细船型、航道图约束或计划航线等因素。

风险预警实验是离线决策支持评估，不是船舶自主避碰验证，也不是COLREGs合规证明。正式运营系统还需要对误报容忍度、漏报后果、人机交互、VTS流程和安全认证进行独立评估。

参考文献仍需在正式投稿前进行最终DOI、卷期页码和期刊格式审校。该审校不影响当前数值证据，但会影响投稿文本的规范性。

## 9 结论

本文建立了从AIS数据审计、短时轨迹预测、时间/船舶泛化评估、场景切片分析到CPA/TCPA风险预警评价的可复现证据链。当前证据表明，Kalman风格常速度模型在时间保持和船舶不相交测试中均为最优ADE模型，常速度模型仍是强短时基线，朴素神经网络基线在当前协议下未能优于简单模型。本文的实践启示是：航行预测论文在提出复杂模型前，应先证明其相对于强运动学基线、严格划分协议和下游预警指标的实际收益。

## 数据与代码可用性

代码、配置、生成图件、紧凑证据文件和论文生成流程已公开于{PUBLIC_REPO_URL}，归档标签为`{PUBLIC_REPO_TAG}`。当前稿件尚无单独归档DOI。源数据来自NOAA MarineCadastre.gov公开历史AIS文件；仓库不重新分发NOAA原始AIS文件或处理后的NumPy数组。

高质量证据包可通过以下命令复现：

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
DATES="2024-01-02 2024-01-09 2024-02-06 2024-03-05" \
bash scripts/run_high_quality_pipeline.sh
```

## 利益冲突声明

作者声明不存在利益冲突。

## 作者贡献

李波为本文唯一作者，负责研究构思、方法设计、软件流程、验证、形式化分析、结果解释、论文撰写和最终审定。

## 基金

本研究未获得任何公共、商业或非营利机构的专项资助。

## 致谢

作者感谢家人在本文准备过程中的耐心、鼓励和支持。

## AI工具使用声明

OpenAI Codex/ChatGPT在2026年5月被用于辅助代码生成、论文初稿组织、文档结构整理以及与仓库证据文件的一致性检查。作者对全部内容负责，并将在投稿前完成数值、参考文献、语言和图表的人工审校。

## 参考文献

{make_references()}
"""


def make_chinese_interpretation(
    generated_at: str,
    data_manifest: dict[str, Any],
    model_rows: list[dict[str, str]],
    risk_metrics: dict[str, Any],
) -> str:
    dataset = data_manifest["dataset_summary"]
    best = best_by_split(model_rows)
    temporal_best = best.get("temporal_test", {})
    vessel_best = best.get("vessel_disjoint_test", {})
    risk_generation = risk_metrics["scenario_generation"]
    kalman_risk = risk_metrics["metrics_by_model"]["kalman_filter_cv"]
    cv_risk = risk_metrics["metrics_by_model"]["constant_velocity"]
    return f"""# 中文版论文解读：这篇论文到底在讲什么

生成时间：{generated_at}

## 一句话概括

这篇论文不是在证明“深度学习一定能让船舶轨迹预测更准”，而是在证明另一件更稳、更容易发表的事：在真实AIS数据上做短时船舶轨迹预测时，简单运动学基线非常强；如果没有可复现的数据处理、严格划分、强基线和风险预警指标，直接宣称复杂模型更好是不可靠的。

## 论文核心问题

论文围绕三个问题展开：

1. 在统一AIS协议下，常速度、Kalman-CV、线性模型和神经网络基线谁更可靠。
2. 时间保持测试和船舶不相交测试下，模型排序是否稳定。
3. 轨迹误差会不会影响CPA/TCPA风险预警的精确率、召回率、误报和漏报。

## 数据和任务

当前证据包来自NOAA历史AIS数据，包含{dataset['sample_count']:,}个轨迹窗口、{dataset['unique_mmsi_count']:,}个MMSI，覆盖数据源日期{', '.join(dataset['source_dates'])}。每个样本用30分钟历史轨迹预测未来15分钟轨迹。误差用Haversine米制距离计算，不把经纬度角度误当成距离。

## 最关键结果

Kalman风格常速度模型是当前最强ADE模型：

- 时间保持测试：{fmt(temporal_best.get('ade_meters'))} m ADE。
- 船舶不相交测试：{fmt(vessel_best.get('ade_meters'))} m ADE。

常速度模型仍然是强基线：

- 时间保持测试中为{fmt(lookup_rows(model_rows)[('temporal_test', 'constant_velocity')].get('ade_meters'))} m ADE。
- 船舶不相交测试中为{fmt(lookup_rows(model_rows)[('vessel_disjoint_test', 'constant_velocity')].get('ade_meters'))} m ADE。

这说明短时预测里“简单模型很强”不是口号，而是当前证据包支持的结果。

## 风险预警结果是什么意思

论文又做了{risk_generation['scenario_count']:,}个AIS衍生会遇场景的风险预警实验。Kalman-CV的预警精确率为{fmt3(kalman_risk['precision'])}、召回率为{fmt3(kalman_risk['recall'])}、误报率为{fmt3(kalman_risk['false_alarm_rate'])}、漏报率为{fmt3(kalman_risk['missed_warning_rate'])}。常速度模型的精确率为{fmt3(cv_risk['precision'])}、召回率为{fmt3(cv_risk['recall'])}，两者都比较强。

这里的意义是：论文不只看“预测点离真实点多远”，还看这些误差会不会影响风险预警。这个角度比单纯模型排行榜更接近航海期刊读者关心的问题。

## 为什么这篇论文有现实意义

现实意义主要有三层：

1. 给AIS轨迹预测研究建立强基线：以后任何复杂模型都应该先和常速度、Kalman-CV等简单模型认真比较。
2. 给航行风险预警提供可复现证据链：从原始AIS、清洗、切片、模型、误差到CPA/TCPA预警，每一步都有产物可查。
3. 可以用更新的历史AIS数据重跑：项目已经有离线最新数据预测和风险预警导出脚本，未来可以换新数据生成新一轮预测结果。

## 不能夸大的地方

这篇论文现在不能说：

- 已经实现实时AIS预测系统。
- 已经完成自主避碰系统验证。
- 已经证明深度学习普遍不如简单模型。
- 已经证明全天候、季节性或全球航区泛化。
- 已经满足COLREGs闭环合规验证。

正确说法是：当前项目支持一个保守、可复现、面向风险预警的AIS轨迹预测基准论文。

## 图1为什么要修

图1是用`matplotlib`程序生成的流程图，不是下载的图片，也不是AI图片生成。之前效果不好，是因为流程框坐标太靠左，第一块框在嵌入PDF后看起来被裁切，且横向框之间过近，长英文标签互相挤压。现在脚本已经调大画布、增加左右边距、缩短文字并分行，重新生成后左侧不会再被切掉。

## 投稿前还需要作者做什么

- 如有ORCID，在ScholarOne中补充。
- 若投稿前取得Zenodo或其他归档DOI，补入数据与代码可用性声明。
- 做参考文献和DOI核对。
- 做一次“去模板化/去AI味”的人工润色，但保留真实AI-use声明。
- 按ScholarOne要求填关键词、利益冲突、数据可用性等元数据，并确认“无专项资助”和“无利益冲突”陈述准确。
- 检查系统生成的投稿PDF。
"""


def make_zip(zip_path: Path, supplementary_md: Path, files: list[Path]) -> float:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(supplementary_md, arcname="jon_supplementary_materials.md")
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)
    return zip_path.stat().st_size / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--final-dir", default="outputs/final_multiday")
    parser.add_argument("--risk-dir", default="outputs/final_risk")
    parser.add_argument("--submission-dir", default="outputs/final_submission")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    audit_dir = Path(args.audit_dir)
    final_dir = Path(args.final_dir)
    risk_dir = Path(args.risk_dir)
    submission_dir = Path(args.submission_dir)
    figures_dir = paper_dir / "figures"

    generated_at = datetime.now(timezone.utc).isoformat()
    data_manifest = read_json(audit_dir / "multiday_data_manifest.json")
    run_manifest = read_json(final_dir / "run_manifest.json")
    readiness = read_json(submission_dir / "readiness_report.json")
    stats = read_json(final_dir / "statistical_tests.json")
    risk_metrics = read_json(risk_dir / "risk_metrics.json")
    neural_protocol = read_json(final_dir / "neural_tuning_protocol.json")
    model_rows = read_csv(final_dir / "model_metrics.csv")
    per_sample_rows = read_csv(final_dir / "per_sample_errors.csv")
    horizon_rows = read_csv(final_dir / "error_summary_by_horizon.csv")
    group_rows = read_csv(final_dir / "error_summary_by_group.csv")
    risk_scenario_rows = read_csv(risk_dir / "risk_scenarios.csv")

    figure_paths = make_figures(figures_dir, model_rows, horizon_rows, group_rows, risk_metrics, risk_dir)

    manuscript = make_manuscript(
        generated_at,
        data_manifest,
        run_manifest,
        readiness,
        model_rows,
        stats,
        risk_metrics,
        group_rows,
        per_sample_rows,
        risk_scenario_rows,
    )
    manuscript_path = paper_dir / "jon_manuscript.md"
    write_text(manuscript_path, manuscript)

    chinese_manuscript = make_chinese_manuscript(
        generated_at,
        data_manifest,
        run_manifest,
        readiness,
        model_rows,
        stats,
        risk_metrics,
    )
    chinese_manuscript_path = paper_dir / "jon_manuscript_zh.md"
    write_text(chinese_manuscript_path, chinese_manuscript)

    chinese_interpretation = make_chinese_interpretation(generated_at, data_manifest, model_rows, risk_metrics)
    chinese_interpretation_path = paper_dir / "jon_manuscript_zh_interpretation.md"
    write_text(chinese_interpretation_path, chinese_interpretation)

    supplementary = make_supplementary(generated_at, data_manifest, model_rows, stats, risk_metrics, neural_protocol)
    supplementary_path = paper_dir / "jon_supplementary_materials.md"
    write_text(supplementary_path, supplementary)

    supplemental_files = [
        audit_dir / "multiday_data_manifest.json",
        final_dir / "model_metrics.csv",
        final_dir / "generalization_metrics.csv",
        final_dir / "error_summary_by_horizon.csv",
        final_dir / "error_summary_by_group.csv",
        final_dir / "statistical_tests.json",
        final_dir / "neural_tuning_protocol.json",
        final_dir / "neural_tuning_results.csv",
        risk_dir / "risk_metrics.json",
        risk_dir / "risk_scenarios.csv",
        submission_dir / "readiness_report.json",
    ]
    zip_path = paper_dir / "jon_supplementary_materials.zip"
    zip_size_mb = make_zip(zip_path, supplementary_path, supplemental_files)

    wc = word_count(manuscript)
    zh_char_count = len(re.findall(r"[\u4e00-\u9fff]", chinese_manuscript))
    cover_letter = make_cover_letter(generated_at)
    checklist = make_checklist(generated_at, wc, zip_size_mb, figure_paths)
    polish_workflow = make_authorial_polish_workflow(generated_at)
    write_text(paper_dir / "jon_cover_letter.md", cover_letter)
    write_text(paper_dir / "jon_submission_checklist.md", checklist)
    write_text(paper_dir / "jon_authorial_polish_workflow.md", polish_workflow)

    manifest = {
        "created_at": generated_at,
        "target_journal": "The Journal of Navigation",
        "article_type": "Research Article",
        "readiness_report_status": readiness.get("overall_status"),
        "blocking_gaps": readiness.get("blocking_gaps", []),
        "word_count_estimate": wc,
        "chinese_character_count_estimate": zh_char_count,
        "supplementary_zip_size_mb": zip_size_mb,
        "artifacts": {
            "manuscript_md": str(manuscript_path),
            "chinese_manuscript_md": str(chinese_manuscript_path),
            "chinese_interpretation_md": str(chinese_interpretation_path),
            "cover_letter_md": str(paper_dir / "jon_cover_letter.md"),
            "checklist_md": str(paper_dir / "jon_submission_checklist.md"),
            "authorial_polish_workflow_md": str(paper_dir / "jon_authorial_polish_workflow.md"),
            "scholarone_metadata_md": str(paper_dir / "jon_scholarone_metadata.md"),
            "reference_audit_md": str(paper_dir / "jon_reference_audit.md"),
            "submission_guide_zh_md": str(paper_dir / "jon_submission_guide_zh.md"),
            "supplementary_md": str(supplementary_path),
            "supplementary_zip": str(zip_path),
            "figures": figure_paths,
        },
        "source_artifacts": {
            "data_manifest": str(audit_dir / "multiday_data_manifest.json"),
            "run_manifest": str(final_dir / "run_manifest.json"),
            "model_metrics": str(final_dir / "model_metrics.csv"),
            "statistical_tests": str(final_dir / "statistical_tests.json"),
            "risk_metrics": str(risk_dir / "risk_metrics.json"),
            "readiness_report": str(submission_dir / "readiness_report.json"),
        },
        "claim_boundary": [
            "Supports reproducible historical AIS trajectory-prediction benchmarking.",
            "Supports temporal and vessel-disjoint holdout results for the current artifact.",
            "Supports AIS-derived CPA/TCPA risk-warning metrics.",
            "Does not validate live AIS deployment.",
            "Does not validate autonomous collision avoidance or COLREGs compliance.",
            "Does not prove all-day, seasonal, or global generalisation.",
        ],
        "human_before_upload": [
            "Add ORCID in ScholarOne if available.",
            "Re-check no-specific-funding and no-competing-interests statements for accuracy.",
            "Create and insert a Zenodo or equivalent DOI if an archival DOI is obtained before submission.",
            "Review final ScholarOne-generated PDF before clicking submit.",
        ],
    }
    write_text(submission_dir / "jon_submission_manifest.json", json.dumps(manifest, indent=2))

    print(f"Wrote {manuscript_path}")
    print(f"Wrote {chinese_manuscript_path}")
    print(f"Wrote {chinese_interpretation_path}")
    print(f"Wrote {paper_dir / 'jon_cover_letter.md'}")
    print(f"Wrote {paper_dir / 'jon_submission_checklist.md'}")
    print(f"Wrote {zip_path} ({zip_size_mb:.2f} MB)")
    print(f"Estimated manuscript word count: {wc}")
    print(f"Estimated Chinese manuscript CJK characters: {zh_char_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
