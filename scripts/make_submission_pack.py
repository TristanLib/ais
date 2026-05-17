#!/usr/bin/env python3
"""Generate high-quality-roadmap submission artifacts from current evidence."""

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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def model_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Split | Model | Status | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Train | Val | Test |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {model} | {status} | {ade} | {fde} | {rmse} | {mae} | {train} | {val} | {test} |".format(
                split=row.get("split_policy", ""),
                model=row.get("model", ""),
                status=row.get("status", ""),
                ade=fmt(row.get("ade_meters")),
                fde=fmt(row.get("fde_meters")),
                rmse=fmt(row.get("rmse_meters")),
                mae=fmt(row.get("mae_meters")),
                train=row.get("n_train", ""),
                val=row.get("n_val", ""),
                test=row.get("n_test", ""),
            )
        )
    return "\n".join(lines)


def risk_table(risk_metrics: dict[str, Any]) -> str:
    rows = risk_metrics.get("metrics_by_model", {})
    lines = [
        "| Model | Scenarios | Precision | Recall | False alarm rate | Missed warning rate | Mean abs CPA error (nmi) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in sorted(rows.items()):
        lines.append(
            "| {model} | {n} | {precision} | {recall} | {far} | {miss} | {cpa} |".format(
                model=model,
                n=row.get("n_scenarios", ""),
                precision=fmt(row.get("precision")),
                recall=fmt(row.get("recall")),
                far=fmt(row.get("false_alarm_rate")),
                miss=fmt(row.get("missed_warning_rate")),
                cpa=fmt(row.get("mean_abs_cpa_error_nmi")),
            )
        )
    return "\n".join(lines)


def best_by_split(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        split = row.get("split_policy", "")
        if split not in best or as_float(row, "ade_meters") < as_float(best[split], "ade_meters"):
            best[split] = row
    return best


def references_bib() -> str:
    return """@misc{noaa_marinecadastre_ais,
  author       = {{NOAA Office for Coastal Management and Bureau of Ocean Energy Management}},
  title        = {{MarineCadastre.gov AIS Data}},
  year         = {2024},
  howpublished = {\\url{https://marinecadastre.gov/ais/}},
  note         = {Accessed 2026-05-16}
}

@misc{noaa_digitalcoast_marinecadastre,
  author       = {{NOAA Office for Coastal Management}},
  title        = {{Marine Cadastre}},
  year         = {2026},
  howpublished = {\\url{https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html}},
  note         = {Accessed 2026-05-16}
}

@misc{imo_colregs,
  author       = {{International Maritime Organization}},
  title        = {{Convention on the International Regulations for Preventing Collisions at Sea, 1972 (COLREGs)}},
  year         = {1972},
  howpublished = {\\url{https://www.imo.org/en/About/Conventions/Pages/COLREG.aspx}},
  note         = {Accessed 2026-05-16}
}

@article{hochreiter1997lstm,
  author  = {Hochreiter, Sepp and Schmidhuber, J{\\\"u}rgen},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  doi     = {10.1162/neco.1997.9.8.1735}
}

@inproceedings{cho2014gru,
  author    = {Cho, Kyunghyun and van Merrienboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title     = {Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing},
  pages     = {1724--1734},
  year      = {2014},
  url       = {https://aclanthology.org/D14-1179/}
}

@inproceedings{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.03762}
}

@article{bai2018tcn,
  author  = {Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen},
  title   = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
  journal = {arXiv preprint arXiv:1803.01271},
  year    = {2018},
  url     = {https://arxiv.org/abs/1803.01271}
}

@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  title     = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019},
  url       = {https://papers.nips.cc/paper/9015-pytorch}
}

@article{pedregosa2011sklearn,
  author  = {Pedregosa, Fabian and Varoquaux, Gael and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  title   = {Scikit-learn: Machine Learning in Python},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011},
  url     = {https://www.jmlr.org/papers/v12/pedregosa11a.html}
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--final-dir", default="outputs/final_multiday")
    parser.add_argument("--risk-dir", default="outputs/final_risk")
    parser.add_argument("--audit-dir", default="outputs/audit")
    parser.add_argument("--submission-dir", default="outputs/final_submission")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    final_dir = Path(args.final_dir)
    risk_dir = Path(args.risk_dir)
    audit_dir = Path(args.audit_dir)
    submission_dir = Path(args.submission_dir)
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = read_json(audit_dir / "multiday_data_manifest.json")
    run_manifest = read_json(final_dir / "run_manifest.json")
    risk_metrics = read_json(risk_dir / "risk_metrics.json")
    readiness = read_json(submission_dir / "readiness_report.json")
    model_rows = read_csv(final_dir / "model_metrics.csv")
    best = best_by_split(model_rows)
    dataset = data_manifest.get("dataset_summary", {})
    source_dates = dataset.get("source_dates", [])
    risk_generation = risk_metrics.get("scenario_generation", {})

    model_table_text = model_table(model_rows)
    risk_table_text = risk_table(risk_metrics)
    write_text(tables_dir / "high_quality_model_metrics.md", model_table_text + "\n")
    write_text(tables_dir / "risk_warning_metrics.md", risk_table_text + "\n")
    write_text(paper_dir / "references.bib", references_bib())

    readiness_status = readiness.get("overall_status", "not_audited")
    stale_generated_file_gaps = {
        "Submission manuscript has not been generated from the high-quality artifacts.",
        "Verified bibliography is missing.",
    }
    blockers = [gap for gap in readiness.get("blocking_gaps", []) if gap not in stale_generated_file_gaps]
    if blockers:
        readiness_status = "not_submission_ready"
    generated_at = datetime.now(timezone.utc).isoformat()
    temporal_best = best.get("temporal_test", {})
    vessel_best = best.get("vessel_disjoint_test", {})
    manuscript = [
        "# Generalization-Aware AIS Trajectory Prediction and Risk-Warning Benchmark",
        "",
        f"Generated from repository artifacts at {generated_at}.",
        "",
        "> Submission-readiness note: this draft is synchronized with the current evidence pack. "
        f"The readiness audit currently reports `{readiness_status}`. If blocking gaps are listed below, "
        "the manuscript is a submission draft rather than a final submission.",
        "",
        "## Abstract",
        "",
        (
            "Short-term vessel trajectory prediction is often evaluated with increasingly complex neural models, "
            "but maritime decision-support systems also require auditable data protocols, strong kinematic baselines, "
            "generalization tests, and downstream risk-warning evidence. This study presents a reproducible AIS "
            "trajectory-prediction pipeline using NOAA MarineCadastre.gov data [@noaa_marinecadastre_ais; "
            "@noaa_digitalcoast_marinecadastre]. The current evidence pack keeps vessel identifiers, source dates, "
            "timestamps, regions, speed, and turning-intensity metadata so that temporal and vessel-disjoint splits can "
            "be evaluated. In the current artifact run, the processed dataset contains "
            f"{dataset.get('sample_count')} trajectory windows from {dataset.get('unique_mmsi_count')} MMSI values "
            f"and source dates {', '.join(source_dates) if source_dates else 'NA'}. "
            f"The best temporal-test ADE in the current run is {fmt(temporal_best.get('ade_meters'))} m "
            f"from `{temporal_best.get('model', 'NA')}`, while the best vessel-disjoint ADE is "
            f"{fmt(vessel_best.get('ade_meters'))} m from `{vessel_best.get('model', 'NA')}`. "
            "A downstream AIS-derived risk-warning evaluation estimates warning precision, recall, false alarms, "
            "missed warnings, and CPA error without claiming autonomous collision avoidance."
        ),
        "",
        "## 1. Introduction",
        "",
        (
            "AIS trajectory prediction is useful only when model comparisons survive simple baselines and when the "
            "forecasting errors can be connected to operational quantities such as closest point of approach and warning "
            "lead time. The objective of this manuscript is therefore not to assert that a neural architecture wins by "
            "default. Instead, it asks whether a transparent AIS evidence pipeline can support stronger claims about "
            "generalization and risk-warning behavior than a single-day benchmark."
        ),
        "",
        "## 2. Data Protocol",
        "",
        (
            "The dataset builder records raw file checksums, row counts, MMSI counts, timestamp ranges, geographic "
            "bounds, processed-file checksum, split policy, and scenario-slice metadata. Each trajectory window uses "
            f"{run_manifest.get('config', {}).get('experiment', {}).get('history_steps')} one-minute history steps "
            f"to forecast {run_manifest.get('config', {}).get('experiment', {}).get('forecast_steps')} one-minute "
            "future positions. Coordinates are WGS84 latitude/longitude, and displacement metrics are Haversine meters."
        ),
        "",
        (
            f"The current processed artifact contains {dataset.get('sample_count')} samples, "
            f"{dataset.get('unique_mmsi_count')} unique MMSI values, and source dates "
            f"{', '.join(source_dates) if source_dates else 'NA'}. Temporal split counts are "
            f"{dataset.get('temporal_split_counts')}; vessel-disjoint split counts are "
            f"{dataset.get('vessel_split_counts')}. Regions represented in the current artifact are "
            f"{', '.join(dataset.get('regions', []))}."
        ),
        "",
        "## 3. Models",
        "",
        (
            "The benchmark includes constant velocity, constant acceleration, a Kalman-style constant-velocity smoother, "
            "ridge least squares, ordinary least squares, LSTM [@hochreiter1997lstm], GRU [@cho2014gru], and Transformer "
            "baselines [@vaswani2017attention], plus a temporal convolutional network baseline [@bai2018tcn]. Neural "
            "models are implemented in PyTorch [@paszke2019pytorch]. Linear "
            "models are included because simple statistical baselines can be more robust than untuned deep sequence "
            "models on short-horizon AIS windows."
        ),
        "",
        "## 4. Evaluation Design",
        "",
        (
            "The primary trajectory metrics are ADE, FDE, RMSE, and MAE in meters. Generalization is evaluated with "
            "both temporal holdout and vessel-disjoint holdout, using aligned per-sample errors for statistical tests. "
            "Scenario slices are produced by region, speed bin, and turning-intensity bin."
        ),
        "",
        (
            "Risk-warning evaluation uses observed future pairwise separation to define AIS-derived warnings within "
            "the forecast horizon. Predicted trajectories are then evaluated for warning precision, recall, false alarms, "
            "missed warnings, lead-time error, and CPA error. This is decision-support evidence, not a validated "
            "closed-loop collision-avoidance system under COLREGs [@imo_colregs]."
        ),
        "",
        "## 5. Trajectory Results",
        "",
        model_table_text,
        "",
        "## 6. Risk-Warning Results",
        "",
        (
            f"The current risk-warning artifact evaluates {risk_generation.get('scenario_count')} AIS-derived encounter "
            f"scenarios from {risk_generation.get('evaluated_samples')} evaluation samples, using a "
            f"{risk_generation.get('warning_threshold_nmi')} nmi warning threshold and "
            f"{risk_generation.get('search_radius_nmi')} nmi search radius."
        ),
        "",
        risk_table_text,
        "",
        "## 7. Reproducibility and Artifact Synchronization",
        "",
        "The synchronized artifact sources are:",
        "",
        "- `outputs/audit/multiday_data_manifest.json`",
        "- `outputs/audit/multiday_split_manifest.csv`",
        "- `outputs/final_multiday/model_metrics.csv`",
        "- `outputs/final_multiday/neural_tuning_protocol.json`",
        "- `outputs/final_multiday/generalization_metrics.csv`",
        "- `outputs/final_multiday/statistical_tests.json`",
        "- `outputs/final_risk/risk_metrics.json`",
        "- `outputs/final_risk/risk_scenarios.csv`",
        "- `outputs/latest_predictions/prediction_manifest.json`",
        "- `outputs/final_submission/readiness_report.json`",
        "",
        "The high-quality roadmap pipeline command is:",
        "",
        "```bash",
        "PYTHON_BIN=.venv/bin/python bash scripts/run_high_quality_pipeline.sh",
        "```",
        "",
        "## 8. Limitations",
        "",
    ]
    if blockers:
        manuscript.extend([f"- {gap}" for gap in blockers])
    else:
        manuscript.append("- No blocking readiness gaps are currently reported by the automated audit.")
    manuscript.extend(
        [
            "- Risk-warning results should not be interpreted as autonomous collision-avoidance validation.",
            "- Model rankings reflect the current stratified time-block protocol; full-day or alternative time-block protocols should be run before making broader all-day traffic claims.",
            "- Neural tuning is documented as a validation-set proxy search; strong architecture-superiority claims require broader search evidence and independent external validation.",
            "",
            "## 9. Conclusion",
            "",
            (
                "The project now provides a synchronized path from AIS data audit to trajectory metrics, generalization "
                "analysis, operational risk-warning evaluation, latest-data offline prediction, and manuscript artifacts. "
                "The current repository state should be treated according to the readiness audit: if blockers remain, "
                "the artifact is a high-quality-journal draft package rather than a submission-ready paper."
            ),
            "",
            "## References",
            "",
            "References are stored in `paper/references.bib`.",
            "",
        ]
    )

    manuscript_path = paper_dir / "submission_manuscript.md"
    write_text(manuscript_path, "\n".join(manuscript))
    submission_manifest = {
        "created_at": generated_at,
        "manuscript": str(manuscript_path),
        "references": str(paper_dir / "references.bib"),
        "tables": {
            "model_metrics": str(tables_dir / "high_quality_model_metrics.md"),
            "risk_warning": str(tables_dir / "risk_warning_metrics.md"),
        },
        "source_artifacts": {
            "data_manifest": str(audit_dir / "multiday_data_manifest.json"),
            "model_metrics": str(final_dir / "model_metrics.csv"),
            "risk_metrics": str(risk_dir / "risk_metrics.json"),
            "readiness_report": str(submission_dir / "readiness_report.json"),
        },
        "readiness_status_at_generation": readiness_status,
        "blocking_gaps_at_generation": blockers,
        "is_debug_run": run_manifest.get("is_debug_run"),
    }
    write_text(submission_dir / "submission_manifest.json", json.dumps(submission_manifest, indent=2))
    print(f"Submission manuscript written to {manuscript_path}")
    print(f"References written to {paper_dir / 'references.bib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
