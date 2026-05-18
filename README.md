# A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support

This repository contains the code, configurations, generated figures, compact evidence files, and manuscript artifacts for a reproducible AIS trajectory-prediction benchmark prepared for submission to *The Journal of Navigation*.

The current public citation tag is:

`jon-submission-v1.3`

## Core Claim

Short-horizon AIS trajectory prediction should be evaluated with strong simple baselines, auditable split definitions, and downstream navigation-warning metrics. In the current four-date NOAA AIS benchmark, a Kalman-style constant-velocity baseline is the strongest mean-ADE model under both temporal and vessel-disjoint holdouts. The documented neural sequence baselines do not outperform the strongest motion baselines under this protocol.

This is not an autonomous collision-avoidance validation. The risk-warning component is an AIS-derived CPA/TCPA decision-support evaluation.

## Current Results

### Trajectory Prediction

| Split | Best mean-ADE model | ADE (m) | FDE (m) |
|---|---|---:|---:|
| Temporal holdout | Kalman-CV | 1,759.7 | 2,704.5 |
| Vessel-disjoint holdout | Kalman-CV | 3,109.4 | 5,979.6 |

Representative baselines:

| Split | Model | ADE (m) | Median ADE (m) |
|---|---|---:|---:|
| Temporal holdout | CV | 2,751.3 | 20.5 |
| Temporal holdout | Ridge | 3,141.7 | 1,563.1 |
| Temporal holdout | Transformer | 56,310.7 | 35,643.0 |
| Vessel-disjoint holdout | CV | 9,553.5 | 22.8 |
| Vessel-disjoint holdout | Ridge | 3,446.9 | 1,277.5 |
| Vessel-disjoint holdout | Transformer | 47,559.1 | 33,691.0 |

### Risk-Warning Evaluation

The risk-warning slice contains 2,000 AIS-derived encounter scenarios. With a 0.5 nautical-mile warning threshold, Kalman-CV reaches:

- Precision: 0.963
- Recall: 0.900
- False-alarm rate: 0.012
- Missed-warning rate: 0.100
- Mean absolute CPA error: 0.092 nmi

## Dataset

The evidence package is generated from public NOAA MarineCadastre.gov historical AIS data:

- Source dates: 2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05
- Processed trajectory windows: 186,326
- Unique MMSI values: 7,425
- Input sequence: 30 one-minute history points
- Forecast horizon: 15 one-minute future positions
- Metrics: Haversine ADE/FDE and local north/east RMSE/MAE in metres

The repository intentionally does not redistribute raw NOAA AIS files, processed NumPy arrays, large per-sample error CSV files, model checkpoints, or local environments.

## Submission Artifacts

The current Journal of Navigation package is in `paper/`:

- `paper/jon_manuscript.md`
- `paper/jon_manuscript.docx`
- `paper/jon_manuscript.pdf`
- `paper/jon_supplementary_materials.md`
- `paper/jon_supplementary_materials.zip`
- `paper/jon_cover_letter.md`
- `paper/jon_scholarone_metadata.md`
- `paper/jon_submission_checklist.md`

The manuscript includes author metadata for Li Bo, China Maritime Service Center, the no-specific-grant funding statement, no-competing-interests declaration, author-contribution statement, acknowledgement, and AI-use declaration.

## Reproduce the Evidence Package

Create an environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the current high-quality/JON evidence pipeline:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
DATES="2024-01-02 2024-01-09 2024-02-06 2024-03-05" \
bash scripts/run_high_quality_pipeline.sh
```

Regenerate the JON manuscript package from existing evidence files:

```bash
.venv/bin/python scripts/make_jon_submission_pack.py
pandoc --resource-path=paper paper/jon_manuscript.md -o paper/jon_manuscript.docx
```

Smoke tests may use smaller caps, but capped runs are not paper evidence. Submission claims should be tied to `outputs/final_submission/readiness_report.json` reporting `overall_status=submission_ready_candidate` and no `blocking_gaps`.

## Evidence Files

Important compact evidence files include:

- `outputs/audit/multiday_data_manifest.json`
- `outputs/audit/split_policy.json`
- `outputs/final_multiday/model_metrics.csv`
- `outputs/final_multiday/generalization_metrics.csv`
- `outputs/final_multiday/error_summary_by_horizon.csv`
- `outputs/final_multiday/error_summary_by_group.csv`
- `outputs/final_multiday/statistical_tests.json`
- `outputs/final_multiday/neural_tuning_protocol.json`
- `outputs/final_multiday/neural_tuning_results.csv`
- `outputs/final_risk/risk_metrics.json`
- `outputs/final_risk/risk_scenarios.csv`
- `outputs/final_submission/readiness_report.json`
- `outputs/final_submission/jon_submission_manifest.json`

## Project Structure

```text
├── configs/                  # Experiment configuration files
├── data/                     # Data notes; raw/processed AIS files are excluded
├── outputs/audit/            # Dataset and split manifests
├── outputs/final_multiday/   # Compact trajectory benchmark evidence
├── outputs/final_risk/       # CPA/TCPA risk-warning evidence
├── outputs/final_submission/ # Readiness and submission manifests
├── paper/                    # Manuscript, figures, checklist, supplementary files
├── scripts/                  # Rebuild, evaluation, and manuscript-generation scripts
└── src/                      # Data, feature, model, risk, and avoidance modules
```

## Citation

Until a separate DOI is minted, cite the stable GitHub tag:

```bibtex
@misc{li_ais_navigation_benchmark_2026,
  title        = {A Reproducible AIS Trajectory Prediction Benchmark for Navigation Risk-Warning Support},
  author       = {Li, Bo},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/TristanLib/ais},
  note         = {Tag: jon-submission-v1.3}
}
```

## License

MIT License. See `LICENSE`.
