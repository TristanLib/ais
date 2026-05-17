# Generalization-Aware AIS Trajectory Prediction and Risk-Warning Benchmark

Generated from repository artifacts at 2026-05-16T14:50:24.936483+00:00.

> Submission-readiness note: this draft is synchronized with the current evidence pack. The readiness audit currently reports `submission_ready_candidate`. If blocking gaps are listed below, the manuscript is a submission draft rather than a final submission.

## Abstract

Short-term vessel trajectory prediction is often evaluated with increasingly complex neural models, but maritime decision-support systems also require auditable data protocols, strong kinematic baselines, generalization tests, and downstream risk-warning evidence. This study presents a reproducible AIS trajectory-prediction pipeline using NOAA MarineCadastre.gov data [@noaa_marinecadastre_ais; @noaa_digitalcoast_marinecadastre]. The current evidence pack keeps vessel identifiers, source dates, timestamps, regions, speed, and turning-intensity metadata so that temporal and vessel-disjoint splits can be evaluated. In the current artifact run, the processed dataset contains 186326 trajectory windows from 7425 MMSI values and source dates 2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05. The best temporal-test ADE in the current run is 1759.7 m from `kalman_filter_cv`, while the best vessel-disjoint ADE is 3109.4 m from `kalman_filter_cv`. A downstream AIS-derived risk-warning evaluation estimates warning precision, recall, false alarms, missed warnings, and CPA error without claiming autonomous collision avoidance.

## 1. Introduction

AIS trajectory prediction is useful only when model comparisons survive simple baselines and when the forecasting errors can be connected to operational quantities such as closest point of approach and warning lead time. The objective of this manuscript is therefore not to assert that a neural architecture wins by default. Instead, it asks whether a transparent AIS evidence pipeline can support stronger claims about generalization and risk-warning behavior than a single-day benchmark.

## 2. Data Protocol

The dataset builder records raw file checksums, row counts, MMSI counts, timestamp ranges, geographic bounds, processed-file checksum, split policy, and scenario-slice metadata. Each trajectory window uses 30 one-minute history steps to forecast 15 one-minute future positions. Coordinates are WGS84 latitude/longitude, and displacement metrics are Haversine meters.

The current processed artifact contains 186326 samples, 7425 unique MMSI values, and source dates 2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05. Temporal split counts are {'train': 130428, 'val': 27948, 'test': 27950}; vessel-disjoint split counts are {'train': 130018, 'val': 28597, 'test': 27711}. Regions represented in the current artifact are east_gulf_coast, hawaii_pacific, other, west_coast.

## 3. Models

The benchmark includes constant velocity, constant acceleration, a Kalman-style constant-velocity smoother, ridge least squares, ordinary least squares, LSTM [@hochreiter1997lstm], GRU [@cho2014gru], and Transformer baselines [@vaswani2017attention], plus a temporal convolutional network baseline [@bai2018tcn]. Neural models are implemented in PyTorch [@paszke2019pytorch]. Linear models are included because simple statistical baselines can be more robust than untuned deep sequence models on short-horizon AIS windows.

## 4. Evaluation Design

The primary trajectory metrics are ADE, FDE, RMSE, and MAE in meters. Generalization is evaluated with both temporal holdout and vessel-disjoint holdout, using aligned per-sample errors for statistical tests. Scenario slices are produced by region, speed bin, and turning-intensity bin.

Risk-warning evaluation uses observed future pairwise separation to define AIS-derived warnings within the forecast horizon. Predicted trajectories are then evaluated for warning precision, recall, false alarms, missed warnings, lead-time error, and CPA error. This is decision-support evidence, not a validated closed-loop collision-avoidance system under COLREGs [@imo_colregs].

## 5. Trajectory Results

| Split | Model | Status | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Train | Val | Test |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| temporal_test | constant_acceleration | ok | 35076.6 | 76360.6 | 737164.2 | 28836.3 | 130428 | 27948 | 27950 |
| temporal_test | constant_velocity | ok | 2751.3 | 4469.8 | 77190.7 | 1742.6 | 130428 | 27948 | 27950 |
| temporal_test | gru_baseline | ok | 25215.5 | 25571.8 | 29923.9 | 14863.6 | 130428 | 27948 | 27950 |
| temporal_test | kalman_filter_cv | ok | 1759.7 | 2704.5 | 22422.7 | 1101.1 | 130428 | 27948 | 27950 |
| temporal_test | linear_lstsq | ok | 3052.3 | 4908.7 | 13349.1 | 1910.3 | 130428 | 27948 | 27950 |
| temporal_test | lstm_baseline | ok | 36039.2 | 36116.8 | 57826.3 | 22176.2 | 130428 | 27948 | 27950 |
| temporal_test | ridge_lstsq | ok | 3141.7 | 5079.5 | 12468.9 | 1965.8 | 130428 | 27948 | 27950 |
| temporal_test | tcn_baseline | ok | 47095.5 | 47078.0 | 48378.2 | 29703.9 | 130428 | 27948 | 27950 |
| temporal_test | transformer_baseline | ok | 56310.7 | 55923.5 | 57766.4 | 34854.0 | 130428 | 27948 | 27950 |
| vessel_disjoint_test | constant_acceleration | ok | 36237.1 | 67988.2 | 1786906.6 | 55853.5 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | constant_velocity | ok | 9553.5 | 17014.2 | 198051.0 | 6181.8 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | gru_baseline | ok | 23989.0 | 24569.4 | 27376.0 | 14911.1 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | kalman_filter_cv | ok | 3109.4 | 5979.6 | 47635.6 | 1941.5 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | linear_lstsq | ok | 8113.3 | 14869.8 | 172502.5 | 5291.8 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | lstm_baseline | ok | 51010.6 | 50890.1 | 77725.3 | 32747.2 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | ridge_lstsq | ok | 3446.9 | 6463.4 | 18536.5 | 2153.1 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | tcn_baseline | ok | 32833.1 | 34379.1 | 34775.3 | 20214.3 | 130018 | 28597 | 27711 |
| vessel_disjoint_test | transformer_baseline | ok | 47559.1 | 45907.0 | 46592.6 | 30158.9 | 130018 | 28597 | 27711 |

## 6. Risk-Warning Results

The current risk-warning artifact evaluates 2000 AIS-derived encounter scenarios from 5000 evaluation samples, using a 0.5 nmi warning threshold and 3.0 nmi search radius.

| Model | Scenarios | Precision | Recall | False alarm rate | Missed warning rate | Mean abs CPA error (nmi) |
|---|---:|---:|---:|---:|---:|---:|
| constant_velocity | 2000 | 0.961 | 0.894 | 0.013 | 0.106 | 0.082 |
| kalman_filter_cv | 2000 | 0.963 | 0.900 | 0.012 | 0.100 | 0.092 |
| linear_lstsq | 2000 | 0.724 | 0.871 | 0.117 | 0.129 | 0.346 |

## 7. Reproducibility and Artifact Synchronization

The synchronized artifact sources are:

- `outputs/audit/multiday_data_manifest.json`
- `outputs/audit/multiday_split_manifest.csv`
- `outputs/final_multiday/model_metrics.csv`
- `outputs/final_multiday/neural_tuning_protocol.json`
- `outputs/final_multiday/generalization_metrics.csv`
- `outputs/final_multiday/statistical_tests.json`
- `outputs/final_risk/risk_metrics.json`
- `outputs/final_risk/risk_scenarios.csv`
- `outputs/latest_predictions/prediction_manifest.json`
- `outputs/final_submission/readiness_report.json`

The high-quality roadmap pipeline command is:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_high_quality_pipeline.sh
```

## 8. Limitations

- No blocking readiness gaps are currently reported by the automated audit.
- Risk-warning results should not be interpreted as autonomous collision-avoidance validation.
- Model rankings reflect the current stratified time-block protocol; full-day or alternative time-block protocols should be run before making broader all-day traffic claims.
- Neural tuning is documented as a validation-set proxy search; strong architecture-superiority claims require broader search evidence and independent external validation.

## 9. Conclusion

The project now provides a synchronized path from AIS data audit to trajectory metrics, generalization analysis, operational risk-warning evaluation, latest-data offline prediction, and manuscript artifacts. The current repository state should be treated according to the readiness audit: if blockers remain, the artifact is a high-quality-journal draft package rather than a submission-ready paper.

## References

References are stored in `paper/references.bib`.
