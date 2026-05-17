# Supplementary Material S1: Reproducibility and Evidence Summary

Generated: 2026-05-17T02:48:32.132551+00:00

This supplementary note documents the artifacts used to generate the JON submission candidate. It is designed as a concise companion to the main manuscript. The large per-sample error CSV is intentionally excluded from the zipped supplementary package because it is about 55 MB, which exceeds the current per-file supplementary guideline.

## Dataset Summary

- Processed sample count: 186,326
- Unique MMSI values: 7,425
- Source dates: 2024-01-02, 2024-01-09, 2024-02-06, 2024-03-05
- Regions: east_gulf_coast, hawaii_pacific, other, west_coast
- Temporal split counts: {'train': 130428, 'val': 27948, 'test': 27950}
- Vessel-disjoint split counts: {'train': 130018, 'val': 28597, 'test': 27711}

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

| Split | Model | Mean ADE (m) | Median ADE (m) | 95% ADE interval (m) | FDE (m) |
|---|---|---:|---:|---:|---:|
| Temporal holdout | Kalman-CV | 1,759.7 | NA | NA to NA | 2,704.5 |
| Temporal holdout | CV | 2,751.3 | NA | NA to NA | 4,469.8 |
| Temporal holdout | Ridge | 3,141.7 | NA | NA to NA | 5,079.5 |
| Temporal holdout | OLS | 3,052.3 | NA | NA to NA | 4,908.7 |
| Temporal holdout | GRU | 25,215.5 | NA | NA to NA | 25,571.8 |
| Temporal holdout | LSTM | 36,039.2 | NA | NA to NA | 36,116.8 |
| Temporal holdout | TCN | 47,095.5 | NA | NA to NA | 47,078.0 |
| Temporal holdout | Transformer | 56,310.7 | NA | NA to NA | 55,923.5 |
| Temporal holdout | CA | 35,076.6 | NA | NA to NA | 76,360.6 |
| Vessel-disjoint holdout | Kalman-CV | 3,109.4 | NA | NA to NA | 5,979.6 |
| Vessel-disjoint holdout | CV | 9,553.5 | NA | NA to NA | 17,014.2 |
| Vessel-disjoint holdout | Ridge | 3,446.9 | NA | NA to NA | 6,463.4 |
| Vessel-disjoint holdout | OLS | 8,113.3 | NA | NA to NA | 14,869.8 |
| Vessel-disjoint holdout | GRU | 23,989.0 | NA | NA to NA | 24,569.4 |
| Vessel-disjoint holdout | LSTM | 51,010.6 | NA | NA to NA | 50,890.1 |
| Vessel-disjoint holdout | TCN | 32,833.1 | NA | NA to NA | 34,379.1 |
| Vessel-disjoint holdout | Transformer | 47,559.1 | NA | NA to NA | 45,907.0 |
| Vessel-disjoint holdout | CA | 36,237.1 | NA | NA to NA | 67,988.2 |

## Risk-Warning Metrics

| Model | TP | FP | FN | TN | Precision | Recall | False alarm | Missed warning | CPA error (nmi) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kalman-CV | 468 | 18 | 52 | 1462 | 0.963 | 0.900 | 0.012 | 0.100 | 0.092 |
| CV | 465 | 19 | 55 | 1461 | 0.961 | 0.894 | 0.013 | 0.106 | 0.082 |
| OLS | 453 | 173 | 67 | 1307 | 0.724 | 0.871 | 0.117 | 0.129 | 0.346 |
