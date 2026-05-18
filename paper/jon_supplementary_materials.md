# Supplementary Material S1: Reproducibility and Evidence Summary

Generated: 2026-05-18T09:48:54.134269+00:00

This supplementary note documents the artifacts used to generate the manuscript. It is designed as a concise companion to the main manuscript. The large per-sample error CSV is intentionally excluded from the zipped supplementary package because it is about 55 MB, which exceeds the per-file supplementary guideline.

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
| Temporal holdout | Kalman-CV | 1,759.7 | 8.2 | 0.6 to 5,673.6 | 2,704.5 |
| Temporal holdout | CV | 2,751.3 | 20.5 | 0.7 to 5,357.1 | 4,469.8 |
| Temporal holdout | Ridge | 3,141.7 | 1,563.1 | 451.4 to 11,189.3 | 5,079.5 |
| Temporal holdout | OLS | 3,052.3 | 1,439.3 | 426.1 to 11,320.0 | 4,908.7 |
| Temporal holdout | GRU | 25,215.5 | 19,597.5 | 3,979.9 to 92,526.6 | 25,571.8 |
| Temporal holdout | LSTM | 36,039.2 | 25,660.5 | 4,365.6 to 116,019.1 | 36,116.8 |
| Temporal holdout | TCN | 47,095.5 | 32,265.2 | 9,361.0 to 149,135.1 | 47,078.0 |
| Temporal holdout | Transformer | 56,310.7 | 35,643.0 | 11,299.8 to 210,486.8 | 55,923.5 |
| Temporal holdout | CA | 35,076.6 | 166.6 | 0.9 to 64,305.6 | 76,360.6 |
| Vessel-disjoint holdout | Kalman-CV | 3,109.4 | 9.2 | 0.6 to 11,946.7 | 5,979.6 |
| Vessel-disjoint holdout | CV | 9,553.5 | 22.8 | 0.8 to 12,053.7 | 17,014.2 |
| Vessel-disjoint holdout | Ridge | 3,446.9 | 1,277.5 | 319.1 to 14,385.8 | 6,463.4 |
| Vessel-disjoint holdout | OLS | 8,113.3 | 930.0 | 261.2 to 12,478.9 | 14,869.8 |
| Vessel-disjoint holdout | GRU | 23,989.0 | 19,508.3 | 5,810.3 to 67,283.9 | 24,569.4 |
| Vessel-disjoint holdout | LSTM | 51,010.6 | 35,421.3 | 8,462.8 to 151,738.0 | 50,890.1 |
| Vessel-disjoint holdout | TCN | 32,833.1 | 24,902.1 | 4,013.2 to 95,171.4 | 34,379.1 |
| Vessel-disjoint holdout | Transformer | 47,559.1 | 33,691.0 | 10,991.1 to 138,971.6 | 45,907.0 |
| Vessel-disjoint holdout | CA | 36,237.1 | 175.1 | 1.0 to 37,008.8 | 67,988.2 |

## Neural Baseline Configuration Summary

The neural sequence baselines use the same 30-step input and 15-step output task as the other models. The proxy search used validation ADE for configuration selection, with final claims tied to the full temporal and vessel-disjoint test runs.

| Model | Selected architecture/configuration | Optimizer | Learning rate | Batch | Epochs | Early stopping | Proxy validation ADE (m) |
|---|---|---|---:|---:|---:|---:|---:|
| GRU | dropout=0.1, hidden_size=64, n_layers=1 | Adam | 0.001 | 256 | 8 | 2 | 159,521.9 |
| LSTM | dropout=0.0, hidden_size=64, n_layers=1 | Adam | 0.001 | 256 | 8 | 2 | 136,080.5 |
| TCN | channels=[48, 64], dropout=0.0, kernel_size=3 | Adam | 0.001 | 256 | 8 | 2 | 99,683.1 |
| Transformer | d_ff=128, d_model=64, dropout=0.0, n_heads=4, n_layers=1 | Adam | 0.001 | 256 | 8 | 2 | 178,999.2 |

## Risk-Warning Metrics

| Model | TP | FP | FN | TN | Precision | Recall | False alarm | Missed warning | Lead-time error (min) | CPA error (nmi) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kalman-CV | 468 | 18 | 52 | 1462 | 0.963 | 0.900 | 0.012 | 0.100 | 0.083 | 0.092 |
| CV | 465 | 19 | 55 | 1461 | 0.961 | 0.894 | 0.013 | 0.106 | 0.045 | 0.082 |
| OLS | 453 | 173 | 67 | 1307 | 0.724 | 0.871 | 0.117 | 0.129 | 0.302 | 0.346 |
