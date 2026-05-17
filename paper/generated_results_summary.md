# Generated Results Summary

Generated at: 2026-05-16T12:56:00.767726+00:00

## Result Table

| Model | Status | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | n test | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| constant_velocity | ok | 306.5 | 659.6 | 691.6 | 195.6 | 9262 |  |
| linear_lstsq | ok | 351.6 | 750.0 | 615.5 | 223.3 | 9262 |  |
| lstm_baseline | ok | 69926.4 | 67654.1 | 233702.3 | 42058.4 | 9262 |  |
| transformer_baseline | ok | 81626.7 | 80757.5 | 131464.7 | 48620.1 | 9262 |  |

## Statistical Summary

- constant_velocity: ADE mean 306.5 m, median 24.430 m, 95% percentile interval [0.000, 2440.5] m.
- linear_lstsq: ADE mean 351.6 m, median 103.8 m, 95% percentile interval [10.100, 2106.8] m.
- lstm_baseline: ADE mean 69926.4 m, median 43210.5 m, 95% percentile interval [13535.0, 464662.0] m.
- transformer_baseline: ADE mean 81626.7 m, median 47833.7 m, 95% percentile interval [16020.1, 191333.3] m.
- Versus constant_velocity, linear_lstsq mean ADE difference 45.076 m; paired t-test p=6.17e-50.
- Versus constant_velocity, lstm_baseline mean ADE difference 69619.9 m; paired t-test p=3.60e-116.
- Versus constant_velocity, transformer_baseline mean ADE difference 81320.2 m; paired t-test p=<1e-300.

## Artifact Checklist

| Artifact | Exists |
|---|---:|
| `outputs/final/model_metrics.csv` | yes |
| `outputs/final/per_sample_errors.csv` | yes |
| `outputs/final/error_summary_by_horizon.csv` | yes |
| `outputs/final/error_summary_by_group.csv` | yes |
| `outputs/final/statistical_tests.json` | yes |
| `outputs/final/reproducibility_check.json` | yes |
| `outputs/final/data_quality_report.json` | yes |
| `outputs/final/figures/model_ade_bar.png` | yes |
| `outputs/final/figures/error_distributions.png` | yes |
| `outputs/final/figures/figure_manifest.json` | yes |
| `outputs/audit/data_manifest.json` | yes |
| `outputs/audit/feature_schema.json` | yes |
| `outputs/audit/split_manifest.csv` | yes |
| `outputs/audit/environment.json` | yes |
| `outputs/audit/git_status.txt` | yes |
| `outputs/audit/run_command.txt` | yes |

## Manuscript Use Rule

Only cite numbers from the table above when the run manifest shows `is_debug_run=false` and all required artifacts exist.
