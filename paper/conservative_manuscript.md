# A Reproducible Evaluation of Simple and Deep Learning Baselines for Short-Term Vessel Trajectory Prediction from AIS Data

Generated from repository artifacts at 2026-05-16T12:56:00.793704+00:00.

## Abstract

Short-term vessel trajectory prediction is often presented as a natural application for deep sequence models, yet operational maritime datasets can make simple kinematic baselines difficult to beat. This study evaluates four baselines on an audited AIS trajectory dataset with 43216 training samples, 9260 validation samples, and 9262 test samples. Using a 30-minute input window and a 15-minute forecast horizon, the constant-velocity baseline achieved 306.5 m ADE and 659.6 m FDE. Linear least squares reached 351.6 m ADE, while the deliberately conservative LSTM and Transformer baselines produced 69926.4 m and 81626.7 m ADE, respectively. The results support a conservative conclusion: before claiming architectural superiority, AIS prediction studies should archive preprocessing, metric definitions, model failures, and per-sample error evidence.

## 1. Introduction

AIS data provide frequent position, speed, and heading observations for vessels, making them attractive for short-horizon forecasting and downstream navigational decision support. The practical challenge is not only to train a model, but to prove that the model comparison is reproducible and that reported meter-level errors are computed from consistent geographic units. This manuscript therefore narrows the project to a benchmark-style claim: under the audited current protocol, simple kinematic models are strong baselines, and naive deep learning runs can fail badly enough that they should be reported as evidence rather than hidden.

## 2. Data and Reproducible Pipeline

The processed data file is `data/processed/real_noaa_processed.npz` with SHA256 `21c7328861915196c551e85b4b2c9e9ba34441fce41bca7f0b1e3c85e196f8f8`. The raw AIS file is `data/raw/real_ais_noaa_2024.csv` with SHA256 `c7ed5be12c89720652b029782a65d0641fc64de58e556b3150c70e3512b84bae`. The raw profile contains 7295616 AIS records from 15130 MMSI values, spanning 2024-01-02T00:00:00 to 2024-01-02T23:59:59. The saved split protocol is `precomputed_npz_splits` with 43216 training, 9260 validation, and 9262 test samples. The processed NPZ does not contain MMSI or timestamp metadata, so this paper does not claim vessel-disjoint splits, temporal cross-validation, regional generalization, or group-specific performance beyond the all-sample summary.

The first two position dimensions were audited as `degrees_latlon_wgs84`. The inferred coordinate range is latitude 13.44598 to 49.5211 and longitude -158.11991 to 144.63076. The final metric distance is `haversine_meters`.

## 3. Models

The main comparison includes constant velocity, linear least squares, an LSTM baseline, and a Transformer baseline. The LSTM and Transformer are intentionally treated as conservative baselines rather than optimized architectures. Their role is to test whether the current repository can regenerate and archive deep-learning behavior, including poor performance, under a single auditable command.

## 4. Metrics and Statistics

ADE and FDE are Haversine displacement errors in meters between predicted and target WGS84 latitude/longitude positions. RMSE and MAE are computed from local north/east component errors in meters. The statistical report uses aligned per-sample errors and paired tests against the constant-velocity reference, with Bonferroni correction metadata stored in `outputs/final/statistical_tests.json`.

## 5. Results

| Model | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Test samples |
|---|---:|---:|---:|---:|---:|
| constant_velocity | 306.5 | 659.6 | 691.6 | 195.6 | 9262 |
| linear_lstsq | 351.6 | 750.0 | 615.5 | 223.3 | 9262 |
| lstm_baseline | 69926.4 | 67654.1 | 233702.3 | 42058.4 | 9262 |
| transformer_baseline | 81626.7 | 80757.5 | 131464.7 | 48620.1 | 9262 |

The constant-velocity model is the strongest model by ADE in the current final evidence pack. Linear least squares is worse by 45.08 m mean ADE. The LSTM and Transformer baselines are much worse than the kinematic baselines, with mean ADE differences of 69619.9 m and 81320.2 m, respectively.

Generated figures:

- `outputs/final/figures/model_ade_bar.png`
- `outputs/final/figures/error_distributions.png`

## 6. Discussion

The current evidence supports a reproducibility-centered interpretation. Constant velocity remains highly competitive for short-horizon AIS forecasting, while the neural baselines demonstrate that model complexity alone is not evidence of better navigational prediction. Because failures are archived with logs and checkpoints, the repository now supports a publishable negative or cautionary result instead of an unsupported architecture-win story.

## 7. Limitations

- The processed NPZ lacks MMSI, timestamps, vessel type, and region fields.
- The paper does not claim temporal cross-validation, vessel-disjoint generalization, or regional robustness.
- GNN, STT, PINN, and recovery experiments are excluded from the main ranking until full-data evidence is regenerated.
- CPA/TCPA and collision-avoidance modules are treated as future downstream work, not as proven operational avoidance.
- The LSTM and Transformer settings are conservative baselines, not tuned state-of-the-art architectures.

## 8. Reproducibility Statement

The final evidence pack was generated with:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh
```

The run is marked `is_debug_run=False`. The Python executable was `/Users/yuebao/research/ship-prediction-avoidance/.venv/bin/python` running Python 3.11.15 (main, Mar  3 2026, 00:52:57) [Clang 21.0.0 (clang-2100.0.123.102)]. Key package versions include numpy 2.4.5, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0, matplotlib 3.10.9, PyYAML 6.0.3, and torch 2.12.0.

The authoritative artifact map is `outputs/final/publication_readiness_report.json`.

## 9. Conclusion

The conservative evidence pack shows that the current project can support a reproducible AIS trajectory prediction benchmark paper. The strongest supported claim is not that deep learning wins, but that auditable preprocessing, metric definitions, and failure reporting are essential before maritime trajectory-prediction models are used to support stronger navigational or collision-avoidance claims.

## References

1. NOAA Office for Coastal Management and BOEM, MarineCadastre.gov AIS data: https://marinecadastre.gov/ais/
2. NOAA Digital Coast Marine Cadastre overview: https://www.coast.noaa.gov/digitalcoast/data/marine-cadastre.html
3. PyTorch project: https://pytorch.org/projects/pytorch/
