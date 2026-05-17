# Conservative Paper Scope

## Working Title

A Reproducible Evaluation of Simple and Deep Learning Baselines for Short-Term Vessel Trajectory Prediction from AIS Data

## Core Thesis

Simple kinematic baselines are strong and reliable for short-term vessel trajectory prediction on real AIS data. Deep learning models can fail catastrophically when training, preprocessing, or coordinate handling are insufficiently controlled. The paper contributes a reproducible evaluation pipeline and an auditable evidence trail before making any architectural-superiority claims.

## Main Claims Allowed

- The project evaluates real AIS trajectory prediction with a fixed data protocol.
- Constant-velocity and linear least-squares baselines are reproducible from repository scripts.
- The processed trajectory arrays use WGS84 latitude/longitude coordinates, and final ADE/FDE/RMSE are computed with Haversine-meter distance.
- Model metrics, per-sample errors, statistical summaries, environment reports, and data manifests are generated into `outputs/final/` and `outputs/audit/`.
- Existing deep-learning failure results can be discussed as historical or preliminary only after they are regenerated into the final evidence pack.

## Claims Not Allowed Yet

- Properly trained LSTM achieves 9.4 m ADE.
- Training methodology causes a verified 23,000x improvement.
- GNN/STT/PINN beat CV on full real-data experiments.
- 5-fold temporal cross-validation or regional generalization has been completed.
- Collision avoidance success has been demonstrated.

## Evidence Contract

Every numerical claim in the manuscript must be traceable to one of these files:

- `outputs/final/model_metrics.csv`
- `outputs/final/per_sample_errors.csv`
- `outputs/final/statistical_tests.json`
- `outputs/final/reproducibility_check.json`
- `outputs/final/error_summary_by_horizon.csv`
- `outputs/final/error_summary_by_group.csv`
- `outputs/final/tables/model_metrics.md`
- `outputs/final/figures/model_ade_bar.png`
- `outputs/final/figures/error_distributions.png`
- `paper/generated_results_summary.md`
- `paper/conservative_manuscript.md`
- `outputs/final/publication_readiness_report.json`
- `outputs/audit/data_manifest.json`
- `outputs/audit/environment.json`

If a number is not present in the evidence pack, it must not appear in the abstract, results, or conclusion.

## Current Status

- The project-local `.venv` created with Python 3.11 passes `scripts/check_env.py`.
- The current `outputs/final/` evidence pack was generated without debug sample caps.
- LSTM and Transformer final runs complete and are archived with logs, checkpoints, and per-sample errors.
- The final readiness audit passes with `error_count=0`.
