# High-Quality Journal Roadmap

Updated: 2026-05-16

## Short Answer

Completing manuscript polish alone is not enough to make the current paper a strong high-quality journal submission. The current repository already supports a conservative reproducibility/benchmark paper, but a high-quality maritime engineering or navigation journal will likely expect stronger external validity and practical relevance.

To aim at higher-quality venues such as Ocean Engineering or The Journal of Navigation, the project needs to move from a single-day benchmark evidence pack to a multi-scenario, generalization-aware, operationally meaningful evaluation.

## Implementation Status

As of 2026-05-16, the roadmap has been implemented as an artifact-complete
high-quality-journal candidate package. The automated audit currently reports
`overall_status=submission_ready_candidate` with no `blocking_gaps`.

Implemented:

- Metadata-rich dataset builder: `configs/experiment_multiday.yaml`,
  `scripts/build_multiday_dataset.py`, `data/processed/multiday_high_quality_processed.npz`.
- Temporal and vessel-disjoint split labels plus scenario metadata:
  `outputs/audit/multiday_data_manifest.json`,
  `outputs/audit/multiday_split_manifest.csv`,
  `outputs/audit/split_policy.json`.
- Expanded benchmark runner with CV, constant acceleration, Kalman-style CV,
  ridge/linear least squares, LSTM, GRU, Transformer, and TCN:
  `scripts/final_train_eval_multiday.py`, `src/models/gru.py`, `src/models/tcn.py`,
  `outputs/final_multiday/model_metrics.csv`.
- Neural validation-set proxy tuning:
  `scripts/tune_neural_baselines.py`,
  `outputs/final_multiday/neural_tuning_protocol.json`,
  `outputs/final_multiday/neural_tuning_results.csv`.
- Generalization/error artifacts:
  `outputs/final_multiday/generalization_metrics.csv`,
  `outputs/final_multiday/error_summary_by_horizon.csv`,
  `outputs/final_multiday/error_summary_by_group.csv`,
  `outputs/final_multiday/statistical_tests.json`.
- AIS-derived risk-warning demonstration:
  `scripts/evaluate_risk_warning.py`,
  `outputs/final_risk/risk_scenarios.csv`,
  `outputs/final_risk/risk_metrics.json`,
  `outputs/final_risk/figures/risk_case_studies.png`.
- Offline latest/new AIS prediction export:
  `scripts/predict_latest_ais.py`,
  `outputs/latest_predictions/trajectory_predictions.csv`,
  `outputs/latest_predictions/risk_warnings.csv`,
  `outputs/latest_predictions/prediction_manifest.json`.
- Readiness audit:
  `scripts/audit_high_quality_pack.py`,
  `outputs/final_submission/readiness_report.json`.

Current audit state:

- `outputs/audit/multiday_data_manifest.json`: 186,326 trajectory windows,
  7,425 MMSI values, four NOAA source dates, computed raw checksums, and a
  documented stratified time-block protocol.
- `outputs/final_multiday/run_manifest.json`: `is_debug_run=false`.
- `outputs/final_submission/readiness_report.json`: all five roadmap phases
  pass with no blocking gaps.
- `paper/submission_manuscript.md`, `paper/references.bib`,
  `paper/tables/high_quality_model_metrics.md`, and
  `paper/tables/risk_warning_metrics.md` are regenerated from current artifacts.
- `paper/submission_manuscript_zh.md` and `paper/submission_manuscript_zh.pdf`
  provide a domestic-journal-style Chinese candidate draft. The Chinese PDF is
  exported with an embedded Chinese-capable TrueType font so the body text is
  visible in PDF viewers.

## Current Supported Contribution

The current project can defend this contribution:

> On an audited real AIS protocol, constant velocity is a strong short-term trajectory baseline; naive LSTM and Transformer baselines fail badly; therefore maritime trajectory-prediction studies should archive preprocessing, metric units, split protocols, failures, and per-sample errors before claiming architectural superiority.

Current high-quality evidence:

- The conservative single-day pipeline still passes and remains available for
  the simpler benchmark/reproducibility paper.
- The high-quality candidate pipeline now covers four source dates and both
  temporal and vessel-disjoint holdouts.
- Final high-quality models: CV, constant acceleration, Kalman-style CV,
  ridge/linear least squares, LSTM, GRU, Transformer, and TCN.
- Best temporal-test ADE: `kalman_filter_cv`, 1759.7 m.
- Best vessel-disjoint ADE: `kalman_filter_cv`, 3109.4 m.
- Risk-warning evaluation: 2,000 AIS-derived encounter scenarios; Kalman-style
  CV has precision 0.963, recall 0.900, false-alarm rate 0.012, and missed
  warning rate 0.100 at the configured 0.5 nmi threshold.

## Remaining Caveats for a High-Quality Journal

### 1. Stronger Dataset Protocol

Current status: implemented for four source dates with retained metadata and a
documented stratified time-block sampling protocol.

Target standard:

- Multiple dates or months.
- Multiple regions or port/coastal contexts.
- Temporal holdout split.
- Vessel-disjoint split if MMSI can be retained safely.
- Scenario slices: speed bins, turning intensity, vessel type, route density, region.
- Data manifest records raw file checksums, record counts, MMSI counts, time range, geographic bounds, and split logic.

### 2. Better Model Baseline Fairness

Current status: implemented with expanded baselines, TCN, early stopping, and a
validation-set proxy tuning artifact. Reviewers may still ask for a broader
full-scale neural hyperparameter search.

Target standard:

- Add persistence/CV, constant acceleration, Kalman filter, linear/ridge regression, LSTM, GRU, Transformer, and one stronger modern sequence model.
- Use the same input horizon, forecast horizon, split, metric, and preprocessing.
- Tune neural models with validation-set early stopping and documented search space.
- Save all model configs, checkpoints, and training curves.
- Report failed or unstable configurations honestly.

### 3. External Validity and Generalization

Current status: implemented for temporal and vessel-disjoint holdouts, plus
region/speed/turning scenario slices. Broader claims about all-day or seasonal
traffic still require additional time-block or full-day protocols.

Target standard:

- Train on earlier dates, test on later dates.
- Train on one region, test on another if raw metadata supports it.
- Evaluate vessel-disjoint generalization.
- Report confidence intervals and paired tests across aligned samples.
- Include horizon-wise degradation curves.

### 4. Operational Relevance

Current status: implemented as a downstream risk-warning evaluation using
AIS-derived encounter scenarios. It is not closed-loop autonomous avoidance.

Target standard:

- Add a downstream risk-warning evaluation, not full autonomous avoidance unless closed-loop simulation is rigorous.
- Use actual model predictions to compute CPA/TCPA warning stability.
- Define encounter scenarios or AIS-derived encounter windows.
- Report whether better/worse trajectory prediction changes warning lead time, false alarms, or missed warnings.
- Keep wording as "risk warning" or "decision support" unless closed-loop avoidance is validated.

### 5. Manuscript Quality

Current status: generated manuscript, tables, and bibliography are synchronized.
Human venue-specific editing and formatting are still required before an actual
journal upload.

Target standard:

- Full related work with verified bibliography.
- Clear research questions and contributions.
- Strong limitations section.
- Venue-specific formatting.
- Journal-quality figures and captions.
- Data availability, code availability, author contribution, funding, ethics, and conflict-of-interest statements.

## Recommended High-Quality Journal Plan

### Phase A: Rebuild Data Protocol

Deliverables:

- `configs/experiment_multiday.yaml`
- `scripts/build_multiday_dataset.py`
- `outputs/audit/multiday_data_manifest.json`
- `outputs/audit/split_policy.json`

Acceptance criteria:

- At least 4 weeks or 4 representative days across different traffic conditions.
- Raw files and processed arrays keep enough metadata for temporal and vessel-disjoint splits.
- Data audit reports record count, MMSI count, timestamp range, geographic bounds, and checksums.

### Phase B: Expand Final Benchmark

Deliverables:

- `scripts/final_train_eval_multiday.py`
- `outputs/final_multiday/model_metrics.csv`
- `outputs/final_multiday/per_sample_errors.csv`
- `outputs/final_multiday/training_logs/`
- `outputs/final_multiday/checkpoints/`

Acceptance criteria:

- CV, constant acceleration, Kalman filter, ridge regression, LSTM/GRU, Transformer, and one stronger tuned model are evaluated.
- Neural models have documented hyperparameter search and early stopping.
- Every model has status, config, logs, and metrics.

### Phase C: Generalization and Error Analysis

Deliverables:

- `outputs/final_multiday/generalization_metrics.csv`
- `outputs/final_multiday/error_summary_by_horizon.csv`
- `outputs/final_multiday/error_summary_by_group.csv`
- `outputs/final_multiday/statistical_tests.json`

Acceptance criteria:

- Temporal holdout and vessel-disjoint results exist.
- Scenario slices include speed and turning-intensity bins.
- Claims about generalization are tied directly to result files.

### Phase D: Practical Risk-Warning Demonstration

Deliverables:

- `scripts/evaluate_risk_warning.py`
- `outputs/final_risk/risk_scenarios.csv`
- `outputs/final_risk/risk_metrics.json`
- `outputs/final_risk/figures/risk_case_studies.png`

Acceptance criteria:

- Uses actual model predictions, not manually invented cases only.
- Reports CPA/TCPA warning lead time, false alarms, and missed warnings where labels are definable.
- Does not claim autonomous collision avoidance.

### Phase E: Submission Package

Deliverables:

- `paper/submission_manuscript.md` or `.docx`
- `paper/references.bib`
- `paper/figures/`
- `paper/tables/`
- `outputs/final_submission/readiness_report.json`

Acceptance criteria:

- No unsupported claims.
- Every table and figure is generated.
- Every number in abstract/results maps to final artifacts.
- Target venue formatting is complete.

### Phase F: Domestic Chinese Journal Packaging

Deliverables:

- `paper/submission_manuscript_zh.md`
- `paper/submission_manuscript_zh.pdf`
- `outputs/final_submission/chinese_submission_manifest.json`

Current status:

- Chinese candidate draft exists and follows a domestic journal structure:
  Chinese title, author/affiliation placeholders, funding placeholder, Chinese
  abstract and keywords, English abstract, introduction, data/task definition,
  method, experiments, discussion, conclusion, data/code availability, conflict
  statement, and references.
- PDF export has been fixed after the initial blank-page rendering problem by
  embedding a local Chinese-capable TrueType font.

Acceptance criteria before actual domestic submission:

- Target journal selected and template locked.
- Author, affiliation, funding, acknowledgements, and corresponding-author
  details completed.
- Related work expanded with Chinese and international AIS trajectory-prediction
  literature.
- References converted to the target journal style, usually GB/T 7714 unless
  specified otherwise.
- Tables and figures numbered and formatted according to the target journal.
- All abstract/result/conclusion numbers verified against repository artifacts.

## Real-World Meaning of the Project

The project is meaningful as a decision-support and evaluation tool, especially for:

- Benchmarking whether complex trajectory predictors truly beat simple motion baselines.
- Auditing AIS preprocessing and metric definitions.
- Short-horizon trajectory extrapolation from historical or recently published AIS data.
- Offline traffic analysis around ports/coastal waters.
- Future risk-warning studies using predicted trajectories plus CPA/TCPA.

It is not yet an operational autonomous collision-avoidance system.

## Can It Predict with the Latest Data?

Yes, with boundaries.

What it can do now or with modest additions:

- Download or ingest a new AIS CSV file in the same schema.
- Clean, slice, and process trajectories.
- Run the trained or retrained conservative models.
- Predict 15-minute future positions for trajectories present in that data.
- Compare CV/linear/LSTM/Transformer results and generate an evidence pack.

What it cannot do yet:

- It does not consume live AIS streams out of the box.
- NOAA MarineCadastre data are historical/published data, not a live feed.
- It does not currently expose a real-time API or dashboard.
- It does not yet perform validated operational collision avoidance.

Recommended practical next feature:

Create `scripts/predict_latest_ais.py` that accepts a new AIS CSV, applies the existing cleaning/slicing protocol, runs CV and any trained model checkpoints, and writes:

- `outputs/latest_predictions/trajectory_predictions.csv`
- `outputs/latest_predictions/risk_warnings.csv`
- `outputs/latest_predictions/prediction_manifest.json`

This would turn the research pipeline into a repeatable offline prediction tool for newly downloaded AIS data.
