# Conservative Publication Implementation Plan

## Implementation Status as of 2026-05-16

This plan has been implemented as a working conservative publication pipeline.
For the separate high-quality-journal target, see
`HIGH_QUALITY_JOURNAL_ROADMAP.md` and
`outputs/final_submission/readiness_report.json`. That route now has an
artifact-complete candidate package; the current audit reports
`overall_status=submission_ready_candidate` with no `blocking_gaps`, while still
requiring human venue polish and additional evidence for broader all-day,
architecture-superiority, or autonomous-avoidance claims.

Authoritative command:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh
```

Current verification state:

- `outputs/final/publication_readiness_report.json`: `status=pass`, `error_count=0`, 63 checks passed.
- `outputs/final/reproducibility_check.json`: same-seed metric stability passed for all four final models.
- `outputs/final/run_manifest.json`: `is_debug_run=false`.
- `outputs/audit/environment.json`: project-local Python 3.11 `.venv` passes required module checks.
- `outputs/audit/data_manifest.json`: raw and processed data checksums are recorded; raw CSV profile records 7,295,616 AIS rows from 15,130 MMSI values on 2024-01-02; coordinates are verified as WGS84 latitude/longitude.
- `paper/conservative_manuscript.md`: generated from repository artifacts and contains no unsupported 9.4 m, 23,000x, GNN/STT/PINN superiority, temporal CV, regional generalization, or collision-avoidance success claims.

Current final full-run metrics:

| Model | ADE (m) | FDE (m) | Interpretation |
|---|---:|---:|---|
| Constant Velocity | 306.5 | 659.6 | Main supported baseline result |
| Linear Least Squares | 351.6 | 750.0 | Main supported baseline result |
| LSTM baseline | 69,926.4 | 67,654.1 | Supported failure/cautionary result |
| Transformer baseline | 81,626.7 | 80,757.5 | Supported failure/cautionary result |

The paper is now evidence-backed as a conservative reproducibility and benchmark manuscript. It is not yet a polished submission package for a high-bar journal: related work, venue-specific formatting, bibliography quality, and narrative polish still need human revision before submission.

## 0. Direction

This plan chooses the conservative route: make the paper publishable by only claiming what the project can reproduce and archive. The paper and repository must mutually verify each other.

The recommended core thesis is:

> On large-scale real AIS data, simple kinematic baselines are strong and reliable for short-term vessel trajectory prediction; deep learning models can fail catastrophically under insufficiently controlled training and preprocessing. A reproducible benchmark pipeline is necessary before claiming architectural superiority.

The current project should not yet claim:

- Properly trained LSTM reaches 9.4 m ADE as the main result.
- Training methodology alone causes a verified 23,000x improvement.
- GNN/STT/PINN outperform CV on full real-data experiments.
- 5-fold temporal cross-validation, regional generalization, or bootstrap confidence intervals, unless these are regenerated from scripts and saved as auditable artifacts.

Those claims can become future work only after the repository produces complete evidence.

## 1. Publication Target Strategy

### Primary Paper Type

Use an empirical reproducibility and benchmark paper:

**Working title**

> A Reproducible Evaluation of Simple and Deep Learning Baselines for Short-Term Vessel Trajectory Prediction from AIS Data

### Recommended Scope

The first publishable version should focus on trajectory prediction and reproducibility. CPA/TCPA and collision-avoidance modules can be included only as a limited downstream implication or appendix unless they are evaluated with the same rigor.

### Candidate Venues

| Target | Fit | Required positioning |
|---|---|---|
| Ocean Engineering | Strong if framed as maritime engineering, vessel operation, safety, and validated realistic data experiments. Its scope includes harbour engineering, vessel manoeuvring/safety, automatic control of marine systems, maritime safety, and risk assessment. | Needs rigorous validation and engineering relevance, not just ML benchmarking. Source: https://www.sciencedirect.com/journal/ocean-engineering |
| The Journal of Navigation | Good if framed around navigation science, risk analysis, operational usefulness, and AIS-based prediction. | Needs stronger navigational interpretation and practical usefulness. Source: https://www.cambridge.org/core/journals/journal-of-navigation |
| Journal of Marine Science and Engineering | Good fallback for a reproducibility-heavy marine engineering article; it explicitly encourages detailed experimental reporting for reproducibility. | Lower barrier, but be careful about institutional expectations around MDPI outlets. Source: https://www.mdpi.com/journal/JMSE/about |
| Conference/workshop in intelligent shipping or maritime informatics | Good if time is limited. | Can publish a shorter benchmark version, then extend to journal. |

Final venue selection should happen after Phase 3, when the real evidence strength is known.

## 2. Claim Ledger

Every paper claim must map to a repository artifact. If an artifact does not exist, the claim is not publishable.

| Claim | Current status | Required artifact |
|---|---:|---|
| Dataset uses NOAA AIS real data with fixed date/source and sample counts | Partially supported | `outputs/audit/data_manifest.json`, raw-data checksum, preprocessing config |
| CV baseline achieves about 306 m ADE | Supported by existing JSON | Regenerate with final pipeline: `outputs/final/model_metrics.csv` |
| Linear regression is close to but worse than CV | Supported by existing JSON | Regenerate with final pipeline and exact training script |
| Naive LSTM/Transformer can fail badly | Supported by existing JSON | Regenerate and archive training logs/checkpoints/errors |
| Results are statistically significant | Weakly supported; existing reports are summary-like | Per-sample error CSV and script-generated tests |
| Project is fully reproducible | Not yet supported | Working environment, one-command pipeline, locked dependencies |
| Proper LSTM reaches 9.4 m ADE | Not supported by current artifacts | Exclude from main paper unless fully rerun and archived |
| GNN/STT/PINN beat CV | Weak quick-run only | Exclude or mark exploratory until full real-data training succeeds |
| Collision avoidance success criteria are met | Not sufficiently tied to final experiment | Exclude from main claims unless fully evaluated |

## 3. Implementation Phases

### Phase 0: Freeze Scope and Remove Unsupported Claims

Goal: prevent the paper from outrunning the project.

Tasks:

- Choose the conservative paper title and thesis.
- Move the current "training methodology vs architecture" manuscript to an archive folder or rename it as exploratory draft.
- Create a new paper draft focused on reproducible baseline evaluation.
- Remove or downgrade all unsupported 9.4 m, 23,000x, 5-fold, regional, and ROI claims.
- Decide whether CPA/TCPA appears in the main paper or only as future work.

Acceptance criteria:

- A one-page paper scope note exists.
- The main draft contains no claim without a planned artifact.
- `README.md` and paper abstract tell the same story.

### Phase 1: Repair Reproducibility Foundation

Goal: anyone can run the final experiments from a clean environment.

Tasks:

- Replace broken `ship_env` assumptions with a fresh reproducible environment.
- Add either `requirements.txt` or `environment.yml`.
- Add `scripts/check_env.py` to verify Python, torch, numpy, pandas, sklearn, scipy, matplotlib.
- Add a top-level runner such as `scripts/run_final_experiment.sh`.
- Standardize paths and remove references to missing files such as `run_full_pipeline.sh`, `requirements.txt`, or `env/conda.yaml` unless created.
- Add an experiment config under `configs/experiment_conservative.yaml`.

Required artifacts:

- `requirements.txt` or `environment.yml`
- `outputs/audit/environment.json`
- `outputs/audit/git_status.txt`
- `outputs/audit/run_command.txt`

Acceptance criteria:

- `python scripts/check_env.py` passes.
- The final experiment runner starts from a documented environment.
- README quick-start commands are true.

### Phase 2: Freeze Data Protocol

Goal: one dataset protocol, one split protocol, one metric definition.

Tasks:

- Decide final horizon: prefer the existing real NOAA protocol, `30 min input -> 15 min output`, because the current `real_noaa_processed.npz` already uses it.
- Stop extending 15-step targets to 30 steps for final reported results.
- Add a dataset audit script that records raw file name, size, checksum, record counts, vessel counts, feature schema, split counts, and coordinate units.
- Save train/val/test split manifests.
- Document whether splits are temporal, vessel-disjoint, or random. Do not claim vessel-disjoint or temporal CV unless implemented.
- Confirm metric units. If arrays are already relative meters, document that. If not, convert via local projection or Haversine before computing ADE/FDE/RMSE.

Required artifacts:

- `outputs/audit/data_manifest.json`
- `outputs/audit/feature_schema.json`
- `outputs/audit/split_manifest.parquet` or `.csv`
- `outputs/final/data_quality_report.json`

Acceptance criteria:

- The paper's dataset section exactly matches the manifest.
- The model scripts and metrics use the same horizon.
- No final metric is computed on mixed coordinate units.

### Phase 3: Rebuild Final Experiments

Goal: produce one final, auditable model comparison table.

Models for the conservative paper:

- Constant Velocity
- Linear Regression
- LSTM naive or baseline implementation
- Transformer naive or baseline implementation
- Optional: SVM only if runtime and configuration are stable

Exploratory appendix only:

- GNN
- STT
- PINN
- Properly trained LSTM or Transformer recovery experiments

Tasks:

- Create a single script, e.g. `scripts/final_train_eval.py`, that trains/evaluates all final models.
- Save per-model predictions or per-sample errors.
- Save model configs and checkpoints.
- Save training logs with losses and runtime.
- Ensure no hardcoded numbers are used in paper tables or figures.
- Generate paper tables directly from final result CSV/JSON.

Required artifacts:

- `outputs/final/model_metrics.csv`
- `outputs/final/per_sample_errors.parquet` or `.csv`
- `outputs/final/training_logs/*.json`
- `outputs/final/checkpoints/*`
- `outputs/final/figures/*.png`
- `outputs/final/tables/*.tex` or `.md`

Acceptance criteria:

- `model_metrics.csv` is the only source for the main performance table.
- Running the script twice with the same seed gives materially identical metrics.
- Failed models are reported honestly instead of omitted silently.

### Phase 4: Statistical and Error Analysis

Goal: turn metrics into publishable evidence.

Tasks:

- Compute per-sample ADE/FDE/RMSE.
- Compute confidence intervals from per-sample errors.
- Run paired tests only for models with aligned test samples.
- Apply Bonferroni or Holm correction if multiple pairwise comparisons are reported.
- Add error distribution plots.
- Add horizon-specific metrics if the horizon is multi-step.
- Add scenario slices only if metadata exists: speed bins, vessel type, turning intensity, or region.

Required artifacts:

- `outputs/final/statistical_tests.json`
- `outputs/final/error_summary_by_horizon.csv`
- `outputs/final/error_summary_by_group.csv`
- `outputs/final/figures/error_distributions.png`

Acceptance criteria:

- Every p-value and confidence interval in the paper can be traced to `statistical_tests.json`.
- The paper does not claim 5-fold cross-validation unless fold result files exist.

### Phase 5: Decide CPA/TCPA Paper Role

Goal: include risk assessment only if it strengthens the conservative paper.

Option A: Keep CPA/TCPA out of main claims.

- Use it as motivation and future work.
- This is the safest path for a first publishable paper.

Option B: Include a small downstream analysis.

- Use actual model predictions to compute trajectory-based CPA/TCPA.
- Compare current-state CPA, CV-based CPA, and model-based CPA only on generated encounter cases or clearly defined AIS encounter windows.
- Report warning precision/recall only if ground-truth risk labels are defined.

Required artifacts for Option B:

- `outputs/final/risk_scenarios.csv`
- `outputs/final/risk_metrics.json`
- `outputs/final/figures/risk_case_studies.png`

Acceptance criteria:

- The paper never claims "collision avoidance" success unless simulated closed-loop avoidance is fully evaluated.
- If using only CPA/TCPA warnings, call it "risk warning" or "decision support", not "autonomous collision avoidance".

### Phase 6: Rewrite the Paper from Artifacts

Goal: the paper becomes a rendering of repository evidence.

Tasks:

- Start from a clean manuscript, not the current unsupported "proper training" final paper.
- Use generated tables and figures only.
- Add a reproducibility statement.
- Fill authors, affiliations, acknowledgments, data availability, code availability, and conflict of interest.
- Replace placeholder references with verified bibliography.
- Write limitations frankly: one date or one month, one source, near-coastal AIS coverage, limited DL tuning, no operational validation.

Recommended structure:

1. Introduction
2. Related Work
3. Data and Reproducible Pipeline
4. Models and Baselines
5. Evaluation Metrics and Statistical Methods
6. Results
7. Discussion
8. Limitations
9. Conclusion

Acceptance criteria:

- Every table and figure has a generating script.
- Every number in the abstract appears in `outputs/final/model_metrics.csv` or `outputs/final/statistical_tests.json`.
- No placeholders remain.

## 4. Paper-Project Synchronization Rules

Use these rules throughout the rewrite:

- The paper cannot contain a metric absent from `outputs/final/`.
- The README cannot advertise a command that fails.
- The paper cannot claim full reproducibility until environment and data manifests exist.
- If a model fails, record and discuss the failure.
- Exploratory quick runs must be clearly labeled and kept out of main ranking tables.
- Generated plots and tables should be regenerated by scripts, not manually typed.

## 5. Suggested Repository Additions

Minimal additions:

- `requirements.txt`
- `configs/experiment_conservative.yaml`
- `scripts/check_env.py`
- `scripts/audit_dataset.py`
- `scripts/final_train_eval.py`
- `scripts/final_stats.py`
- `scripts/make_paper_assets.py`
- `outputs/final/`
- `paper/conservative_draft.md`

Optional later additions:

- `paper/references.bib`
- `paper/figures/`
- `paper/tables/`
- `scripts/evaluate_risk_warning.py`

## 6. Timeline

### Week 1: Scope and Reproducibility

- Freeze thesis and paper outline.
- Repair environment.
- Add final experiment config.
- Add dataset audit and manifests.

Deliverable:

- Reproducible setup and data audit.

### Week 2: Final Baseline Experiments

- Run CV, Linear Regression, LSTM, Transformer.
- Save metrics, predictions/errors, logs, checkpoints.
- Fix any model/runtime failures.

Deliverable:

- Final auditable model comparison table.

### Week 3: Statistics and Figures

- Generate confidence intervals, paired tests, error distributions.
- Generate final figures and tables from scripts.
- Decide whether CPA/TCPA is excluded, appendix-only, or a small downstream section.

Deliverable:

- Complete evidence pack under `outputs/final/`.

### Week 4: Manuscript Rewrite

- Write conservative paper from generated artifacts.
- Remove unsupported claims.
- Finalize references and limitations.
- Align README with the paper.

Deliverable:

- Submission-ready manuscript draft.

### Week 5: Internal Review and Target Selection

- Audit every claim against artifacts.
- Run final pipeline once from scratch if feasible.
- Select target journal based on final evidence strength.
- Prepare submission formatting.

Deliverable:

- Submission package.

## 7. Go/No-Go Criteria for Submission

Submit only if all are true:

- A clean environment can run the final scripts.
- Dataset manifest and split protocol are saved.
- Main metrics are generated from scripts, not manually typed.
- Statistical tests use per-sample errors.
- Paper and README agree on dataset, horizon, models, and results.
- No placeholders remain.
- Claims about deep learning, reproducibility, and risk are limited to supported evidence.

Do not submit if any are true:

- The main result depends on quick-run subsets only.
- The paper still claims 9.4 m proper LSTM without archived proof.
- The full-data training result file reports model failures but the paper reports success.
- The metric units are ambiguous.
- Figures or tables are manually invented.

## 8. Near-Term Next Step

Phase 1 through Phase 4 infrastructure is implemented and verified. The next concrete tasks are submission preparation rather than evidence scaffolding:

1. Polish `paper/conservative_manuscript.md` into the target journal format.
2. Expand and verify the related-work bibliography.
3. Decide target venue based on the conservative contribution, likely a reproducibility/benchmark framing rather than a novel deep-learning superiority framing.
4. Add author, affiliation, acknowledgments, data availability, code availability, conflict-of-interest, and ethics statements.
5. Decide whether CPA/TCPA remains future work or becomes a separately evaluated appendix.
6. If aiming at a higher-bar engineering journal, add a richer data protocol with temporal or vessel-disjoint splits before claiming generalization.
