# Current Publication Status

Updated: 2026-05-17

## Bottom Line

The active submission target is now **The Journal of Navigation**. The working
roadmap is `JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md`.

The repository now supports a conservative, evidence-backed manuscript about
reproducible AIS trajectory-prediction benchmarking. A JON-specific
submission-candidate package has now been generated from the current evidence
pack. It is not an external submission yet: author metadata, funding,
acknowledgements, competing-interest confirmation, final reference audit, final
language edit, ScholarOne metadata, and optional reviewer suggestions still need
human completion before upload.

The higher-bar journal route has also been implemented as an artifact-complete
candidate package. The repository now has a metadata-rich multiday protocol,
expanded benchmark runner, vessel-disjoint evaluation, neural proxy-tuning
records, risk-warning evaluation, offline latest-AIS prediction export, and a
synchronized submission draft. `outputs/final_submission/readiness_report.json`
currently reports `overall_status=submission_ready_candidate` with no
`blocking_gaps`.

The current paper should not be framed as "deep learning improves ship trajectory prediction." It should be framed as:

> Simple kinematic baselines are strong on the audited real AIS protocol, and naive LSTM/Transformer baselines fail badly under the current controlled run. The contribution is the reproducible evidence chain and the cautionary benchmark result.

## Public Repository and Reproducibility Status

The clean public GitHub repository has been prepared and pushed:

- Repository: <https://github.com/TristanLib/ais>
- Current recommended public tag for citation/pre-DOI discussion:
  `jon-submission-v1.2`
- The exact final commit hash should be copied from the release tag immediately
  before manuscript upload.

The repository intentionally does not store raw NOAA AIS files, processed
`.npz` arrays, per-sample large CSV files, model checkpoints, or virtual
environments. This is the normal source-repository boundary, similar to a
Node.js project not committing `node_modules/`.

The public repository does include the data installation/rebuild path. The
high-quality/JON pipeline can download the planned NOAA source dates, build the
processed dataset, run the benchmark, run the risk-warning evaluation, generate
submission artifacts, and audit readiness:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
bash scripts/run_high_quality_pipeline.sh
```

To reproduce the current JON candidate explicitly:

```bash
PYTHON_BIN=.venv/bin/python \
DOWNLOAD_DATES=true \
DATES="2024-01-02 2024-01-09 2024-02-06 2024-03-05" \
bash scripts/run_high_quality_pipeline.sh
```

Current interpretation:

- `git clone` alone does not contain the full AIS data payload.
- `git clone` plus the documented download command is the intended full
  reproduction route.
- The GitHub repository is sufficient as a code and compact evidence release,
  but a formal paper submission should ideally cite both the GitHub tag/commit
  and a Zenodo or equivalent archived release DOI.

## Submission Cost Strategy

For `The Journal of Navigation`, the recommended route is ordinary
subscription/hybrid-journal publication, not paid Gold Open Access at first.

Current fee understanding from the Cambridge author pages:

- There is no separate submission fee identified in the journal instructions.
- Gold Open Access is optional. If selected and not covered by an institutional
  agreement, the listed APC is GBP 2610 or USD 3655, plus possible taxes.
- Online colour figures are free.
- Printed colour figures are charged separately at GBP 200 or USD 320 per
  figure, with a cap of GBP 1000 or USD 1600 per article.

Recommended author choice:

1. Submit as a regular Research Article.
2. Do not choose paid Gold Open Access unless an institution, funder, or
   agreement covers the APC.
3. Do not request printed colour figures; keep figures colour online and
   readable in grayscale.
4. Reserve or publish a Zenodo/GitHub release DOI before final manuscript upload
   if possible, then cite it in Data and Code Availability.
5. Treat professional English editing as optional. It may help, but it does not
   guarantee acceptance and should not be bought until the authorial polish pass
   is complete.

## Current Manuscript Assets

The repository now contains a JON submission-candidate package, the earlier
evidence-synchronized English artifacts, and a domestic-journal-style Chinese
candidate draft.

The Journal of Navigation candidate artifacts:

- `paper/jon_manuscript.md`: JON-style Research Article candidate, about 6,152
  words, generated from the current evidence pack.
- `paper/jon_manuscript.docx`: Word upload candidate.
- `paper/jon_manuscript.pdf`: 15-page A4 review PDF with embedded figures.
- `paper/jon_manuscript_zh.md`: Chinese working version of the JON manuscript,
  about 4,278 CJK characters.
- `paper/jon_manuscript_zh.docx`: Chinese Word export.
- `paper/jon_manuscript_zh.pdf`: 9-page A4 Chinese review PDF with embedded
  figures.
- `paper/jon_manuscript_zh_interpretation.md`: Chinese explanation of the
  paper's purpose, evidence, limitations, and practical meaning.
- `paper/jon_manuscript_zh_interpretation.pdf`: 2-page A4 PDF explanation.
- `paper/jon_cover_letter.md`: cover letter draft.
- `paper/jon_submission_checklist.md`: ScholarOne-oriented checklist.
- `paper/jon_authorial_polish_workflow.md`: mandatory authorial
  polish/de-template workflow added to remove generated-report flavour while
  preserving required AI-use disclosure.
- `paper/jon_supplementary_materials.md`: supplementary note and evidence
  index.
- `paper/jon_supplementary_materials.zip`: compact supplementary archive,
  about 0.31 MB and below the current 10 MB per-file guideline.
- `paper/figures/jon_*.png`: six JON figures, including protocol, model
  performance, horizon degradation, risk-warning metrics, scenario slices, and
  risk case studies.
- `outputs/final_submission/jon_submission_manifest.json`: package manifest,
  word-count estimate, source artifacts, and claim boundary.
- `scripts/make_jon_submission_pack.py`: generation script for the JON package.

English/high-quality candidate artifacts:

- `paper/submission_manuscript.md`: generated English evidence-synchronized
  draft.
- `paper/submission_manuscript.pdf`: readable PDF export of the English draft.
- `paper/references.bib`: bibliography used by the English draft.
- `outputs/final_submission/submission_manifest.json`: generation manifest.

Chinese/domestic journal candidate artifacts:

- `paper/submission_manuscript_zh.md`: Chinese manuscript candidate with title,
  author/affiliation placeholders, funding placeholder, Chinese abstract,
  keywords, English abstract, introduction, data/task definition, method,
  experiments, discussion, conclusion, data/code availability, conflict of
  interest, and references.
- `paper/submission_manuscript_zh.pdf`: A4 Chinese PDF export. The current PDF
  embeds a local TrueType Chinese-capable font (`ArialUnicodeMS`) so Chinese
  text renders visibly in PDF viewers.
- `outputs/final_submission/chinese_submission_manifest.json`: Chinese draft
  generation manifest and scope note.

Important interpretation:

- The JON package is the active English submission candidate.
- The English and Chinese drafts are now usable as manuscript starting points,
  not final journal uploads.
- The JON manuscript has JON-oriented framing, figures, declarations, cover
  letter, checklist, Chinese working version, Chinese interpretation, and compact
  supplementary archive, but still has human placeholders that must be resolved
  before ScholarOne submission.
- The workflow now includes a required authorial polish/de-template pass before
  final language editing. This is a style and clarity pass, not removal of
  disclosure; the AI-use declaration must remain accurate.
- The Chinese PDF issue where pages appeared blank has been fixed by embedding
  a local Chinese-capable font during PDF export.
- The current Chinese draft is closer to a domestic journal structure than the
  earlier English evidence report, but it still needs target-journal formatting,
  author metadata, funding information, figure placement, and reference style
  cleanup before submission.

## Evidence Pack Status

The strict final pipeline completed successfully:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_final_experiment.sh
```

Verification artifacts:

- `outputs/final/publication_readiness_report.json`: `status=pass`, `error_count=0`, 63 checks passed.
- `outputs/final/reproducibility_check.json`: same-seed metric stability passed for all final models.
- `outputs/final/run_manifest.json`: `is_debug_run=false`.
- `outputs/audit/environment.json`: Python 3.11 `.venv` passes dependency checks.
- `outputs/audit/data_manifest.json`: raw and processed checksums recorded; raw profile records 7,295,616 AIS rows from 15,130 MMSI values on 2024-01-02; coordinate units verified.
- `paper/conservative_manuscript.md`: generated manuscript synchronized with the evidence pack.

## Final Full-Run Results

| Model | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | Interpretation |
|---|---:|---:|---:|---:|---|
| constant_velocity | 306.5 | 659.6 | 691.6 | 195.6 | Strongest supported baseline |
| linear_lstsq | 351.6 | 750.0 | 615.5 | 223.3 | Worse ADE than CV but reproducible |
| lstm_baseline | 69,926.4 | 67,654.1 | 233,702.3 | 42,058.4 | Supported deep-learning failure result |
| transformer_baseline | 81,626.7 | 80,757.5 | 131,464.7 | 48,620.1 | Supported deep-learning failure result |

These values come from `outputs/final/model_metrics.csv`.

## Does the Paper Reach Publishable Level?

### Technically, yes for a conservative benchmark/reproducibility paper

The core result is now supported by a reproducible pipeline, data audit, per-sample errors, statistical tests, model logs, checkpoints, generated tables, figures, and an automated readiness audit. The paper can credibly claim:

- The audited protocol uses 30 one-minute input steps to forecast 15 one-minute future positions.
- The raw AIS source profile and processed split counts are archived in the data manifest.
- Coordinates are WGS84 latitude/longitude and metrics are computed in meters.
- CV is the best model by ADE in the current evidence pack.
- Linear least squares is close but worse than CV by ADE.
- Naive LSTM and Transformer baselines fail dramatically under the current controlled run.
- The repository provides a reproducible artifact trail for every numerical claim in the generated manuscript.

### Not yet submission-polished

Before journal submission, the manuscript still needs:

- A stronger related-work section with verified citations.
- Venue-specific formatting and figure/table styling.
- Author, affiliation, funding, data availability, code availability, conflict-of-interest, and ethics statements.
- A careful discussion of why the LSTM/Transformer baselines fail and what this means methodologically.
- A final human pass to improve narrative flow and remove generated-prose roughness.

For a higher-bar journal such as Ocean Engineering or The Journal of Navigation, the current manuscript is probably not enough yet unless positioned very carefully as a reproducibility/benchmark cautionary study. For a more methods/reproducibility-friendly venue or workshop, it is much closer.

## Project Support for Paper Claims

| Paper claim | Current support | Evidence |
|---|---|---|
| Real AIS data are used with fixed split counts | Supported | `outputs/audit/data_manifest.json` |
| Dataset checksums are archived | Supported | `outputs/audit/data_manifest.json` |
| Metric units are verified | Supported | `outputs/final/run_manifest.json`, `outputs/audit/data_manifest.json` |
| CV reaches about 306.5 m ADE | Supported | `outputs/final/model_metrics.csv` |
| Linear least squares is worse than CV by ADE | Supported | `outputs/final/model_metrics.csv`, `outputs/final/statistical_tests.json` |
| LSTM/Transformer fail under the conservative baseline run | Supported | `outputs/final/model_metrics.csv`, `outputs/final/training_logs/*.json`, `outputs/final/checkpoints/*` |
| Results use per-sample statistical tests | Supported | `outputs/final/per_sample_errors.csv`, `outputs/final/statistical_tests.json` |
| Same-seed metrics are stable | Supported | `outputs/final/reproducibility_check.json` |
| Paper and project are synchronized | Supported | `paper/conservative_manuscript.md`, `paper/generated_results_summary.md`, `outputs/final/publication_readiness_report.json` |
| Temporal CV or vessel-disjoint split generalization | Not supported | NPZ lacks MMSI/timestamp metadata |
| Collision avoidance success | Not supported for main paper | CPA/TCPA and COLREGs modules are outside the final evidence pack |
| Proper LSTM 9.4 m ADE | Not supported | Excluded from current manuscript |
| GNN/STT/PINN superiority | Not supported | Excluded from current manuscript |

## Recommended Next Move

Keep the current paper conservative. The most defensible next work is manuscript
polishing, related-work expansion, target-journal formatting, and a small number
of evidence-hardening experiments, not adding unsupported model claims.

Target journal selection is now documented in
`TARGET_JOURNAL_SELECTION.md`.

Current recommended domestic order:

1. `中国航海`
2. `大连海事大学学报`
3. `上海海事大学学报`
4. `交通信息与安全`
5. `航海技术` as a shorter practical/application fallback
6. `中国海事` only if rewritten as a maritime supervision practice note

Current recommended English order:

1. `The Journal of Navigation`
2. `Ocean Engineering`
3. `IEEE ITSC` as the best conference route
4. `IEEE Transactions on Intelligent Transportation Systems` or
   `Transportation Research Part C` only after stronger external-validity and
   uncertainty/risk-warning evidence
5. `Applied Ocean Research`, `IEEE Open Journal of ITS`, or `JMSE` as practical
   fallback options depending on open-access budget and review-speed needs

The detailed The Journal of Navigation route is recorded in
`JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md`, including manuscript
optimization phases, evidence-hardening tasks, formatting constraints, and the
ScholarOne submission checklist.

Current active plan:

1. Treat `JOURNAL_OF_NAVIGATION_SUBMISSION_ROADMAP.md` as the primary roadmap.
2. Use `paper/jon_manuscript.docx` and `paper/jon_manuscript.pdf` as the active
   English submission candidates.
3. Complete the human submission tasks listed in
   `paper/jon_submission_checklist.md`: author/affiliation/ORCID metadata,
   funding, acknowledgements, competing-interest confirmation, reference audit,
   authorial polish/de-template pass, final language edit, ScholarOne keywords,
   and proof review.
4. Keep the domestic Chinese manuscript as a fallback branch, not the main
   route.

For a domestic Chinese journal submission, use this order:

1. Select one target venue and lock its template, word/page limit, citation
   style, figure/table requirements, and data/code availability expectations.
2. Replace placeholders in `paper/submission_manuscript_zh.md`: authors,
   affiliations, funding, acknowledgements, conflict-of-interest wording, and
   corresponding author details.
3. Expand the related-work section in Chinese, especially AIS trajectory
   prediction, CPA/TCPA risk assessment, maritime decision support, and
   reproducible benchmark methodology.
4. Convert the current Markdown tables into journal-ready numbered tables and
   add at least two figures: model ADE comparison and risk-warning/case-study
   visualization. Figures should be generated from `outputs/final_multiday/`
   and `outputs/final_risk/`, not drawn manually.
5. Convert references to the target journal format, preferably GB/T 7714 for
   domestic journals unless the venue specifies otherwise.
6. Add a limitations paragraph that explicitly says the current protocol is
   stratified time-block historical AIS evaluation, not live AIS prediction and
   not autonomous collision avoidance.
7. Run a final consistency audit: every number in the abstract, tables, and
   conclusion must map to `outputs/audit/`, `outputs/final_multiday/`,
   `outputs/final_risk/`, or `outputs/final_submission/`.
8. Only after the target venue is chosen, export the journal-specific final PDF
   or Word document from the polished manuscript.

## High-Quality Journal Roadmap Status

Implemented infrastructure:

- `configs/experiment_multiday.yaml`
- `scripts/build_multiday_dataset.py`
- `scripts/final_train_eval_multiday.py`
- `scripts/tune_neural_baselines.py`
- `scripts/evaluate_risk_warning.py`
- `scripts/predict_latest_ais.py`
- `scripts/audit_high_quality_pack.py`
- `scripts/run_high_quality_pipeline.sh`
- `src/models/gru.py`
- `src/models/tcn.py`

Current high-quality candidate artifacts:

- `outputs/audit/multiday_data_manifest.json`: metadata-rich processed dataset
  exists, with temporal and vessel-disjoint split labels. The current build
  covers four source dates, contains 186,326 trajectory windows from 7,425 MMSI
  values, uses a documented stratified time-block sampling protocol, and has raw
  checksums computed.
- `outputs/final_multiday/model_metrics.csv`: expanded non-debug benchmark
  covers temporal and vessel-disjoint test policies across CV, constant
  acceleration, Kalman-style CV, ridge/linear least squares, LSTM, GRU,
  Transformer, and TCN.
- `outputs/final_multiday/neural_tuning_protocol.json`: validation-set proxy
  search records the neural-model search space, early-stopping settings, and
  selected proxy configurations.
- `outputs/final_multiday/statistical_tests.json`: paired tests and confidence
  summaries for the expanded run.
- `outputs/final_risk/risk_metrics.json`: 2,000 AIS-derived encounter scenarios
  were evaluated; Kalman-style CV has precision 0.963, recall 0.900,
  false-alarm rate 0.012, and missed-warning rate 0.100 for the chosen 0.5 nmi
  warning threshold.
- `outputs/latest_predictions/trajectory_predictions.csv`: offline prediction
  export ran on a raw AIS CSV subset and produced 2250 forecast rows for 50
  latest trajectory windows.
- `paper/submission_manuscript.md`, `paper/references.bib`, and
  `outputs/final_submission/submission_manifest.json`: synchronized submission
  draft artifacts created.
- `outputs/final_submission/readiness_report.json`: all five roadmap phases
  pass; `blocking_gaps=[]`.

Current caveats for a high-quality journal submission:

- This is a submission-ready-candidate evidence package, not a guarantee of
  acceptance by Ocean Engineering, The Journal of Navigation, or another
  high-quality venue.
- The current multiday protocol uses stratified time blocks, not every minute of
  every selected source date. Broader all-day/seasonal traffic claims require
  additional full-day or alternative time-block experiments.
- Neural tuning is documented as a validation-set proxy search. Strong
  architecture-superiority claims require a broader search and preferably
  independent external validation.
- The risk module supports warning/decision-support claims only; it does not
  validate autonomous collision avoidance.

Therefore: the project and paper now fully support a conservative,
artifact-synchronized candidate submission. The next improvement is not to make
bigger claims, but to harden venue formatting, related work, external
validation, and full-day sensitivity experiments.
