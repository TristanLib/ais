# Public Release Manifest

This repository is prepared for a clean GitHub release at:

https://github.com/TristanLib/ais

## Included

- Source code under `src/`
- Experiment and model configurations under `configs/`
- Reproducibility, audit, paper-generation, and submission-pack scripts under
  `scripts/`
- Documentation, publication roadmaps, and claim-boundary notes
- JON manuscript candidates and generated figures under `paper/`
- Compact evidence artifacts:
  - dataset manifests
  - model metric summaries
  - statistical summaries
  - risk-warning metrics
  - submission/readiness manifests

## Excluded

- NOAA raw AIS CSV/ZIP files
- Processed `.npz` arrays
- Large split manifests
- Per-sample error CSV files
- Model checkpoints and tuning work directories
- Virtual environments, caches, and OS metadata

## Verification Boundary

This public repository is a compact code-and-evidence release. The JON/high-
quality submission audit is the primary public readiness check and should report
`outputs/final_submission/readiness_report.json` with
`overall_status=submission_ready_candidate` and no `blocking_gaps`.

The older conservative-package audit expects full private artifacts such as
`outputs/final/per_sample_errors.csv` and early split manifests. Running that
audit on this compact GitHub release can therefore report missing-file failures
unless the large excluded artifact bundle is restored locally.

## Recommended Citation Path

For journal submission, cite a stable release rather than a moving branch:

1. Push this clean repository to GitHub.
2. Create a version tag such as `jon-submission-v1.3`.
3. Archive that release with Zenodo or a similar service.
4. Add the resulting DOI and commit hash to the manuscript's Data and Code
   Availability section.
